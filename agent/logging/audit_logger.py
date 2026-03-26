"""
audit_logger.py — Append-only audit trail for every pipeline stage.

Architecture:
  AuditLogger (abstract base class)
    ├── SQLiteLogger  — local dev, zero setup, append-only SQLite
    └── PostgresLogger — production, PostgreSQL, append-only enforced by DB trigger

Critical design constraints:
  - APPEND ONLY — no UPDATE or DELETE ever. Immutability is the audit guarantee.
  - No raw PHI — AuditEntry.input_snapshot and output_snapshot contain only
    token IDs, code lists, and numeric values. Never real patient names or dates.
  - Every pipeline stage writes exactly one entry — success OR error, never silent.
  - All entries keyed by audit_trace_id — entire claim history queryable in one shot.

Switch backends with AUDIT_BACKEND=sqlite|postgres in .env. Zero code change.
"""

import json
import logging
import os
import sqlite3
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from agent.models.enums import AuditStatus, PipelineStage, VaultBackend
from agent.models.schemas import AuditEntry

logger = logging.getLogger(__name__)

_AUDIT_BACKEND_ENV = "AUDIT_BACKEND"


# ─────────────────────────────────────────────
# ABSTRACT BASE CLASS
# ─────────────────────────────────────────────

class AuditLogger(ABC):
    """
    Abstract base for audit loggers.
    All pipeline stages call log() — never instantiate backends directly.
    Use get_audit_logger() factory instead.
    """

    @abstractmethod
    def log(self, entry: AuditEntry) -> None:
        """
        Appends one AuditEntry to the audit trail.
        Must never raise — log failures are caught and reported separately.
        """

    @abstractmethod
    def get_trace(self, audit_trace_id: str) -> list[AuditEntry]:
        """Returns all AuditEntries for a given audit_trace_id, ordered by timestamp."""

    @abstractmethod
    def get_claim_history(self, claim_id: str) -> list[AuditEntry]:
        """Returns all AuditEntries for a given claim_id across all traces."""

    @abstractmethod
    def close(self) -> None:
        """Closes the logger connection."""


# ─────────────────────────────────────────────
# SQLITE LOGGER (local dev)
# ─────────────────────────────────────────────

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    audit_trace_id  TEXT    NOT NULL,
    claim_id        TEXT    NOT NULL,
    stage           TEXT    NOT NULL,
    status          TEXT    NOT NULL,
    timestamp       TEXT    NOT NULL,
    input_snapshot  TEXT    NOT NULL DEFAULT '{}',
    output_snapshot TEXT    NOT NULL DEFAULT '{}',
    duration_ms     INTEGER NOT NULL DEFAULT 0,
    error_message   TEXT
);

CREATE INDEX IF NOT EXISTS idx_trace   ON audit_log (audit_trace_id);
CREATE INDEX IF NOT EXISTS idx_claim   ON audit_log (claim_id);
CREATE INDEX IF NOT EXISTS idx_stage   ON audit_log (stage);
CREATE INDEX IF NOT EXISTS idx_status  ON audit_log (status);
"""

# SQLite trigger — enforces append-only at the DB level
_NO_UPDATE_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS no_update_audit
BEFORE UPDATE ON audit_log
BEGIN
    SELECT RAISE(ABORT, 'audit_log is append-only — UPDATE not permitted');
END;
"""

_NO_DELETE_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS no_delete_audit
BEFORE DELETE ON audit_log
BEGIN
    SELECT RAISE(ABORT, 'audit_log is append-only — DELETE not permitted');
END;
"""


class SQLiteLogger(AuditLogger):
    """
    SQLite-backed audit logger for local development.

    - Zero setup — creates the SQLite file automatically on first use.
    - Thread-safe via threading.Lock on all writes.
    - Append-only enforced by DB-level triggers (no UPDATE / DELETE).
    - Snapshots serialised as JSON — compact and queryable.

    Configured by AUDIT_SQLITE_PATH env var (default: ./audit.sqlite).
    """

    def __init__(self, db_path: Optional[str] = None):
        self._path = db_path or os.environ.get("AUDIT_SQLITE_PATH", "./audit.sqlite")
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._conn:
            self._conn.executescript(_SQLITE_SCHEMA)
            self._conn.executescript(_NO_UPDATE_TRIGGER)
            self._conn.executescript(_NO_DELETE_TRIGGER)

    def log(self, entry: AuditEntry) -> None:
        try:
            with self._lock:
                with self._conn:
                    self._conn.execute(
                        """INSERT INTO audit_log
                           (audit_trace_id, claim_id, stage, status, timestamp,
                            input_snapshot, output_snapshot, duration_ms, error_message)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            entry.audit_trace_id,
                            entry.claim_id,
                            entry.stage.value,
                            entry.status.value,
                            entry.timestamp.isoformat(),
                            json.dumps(entry.input_snapshot,  default=str),
                            json.dumps(entry.output_snapshot, default=str),
                            entry.duration_ms,
                            entry.error_message,
                        ),
                    )
        except Exception as e:
            # Audit failures must never crash the pipeline
            logger.error(f"Audit log write failed for trace {entry.audit_trace_id}: {e}")

    def get_trace(self, audit_trace_id: str) -> list[AuditEntry]:
        rows = self._conn.execute(
            "SELECT * FROM audit_log WHERE audit_trace_id = ? ORDER BY timestamp ASC",
            (audit_trace_id,),
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def get_claim_history(self, claim_id: str) -> list[AuditEntry]:
        rows = self._conn.execute(
            "SELECT * FROM audit_log WHERE claim_id = ? ORDER BY timestamp ASC",
            (claim_id,),
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def count(self) -> int:
        """Returns total number of audit entries — used in tests."""
        return self._conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]

    def close(self) -> None:
        self._conn.close()

    def _row_to_entry(self, row: sqlite3.Row) -> AuditEntry:
        return AuditEntry(
            audit_trace_id  = row["audit_trace_id"],
            claim_id        = row["claim_id"],
            stage           = PipelineStage(row["stage"]),
            status          = AuditStatus(row["status"]),
            timestamp       = datetime.fromisoformat(row["timestamp"]),
            input_snapshot  = json.loads(row["input_snapshot"]  or "{}"),
            output_snapshot = json.loads(row["output_snapshot"] or "{}"),
            duration_ms     = row["duration_ms"],
            error_message   = row["error_message"],
        )


# ─────────────────────────────────────────────
# POSTGRES LOGGER (production)
# ─────────────────────────────────────────────

_PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    id              BIGSERIAL    PRIMARY KEY,
    audit_trace_id  TEXT         NOT NULL,
    claim_id        TEXT         NOT NULL,
    stage           TEXT         NOT NULL,
    status          TEXT         NOT NULL,
    timestamp       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    input_snapshot  JSONB        NOT NULL DEFAULT '{}',
    output_snapshot JSONB        NOT NULL DEFAULT '{}',
    duration_ms     INTEGER      NOT NULL DEFAULT 0,
    error_message   TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_trace  ON audit_log (audit_trace_id);
CREATE INDEX IF NOT EXISTS idx_audit_claim  ON audit_log (claim_id);
CREATE INDEX IF NOT EXISTS idx_audit_stage  ON audit_log (stage);
CREATE INDEX IF NOT EXISTS idx_audit_status ON audit_log (status);
"""

_PG_NO_UPDATE = """
CREATE OR REPLACE RULE no_update_audit AS ON UPDATE TO audit_log DO INSTEAD NOTHING;
"""

_PG_NO_DELETE = """
CREATE OR REPLACE RULE no_delete_audit AS ON DELETE TO audit_log DO INSTEAD NOTHING;
"""


class PostgresLogger(AuditLogger):
    """
    PostgreSQL-backed audit logger for production.

    - Append-only enforced by PostgreSQL rules (no UPDATE / DELETE).
    - JSONB columns for efficient querying of snapshots.
    - Connection pooling via psycopg2.ThreadedConnectionPool.
    - Requires: pip install psycopg2-binary
    - Configured by DATABASE_URL env var.
    """

    def __init__(self, dsn: Optional[str] = None):
        try:
            import psycopg2
            import psycopg2.pool
            import psycopg2.extras
            self._psycopg2 = psycopg2
            self._extras   = psycopg2.extras
        except ImportError:
            raise ImportError("psycopg2-binary required for PostgresLogger.")

        self._dsn  = dsn or os.environ.get("DATABASE_URL")
        if not self._dsn:
            raise ValueError("DATABASE_URL must be set for PostgresLogger.")

        self._pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1, maxconn=10, dsn=self._dsn
        )
        self._ensure_schema()

    def _ensure_schema(self):
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(_PG_SCHEMA)
                cur.execute(_PG_NO_UPDATE)
                cur.execute(_PG_NO_DELETE)
            conn.commit()
        finally:
            self._pool.putconn(conn)

    def log(self, entry: AuditEntry) -> None:
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO audit_log
                       (audit_trace_id, claim_id, stage, status, timestamp,
                        input_snapshot, output_snapshot, duration_ms, error_message)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        entry.audit_trace_id,
                        entry.claim_id,
                        entry.stage.value,
                        entry.status.value,
                        entry.timestamp,
                        json.dumps(entry.input_snapshot,  default=str),
                        json.dumps(entry.output_snapshot, default=str),
                        entry.duration_ms,
                        entry.error_message,
                    ),
                )
            conn.commit()
        except Exception as e:
            logger.error(f"Postgres audit log failed for {entry.audit_trace_id}: {e}")
        finally:
            self._pool.putconn(conn)

    def get_trace(self, audit_trace_id: str) -> list[AuditEntry]:
        conn = self._pool.getconn()
        try:
            with conn.cursor(cursor_factory=self._extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM audit_log WHERE audit_trace_id = %s ORDER BY timestamp ASC",
                    (audit_trace_id,),
                )
                return [self._row_to_entry(dict(r)) for r in cur.fetchall()]
        finally:
            self._pool.putconn(conn)

    def get_claim_history(self, claim_id: str) -> list[AuditEntry]:
        conn = self._pool.getconn()
        try:
            with conn.cursor(cursor_factory=self._extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM audit_log WHERE claim_id = %s ORDER BY timestamp ASC",
                    (claim_id,),
                )
                return [self._row_to_entry(dict(r)) for r in cur.fetchall()]
        finally:
            self._pool.putconn(conn)

    def close(self) -> None:
        self._pool.closeall()

    def _row_to_entry(self, row: dict) -> AuditEntry:
        snap_in  = row["input_snapshot"]
        snap_out = row["output_snapshot"]
        return AuditEntry(
            audit_trace_id  = row["audit_trace_id"],
            claim_id        = row["claim_id"],
            stage           = PipelineStage(row["stage"]),
            status          = AuditStatus(row["status"]),
            timestamp       = row["timestamp"],
            input_snapshot  = json.loads(snap_in)  if isinstance(snap_in,  str) else snap_in  or {},
            output_snapshot = json.loads(snap_out) if isinstance(snap_out, str) else snap_out or {},
            duration_ms     = row["duration_ms"],
            error_message   = row["error_message"],
        )


# ─────────────────────────────────────────────
# FACTORY
# ─────────────────────────────────────────────

_logger_instance: Optional[AuditLogger] = None
_logger_lock = threading.Lock()


def get_audit_logger() -> AuditLogger:
    """
    Returns the singleton AuditLogger for the configured backend.
    Reads AUDIT_BACKEND from .env (default: sqlite).
    This is the ONLY function pipeline.py should call.
    """
    global _logger_instance
    if _logger_instance is not None:
        return _logger_instance

    with _logger_lock:
        if _logger_instance is not None:
            return _logger_instance

        backend = os.environ.get(_AUDIT_BACKEND_ENV, "sqlite").lower()

        if backend == "sqlite":
            _logger_instance = SQLiteLogger()
        elif backend == "postgres":
            _logger_instance = PostgresLogger()
        else:
            raise ValueError(
                f"Unknown AUDIT_BACKEND='{backend}'. Use 'sqlite' or 'postgres'."
            )

    logger.info(f"AuditLogger initialised: {type(_logger_instance).__name__}")
    return _logger_instance


def reset_audit_logger() -> None:
    """Resets singleton for tests. Never call in production."""
    global _logger_instance
    with _logger_lock:
        if _logger_instance is not None:
            _logger_instance.close()
        _logger_instance = None
