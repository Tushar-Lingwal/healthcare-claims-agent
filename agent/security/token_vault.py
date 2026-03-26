"""
token_vault.py — PHI token vault: SQLiteVault (local) + PostgresVault (production).

Token format:  PHI_<FIELDTYPE>_<8hexchars>
Example:       PHI_NAME_a3f9b12e

Switch backends via VAULT_BACKEND=sqlite|postgres in .env. Zero code change.
"""

import os
import uuid
import sqlite3
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from agent.models.enums import PHIFieldType, VaultBackend
from agent.security.encryption import encrypt, decrypt


def generate_token(phi_type: PHIFieldType) -> str:
    """Generates PHI_<FIELDTYPE>_<8 random hex chars> token."""
    return f"PHI_{phi_type.value}_{uuid.uuid4().hex[:8]}"


# ─────────────────────────────────────────────
# ABSTRACT BASE
# ─────────────────────────────────────────────

class PHIVault(ABC):
    @abstractmethod
    def store(self, phi_type: PHIFieldType, raw_value: str, claim_id: str) -> str:
        """Encrypts raw_value, stores it, returns token."""

    @abstractmethod
    def retrieve(self, token: str) -> Optional[str]:
        """Decrypts and returns raw PHI for token. None if not found."""

    @abstractmethod
    def delete_claim(self, claim_id: str) -> int:
        """Deletes all tokens for a claim. Returns count deleted."""

    @abstractmethod
    def token_exists(self, token: str) -> bool:
        """Returns True if token exists in vault."""

    @abstractmethod
    def close(self) -> None:
        """Closes vault connection."""


# ─────────────────────────────────────────────
# SQLITE VAULT (local dev)
# ─────────────────────────────────────────────

_SQLITE_CREATE = """
CREATE TABLE IF NOT EXISTS phi_vault (
    token           TEXT PRIMARY KEY,
    phi_type        TEXT NOT NULL,
    encrypted_value TEXT NOT NULL,
    claim_id        TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_claim_id ON phi_vault (claim_id);
"""


class SQLiteVault(PHIVault):
    """
    Local SQLite vault — zero setup, thread-safe, AES-256-GCM encrypted.
    File path from PHI_VAULT_SQLITE_PATH env var (default: ./vault.sqlite).
    Add vault.sqlite to .gitignore — never commit it.
    """

    def __init__(self, db_path: Optional[str] = None):
        self._path = db_path or os.environ.get("PHI_VAULT_SQLITE_PATH", "./vault.sqlite")
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._conn:
            self._conn.executescript(_SQLITE_CREATE)

    def store(self, phi_type: PHIFieldType, raw_value: str, claim_id: str) -> str:
        token     = generate_token(phi_type)
        encrypted = encrypt(raw_value)
        now       = datetime.utcnow().isoformat()
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO phi_vault (token, phi_type, encrypted_value, claim_id, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (token, phi_type.value, encrypted, claim_id, now),
                )
        return token

    def retrieve(self, token: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT encrypted_value FROM phi_vault WHERE token = ?", (token,)
        ).fetchone()
        return decrypt(row["encrypted_value"]) if row else None

    def delete_claim(self, claim_id: str) -> int:
        with self._lock:
            with self._conn:
                cur = self._conn.execute(
                    "DELETE FROM phi_vault WHERE claim_id = ?", (claim_id,)
                )
        return cur.rowcount

    def token_exists(self, token: str) -> bool:
        return self._conn.execute(
            "SELECT 1 FROM phi_vault WHERE token = ?", (token,)
        ).fetchone() is not None

    def close(self) -> None:
        self._conn.close()

    def _count(self) -> int:
        """Total stored tokens — used in tests."""
        return self._conn.execute("SELECT COUNT(*) FROM phi_vault").fetchone()[0]


# ─────────────────────────────────────────────
# POSTGRES VAULT (production)
# ─────────────────────────────────────────────

class PostgresVault(PHIVault):
    """
    Production PostgreSQL vault — runs in private VPC, AES-256-GCM encrypted.
    Configured by DATABASE_URL env var.
    Requires: pip install psycopg2-binary
    """

    def __init__(self, dsn: Optional[str] = None):
        try:
            import psycopg2
            import psycopg2.pool
            self._psycopg2 = psycopg2
        except ImportError:
            raise ImportError("psycopg2-binary required for PostgresVault.")

        self._dsn = dsn or os.environ.get("DATABASE_URL")
        if not self._dsn:
            raise ValueError("DATABASE_URL must be set for PostgresVault.")

        self._pool = psycopg2.pool.ThreadedConnectionPool(minconn=1, maxconn=10, dsn=self._dsn)
        self._ensure_table()

    def _ensure_table(self):
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS phi_vault (
                        token           TEXT PRIMARY KEY,
                        phi_type        TEXT NOT NULL,
                        encrypted_value TEXT NOT NULL,
                        claim_id        TEXT NOT NULL,
                        created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    CREATE INDEX IF NOT EXISTS idx_phi_vault_claim ON phi_vault (claim_id);
                """)
            conn.commit()
        finally:
            self._pool.putconn(conn)

    def store(self, phi_type: PHIFieldType, raw_value: str, claim_id: str) -> str:
        token = generate_token(phi_type)
        encrypted = encrypt(raw_value)
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO phi_vault (token, phi_type, encrypted_value, claim_id) "
                    "VALUES (%s, %s, %s, %s)",
                    (token, phi_type.value, encrypted, claim_id),
                )
            conn.commit()
        finally:
            self._pool.putconn(conn)
        return token

    def retrieve(self, token: str) -> Optional[str]:
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT encrypted_value FROM phi_vault WHERE token = %s", (token,))
                row = cur.fetchone()
        finally:
            self._pool.putconn(conn)
        return decrypt(row[0]) if row else None

    def delete_claim(self, claim_id: str) -> int:
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM phi_vault WHERE claim_id = %s", (claim_id,))
                count = cur.rowcount
            conn.commit()
        finally:
            self._pool.putconn(conn)
        return count

    def token_exists(self, token: str) -> bool:
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM phi_vault WHERE token = %s", (token,))
                return cur.fetchone() is not None
        finally:
            self._pool.putconn(conn)

    def close(self) -> None:
        self._pool.closeall()


# ─────────────────────────────────────────────
# FACTORY
# ─────────────────────────────────────────────

_vault_instance: Optional[PHIVault] = None
_vault_lock = threading.Lock()


def get_vault() -> PHIVault:
    """
    Returns the singleton vault for the configured VAULT_BACKEND.
    This is the ONLY function the rest of the codebase should call.
    """
    global _vault_instance
    if _vault_instance is not None:
        return _vault_instance
    with _vault_lock:
        if _vault_instance is not None:
            return _vault_instance
        backend = os.environ.get("VAULT_BACKEND", VaultBackend.SQLITE.value).lower()
        if backend == VaultBackend.SQLITE.value:
            _vault_instance = SQLiteVault()
        elif backend == VaultBackend.POSTGRES.value:
            _vault_instance = PostgresVault()
        else:
            raise ValueError(f"Unknown VAULT_BACKEND='{backend}'. Use 'sqlite' or 'postgres'.")
    return _vault_instance


def reset_vault() -> None:
    """Resets singleton for tests. Never call in production."""
    global _vault_instance
    with _vault_lock:
        if _vault_instance is not None:
            _vault_instance.close()
        _vault_instance = None
