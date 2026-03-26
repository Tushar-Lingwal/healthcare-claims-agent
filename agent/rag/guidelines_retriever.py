"""
guidelines_retriever.py — Stage 5: Retrieves relevant clinical guideline passages.

Finds guideline passages most relevant to the current claim's diagnosis
and procedure codes, providing evidence for medical necessity decisions.

Two backends:
  LocalRetriever   — in-memory keyword similarity, zero setup, local dev
  PgvectorRetriever — PostgreSQL + pgvector, semantic search, production

The LocalRetriever loads guideline chunks from data/guidelines/*.json
and ranks them by keyword overlap with the coding result. Sufficient for
demo and development. Switch to PgvectorRetriever in production for
genuine semantic search with Anthropic or sentence-transformer embeddings.

Switch via RAG_BACKEND=local|pgvector in .env.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from agent.models.schemas import CodingResult, GuidelinePassage, RAGResult
from agent.rag.embedder import KeywordEmbedder

logger = logging.getLogger(__name__)

_GUIDELINES_DIR  = Path(__file__).parent.parent.parent / "data" / "guidelines"
_DEFAULT_TOP_K   = 3
_MIN_RELEVANCE   = 0.05   # Minimum similarity score to include a passage
_RAG_BACKEND_ENV = "RAG_BACKEND"


# ─────────────────────────────────────────────
# ABSTRACT BASE
# ─────────────────────────────────────────────

class GuidelinesRetriever(ABC):

    @abstractmethod
    def retrieve(self, coding: CodingResult, top_k: int = _DEFAULT_TOP_K) -> RAGResult:
        """Retrieves top_k most relevant guideline passages for a coding result."""


# ─────────────────────────────────────────────
# LOCAL RETRIEVER (dev, zero setup)
# ─────────────────────────────────────────────

class LocalRetriever(GuidelinesRetriever):
    """
    In-memory keyword-based retrieval from JSON guideline files.

    Ranks passages by:
      1. Code overlap — passages mentioning the claim's ICD/CPT codes
      2. Keyword similarity — KeywordEmbedder cosine similarity

    Zero setup — loads from data/guidelines/*.json automatically.
    """

    def __init__(self):
        self._embedder = KeywordEmbedder()
        self._chunks   = self._load_all_chunks()
        logger.info(f"LocalRetriever loaded {len(self._chunks)} guideline chunks")

    def _load_all_chunks(self) -> list[dict]:
        """Loads all JSON guideline files from data/guidelines/."""
        chunks = []
        if not _GUIDELINES_DIR.exists():
            logger.warning(f"Guidelines directory not found: {_GUIDELINES_DIR}")
            return chunks

        for json_file in sorted(_GUIDELINES_DIR.glob("*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    chunks.extend(data)
                    logger.debug(f"Loaded {len(data)} chunks from {json_file.name}")
            except Exception as e:
                logger.error(f"Failed to load guidelines file {json_file}: {e}")

        return chunks

    def retrieve(self, coding: CodingResult, top_k: int = _DEFAULT_TOP_K) -> RAGResult:
        """
        Retrieves top_k most relevant passages for the coding result.

        Scoring:
          - Code overlap bonus: +0.3 per matching ICD/CPT code in passage metadata
          - Keyword similarity: cosine similarity of content vs query string
        """
        if not self._chunks:
            return RAGResult(passages=[], query_used="no chunks loaded")

        # Build query string from codes and their descriptions
        query_parts = []
        for c in coding.icd10_codes:
            query_parts.append(f"{c.code} {c.description}")
        for c in coding.cpt_codes:
            query_parts.append(f"{c.code} {c.description}")
        query = " ".join(query_parts).lower()

        if not query.strip():
            return RAGResult(passages=[], query_used="empty query")

        claim_icd_codes = {c.code for c in coding.icd10_codes}
        claim_cpt_codes = {c.code for c in coding.cpt_codes}

        scored: list[tuple[float, dict]] = []

        for chunk in self._chunks:
            content = chunk.get("content", "")

            # Code overlap bonus
            chunk_icd = set(chunk.get("icd_relevant", []))
            chunk_cpt = set(chunk.get("cpt_relevant", []))
            icd_overlap = len(claim_icd_codes & chunk_icd)
            cpt_overlap = len(claim_cpt_codes & chunk_cpt)
            code_bonus  = (icd_overlap + cpt_overlap) * 0.3

            # Keyword similarity
            kw_sim = self._embedder.similarity(query, content)

            total_score = round(min(1.0, kw_sim + code_bonus), 4)
            scored.append((total_score, chunk))

        # Sort descending by score, take top_k above threshold
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [
            (score, chunk)
            for score, chunk in scored[:top_k]
            if score >= _MIN_RELEVANCE
        ]

        passages = [
            GuidelinePassage(
                source     = chunk.get("source", "Unknown"),
                passage_id = chunk.get("passage_id", f"chunk_{i}"),
                content    = chunk.get("content", ""),
                relevance  = score,
            )
            for i, (score, chunk) in enumerate(top)
        ]

        logger.info(
            f"RAG retrieved {len(passages)} passages "
            f"(top score={top[0][0]:.3f} if passages else 0)"
        )

        return RAGResult(passages=passages, query_used=query[:200])


# ─────────────────────────────────────────────
# PGVECTOR RETRIEVER (production)
# ─────────────────────────────────────────────

class PgvectorRetriever(GuidelinesRetriever):
    """
    PostgreSQL + pgvector semantic retrieval for production.

    Uses pgvector cosine similarity search over pre-computed embeddings.
    Embeddings generated by seed_guidelines.py at setup time.

    Requires:
      - pip install psycopg2-binary pgvector
      - PostgreSQL with pgvector extension installed
      - DATABASE_URL env var set
      - seed_guidelines.py run at least once to populate the embeddings table
    """

    def __init__(self, dsn: Optional[str] = None):
        try:
            import psycopg2
            import psycopg2.extras
            self._psycopg2 = psycopg2
            self._extras   = psycopg2.extras
        except ImportError:
            raise ImportError("psycopg2-binary required for PgvectorRetriever.")

        self._dsn = dsn or os.environ.get("DATABASE_URL")
        if not self._dsn:
            raise ValueError("DATABASE_URL must be set for PgvectorRetriever.")

        self._embedder = KeywordEmbedder()
        self._conn     = psycopg2.connect(self._dsn)
        self._ensure_table()

    def _ensure_table(self):
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS guideline_chunks (
                    id          SERIAL PRIMARY KEY,
                    passage_id  TEXT NOT NULL UNIQUE,
                    source      TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    icd_relevant TEXT[] DEFAULT '{}',
                    cpt_relevant TEXT[] DEFAULT '{}',
                    embedding   vector(58)
                )
            """)
        self._conn.commit()

    def retrieve(self, coding: CodingResult, top_k: int = _DEFAULT_TOP_K) -> RAGResult:
        query_parts = [f"{c.code} {c.description}" for c in coding.icd10_codes + coding.cpt_codes]
        query       = " ".join(query_parts)
        if not query.strip():
            return RAGResult(passages=[], query_used="empty query")

        query_embedding = self._embedder.embed(query)

        with self._conn.cursor(cursor_factory=self._extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT passage_id, source, content,
                       1 - (embedding <=> %s::vector) AS relevance
                FROM   guideline_chunks
                ORDER  BY embedding <=> %s::vector
                LIMIT  %s
                """,
                (query_embedding, query_embedding, top_k),
            )
            rows = cur.fetchall()

        passages = [
            GuidelinePassage(
                source     = r["source"],
                passage_id = r["passage_id"],
                content    = r["content"],
                relevance  = round(float(r["relevance"]), 4),
            )
            for r in rows
            if float(r["relevance"]) >= _MIN_RELEVANCE
        ]

        return RAGResult(passages=passages, query_used=query[:200])

    def close(self):
        self._conn.close()


# ─────────────────────────────────────────────
# FACTORY
# ─────────────────────────────────────────────

def get_retriever() -> GuidelinesRetriever:
    """
    Returns the configured retriever backend.
    Reads RAG_BACKEND from .env (default: local).
    """
    backend = os.environ.get(_RAG_BACKEND_ENV, "local").lower()

    if backend == "local":
        return LocalRetriever()
    elif backend == "pgvector":
        return PgvectorRetriever()
    else:
        raise ValueError(
            f"Unknown RAG_BACKEND='{backend}'. Use 'local' or 'pgvector'."
        )
