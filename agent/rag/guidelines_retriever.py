"""
guidelines_retriever.py — Stage 5: Multi-factor clinical guideline retrieval.

Upgraded from simple keyword overlap to a 3-signal scoring system:
  1. TF-IDF semantic similarity (40%) — vector cosine similarity
  2. Code overlap bonus (40%)        — ICD-10/CPT exact matches in passage
  3. Specialty relevance (20%)       — specialty tag matching

This produces dramatically better results for clinical queries because
a brain tumor claim correctly retrieves CNS oncology guidelines rather
than incidentally matching cardiac or orthopedic passages.

Loads all JSON files from data/guidelines/ automatically — just add
more JSON files to expand coverage, no code changes needed.

Switch to pgvector for production semantic search via RAG_BACKEND=pgvector.
"""

import json
import logging
import math
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from agent.models.schemas import CodingResult, GuidelinePassage, RAGResult
from agent.rag.embedder import TFIDFEmbedder

logger = logging.getLogger(__name__)

_GUIDELINES_DIR  = Path(__file__).parent.parent.parent / "data" / "guidelines"
_DEFAULT_TOP_K   = 3
_MIN_RELEVANCE   = 0.05
_RAG_BACKEND_ENV = "RAG_BACKEND"

# Specialty → ICD-10 prefix mapping for specialty relevance scoring
_SPECIALTY_ICD_MAP = {
    "oncology":          ["C", "D0", "D1", "D2", "D3", "D4"],
    "neurology":         ["G", "F0", "F1", "F2", "F3"],
    "cardiology":        ["I"],
    "endocrinology":     ["E"],
    "orthopedics":       ["M", "S"],
    "nephrology":        ["N"],
    "pulmonology":       ["J"],
    "psychiatry":        ["F3", "F4", "F6", "F9"],
    "infectious_disease":["A", "B", "J1"],
    "rheumatology":      ["M0", "M1"],
    "radiology":         [],   # matches all
    "preventive":        ["Z"],
    "policy":            [],   # matches all
}


class GuidelinesRetriever(ABC):
    @abstractmethod
    def retrieve(self, coding: CodingResult, top_k: int = _DEFAULT_TOP_K) -> RAGResult:
        pass


class LocalRetriever(GuidelinesRetriever):
    """
    In-memory multi-factor retrieval from JSON guideline files.

    Scoring formula per passage:
      score = 0.40 * tfidf_sim + 0.40 * code_overlap + 0.20 * specialty_match

    TF-IDF embedder is fitted on the full guideline corpus so rare
    medical terms (glioblastoma, arthroplasty) have higher IDF weights
    than common words (patient, diagnosis, treatment).
    """

    def __init__(self):
        self._chunks = self._load_all_chunks()

        # Fit TF-IDF on the corpus
        corpus_texts = [c.get("content", "") for c in self._chunks]
        self._embedder = TFIDFEmbedder(corpus_texts=corpus_texts)

        # Pre-compute chunk embeddings
        self._chunk_vectors = [
            self._embedder.embed(c.get("content", ""))
            for c in self._chunks
        ]

        logger.info(
            f"LocalRetriever: {len(self._chunks)} chunks loaded and embedded "
            f"(dim={self._embedder.dim})"
        )

    def _load_all_chunks(self) -> list[dict]:
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
                logger.error(f"Failed to load {json_file}: {e}")
        return chunks

    def retrieve(self, coding: CodingResult, top_k: int = _DEFAULT_TOP_K) -> RAGResult:
        """
        Retrieves top_k most relevant guideline passages.

        Query is built from ICD-10 and CPT code descriptions,
        providing rich medical vocabulary for TF-IDF matching.
        """
        if not self._chunks:
            return RAGResult(passages=[], query_used="no chunks loaded")

        # Build rich query from codes and descriptions
        query_parts = []
        claim_icd = set()
        claim_cpt = set()

        for c in coding.icd10_codes:
            query_parts.append(f"{c.code} {c.description}")
            claim_icd.add(c.code)

        for c in coding.cpt_codes:
            query_parts.append(f"{c.code} {c.description}")
            claim_cpt.add(c.code)

        query = " ".join(query_parts).lower()
        if not query.strip():
            return RAGResult(passages=[], query_used="empty query")

        # Infer specialties from ICD-10 codes for specialty matching
        claim_specialties = self._infer_specialties(claim_icd)

        # Embed query
        query_vector = self._embedder.embed(query)

        # Score all chunks
        scored: list[tuple[float, dict]] = []
        for i, chunk in enumerate(self._chunks):
            score = self._score_chunk(
                chunk          = chunk,
                chunk_vector   = self._chunk_vectors[i],
                query_vector   = query_vector,
                claim_icd      = claim_icd,
                claim_cpt      = claim_cpt,
                claim_specialties = claim_specialties,
            )
            scored.append((score, chunk))

        # Sort and filter
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
                relevance  = round(score, 4),
            )
            for i, (score, chunk) in enumerate(top)
        ]

        if passages:
            logger.info(
                f"RAG retrieved {len(passages)}/{len(self._chunks)} passages "
                f"(top: '{passages[0].source[:40]}' score={passages[0].relevance:.3f})"
            )

        return RAGResult(passages=passages, query_used=query[:300])

    def _score_chunk(
        self,
        chunk:             dict,
        chunk_vector:      list[float],
        query_vector:      list[float],
        claim_icd:         set[str],
        claim_cpt:         set[str],
        claim_specialties: set[str],
    ) -> float:
        """
        Computes composite relevance score for a chunk:
          40% TF-IDF cosine similarity
          40% code overlap (ICD-10 + CPT exact matches)
          20% specialty alignment
        """
        # Signal 1: TF-IDF cosine similarity
        dot = sum(a * b for a, b in zip(query_vector, chunk_vector))
        tfidf_score = max(0.0, dot)

        # Signal 2: Code overlap
        chunk_icd = set(chunk.get("icd_relevant", []))
        chunk_cpt = set(chunk.get("cpt_relevant", []))

        icd_overlap = len(claim_icd & chunk_icd)
        cpt_overlap = len(claim_cpt & chunk_cpt)

        # Also check prefix matches (e.g. claim has C71.9, chunk has C71 prefix)
        icd_prefix_matches = sum(
            1 for claim_code in claim_icd
            for chunk_code in chunk_icd
            if claim_code[:3] == chunk_code[:3]
        )

        code_score = min(1.0, (icd_overlap * 0.35 + cpt_overlap * 0.25 + icd_prefix_matches * 0.15))

        # Signal 3: Specialty alignment
        chunk_specialty = chunk.get("specialty", "")
        specialty_score = 0.0
        if chunk_specialty in claim_specialties:
            specialty_score = 1.0
        elif chunk_specialty in ("policy", "radiology"):
            specialty_score = 0.3  # policy/radiology are always somewhat relevant

        composite = (0.40 * tfidf_score) + (0.40 * code_score) + (0.20 * specialty_score)
        return round(min(1.0, composite), 4)

    def _infer_specialties(self, icd_codes: set[str]) -> set[str]:
        """
        Infers relevant medical specialties from ICD-10 codes.
        Used to boost specialty-matched guideline passages.
        """
        specialties = set()
        for code in icd_codes:
            for specialty, prefixes in _SPECIALTY_ICD_MAP.items():
                for prefix in prefixes:
                    if code.upper().startswith(prefix.upper()):
                        specialties.add(specialty)
                        break
        return specialties


# ─────────────────────────────────────────────
# PGVECTOR RETRIEVER (production)
# ─────────────────────────────────────────────

class PgvectorRetriever(GuidelinesRetriever):
    """
    PostgreSQL + pgvector semantic retrieval for production.
    Seed with: python scripts/seed_guidelines.py
    Requires: pip install psycopg2-binary pgvector
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

        self._embedder = TFIDFEmbedder()
        self._conn     = psycopg2.connect(self._dsn)
        self._ensure_table()

    def _ensure_table(self):
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS guideline_chunks (
                    id          SERIAL PRIMARY KEY,
                    passage_id  TEXT NOT NULL UNIQUE,
                    source      TEXT NOT NULL,
                    specialty   TEXT,
                    content     TEXT NOT NULL,
                    icd_relevant TEXT[] DEFAULT '{{}}',
                    cpt_relevant TEXT[] DEFAULT '{{}}',
                    embedding   vector({self._embedder.dim})
                )
            """)
        self._conn.commit()

    def retrieve(self, coding: CodingResult, top_k: int = _DEFAULT_TOP_K) -> RAGResult:
        query_parts = [f"{c.code} {c.description}" for c in coding.icd10_codes + coding.cpt_codes]
        query = " ".join(query_parts)
        if not query.strip():
            return RAGResult(passages=[], query_used="empty query")

        query_emb = self._embedder.embed(query)

        with self._conn.cursor(cursor_factory=self._extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT passage_id, source, content,
                          1 - (embedding <=> %s::vector) AS relevance
                   FROM   guideline_chunks
                   ORDER  BY embedding <=> %s::vector
                   LIMIT  %s""",
                (query_emb, query_emb, top_k),
            )
            rows = cur.fetchall()

        passages = [
            GuidelinePassage(
                source=r["source"], passage_id=r["passage_id"],
                content=r["content"], relevance=round(float(r["relevance"]), 4),
            )
            for r in rows if float(r["relevance"]) >= _MIN_RELEVANCE
        ]
        return RAGResult(passages=passages, query_used=query[:300])

    def close(self):
        self._conn.close()


# ─────────────────────────────────────────────
# FACTORY
# ─────────────────────────────────────────────

_retriever_instance: Optional[GuidelinesRetriever] = None


def get_retriever() -> GuidelinesRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        backend = os.environ.get(_RAG_BACKEND_ENV, "local").lower()
        if backend == "local":
            _retriever_instance = LocalRetriever()
        elif backend == "pgvector":
            _retriever_instance = PgvectorRetriever()
        else:
            raise ValueError(f"Unknown RAG_BACKEND='{backend}'. Use 'local' or 'pgvector'.")
    return _retriever_instance


def reset_retriever():
    """Reset singleton — used in tests."""
    global _retriever_instance
    _retriever_instance = None
