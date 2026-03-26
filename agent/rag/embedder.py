"""
embedder.py — Generates and stores embeddings for clinical guideline chunks.

Used by the seed_guidelines.py script (run once at setup) to embed
NCCN, CMS, and AHA guideline passages into pgvector for RAG retrieval.

At runtime, the same embedding function is used to embed the query
(coding result codes + diagnoses) before querying the vector store.

Embedding strategy:
  - Each guideline chunk is a 200-500 word passage from a clinical guideline.
  - Chunks are embedded using Claude's text representation via the
    Anthropic embeddings API (or a local sentence-transformer as fallback).
  - Stored in pgvector with metadata: source, passage_id, content, chunk_index.

Local dev fallback:
  When pgvector is not available, uses an in-memory list of pre-embedded
  chunks from data/guidelines/ JSON files. Cosine similarity computed in Python.
  Zero setup required — works out of the box.
"""

import json
import logging
import math
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_GUIDELINES_DIR = Path(__file__).parent.parent.parent / "data" / "guidelines"


# ─────────────────────────────────────────────
# COSINE SIMILARITY (local fallback)
# ─────────────────────────────────────────────

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Computes cosine similarity between two equal-length vectors."""
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ─────────────────────────────────────────────
# SIMPLE KEYWORD EMBEDDER (local dev, no API)
# ─────────────────────────────────────────────

class KeywordEmbedder:
    """
    Lightweight keyword-overlap embedder for local development.
    No API key or vector DB required.

    Represents text as a sparse TF-IDF-style vector over a
    medical vocabulary. Sufficient for development and demo purposes.

    Production upgrade: swap with Anthropic embeddings API or
    sentence-transformers (e.g. all-MiniLM-L6-v2) for semantic search.
    """

    # Core medical vocabulary for sparse embedding
    _VOCAB = [
        "glioma", "glioblastoma", "meningioma", "pituitary", "brain", "tumor",
        "neoplasm", "malignant", "benign", "cranial", "intracranial", "cerebral",
        "dementia", "alzheimer", "cognitive", "impairment", "memory", "neurological",
        "mri", "imaging", "scan", "contrast", "radiation", "chemotherapy", "surgery",
        "treatment", "therapy", "diagnosis", "staging", "grade", "resection",
        "biopsy", "pathology", "histology", "icd", "cpt", "code", "claim",
        "authorization", "prior", "coverage", "plan", "insurance", "benefit",
        "medical", "necessity", "clinical", "guideline", "standard", "care",
        "patient", "age", "adult", "pediatric", "symptom", "sign", "history",
        "nccn", "cms", "aha", "policy", "criteria", "evidence", "trial",
        "diabetes", "hypertension", "cardiac", "oncology", "neurology", "surgery",
        "approved", "rejected", "review", "documentation", "record",
    ]

    def __init__(self):
        self._vocab_index = {word: i for i, word in enumerate(self._VOCAB)}
        self._dim = len(self._VOCAB)

    def embed(self, text: str) -> list[float]:
        """Returns a sparse keyword-overlap vector for a text string."""
        words  = text.lower().split()
        vector = [0.0] * self._dim

        for word in words:
            # Strip punctuation
            clean = word.strip(".,;:()[]\"'")
            if clean in self._vocab_index:
                vector[self._vocab_index[clean]] += 1.0

        # L2 normalise
        mag = math.sqrt(sum(x * x for x in vector))
        if mag > 0:
            vector = [x / mag for x in vector]

        return vector

    def similarity(self, a: str, b: str) -> float:
        """Returns cosine similarity between two text strings."""
        return _cosine_similarity(self.embed(a), self.embed(b))
