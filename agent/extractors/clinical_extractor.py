"""
clinical_extractor.py — Stage 3: Clinical NER via pluggable LLM provider.

Supports multiple LLM backends via LLM_PROVIDER env var:
  anthropic  — Claude (default, requires: pip install anthropic)
  gemini     — Google Gemini (free tier, requires: pip install google-generativeai)
  groq       — Groq/LLaMA (free tier, requires: pip install groq)
  openai     — OpenAI (requires: pip install openai)
  stub       — no LLM, uses structured_data only (zero dependencies, for testing)

Set in .env:
  LLM_PROVIDER=gemini
  GEMINI_API_KEY=your-key-here

  LLM_PROVIDER=groq
  GROQ_API_KEY=your-key-here

  LLM_PROVIDER=openai
  OPENAI_API_KEY=your-key-here
"""

import json
import logging
import os
import time
from typing import Optional

from agent.models.enums import EntityCategory
from agent.models.schemas import (
    ClinicalEntity,
    ExtractionResult,
    TokenizedClaimInput,
)

logger = logging.getLogger(__name__)

_DEFAULT_PROVIDER    = "stub"
_MAX_TOKENS          = 1024
_TEMPERATURE         = 0.0
_LOW_CONF_THRESHOLD  = 0.75
_RETRY_ATTEMPTS      = 2
_RETRY_DELAY_SECONDS = 1.0

_SYSTEM_PROMPT = """You are a clinical NER engine for a healthcare claims system.
Extract medical entities from tokenized clinical notes and return ONLY raw JSON.

RULES:
1. Return ONLY raw JSON — no markdown, no code blocks, no explanation.
2. Tokens like PHI_NAME_a3f9 are identifiers — ignore them completely.
3. Only extract genuine medical entities present in the text.
4. Do NOT invent entities not in the text.

OUTPUT SCHEMA:
{
  "diagnoses":   [{"text":"string","normalized":"string","confidence":float,"source_span":"string"}],
  "procedures":  [{"text":"string","normalized":"string","confidence":float,"source_span":"string"}],
  "symptoms":    [{"text":"string","normalized":"string","confidence":float,"source_span":"string"}],
  "medications": [{"text":"string","normalized":"string","confidence":float,"source_span":"string"}]
}

Confidence: 0.95=explicit statement, 0.85=clear implication, 0.70=uncertain.
Empty category = empty list [].
"""


# ─────────────────────────────────────────────
# BASE PROVIDER
# ─────────────────────────────────────────────

class _LLMProvider:
    def call(self, notes: str, claim_id: str) -> Optional[str]:
        raise NotImplementedError


# ─────────────────────────────────────────────
# ANTHROPIC PROVIDER
# ─────────────────────────────────────────────

class _AnthropicProvider(_LLMProvider):
    def __init__(self):
        try:
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY", "")
            )
            self._model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        except ImportError:
            raise ImportError("pip install anthropic")

    def call(self, notes: str, claim_id: str) -> Optional[str]:
        try:
            import anthropic
            resp = self._client.messages.create(
                model      = self._model,
                max_tokens = _MAX_TOKENS,
                system     = _SYSTEM_PROMPT,
                messages   = [{"role": "user", "content": f"Extract entities:\n\n{notes}"}],
            )
            return resp.content[0].text.strip()
        except Exception as e:
            logger.error(f"[{claim_id}] Anthropic error: {e}")
            return None


# ─────────────────────────────────────────────
# GEMINI PROVIDER
# ─────────────────────────────────────────────

class _GeminiProvider(_LLMProvider):
    def __init__(self):
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
            model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
            self._model = genai.GenerativeModel(
                model_name       = model_name,
                system_instruction = _SYSTEM_PROMPT,
            )
            logger.info(f"Gemini provider initialised: {model_name}")
        except ImportError:
            raise ImportError("pip install google-generativeai")

    def call(self, notes: str, claim_id: str) -> Optional[str]:
        try:
            resp = self._model.generate_content(
                f"Extract medical entities:\n\n{notes}",
                generation_config={"temperature": _TEMPERATURE, "max_output_tokens": _MAX_TOKENS},
            )
            raw = resp.text.strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(l for l in lines if not l.startswith("```")).strip()
            return raw
        except Exception as e:
            logger.error(f"[{claim_id}] Gemini error: {e}")
            return None


# ─────────────────────────────────────────────
# GROQ PROVIDER
# ─────────────────────────────────────────────

class _GroqProvider(_LLMProvider):
    def __init__(self):
        try:
            from groq import Groq
            self._client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
            self._model  = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
            logger.info(f"Groq provider initialised: {self._model}")
        except ImportError:
            raise ImportError("pip install groq")

    def call(self, notes: str, claim_id: str) -> Optional[str]:
        try:
            resp = self._client.chat.completions.create(
                model       = self._model,
                max_tokens  = _MAX_TOKENS,
                temperature = _TEMPERATURE,
                messages    = [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Extract medical entities:\n\n{notes}"},
                ],
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(l for l in lines if not l.startswith("```")).strip()
            return raw
        except Exception as e:
            logger.error(f"[{claim_id}] Groq error: {e}")
            return None


# ─────────────────────────────────────────────
# OPENAI PROVIDER
# ─────────────────────────────────────────────

class _OpenAIProvider(_LLMProvider):
    def __init__(self):
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
            self._model  = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            logger.info(f"OpenAI provider initialised: {self._model}")
        except ImportError:
            raise ImportError("pip install openai")

    def call(self, notes: str, claim_id: str) -> Optional[str]:
        try:
            resp = self._client.chat.completions.create(
                model       = self._model,
                max_tokens  = _MAX_TOKENS,
                temperature = _TEMPERATURE,
                messages    = [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Extract medical entities:\n\n{notes}"},
                ],
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(l for l in lines if not l.startswith("```")).strip()
            return raw
        except Exception as e:
            logger.error(f"[{claim_id}] OpenAI error: {e}")
            return None


# ─────────────────────────────────────────────
# STUB PROVIDER (zero dependencies, uses structured_data)
# ─────────────────────────────────────────────

class _StubProvider(_LLMProvider):
    """
    Returns empty extraction — structured_data merged in by entity_merger.
    Use when no LLM API key is available.
    Works perfectly with sample_claims.json since all claims have structured_data.
    """
    def call(self, notes: str, claim_id: str) -> Optional[str]:
        logger.debug(f"[{claim_id}] Stub provider — returning empty extraction")
        return json.dumps({"diagnoses": [], "procedures": [], "symptoms": [], "medications": []})


# ─────────────────────────────────────────────
# PROVIDER FACTORY
# ─────────────────────────────────────────────

def _get_provider(provider_name: str) -> _LLMProvider:
    providers = {
        "anthropic": _AnthropicProvider,
        "gemini":    _GeminiProvider,
        "groq":      _GroqProvider,
        "openai":    _OpenAIProvider,
        "stub":      _StubProvider,
    }
    cls = providers.get(provider_name.lower())
    if cls is None:
        raise ValueError(
            f"Unknown LLM_PROVIDER='{provider_name}'. "
            f"Valid: {list(providers.keys())}"
        )
    return cls()


# ─────────────────────────────────────────────
# CLINICAL EXTRACTOR
# ─────────────────────────────────────────────

class ClinicalExtractor:
    """
    Stage 3 — Extracts clinical entities from tokenized clinical notes.

    Usage:
        extractor = ClinicalExtractor()          # uses LLM_PROVIDER from .env
        extractor = ClinicalExtractor("gemini")  # explicit provider
        result    = extractor.extract(claim)
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        client:   Optional[object] = None,   # legacy — accepts mock client for tests
    ):
        if client is not None:
            # Test mode — wrap mock client in a shim
            self._provider = self._MockShim(client)
        else:
            provider_name = provider or os.environ.get("LLM_PROVIDER", _DEFAULT_PROVIDER)
            self._provider = _get_provider(provider_name)
            logger.info(f"ClinicalExtractor using provider: {provider_name}")

    class _MockShim(_LLMProvider):
        """Wraps a unittest.mock client for backward-compatible testing."""
        def __init__(self, mock_client):
            self._c = mock_client
        def call(self, notes, claim_id):
            try:
                resp = self._c.messages.create(
                    model="test", max_tokens=1024, system="", messages=[]
                )
                raw = resp.content[0].text.strip()
                if raw.startswith("```"):
                    lines = raw.split("\n")
                    raw = "\n".join(l for l in lines if not l.startswith("```")).strip()
                return raw
            except Exception as e:
                logger.error(f"Mock client error: {e}")
                return None

    # ── Primary entry point ──────────────────────────────────────────────────

    def extract(self, claim: TokenizedClaimInput) -> ExtractionResult:
        notes = claim.clinical_notes.strip()

        if not notes:
            return ExtractionResult(normalized_text=notes, overall_confidence=0.0)

        normalized = " ".join(notes.split())

        raw_json = self._call_with_retry(normalized, claim.claim_id)

        if raw_json is None:
            logger.error(f"[{claim.claim_id}] LLM extraction failed")
            return ExtractionResult(normalized_text=normalized, overall_confidence=0.0)

        return self._parse_response(raw_json, normalized, claim.claim_id)

    # ── Retry logic ──────────────────────────────────────────────────────────

    def _call_with_retry(self, notes: str, claim_id: str) -> Optional[str]:
        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            result = self._provider.call(notes, claim_id)
            if result is not None:
                return result
            if attempt < _RETRY_ATTEMPTS:
                time.sleep(_RETRY_DELAY_SECONDS * attempt)
        return None

    # ── Response parsing ─────────────────────────────────────────────────────

    def _parse_response(self, raw_json: str, normalized_text: str, claim_id: str) -> ExtractionResult:
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as e:
            logger.error(f"[{claim_id}] JSON parse error: {e}\nRaw: {raw_json[:200]}")
            return ExtractionResult(normalized_text=normalized_text, overall_confidence=0.0)

        diagnoses   = self._parse_entities(data.get("diagnoses",   []), EntityCategory.DIAGNOSIS,  claim_id)
        procedures  = self._parse_entities(data.get("procedures",  []), EntityCategory.PROCEDURE,  claim_id)
        symptoms    = self._parse_entities(data.get("symptoms",    []), EntityCategory.SYMPTOM,    claim_id)
        medications = self._parse_entities(data.get("medications", []), EntityCategory.MEDICATION, claim_id)

        all_entities = diagnoses + procedures + symptoms + medications
        overall = sum(e.confidence for e in all_entities) / len(all_entities) if all_entities else 0.0

        return ExtractionResult(
            diagnoses          = diagnoses,
            procedures         = procedures,
            symptoms           = symptoms,
            medications        = medications,
            normalized_text    = normalized_text,
            overall_confidence = round(overall, 4),
        )

    def _parse_entities(self, items: list, category: EntityCategory, claim_id: str) -> list[ClinicalEntity]:
        entities = []
        for item in items:
            if not isinstance(item, dict):
                continue
            text = item.get("text", "").strip()
            if not text or text.startswith("PHI_"):
                continue
            try:
                confidence = float(item.get("confidence", 0.85))
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, TypeError):
                confidence = 0.85

            entities.append(ClinicalEntity(
                text        = text,
                category    = category,
                confidence  = confidence,
                source_span = (item.get("source_span", "") or "")[:120] or None,
                normalized  = item.get("normalized", text) or text,
            ))
        return entities

    # ── Helpers ──────────────────────────────────────────────────────────────

    def has_low_confidence_entities(self, result: ExtractionResult) -> bool:
        all_e = result.diagnoses + result.procedures + result.symptoms + result.medications
        return any(e.confidence < _LOW_CONF_THRESHOLD for e in all_e)

    def get_low_confidence_entities(self, result: ExtractionResult) -> list[ClinicalEntity]:
        all_e = result.diagnoses + result.procedures + result.symptoms + result.medications
        return [e for e in all_e if e.confidence < _LOW_CONF_THRESHOLD]