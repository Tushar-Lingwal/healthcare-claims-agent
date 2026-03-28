"""
base_expert.py — Abstract base class for MoE expert agents.

Each expert receives:
  - Extracted clinical entities
  - Mapped ICD-10 + CPT codes
  - Optional imaging result from Swin model
  - Router score (confidence that this expert is relevant)

Each expert returns an ExpertFinding with:
  - Clinical assessment in structured form
  - Risk flags
  - Recommended additional codes
  - Confidence in its own assessment
  - Raw LLM narrative (when LLM is used)
"""

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ExpertFinding:
    expert_id:          str               # e.g. "oncology"
    expert_name:        str               # e.g. "Oncology Expert"
    router_score:       float             # how confident router was
    expert_confidence:  float             # expert's own confidence 0..1
    assessment:         str               # one-line clinical summary
    risk_level:         str               # "low" | "moderate" | "high" | "critical"
    risk_flags:         list[str]         = field(default_factory=list)
    recommendations:    list[str]         = field(default_factory=list)
    additional_codes:   list[dict]        = field(default_factory=list)
    imaging_assessment: Optional[str]     = None
    narrative:          Optional[str]     = None  # LLM-written prose
    source:             str               = "rule_based"  # "rule_based" | "llm" | "hybrid"
    warnings:           list[str]         = field(default_factory=list)


def _imaging_section(imaging_result: Optional[dict]) -> str:
    """Formats imaging model output for LLM prompt."""
    if not imaging_result or not imaging_result.get("predicted_class"):
        return "No imaging model output available for this claim."
    cls  = imaging_result.get("predicted_class", "Unknown")
    conf = imaging_result.get("confidence", 0)
    cat  = imaging_result.get("category", "unknown")
    icd  = imaging_result.get("icd10_code", "")
    desc = imaging_result.get("icd10_description", "")
    probs = imaging_result.get("all_probabilities", {})
    top3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
    top3_str = ", ".join(f"{k}: {v:.1%}" for k, v in top3)
    return (
        f"Swin Transformer classification: {cls} ({conf:.1%} confidence)\n"
        f"Category: {cat} | Suggested ICD-10: {icd} — {desc}\n"
        f"Top-3 probabilities: {top3_str}\n"
        f"Interpretation task: State if this matches the clinical notes, "
        f"flag any discordance, and note if confidence is decisive (>85%) or ambiguous (<70%)."
    )


class BaseExpert(ABC):
    """Base class for all MoE expert agents."""

    expert_id:   str = "base"
    expert_name: str = "Base Expert"

    def __init__(self):
        self.llm_provider = os.environ.get("LLM_PROVIDER", "rules").lower()
        self.logger = logging.getLogger(f"agent.moe.{self.expert_id}")

    async def analyze(
        self,
        extracted_entities: dict,
        icd10_codes:        list[dict],
        cpt_codes:          list[dict],
        router_score:       float,
        imaging_result:     Optional[dict] = None,
        clinical_notes:     str = "",
    ) -> ExpertFinding:
        """
        Main entry point. Runs rule-based analysis first,
        then enriches with LLM if provider is configured.
        """
        # Step 1: Rule-based analysis (always runs, zero latency)
        finding = self._rule_based_analysis(
            extracted_entities, icd10_codes, cpt_codes,
            router_score, imaging_result, clinical_notes,
        )

        # Step 2: LLM enrichment (only if provider configured and not rules-only)
        if self.llm_provider != "rules" and router_score >= 0.35:
            try:
                narrative = await self._llm_analysis(
                    extracted_entities, icd10_codes, cpt_codes,
                    imaging_result, clinical_notes, finding,
                )
                if narrative:
                    finding.narrative = narrative
                    finding.source    = "hybrid"
            except Exception as e:
                self.logger.warning(f"LLM enrichment failed: {e} — using rule-based only")
                finding.warnings.append(f"LLM unavailable: {e}")

        self.logger.info(
            f"{self.expert_name}: risk={finding.risk_level} "
            f"conf={finding.expert_confidence:.0%} source={finding.source}"
        )
        return finding

    @abstractmethod
    def _rule_based_analysis(
        self,
        extracted_entities: dict,
        icd10_codes:        list[dict],
        cpt_codes:          list[dict],
        router_score:       float,
        imaging_result:     Optional[dict],
        clinical_notes:     str,
    ) -> ExpertFinding:
        """Deterministic rule-based analysis. Must be implemented by each expert."""
        ...

    @abstractmethod
    def _build_llm_prompt(
        self,
        extracted_entities: dict,
        icd10_codes:        list[dict],
        cpt_codes:          list[dict],
        imaging_result:     Optional[dict],
        clinical_notes:     str,
        rule_finding:       ExpertFinding,
    ) -> tuple[str, str]:
        """Returns (system_prompt, user_prompt) for LLM call."""
        ...

    async def _llm_analysis(
        self,
        extracted_entities: dict,
        icd10_codes:        list[dict],
        cpt_codes:          list[dict],
        imaging_result:     Optional[dict],
        clinical_notes:     str,
        rule_finding:       ExpertFinding,
    ) -> Optional[str]:
        """Calls the configured LLM provider and returns narrative text."""
        system_prompt, user_prompt = self._build_llm_prompt(
            extracted_entities, icd10_codes, cpt_codes,
            imaging_result, clinical_notes, rule_finding,
        )

        provider = self.llm_provider
        try:
            if provider == "groq":
                return await asyncio.to_thread(
                    self._call_groq, system_prompt, user_prompt
                )
            elif provider == "gemini":
                return await asyncio.to_thread(
                    self._call_gemini, system_prompt, user_prompt
                )
            elif provider == "anthropic":
                return await asyncio.to_thread(
                    self._call_anthropic, system_prompt, user_prompt
                )
            elif provider == "openai":
                return await asyncio.to_thread(
                    self._call_openai, system_prompt, user_prompt
                )
        except Exception as e:
            self.logger.warning(f"LLM call failed ({provider}): {e}")
            return None

    def _call_groq(self, system: str, user: str) -> str:
        from groq import Groq
        client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
        model  = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        resp   = client.chat.completions.create(
            model=model, max_tokens=600, temperature=0.2,
            messages=[{"role":"system","content":system},
                      {"role":"user","content":user}],
        )
        return resp.choices[0].message.content.strip()

    def _call_gemini(self, system: str, user: str) -> str:
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
        model = genai.GenerativeModel(
            model_name=os.environ.get("GEMINI_MODEL","gemini-1.5-flash"),
            system_instruction=system,
        )
        resp = model.generate_content(user,
            generation_config={"temperature":0.2,"max_output_tokens":600})
        return resp.text.strip()

    def _call_anthropic(self, system: str, user: str) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY",""))
        resp   = client.messages.create(
            model=os.environ.get("ANTHROPIC_MODEL","claude-sonnet-4-6"),
            max_tokens=600, system=system,
            messages=[{"role":"user","content":user}],
        )
        return resp.content[0].text.strip()

    def _call_openai(self, system: str, user: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY",""))
        resp   = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL","gpt-4o-mini"),
            max_tokens=600, temperature=0.2,
            messages=[{"role":"system","content":system},
                      {"role":"user","content":user}],
        )
        return resp.choices[0].message.content.strip()

    # ── Shared helpers ─────────────────────────────────────────────────────────

    def _get_text(self, entities: dict) -> str:
        parts = []
        for k in ["diagnoses","procedures","symptoms","medications"]:
            v = entities.get(k,[])
            if isinstance(v,list): parts.extend([str(x).lower() for x in v])
            elif v: parts.append(str(v).lower())
        raw = entities.get("raw_text","") or entities.get("clinical_notes","")
        parts.append(str(raw).lower())
        return " ".join(parts)

    def _has_keyword(self, text: str, *keywords) -> bool:
        return any(kw.lower() in text for kw in keywords)

    def _icd_starts(self, codes: list[dict], *prefixes) -> list[dict]:
        return [c for c in codes if any(c["code"].startswith(p) for p in prefixes)]

    def _format_codes(self, codes: list[dict]) -> str:
        return ", ".join(f"{c['code']} ({c['description']})" for c in codes) or "none"