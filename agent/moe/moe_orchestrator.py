"""
moe_orchestrator.py — MoE Pipeline Orchestrator

Entry point for the MoE system. Called from api.py during adjudication.

Flow:
  1. Router scores all experts against extracted data
  2. Activated experts run in parallel (asyncio.gather)
  3. Findings are merged into a MoEResult
  4. MoEResult is injected into the adjudication pipeline
     as additional context for the rule engine
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

from .router import route, RouterDecision
from .base_expert import ExpertFinding
from .oncology_expert import OncologyExpert
from .neurology_expert import NeurologyExpert
from .cardiology_expert import CardiologyExpert
from .radiology_expert import RadiologyExpert
from .orthopedics_expert import OrthopedicsExpert
from .psychiatry_expert import PsychiatryExpert

logger = logging.getLogger(__name__)

EXPERT_REGISTRY = {
    "oncology":    OncologyExpert,
    "neurology":   NeurologyExpert,
    "cardiology":  CardiologyExpert,
    "radiology":   RadiologyExpert,
    "orthopedics": OrthopedicsExpert,
    "psychiatry":  PsychiatryExpert,
}


@dataclass
class MoEResult:
    activated_experts:  list[str]
    findings:           list[ExpertFinding]
    router_decision:    RouterDecision
    consensus_risk:     str               # highest risk across all experts
    merged_risk_flags:  list[str]
    merged_recommendations: list[str]
    suggested_codes:    list[dict]
    expert_summary:     str               # one-line summary for adjudication
    imaging_assessments: list[str]
    moe_confidence:     float             # average expert confidence
    skipped:            bool = False      # True if no experts activated


async def run_moe(
    extracted_entities: dict,
    icd10_codes:        list[dict],
    cpt_codes:          list[dict],
    clinical_notes:     str = "",
    imaging_result:     Optional[dict] = None,
) -> MoEResult:
    """
    Run the full MoE pipeline. Called during adjudication.
    Returns MoEResult with merged findings from all activated experts.
    """
    # Step 1: Route
    decision = route(
        extracted_entities=extracted_entities,
        icd10_codes=icd10_codes,
        cpt_codes=cpt_codes,
        imaging_result=imaging_result,
    )

    if not decision.activated_experts:
        logger.info("MoE: No experts activated — pipeline continues without MoE")
        return MoEResult(
            activated_experts   = [],
            findings            = [],
            router_decision     = decision,
            consensus_risk      = "low",
            merged_risk_flags   = [],
            merged_recommendations=[],
            suggested_codes     = [],
            expert_summary      = "No specialist review triggered",
            imaging_assessments = [],
            moe_confidence      = 0.0,
            skipped             = True,
        )

    # Step 2: Run activated experts in parallel
    logger.info(f"MoE: Running experts: {decision.activated_experts}")
    tasks = []
    for expert_id in decision.activated_experts:
        if expert_id not in EXPERT_REGISTRY:
            logger.warning(f"Expert '{expert_id}' not found in registry — skipping")
            continue
        expert = EXPERT_REGISTRY[expert_id]()
        tasks.append(expert.analyze(
            extracted_entities = extracted_entities,
            icd10_codes        = icd10_codes,
            cpt_codes          = cpt_codes,
            router_score       = decision.scores[expert_id],
            imaging_result     = imaging_result if decision.imaging_relevant else None,
            clinical_notes     = clinical_notes,
        ))

    findings: list[ExpertFinding] = await asyncio.gather(*tasks, return_exceptions=False)

    # Step 3: Merge findings
    return _merge_findings(findings, decision)


def _merge_findings(findings: list[ExpertFinding], decision: RouterDecision) -> MoEResult:
    """Merge multiple expert findings into a single consolidated result."""

    RISK_ORDER = {"low": 0, "moderate": 1, "high": 2, "critical": 3}

    merged_flags   = []
    merged_recs    = []
    merged_codes   = []
    imaging_assess = []
    confidences    = []
    max_risk       = "low"

    for f in findings:
        # Deduplicate flags
        for flag in f.risk_flags:
            if flag not in merged_flags:
                merged_flags.append(flag)

        # Deduplicate recommendations
        for rec in f.recommendations:
            if rec not in merged_recs:
                merged_recs.append(rec)

        # Merge suggested codes (deduplicate by code)
        existing_codes = {c["code"] for c in merged_codes}
        for code in f.additional_codes:
            if code["code"] not in existing_codes:
                merged_codes.append(code)
                existing_codes.add(code["code"])

        if f.imaging_assessment:
            imaging_assess.append(f.imaging_assessment)

        confidences.append(f.expert_confidence)

        # Track highest risk
        if RISK_ORDER.get(f.risk_level, 0) > RISK_ORDER.get(max_risk, 0):
            max_risk = f.risk_level

    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    # Build expert summary
    summaries = [f"{f.expert_name}: {f.assessment}" for f in findings]
    expert_summary = " | ".join(summaries)

    # Add MoE prefix to all flags for traceability
    prefixed_flags = [f"[MOE:{f.expert_id.upper()}] {flag}"
                      for f in findings for flag in f.risk_flags]

    logger.info(
        f"MoE merged: risk={max_risk} flags={len(merged_flags)} "
        f"recs={len(merged_recs)} codes={len(merged_codes)} "
        f"conf={avg_confidence:.0%}"
    )

    return MoEResult(
        activated_experts    = [f.expert_id for f in findings],
        findings             = findings,
        router_decision      = decision,
        consensus_risk       = max_risk,
        merged_risk_flags    = prefixed_flags,
        merged_recommendations=merged_recs,
        suggested_codes      = merged_codes,
        expert_summary       = expert_summary,
        imaging_assessments  = imaging_assess,
        moe_confidence       = round(avg_confidence, 3),
        skipped              = False,
    )


def moe_result_to_dict(result: MoEResult) -> dict:
    """Serialize MoEResult to a JSON-safe dict for the API response."""
    if result.skipped:
        return {"skipped": True, "reason": "No experts activated"}

    return {
        "activated_experts":   result.activated_experts,
        "consensus_risk":      result.consensus_risk,
        "moe_confidence":      result.moe_confidence,
        "expert_summary":      result.expert_summary,
        "risk_flags":          result.merged_risk_flags,
        "recommendations":     result.merged_recommendations,
        "suggested_codes":     result.suggested_codes,
        "imaging_assessments": result.imaging_assessments,
        "router_scores":       result.router_decision.scores,
        "routing_reason":      result.router_decision.routing_reason,
        "findings": [
            {
                "expert_id":         f.expert_id,
                "expert_name":       f.expert_name,
                "router_score":      f.router_score,
                "expert_confidence": f.expert_confidence,
                "assessment":        f.assessment,
                "risk_level":        f.risk_level,
                "risk_flags":        f.risk_flags,
                "recommendations":   f.recommendations,
                "additional_codes":  f.additional_codes,
                "imaging_assessment":f.imaging_assessment,
                "narrative":         f.narrative,
                "source":            f.source,
                "warnings":          f.warnings,
            }
            for f in result.findings
        ],
        "skipped": False,
    }