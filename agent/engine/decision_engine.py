"""
decision_engine.py — Stage 8: Final adjudication with weighted confidence scoring.

Synthesises all prior stage outputs into a single ClaimDecision.

Decision priority cascade (strict order):
  1. Any REJECT rule failure           → REJECTED
  2. Any blocking edge case (HIGH/CRIT)→ NEEDS_REVIEW
  3. Any FLAG_REVIEW rule warning      → NEEDS_REVIEW
  4. Any medium edge case              → NEEDS_REVIEW
  5. All clear                         → APPROVED

Confidence score = weighted average of three components:
  40% coding confidence   — average of all ICD-10 + CPT mapping confidences
  35% rule pass rate      — proportion of evaluated rules that passed
  25% edge case penalty   — 1.0 minus severity-weighted penalty sum

This composite score means a claim with perfect coding but a blocking rule
failure still shows reduced confidence, accurately reflecting claim quality.
"""

import logging
from typing import Optional

from agent.models.enums import ClaimDecision, Severity
from agent.models.schemas import (
    AdjudicationResult,
    CodingResult,
    EdgeCaseResult,
    ExtractionResult,
    ImagingResult,
    RAGResult,
    RuleEngineResult,
    TokenizedClaimInput,
)

logger = logging.getLogger(__name__)

# Confidence weight constants
_W_CODING    = 0.40
_W_RULES     = 0.35
_W_EDGE      = 0.25


class DecisionEngine:
    """
    Stage 8 — Produces the final AdjudicationResult.

    Usage (called by pipeline.py):
        engine = DecisionEngine()
        result = engine.decide(claim, extraction, coding, rules, edge_cases, rag)
    """

    def decide(
        self,
        claim:       TokenizedClaimInput,
        extraction:  ExtractionResult,
        coding:      CodingResult,
        rules:       RuleEngineResult,
        edge_cases:  EdgeCaseResult,
        rag:         Optional[RAGResult]    = None,
        imaging:     Optional[ImagingResult]= None,
    ) -> AdjudicationResult:
        """
        Makes the final adjudication decision.

        Returns a fully populated AdjudicationResult with decision,
        confidence score, all codes, rule evaluations, edge cases, and reasons.
        The explainability report is attached by explainability.py (Stage 9).
        """
        decision  = self._make_decision(rules, edge_cases)
        confidence= self._calculate_confidence(coding, rules, edge_cases)
        reasons   = self._build_reasons(decision, rules, edge_cases)

        logger.info(
            f"[{claim.claim_id}] Decision: {decision.value} "
            f"confidence={confidence:.4f} "
            f"rules_failed={len(rules.blocking_failures)} "
            f"edge_cases={len(edge_cases.edge_cases)}"
        )

        return AdjudicationResult(
            claim_id         = claim.claim_id,
            audit_trace_id   = claim.audit_trace_id,
            decision         = decision,
            confidence_score = confidence,
            icd10_codes      = coding.icd10_codes,
            cpt_codes        = coding.cpt_codes,
            rule_evaluations = rules.evaluations,
            edge_cases       = edge_cases.edge_cases,
            rag_passages     = rag.passages if rag else [],
            imaging_result   = imaging,
            reasons          = reasons,
        )

    # ── Decision cascade ─────────────────────────────────────────────────────

    def _make_decision(
        self,
        rules:      RuleEngineResult,
        edge_cases: EdgeCaseResult,
    ) -> ClaimDecision:
        """
        Strict priority cascade — first matching condition wins.
        """
        # 1. Hard REJECT — any blocking rule failure
        if rules.blocking_failures:
            return ClaimDecision.REJECTED

        # 2. Edge case blocking issues (CRITICAL or HIGH severity)
        if edge_cases.has_blocking_issues:
            return ClaimDecision.NEEDS_REVIEW

        # 3. Rule engine FLAG_REVIEW warnings
        if rules.warnings:
            return ClaimDecision.NEEDS_REVIEW

        # 4. Any medium/low edge case still routes to review
        if edge_cases.edge_cases:
            return ClaimDecision.NEEDS_REVIEW

        # 5. All clear
        return ClaimDecision.APPROVED

    # ── Confidence scoring ───────────────────────────────────────────────────

    def _calculate_confidence(
        self,
        coding:     CodingResult,
        rules:      RuleEngineResult,
        edge_cases: EdgeCaseResult,
    ) -> float:
        """
        Weighted confidence score:
          40% coding confidence
          35% rule pass rate
          25% edge case penalty
        """
        # Component 1: coding confidence
        all_codes = coding.icd10_codes + coding.cpt_codes
        if all_codes:
            coding_conf = sum(c.confidence for c in all_codes) / len(all_codes)
        else:
            coding_conf = 0.0

        # Component 2: rule pass rate
        total_rules = rules.rules_evaluated
        if total_rules > 0:
            failed = len(rules.blocking_failures) + len(rules.warnings)
            rule_conf = max(0.0, (total_rules - failed) / total_rules)
        else:
            rule_conf = 1.0   # No rules evaluated = no failures

        # Component 3: edge case penalty
        penalty = sum(ec.severity.penalty() for ec in edge_cases.edge_cases)
        edge_conf = max(0.0, 1.0 - penalty)

        # Weighted average
        score = (
            _W_CODING * coding_conf +
            _W_RULES  * rule_conf   +
            _W_EDGE   * edge_conf
        )

        return round(min(1.0, max(0.0, score)), 4)

    # ── Human-readable reasons ───────────────────────────────────────────────

    def _build_reasons(
        self,
        decision:   ClaimDecision,
        rules:      RuleEngineResult,
        edge_cases: EdgeCaseResult,
    ) -> list[str]:
        """
        Builds a concise list of human-readable reason strings.
        These appear in the dashboard and audit trail.
        """
        reasons = []

        if decision == ClaimDecision.APPROVED:
            reasons.append("All policy rules passed with no edge cases detected.")
            return reasons

        # Blocking rule failures
        for ev in rules.blocking_failures:
            reasons.append(f"[REJECTED] Rule {ev.rule_id} ({ev.rule_name}): {ev.reason}")

        # Rule warnings
        for ev in rules.warnings:
            reasons.append(f"[REVIEW] Rule {ev.rule_id} ({ev.rule_name}): {ev.reason}")

        # Edge cases
        for ec in edge_cases.edge_cases:
            prefix = "[BLOCKING]" if ec.severity.is_blocking() else "[FLAG]"
            reasons.append(f"{prefix} {ec.edge_case_type.value}: {ec.description}")

        return reasons
