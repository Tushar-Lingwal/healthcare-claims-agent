"""
explainability.py — Stage 9: Builds the auditable reasoning chain.

Attaches a structured ExplainabilityReport to the AdjudicationResult.
This is the "why" layer — every decision is backed by a traceable
chain of reasoning steps that a reviewer, payer, or regulator can follow.

The report contains:
  - summary: one plain-English sentence describing the decision
  - reasoning_chain: ordered ReasoningStep objects (one per pipeline stage)
  - key_factors: the top factors that drove the decision
  - risk_flags: any concerns the reviewer should act on
  - guideline_citations: RAG passages cited as medical necessity justification
"""

import logging
from typing import Optional

from agent.models.enums import ClaimDecision, PipelineStage, Severity
from agent.models.schemas import (
    AdjudicationResult,
    CodingResult,
    EdgeCaseResult,
    ExplainabilityReport,
    ExtractionResult,
    ImagingResult,
    RAGResult,
    ReasoningStep,
    RuleEngineResult,
    TokenizedClaimInput,
)

logger = logging.getLogger(__name__)


class ExplainabilityEngine:
    """
    Stage 9 — Attaches an ExplainabilityReport to AdjudicationResult.

    Usage (called by pipeline.py after decision_engine):
        engine = ExplainabilityEngine()
        result = engine.explain(result, claim, extraction, coding, rules, edge_cases, rag, imaging)
    """

    def explain(
        self,
        result:     AdjudicationResult,
        claim:      TokenizedClaimInput,
        extraction: ExtractionResult,
        coding:     CodingResult,
        rules:      RuleEngineResult,
        edge_cases: EdgeCaseResult,
        rag:        Optional[RAGResult]     = None,
        imaging:    Optional[ImagingResult] = None,
    ) -> AdjudicationResult:
        """
        Builds and attaches ExplainabilityReport to result.
        Returns the same result object with explainability populated.
        """
        chain    = self._build_chain(claim, extraction, coding, rules, edge_cases, rag, imaging)
        factors  = self._build_key_factors(extraction, coding, rules, edge_cases)
        flags    = self._build_risk_flags(rules, edge_cases)
        citations= self._build_citations(rag)
        summary  = self._build_summary(result, rules, edge_cases)

        result.explainability = ExplainabilityReport(
            summary             = summary,
            reasoning_chain     = chain,
            key_factors         = factors,
            risk_flags          = flags,
            guideline_citations = citations,
        )

        logger.info(
            f"[{claim.claim_id}] Explainability built: "
            f"{len(chain)} steps, {len(flags)} risk flags, {len(citations)} citations"
        )

        return result

    # ── Reasoning chain ──────────────────────────────────────────────────────

    def _build_chain(
        self,
        claim:      TokenizedClaimInput,
        extraction: ExtractionResult,
        coding:     CodingResult,
        rules:      RuleEngineResult,
        edge_cases: EdgeCaseResult,
        rag:        Optional[RAGResult],
        imaging:    Optional[ImagingResult],
    ) -> list[ReasoningStep]:
        """
        Builds one ReasoningStep per pipeline stage.
        Steps are ordered by stage — reviewers can follow the decision path.
        """
        chain   = []
        step_no = 1

        # Step 1 — Ingestion
        chain.append(ReasoningStep(
            step        = step_no,
            stage       = PipelineStage.INGESTION,
            description = "Claim ingested and PHI tokenized at gateway.",
            outcome     = f"Claim ID {claim.claim_id} assigned. All PHI fields replaced with vault tokens.",
            confidence  = 1.0,
        ))
        step_no += 1

        # Step 2 — Imaging (conditional)
        if imaging is not None:
            from agent.models.enums import ImagingMode
            if imaging.mode_used != ImagingMode.SKIPPED:
                chain.append(ReasoningStep(
                    step        = step_no,
                    stage       = PipelineStage.IMAGING,
                    description = f"Imaging analysis via {imaging.mode_used.value} model.",
                    outcome     = (
                        f"Classification: '{imaging.class_label}' "
                        f"(confidence {imaging.confidence:.0%}). "
                        f"ICD-10 suggestion: {imaging.icd10_suggestion or 'none'}."
                    ),
                    confidence  = imaging.confidence,
                ))
                step_no += 1

        # Step 3 — Extraction
        n_diag  = len(extraction.diagnoses)
        n_proc  = len(extraction.procedures)
        n_sym   = len(extraction.symptoms)
        n_med   = len(extraction.medications)
        chain.append(ReasoningStep(
            step        = step_no,
            stage       = PipelineStage.EXTRACTION,
            description = "Claude NER extracted clinical entities from tokenized notes.",
            outcome     = (
                f"Found {n_diag} diagnosis(es), {n_proc} procedure(s), "
                f"{n_sym} symptom(s), {n_med} medication(s). "
                f"Overall confidence: {extraction.overall_confidence:.0%}."
            ),
            confidence  = extraction.overall_confidence,
        ))
        step_no += 1

        # Step 4 — Coding
        n_icd = len(coding.icd10_codes)
        n_cpt = len(coding.cpt_codes)
        n_unm = len(coding.unmapped_diagnoses) + len(coding.unmapped_procedures)
        icd_list = ", ".join(c.code for c in coding.icd10_codes[:5])
        cpt_list = ", ".join(c.code for c in coding.cpt_codes[:5])
        chain.append(ReasoningStep(
            step        = step_no,
            stage       = PipelineStage.CODING,
            description = "Medical coder mapped entities to ICD-10-CM and CPT codes.",
            outcome     = (
                f"Mapped {n_icd} ICD-10 code(s) [{icd_list or 'none'}] and "
                f"{n_cpt} CPT code(s) [{cpt_list or 'none'}]. "
                f"{n_unm} entity(ies) could not be mapped."
            ),
            confidence  = coding.overall_confidence,
        ))
        step_no += 1

        # Step 5 — RAG (conditional)
        if rag and rag.passages:
            chain.append(ReasoningStep(
                step        = step_no,
                stage       = PipelineStage.RAG,
                description = "Clinical guidelines retrieved to assess medical necessity.",
                outcome     = (
                    f"Retrieved {len(rag.passages)} guideline passage(s). "
                    f"Top source: {rag.passages[0].source if rag.passages else 'none'}."
                ),
                confidence  = 1.0,
            ))
            step_no += 1

        # Step 6 — Rules
        n_pass  = rules.rules_evaluated - len(rules.blocking_failures) - len(rules.warnings)
        n_fail  = len(rules.blocking_failures)
        n_warn  = len(rules.warnings)
        chain.append(ReasoningStep(
            step        = step_no,
            stage       = PipelineStage.RULE_EXECUTION,
            description = f"Deterministic rule engine evaluated {rules.rules_evaluated} policy rule(s).",
            outcome     = (
                f"{n_pass} passed, {n_fail} blocking failure(s), {n_warn} warning(s). "
                + (
                    "Blocking rules: " + ", ".join(ev.rule_id for ev in rules.blocking_failures)
                    if rules.blocking_failures else "No blocking failures."
                )
            ),
            confidence  = 1.0 if rules.all_passed else max(0.0, n_pass / max(rules.rules_evaluated, 1)),
        ))
        step_no += 1

        # Step 7 — Edge detection
        n_ec = len(edge_cases.edge_cases)
        ec_summary = (
            ", ".join(ec.edge_case_type.value for ec in edge_cases.edge_cases[:3])
            or "none"
        )
        chain.append(ReasoningStep(
            step        = step_no,
            stage       = PipelineStage.EDGE_DETECTION,
            description = "Edge case detector scanned for anomalies.",
            outcome     = (
                f"{n_ec} edge case(s) detected: [{ec_summary}]. "
                f"Blocking: {edge_cases.has_blocking_issues}. "
                f"Image/text mismatch: {edge_cases.image_text_mismatch}."
            ),
            confidence  = 1.0 - sum(ec.severity.penalty() for ec in edge_cases.edge_cases),
        ))
        step_no += 1

        return chain

    # ── Key factors ──────────────────────────────────────────────────────────

    def _build_key_factors(
        self,
        extraction: ExtractionResult,
        coding:     CodingResult,
        rules:      RuleEngineResult,
        edge_cases: EdgeCaseResult,
    ) -> list[str]:
        """Top factors that influenced the decision — shown prominently in dashboard."""
        factors = []

        if coding.icd10_codes:
            codes = ", ".join(f"{c.code} ({c.description})" for c in coding.icd10_codes[:3])
            factors.append(f"Diagnosis codes: {codes}")

        if coding.cpt_codes:
            codes = ", ".join(f"{c.code} ({c.description})" for c in coding.cpt_codes[:3])
            factors.append(f"Procedure codes: {codes}")

        factors.append(f"Rules evaluated: {rules.rules_evaluated}")

        if rules.blocking_failures:
            factors.append(f"Blocking rule failures: {len(rules.blocking_failures)}")

        if edge_cases.edge_cases:
            factors.append(f"Edge cases: {len(edge_cases.edge_cases)}")

        if edge_cases.image_text_mismatch:
            factors.append("IMAGE_TEXT_MISMATCH: imaging and clinical text diagnoses disagree")

        return factors

    # ── Risk flags ───────────────────────────────────────────────────────────

    def _build_risk_flags(
        self,
        rules:      RuleEngineResult,
        edge_cases: EdgeCaseResult,
    ) -> list[str]:
        """Actionable concerns for the human reviewer."""
        flags = []

        for ev in rules.blocking_failures:
            flags.append(f"RULE FAILURE [{ev.rule_id}]: {ev.reason}")

        for ec in edge_cases.edge_cases:
            if ec.severity in (Severity.HIGH, Severity.CRITICAL):
                flags.append(
                    f"{ec.severity.value.upper()} [{ec.edge_case_type.value}]: "
                    f"{ec.description} — {ec.recommendation}"
                )

        return flags

    # ── Guideline citations ──────────────────────────────────────────────────

    def _build_citations(self, rag: Optional[RAGResult]) -> list[str]:
        """Formats RAG passages as citable references."""
        if not rag or not rag.passages:
            return []
        return [
            f"{p.source} (relevance {p.relevance:.2f}): ...{p.content[:120]}..."
            for p in rag.passages[:5]
        ]

    # ── Decision summary sentence ─────────────────────────────────────────────

    def _build_summary(
        self,
        result:     AdjudicationResult,
        rules:      RuleEngineResult,
        edge_cases: EdgeCaseResult,
    ) -> str:
        """One plain-English sentence describing what happened and why."""
        conf_pct = f"{result.confidence_score:.0%}"

        if result.decision == ClaimDecision.APPROVED:
            return (
                f"Claim approved with {conf_pct} confidence — "
                f"all {rules.rules_evaluated} policy rule(s) passed and no anomalies detected."
            )

        if result.decision == ClaimDecision.REJECTED:
            primary = rules.blocking_failures[0] if rules.blocking_failures else None
            reason  = primary.reason if primary else "policy rule failure"
            rule_id = primary.rule_id if primary else "unknown"
            return (
                f"Claim rejected with {conf_pct} confidence — "
                f"rule {rule_id} failed: {reason}"
            )

        # NEEDS_REVIEW
        top_issues = []
        if rules.blocking_failures:
            top_issues.append(f"{len(rules.blocking_failures)} blocking rule failure(s)")
        if rules.warnings:
            top_issues.append(f"{len(rules.warnings)} rule warning(s)")
        if edge_cases.has_blocking_issues:
            top_issues.append(f"{len(edge_cases.edge_cases)} edge case(s) requiring review")
        if edge_cases.image_text_mismatch:
            top_issues.append("imaging/text diagnosis mismatch")

        issue_str = "; ".join(top_issues) if top_issues else "policy concerns"
        return (
            f"Claim flagged for human review with {conf_pct} confidence — "
            f"{issue_str}."
        )
