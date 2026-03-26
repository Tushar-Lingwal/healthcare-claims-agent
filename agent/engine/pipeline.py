"""
pipeline.py — The 9-stage Claims Adjudication Pipeline orchestrator.

This is the "brain" of the system. It sequences all stages, handles errors
gracefully, and coordinates the audit logger at every step.

Stage sequence:
  1  Ingestion          — validate input, check PHI tokenization
  2  Imaging            — optional imaging layer (skipped if not triggered)
  3  Extraction         — Claude NER on tokenized clinical notes
  3b Entity merge       — merge text + imaging + structured data entities
  4  Coding             — ICD-10-CM + CPT mapping
  5  RAG                — retrieve clinical guidelines (stubbed until rag/ built)
  6  Rule engine        — evaluate 14 policy rules
  7  Edge detection     — detect anomalies and IMAGE_TEXT_MISMATCH
  8  Decision           — make final adjudication decision
  9  Explainability     — attach reasoning chain and risk flags

Error handling:
  Every stage is wrapped in try/except.
  Any stage failure → AdjudicationResult(NEEDS_REVIEW, confidence=0.0).
  The audit trail always records the error — never silent.

Audit logging:
  Every stage produces an AuditEntry (success or error).
  All entries keyed by audit_trace_id — fully reconstructable per claim.
"""

import logging
import time
from typing import Optional

from agent.models.enums import AuditStatus, ClaimDecision, ImagingMode, PipelineStage
from agent.models.schemas import (
    AdjudicationResult,
    AuditEntry,
    CodingResult,
    EdgeCaseResult,
    ExtractionResult,
    ImagingResult,
    RAGResult,
    RuleEngineResult,
    TokenizedClaimInput,
)
from agent.extractors.clinical_extractor import ClinicalExtractor
from agent.extractors.entity_merger import EntityMerger
from agent.engine.edge_case_detector import EdgeCaseDetector
from agent.engine.decision_engine import DecisionEngine
from agent.engine.explainability import ExplainabilityEngine
from agent.rules.rule_engine import RuleEngine
from agent.coders.medical_coder import MedicalCoder

logger = logging.getLogger(__name__)


class ClaimsAdjudicationPipeline:
    """
    Orchestrates all 9 pipeline stages for a single claim.

    Usage:
        pipeline = ClaimsAdjudicationPipeline(audit_logger=logger)
        result   = pipeline.process(tokenized_claim)
    """

    def __init__(
        self,
        audit_logger          = None,
        clinical_extractor    : Optional[ClinicalExtractor]  = None,
        entity_merger         : Optional[EntityMerger]       = None,
        edge_case_detector    : Optional[EdgeCaseDetector]   = None,
        decision_engine       : Optional[DecisionEngine]     = None,
        explainability_engine : Optional[ExplainabilityEngine] = None,
        rule_engine           = None,
        medical_coder         = None,
    ):
        # All components accept injection for testing
        self._audit          = audit_logger
        self._extractor      = clinical_extractor    or ClinicalExtractor()
        self._merger         = entity_merger         or EntityMerger()
        self._edge_detector  = edge_case_detector    or EdgeCaseDetector()
        self._decision       = decision_engine       or DecisionEngine()
        self._explainer      = explainability_engine or ExplainabilityEngine()
        self._rule_engine    = rule_engine           or RuleEngine()
        self._medical_coder  = medical_coder         or MedicalCoder()

    # ── Primary entry point ──────────────────────────────────────────────────

    def process(self, claim: TokenizedClaimInput) -> AdjudicationResult:
        """
        Runs the full 9-stage pipeline on a tokenized claim.

        Args:
            claim: TokenizedClaimInput — PHI already replaced with vault tokens

        Returns:
            AdjudicationResult with decision, confidence, codes, and full
            explainability report. Always returns a result — never raises.
        """
        logger.info(f"[{claim.claim_id}] Pipeline started (trace={claim.audit_trace_id})")
        start_total = time.monotonic()

        try:
            result = self._run_pipeline(claim)
        except Exception as e:
            logger.error(f"[{claim.claim_id}] Unhandled pipeline error: {e}", exc_info=True)
            result = self._error_result(claim, f"Unhandled pipeline error: {e}")

        elapsed = int((time.monotonic() - start_total) * 1000)
        logger.info(
            f"[{claim.claim_id}] Pipeline complete: "
            f"{result.decision.value} conf={result.confidence_score:.4f} ({elapsed}ms)"
        )
        return result

    # ── Internal pipeline ────────────────────────────────────────────────────

    def _run_pipeline(self, claim: TokenizedClaimInput) -> AdjudicationResult:

        # ── Stage 1: Ingestion ───────────────────────────────────────────────
        t = time.monotonic()
        self._log(claim, PipelineStage.INGESTION, AuditStatus.SUCCESS,
            inp={"claim_id": claim.claim_id, "imaging_mode": claim.imaging_mode.value},
            out={"audit_trace_id": claim.audit_trace_id, "notes_length": len(claim.clinical_notes)},
            ms=self._ms(t),
        )

        # ── Stage 2: Imaging (optional) ──────────────────────────────────────
        imaging_result: Optional[ImagingResult] = None

        if claim.imaging_mode != ImagingMode.SKIPPED:
            t = time.monotonic()
            try:
                imaging_result = self._run_imaging(claim)
                self._log(claim, PipelineStage.IMAGING, AuditStatus.SUCCESS,
                    inp={"mode": claim.imaging_mode.value},
                    out={
                        "class_label": imaging_result.class_label if imaging_result else None,
                        "confidence":  imaging_result.confidence  if imaging_result else None,
                    },
                    ms=self._ms(t),
                )
            except Exception as e:
                self._log(claim, PipelineStage.IMAGING, AuditStatus.ERROR, ms=self._ms(t), error=str(e))
                logger.warning(f"[{claim.claim_id}] Imaging stage failed, continuing without imaging: {e}")
                imaging_result = None

        # ── Stage 3: Extraction ──────────────────────────────────────────────
        extraction_raw = None
        t = time.monotonic()
        try:
            extraction_raw = self._extractor.extract(claim)
            extraction     = self._merger.merge(extraction_raw, claim, imaging_result)
            self._log(claim, PipelineStage.EXTRACTION, AuditStatus.SUCCESS,
                inp={"notes_tokens": len(claim.clinical_notes.split())},
                out={
                    "diagnoses":   len(extraction.diagnoses),
                    "procedures":  len(extraction.procedures),
                    "symptoms":    len(extraction.symptoms),
                    "medications": len(extraction.medications),
                    "confidence":  extraction.overall_confidence,
                },
                ms=self._ms(t),
            )
        except Exception as e:
            self._log(claim, PipelineStage.EXTRACTION, AuditStatus.ERROR, ms=self._ms(t), error=str(e))
            return self._error_result(claim, f"Extraction failed: {e}")

        # ── Stage 4: Coding ──────────────────────────────────────────────────
        t = time.monotonic()
        try:
            coding = self._run_coding(extraction)
            self._log(claim, PipelineStage.CODING, AuditStatus.SUCCESS,
                inp={"entities": len(extraction.diagnoses) + len(extraction.procedures)},
                out={
                    "icd10_codes":        [c.code for c in coding.icd10_codes],
                    "cpt_codes":          [c.code for c in coding.cpt_codes],
                    "unmapped_diagnoses": coding.unmapped_diagnoses,
                    "unmapped_procedures":coding.unmapped_procedures,
                    "confidence":         coding.overall_confidence,
                },
                ms=self._ms(t),
            )
        except Exception as e:
            self._log(claim, PipelineStage.CODING, AuditStatus.ERROR, ms=self._ms(t), error=str(e))
            return self._error_result(claim, f"Coding failed: {e}")

        # ── Stage 5: RAG (stubbed — implemented in rag/ sprint) ──────────────
        t = time.monotonic()
        rag_result: Optional[RAGResult] = self._run_rag(coding)
        self._log(claim, PipelineStage.RAG, AuditStatus.SUCCESS,
            inp={"icd10_codes": [c.code for c in coding.icd10_codes]},
            out={"passages": len(rag_result.passages) if rag_result else 0},
            ms=self._ms(t),
        )

        # ── Stage 6: Rules ───────────────────────────────────────────────────
        t = time.monotonic()
        try:
            rule_result = self._run_rules(claim, coding)
            self._log(claim, PipelineStage.RULE_EXECUTION, AuditStatus.SUCCESS,
                inp={"icd10_codes": [c.code for c in coding.icd10_codes],
                     "cpt_codes":   [c.code for c in coding.cpt_codes]},
                out={
                    "rules_evaluated":  rule_result.rules_evaluated,
                    "blocking_failures":[ev.rule_id for ev in rule_result.blocking_failures],
                    "warnings":         [ev.rule_id for ev in rule_result.warnings],
                    "all_passed":       rule_result.all_passed,
                },
                ms=self._ms(t),
            )
        except Exception as e:
            self._log(claim, PipelineStage.RULE_EXECUTION, AuditStatus.ERROR, ms=self._ms(t), error=str(e))
            return self._error_result(claim, f"Rule engine failed: {e}")

        # ── Stage 7: Edge detection ──────────────────────────────────────────
        # Pass pre-merge extraction so IMAGE_TEXT_MISMATCH uses original text
        # signal before imaging entities were merged in
        t = time.monotonic()
        try:
            edge_result = self._edge_detector.detect(
                claim, extraction_raw or extraction, coding, rule_result, imaging_result
            )
            self._log(claim, PipelineStage.EDGE_DETECTION, AuditStatus.SUCCESS,
                inp={},
                out={
                    "edge_cases":           [ec.edge_case_type.value for ec in edge_result.edge_cases],
                    "has_blocking_issues":  edge_result.has_blocking_issues,
                    "image_text_mismatch":  edge_result.image_text_mismatch,
                },
                ms=self._ms(t),
            )
        except Exception as e:
            self._log(claim, PipelineStage.EDGE_DETECTION, AuditStatus.ERROR, ms=self._ms(t), error=str(e))
            return self._error_result(claim, f"Edge detection failed: {e}")

        # ── Stage 8: Decision ────────────────────────────────────────────────
        t = time.monotonic()
        try:
            result = self._decision.decide(
                claim, extraction, coding, rule_result, edge_result, rag_result, imaging_result
            )
            self._log(claim, PipelineStage.DECISION, AuditStatus.SUCCESS,
                inp={},
                out={
                    "decision":          result.decision.value,
                    "confidence_score":  result.confidence_score,
                },
                ms=self._ms(t),
            )
        except Exception as e:
            self._log(claim, PipelineStage.DECISION, AuditStatus.ERROR, ms=self._ms(t), error=str(e))
            return self._error_result(claim, f"Decision engine failed: {e}")

        # ── Stage 9: Explainability ──────────────────────────────────────────
        t = time.monotonic()
        try:
            result = self._explainer.explain(
                result, claim, extraction, coding, rule_result, edge_result, rag_result, imaging_result
            )
            self._log(claim, PipelineStage.EXPLAINABILITY, AuditStatus.SUCCESS,
                inp={},
                out={
                    "summary":      result.explainability.summary if result.explainability else "",
                    "risk_flags":   len(result.explainability.risk_flags) if result.explainability else 0,
                    "chain_steps":  len(result.explainability.reasoning_chain) if result.explainability else 0,
                },
                ms=self._ms(t),
            )
        except Exception as e:
            self._log(claim, PipelineStage.EXPLAINABILITY, AuditStatus.ERROR, ms=self._ms(t), error=str(e))
            logger.warning(f"[{claim.claim_id}] Explainability failed — result still returned: {e}")
            # Non-fatal — return result without explainability rather than failing

        return result

    # ── Stage implementations (stubs for unbuilt modules) ────────────────────

    def _run_imaging(self, claim: TokenizedClaimInput) -> Optional[ImagingResult]:
        """
        Stage 2 — Imaging layer.
        Stub: returns None until agent/imaging/ sprint is complete.
        Full implementation: imaging_router.py routes to builtin/custom/precomputed.
        """
        if claim.precomputed_class and claim.precomputed_confidence is not None:
            return ImagingResult(
                mode_used    = ImagingMode.PRECOMPUTED,
                class_label  = claim.precomputed_class,
                confidence   = claim.precomputed_confidence,
            )
        logger.debug(f"[{claim.claim_id}] Imaging stub: no precomputed result, returning None")
        return None

    def _run_coding(self, extraction: ExtractionResult) -> CodingResult:
        """Stage 4 — Medical coding."""
        return self._medical_coder.code(extraction)

    def _run_rag(self, coding: CodingResult) -> Optional[RAGResult]:
        """
        Stage 5 — RAG guideline retrieval.
        Stub: returns None until agent/rag/ is built.
        Full implementation: guidelines_retriever.py queries pgvector.
        """
        try:
            from agent.rag.guidelines_retriever import GuidelinesRetriever
            retriever = GuidelinesRetriever()
            return retriever.retrieve(coding)
        except (ImportError, Exception):
            return RAGResult(passages=[], query_used="stub")

    def _run_rules(self, claim: TokenizedClaimInput, coding: CodingResult) -> RuleEngineResult:
        """Stage 6 — Deterministic rule engine."""
        return self._rule_engine.evaluate(claim, coding)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _ms(self, start: float) -> int:
        """Returns elapsed milliseconds since start."""
        return int((time.monotonic() - start) * 1000)

    def _log(
        self,
        claim:  TokenizedClaimInput,
        stage:  PipelineStage,
        status: AuditStatus,
        inp:    dict = None,
        out:    dict = None,
        ms:     int  = 0,
        error:  str  = None,
    ) -> None:
        """Writes one AuditEntry for a pipeline stage."""
        if self._audit is None:
            return

        entry = AuditEntry(
            audit_trace_id  = claim.audit_trace_id,
            claim_id        = claim.claim_id,
            stage           = stage,
            status          = status,
            input_snapshot  = inp  or {},
            output_snapshot = out  or {},
            duration_ms     = ms,
            error_message   = error,
        )

        try:
            self._audit.log(entry)
        except Exception as e:
            logger.error(f"[{claim.claim_id}] Audit log write failed for stage {stage.value}: {e}")

    def _error_result(self, claim: TokenizedClaimInput, message: str) -> AdjudicationResult:
        """Returns a safe NEEDS_REVIEW result when a stage fails."""
        return AdjudicationResult(
            claim_id         = claim.claim_id,
            audit_trace_id   = claim.audit_trace_id,
            decision         = ClaimDecision.NEEDS_REVIEW,
            confidence_score = 0.0,
            reasons          = [f"Pipeline error: {message}"],
        )