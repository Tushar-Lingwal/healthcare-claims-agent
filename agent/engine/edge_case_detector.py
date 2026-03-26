"""
edge_case_detector.py — Stage 7: Detects anomalies the rule engine cannot catch.

Runs after Stage 6 (rule engine) and before Stage 8 (decision engine).
Produces EdgeCaseResult with severity-graded findings that influence the
final adjudication decision.

The novel addition: IMAGE_TEXT_MISMATCH — fires when the imaging model
and clinical text extraction disagree on the primary diagnosis.
This is cross-validated using EntityMerger.detect_imaging_text_conflict().

Severity → Decision mapping:
  CRITICAL  → always blocks (NEEDS_REVIEW minimum, likely REJECTED)
  HIGH      → blocks (NEEDS_REVIEW)
  MEDIUM    → soft flag (human review recommended but not forced)
  LOW       → informational only
"""

import logging
from typing import Optional

from agent.models.enums import EdgeCaseType, Severity
from agent.models.schemas import (
    CodingResult,
    EdgeCase,
    EdgeCaseResult,
    ExtractionResult,
    ImagingResult,
    RuleEngineResult,
    TokenizedClaimInput,
)
from agent.extractors.entity_merger import EntityMerger

logger = logging.getLogger(__name__)

_LOW_CONF_THRESHOLD  = 0.75   # Below this triggers LOW_CONFIDENCE_EXTRACTION
_MISMATCH_THRESHOLD  = 0.50   # Mismatch score above this triggers IMAGE_TEXT_MISMATCH

# Mutually exclusive condition pairs — both cannot be true simultaneously
_CONFLICTING_PAIRS = [
    ("diabetes", "hypoglycemia"),
    ("hypertension", "hypotension"),
    ("tachycardia", "bradycardia"),
    ("hyperkalemia", "hypokalemia"),
    ("malignant", "benign"),
]


class EdgeCaseDetector:
    """
    Stage 7 — Detects anomalies across all prior stage outputs.

    Usage (called by pipeline.py):
        detector = EdgeCaseDetector()
        result   = detector.detect(claim, extraction, coding, rule_result, imaging)
    """

    def __init__(self):
        self._merger = EntityMerger()

    def detect(
        self,
        claim:          TokenizedClaimInput,
        extraction:     ExtractionResult,
        coding:         CodingResult,
        rule_result:    RuleEngineResult,
        imaging_result: Optional[ImagingResult] = None,
    ) -> EdgeCaseResult:
        """
        Runs all edge case checks and returns aggregated EdgeCaseResult.
        Never raises — all check failures are caught and logged.
        """
        edge_cases: list[EdgeCase] = []

        checks = [
            self._check_missing_diagnosis,
            self._check_missing_procedure,
            self._check_low_confidence,
            self._check_unknown_codes,
            self._check_conflicting_conditions,
            self._check_coverage_gap,
            self._check_image_text_mismatch,
        ]

        for check in checks:
            try:
                found = check(claim, extraction, coding, rule_result, imaging_result)
                edge_cases.extend(found)
            except Exception as e:
                logger.error(f"[{claim.claim_id}] Edge case check {check.__name__} failed: {e}")

        # ── Aggregate flags ──────────────────────────────────────────────────
        has_blocking   = any(ec.severity.is_blocking() for ec in edge_cases)
        needs_review   = any(
            ec.severity in (Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL)
            for ec in edge_cases
        )
        mismatch_found = any(
            ec.edge_case_type == EdgeCaseType.IMAGE_TEXT_MISMATCH
            for ec in edge_cases
        )

        if edge_cases:
            logger.info(
                f"[{claim.claim_id}] {len(edge_cases)} edge case(s) detected: "
                f"{[ec.edge_case_type.value for ec in edge_cases]}"
            )

        return EdgeCaseResult(
            edge_cases            = edge_cases,
            has_blocking_issues   = has_blocking,
            requires_human_review = needs_review,
            image_text_mismatch   = mismatch_found,
        )

    # ── Individual checks ────────────────────────────────────────────────────

    def _check_missing_diagnosis(self, claim, extraction, coding, rule_result, imaging) -> list[EdgeCase]:
        """CRITICAL if no diagnoses at all — cannot adjudicate without a diagnosis."""
        cases = []

        if not extraction.diagnoses and not coding.icd10_codes:
            cases.append(EdgeCase(
                edge_case_type = EdgeCaseType.MISSING_DIAGNOSIS,
                description    = "No diagnoses extracted from clinical notes and no ICD-10 codes mapped.",
                severity       = Severity.CRITICAL,
                recommendation = "Request additional clinical documentation or manual coding review.",
            ))

        elif coding.cpt_codes and not coding.icd10_codes:
            cases.append(EdgeCase(
                edge_case_type = EdgeCaseType.MISSING_DIAGNOSIS,
                description    = "Procedures present but no diagnosis codes — claim cannot be adjudicated without a supporting diagnosis.",
                severity       = Severity.CRITICAL,
                recommendation = "Obtain diagnosis codes before resubmitting.",
            ))

        return cases

    def _check_missing_procedure(self, claim, extraction, coding, rule_result, imaging) -> list[EdgeCase]:
        """HIGH if clinical notes mention procedures but none were coded."""
        cases = []

        if extraction.procedures and not coding.cpt_codes:
            cases.append(EdgeCase(
                edge_case_type  = EdgeCaseType.MISSING_PROCEDURE,
                description     = f"{len(extraction.procedures)} procedure(s) extracted from notes but none mapped to CPT codes.",
                severity        = Severity.HIGH,
                recommendation  = "Verify if this is an E/M-only claim; otherwise obtain procedure codes.",
                affected_codes  = [e.text for e in extraction.procedures],
            ))

        return cases

    def _check_low_confidence(self, claim, extraction, coding, rule_result, imaging) -> list[EdgeCase]:
        """MEDIUM if any extracted entity or code mapping is below confidence threshold."""
        cases = []

        # Low-confidence extracted entities
        low_entities = [
            e for e in (extraction.diagnoses + extraction.procedures + extraction.symptoms + extraction.medications)
            if e.confidence < _LOW_CONF_THRESHOLD
        ]

        # Low-confidence code mappings
        low_codes = [
            c for c in (coding.icd10_codes + coding.cpt_codes)
            if c.confidence < _LOW_CONF_THRESHOLD
        ]

        if low_entities or low_codes:
            affected = [e.text for e in low_entities] + [c.code for c in low_codes]
            cases.append(EdgeCase(
                edge_case_type = EdgeCaseType.LOW_CONFIDENCE_EXTRACTION,
                description    = (
                    f"{len(low_entities)} low-confidence entity(ies) and "
                    f"{len(low_codes)} low-confidence code mapping(s) detected."
                ),
                severity       = Severity.MEDIUM,
                recommendation = "Clinical review recommended to verify extracted entities and code mappings.",
                affected_codes = affected[:10],   # cap to avoid oversized payloads
            ))

        return cases

    def _check_unknown_codes(self, claim, extraction, coding, rule_result, imaging) -> list[EdgeCase]:
        """HIGH if diagnoses or procedures could not be mapped to codes."""
        cases = []

        if coding.unmapped_diagnoses:
            cases.append(EdgeCase(
                edge_case_type = EdgeCaseType.UNKNOWN_CODE_DIAGNOSIS,
                description    = f"{len(coding.unmapped_diagnoses)} diagnosis(es) could not be mapped to ICD-10 codes.",
                severity       = Severity.HIGH,
                recommendation = "Manual coding review required for unmapped diagnoses.",
                affected_codes = coding.unmapped_diagnoses,
            ))

        if coding.unmapped_procedures:
            cases.append(EdgeCase(
                edge_case_type = EdgeCaseType.UNKNOWN_CODE_PROCEDURE,
                description    = f"{len(coding.unmapped_procedures)} procedure(s) could not be mapped to CPT codes.",
                severity       = Severity.HIGH,
                recommendation = "Manual coding review required for unmapped procedures.",
                affected_codes = coding.unmapped_procedures,
            ))

        return cases

    def _check_conflicting_conditions(self, claim, extraction, coding, rule_result, imaging) -> list[EdgeCase]:
        """HIGH if mutually exclusive conditions are both present."""
        cases = []

        diag_texts = [
            (e.normalized or e.text).lower()
            for e in extraction.diagnoses
        ]

        for term_a, term_b in _CONFLICTING_PAIRS:
            a_found = any(term_a in d for d in diag_texts)
            b_found = any(term_b in d for d in diag_texts)

            if a_found and b_found:
                cases.append(EdgeCase(
                    edge_case_type = EdgeCaseType.MULTIPLE_CONFLICTING_CONDITIONS,
                    description    = f"Conflicting conditions detected: '{term_a}' and '{term_b}' cannot both be primary diagnoses.",
                    severity       = Severity.HIGH,
                    recommendation = "Clinical review required to resolve conflicting diagnoses.",
                    affected_codes = [term_a, term_b],
                ))

        return cases

    def _check_coverage_gap(self, claim, extraction, coding, rule_result, imaging) -> list[EdgeCase]:
        """MEDIUM if there are rule warnings related to coverage that don't outright reject."""
        cases = []

        coverage_warnings = [
            ev for ev in rule_result.warnings
            if "coverage" in ev.rule_id.lower() or "cov" in ev.rule_id.lower()
        ]

        if coverage_warnings:
            cases.append(EdgeCase(
                edge_case_type = EdgeCaseType.COVERAGE_GAP,
                description    = f"{len(coverage_warnings)} coverage warning(s) flagged by rule engine.",
                severity       = Severity.MEDIUM,
                recommendation = "Verify patient coverage and plan eligibility before resubmission.",
                affected_codes = [ev.rule_id for ev in coverage_warnings],
            ))

        return cases

    def _check_image_text_mismatch(self, claim, extraction, coding, rule_result, imaging) -> list[EdgeCase]:
        """
        HIGH if imaging model diagnosis conflicts with text-extracted diagnosis.
        This is the novel edge case — no standard claims system checks this.

        Uses EntityMerger.detect_imaging_text_conflict() to compare signals.
        Only fires when imaging was actually used (not SKIPPED).
        """
        cases = []

        from agent.models.enums import ImagingMode

        if imaging is None or imaging.mode_used == ImagingMode.SKIPPED:
            return cases

        conflict = self._merger.detect_imaging_text_conflict(extraction, imaging)

        if conflict is not None:
            image_signal, text_signal, mismatch_score = conflict

            # Escalate severity based on mismatch score
            severity = Severity.HIGH if mismatch_score >= 0.7 else Severity.MEDIUM

            cases.append(EdgeCase(
                edge_case_type = EdgeCaseType.IMAGE_TEXT_MISMATCH,
                description    = (
                    f"Imaging model ({imaging.mode_used.value}) suggests '{image_signal}' "
                    f"but clinical text extraction suggests '{text_signal}'. "
                    f"Mismatch score: {mismatch_score:.2f}."
                ),
                severity       = severity,
                recommendation = (
                    "Radiologist or specialist review required before adjudication. "
                    "Both imaging and clinical text signals must agree for auto-approval."
                ),
                image_signal   = image_signal,
                text_signal    = text_signal,
                mismatch_score = mismatch_score,
            ))

        return cases
