"""
entity_merger.py — Merges ClinicalExtractor output with ImagingResult and StructuredClinicalData.

Called after Stage 3 extraction to combine three possible entity sources:
  1. Text entities    — extracted by ClinicalExtractor from clinical notes
  2. Imaging entities — produced by ImagingLayer (Stage 2), if present
  3. Structured data  — pre-supplied diagnosis/procedure lists on the claim

Merge rules:
  - Imaging entities take HIGHER confidence weight when they agree with text entities.
  - When imaging and text diagnoses CONFLICT, both are kept and flagged.
    The conflict is later caught by EdgeCaseDetector as IMAGE_TEXT_MISMATCH.
  - Structured data entities are added if not already present in text entities.
  - Deduplication uses normalized text — same normalized form = same entity.
  - Result is a single ExtractionResult with imaging_result attached.
"""

import logging
from difflib import SequenceMatcher
from typing import Optional

from agent.models.enums import EntityCategory
from agent.models.schemas import (
    ClinicalEntity,
    ExtractionResult,
    ImagingResult,
    StructuredClinicalData,
    TokenizedClaimInput,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

_SIMILARITY_THRESHOLD   = 0.80   # Normalized text similarity for dedup
_IMAGING_CONFIDENCE     = 0.92   # Base confidence for imaging-derived entities
_STRUCTURED_CONFIDENCE  = 0.99   # Structured input is assumed pre-validated
_STRUCTURED_SYMPTOM_CONF= 0.90
_IMAGING_BOOST          = 0.03   # Confidence boost when image + text agree


# ─────────────────────────────────────────────
# SIMILARITY HELPER
# ─────────────────────────────────────────────

def _similarity(a: str, b: str) -> float:
    """
    Returns text similarity ratio between two strings (0.0–1.0).
    Used to detect duplicate entities with slightly different wording.
    """
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _is_duplicate(entity: ClinicalEntity, existing: list[ClinicalEntity]) -> bool:
    """Returns True if entity is already represented in existing list."""
    for e in existing:
        if _similarity(
            entity.normalized or entity.text,
            e.normalized or e.text
        ) >= _SIMILARITY_THRESHOLD:
            return True
    return False


# ─────────────────────────────────────────────
# ENTITY MERGER
# ─────────────────────────────────────────────

class EntityMerger:
    """
    Merges entities from three sources into a single ExtractionResult.

    Usage (called by pipeline.py after Stage 3):
        merger = EntityMerger()
        merged = merger.merge(extraction_result, claim)
    """

    def merge(
        self,
        extraction:    ExtractionResult,
        claim:         TokenizedClaimInput,
        imaging_result: Optional[ImagingResult] = None,
    ) -> ExtractionResult:
        """
        Merges text extraction + imaging result + structured data.

        Args:
            extraction:     Output from ClinicalExtractor (Stage 3)
            claim:          TokenizedClaimInput (carries structured_data)
            imaging_result: ImagingResult from Stage 2, or None if skipped

        Returns:
            Merged ExtractionResult with imaging_result attached
        """
        diagnoses  = list(extraction.diagnoses)
        procedures = list(extraction.procedures)
        symptoms   = list(extraction.symptoms)
        medications= list(extraction.medications)

        # ── 1. Merge structured clinical data ──────────────────────────────
        if claim.structured_data:
            diagnoses, procedures, symptoms, medications = self._merge_structured(
                claim.structured_data, diagnoses, procedures, symptoms, medications
            )

        # ── 2. Merge imaging result ─────────────────────────────────────────
        if imaging_result is not None:
            diagnoses = self._merge_imaging(imaging_result, diagnoses, claim.claim_id)

        # ── 3. Boost confidence on agreed entities ──────────────────────────
        if imaging_result is not None:
            diagnoses = self._apply_imaging_boost(imaging_result, diagnoses)

        # ── 4. Recalculate overall confidence ───────────────────────────────
        all_entities = diagnoses + procedures + symptoms + medications
        if all_entities:
            overall = sum(e.confidence for e in all_entities) / len(all_entities)
        else:
            overall = 0.0

        return ExtractionResult(
            diagnoses          = diagnoses,
            procedures         = procedures,
            symptoms           = symptoms,
            medications        = medications,
            normalized_text    = extraction.normalized_text,
            overall_confidence = round(overall, 4),
            imaging_result     = imaging_result,
        )

    # ── Structured data merge ────────────────────────────────────────────────

    def _merge_structured(
        self,
        structured:  StructuredClinicalData,
        diagnoses:   list[ClinicalEntity],
        procedures:  list[ClinicalEntity],
        symptoms:    list[ClinicalEntity],
        medications: list[ClinicalEntity],
    ) -> tuple[list, list, list, list]:
        """Adds structured entities not already present in text extraction."""

        for text in structured.diagnoses:
            entity = ClinicalEntity(
                text        = text,
                category    = EntityCategory.DIAGNOSIS,
                confidence  = _STRUCTURED_CONFIDENCE,
                normalized  = text,
                source_span = "structured_input",
            )
            if not _is_duplicate(entity, diagnoses):
                diagnoses.append(entity)

        for text in structured.procedures:
            entity = ClinicalEntity(
                text        = text,
                category    = EntityCategory.PROCEDURE,
                confidence  = _STRUCTURED_CONFIDENCE,
                normalized  = text,
                source_span = "structured_input",
            )
            if not _is_duplicate(entity, procedures):
                procedures.append(entity)

        for text in structured.symptoms:
            entity = ClinicalEntity(
                text        = text,
                category    = EntityCategory.SYMPTOM,
                confidence  = _STRUCTURED_SYMPTOM_CONF,
                normalized  = text,
                source_span = "structured_input",
            )
            if not _is_duplicate(entity, symptoms):
                symptoms.append(entity)

        for text in structured.medications:
            entity = ClinicalEntity(
                text        = text,
                category    = EntityCategory.MEDICATION,
                confidence  = _STRUCTURED_CONFIDENCE,
                normalized  = text,
                source_span = "structured_input",
            )
            if not _is_duplicate(entity, medications):
                medications.append(entity)

        return diagnoses, procedures, symptoms, medications

    # ── Imaging entity merge ─────────────────────────────────────────────────

    def _merge_imaging(
        self,
        imaging:    ImagingResult,
        diagnoses:  list[ClinicalEntity],
        claim_id:   str,
    ) -> list[ClinicalEntity]:
        """
        Adds imaging-derived diagnosis entity if not already in list.

        The imaging class label (e.g. 'glioma_stage_2') is converted to a
        readable normalized form and added as a DIAGNOSIS entity.

        If a similar diagnosis is already present from text extraction,
        it is kept (the boost step handles the confidence adjustment).
        If no matching diagnosis exists, the imaging entity is added as new.
        """
        if not imaging.class_label:
            return diagnoses

        # Convert snake_case class label to readable form
        readable = imaging.class_label.replace("_", " ").strip()
        normalized = readable

        imaging_entity = ClinicalEntity(
            text        = readable,
            category    = EntityCategory.DIAGNOSIS,
            confidence  = imaging.confidence * _IMAGING_CONFIDENCE,
            normalized  = normalized,
            source_span = f"imaging_model:{imaging.mode_used.value}",
        )

        if _is_duplicate(imaging_entity, diagnoses):
            logger.debug(f"[{claim_id}] Imaging entity '{readable}' matches existing text entity — skipping add, will boost")
        else:
            logger.info(f"[{claim_id}] Imaging entity '{readable}' not in text extraction — adding as new diagnosis")
            diagnoses.append(imaging_entity)

        return diagnoses

    # ── Confidence boosting ──────────────────────────────────────────────────

    def _apply_imaging_boost(
        self,
        imaging:   ImagingResult,
        diagnoses: list[ClinicalEntity],
    ) -> list[ClinicalEntity]:
        """
        Boosts confidence of text-extracted diagnoses that agree with
        the imaging model result. The boost is small (_IMAGING_BOOST = 0.03)
        but measurable — recorded in the audit trail.
        """
        if not imaging.class_label:
            return diagnoses

        imaging_normalized = imaging.class_label.replace("_", " ").strip()

        boosted = []
        for entity in diagnoses:
            en = entity.normalized or entity.text
            sim = _similarity(en, imaging_normalized)
            if sim >= _SIMILARITY_THRESHOLD:
                boosted_conf = min(1.0, entity.confidence + _IMAGING_BOOST)
                boosted.append(ClinicalEntity(
                    text        = entity.text,
                    category    = entity.category,
                    confidence  = boosted_conf,
                    normalized  = entity.normalized,
                    source_span = entity.source_span,
                ))
            else:
                boosted.append(entity)

        return boosted

    # ── Conflict detection (used by EdgeCaseDetector) ────────────────────────

    def detect_imaging_text_conflict(
        self,
        extraction:     ExtractionResult,
        imaging_result: ImagingResult,
    ) -> Optional[tuple[str, str, float]]:
        """
        Checks whether the imaging class label conflicts with text diagnoses.

        Returns (image_signal, text_signal, mismatch_score) if conflict found.
        Returns None if they agree or there are no text diagnoses to compare.

        mismatch_score: 0.0 = identical, 1.0 = completely different.
        Used to populate EdgeCase.mismatch_score for IMAGE_TEXT_MISMATCH.
        """
        if not extraction.diagnoses or not imaging_result.class_label:
            return None

        imaging_normalized = imaging_result.class_label.replace("_", " ").strip()

        # Check similarity against each text diagnosis
        max_similarity = 0.0
        best_text_match = ""

        for entity in extraction.diagnoses:
            en  = entity.normalized or entity.text
            sim = _similarity(en, imaging_normalized)
            if sim > max_similarity:
                max_similarity = sim
                best_text_match = en

        mismatch_score = round(1.0 - max_similarity, 4)

        # Only report as conflict if similarity is low
        if max_similarity < _SIMILARITY_THRESHOLD:
            logger.warning(
                f"IMAGE_TEXT_MISMATCH: imaging='{imaging_normalized}' "
                f"vs text='{best_text_match}' "
                f"similarity={max_similarity:.2f} mismatch={mismatch_score:.2f}"
            )
            return (imaging_normalized, best_text_match, mismatch_score)

        return None
