"""
schemas.py — All typed data contracts for the Healthcare Claims Adjudication Agent.

Every pipeline stage consumes and produces objects defined here.
These are the strict interfaces between stages — changing a field here
is the ONLY place you need to change it system-wide.

Design rules:
  - All fields have explicit types — no bare dicts, no untyped lists.
  - Optional fields use Optional[T] with a default of None — never missing keys.
  - Every output schema has a confidence_score (0.0–1.0).
  - PHI-bearing fields are marked with a # PHI comment — tokenizer targets these.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import uuid

from agent.models.enums import (
    AuditStatus,
    ClaimDecision,
    CodeSystem,
    EdgeCaseType,
    EntityCategory,
    ImagingMode,
    InsurancePlan,
    PipelineStage,
    RuleAction,
    RuleType,
    Severity,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _claim_id() -> str:
    """Auto-generates a claim ID: CLM-XXXXXXXX"""
    return f"CLM-{uuid.uuid4().hex[:8].upper()}"


def _audit_trace_id() -> str:
    """Auto-generates an audit trace ID: AUD-XXXXXXXXXXXX"""
    return f"AUD-{uuid.uuid4().hex[:12].upper()}"


def _now() -> datetime:
    return datetime.utcnow()


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 INPUT — RAW CLAIM (contains PHI)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PatientInfo:
    """
    Raw patient demographics — CONTAINS PHI.
    This object never passes the API gateway.
    PHI tokenizer converts this into TokenizedPatientInfo before pipeline entry.
    """
    patient_id:     str                      # PHI — MRN or system ID
    name:           str                      # PHI — full name
    date_of_birth:  str                      # PHI — ISO date string YYYY-MM-DD
    age:            int                      # PHI if > 89
    sex:            str                      # M / F / Other
    insurance_plan: InsurancePlan            # BASIC / STANDARD / PREMIUM
    policy_number:  str                      # PHI — insurance policy number
    phone:          Optional[str] = None     # PHI
    email:          Optional[str] = None     # PHI
    address:        Optional[str] = None     # PHI


@dataclass
class StructuredClinicalData:
    """
    Optional pre-structured clinical data supplied alongside free text.
    When provided, merged with text-extracted entities in Stage 3.
    """
    diagnoses:   list[str] = field(default_factory=list)
    procedures:  list[str] = field(default_factory=list)
    symptoms:    list[str] = field(default_factory=list)
    medications: list[str] = field(default_factory=list)


@dataclass
class ClaimInput:
    """
    Top-level raw input — CONTAINS PHI via patient_info.
    Accepted by POST /adjudicate but never enters the pipeline directly.
    PHI tokenizer converts this to TokenizedClaimInput at the gateway.
    """
    patient_info:           PatientInfo
    clinical_notes:         str                                  # Free text — may contain PHI
    structured_data:        Optional[StructuredClinicalData] = None
    claim_id:               str = field(default_factory=_claim_id)
    submitted_at:           datetime = field(default_factory=_now)
    imaging_mode:           ImagingMode = ImagingMode.SKIPPED    # Doctor selects at submission
    image_path:             Optional[str] = None                 # Path to uploaded scan (if any)
    precomputed_class:      Optional[str] = None                 # Mode C: pasted class label
    precomputed_confidence: Optional[float] = None              # Mode C: pasted confidence


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 INPUT — TOKENIZED CLAIM (PHI replaced with tokens)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TokenizedPatientInfo:
    """
    PHI-safe version of PatientInfo.
    All PHI fields replaced with vault tokens (e.g. PHI_NAME_a3f9).
    This is what the pipeline sees — never the raw values.
    """
    token_patient_id:   str           # Token replacing patient_id
    token_name:         str           # Token replacing name
    token_dob:          str           # Token replacing date_of_birth
    age_band:           str           # "60s" not exact DOB — safe to send to Claude
    sex:                str
    insurance_plan:     InsurancePlan
    token_policy:       str           # Token replacing policy_number


@dataclass
class TokenizedClaimInput:
    """
    PHI-safe version of ClaimInput.
    This is the ONLY input type accepted by the pipeline.
    Created by PHITokenizer.tokenize(raw_claim) at the API gateway.
    """
    patient_info:           TokenizedPatientInfo
    clinical_notes:         str                                   # PHI replaced with tokens
    structured_data:        Optional[StructuredClinicalData] = None
    claim_id:               str = field(default_factory=_claim_id)
    audit_trace_id:         str = field(default_factory=_audit_trace_id)
    submitted_at:           datetime = field(default_factory=_now)
    imaging_mode:           ImagingMode = ImagingMode.SKIPPED
    image_path:             Optional[str] = None
    precomputed_class:      Optional[str] = None
    precomputed_confidence: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — IMAGING RESULT (optional)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ImagingResult:
    """
    Output of Stage 2 imaging layer.
    Present only when imaging_mode != SKIPPED.
    Fed into Stage 3 (entity merger) and Stage 7 (contradiction detector).
    """
    mode_used:          ImagingMode
    class_label:        str                      # e.g. "glioma_stage_2"
    confidence:         float                    # 0.0 – 1.0
    icd10_suggestion:   Optional[str] = None     # e.g. "C71.1"
    heatmap_path:       Optional[str] = None     # Path to Grad-CAM overlay image
    raw_probabilities:  Optional[dict[str, float]] = None  # All class probabilities
    model_source:       Optional[str] = None     # model filename or "builtin"


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — CLINICAL EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClinicalEntity:
    """
    A single extracted clinical entity from Stage 3.
    Source text is the tokenized text span — never raw PHI.
    """
    text:        str             # Extracted term (from tokenized notes)
    category:    EntityCategory  # DIAGNOSIS / PROCEDURE / SYMPTOM / MEDICATION
    confidence:  float           # 0.0 – 1.0
    source_span: Optional[str] = None  # Original substring that matched
    normalized:  Optional[str] = None  # Canonical form (e.g. "type 2 diabetes mellitus")


@dataclass
class ExtractionResult:
    """
    Output of Stage 3 — aggregated clinical entity extraction.
    """
    diagnoses:          list[ClinicalEntity] = field(default_factory=list)
    procedures:         list[ClinicalEntity] = field(default_factory=list)
    symptoms:           list[ClinicalEntity] = field(default_factory=list)
    medications:        list[ClinicalEntity] = field(default_factory=list)
    normalized_text:    str = ""          # Tokenized + cleaned clinical notes
    overall_confidence: float = 0.0
    imaging_result:     Optional[ImagingResult] = None   # Merged from Stage 2


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — MEDICAL CODING
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CodeMapping:
    """
    A single medical code mapping produced by Stage 4.
    """
    original_text:  str         # The entity text that was mapped
    code:           str         # e.g. "E11.9" or "27447"
    code_system:    CodeSystem  # ICD-10-CM or CPT
    description:    str         # Human-readable code description
    confidence:     float       # 0.0 – 1.0
    is_exact_match: bool = True # False = fuzzy match (lower confidence)


@dataclass
class CodingResult:
    """
    Output of Stage 4 — all ICD-10 and CPT code mappings.
    """
    icd10_codes:          list[CodeMapping] = field(default_factory=list)
    cpt_codes:            list[CodeMapping] = field(default_factory=list)
    unmapped_diagnoses:   list[str] = field(default_factory=list)  # Triggers edge case
    unmapped_procedures:  list[str] = field(default_factory=list)  # Triggers edge case
    overall_confidence:   float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — RAG GUIDELINE RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GuidelinePassage:
    """
    A single retrieved guideline passage from Stage 5 RAG.
    """
    source:       str    # e.g. "NCCN 2024 — Brain Tumors v1.0"
    passage_id:   str    # Unique chunk ID in the vector store
    content:      str    # The retrieved text passage
    relevance:    float  # Cosine similarity score


@dataclass
class RAGResult:
    """
    Output of Stage 5 — retrieved clinical guideline passages.
    Used by the decision engine to justify medical necessity.
    """
    passages:       list[GuidelinePassage] = field(default_factory=list)
    query_used:     str = ""      # The query sent to the vector store
    retrieval_ms:   int = 0       # Latency in milliseconds


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6 — RULE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PolicyRule:
    """
    Definition of a single policy rule — loaded from default_rules.json.
    """
    rule_id:      str
    rule_name:    str
    description:  str
    rule_type:    RuleType
    conditions:   dict              # Type-specific condition parameters
    action:       RuleAction        # REJECT or FLAG_REVIEW on failure
    severity:     Severity


@dataclass
class RuleEvaluation:
    """
    Result of evaluating a single policy rule against a claim.
    """
    rule_id:    str
    rule_name:  str
    rule_type:  RuleType
    passed:     bool
    action:     RuleAction
    reason:     str        # Human-readable explanation
    severity:   Severity


@dataclass
class RuleEngineResult:
    """
    Output of Stage 6 — all rule evaluations aggregated.
    """
    evaluations:       list[RuleEvaluation] = field(default_factory=list)
    blocking_failures: list[RuleEvaluation] = field(default_factory=list)  # REJECT actions
    warnings:          list[RuleEvaluation] = field(default_factory=list)  # FLAG_REVIEW
    all_passed:        bool = False
    rules_evaluated:   int = 0


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 7 — EDGE CASE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EdgeCase:
    """
    A single detected anomaly from Stage 7.
    IMAGE_TEXT_MISMATCH fires when imaging and text diagnoses contradict.
    """
    edge_case_type:   EdgeCaseType
    description:      str
    severity:         Severity
    recommendation:   str
    affected_codes:   list[str] = field(default_factory=list)
    # Image vs text specific fields
    image_signal:     Optional[str] = None   # What the imaging model said
    text_signal:      Optional[str] = None   # What the text extractor said
    mismatch_score:   Optional[float] = None # 0.0 = identical, 1.0 = total contradiction


@dataclass
class EdgeCaseResult:
    """
    Output of Stage 7 — all detected edge cases.
    """
    edge_cases:           list[EdgeCase] = field(default_factory=list)
    has_blocking_issues:  bool = False   # CRITICAL or HIGH severity present
    requires_human_review: bool = False  # Any edge case flags human review
    image_text_mismatch:  bool = False   # Convenience flag for dashboard


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 8–9 — DECISION + EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReasoningStep:
    """
    One step in the explainability reasoning chain.
    Ordered list of these forms the full audit-ready reasoning trace.
    """
    step:        int
    stage:       PipelineStage
    description: str          # What happened at this step
    outcome:     str          # What it produced / concluded
    confidence:  float        # Confidence at this step


@dataclass
class ExplainabilityReport:
    """
    Human-readable + machine-readable reasoning chain for a claim decision.
    Attached to every AdjudicationResult — mandatory, never optional.
    """
    summary:        str                    # One-sentence plain-English decision summary
    reasoning_chain: list[ReasoningStep] = field(default_factory=list)
    key_factors:    list[str] = field(default_factory=list)   # Top factors that drove decision
    risk_flags:     list[str] = field(default_factory=list)   # Any concerns flagged
    guideline_citations: list[str] = field(default_factory=list)  # RAG passages cited


@dataclass
class AdjudicationResult:
    """
    Final output of the full pipeline — returned by POST /adjudicate.
    This is what the dashboard displays and what gets stored in the audit DB.

    audit_trace_id links this result to all 9 stage audit log entries.
    All PHI fields are absent — only token IDs and medical codes present.
    """
    # Core decision
    claim_id:         str
    audit_trace_id:   str
    decision:         ClaimDecision
    confidence_score: float          # 0.0 – 1.0 composite

    # Medical codes
    icd10_codes:      list[CodeMapping] = field(default_factory=list)
    cpt_codes:        list[CodeMapping] = field(default_factory=list)

    # Stage outputs
    rule_evaluations:  list[RuleEvaluation] = field(default_factory=list)
    edge_cases:        list[EdgeCase] = field(default_factory=list)
    rag_passages:      list[GuidelinePassage] = field(default_factory=list)
    imaging_result:    Optional[ImagingResult] = None

    # Explainability
    explainability:   Optional[ExplainabilityReport] = None
    reasons:          list[str] = field(default_factory=list)  # Short human-readable reasons

    # Metadata
    decided_at:       datetime = field(default_factory=_now)
    pipeline_version: str = "1.0.0"


# ─────────────────────────────────────────────────────────────────────────────
# AUDIT LOG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AuditEntry:
    """
    One log entry for a single pipeline stage.
    Stored by AuditLogger — append-only, never updated or deleted.

    input_snapshot and output_snapshot contain ONLY tokenized data.
    Raw PHI is never written to the audit log.
    """
    audit_trace_id:   str
    claim_id:         str
    stage:            PipelineStage
    status:           AuditStatus
    timestamp:        datetime = field(default_factory=_now)
    input_snapshot:   dict = field(default_factory=dict)   # Tokenized stage input summary
    output_snapshot:  dict = field(default_factory=dict)   # Tokenized stage output summary
    duration_ms:      int = 0
    error_message:    Optional[str] = None                 # Populated on AuditStatus.ERROR
