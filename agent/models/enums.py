"""
enums.py — All enumerations for the Healthcare Claims Adjudication Agent.

Every enum in this file is a strict contract used across all pipeline stages.
Adding a new value here is the ONLY change needed to extend any category system-wide.
"""

from enum import Enum


# ─────────────────────────────────────────────
# PIPELINE STAGES
# ─────────────────────────────────────────────

class PipelineStage(str, Enum):
    """
    Identifies each of the 9 pipeline stages.
    Used by AuditLogger to tag every log entry with its originating stage.
    String enum so stage names serialize cleanly to JSON in audit records.
    """
    INGESTION       = "ingestion"          # Stage 1 — raw input + PHI validation
    IMAGING         = "imaging"            # Stage 2 — optional imaging layer
    EXTRACTION      = "extraction"         # Stage 3 — Claude NER
    CODING          = "coding"             # Stage 4 — ICD-10 + CPT mapping
    RAG             = "rag"                # Stage 5 — guideline retrieval
    RULE_RETRIEVAL  = "rule_retrieval"     # Stage 6a — fetch applicable rules
    RULE_EXECUTION  = "rule_execution"     # Stage 6b — evaluate rules
    EDGE_DETECTION  = "edge_detection"     # Stage 7 — anomaly detection
    DECISION        = "decision"           # Stage 8 — final adjudication
    EXPLAINABILITY  = "explainability"     # Stage 9 — reasoning report


# ─────────────────────────────────────────────
# CLAIM DECISION OUTCOMES
# ─────────────────────────────────────────────

class ClaimDecision(str, Enum):
    """
    The three possible outcomes of claim adjudication.
    String enum so decisions serialize cleanly in API responses and audit logs.
    """
    APPROVED      = "approved"       # All rules passed, no blocking issues
    REJECTED      = "rejected"       # One or more blocking rule failures
    NEEDS_REVIEW  = "needs_review"   # Ambiguous — requires human reviewer


# ─────────────────────────────────────────────
# SEVERITY LEVELS
# ─────────────────────────────────────────────

class Severity(str, Enum):
    """
    Severity grading used by both the rule engine and edge case detector.
    Order matters: CRITICAL > HIGH > MEDIUM > LOW.
    Used in confidence penalty calculations in the decision engine.
    """
    LOW      = "low"       # Informational — no action required
    MEDIUM   = "medium"    # Soft flag — review recommended
    HIGH     = "high"      # Hard flag — likely blocks or requires review
    CRITICAL = "critical"  # Blocking — claim cannot proceed without resolution

    def penalty(self) -> float:
        """
        Returns the confidence score penalty applied by the decision engine.
        CRITICAL = -0.50, HIGH = -0.30, MEDIUM = -0.15, LOW = -0.05
        """
        _penalties = {
            Severity.LOW:      0.05,
            Severity.MEDIUM:   0.15,
            Severity.HIGH:     0.30,
            Severity.CRITICAL: 0.50,
        }
        return _penalties[self]

    def is_blocking(self) -> bool:
        """Returns True if this severity level blocks claim approval."""
        return self in (Severity.HIGH, Severity.CRITICAL)


# ─────────────────────────────────────────────
# EDGE CASE TYPES
# ─────────────────────────────────────────────

class EdgeCaseType(str, Enum):
    """
    Classification of anomalies detected by the EdgeCaseDetector (Stage 7).
    Each type maps to a severity level and a recommended action.
    IMAGE_TEXT_MISMATCH is the novel addition — fired when the imaging model
    output contradicts the clinical text diagnosis.
    """
    MISSING_DIAGNOSIS               = "missing_diagnosis"
    MISSING_PROCEDURE               = "missing_procedure"
    UNKNOWN_CODE_DIAGNOSIS          = "unknown_code_diagnosis"
    UNKNOWN_CODE_PROCEDURE          = "unknown_code_procedure"
    LOW_CONFIDENCE_EXTRACTION       = "low_confidence_extraction"
    MULTIPLE_CONFLICTING_CONDITIONS = "multiple_conflicting_conditions"
    COVERAGE_GAP                    = "coverage_gap"
    IMAGE_TEXT_MISMATCH             = "image_text_mismatch"

    def default_severity(self) -> Severity:
        """
        Returns the default severity for each edge case type.
        Can be overridden by the detector based on context.
        """
        _severities = {
            EdgeCaseType.MISSING_DIAGNOSIS:               Severity.CRITICAL,
            EdgeCaseType.MISSING_PROCEDURE:               Severity.HIGH,
            EdgeCaseType.UNKNOWN_CODE_DIAGNOSIS:          Severity.HIGH,
            EdgeCaseType.UNKNOWN_CODE_PROCEDURE:          Severity.HIGH,
            EdgeCaseType.LOW_CONFIDENCE_EXTRACTION:       Severity.MEDIUM,
            EdgeCaseType.MULTIPLE_CONFLICTING_CONDITIONS: Severity.HIGH,
            EdgeCaseType.COVERAGE_GAP:                    Severity.MEDIUM,
            EdgeCaseType.IMAGE_TEXT_MISMATCH:             Severity.HIGH,
        }
        return _severities[self]


# ─────────────────────────────────────────────
# RULE TYPES
# ─────────────────────────────────────────────

class RuleType(str, Enum):
    """
    The five categories of policy rules evaluated by the RuleEngine (Stage 6).
    Each type has a distinct evaluation strategy in rule_engine.py.
    """
    COMPATIBILITY   = "compatibility"
    PRIOR_AUTH      = "prior_auth"
    AGE_RESTRICTION = "age_restriction"
    COVERAGE        = "coverage"
    FREQUENCY       = "frequency"


class RuleAction(str, Enum):
    """
    The action a rule triggers when it fails.
    REJECT = blocking failure. FLAG_REVIEW = routes to human review.
    """
    REJECT      = "reject"
    FLAG_REVIEW = "flag_review"


# ─────────────────────────────────────────────
# INSURANCE PLAN TYPES
# ─────────────────────────────────────────────

class InsurancePlan(str, Enum):
    """
    Patient insurance plan tiers.
    Directly used by coverage rules in the rule engine.
    """
    BASIC    = "basic"
    STANDARD = "standard"
    PREMIUM  = "premium"


# ─────────────────────────────────────────────
# PHI FIELD TYPES
# ─────────────────────────────────────────────

class PHIFieldType(str, Enum):
    """
    The 18 HIPAA-defined Protected Health Information identifiers.
    Used by phi_fields.py and phi_tokenizer.py to detect and tokenize PHI.
    Each value becomes the prefix of the generated token:
    e.g. PHIFieldType.NAME → token "PHI_NAME_a3f9"
    """
    NAME           = "NAME"
    DATE           = "DATE"
    AGE            = "AGE"
    PHONE          = "PHONE"
    FAX            = "FAX"
    EMAIL          = "EMAIL"
    SSN            = "SSN"
    MRN            = "MRN"
    PLAN_NUMBER    = "PLAN_NUMBER"
    ACCOUNT_NUMBER = "ACCOUNT"
    CERTIFICATE    = "CERTIFICATE"
    VIN            = "VIN"
    DEVICE_ID      = "DEVICE_ID"
    URL            = "URL"
    IP_ADDRESS     = "IP"
    BIOMETRIC      = "BIOMETRIC"
    PHOTO          = "PHOTO"
    OTHER_ID       = "OTHER_ID"


# ─────────────────────────────────────────────
# VAULT BACKEND
# ─────────────────────────────────────────────

class VaultBackend(str, Enum):
    """
    Selects the PHI token vault storage backend.
    Controlled by VAULT_BACKEND env var — zero code change to switch.
    """
    SQLITE   = "sqlite"
    POSTGRES = "postgres"


# ─────────────────────────────────────────────
# AUDIT LOG STATUS
# ─────────────────────────────────────────────

class AuditStatus(str, Enum):
    """
    Status of each pipeline stage entry in the audit log.
    Every stage logs either SUCCESS or ERROR — never silent.
    """
    SUCCESS = "success"
    ERROR   = "error"


# ─────────────────────────────────────────────
# IMAGING MODE
# ─────────────────────────────────────────────

class ImagingMode(str, Enum):
    """
    The three doctor-controlled imaging input modes (Stage 2).
    Defined here now so schemas.py can reference it.
    Fully implemented in the imaging sprint.
    """
    BUILTIN     = "builtin"      # Mode A — run built-in 8-class model
    CUSTOM      = "custom"       # Mode B — doctor uploads their own model
    PRECOMPUTED = "precomputed"  # Mode C — doctor pastes model output directly
    SKIPPED     = "skipped"      # No imaging input — stage bypassed


# ─────────────────────────────────────────────
# ENTITY CATEGORIES
# ─────────────────────────────────────────────

class EntityCategory(str, Enum):
    """
    Categories of clinical entities extracted by the ClinicalExtractor (Stage 3).
    Matches the structured output schema sent to Claude for NER.
    """
    DIAGNOSIS  = "diagnosis"
    PROCEDURE  = "procedure"
    SYMPTOM    = "symptom"
    MEDICATION = "medication"


# ─────────────────────────────────────────────
# CODE SYSTEMS
# ─────────────────────────────────────────────

class CodeSystem(str, Enum):
    """
    Medical coding systems used by the MedicalCoder (Stage 4).
    """
    ICD10_CM = "ICD-10-CM"
    CPT      = "CPT"
