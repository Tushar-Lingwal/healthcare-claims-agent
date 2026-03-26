"""
code_validator.py — Anti-hallucination validation for medical codes.

Validates CodeMappings produced by MedicalCoder before they enter
the rule engine. Catches:
  1. Hallucinated codes  — format looks right but code doesn't exist in DB
  2. Format violations   — wrong structure (e.g. "C7" instead of "C71.9")
  3. Cross-system errors — ICD-10 code stored as CPT or vice versa
  4. Low-confidence codes — fuzzy matches below accept threshold

Every rejected code is logged to the audit trail as a validation failure.
The pipeline continues with only the valid codes — it does not crash.

ICD-10-CM format rules:
  - 3–7 characters
  - Starts with a letter (A–Z)
  - Followed by 2 digits
  - Optional decimal + 1–4 alphanumeric chars
  - Example: C71.9, E11.9, G30.9, M19.90

CPT format rules:
  - Exactly 5 digits
  - Optional 1-char category suffix (F, T, U, M)
  - Example: 70553, 96413, 27447
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional

from agent.models.enums import CodeSystem
from agent.models.schemas import CodeMapping, CodingResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# REGEX PATTERNS
# ─────────────────────────────────────────────

_ICD10_PATTERN = re.compile(
    r'^[A-Z][0-9]{2}(\.[0-9A-Z]{1,4})?$',
    re.IGNORECASE
)

_CPT_PATTERN = re.compile(
    r'^\d{5}[FTUM]?$'
)

# Minimum confidence to accept a code mapping
_MIN_CONFIDENCE = 0.65


# ─────────────────────────────────────────────
# VALIDATION RESULT
# ─────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Result of validating a single CodeMapping."""
    code:       str
    is_valid:   bool
    reason:     Optional[str] = None   # Why it failed (None if valid)

    def __repr__(self):
        status = "VALID" if self.is_valid else f"INVALID ({self.reason})"
        return f"ValidationResult({self.code}: {status})"


# ─────────────────────────────────────────────
# CODE VALIDATOR
# ─────────────────────────────────────────────

class CodeValidator:
    """
    Validates CodeMappings from MedicalCoder before pipeline continues.

    Usage (called by pipeline or medical_coder post-processing):
        validator = CodeValidator()
        clean_result = validator.validate(coding_result)
    """

    def __init__(self, strict_db_check: bool = True):
        """
        Args:
            strict_db_check: If True, codes must exist in the known DB.
                             If False, only format validation is applied.
                             Set to False for production with full CMS DB.
        """
        self._strict = strict_db_check
        if strict_db_check:
            from agent.coders.medical_coder import MedicalCoder
            self._valid_icd10 = MedicalCoder.get_all_icd10_codes()
            self._valid_cpt   = MedicalCoder.get_all_cpt_codes()
        else:
            self._valid_icd10 = set()
            self._valid_cpt   = set()

    # ── Primary entry point ──────────────────────────────────────────────────

    def validate(self, coding_result: CodingResult) -> CodingResult:
        """
        Validates all codes in a CodingResult.
        Removes invalid codes and logs each failure.
        Returns a new CodingResult with only validated codes.
        """
        valid_icd10, invalid_icd10 = self._filter_codes(
            coding_result.icd10_codes, CodeSystem.ICD10_CM
        )
        valid_cpt, invalid_cpt = self._filter_codes(
            coding_result.cpt_codes, CodeSystem.CPT
        )

        # Log all invalid codes
        for cm in invalid_icd10 + invalid_cpt:
            logger.warning(
                f"Code validation FAILED: {cm.code} ({cm.code_system.value}) "
                f"from '{cm.original_text}' — removed from coding result"
            )

        # Recalculate overall confidence with only valid codes
        all_valid = valid_icd10 + valid_cpt
        overall = (
            sum(c.confidence for c in all_valid) / len(all_valid)
            if all_valid else 0.0
        )

        return CodingResult(
            icd10_codes          = valid_icd10,
            cpt_codes            = valid_cpt,
            unmapped_diagnoses   = coding_result.unmapped_diagnoses,
            unmapped_procedures  = coding_result.unmapped_procedures,
            overall_confidence   = round(overall, 4),
        )

    # ── Filter by code system ────────────────────────────────────────────────

    def _filter_codes(
        self,
        codes:      list[CodeMapping],
        system:     CodeSystem,
    ) -> tuple[list[CodeMapping], list[CodeMapping]]:
        """
        Splits codes into (valid, invalid) lists.
        """
        valid   = []
        invalid = []
        for cm in codes:
            vr = self._validate_one(cm, system)
            if vr.is_valid:
                valid.append(cm)
            else:
                invalid.append(cm)
        return valid, invalid

    # ── Single code validation ───────────────────────────────────────────────

    def _validate_one(
        self,
        cm:     CodeMapping,
        system: CodeSystem,
    ) -> ValidationResult:
        """
        Validates a single CodeMapping against:
          1. Confidence threshold
          2. Format (regex)
          3. Cross-system check
          4. DB existence (if strict mode)
        """
        code = cm.code.strip().upper()

        # 1. Confidence threshold
        if cm.confidence < _MIN_CONFIDENCE:
            return ValidationResult(
                code     = code,
                is_valid = False,
                reason   = f"Confidence {cm.confidence:.2f} below minimum {_MIN_CONFIDENCE}",
            )

        # 2. Format validation + cross-system check
        if system == CodeSystem.ICD10_CM:
            if _CPT_PATTERN.match(code):
                return ValidationResult(
                    code     = code,
                    is_valid = False,
                    reason   = "CPT format code stored as ICD-10 (cross-system error)",
                )
            if not _ICD10_PATTERN.match(code):
                return ValidationResult(
                    code     = code,
                    is_valid = False,
                    reason   = f"Invalid ICD-10-CM format: '{code}'",
                )

        elif system == CodeSystem.CPT:
            if _ICD10_PATTERN.match(code) and not _CPT_PATTERN.match(code):
                return ValidationResult(
                    code     = code,
                    is_valid = False,
                    reason   = "ICD-10 format code stored as CPT (cross-system error)",
                )
            if not _CPT_PATTERN.match(code):
                return ValidationResult(
                    code     = code,
                    is_valid = False,
                    reason   = f"Invalid CPT format: '{code}'",
                )

        # 3. DB existence check (strict mode only)
        if self._strict:
            if system == CodeSystem.ICD10_CM and code not in self._valid_icd10:
                return ValidationResult(
                    code     = code,
                    is_valid = False,
                    reason   = f"ICD-10 code '{code}' not found in validated code database",
                )
            if system == CodeSystem.CPT and code not in self._valid_cpt:
                return ValidationResult(
                    code     = code,
                    is_valid = False,
                    reason   = f"CPT code '{code}' not found in validated code database",
                )

        return ValidationResult(code=code, is_valid=True)

    # ── Standalone validators (used in tests + api input validation) ─────────

    @staticmethod
    def is_valid_icd10_format(code: str) -> bool:
        """Returns True if code matches ICD-10-CM format regex."""
        return bool(_ICD10_PATTERN.match(code.strip().upper()))

    @staticmethod
    def is_valid_cpt_format(code: str) -> bool:
        """Returns True if code matches CPT format regex."""
        return bool(_CPT_PATTERN.match(code.strip().upper()))

    def validate_icd10(self, code: str) -> ValidationResult:
        """Validates a single ICD-10 code string."""
        cm = CodeMapping(
            original_text  = code,
            code           = code,
            code_system    = CodeSystem.ICD10_CM,
            description    = "",
            confidence     = 1.0,
            is_exact_match = True,
        )
        return self._validate_one(cm, CodeSystem.ICD10_CM)

    def validate_cpt(self, code: str) -> ValidationResult:
        """Validates a single CPT code string."""
        cm = CodeMapping(
            original_text  = code,
            code           = code,
            code_system    = CodeSystem.CPT,
            description    = "",
            confidence     = 1.0,
            is_exact_match = True,
        )
        return self._validate_one(cm, CodeSystem.CPT)
