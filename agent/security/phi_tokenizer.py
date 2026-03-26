"""
phi_tokenizer.py — Tokenizes PHI in ClaimInput → TokenizedClaimInput.

This is the ONLY file api.py needs to call. Flow:
  ClaimInput (real PHI) → tokenize() → TokenizedClaimInput (tokens only)
  AdjudicationResult    → detokenize_result() → real names restored (auth users only)
"""

import re
from typing import Optional

from agent.models.enums import PHIFieldType
from agent.models.schemas import (
    ClaimInput, TokenizedClaimInput, TokenizedPatientInfo,
    PatientInfo, AdjudicationResult,
)
from agent.security.phi_fields import scan_all, AUTO_TOKENIZE_THRESHOLD
from agent.security.token_vault import PHIVault, get_vault

_TOKEN_PATTERN = re.compile(r'PHI_[A-Z_]+_[0-9a-f]{8}')


def _age_to_band(age: int) -> str:
    """
    Converts exact age to a decade band — safe to send to Claude.
    67 → '60s', 43 → '40s', 91 → '90s+'
    Ages >89 are HIPAA PHI; all ages use bands for consistency.
    """
    if age >= 90:
        return "90s+"
    return f"{(age // 10) * 10}s"


class PHITokenizer:
    """
    Tokenizes a raw ClaimInput into a TokenizedClaimInput safe for pipeline entry.

    Usage in api.py (one call — that's it):
        tokenizer = PHITokenizer()
        tokenized = tokenizer.tokenize(raw_claim)
        result    = pipeline.process(tokenized)
        output    = tokenizer.detokenize_result(result)  # authorized users only
    """

    def __init__(self, vault: Optional[PHIVault] = None):
        self._vault = vault or get_vault()

    # ── Primary entry point ─────────────────────────────────────────────────

    def tokenize(self, claim: ClaimInput) -> TokenizedClaimInput:
        """
        Converts raw ClaimInput → TokenizedClaimInput.
          1. Tokenizes all PatientInfo fields (explicit, field-by-field).
          2. Tokenizes free-text clinical_notes (regex scan).
          3. Returns safe TokenizedClaimInput for pipeline.
        """
        claim_id = claim.claim_id
        return TokenizedClaimInput(
            patient_info           = self._tokenize_patient(claim.patient_info, claim_id),
            clinical_notes         = self._tokenize_text(claim.clinical_notes, claim_id),
            structured_data        = claim.structured_data,
            claim_id               = claim_id,
            submitted_at           = claim.submitted_at,
            imaging_mode           = claim.imaging_mode,
            image_path             = claim.image_path,
            precomputed_class      = claim.precomputed_class,
            precomputed_confidence = claim.precomputed_confidence,
        )

    # ── Patient field tokenization ──────────────────────────────────────────

    def _tokenize_patient(self, patient: PatientInfo, claim_id: str) -> TokenizedPatientInfo:
        """Explicitly tokenizes every PHI-bearing field on PatientInfo."""
        return TokenizedPatientInfo(
            token_patient_id = self._store(PHIFieldType.MRN,         patient.patient_id,   claim_id),
            token_name       = self._store(PHIFieldType.NAME,        patient.name,          claim_id),
            token_dob        = self._store(PHIFieldType.DATE,        patient.date_of_birth, claim_id),
            age_band         = _age_to_band(patient.age),  # decade band — not exact age
            sex              = patient.sex,                 # not a HIPAA identifier
            insurance_plan   = patient.insurance_plan,      # plan tier — not account number
            token_policy     = self._store(PHIFieldType.PLAN_NUMBER, patient.policy_number, claim_id),
        )

    # ── Free-text tokenization ──────────────────────────────────────────────

    def _tokenize_text(self, text: str, claim_id: str) -> str:
        """
        Scans free-text clinical notes for PHI using scan_all(),
        replaces each match with a vault token.

        Replacement is right-to-left to preserve character offsets.
        Only auto-tokenizes fields at or above AUTO_TOKENIZE_THRESHOLD confidence.
        Low-confidence matches (URLs, OTHER_ID) are left for human review.
        """
        matches = [
            m for m in scan_all(text)
            if m[4] >= AUTO_TOKENIZE_THRESHOLD  # m[4] is confidence
        ]

        if not matches:
            return text

        result = text
        for phi_type, start, end, matched, _ in reversed(matches):
            token  = self._store(phi_type, matched.strip(), claim_id)
            result = result[:start] + token + result[end:]

        return result

    # ── Vault helper ────────────────────────────────────────────────────────

    def _store(self, phi_type: PHIFieldType, value: str, claim_id: str) -> str:
        return self._vault.store(phi_type, value, claim_id)

    # ── De-tokenization (output layer only) ─────────────────────────────────

    def detokenize_text(self, text: str) -> str:
        """
        Replaces PHI tokens in text with real values from vault.
        Only called at output layer for authorized users.
        Pipeline NEVER calls this.
        """
        tokens = _TOKEN_PATTERN.findall(text)
        result = text
        for token in set(tokens):
            real = self._vault.retrieve(token)
            if real is not None:
                result = result.replace(token, real)
        return result

    def detokenize_result(self, result: AdjudicationResult) -> AdjudicationResult:
        """
        Restores PHI in AdjudicationResult for display to authorized user.
        De-tokenizes explainability summary and reason strings only.
        Medical codes and audit fields stay tokenized — clinically sufficient.
        """
        if result.explainability and result.explainability.summary:
            result.explainability.summary = self.detokenize_text(
                result.explainability.summary
            )
        result.reasons = [self.detokenize_text(r) for r in result.reasons]
        return result

    def delete_claim_phi(self, claim_id: str) -> int:
        """Deletes all PHI tokens for a claim. Call on retention policy expiry."""
        return self._vault.delete_claim(claim_id)
