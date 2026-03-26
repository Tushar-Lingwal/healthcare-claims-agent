"""
phi_fields.py — 18 HIPAA-defined PHI identifier definitions with regex patterns.

This file is the single source of truth for WHAT counts as PHI in this system.
The PHITokenizer (File #6) uses these definitions to scan and replace PHI.

HIPAA Safe Harbor method defines 18 identifier categories that must be removed
or replaced before health information is considered de-identified.
Reference: 45 CFR § 164.514(b)(2)

Design rules:
  - Each PHIField has ONE primary regex (catches the most common format).
  - Secondary patterns handle common variants (abbreviations, alternate formats).
  - All patterns are case-insensitive by default.
  - Patterns are ordered longest-match first to prevent partial replacements.
  - NO pattern should ever match a medical code (ICD-10, CPT) — those must pass through.
  - Each field has a CONFIDENCE score (0.0-1.0) — how certain we are a match is truly PHI.
    High confidence (>0.9) = auto-tokenize. Low confidence (<0.7) = flag for review.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from agent.models.enums import PHIFieldType, Severity


# ─────────────────────────────────────────────
# CORE DATA STRUCTURE
# ─────────────────────────────────────────────

@dataclass
class PHIField:
    """
    Definition of one HIPAA PHI identifier category.

    field_type      : The PHIFieldType enum this field belongs to.
    display_name    : Human-readable name for dashboard and audit logs.
    description     : What this identifier is and why it's PHI.
    primary_pattern : Main compiled regex — catches the most common format.
    alt_patterns    : Additional compiled regexes for common variants.
    confidence      : How certain a regex match is truly PHI (0.0–1.0).
    severity        : How critical a leak of this field type would be.
    examples        : Real-looking examples for test generation.
    safe_exceptions : Patterns that look like PHI but are NOT (e.g. ICD codes).
    """
    field_type:       PHIFieldType
    display_name:     str
    description:      str
    primary_pattern:  re.Pattern
    alt_patterns:     list[re.Pattern] = field(default_factory=list)
    confidence:       float = 0.95
    severity:         Severity = Severity.CRITICAL
    examples:         list[str] = field(default_factory=list)
    safe_exceptions:  list[re.Pattern] = field(default_factory=list)

    def all_patterns(self) -> list[re.Pattern]:
        """Returns primary + all alt patterns in order."""
        return [self.primary_pattern] + self.alt_patterns

    def scan(self, text: str) -> list[tuple[int, int, str]]:
        """
        Scans text for all matches of this PHI field.
        Returns list of (start, end, matched_text) tuples.
        Filters out safe exceptions before returning.
        """
        matches = []
        for pattern in self.all_patterns():
            for m in pattern.finditer(text):
                matched = m.group()
                # Skip if it matches a safe exception
                is_exception = any(
                    exc.fullmatch(matched.strip())
                    for exc in self.safe_exceptions
                )
                if not is_exception:
                    matches.append((m.start(), m.end(), matched))

        # Deduplicate overlapping matches — keep longest
        matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))
        deduped = []
        last_end = -1
        for start, end, text_match in matches:
            if start >= last_end:
                deduped.append((start, end, text_match))
                last_end = end
        return deduped


# ─────────────────────────────────────────────
# FLAGS SHARED ACROSS ALL PATTERNS
# ─────────────────────────────────────────────

_F = re.IGNORECASE


# ─────────────────────────────────────────────
# SAFE EXCEPTION PATTERNS (not PHI)
# These prevent ICD-10 and CPT codes from being
# mistakenly tokenized as PHI identifiers.
# ─────────────────────────────────────────────

# ICD-10: letter + 2 digits + optional dot + optional 1-4 chars
# e.g. E11.9, C71.1, Z23, M19.90
_ICD10_EXCEPTION = re.compile(
    r'^[A-Z]\d{2}(\.\d{1,4})?$', re.IGNORECASE
)

# CPT: exactly 5 digits, optionally with F/T/U/M modifier
# e.g. 27447, 96413, 70553
_CPT_EXCEPTION = re.compile(
    r'^\d{5}[FTUM]?$'
)

# NPI (10-digit provider ID) — not patient PHI
_NPI_EXCEPTION = re.compile(r'^1[0-9]{9}$')


# ─────────────────────────────────────────────
# THE 18 PHI FIELD DEFINITIONS
# ─────────────────────────────────────────────

PHI_FIELDS: list[PHIField] = [

    # ── 1. NAMES ──────────────────────────────────────────────────────────────
    # Most complex field — names appear everywhere in clinical notes.
    # Strategy: match "Firstname Lastname" patterns, known titles, and
    # "Patient: Name" / "Patient Name:" labeled patterns.
    PHIField(
        field_type=PHIFieldType.NAME,
        display_name="Patient name",
        description="Full name, first name, or last name of the patient",
        primary_pattern=re.compile(
            r'\b(?:patient|pt|name)\s*:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})(?=\s*\n|\s*,|\s*DOB|\s*$)',
            _F
        ),
        alt_patterns=[
            # "Dr. Jane Smith" / "Mr. John Doe" / "Ms. Alice Brown"
            re.compile(
                r'\b(?:Dr\.?|Mr\.?|Mrs\.?|Ms\.?|Miss)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
                _F
            ),
            # Labeled: "Name: Jane Smith" or "Patient Name: Jane Smith"
            re.compile(
                r'\b(?:patient\s+)?name\s*:\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
                _F
            ),
        ],
        confidence=0.85,
        severity=Severity.CRITICAL,
        examples=["Jane Smith", "Robert Chen", "Maria Garcia", "Patient: Linda Taylor"],
    ),

    # ── 2. DATES ──────────────────────────────────────────────────────────────
    # Dates directly related to an individual — DOB, admission, discharge, death.
    # Year-only dates (e.g. "2024") are NOT PHI — only specific dates.
    PHIField(
        field_type=PHIFieldType.DATE,
        display_name="Individual date",
        description="Dates directly related to the individual: DOB, admission, discharge",
        primary_pattern=re.compile(
            # ISO format: 1964-03-12
            r'\b(19|20)\d{2}[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b'
        ),
        alt_patterns=[
            # US format: 03/12/1964 or 3-12-1964
            re.compile(
                r'\b(0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])[-/](19|20)\d{2}\b'
            ),
            # Long format: March 12, 1964 / 12 March 1964
            re.compile(
                r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
                r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|'
                r'Dec(?:ember)?)\s+\d{1,2},?\s+(19|20)\d{2}\b',
                _F
            ),
            # Labeled: DOB: 1964-03-12
            re.compile(
                r'\b(?:dob|date\s+of\s+birth|birth\s+date|d\.o\.b\.?)\s*:?\s*'
                r'([\d]{1,2}[-/][\d]{1,2}[-/][\d]{2,4}|[\d]{4}[-/][\d]{1,2}[-/][\d]{1,2})',
                _F
            ),
            # Admission / discharge dates
            re.compile(
                r'\b(?:admitted|admission|discharged?|discharge\s+date)\s*:?\s*'
                r'([\d]{1,2}[-/][\d]{1,2}[-/][\d]{2,4})',
                _F
            ),
        ],
        confidence=0.95,
        severity=Severity.CRITICAL,
        examples=["1964-03-12", "03/12/1964", "March 12, 1964", "DOB: 1964-03-12"],
    ),

    # ── 3. AGE (over 89) ──────────────────────────────────────────────────────
    # Ages 90+ are PHI because they are specific enough to identify individuals.
    # Ages under 90 are safe and intentionally NOT matched.
    PHIField(
        field_type=PHIFieldType.AGE,
        display_name="Age over 89",
        description="Exact age if 90 or older — specific enough to identify",
        primary_pattern=re.compile(
            r'\b(?:age[d]?|aged)\s*:?\s*(9\d|1[0-9]{2})\s*(?:years?|yrs?|yo|y\.o\.)?',
            _F
        ),
        alt_patterns=[
            re.compile(r'\b(9\d|1[0-9]{2})[- ]?(?:year[- ]?old|yo|y\.o\.)\b', _F),
        ],
        confidence=0.90,
        severity=Severity.HIGH,
        examples=["age 92", "91-year-old", "aged 95", "100 yo"],
    ),

    # ── 4. PHONE NUMBERS ──────────────────────────────────────────────────────
    PHIField(
        field_type=PHIFieldType.PHONE,
        display_name="Phone number",
        description="Telephone numbers including mobile, home, work",
        primary_pattern=re.compile(
            # (555) 123-4567 or 555-123-4567 or 555.123.4567
            r'\b(\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b'
        ),
        alt_patterns=[
            # Labeled: Phone: 555-123-4567
            re.compile(
                r'\b(?:phone|tel|telephone|mobile|cell|fax)\s*:?\s*'
                r'(\+?[\d\s\-().]{10,15})',
                _F
            ),
        ],
        confidence=0.92,
        severity=Severity.HIGH,
        examples=["(555) 123-4567", "555-123-4567", "+1 555 123 4567"],
    ),

    # ── 5. FAX NUMBERS ────────────────────────────────────────────────────────
    PHIField(
        field_type=PHIFieldType.FAX,
        display_name="Fax number",
        description="Fax numbers",
        primary_pattern=re.compile(
            r'\b(?:fax)\s*:?\s*(\+?[\d\s\-().]{10,15})',
            _F
        ),
        confidence=0.95,
        severity=Severity.MEDIUM,
        examples=["Fax: 555-987-6543"],
    ),

    # ── 6. EMAIL ADDRESSES ────────────────────────────────────────────────────
    PHIField(
        field_type=PHIFieldType.EMAIL,
        display_name="Email address",
        description="Electronic mail addresses",
        primary_pattern=re.compile(
            r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'
        ),
        confidence=0.98,
        severity=Severity.HIGH,
        examples=["jane.smith@email.com", "patient@hospital.org"],
    ),

    # ── 7. SOCIAL SECURITY NUMBERS ────────────────────────────────────────────
    PHIField(
        field_type=PHIFieldType.SSN,
        display_name="Social Security Number",
        description="US Social Security Numbers — 9 digits in AAA-BB-CCCC format",
        primary_pattern=re.compile(
            r'\b(?!000|666|9\d{2})\d{3}[-\s]?(?!00)\d{2}[-\s]?(?!0000)\d{4}\b'
        ),
        alt_patterns=[
            re.compile(r'\b(?:ssn|social\s+security)\s*:?\s*\d{3}[-\s]?\d{2}[-\s]?\d{4}\b', _F),
        ],
        confidence=0.97,
        severity=Severity.CRITICAL,
        examples=["123-45-6789", "SSN: 123-45-6789"],
    ),

    # ── 8. MEDICAL RECORD NUMBERS ─────────────────────────────────────────────
    PHIField(
        field_type=PHIFieldType.MRN,
        display_name="Medical record number",
        description="Medical record numbers, patient IDs, encounter numbers",
        primary_pattern=re.compile(
            r'\b(?:mrn|medical\s+record(?:\s+number)?|patient\s+id|encounter)\s*[:#]?\s*'
            r'([A-Z0-9\-]{4,20})\b',
            _F
        ),
        alt_patterns=[
            # Standalone MRN-format patterns: MRN-12345 or MR-98234-A
            re.compile(r'\bMR[N]?[-#]?\s*(\d{4,10}[A-Z]?)\b', _F),
            # Labeled with hash: MRN #98234
            re.compile(r'\bMRN\s*#\s*(\d{4,10})\b', _F),
        ],
        confidence=0.93,
        severity=Severity.CRITICAL,
        examples=["MRN: 98234-A", "MRN #12345", "Patient ID: PT-00123", "MR-98234"],
        safe_exceptions=[_ICD10_EXCEPTION, _CPT_EXCEPTION],
    ),

    # ── 9. HEALTH PLAN BENEFICIARY NUMBERS ────────────────────────────────────
    PHIField(
        field_type=PHIFieldType.PLAN_NUMBER,
        display_name="Health plan number",
        description="Insurance plan numbers, member IDs, beneficiary numbers",
        primary_pattern=re.compile(
            r'\b(?:policy|plan|member|beneficiary|insurance\s+id)\s*(?:number|no|#|id)?\s*'
            r'[:#]?\s*([A-Z0-9\-]{5,20})\b',
            _F
        ),
        alt_patterns=[
            re.compile(r'\bPOL[-#]?\s*([A-Z0-9\-]{4,15})\b', _F),
        ],
        confidence=0.88,
        severity=Severity.CRITICAL,
        examples=["Policy: POL-99234", "Member ID: MBR-12345", "Plan #: A1234567"],
    ),

    # ── 10. ACCOUNT NUMBERS ───────────────────────────────────────────────────
    PHIField(
        field_type=PHIFieldType.ACCOUNT_NUMBER,
        display_name="Account number",
        description="Account numbers associated with the individual",
        primary_pattern=re.compile(
            r'\b(?:account|acct)\s*(?:number|no|#)?\s*[:#]?\s*([A-Z0-9\-]{4,20})\b',
            _F
        ),
        confidence=0.85,
        severity=Severity.HIGH,
        examples=["Account #: 99234-A", "Acct: 12345678"],
    ),

    # ── 11. CERTIFICATE / LICENSE NUMBERS ─────────────────────────────────────
    PHIField(
        field_type=PHIFieldType.CERTIFICATE,
        display_name="Certificate / license number",
        description="Certificate or license numbers associated with the individual",
        primary_pattern=re.compile(
            r'\b(?:certificate|license|licence|cert)\s*(?:number|no|#|id)?\s*'
            r'[:#]?\s*([A-Z0-9\-]{4,20})\b',
            _F
        ),
        confidence=0.80,
        severity=Severity.MEDIUM,
        examples=["License #: DL-123456", "Certificate: CERT-9876"],
    ),

    # ── 12. VEHICLE IDENTIFIERS ───────────────────────────────────────────────
    PHIField(
        field_type=PHIFieldType.VIN,
        display_name="Vehicle identifier",
        description="Vehicle identification numbers, license plates",
        primary_pattern=re.compile(
            # Standard 17-char VIN
            r'\b[A-HJ-NPR-Z0-9]{17}\b'
        ),
        alt_patterns=[
            re.compile(r'\b(?:vin|vehicle\s+id)\s*[:#]?\s*([A-HJ-NPR-Z0-9]{17})\b', _F),
            # License plates — US format (2-3 letters + 3-4 digits or variations)
            re.compile(r'\b(?:plate|license\s+plate)\s*[:#]?\s*([A-Z]{2,3}\d{3,4})\b', _F),
        ],
        confidence=0.88,
        severity=Severity.MEDIUM,
        examples=["1HGCM82633A123456", "VIN: 2FAFP71W45X123456"],
    ),

    # ── 13. DEVICE IDENTIFIERS ────────────────────────────────────────────────
    PHIField(
        field_type=PHIFieldType.DEVICE_ID,
        display_name="Device identifier",
        description="Device identifiers, serial numbers for implanted or personal devices",
        primary_pattern=re.compile(
            r'\b(?:device|serial|implant|pacemaker|defibrillator)\s*'
            r'(?:id|number|no|serial|sn)?\s*[:#]?\s*([A-Z0-9\-]{6,20})\b',
            _F
        ),
        confidence=0.82,
        severity=Severity.MEDIUM,
        examples=["Device ID: DEV-123456", "Serial: SN-987654"],
    ),

    # ── 14. WEB URLS ──────────────────────────────────────────────────────────
    PHIField(
        field_type=PHIFieldType.URL,
        display_name="Web URL",
        description="Web URLs that could identify an individual",
        primary_pattern=re.compile(
            r'\bhttps?://[^\s<>"\']+',
            _F
        ),
        alt_patterns=[
            re.compile(r'\bwww\.[a-z0-9\-]+\.[a-z]{2,}[^\s]*', _F),
        ],
        confidence=0.75,
        severity=Severity.MEDIUM,
        examples=["https://patientportal.hospital.com/jane-smith", "www.example.com/profile"],
    ),

    # ── 15. IP ADDRESSES ──────────────────────────────────────────────────────
    PHIField(
        field_type=PHIFieldType.IP_ADDRESS,
        display_name="IP address",
        description="Internet Protocol addresses",
        primary_pattern=re.compile(
            # IPv4
            r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}'
            r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b'
        ),
        alt_patterns=[
            # IPv6 (simplified)
            re.compile(r'\b([0-9a-f]{1,4}:){7}[0-9a-f]{1,4}\b', _F),
        ],
        confidence=0.90,
        severity=Severity.MEDIUM,
        examples=["192.168.1.100", "10.0.0.1"],
    ),

    # ── 16. BIOMETRIC IDENTIFIERS ─────────────────────────────────────────────
    PHIField(
        field_type=PHIFieldType.BIOMETRIC,
        display_name="Biometric identifier",
        description="Fingerprints, retinal scans, voice prints — usually appear as file references",
        primary_pattern=re.compile(
            r'\b(?:fingerprint|retinal?\s+scan|voice\s*print|biometric)\s*'
            r'(?:id|file|scan|data|sample)?\s*[:#]?\s*([A-Z0-9\-_.]{4,40})\b',
            _F
        ),
        confidence=0.88,
        severity=Severity.CRITICAL,
        examples=["Fingerprint ID: FP-12345", "Biometric: BIO-9876.dat"],
    ),

    # ── 17. FULL-FACE PHOTOS / IMAGES ─────────────────────────────────────────
    PHIField(
        field_type=PHIFieldType.PHOTO,
        display_name="Photo reference",
        description="References to full-face photographs or comparable images",
        primary_pattern=re.compile(
            r'\b(?:photo|photograph|image|picture|headshot)\s*'
            r'(?:file|path|ref|id)?\s*[:#]?\s*([A-Z0-9_\-./]{4,60}\.(?:jpg|jpeg|png|gif|bmp|tiff?))\b',
            _F
        ),
        confidence=0.80,
        severity=Severity.HIGH,
        examples=["photo: patient_jane_smith.jpg", "Image: face_scan_001.png"],
    ),

    # ── 18. ANY OTHER UNIQUE IDENTIFYING NUMBER ───────────────────────────────
    # Catch-all for labeled identifiers not covered above.
    # Lower confidence — used for review flagging rather than auto-tokenize.
    PHIField(
        field_type=PHIFieldType.OTHER_ID,
        display_name="Other unique identifier",
        description="Any other unique identifier number not covered by the above fields",
        primary_pattern=re.compile(
            r'\b(?:id|identifier|ref|reference|code)\s*[:#]?\s*'
            r'([A-Z]{2,4}[-#]?\d{4,12})\b',
            _F
        ),
        confidence=0.65,   # Lower — review flag rather than auto-tokenize
        severity=Severity.MEDIUM,
        examples=["ID: PT-12345", "Ref: REF-9876-X"],
        safe_exceptions=[_ICD10_EXCEPTION, _CPT_EXCEPTION, _NPI_EXCEPTION],
    ),
]


# ─────────────────────────────────────────────
# LOOKUP HELPERS
# ─────────────────────────────────────────────

# Fast lookup by PHIFieldType
_FIELD_BY_TYPE: dict[PHIFieldType, PHIField] = {
    f.field_type: f for f in PHI_FIELDS
}

# Fields that auto-tokenize (confidence >= threshold)
AUTO_TOKENIZE_THRESHOLD = 0.80


def get_field(phi_type: PHIFieldType) -> PHIField:
    """Returns the PHIField definition for a given PHIFieldType."""
    return _FIELD_BY_TYPE[phi_type]


def get_all_fields() -> list[PHIField]:
    """Returns all 18 PHI field definitions."""
    return PHI_FIELDS


def get_auto_tokenize_fields() -> list[PHIField]:
    """
    Returns fields with confidence >= AUTO_TOKENIZE_THRESHOLD.
    These are tokenized automatically without human review.
    """
    return [f for f in PHI_FIELDS if f.confidence >= AUTO_TOKENIZE_THRESHOLD]


def get_review_fields() -> list[PHIField]:
    """
    Returns fields with confidence < AUTO_TOKENIZE_THRESHOLD.
    These are flagged for human review rather than auto-tokenized.
    """
    return [f for f in PHI_FIELDS if f.confidence < AUTO_TOKENIZE_THRESHOLD]


def scan_all(text: str) -> list[tuple[PHIFieldType, int, int, str, float]]:
    """
    Scans text for ALL PHI field types.
    Returns a list of (field_type, start, end, matched_text, confidence) tuples,
    sorted by position in the text.

    Used by phi_tokenizer.py to find every PHI occurrence in one pass.
    """
    all_matches: list[tuple[PHIFieldType, int, int, str, float]] = []

    for phi_field in PHI_FIELDS:
        for start, end, matched_text in phi_field.scan(text):
            all_matches.append((
                phi_field.field_type,
                start,
                end,
                matched_text,
                phi_field.confidence,
            ))

    # Sort by position, then by length descending (longest match wins)
    all_matches.sort(key=lambda x: (x[1], -(x[2] - x[1])))

    # Remove overlapping matches — first (leftmost, longest) wins
    deduped: list[tuple[PHIFieldType, int, int, str, float]] = []
    last_end = -1
    for match in all_matches:
        _, start, end, _, _ = match
        if start >= last_end:
            deduped.append(match)
            last_end = end

    return deduped
