"""
oncology_expert.py — Oncology Expert Agent

Specialises in:
  - Brain tumor classification and grading (Glioma I-IV, Meningioma, Pituitary)
  - Cancer staging validation
  - Treatment protocol appropriateness (chemo, radiation, surgery)
  - Imaging necessity for oncology cases
  - Prior authorization requirements for oncology procedures

Rule-based layer: deterministic scoring from entities + codes
LLM layer: clinical reasoning narrative when score >= 0.35
"""

from typing import Optional
from .base_expert import BaseExpert, ExpertFinding

# WHO grading rules for brain tumors
TUMOR_GRADING = {
    "glioma stage 1": {"grade": "WHO I",  "urgency": "moderate", "risk": "moderate"},
    "glioma stage 2": {"grade": "WHO II", "urgency": "high",     "risk": "high"},
    "glioma stage 3": {"grade": "WHO III","urgency": "high",     "risk": "high"},
    "glioma stage 4": {"grade": "WHO IV", "urgency": "critical", "risk": "critical"},
    "glioblastoma":   {"grade": "WHO IV", "urgency": "critical", "risk": "critical"},
    "meningioma":     {"grade": "WHO I",  "urgency": "moderate", "risk": "moderate"},
    "pituitary":      {"grade": "N/A",    "urgency": "moderate", "risk": "moderate"},
    "glioma":         {"grade": "WHO II+","urgency": "high",     "risk": "high"},
}

# Imaging classes from Swin model that map to oncology
ONCOLOGY_IMAGING = {
    "Glioma":      {"grade":"WHO II+","risk":"high",    "icd":"C71.9"},
    "Meningioma":  {"grade":"WHO I",  "risk":"moderate","icd":"D32.9"},
    "Pituitary":   {"grade":"benign", "risk":"moderate","icd":"D35.2"},
}

PRIOR_AUTH_CPTS = {"70553","70554","77261","77262","77263","77301","77316","96450"}
HIGH_RISK_CHEMO = ["temozolomide","bevacizumab","carmustine","lomustine","procarbazine"]


class OncologyExpert(BaseExpert):
    expert_id   = "oncology"
    expert_name = "Oncology Expert"

    def _rule_based_analysis(
        self,
        extracted_entities: dict,
        icd10_codes:        list[dict],
        cpt_codes:          list[dict],
        router_score:       float,
        imaging_result:     Optional[dict],
        clinical_notes:     str,
    ) -> ExpertFinding:

        text      = self._get_text(extracted_entities)
        cpt_list  = [c["code"] for c in cpt_codes]
        diags     = extracted_entities.get("diagnoses", [])
        meds      = extracted_entities.get("medications", [])
        procs     = extracted_entities.get("procedures", [])

        risk_flags      = []
        recommendations = []
        additional_codes= []
        risk_level      = "low"
        confidence      = router_score * 0.6  # start from router score

        # ── 1. Tumor grade detection ───────────────────────────────────────
        detected_grade  = None
        detected_urgency= "routine"
        for dx in diags:
            dx_lower = str(dx).lower()
            for pattern, info in TUMOR_GRADING.items():
                if pattern in dx_lower:
                    detected_grade   = info["grade"]
                    detected_urgency = info["urgency"]
                    risk_level       = info["risk"]
                    confidence       = min(0.95, confidence + 0.25)
                    break
            if detected_grade:
                break

        # ── 2. Imaging model output ────────────────────────────────────────
        imaging_assessment = None
        if imaging_result and imaging_result.get("predicted_class") in ONCOLOGY_IMAGING:
            img_class  = imaging_result["predicted_class"]
            img_conf   = imaging_result.get("confidence", 0)
            img_info   = ONCOLOGY_IMAGING[img_class]
            imaging_assessment = (
                f"Imaging model classified scan as {img_class} "
                f"({img_conf:.0%} confidence) — WHO {img_info['grade']}, "
                f"risk: {img_info['risk']}."
            )
            confidence = min(0.95, confidence + 0.15)

            # Cross-validate imaging vs clinical text
            text_mentions_tumor = self._has_keyword(
                text, "glioma", "meningioma", "tumor", "pituitary", "cancer"
            )
            if not text_mentions_tumor:
                risk_flags.append(
                    "IMAGING_TEXT_MISMATCH: Imaging suggests tumor but clinical notes "
                    "do not mention malignancy — radiologist correlation required"
                )
                risk_level = "high" if risk_level == "low" else risk_level

        # ── 3. Prior auth requirements ─────────────────────────────────────
        auth_needed = [c for c in cpt_list if c in PRIOR_AUTH_CPTS]
        if auth_needed:
            risk_flags.append(
                f"PRIOR_AUTH_REQUIRED: CPT {', '.join(auth_needed)} requires "
                "prior authorization for oncology indication"
            )
            recommendations.append(
                f"Submit prior authorization for CPT {', '.join(auth_needed)} "
                "with oncology clinical notes and tumor staging documentation"
            )

        # ── 4. Chemotherapy validation ─────────────────────────────────────
        high_risk_meds = [m for m in meds
                          if any(hr in str(m).lower() for hr in HIGH_RISK_CHEMO)]
        if high_risk_meds:
            risk_flags.append(
                f"HIGH_RISK_CHEMOTHERAPY: {', '.join(str(m) for m in high_risk_meds)} "
                "— requires oncology board review and treatment plan documentation"
            )
            risk_level = "high"
            confidence = min(0.95, confidence + 0.10)
            recommendations.append(
                "Include signed oncology treatment plan and tumor board approval "
                "with chemotherapy claim submission"
            )

        # ── 5. Missing staging documentation ──────────────────────────────
        cancer_codes = self._icd_starts(icd10_codes, "C")
        if cancer_codes and not detected_grade:
            risk_flags.append(
                "MISSING_STAGING: Malignancy ICD-10 code present but no tumor grade "
                "or staging documented in clinical notes"
            )
            recommendations.append(
                "Add tumor grade (WHO classification) and clinical staging "
                "to clinical notes before resubmission"
            )

        # ── 6. MRI necessity for brain tumors ─────────────────────────────
        has_brain_tumor = self._has_keyword(text, "glioma","meningioma","brain tumor")
        has_mri_cpt     = any(c in cpt_list for c in ["70551","70552","70553","70554"])
        if has_brain_tumor and not has_mri_cpt:
            recommendations.append(
                "MRI brain with contrast (CPT 70553) is standard of care for "
                "brain tumor surveillance — consider adding if not already ordered"
            )
            additional_codes.append({
                "code": "70553",
                "description": "MRI brain with contrast",
                "reason": "Standard oncology surveillance for brain tumor"
            })

        # ── Build assessment ───────────────────────────────────────────────
        if detected_grade:
            assessment = (
                f"Brain tumor detected — {detected_grade} grade, "
                f"urgency: {detected_urgency}. "
                f"{len(risk_flags)} flag(s) identified."
            )
        elif cancer_codes:
            assessment = (
                f"Malignancy codes present ({self._format_codes(cancer_codes[:2])}). "
                f"Staging documentation required."
            )
        else:
            assessment = (
                f"Oncology routing triggered by clinical text. "
                f"No definitive malignancy codes mapped — verify coding accuracy."
            )

        if not risk_flags:
            recommendations.append(
                "Oncology documentation appears adequate — standard review applies"
            )

        return ExpertFinding(
            expert_id         = self.expert_id,
            expert_name       = self.expert_name,
            router_score      = router_score,
            expert_confidence = round(min(confidence, 0.95), 3),
            assessment        = assessment,
            risk_level        = risk_level,
            risk_flags        = risk_flags,
            recommendations   = recommendations,
            additional_codes  = additional_codes,
            imaging_assessment= imaging_assessment,
            source            = "rule_based",
        )

    def _build_llm_prompt(self, extracted_entities, icd10_codes, cpt_codes,
                          imaging_result, clinical_notes, rule_finding):
        system = """You are a board-certified oncologist reviewing a healthcare insurance claim.

Write a structured expert assessment using EXACTLY these three section headers:

## Clinical Assessment
Summarise the clinical picture, diagnoses, severity, and what the evidence shows.
If imaging model output is provided, interpret it and state whether it is concordant with the clinical notes.
Reference specific findings from the data. Be clinically precise. (3-5 sentences)

## Risk Flags
List each identified risk as a separate line starting with "- ".
For each flag: state what the problem is, why it matters clinically or administratively, and the consequence if unresolved.
If imaging and clinical text disagree, flag it explicitly with the mismatch and its clinical significance.
If no flags: write "- No significant risk flags identified."

## Recommendations
List each recommendation as a separate numbered item starting with "1.", "2." etc.
Be specific: name the exact document, test, code, or action required.
Reference relevant guideline (e.g. "per ACC/AHA Class I recommendation" or "per CMS NCD 220.6.20").
Minimum 2 recommendations even if claim appears appropriate.

Guidelines to reference where relevant: ACC/AHA, NCCN CNS Tumors, WHO Classification of Brain Tumors

IMPORTANT:
- Do not use bold, italics, or markdown formatting inside sections
- Each section must have content — do not skip any section
- Total length: 250-350 words
- Write for a medical director who will make the final payment decision"""

        diags = extracted_entities.get("diagnoses", [])
        procs = extracted_entities.get("procedures", [])
        meds  = extracted_entities.get("medications", [])
        img   = ""
        if imaging_result and imaging_result.get("predicted_class"):
            img = (f"\nImaging Model Output: {imaging_result['predicted_class']} "
                   f"({imaging_result.get('confidence',0):.0%} confidence)")

        img_section = _imaging_section(imaging_result)
        diags_str   = ', '.join(str(d) for d in extracted_entities.get('diagnoses', []))
        procs_str   = ', '.join(str(p) for p in extracted_entities.get('procedures', []))
        meds_str    = ', '.join(str(m) for m in extracted_entities.get('medications', []))
        flags_str   = '; '.join(rule_finding.risk_flags) if rule_finding.risk_flags else 'none'
        recs_str    = '; '.join(rule_finding.recommendations) if rule_finding.recommendations else 'none'

        user = f"""=== CLAIM DATA ===
Clinical Notes: {clinical_notes[:700]}

Extracted Diagnoses: {diags_str}
Procedures: {procs_str}
Medications: {meds_str}
ICD-10 Codes: {self._format_codes(icd10_codes[:5])}
CPT Codes: {self._format_codes(cpt_codes[:5])}

=== IMAGING MODEL OUTPUT ===
{img_section}

=== RULE-BASED PRE-ANALYSIS ===
Assessment: {rule_finding.assessment}
Risk Level: {rule_finding.risk_level}
Flags identified: {flags_str}
Recommendations identified: {recs_str}

=== YOUR TASK ===
Write your structured expert assessment with the three sections: Clinical Assessment, Risk Flags, Recommendations.
Explicitly interpret the imaging model output — state concordance or discordance with clinical notes."""


        return system, user