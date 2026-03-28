"""
neurology_expert.py — Neurology Expert Agent

Specialises in:
  - Alzheimer / dementia staging (CDR scale)
  - MCI assessment and progression risk
  - Neurological imaging necessity (MRI, PET)
  - Cognitive assessment procedure validation
  - Drug appropriateness for dementia stage
  - Brain tumor neurological impact
"""

from typing import Optional
from .base_expert import BaseExpert, ExpertFinding

# CDR staging from imaging model output
DEMENTIA_STAGING = {
    "Non Demented":       {"cdr": "0",   "stage": "Normal",   "risk": "low",      "confidence_boost": 0.20},
    "Very Mild Dementia": {"cdr": "0.5", "stage": "MCI/Early","risk": "moderate", "confidence_boost": 0.25},
    "Mild Dementia":      {"cdr": "1",   "stage": "Mild",     "risk": "moderate", "confidence_boost": 0.25},
    "Moderate Dementia":  {"cdr": "2",   "stage": "Moderate", "risk": "high",     "confidence_boost": 0.30},
    # Tumors that have neurological impact
    "Glioma":             {"cdr": "N/A", "stage": "Tumor",    "risk": "high",     "confidence_boost": 0.20},
    "Meningioma":         {"cdr": "N/A", "stage": "Tumor",    "risk": "moderate", "confidence_boost": 0.15},
    "Pituitary":          {"cdr": "N/A", "stage": "Tumor",    "risk": "moderate", "confidence_boost": 0.15},
}

# Dementia medications and their appropriate stages
MED_STAGE_MAP = {
    "donepezil":   ["mild", "moderate", "severe"],
    "rivastigmine":["mild", "moderate", "severe"],
    "galantamine": ["mild", "moderate"],
    "memantine":   ["moderate", "severe"],
    "aricept":     ["mild", "moderate", "severe"],
    "namenda":     ["moderate", "severe"],
    "exelon":      ["mild", "moderate", "severe"],
}

# Imaging CPT codes for neurology
NEURO_IMAGING_CPTS = {
    "70551": "MRI brain without contrast",
    "70552": "MRI brain with contrast",
    "70553": "MRI brain with/without contrast",
    "78607": "Brain SPECT",
    "78608": "PET brain (FDG)",
    "78609": "PET brain (amyloid)",
}

# Cognitive assessment CPTs
COG_ASSESSMENT_CPTS = {
    "96116": "Neurobehavioral status exam",
    "96121": "Neurobehavioral status exam, additional hour",
    "96132": "Neuropsychological testing",
    "96133": "Neuropsychological testing, additional hour",
}


class NeurologyExpert(BaseExpert):
    expert_id   = "neurology"
    expert_name = "Neurology Expert"

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
        confidence      = router_score * 0.6
        detected_stage  = None
        imaging_assessment = None

        # ── 1. Imaging model output (primary signal) ──────────────────────
        if imaging_result and imaging_result.get("predicted_class") in DEMENTIA_STAGING:
            img_class = imaging_result["predicted_class"]
            img_conf  = imaging_result.get("confidence", 0)
            stage_info = DEMENTIA_STAGING[img_class]
            detected_stage = stage_info["stage"]
            risk_level     = stage_info["risk"]
            confidence     = min(0.95, confidence + stage_info["confidence_boost"])

            if stage_info["cdr"] != "N/A":
                imaging_assessment = (
                    f"MRI pattern consistent with {img_class} "
                    f"(CDR ~{stage_info['cdr']}, {img_conf:.0%} confidence). "
                    f"Stage: {detected_stage}."
                )
                # Cross-validate: does clinical text mention cognitive symptoms?
                cognitive_keywords = ["memory","cognitive","dementia","alzheimer",
                                      "confusion","forgetful","decline"]
                if not any(kw in text for kw in cognitive_keywords):
                    risk_flags.append(
                        "IMAGING_TEXT_MISMATCH: Imaging suggests cognitive impairment "
                        "but clinical notes lack cognitive symptom documentation — "
                        "neuropsychological testing recommended"
                    )
            else:
                imaging_assessment = (
                    f"Imaging model detected {img_class} with neurological implications "
                    f"({img_conf:.0%} confidence). Neurology consult recommended."
                )

        # ── 2. Dementia staging from clinical text ─────────────────────────
        if not detected_stage:
            if self._has_keyword(text, "moderate dementia", "cdr 2", "cdr-2"):
                detected_stage = "Moderate"; risk_level = "high"
                confidence = min(0.95, confidence + 0.20)
            elif self._has_keyword(text, "mild dementia", "early alzheimer", "cdr 1"):
                detected_stage = "Mild"; risk_level = "moderate"
                confidence = min(0.95, confidence + 0.18)
            elif self._has_keyword(text, "mci", "mild cognitive", "very mild", "cdr 0.5"):
                detected_stage = "MCI"; risk_level = "moderate"
                confidence = min(0.95, confidence + 0.15)
            elif self._has_keyword(text, "alzheimer", "dementia"):
                detected_stage = "Unspecified dementia"; risk_level = "moderate"
                confidence = min(0.95, confidence + 0.12)

        # ── 3. Medication-stage appropriateness ───────────────────────────
        detected_meds = []
        for med in meds:
            med_lower = str(med).lower()
            for drug, appropriate_stages in MED_STAGE_MAP.items():
                if drug in med_lower:
                    detected_meds.append((drug, appropriate_stages))
                    confidence = min(0.95, confidence + 0.05)

                    if detected_stage:
                        stage_lower = detected_stage.lower()
                        if not any(s in stage_lower for s in appropriate_stages):
                            risk_flags.append(
                                f"MED_STAGE_MISMATCH: {drug.title()} is typically used "
                                f"for {'/'.join(appropriate_stages)} stage dementia but "
                                f"patient appears to be {detected_stage} stage"
                            )

        # ── 4. Imaging necessity for Alzheimer workup ─────────────────────
        has_dementia_dx  = self._has_keyword(text, "alzheimer","dementia","mci",
                                             "cognitive impairment")
        has_neuro_imaging= any(c in cpt_list for c in NEURO_IMAGING_CPTS)
        has_pet          = any(c in cpt_list for c in ["78607","78608","78609"])

        if has_dementia_dx and not has_neuro_imaging:
            recommendations.append(
                "MRI brain (CPT 70553) is standard of care for dementia workup — "
                "essential to rule out structural causes and vascular contributions"
            )
            additional_codes.append({
                "code": "70553",
                "description": "MRI brain with/without contrast",
                "reason": "Standard dementia workup — structural assessment"
            })

        # PET appropriateness — only for diagnostic uncertainty
        if has_pet and detected_stage in ["MCI", "Unspecified dementia"]:
            recommendations.append(
                "PET brain (CPT 78608) may require prior authorization — "
                "document diagnostic uncertainty and failure of standard workup"
            )
            risk_flags.append(
                "PET_AUTH_REQUIRED: Brain PET requires documentation of diagnostic "
                "uncertainty after standard MRI workup under CMS NCD 220.6.20"
            )

        # ── 5. Cognitive assessment codes ─────────────────────────────────
        has_cog_assess = any(c in cpt_list for c in COG_ASSESSMENT_CPTS)
        if has_dementia_dx and not has_cog_assess and detected_stage in [None, "Unspecified dementia", "MCI"]:
            recommendations.append(
                "Neurobehavioral status exam (CPT 96116) or neuropsychological "
                "testing (CPT 96132) recommended to formally document cognitive baseline"
            )

        # ── 6. Missing formal diagnosis for drug claims ────────────────────
        neuro_icd = self._icd_starts(icd10_codes, "G30","G31","F01","F02","F03")
        if detected_meds and not neuro_icd:
            risk_flags.append(
                "MISSING_NEURO_DX: Dementia medications claimed without formal "
                "G30/G31/F0x ICD-10 diagnosis code — coding may be incomplete"
            )
            additional_codes.append({
                "code": "G30.9",
                "description": "Alzheimer's disease, unspecified",
                "reason": "Required diagnosis code for dementia medication claims"
            })

        # ── 7. Rapid progression flag ─────────────────────────────────────
        if self._has_keyword(text, "rapid", "aggressive", "quickly", "fast progress"):
            if detected_stage in ["Moderate", "Mild"]:
                risk_flags.append(
                    "RAPID_PROGRESSION: Clinical notes suggest rapid cognitive decline — "
                    "recommend expedited neurology referral and treatment review"
                )
                risk_level = "high"

        # ── Build assessment ───────────────────────────────────────────────
        if detected_stage:
            drug_str = (f" Medications: {', '.join(d[0].title() for d in detected_meds)}."
                        if detected_meds else "")
            assessment = (
                f"Neurological assessment: {detected_stage} stage. "
                f"{len(risk_flags)} flag(s).{drug_str}"
            )
        elif neuro_icd:
            assessment = (
                f"Neurological ICD-10 codes present: {self._format_codes(neuro_icd[:2])}. "
                f"Stage not determinable from available data."
            )
        else:
            assessment = (
                "Neurology routing triggered by clinical keywords. "
                "No formal neurological diagnosis codes mapped."
            )

        if not risk_flags and not recommendations:
            recommendations.append(
                "Neurological documentation appears adequate for the claimed services"
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
        system = """You are a board-certified neurologist reviewing a healthcare insurance claim.

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

Guidelines to reference where relevant: AAN Practice Guidelines, CMS NCD 220.6.20, DSM-5

IMPORTANT:
- Do not use bold, italics, or markdown formatting inside sections
- Each section must have content — do not skip any section
- Total length: 250-350 words
- Write for a medical director who will make the final payment decision"""

        img = ""
        if imaging_result and imaging_result.get("predicted_class"):
            img = (f"\nImaging Model: {imaging_result['predicted_class']} "
                   f"({imaging_result.get('confidence',0):.0%} confidence, "
                   f"CDR: {DEMENTIA_STAGING.get(imaging_result['predicted_class'],{}).get('cdr','N/A')})")

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