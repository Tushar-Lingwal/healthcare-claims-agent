"""
radiology_expert.py — Radiology Expert Agent

Specialises in:
  - Imaging appropriateness criteria (ACR guidelines)
  - Brain MRI / CT / PET indication validation
  - Imaging-clinical text concordance (cross-validates Swin model output)
  - Contrast vs non-contrast necessity
  - Radiation exposure justification (CT, nuclear medicine)
  - Repeat imaging frequency checks
  - DICOM modality-procedure code matching
  - Image-guided procedure validation
"""

from typing import Optional
from .base_expert import BaseExpert, ExpertFinding

# Imaging CPT codes organised by modality
IMAGING_CPTS = {
    # Brain MRI
    "70551": {"modality":"MRI","region":"brain","contrast":False,"desc":"MRI brain w/o contrast"},
    "70552": {"modality":"MRI","region":"brain","contrast":True, "desc":"MRI brain w/ contrast"},
    "70553": {"modality":"MRI","region":"brain","contrast":"both","desc":"MRI brain w/ & w/o contrast"},
    # Brain CT
    "70450": {"modality":"CT", "region":"brain","contrast":False,"desc":"CT head/brain w/o contrast"},
    "70460": {"modality":"CT", "region":"brain","contrast":True, "desc":"CT head/brain w/ contrast"},
    "70470": {"modality":"CT", "region":"brain","contrast":"both","desc":"CT head/brain w/ & w/o contrast"},
    # Nuclear / PET
    "78608": {"modality":"PET","region":"brain","contrast":False,"desc":"PET brain FDG"},
    "78609": {"modality":"PET","region":"brain","contrast":False,"desc":"PET brain amyloid"},
    "78607": {"modality":"SPECT","region":"brain","contrast":False,"desc":"Brain SPECT"},
    # Spine MRI
    "72141": {"modality":"MRI","region":"cervical_spine","contrast":False,"desc":"MRI cervical spine w/o"},
    "72156": {"modality":"MRI","region":"cervical_spine","contrast":True, "desc":"MRI cervical spine w/"},
    "72148": {"modality":"MRI","region":"lumbar_spine","contrast":False,"desc":"MRI lumbar spine w/o"},
    "72158": {"modality":"MRI","region":"lumbar_spine","contrast":True, "desc":"MRI lumbar spine w/"},
    # Cardiac
    "93303": {"modality":"Echo","region":"heart","contrast":False,"desc":"Transthoracic echo"},
    "93306": {"modality":"Echo","region":"heart","contrast":True, "desc":"Echo with Doppler"},
    "78452": {"modality":"SPECT","region":"heart","contrast":False,"desc":"Myocardial perfusion"},
    "78459": {"modality":"PET","region":"heart","contrast":False,"desc":"Cardiac PET"},
    # Chest
    "71046": {"modality":"XR","region":"chest","contrast":False,"desc":"Chest X-ray 2 views"},
    "71250": {"modality":"CT","region":"chest","contrast":False,"desc":"CT chest w/o contrast"},
    "71260": {"modality":"CT","region":"chest","contrast":True, "desc":"CT chest w/ contrast"},
    "71550": {"modality":"MRI","region":"chest","contrast":False,"desc":"MRI chest w/o"},
    # Abdomen/Pelvis
    "74177": {"modality":"CT","region":"abdomen","contrast":True, "desc":"CT abdomen/pelvis w/ contrast"},
    "74178": {"modality":"CT","region":"abdomen","contrast":"both","desc":"CT abdomen/pelvis w/ & w/o"},
    "74183": {"modality":"MRI","region":"abdomen","contrast":True, "desc":"MRI abdomen w/ contrast"},
}

# ACR Appropriateness Criteria: diagnosis → recommended imaging
ACR_CRITERIA = {
    "alzheimer":          ["70553","78608","78609"],  # MRI brain w/wo, PET FDG/amyloid
    "dementia":           ["70553","78608"],
    "glioma":             ["70553","70552"],          # MRI with contrast standard of care
    "meningioma":         ["70553","70552"],
    "pituitary":          ["70552","70553"],          # MRI with contrast required
    "brain tumor":        ["70553","70552"],
    "stroke":             ["70450","70553","70460"],  # CT first, then MRI
    "seizure":            ["70551","70553"],
    "headache":           ["70551","70450"],
    "multiple sclerosis": ["70553"],
    "spine pain":         ["72148","72141"],
    "chest pain":         ["71046","93306","78452"],
    "pulmonary embolism": ["71260"],
    "lung cancer":        ["71250","71260"],
}

# Diagnoses that REQUIRE contrast for imaging
CONTRAST_REQUIRED_DX = [
    "glioma","meningioma","pituitary","tumor","cancer","malignant",
    "metastasis","abscess","inflammation","infection","enhancement",
    "blood brain barrier","vascular malformation",
]

# High radiation modalities requiring justification
HIGH_RADIATION = ["CT","SPECT","PET","XR"]

# Swin model classes and their appropriate imaging follow-up
IMAGING_FOLLOWUP = {
    "Glioma":             {"required":["70553"],"note":"MRI brain w/contrast mandatory for glioma staging"},
    "Meningioma":         {"required":["70553"],"note":"MRI brain w/contrast standard for meningioma characterisation"},
    "Pituitary":          {"required":["70552"],"note":"Dedicated pituitary MRI protocol recommended"},
    "Mild Dementia":      {"required":["70553"],"note":"MRI brain to exclude structural causes of dementia"},
    "Moderate Dementia":  {"required":["70553","78608"],"note":"MRI + PET FDG for moderate dementia differential"},
    "Non Demented":       {"required":[],"note":"Normal MRI pattern — no follow-up imaging indicated"},
    "Very Mild Dementia": {"required":["70553"],"note":"MRI brain for baseline structural assessment"},
    "Healthy":            {"required":[],"note":"No imaging abnormality — routine screening applies"},
}


class RadiologyExpert(BaseExpert):
    expert_id   = "radiology"
    expert_name = "Radiology Expert"

    def _rule_based_analysis(
        self,
        extracted_entities: dict,
        icd10_codes:        list[dict],
        cpt_codes:          list[dict],
        router_score:       float,
        imaging_result:     Optional[dict],
        clinical_notes:     str,
    ) -> ExpertFinding:

        text     = self._get_text(extracted_entities)
        cpt_list = [c["code"] for c in cpt_codes]
        diags    = extracted_entities.get("diagnoses", [])

        risk_flags      = []
        recommendations = []
        additional_codes= []
        confidence      = router_score * 0.6
        risk_level      = "low"
        imaging_assessment = None

        # ── 1. Identify claimed imaging procedures ─────────────────────────
        claimed_imaging = {c: IMAGING_CPTS[c] for c in cpt_list if c in IMAGING_CPTS}

        # ── 2. Imaging model output validation ─────────────────────────────
        if imaging_result and imaging_result.get("predicted_class"):
            img_class = imaging_result["predicted_class"]
            img_conf  = imaging_result.get("confidence", 0)
            followup  = IMAGING_FOLLOWUP.get(img_class, {})

            # Check if required follow-up imaging is claimed
            required_cpts = followup.get("required", [])
            missing_cpts  = [c for c in required_cpts if c not in cpt_list]

            if missing_cpts:
                missing_descs = [IMAGING_CPTS[c]["desc"] for c in missing_cpts if c in IMAGING_CPTS]
                risk_flags.append(
                    f"MISSING_FOLLOWUP_IMAGING: Swin model classified scan as {img_class} "
                    f"({img_conf:.0%} confidence) but recommended follow-up imaging not claimed: "
                    f"{', '.join(missing_descs)}"
                )
                for c in missing_cpts:
                    if c in IMAGING_CPTS:
                        additional_codes.append({
                            "code": c,
                            "description": IMAGING_CPTS[c]["desc"],
                            "reason": f"Recommended follow-up for {img_class} per ACR criteria"
                        })
                confidence = min(0.95, confidence + 0.15)

            # Cross-validate imaging class vs clinical text
            img_text_match = self._validate_imaging_text_concordance(
                img_class, img_conf, text
            )
            if img_text_match["mismatch"]:
                risk_flags.append(img_text_match["flag"])
                risk_level = "high"
                confidence = min(0.95, confidence + 0.20)

            imaging_assessment = (
                f"Swin Transformer: {img_class} ({img_conf:.0%} confidence). "
                f"{followup.get('note','')}"
            )

        # ── 3. ACR appropriateness check ───────────────────────────────────
        for dx in diags:
            dx_lower = str(dx).lower()
            for condition, recommended in ACR_CRITERIA.items():
                if condition in dx_lower:
                    claimed_relevant = [c for c in cpt_list if c in recommended]
                    if not claimed_relevant and claimed_imaging:
                        # Imaging claimed but not the recommended type
                        rec_descs = [IMAGING_CPTS[c]["desc"]
                                    for c in recommended if c in IMAGING_CPTS]
                        risk_flags.append(
                            f"ACR_APPROPRIATENESS: For {dx}, ACR recommends "
                            f"{', '.join(rec_descs[:2])} — verify claimed imaging is appropriate"
                        )
                    confidence = min(0.95, confidence + 0.10)
                    break

        # ── 4. Contrast necessity validation ──────────────────────────────
        for cpt, info in claimed_imaging.items():
            if not info["contrast"]:  # Non-contrast imaging claimed
                needs_contrast = any(kw in text for kw in CONTRAST_REQUIRED_DX)
                if needs_contrast:
                    contrast_cpt = None
                    if info["region"] == "brain":
                        contrast_cpt = "70553"
                    if contrast_cpt and contrast_cpt not in cpt_list:
                        risk_flags.append(
                            f"CONTRAST_INDICATED: {info['desc']} claimed but diagnosis suggests "
                            f"contrast enhancement needed — consider upgrading to w/wo contrast protocol"
                        )
                        additional_codes.append({
                            "code": contrast_cpt,
                            "description": IMAGING_CPTS[contrast_cpt]["desc"],
                            "reason": "Contrast required for tumor/lesion characterisation"
                        })

        # ── 5. Radiation justification for CT/nuclear ─────────────────────
        high_rad_imaging = {c: info for c, info in claimed_imaging.items()
                           if info["modality"] in HIGH_RADIATION}
        if high_rad_imaging:
            # Check if MRI alternative was considered
            has_mri = any(info["modality"] == "MRI" for info in claimed_imaging.values())
            brain_ct = [c for c, info in high_rad_imaging.items()
                       if info["region"] == "brain" and info["modality"] == "CT"]
            if brain_ct and not has_mri:
                recommendations.append(
                    "Brain CT involves ionising radiation — document why MRI is not "
                    "appropriate (e.g. metallic implant, acute emergency, claustrophobia) "
                    "per ACR appropriateness criteria"
                )
            confidence = min(0.95, confidence + 0.08)

        # ── 6. Duplicate imaging check ─────────────────────────────────────
        modality_regions = [(info["modality"], info["region"])
                           for info in claimed_imaging.values()]
        seen = set()
        for mr in modality_regions:
            if mr in seen:
                risk_flags.append(
                    f"DUPLICATE_IMAGING: Multiple {mr[0]} {mr[1]} studies claimed — "
                    "verify each study has distinct clinical indication"
                )
                risk_level = "moderate"
            seen.add(mr)

        # ── 7. PET brain specific requirements (CMS NCD 220.6.20) ─────────
        pet_brain = [c for c in cpt_list if c in ["78608","78609"]]
        if pet_brain:
            has_prior_mri = self._has_keyword(text, "mri","prior imaging","previous scan")
            if not has_prior_mri:
                risk_flags.append(
                    "PET_CMS_NCD: Brain PET (NCD 220.6.20) requires prior MRI/CT "
                    "and documented diagnostic uncertainty — add prior imaging results "
                    "and clinical rationale to support medical necessity"
                )
                risk_level = max(risk_level, "moderate",
                                key=lambda x:{"low":0,"moderate":1,"high":2}[x])
                recommendations.append(
                    "Include prior MRI/CT results and document diagnostic uncertainty "
                    "per CMS NCD 220.6.20 for brain PET reimbursement"
                )
            confidence = min(0.95, confidence + 0.12)

        # ── Build assessment ───────────────────────────────────────────────
        if claimed_imaging:
            modalities = list({i["modality"] for i in claimed_imaging.values()})
            assessment = (
                f"{len(claimed_imaging)} imaging study(ies) claimed "
                f"({', '.join(modalities)}). "
                f"{len(risk_flags)} appropriateness flag(s)."
            )
        elif imaging_result:
            assessment = (
                f"Imaging model result reviewed: {imaging_result.get('predicted_class')}. "
                f"No imaging CPT codes in current claim."
            )
        else:
            assessment = (
                "Radiology routing triggered but no imaging CPT codes identified. "
                "Verify imaging claims are correctly coded."
            )

        if not risk_flags:
            recommendations.append(
                "Imaging claims appear appropriate for the documented clinical indications"
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

    def _validate_imaging_text_concordance(
        self, img_class: str, img_conf: float, text: str
    ) -> dict:
        """Cross-validate Swin model output against clinical text."""
        CONCORDANCE_RULES = {
            "Glioma":       ["glioma","tumor","brain cancer","malignant","mass"],
            "Meningioma":   ["meningioma","tumor","mass","intracranial"],
            "Pituitary":    ["pituitary","adenoma","tumor","sellar","visual field"],
            "Mild Dementia":      ["dementia","alzheimer","cognitive","memory"],
            "Moderate Dementia":  ["dementia","alzheimer","cognitive","confusion"],
            "Very Mild Dementia": ["dementia","mci","cognitive","memory","forgetful"],
            "Non Demented":       [],  # Normal — no specific text required
            "Healthy":            [],
        }
        expected_keywords = CONCORDANCE_RULES.get(img_class, [])
        if not expected_keywords:
            return {"mismatch": False}

        text_confirms = any(kw in text for kw in expected_keywords)
        if not text_confirms and img_conf > 0.70:
            return {
                "mismatch": True,
                "flag": (
                    f"IMAGE_TEXT_MISMATCH: Swin model classified as {img_class} "
                    f"({img_conf:.0%} confidence) but clinical notes do not mention "
                    f"expected findings ({', '.join(expected_keywords[:3])}). "
                    "Radiologist correlation and clinical reconciliation required."
                )
            }
        return {"mismatch": False}

    def _build_llm_prompt(self, extracted_entities, icd10_codes, cpt_codes,
                          imaging_result, clinical_notes, rule_finding):
        system = """You are a board-certified radiologist reviewing a healthcare insurance claim.

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

Guidelines to reference where relevant: ACR Appropriateness Criteria, CMS NCD 220.6.20

IMPORTANT:
- Do not use bold, italics, or markdown formatting inside sections
- Each section must have content — do not skip any section
- Total length: 250-350 words
- Write for a medical director who will make the final payment decision"""

        img = ""
        if imaging_result and imaging_result.get("predicted_class"):
            img = (f"\nSwin Model Output: {imaging_result['predicted_class']} "
                   f"({imaging_result.get('confidence',0):.0%} confidence)\n"
                   f"All probabilities: {imaging_result.get('all_probabilities',{})}")

        claimed = [f"CPT {c['code']}: {c['description']}" for c in cpt_codes
                  if c['code'] in IMAGING_CPTS]

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
