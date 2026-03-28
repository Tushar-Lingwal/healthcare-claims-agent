"""
orthopedics_expert.py — Orthopedics Expert Agent

Specialises in:
  - Joint replacement necessity (conservative treatment failure)
  - Fracture management and coding accuracy
  - Spine surgery appropriateness (SPORT trial criteria)
  - Arthroscopy vs open surgery indication
  - Physical therapy prerequisite documentation
  - Implant and device coding validation
  - Post-surgical complication coding
  - Age and BMI restrictions for elective procedures
"""

from typing import Optional
from .base_expert import BaseExpert, ExpertFinding

JOINT_REPLACEMENT_CPTS = {
    "27447": "Total knee arthroplasty",
    "27130": "Total hip arthroplasty",
    "27236": "Femoral neck fracture ORIF",
    "27245": "Intertrochanteric fracture nailing",
    "23472": "Total shoulder arthroplasty",
    "25800": "Total wrist arthroplasty",
}

SPINE_SURGERY_CPTS = {
    "22612": "Lumbar fusion posterior",
    "22630": "Lumbar interbody fusion",
    "63030": "Lumbar discectomy",
    "63047": "Lumbar laminectomy",
    "22551": "Cervical discectomy/fusion anterior",
    "22600": "Cervical fusion posterior",
}

ARTHROSCOPY_CPTS = {
    "29881": "Knee arthroscopy with meniscectomy",
    "29880": "Knee arthroscopy with medial/lateral meniscectomy",
    "29827": "Shoulder arthroscopy with rotator cuff repair",
    "29822": "Shoulder arthroscopy with debridement",
    "29830": "Elbow arthroscopy",
}

CONSERVATIVE_TX_KEYWORDS = [
    "physical therapy", "pt", "conservative", "nsaid", "anti-inflammatory",
    "corticosteroid injection", "steroid injection", "cortisone",
    "weight loss", "activity modification", "bracing", "orthotics",
    "failed conservative", "failed pt", "failed physical therapy",
    "6 weeks", "3 months", "6 months", "12 weeks",
]

SEVERITY_KEYWORDS = {
    "severe": ["severe","advanced","end-stage","grade 4","kellgren-lawrence 4",
               "bone on bone","complete loss","destroyed"],
    "moderate": ["moderate","grade 3","kellgren-lawrence 3","significant",
                 "substantial","considerable"],
    "mild": ["mild","grade 1","grade 2","early","minimal"],
}


class OrthopedicsExpert(BaseExpert):
    expert_id   = "orthopedics"
    expert_name = "Orthopedics Expert"

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

        # ── 1. Joint replacement validation ───────────────────────────────
        replacement_claimed = {c: JOINT_REPLACEMENT_CPTS[c]
                               for c in cpt_list if c in JOINT_REPLACEMENT_CPTS}
        if replacement_claimed:
            conservative_documented = any(kw in text for kw in CONSERVATIVE_TX_KEYWORDS)
            if not conservative_documented:
                risk_flags.append(
                    "CONSERVATIVE_TX_MISSING: Joint replacement claimed without "
                    "documented failure of conservative treatment (PT, NSAIDs, injections). "
                    "Most payers require 3-6 months conservative therapy documentation."
                )
                recommendations.append(
                    "Document failed conservative treatment: duration of PT, "
                    "medications tried, injection history, and functional limitation scores"
                )
                risk_level = "high"
                confidence = min(0.95, confidence + 0.20)

            # Severity check
            severity_found = None
            for level, keywords in SEVERITY_KEYWORDS.items():
                if any(kw in text for kw in keywords):
                    severity_found = level
                    break
            if not severity_found:
                risk_flags.append(
                    "MISSING_SEVERITY: Joint replacement claimed without documented "
                    "OA severity grade (Kellgren-Lawrence scale or equivalent) — "
                    "required for medical necessity justification"
                )
            elif severity_found == "mild":
                risk_flags.append(
                    "MILD_SEVERITY_REPLACEMENT: Joint replacement claimed for mild OA — "
                    "medical necessity may be questioned without exceptional circumstances"
                )
                risk_level = "high"

        # ── 2. Spine surgery criteria ─────────────────────────────────────
        spine_claimed = {c: SPINE_SURGERY_CPTS[c]
                        for c in cpt_list if c in SPINE_SURGERY_CPTS}
        if spine_claimed:
            # SPORT trial: 6 weeks conservative for discectomy, 3 months for fusion
            conservative_documented = any(kw in text for kw in CONSERVATIVE_TX_KEYWORDS)
            imaging_confirmed = self._has_keyword(text, "mri","herniated disc","stenosis",
                                                   "radiculopathy","confirmed on imaging")
            if not conservative_documented:
                risk_flags.append(
                    "SPINE_CONSERVATIVE_MISSING: Spine surgery claimed without "
                    "documented conservative treatment failure per SPORT trial criteria"
                )
                risk_level = "high"
            if not imaging_confirmed:
                risk_flags.append(
                    "SPINE_IMAGING_MISSING: Spine surgery claimed without documented "
                    "imaging confirmation (MRI/CT myelogram) of surgical pathology"
                )
                recommendations.append(
                    "Include MRI/CT results confirming disc herniation, stenosis, "
                    "or other surgical pathology — required for spine surgery authorisation"
                )
            confidence = min(0.95, confidence + 0.18)

        # ── 3. Arthroscopy appropriateness ────────────────────────────────
        arthro_claimed = {c: ARTHROSCOPY_CPTS[c]
                         for c in cpt_list if c in ARTHROSCOPY_CPTS}
        if arthro_claimed:
            # Check for mechanical symptoms (locking, catching) for meniscectomy
            if "29881" in arthro_claimed or "29880" in arthro_claimed:
                has_mechanical = self._has_keyword(
                    text, "locking","catching","mechanical","bucket handle",
                    "meniscus tear","meniscal","mri confirmed"
                )
                if not has_mechanical:
                    risk_flags.append(
                        "ARTHROSCOPY_INDICATION: Knee arthroscopy meniscectomy "
                        "should document mechanical symptoms (locking/catching) "
                        "or MRI-confirmed meniscal tear requiring surgical repair"
                    )
            confidence = min(0.95, confidence + 0.12)

        # ── 4. Fracture coding accuracy ───────────────────────────────────
        fracture_icd = self._icd_starts(icd10_codes, "S", "M80", "M84")
        if fracture_icd:
            # Check for 7th character specificity
            incomplete_codes = [c for c in fracture_icd
                               if len(c["code"]) < 7]
            if incomplete_codes:
                risk_flags.append(
                    f"FRACTURE_CODING_INCOMPLETE: Fracture ICD-10 codes lack required "
                    f"7th character specificity: {', '.join(c['code'] for c in incomplete_codes)} — "
                    "must specify encounter type (A=initial, D=subsequent, S=sequela)"
                )
            confidence = min(0.95, confidence + 0.10)

        # ── 5. Age/weight restrictions for elective procedures ────────────
        if replacement_claimed:
            age_restricted = self._has_keyword(text, "young","age 40","age 45",
                                                "age 50","active","athletic","sports")
            if age_restricted:
                risk_flags.append(
                    "AGE_CONSIDERATION: Joint replacement in younger/active patient — "
                    "document discussion of alternative procedures and implant longevity"
                )

        # ── Build assessment ───────────────────────────────────────────────
        all_ortho = {**replacement_claimed, **spine_claimed, **arthro_claimed}
        if all_ortho:
            assessment = (
                f"{len(all_ortho)} orthopedic procedure(s) claimed: "
                f"{', '.join(all_ortho.values()[:2])}. "
                f"{len(risk_flags)} flag(s)."
            )
        else:
            assessment = "Orthopedics routing triggered — no specific procedure codes identified."

        if not risk_flags:
            recommendations.append(
                "Orthopedic documentation appears adequate for the claimed procedures"
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
            source            = "rule_based",
        )

    def _build_llm_prompt(self, extracted_entities, icd10_codes, cpt_codes,
                          imaging_result, clinical_notes, rule_finding):
        system = """You are a board-certified orthopedic surgeon reviewing a healthcare insurance claim.

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

Guidelines to reference where relevant: AAOS Clinical Practice Guidelines, SPORT Trial criteria

IMPORTANT:
- Do not use bold, italics, or markdown formatting inside sections
- Each section must have content — do not skip any section
- Total length: 250-350 words
- Write for a medical director who will make the final payment decision"""

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
