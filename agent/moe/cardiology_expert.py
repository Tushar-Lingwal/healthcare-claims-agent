"""
cardiology_expert.py — Cardiology Expert Agent

Specialises in:
  - Cardiovascular risk stratification (ACC/AHA guidelines)
  - Hypertension management and complication assessment
  - Cardiac imaging necessity (Echo, stress test, cardiac cath)
  - Heart failure staging (NYHA classification)
  - Arrhythmia and conduction disorder assessment
  - Anticoagulation therapy validation
  - Prior authorization for high-cost cardiac procedures
  - Drug-procedure compatibility for cardiac medications
"""

from typing import Optional
from .base_expert import BaseExpert, ExpertFinding

# ACC/AHA cardiovascular risk categories
CV_RISK_KEYWORDS = {
    "critical": [
        "stemi", "nstemi", "myocardial infarction", "cardiac arrest",
        "ventricular fibrillation", "ventricular tachycardia",
        "acute coronary syndrome", "aortic dissection", "cardiac tamponade",
        "complete heart block",
    ],
    "high": [
        "heart failure", "hf", "cardiomyopathy", "ejection fraction",
        "atrial fibrillation", "afib", "af", "pulmonary embolism",
        "deep vein thrombosis", "dvt", "unstable angina",
        "coronary artery disease", "cad", "left main", "three vessel",
        "aortic stenosis", "severe", "mitral regurgitation",
    ],
    "moderate": [
        "hypertension", "htn", "high blood pressure", "hypertensive",
        "hyperlipidemia", "dyslipidemia", "hypercholesterolemia",
        "stable angina", "palpitations", "syncope", "chest pain",
        "shortness of breath", "dyspnea", "edema", "peripheral artery",
        "carotid stenosis",
    ],
}

# NYHA heart failure classification
NYHA_KEYWORDS = {
    "IV":  ["rest","minimal activity","unable to carry on","discomfort at rest"],
    "III": ["less than ordinary","marked limitation","comfortable only at rest"],
    "II":  ["ordinary physical activity","slight limitation","comfortable at rest"],
    "I":   ["no symptoms","no limitation","ordinary activity"],
}

# Cardiac medications and their indications
CARDIAC_MEDS = {
    "warfarin":       {"class":"anticoagulant","indication":"afib,dvt,pe,mechanical valve"},
    "apixaban":       {"class":"anticoagulant","indication":"afib,dvt,pe"},
    "rivaroxaban":    {"class":"anticoagulant","indication":"afib,dvt,pe"},
    "dabigatran":     {"class":"anticoagulant","indication":"afib,dvt,pe"},
    "metoprolol":     {"class":"beta_blocker","indication":"htn,hf,afib,angina"},
    "carvedilol":     {"class":"beta_blocker","indication":"hf,htn"},
    "lisinopril":     {"class":"ace_inhibitor","indication":"htn,hf,post_mi"},
    "enalapril":      {"class":"ace_inhibitor","indication":"htn,hf"},
    "losartan":       {"class":"arb","indication":"htn,hf,diabetic_nephropathy"},
    "furosemide":     {"class":"diuretic","indication":"hf,edema,htn"},
    "spironolactone": {"class":"aldosterone_antagonist","indication":"hf,htn"},
    "amlodipine":     {"class":"ccb","indication":"htn,angina"},
    "atorvastatin":   {"class":"statin","indication":"hyperlipidemia,cad_prevention"},
    "rosuvastatin":   {"class":"statin","indication":"hyperlipidemia,cad_prevention"},
    "aspirin":        {"class":"antiplatelet","indication":"cad,post_mi,stroke_prevention"},
    "clopidogrel":    {"class":"antiplatelet","indication":"acs,pci,cad"},
    "digoxin":        {"class":"cardiac_glycoside","indication":"hf,afib"},
    "amiodarone":     {"class":"antiarrhythmic","indication":"afib,vt,vf"},
    "nitroglycerin":  {"class":"nitrate","indication":"angina,acs"},
}

# Cardiac CPT codes requiring prior auth
CARDIAC_AUTH_CPTS = {
    "93452": "Left heart catheterization",
    "93454": "Coronary angiography",
    "93458": "Left heart cath with coronary angiography",
    "93461": "Right and left heart cath with coronary angiography",
    "33533": "CABG arterial",
    "33534": "CABG arterial, two vessels",
    "33535": "CABG arterial, three vessels",
    "33206": "Pacemaker insertion",
    "33249": "ICD insertion",
    "33274": "Transcatheter pacing system",
    "92928": "Percutaneous coronary stenting",
    "92933": "PCI with atherectomy",
    "93303": "Transthoracic echocardiogram",
    "93306": "Echo with Doppler",
    "93351": "Stress echo",
    "78452": "Myocardial perfusion imaging",
    "78459": "Cardiac PET",
}

# ICD-10 prefixes for cardiology
CARDIO_ICD_PREFIXES = [
    "I10","I11","I12","I13","I20","I21","I22","I23","I24","I25",
    "I26","I27","I30","I31","I32","I33","I34","I35","I36","I37",
    "I38","I40","I41","I42","I43","I44","I45","I46","I47","I48",
    "I49","I50","I51","I52","I60","I61","I62","I63","I65","I66",
    "I70","I71","I72","I73","I74","I80","I82","I83","I87",
]


class CardiologyExpert(BaseExpert):
    expert_id   = "cardiology"
    expert_name = "Cardiology Expert"

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
        meds     = extracted_entities.get("medications", [])
        diags    = extracted_entities.get("diagnoses", [])

        risk_flags      = []
        recommendations = []
        additional_codes= []
        confidence      = router_score * 0.6
        risk_level      = "low"

        # ── 1. Cardiovascular risk stratification ─────────────────────────
        detected_risk = "low"
        for level in ["critical", "high", "moderate"]:
            if any(kw in text for kw in CV_RISK_KEYWORDS[level]):
                detected_risk = level
                risk_level    = level
                confidence    = min(0.95, confidence + {"critical":0.30,"high":0.25,"moderate":0.15}[level])
                break

        # ── 2. NYHA classification ─────────────────────────────────────────
        nyha_class = None
        if self._has_keyword(text, "heart failure", "hf", "cardiomyopathy"):
            for cls, keywords in NYHA_KEYWORDS.items():
                if any(kw in text for kw in keywords):
                    nyha_class = cls
                    if cls in ["III", "IV"]:
                        risk_level = "high"
                        risk_flags.append(
                            f"SEVERE_HF: NYHA Class {cls} heart failure detected — "
                            "expedited cardiology review required"
                        )
                    break
            if not nyha_class:
                risk_flags.append(
                    "MISSING_NYHA: Heart failure documented but NYHA functional "
                    "class not specified — required for HF management coding"
                )
                recommendations.append(
                    "Document NYHA functional classification (I-IV) in clinical notes "
                    "for accurate heart failure coding and treatment planning"
                )

        # ── 3. Anticoagulation therapy validation ─────────────────────────
        anticoag_meds = []
        for med in meds:
            med_lower = str(med).lower()
            for drug, info in CARDIAC_MEDS.items():
                if drug in med_lower and info["class"] == "anticoagulant":
                    anticoag_meds.append(drug)

        if anticoag_meds:
            has_anticoag_dx = self._has_keyword(
                text, "atrial fibrillation","afib","dvt","pulmonary embolism",
                "pe","thrombosis","mechanical valve","stroke prevention"
            )
            if not has_anticoag_dx:
                risk_flags.append(
                    f"ANTICOAG_INDICATION_MISSING: {', '.join(anticoag_meds)} prescribed "
                    "without documented indication (AFib, DVT, PE, or valve disease)"
                )
                recommendations.append(
                    "Document anticoagulation indication (ICD-10 I48.x for AFib, "
                    "I82.x for DVT, I26.x for PE) to support anticoagulant claims"
                )
            confidence = min(0.95, confidence + 0.08)

        # ── 4. Prior authorization for high-cost cardiac procedures ────────
        auth_needed = {c: CARDIAC_AUTH_CPTS[c] for c in cpt_list if c in CARDIAC_AUTH_CPTS}
        if auth_needed:
            for code, desc in auth_needed.items():
                risk_flags.append(
                    f"PRIOR_AUTH_REQUIRED: CPT {code} ({desc}) requires prior "
                    "authorization with cardiology documentation"
                )
            recommendations.append(
                f"Submit prior authorization for: {', '.join(auth_needed.keys())} — "
                "include stress test results, echo findings, and cardiology consultation notes"
            )

        # ── 5. Hypertension complication check ────────────────────────────
        has_htn = self._has_keyword(text, "hypertension","htn","high blood pressure")
        cardio_icd = self._icd_starts(icd10_codes, *CARDIO_ICD_PREFIXES)

        if has_htn and cardio_icd:
            htn_with_ckd = self._has_keyword(text, "kidney","renal","ckd","creatinine")
            htn_with_hf  = self._has_keyword(text, "heart failure","hf","ejection")
            if htn_with_ckd and not any(c["code"].startswith("I12") for c in icd10_codes):
                additional_codes.append({
                    "code": "I12.9",
                    "description": "Hypertensive chronic kidney disease",
                    "reason": "HTN + CKD documented — combination code may be more accurate"
                })
            if htn_with_hf and not any(c["code"].startswith("I11") for c in icd10_codes):
                additional_codes.append({
                    "code": "I11.0",
                    "description": "Hypertensive heart disease with heart failure",
                    "reason": "HTN + HF documented — combination code required per ICD-10 guidelines"
                })

        # ── 6. Drug interaction flags ──────────────────────────────────────
        med_classes = []
        for med in meds:
            med_lower = str(med).lower()
            for drug, info in CARDIAC_MEDS.items():
                if drug in med_lower:
                    med_classes.append(info["class"])

        if "anticoagulant" in med_classes and "antiplatelet" in med_classes:
            risk_flags.append(
                "DUAL_ANTITHROMBOTIC: Both anticoagulant and antiplatelet therapy present — "
                "verify dual therapy is intentional (ACS post-PCI) and bleeding risk assessed"
            )
            risk_level = max(risk_level, "moderate",
                            key=lambda x: {"low":0,"moderate":1,"high":2,"critical":3}[x])

        # ── 7. Missing cardiac dx for drug claims ─────────────────────────
        has_statin = any("statin" in CARDIAC_MEDS.get(str(m).lower(),{}).get("class","")
                         for m in meds)
        if has_statin and not cardio_icd:
            risk_flags.append(
                "MISSING_CARDIAC_DX: Statin therapy claimed without cardiovascular "
                "ICD-10 diagnosis — add E78.x (hyperlipidemia) or I25.x (CAD) code"
            )

        # ── Build assessment ───────────────────────────────────────────────
        nyha_str = f" NYHA Class {nyha_class}." if nyha_class else ""
        if detected_risk != "low":
            assessment = (
                f"Cardiovascular risk: {detected_risk.upper()}.{nyha_str} "
                f"{len(risk_flags)} flag(s), {len(auth_needed)} prior auth(s) required."
            )
        elif cardio_icd:
            assessment = (
                f"Cardiac ICD-10 codes present: {self._format_codes(cardio_icd[:2])}. "
                f"Risk level: {risk_level}."
            )
        else:
            assessment = (
                "Cardiology routing triggered. No definitive cardiac ICD-10 codes — "
                "verify coding accuracy."
            )

        if not risk_flags:
            recommendations.append(
                "Cardiovascular documentation appears adequate for the claimed services"
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
            imaging_assessment= None,
            source            = "rule_based",
        )

    def _build_llm_prompt(self, extracted_entities, icd10_codes, cpt_codes,
                          imaging_result, clinical_notes, rule_finding):
        system = """You are a board-certified cardiologist reviewing a healthcare insurance claim.

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

Guidelines to reference where relevant: ACC/AHA Guidelines, CMS NCD for Cardiac Procedures

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