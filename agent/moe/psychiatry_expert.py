"""
psychiatry_expert.py — Psychiatry Expert Agent

Specialises in:
  - DSM-5 diagnosis code accuracy
  - Mental health parity compliance (MHPAEA)
  - Inpatient psychiatric necessity criteria
  - Medication management appropriateness
  - Psychotherapy level of care validation
  - Substance use disorder treatment coding
  - Crisis intervention necessity
  - Co-occurring disorder documentation
"""

from typing import Optional
from .base_expert import BaseExpert, ExpertFinding

PSYCH_SEVERITY = {
    "critical": [
        "suicidal","suicide attempt","homicidal","psychosis","acute psychosis",
        "command hallucinations","grave disability","danger to self","danger to others",
        "acute mania","severe agitation","5150","involuntary","inpatient admission",
    ],
    "high": [
        "major depressive","severe depression","bipolar","schizophrenia",
        "schizoaffective","acute anxiety","panic disorder","ptsd","trauma",
        "eating disorder","anorexia","bulimia","ocd severe",
        "self-harm","self harm","cutting","substance abuse",
    ],
    "moderate": [
        "depression","anxiety","adhd","add","ocd","phobia","insomnia",
        "adjustment disorder","mild anxiety","dysthymia","cyclothymia",
        "alcohol use","drug use","cannabis","substance use",
    ],
}

PSYCH_MEDS = {
    # Antidepressants
    "sertraline":    {"class":"ssri","indication":"depression,anxiety,ptsd,ocd"},
    "fluoxetine":    {"class":"ssri","indication":"depression,anxiety,ocd,bulimia"},
    "escitalopram":  {"class":"ssri","indication":"depression,anxiety"},
    "venlafaxine":   {"class":"snri","indication":"depression,anxiety,ptsd"},
    "duloxetine":    {"class":"snri","indication":"depression,anxiety,pain"},
    "bupropion":     {"class":"ndri","indication":"depression,smoking"},
    "mirtazapine":   {"class":"nassa","indication":"depression,insomnia"},
    # Antipsychotics
    "quetiapine":    {"class":"atypical_ap","indication":"schizophrenia,bipolar,depression"},
    "olanzapine":    {"class":"atypical_ap","indication":"schizophrenia,bipolar"},
    "risperidone":   {"class":"atypical_ap","indication":"schizophrenia,bipolar"},
    "aripiprazole":  {"class":"atypical_ap","indication":"schizophrenia,bipolar,depression"},
    "haloperidol":   {"class":"typical_ap","indication":"schizophrenia,acute_psychosis"},
    # Mood stabilizers
    "lithium":       {"class":"mood_stabilizer","indication":"bipolar"},
    "valproate":     {"class":"mood_stabilizer","indication":"bipolar,seizure"},
    "lamotrigine":   {"class":"mood_stabilizer","indication":"bipolar,seizure"},
    # Anxiolytics (controlled)
    "lorazepam":     {"class":"benzodiazepine","indication":"anxiety,acute_agitation"},
    "clonazepam":    {"class":"benzodiazepine","indication":"anxiety,panic,seizure"},
    "diazepam":      {"class":"benzodiazepine","indication":"anxiety,seizure,alcohol_wd"},
    "alprazolam":    {"class":"benzodiazepine","indication":"anxiety,panic"},
    # ADHD
    "methylphenidate":{"class":"stimulant","indication":"adhd"},
    "amphetamine":   {"class":"stimulant","indication":"adhd,narcolepsy"},
    "atomoxetine":   {"class":"snri","indication":"adhd"},
}

PSYCH_CPT_LEVELS = {
    "90791": {"type":"assessment","level":"initial","desc":"Psychiatric diagnostic evaluation"},
    "90792": {"type":"assessment","level":"initial","desc":"Psychiatric diagnostic eval with medical services"},
    "90832": {"type":"therapy","level":"individual","minutes":30,"desc":"Psychotherapy 30 min"},
    "90834": {"type":"therapy","level":"individual","minutes":45,"desc":"Psychotherapy 45 min"},
    "90837": {"type":"therapy","level":"individual","minutes":60,"desc":"Psychotherapy 60 min"},
    "90847": {"type":"therapy","level":"family","desc":"Family psychotherapy with patient"},
    "90853": {"type":"therapy","level":"group","desc":"Group psychotherapy"},
    "90839": {"type":"crisis","level":"crisis","minutes":60,"desc":"Psychotherapy for crisis 60 min"},
    "90840": {"type":"crisis","level":"crisis","minutes":30,"desc":"Psychotherapy for crisis additional 30 min"},
    "90863": {"type":"medication","level":"med_mgmt","desc":"Pharmacologic management"},
    "99213": {"type":"e_m","level":"established","desc":"Office visit E&M level 3"},
    "99214": {"type":"e_m","level":"established","desc":"Office visit E&M level 4"},
    "H0015": {"type":"sud","level":"intensive","desc":"Alcohol/drug treatment intensive outpatient"},
    "H2019": {"type":"sud","level":"standard","desc":"Therapeutic behavioral services"},
}

DSM5_ICD_PREFIXES = [
    "F10","F11","F12","F13","F14","F15","F16","F17","F18","F19",  # SUDs
    "F20","F21","F22","F23","F24","F25","F28","F29",               # Psychotic
    "F30","F31","F32","F33","F34","F39",                           # Mood
    "F40","F41","F42","F43","F44","F45","F48",                     # Anxiety
    "F50","F51","F52","F53","F54","F55","F59",                     # Somatic/eating
    "F60","F61","F62","F63","F64","F65","F66","F68","F69",         # Personality
    "F70","F71","F72","F73","F74","F78","F79",                     # Intellectual
    "F80","F81","F82","F84","F88","F89",                           # Neurodevelopmental
    "F90","F91","F93","F94","F95","F98","F99",                     # Childhood
]


class PsychiatryExpert(BaseExpert):
    expert_id   = "psychiatry"
    expert_name = "Psychiatry Expert"

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

        # ── 1. Severity stratification ─────────────────────────────────────
        detected_severity = "low"
        for level in ["critical","high","moderate"]:
            if any(kw in text for kw in PSYCH_SEVERITY[level]):
                detected_severity = level
                risk_level        = level
                confidence = min(0.95, confidence +
                                 {"critical":0.30,"high":0.22,"moderate":0.14}[level])
                break

        # ── 2. Inpatient necessity criteria ───────────────────────────────
        inpatient_cpts = [c for c in cpt_list if c.startswith("99")]
        is_inpatient   = self._has_keyword(text, "inpatient","hospitalization",
                                           "admission","admitted","psychiatric unit")
        if is_inpatient and detected_severity not in ["critical","high"]:
            risk_flags.append(
                "INPATIENT_NECESSITY: Inpatient psychiatric stay claimed but "
                "clinical notes do not document imminent danger criteria — "
                "payer requires evidence of safety risk or grave disability"
            )
            risk_level = "high"
            recommendations.append(
                "Document inpatient necessity: imminent danger to self/others, "
                "grave disability, or failure of less restrictive treatment options"
            )

        # ── 3. Controlled substance monitoring ───────────────────────────
        benzo_meds = [str(m).lower() for m in meds
                      if any(b in str(m).lower() for b in
                             ["lorazepam","clonazepam","diazepam","alprazolam",
                              "xanax","valium","klonopin","ativan"])]
        stimulant_meds = [str(m).lower() for m in meds
                         if any(s in str(m).lower() for s in
                                ["methylphenidate","amphetamine","adderall",
                                 "ritalin","vyvanse","dexedrine"])]

        if benzo_meds:
            if not self._has_keyword(text, "anxiety","panic","insomnia","seizure","alcohol withdrawal"):
                risk_flags.append(
                    f"BENZO_INDICATION_MISSING: Benzodiazepine(s) "
                    f"({', '.join(benzo_meds)}) prescribed without documented "
                    "anxiety, panic, insomnia, or seizure indication"
                )
            if self._has_keyword(text, "substance use","sud","alcohol","drug use"):
                risk_flags.append(
                    "BENZO_SUD_RISK: Benzodiazepine prescribed in patient with "
                    "documented substance use disorder — document risk/benefit analysis"
                )
            confidence = min(0.95, confidence + 0.08)

        if stimulant_meds:
            if not self._has_keyword(text, "adhd","add","attention deficit"):
                risk_flags.append(
                    f"STIMULANT_INDICATION_MISSING: Stimulant medication "
                    f"({', '.join(stimulant_meds)}) without documented ADHD diagnosis"
                )
            confidence = min(0.95, confidence + 0.07)

        # ── 4. Mental health parity check (MHPAEA) ────────────────────────
        psych_icd = self._icd_starts(icd10_codes, *DSM5_ICD_PREFIXES)
        if psych_icd:
            therapy_cpts = [c for c in cpt_list if c in PSYCH_CPT_LEVELS
                           and PSYCH_CPT_LEVELS[c]["type"] == "therapy"]
            if not therapy_cpts and detected_severity in ["high","critical"]:
                recommendations.append(
                    "MHPAEA: Consider adding psychotherapy CPT codes — "
                    "combined pharmacotherapy + therapy shows superior outcomes "
                    "for mood/anxiety disorders per APA guidelines"
                )
            confidence = min(0.95, confidence + 0.10)

        # ── 5. DSM-5 code specificity ─────────────────────────────────────
        unspecified_codes = [c for c in psych_icd
                            if c["code"].endswith("9") or "unspecified" in c["description"].lower()]
        if unspecified_codes and len(unspecified_codes) > 1:
            risk_flags.append(
                f"UNSPECIFIED_DX_OVERUSE: Multiple unspecified mental health codes — "
                f"({', '.join(c['code'] for c in unspecified_codes)}) — "
                "document specific DSM-5 specifiers for more precise coding"
            )

        # ── 6. Crisis intervention documentation ─────────────────────────
        crisis_cpts = [c for c in cpt_list if c in ["90839","90840"]]
        if crisis_cpts:
            if not self._has_keyword(text, "crisis","suicidal","danger","emergency",
                                     "acute","urgent","immediate"):
                risk_flags.append(
                    "CRISIS_DOCUMENTATION: Crisis psychotherapy CPT (90839/90840) "
                    "claimed without crisis documentation in clinical notes — "
                    "must document nature of crisis and interventions"
                )
            confidence = min(0.95, confidence + 0.12)

        # ── 7. Medication without psychotherapy for moderate+ ─────────────
        has_psych_meds = any(str(m).lower() in PSYCH_MEDS for m in meds)
        has_therapy    = any(c in cpt_list for c in ["90832","90834","90837"])
        if has_psych_meds and not has_therapy and detected_severity == "high":
            recommendations.append(
                "APA guidelines recommend combined pharmacotherapy + psychotherapy "
                "for moderate-severe depression/anxiety — consider adding therapy referral"
            )

        # ── Build assessment ───────────────────────────────────────────────
        psych_icd_str = self._format_codes(psych_icd[:2]) if psych_icd else "none"
        assessment = (
            f"Psychiatric severity: {detected_severity.upper()}. "
            f"DSM-5 codes: {psych_icd_str}. "
            f"{len(risk_flags)} flag(s)."
        )

        if not risk_flags:
            recommendations.append(
                "Psychiatric documentation appears adequate for the claimed services"
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
        system = """You are a board-certified psychiatrist reviewing a healthcare insurance claim.

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

Guidelines to reference where relevant: APA Practice Guidelines, DSM-5, MHPAEA

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
