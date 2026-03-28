"""
router.py — Rule-based MoE Router

Reads extracted clinical entities + imaging output and decides which
expert agents to activate. Scoring is deterministic — no LLM call here.
The router runs fast so it adds zero latency to the happy path.

Scoring logic:
  - Each expert has a set of trigger keywords and ICD-10/CPT prefixes
  - Router scores each expert against the extracted data
  - Experts above threshold are activated (max 2 per claim)
  - If no expert scores above threshold, pipeline continues without MoE
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RouterDecision:
    activated_experts: list[str]          # e.g. ["oncology", "neurology"]
    scores: dict[str, float]              # expert_name -> confidence 0..1
    routing_reason: str                   # human-readable explanation
    imaging_relevant: bool                # whether imaging output should be passed to experts
    imaging_class: Optional[str] = None  # from Swin model if available


# ── EXPERT DEFINITIONS ────────────────────────────────────────────────────────

EXPERT_RULES = {

    "oncology": {
        "display_name": "Oncology Expert",
        "description":  "Cancer staging, tumor classification, treatment protocol validation",
        "keyword_triggers": [
            "glioma", "glioblastoma", "astrocytoma", "oligodendroglioma",
            "meningioma", "tumor", "tumour", "carcinoma", "cancer",
            "malignant", "neoplasm", "metastasis", "metastatic",
            "lymphoma", "leukemia", "sarcoma", "melanoma",
            "chemotherapy", "radiation therapy", "radiotherapy",
            "immunotherapy", "targeted therapy", "oncology",
            "mastectomy", "lumpectomy", "resection", "biopsy",
            "staging", "grade", "invasive", "ductal",
        ],
        "icd10_prefixes": ["C", "D00", "D01", "D02", "D03", "D04",
                           "D05", "D06", "D07", "D08", "D09", "D3", "D4"],
        "cpt_codes":      ["70553", "70554", "77261", "77262", "77263",
                           "77301", "77316", "19301", "19303", "19307"],
        "imaging_classes": ["Glioma", "Meningioma", "Pituitary"],
        "min_score": 0.25,
    },

    "neurology": {
        "display_name": "Neurology Expert",
        "description":  "Cognitive assessment, dementia staging, neurological procedure validation",
        "keyword_triggers": [
            "alzheimer", "dementia", "cognitive", "memory loss", "confusion",
            "mci", "mild cognitive impairment", "vascular dementia",
            "parkinson", "multiple sclerosis", "epilepsy", "seizure",
            "stroke", "tia", "transient ischemic", "cerebrovascular",
            "neuropathy", "encephalopathy", "myelopathy",
            "donepezil", "memantine", "aricept", "namenda",
            "neuropsychological", "mmse", "moca", "mini mental",
            "mri brain", "brain mri", "pet scan", "pet brain",
            "neurologist", "neurology", "brain",
            "headache", "migraine", "glioma", "meningioma",
        ],
        "icd10_prefixes": ["G30", "G31", "G20", "G35", "G40", "G43",
                           "G45", "G60", "G70", "G80", "G89", "G91",
                           "I60", "I61", "I62", "I63", "I65", "I66",
                           "F01", "F02", "F03", "F05", "F06", "F09"],
        "cpt_codes":      ["70551", "70552", "70553", "70554", "78607",
                           "78608", "96116", "96121", "96132", "96133",
                           "95819", "95822", "95827"],
        "imaging_classes": ["Mild Dementia", "Moderate Dementia",
                            "Non Demented", "Very Mild Dementia",
                            "Glioma", "Meningioma", "Pituitary"],
        "min_score": 0.20,
    },

    "cardiology": {
        "display_name": "Cardiology Expert",
        "description":  "Cardiovascular risk stratification, cardiac procedure validation, ACC/AHA guidelines",
        "keyword_triggers": [
            "heart failure", "cardiomyopathy", "atrial fibrillation", "afib",
            "myocardial infarction", "stemi", "nstemi", "coronary artery disease",
            "hypertension", "htn", "angina", "chest pain", "dyspnea",
            "palpitations", "syncope", "edema", "ejection fraction",
            "pulmonary embolism", "deep vein thrombosis", "dvt",
            "aortic stenosis", "mitral regurgitation", "pacemaker", "icd",
            "statin", "warfarin", "apixaban", "rivaroxaban", "metoprolol",
            "lisinopril", "furosemide", "carvedilol", "digoxin", "amiodarone",
            "echocardiogram", "stress test", "cardiac catheterization",
            "angioplasty", "stent", "cabg", "bypass", "cardiology",
        ],
        "icd10_prefixes": ["I10", "I11", "I12", "I13", "I20", "I21", "I22",
                           "I25", "I26", "I27", "I42", "I44", "I48", "I50",
                           "I63", "I65", "I70", "I80", "I82"],
        "cpt_codes":      ["93303", "93306", "93351", "93452", "93454", "93458",
                           "78452", "78459", "92928", "33206", "33249"],
        "imaging_classes": [],
        "min_score": 0.22,
    },

    "radiology": {
        "display_name": "Radiology Expert",
        "description":  "Imaging appropriateness, ACR criteria, imaging-text concordance validation",
        "keyword_triggers": [
            "mri", "ct scan", "computed tomography", "pet scan", "x-ray",
            "ultrasound", "mammogram", "fluoroscopy", "angiography",
            "contrast", "gadolinium", "iodinated contrast",
            "brain imaging", "spine imaging", "chest imaging",
            "radiology", "radiologist", "imaging", "scan",
            "dicom", "pacs", "prior imaging", "baseline imaging",
        ],
        "icd10_prefixes": [],
        "cpt_codes":      ["70450", "70460", "70470", "70551", "70552", "70553",
                           "70551", "71046", "71250", "71260", "72141", "72148",
                           "74177", "78608", "78609", "93303", "93306"],
        "imaging_classes": ["Glioma", "Meningioma", "Pituitary",
                            "Mild Dementia", "Moderate Dementia",
                            "Very Mild Dementia", "Non Demented", "Healthy"],
        "min_score": 0.18,
    },

    "orthopedics": {
        "display_name": "Orthopedics Expert",
        "description":  "Joint replacement, spine surgery, fracture management, AAOS guidelines",
        "keyword_triggers": [
            "knee replacement", "hip replacement", "joint replacement", "arthroplasty",
            "total knee", "total hip", "rotator cuff", "meniscus", "arthroscopy",
            "fracture", "broken bone", "dislocation", "ligament tear", "acl", "mcl",
            "spine fusion", "discectomy", "laminectomy", "herniated disc",
            "spinal stenosis", "scoliosis", "osteoarthritis", "oa", "arthritis",
            "physical therapy", "orthopedic", "orthopaedic",
            "bone", "joint", "tendon", "cartilage", "implant",
        ],
        "icd10_prefixes": ["M00", "M10", "M15", "M16", "M17", "M18", "M19",
                           "M20", "M21", "M22", "M23", "M40", "M41", "M43",
                           "M47", "M48", "M50", "M51", "M54", "M75", "M79",
                           "S00", "S10", "S20", "S30", "S40", "S50", "S60",
                           "S70", "S80", "S90"],
        "cpt_codes":      ["27447", "27130", "27236", "23472", "29881", "29827",
                           "22612", "22630", "63030", "63047", "22551"],
        "imaging_classes": [],
        "min_score": 0.22,
    },

    "psychiatry": {
        "display_name": "Psychiatry Expert",
        "description":  "DSM-5 accuracy, mental health parity, medication management, APA guidelines",
        "keyword_triggers": [
            "depression", "anxiety", "bipolar", "schizophrenia", "psychosis",
            "adhd", "attention deficit", "ptsd", "trauma", "ocd",
            "suicidal", "self-harm", "eating disorder", "anorexia", "bulimia",
            "substance abuse", "alcohol use", "drug use", "addiction",
            "dementia", "cognitive", "alzheimer",
            "antidepressant", "antipsychotic", "mood stabilizer",
            "sertraline", "fluoxetine", "quetiapine", "lithium", "valproate",
            "benzodiazepine", "lorazepam", "alprazolam", "clonazepam",
            "methylphenidate", "amphetamine", "adderall", "ritalin",
            "psychotherapy", "therapy", "counseling", "psychiatric",
            "mental health", "behavioral health", "inpatient psychiatric",
        ],
        "icd10_prefixes": ["F10", "F11", "F12", "F13", "F14", "F15", "F16",
                           "F20", "F25", "F30", "F31", "F32", "F33", "F40",
                           "F41", "F42", "F43", "F50", "F60", "F84", "F90"],
        "cpt_codes":      ["90791", "90792", "90832", "90834", "90837",
                           "90847", "90853", "90839", "90863"],
        "imaging_classes": ["Mild Dementia", "Moderate Dementia", "Very Mild Dementia"],
        "min_score": 0.20,
    },
}

MAX_ACTIVE_EXPERTS = 2
ACTIVATION_THRESHOLD = 0.20


def route(
    extracted_entities: dict,
    icd10_codes: list[dict],
    cpt_codes:   list[dict],
    imaging_result: Optional[dict] = None,
) -> RouterDecision:
    """
    Scores each expert and returns a RouterDecision.

    Args:
        extracted_entities: output from ClinicalExtractor
            {diagnoses, procedures, symptoms, medications, raw_text}
        icd10_codes: list of {code, description, confidence}
        cpt_codes:   list of {code, description, confidence}
        imaging_result: optional dict from Swin classifier
    """
    # Flatten all text for keyword matching
    raw_text  = _flatten_text(extracted_entities)
    icd_codes = [c["code"] for c in icd10_codes]
    cpt_list  = [c["code"] for c in cpt_codes]
    img_class = imaging_result.get("predicted_class") if imaging_result else None

    scores = {}
    reasons = {}

    for expert_id, rules in EXPERT_RULES.items():
        score = 0.0
        hits  = []

        # 1. Keyword matching in clinical text
        kw_score = _keyword_score(raw_text, rules["keyword_triggers"])
        if kw_score > 0:
            score += kw_score * 0.50
            hits.append(f"keywords({kw_score:.2f})")

        # 2. ICD-10 prefix matching
        icd_score = _icd_score(icd_codes, rules["icd10_prefixes"])
        if icd_score > 0:
            score += icd_score * 0.30
            hits.append(f"ICD-10({icd_score:.2f})")

        # 3. CPT code matching
        cpt_score = _cpt_score(cpt_list, rules["cpt_codes"])
        if cpt_score > 0:
            score += cpt_score * 0.10
            hits.append(f"CPT({cpt_score:.2f})")

        # 4. Imaging class bonus
        if img_class and img_class in rules["imaging_classes"]:
            score += 0.15
            hits.append(f"imaging({img_class})")

        scores[expert_id]  = round(min(score, 1.0), 3)
        reasons[expert_id] = ", ".join(hits) if hits else "no match"

    # Select top experts above threshold
    ranked = sorted(
        [(k, v) for k, v in scores.items() if v >= ACTIVATION_THRESHOLD],
        key=lambda x: x[1], reverse=True,
    )[:MAX_ACTIVE_EXPERTS]

    activated = [k for k, _ in ranked]
    imaging_relevant = bool(img_class and any(
        img_class in EXPERT_RULES[e]["imaging_classes"]
        for e in activated
    ))

    if activated:
        reason_parts = [f"{e}(score={scores[e]:.2f}: {reasons[e]})" for e in activated]
        routing_reason = "Activated: " + " | ".join(reason_parts)
    else:
        routing_reason = "No experts activated — scores below threshold: " + \
                         ", ".join(f"{k}={v:.2f}" for k, v in scores.items())

    logger.info(f"MoE Router: {routing_reason}")

    return RouterDecision(
        activated_experts=activated,
        scores=scores,
        routing_reason=routing_reason,
        imaging_relevant=imaging_relevant,
        imaging_class=img_class,
    )


# ── SCORING HELPERS ───────────────────────────────────────────────────────────

def _flatten_text(entities: dict) -> str:
    parts = []
    for key in ["diagnoses", "procedures", "symptoms", "medications"]:
        val = entities.get(key, [])
        if isinstance(val, list):
            parts.extend([str(v).lower() for v in val])
        elif isinstance(val, str):
            parts.append(val.lower())
    raw = entities.get("raw_text", "") or entities.get("clinical_notes", "")
    parts.append(str(raw).lower())
    return " ".join(parts)


def _keyword_score(text: str, keywords: list[str]) -> float:
    if not text:
        return 0.0
    hits = sum(1 for kw in keywords if kw.lower() in text)
    # Diminishing returns: 1 hit = 0.5, 2 = 0.7, 3+ = 0.9
    if hits == 0:   return 0.0
    if hits == 1:   return 0.5
    if hits == 2:   return 0.7
    if hits == 3:   return 0.8
    return min(0.95, 0.8 + (hits - 3) * 0.03)


def _icd_score(codes: list[str], prefixes: list[str]) -> float:
    if not codes:
        return 0.0
    hits = sum(1 for code in codes
               if any(code.startswith(p) for p in prefixes))
    return min(1.0, hits * 0.5)


def _cpt_score(cpt_list: list[str], target_cpts: list[str]) -> float:
    if not cpt_list:
        return 0.0
    hits = sum(1 for c in cpt_list if c in target_cpts)
    return min(1.0, hits * 0.6)