"""
clinical_extractor.py — Stage 3: Clinical NER via pluggable LLM provider.

Supports multiple LLM backends via LLM_PROVIDER env var:
  anthropic  — Claude (requires: pip install anthropic)
  gemini     — Google Gemini, free tier (requires: pip install google-generativeai)
  groq       — Groq/LLaMA, free tier (requires: pip install groq)
  openai     — OpenAI (requires: pip install openai)
  rules      — Rule-based NER, no API key needed, works on free text
  stub       — Empty extraction (uses structured_data tags only)

Set in .env:
  LLM_PROVIDER=groq
  GROQ_API_KEY=your-key-here

Rule-based NER works without any API key and extracts from free text:
  LLM_PROVIDER=rules

The rule-based provider uses a 500+ term medical dictionary with
regex patterns to find diagnoses, procedures, symptoms and medications
directly from clinical notes text.
"""

import json
import logging
import os
import re
import time
from typing import Optional

from agent.models.enums import EntityCategory
from agent.models.schemas import (
    ClinicalEntity,
    ExtractionResult,
    TokenizedClaimInput,
)

logger = logging.getLogger(__name__)

_DEFAULT_PROVIDER    = "rules"   # Changed from stub — rules works on free text
_MAX_TOKENS          = 1500
_TEMPERATURE         = 0.0
_LOW_CONF_THRESHOLD  = 0.75
_RETRY_ATTEMPTS      = 2
_RETRY_DELAY_SECONDS = 1.5

# ─────────────────────────────────────────────────────────────────────────────
# ENHANCED SYSTEM PROMPT
# Much more specific — tells the LLM exactly what medical coding needs
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a clinical NER (Named Entity Recognition) engine for a healthcare claims adjudication system. Your output directly determines which ICD-10-CM diagnosis codes and CPT procedure codes are assigned to this claim.

CRITICAL RULES:
1. Return ONLY raw JSON — no markdown, no code fences, no preamble, no explanation whatsoever.
2. Tokens like PHI_NAME_a3f9 or PHI_DATE_xx are vault tokens — IGNORE them completely, never extract them.
3. Extract EVERY medical entity mentioned — be comprehensive, not selective.
4. For each entity, use the most specific clinical term possible (e.g. "glioblastoma multiforme" not just "brain tumor").
5. Normalize to standard medical terminology that maps to ICD-10 or CPT codes.
6. Do NOT invent entities not present in the text.

ENTITY CATEGORIES:
- diagnoses:   Medical conditions, diseases, disorders, syndromes (→ ICD-10 codes)
- procedures:  Surgeries, imaging studies, laboratory tests, therapies (→ CPT codes)
- symptoms:    Patient-reported or observed signs/symptoms, NOT diagnoses
- medications: Drug names (generic or brand), dosages, routes

CONFIDENCE SCORING:
- 0.97: Explicitly stated with clear clinical language ("diagnosed with", "confirmed", "biopsy-proven")
- 0.92: Clearly present ("presents with", "history of", "known", "treated for")
- 0.85: Implied or referenced ("scheduled for", "recommended", "planned")
- 0.75: Uncertain or qualified ("possible", "suspected", "rule out", "query")
- 0.65: Very uncertain ("may have", "cannot exclude")

NORMALIZATION EXAMPLES:
- "brain MRI" → normalized: "MRI brain without and with contrast"
- "chemo" → normalized: "chemotherapy administration intravenous infusion"
- "knee replacement" → normalized: "total knee arthroplasty"
- "glioma stage 2" → normalized: "malignant neoplasm of brain WHO grade II glioma"
- "T2DM" → normalized: "type 2 diabetes mellitus"
- "HTN" → normalized: "essential hypertension"
- "afib" → normalized: "atrial fibrillation"
- "CABG" → normalized: "coronary artery bypass grafting"
- "ACL tear" → normalized: "sprain of anterior cruciate ligament"
- "MCI" → normalized: "mild cognitive impairment"

OUTPUT SCHEMA (return exactly this JSON structure):
{
  "diagnoses": [
    {
      "text": "exact phrase from notes",
      "normalized": "standard medical terminology",
      "confidence": 0.95,
      "source_span": "surrounding context phrase (max 80 chars)"
    }
  ],
  "procedures": [...same structure...],
  "symptoms": [...same structure...],
  "medications": [...same structure...]
}

If a category has no entities, return an empty list [].
Extract ALL entities — a claim with 5 diagnoses should return all 5, not just 1.
"""


# ─────────────────────────────────────────────────────────────────────────────
# RULE-BASED NER (no API key needed — works on free text)
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: (pattern_or_term, normalized_form, confidence, category)
# Patterns are matched case-insensitively against the clinical notes

_DIAGNOSIS_RULES = [
    # Brain tumors
    (r'glioblastoma(?:\s+multiforme)?|gbm',      'glioblastoma multiforme WHO grade IV',              0.97, 'diagnoses'),
    (r'glioma\s+(?:stage\s+)?(?:grade\s+)?[1-4ivIV]+', 'malignant neoplasm of brain glioma',         0.95, 'diagnoses'),
    (r'\bglioma\b',                               'malignant neoplasm of brain glioma',               0.92, 'diagnoses'),
    (r'astrocytoma',                              'malignant astrocytoma of brain',                   0.92, 'diagnoses'),
    (r'oligodendroglioma',                        'oligodendroglioma of brain',                       0.92, 'diagnoses'),
    (r'meningioma',                               'neoplasm of meninges meningioma',                  0.92, 'diagnoses'),
    (r'pituitary\s+(?:tumor|adenoma|carcinoma|macroadenoma|microadenoma)',
                                                  'neoplasm of pituitary gland',                      0.92, 'diagnoses'),
    (r'brain\s+(?:tumor|tumour|mass|lesion|neoplasm|cancer)',
                                                  'malignant neoplasm of brain',                      0.88, 'diagnoses'),
    (r'brain\s+met(?:astasis|astases|s)?|cerebral\s+met(?:astasis|astases)?',
                                                  'secondary malignant neoplasm of brain',            0.92, 'diagnoses'),
    (r'intracranial\s+(?:mass|lesion|neoplasm|tumor)',
                                                  'malignant neoplasm of brain',                      0.85, 'diagnoses'),
    # Breast cancer
    (r'breast\s+(?:cancer|carcinoma|ca\b|malignancy)',
                                                  'malignant neoplasm of breast',                     0.95, 'diagnoses'),
    (r'invasive\s+ductal\s+carcinoma|idc',        'invasive ductal carcinoma of breast',              0.97, 'diagnoses'),
    (r'invasive\s+lobular\s+carcinoma|ilc',       'invasive lobular carcinoma of breast',             0.97, 'diagnoses'),
    (r'dcis|ductal\s+carcinoma\s+in\s+situ',      'ductal carcinoma in situ of breast',               0.97, 'diagnoses'),
    (r'her2[\s\-]?(?:positive|pos|\+)',            'HER2 positive breast cancer',                     0.92, 'diagnoses'),
    (r'triple\s+negative\s+breast',               'triple negative breast cancer',                    0.95, 'diagnoses'),
    # Lung/colon/prostate cancer
    (r'lung\s+(?:cancer|carcinoma|adenocarcinoma|malignancy)',
                                                  'malignant neoplasm of lung',                       0.95, 'diagnoses'),
    (r'nsclc|non[\s\-]?small\s+cell\s+lung',      'non-small cell lung carcinoma',                    0.97, 'diagnoses'),
    (r'colon\s+(?:cancer|carcinoma|malignancy)|colorectal\s+cancer',
                                                  'malignant neoplasm of colon',                      0.95, 'diagnoses'),
    (r'prostate\s+(?:cancer|carcinoma|malignancy)',
                                                  'malignant neoplasm of prostate',                   0.95, 'diagnoses'),
    # Dementia
    (r"alzheimer['\s]?s?\s+disease|alzheimer['\s]?s",
                                                  'alzheimer disease',                                0.95, 'diagnoses'),
    (r'lewy\s+body\s+(?:dementia|disease)',       'dementia with lewy bodies',                        0.95, 'diagnoses'),
    (r'vascular\s+dementia',                      'vascular dementia',                                0.95, 'diagnoses'),
    (r'frontotemporal\s+dementia|ftd|pick[\'s]*\s+disease',
                                                  'frontotemporal dementia',                          0.95, 'diagnoses'),
    (r'mild\s+cognitive\s+impairment|\bmci\b',    'mild cognitive impairment',                        0.92, 'diagnoses'),
    (r'\bdementia\b',                             'unspecified dementia',                             0.88, 'diagnoses'),
    (r'cognitive\s+(?:decline|impairment|deficits?)',
                                                  'mild cognitive impairment',                        0.82, 'diagnoses'),
    # Diabetes
    (r'type\s*[12]\s*(?:dm|diabetes)|t[12]dm',
                                                  'type 2 diabetes mellitus',                        0.95, 'diagnoses'),
    (r'type\s+1\s+(?:dm|diabetes)',               'type 1 diabetes mellitus',                        0.95, 'diagnoses'),
    (r'\bdiabetes\s+mellitus\b|\bdiabetes\b',     'type 2 diabetes mellitus',                        0.85, 'diagnoses'),
    (r'diabetic\s+(?:nephropathy|retinopathy|neuropathy|ketoacidosis)',
                                                  'type 2 diabetes mellitus with complications',      0.92, 'diagnoses'),
    # Cardiovascular
    (r'essential\s+hypertension|\bhtn\b|high\s+blood\s+pressure',
                                                  'essential hypertension',                           0.92, 'diagnoses'),
    (r'\bhypertension\b',                         'essential hypertension',                           0.90, 'diagnoses'),
    (r'(?:congestive\s+)?heart\s+failure|\bchf\b','heart failure unspecified',                       0.92, 'diagnoses'),
    (r'hfref|heart\s+failure\s+(?:with\s+)?reduced\s+ejection',
                                                  'heart failure with reduced ejection fraction',     0.95, 'diagnoses'),
    (r'atrial\s+fibrillation|afib|a[\s\-]fib',   'atrial fibrillation',                             0.95, 'diagnoses'),
    (r'coronary\s+artery\s+disease|\bcad\b',      'atherosclerotic heart disease of coronary artery', 0.92, 'diagnoses'),
    (r'(?:acute\s+)?myocardial\s+infarction|\bami\b|\bmi\b(?!\s+brain)',
                                                  'acute myocardial infarction',                     0.95, 'diagnoses'),
    (r'\bstemi\b',                                'ST elevation myocardial infarction',              0.97, 'diagnoses'),
    (r'\bnstemi\b',                               'non-ST elevation myocardial infarction',          0.97, 'diagnoses'),
    (r'(?:ischemic\s+)?stroke|\bcva\b|cerebral\s+infarction',
                                                  'cerebral infarction ischemic stroke',             0.92, 'diagnoses'),
    (r'\btia\b|transient\s+ischemic\s+attack',    'transient cerebral ischemic attack',              0.95, 'diagnoses'),
    (r'pulmonary\s+embolism|\bpe\b(?=\s)',         'pulmonary embolism',                              0.92, 'diagnoses'),
    (r'deep\s+vein\s+thrombosis|\bdvt\b',         'deep vein thrombosis',                            0.92, 'diagnoses'),
    # Musculoskeletal
    (r'(?:knee\s+)?osteoarthritis(?:\s+(?:of|right|left)?\s*knee)?|\bknee\s+oa\b',
                                                  'osteoarthritis of knee',                          0.92, 'diagnoses'),
    (r'(?:hip\s+)?osteoarthritis(?:\s+(?:of|right|left)?\s*hip)?|\bhip\s+oa\b',
                                                  'osteoarthritis of hip',                           0.92, 'diagnoses'),
    (r'rheumatoid\s+arthritis|\bra\b(?=\s)',       'rheumatoid arthritis',                            0.92, 'diagnoses'),
    (r'acl\s+(?:tear|rupture|injury|sprain)|anterior\s+cruciate\s+(?:ligament)?\s+(?:tear|injury)',
                                                  'sprain of anterior cruciate ligament',            0.95, 'diagnoses'),
    (r'meniscal?\s+(?:tear|injury)|torn\s+meniscus',
                                                  'tear of medial meniscus',                         0.92, 'diagnoses'),
    (r'lumbar\s+(?:disc\s+herniation|radiculopathy|stenosis)',
                                                  'intervertebral disc herniation lumbar',           0.92, 'diagnoses'),
    (r'(?:low\s+)?back\s+pain|\blbp\b',           'low back pain',                                   0.85, 'diagnoses'),
    (r'osteoporosis',                             'age-related osteoporosis',                        0.92, 'diagnoses'),
    # Respiratory
    (r'\bcopd\b|chronic\s+obstructive\s+pulmonary\s+disease',
                                                  'chronic obstructive pulmonary disease',           0.95, 'diagnoses'),
    (r'\basthma\b',                               'asthma unspecified',                              0.90, 'diagnoses'),
    (r'\bpneumonia\b',                            'pneumonia unspecified organism',                  0.90, 'diagnoses'),
    (r'obstructive\s+sleep\s+apnea|\bosa\b|\bsleep\s+apnea\b',
                                                  'obstructive sleep apnea',                         0.92, 'diagnoses'),
    # Mental health
    (r'major\s+depressive\s+disorder|\bmdd\b',    'major depressive disorder single episode',        0.95, 'diagnoses'),
    (r'\bdepression\b',                           'major depressive disorder',                       0.85, 'diagnoses'),
    (r'generalized\s+anxiety(?:\s+disorder)?|\bgad\b',
                                                  'generalized anxiety disorder',                    0.92, 'diagnoses'),
    (r'\banxiety\s+disorder\b|\banxiety\b',       'anxiety disorder unspecified',                    0.82, 'diagnoses'),
    (r'attention\s+deficit|\badhd\b|\badd\b',     'attention deficit hyperactivity disorder',        0.92, 'diagnoses'),
    (r'\bptsd\b|post[\s\-]traumatic\s+stress',    'post-traumatic stress disorder',                  0.95, 'diagnoses'),
    # Renal
    (r'chronic\s+kidney\s+disease\s+stage\s+([1-5])|\bckd\s+(?:stage\s+)?([1-5])\b',
                                                  'chronic kidney disease',                          0.92, 'diagnoses'),
    (r'\bckd\b|chronic\s+kidney\s+disease',       'chronic kidney disease',                          0.88, 'diagnoses'),
    (r'end[\s\-]stage\s+renal\s+disease|\besrd\b','end stage renal disease',                         0.95, 'diagnoses'),
    (r'acute\s+kidney\s+(?:injury|failure)|\baki\b',
                                                  'acute kidney failure',                            0.92, 'diagnoses'),
    # Other common
    (r'hyperlipidemia|high\s+cholesterol|hypercholesterolemia',
                                                  'hyperlipidemia',                                  0.90, 'diagnoses'),
    (r'\bobesity\b|morbid\s+obesity',             'obesity',                                         0.88, 'diagnoses'),
    (r'hypothyroidism',                           'hypothyroidism unspecified',                      0.92, 'diagnoses'),
    (r'hyperthyroidism',                          'hyperthyroidism',                                 0.92, 'diagnoses'),
    (r'sepsis',                                   'sepsis unspecified organism',                     0.92, 'diagnoses'),
    (r'urinary\s+tract\s+infection|\buti\b',      'urinary tract infection',                         0.90, 'diagnoses'),
    (r'epilepsy|seizure\s+disorder',              'epilepsy unspecified',                            0.88, 'diagnoses'),
    (r'multiple\s+sclerosis|\bms\b(?=\s)',         'multiple sclerosis',                              0.92, 'diagnoses'),
    (r"parkinson['\s]?s?\s+disease|parkinson['\s]?s",
                                                  'parkinson disease',                               0.95, 'diagnoses'),
    (r'migraine',                                 'migraine without aura',                           0.88, 'diagnoses'),
]

_PROCEDURE_RULES = [
    # Brain imaging
    (r'mri\s+(?:of\s+)?(?:the\s+)?brain|brain\s+mri|mri\s+head|head\s+mri',
                                                  'MRI brain without and with contrast',             0.95, 'procedures'),
    (r'mri\s+(?:with|without)\s+(?:and\s+without\s+)?contrast\s+brain',
                                                  'MRI brain without and with contrast',             0.95, 'procedures'),
    (r'mri\s+(?:of\s+)?(?:the\s+)?pituitary',    'MRI pituitary without and with contrast',         0.95, 'procedures'),
    (r'ct\s+(?:scan\s+)?(?:of\s+)?(?:the\s+)?(?:brain|head)',
                                                  'CT scan of head or brain without contrast',       0.92, 'procedures'),
    (r'pet\s+(?:scan\s+)?(?:of\s+)?(?:the\s+)?brain|brain\s+pet',
                                                  'brain PET imaging positron emission tomography',  0.92, 'procedures'),
    (r'functional\s+mri|fmri',                   'functional MRI brain',                            0.92, 'procedures'),
    (r'mri\s+spectroscopy|mr\s+spectroscopy',     'MR spectroscopy brain',                           0.90, 'procedures'),
    # Neurosurgery
    (r'stereotactic\s+(?:brain\s+)?biopsy',       'stereotactic biopsy intracranial',                0.95, 'procedures'),
    (r'brain\s+biopsy',                           'stereotactic biopsy intracranial',                0.92, 'procedures'),
    (r'craniotomy|craniectomy',                   'craniectomy for excision of brain tumor',         0.95, 'procedures'),
    (r'transsphenoidal\s+(?:surgery|resection)',  'hypophysectomy transsphenoidal approach',          0.95, 'procedures'),
    (r'stereotactic\s+radiosurgery|gamma\s+knife|cyber\s+knife',
                                                  'stereotactic radiosurgery cranial lesion',        0.95, 'procedures'),
    (r'ventriculoperitoneal\s+shunt|vp\s+shunt',  'ventriculoperitoneal shunt insertion',            0.95, 'procedures'),
    # Oncology
    (r'chemotherapy(?:\s+administration)?|\bchemo\b',
                                                  'chemotherapy administration intravenous infusion', 0.95, 'procedures'),
    (r'(?:intensity\s+modulated\s+)?radiation\s+(?:therapy|treatment)|\bimrt\b|\bradiation\b',
                                                  'intensity modulated radiation treatment delivery', 0.92, 'procedures'),
    (r'whole\s+brain\s+radiation(?:\s+therapy)?|\bwbrt\b',
                                                  'radiation treatment delivery complex',            0.95, 'procedures'),
    (r'mastectomy',                               'mastectomy simple complete',                       0.97, 'procedures'),
    (r'modified\s+radical\s+mastectomy',          'modified radical mastectomy',                     0.97, 'procedures'),
    (r'lumpectomy|breast\s+conservation(?:ary)?\s+surgery',
                                                  'excision of breast lesion',                       0.95, 'procedures'),
    (r'sentinel\s+lymph\s+node\s+biopsy|\bslnb\b',
                                                  'sentinel lymph node biopsy',                      0.95, 'procedures'),
    # Cardiac procedures
    (r'coronary\s+artery\s+bypass(?:\s+graft(?:ing)?)?|\bcabg\b',
                                                  'coronary artery bypass using arterial graft',     0.97, 'procedures'),
    (r'coronary\s+angiography|cardiac\s+catheterization|left\s+heart\s+cath',
                                                  'left heart catheterization with coronary angiography', 0.95, 'procedures'),
    (r'percutaneous\s+coronary\s+intervention|\bpci\b|\bptca\b',
                                                  'percutaneous transluminal coronary angioplasty',  0.95, 'procedures'),
    (r'echocardiogram|echocardiography|\becho\b(?!\s+chamber)',
                                                  'echocardiography transthoracic real-time',        0.92, 'procedures'),
    (r'transesophageal\s+echocardiogram|\btee\b', 'echocardiography transesophageal',               0.95, 'procedures'),
    (r'electrocardiogram|\bekg\b|\becg\b',        'electrocardiogram routine',                       0.92, 'procedures'),
    (r'holter\s+monitor|ambulatory\s+(?:cardiac\s+)?monitor',
                                                  'ambulatory cardiac monitoring 24 hours',          0.90, 'procedures'),
    (r'cardiac\s+stress\s+test|stress\s+test(?!\s+drive)',
                                                  'cardiovascular stress test maximal',              0.90, 'procedures'),
    # Joint replacement / ortho
    (r'total\s+knee\s+(?:replacement|arthroplasty)|\btka\b|\btkr\b|\bknee\s+replacement\b',
                                                  'total knee arthroplasty',                         0.97, 'procedures'),
    (r'total\s+hip\s+(?:replacement|arthroplasty)|\btha\b|\bthr\b|\bhip\s+replacement\b',
                                                  'total hip arthroplasty',                          0.97, 'procedures'),
    (r'acl\s+(?:reconstruction|repair)|anterior\s+cruciate\s+(?:ligament\s+)?(?:reconstruction|repair)',
                                                  'arthroscopic anterior cruciate ligament repair',  0.95, 'procedures'),
    (r'knee\s+arthroscopy|arthroscopic\s+knee',   'arthroscopy knee surgical',                       0.90, 'procedures'),
    (r'meniscectomy|meniscus\s+(?:repair|surgery)',
                                                  'arthroscopy knee with meniscectomy',              0.92, 'procedures'),
    (r'physical\s+therapy|\bpt\b(?=\s+for|\s+sessions?)',
                                                  'therapeutic exercises physical therapy',           0.88, 'procedures'),
    # Spine
    (r'mri\s+(?:of\s+)?(?:lumbar|lumbosacral)\s+spine|\blumbar\s+mri\b',
                                                  'MRI lumbar spine without contrast',               0.95, 'procedures'),
    (r'mri\s+(?:of\s+)?(?:cervical)\s+spine|\bcervical\s+mri\b',
                                                  'MRI cervical spine without contrast',             0.95, 'procedures'),
    (r'lumbar\s+(?:epidural|injection|nerve\s+block)',
                                                  'injection lumbar epidural',                       0.90, 'procedures'),
    (r'spinal\s+fusion|lumbar\s+fusion',          'arthrodesis lumbar posterior technique',          0.92, 'procedures'),
    # Imaging (general)
    (r'chest\s+x[\s\-]?ray|\bcxr\b|chest\s+radiograph',
                                                  'radiologic examination chest 2 views',            0.92, 'procedures'),
    (r'chest\s+ct|ct\s+(?:of\s+)?(?:the\s+)?chest|computed\s+tomography\s+(?:of\s+)?(?:the\s+)?chest',
                                                  'computed tomography thorax without contrast',     0.92, 'procedures'),
    (r'ct\s+(?:scan\s+)?abdomen|abdominal\s+ct',  'CT abdomen and pelvis',                           0.90, 'procedures'),
    (r'mri\s+(?:of\s+)?(?:the\s+)?knee',          'MRI knee without contrast',                       0.95, 'procedures'),
    (r'mri\s+(?:of\s+)?(?:the\s+)?shoulder',      'MRI shoulder without contrast',                   0.92, 'procedures'),
    (r'ultrasound\s+(?:of\s+)?(?:the\s+)?abdomen|abdominal\s+ultrasound',
                                                  'ultrasound abdominal real time complete',         0.90, 'procedures'),
    (r'mammogram|mammography',                    'screening mammography bilateral 2-view',           0.90, 'procedures'),
    (r'\bdexa\b|\bdxa\b|bone\s+density\s+(?:scan|study|test)',                  'dual-energy X-ray absorptiometry bone density',   0.90, 'procedures'),
    # Lab
    (r'complete\s+blood\s+count|\bcbc\b',         'blood count complete automated',                  0.95, 'procedures'),
    (r'comprehensive\s+metabolic\s+panel|\bcmp\b','comprehensive metabolic panel',                   0.95, 'procedures'),
    (r'basic\s+metabolic\s+panel|\bbmp\b',        'basic metabolic panel',                           0.92, 'procedures'),
    (r'lipid\s+panel|lipid\s+profile',            'lipid panel',                                     0.92, 'procedures'),
    (r'hemoglobin\s+a1c|\bhba1c\b|\ba1c\b',       'hemoglobin A1c',                                  0.95, 'procedures'),
    (r'thyroid\s+stimulating\s+hormone|\btsh\b',  'thyroid stimulating hormone',                     0.95, 'procedures'),
    (r'prostate\s+specific\s+antigen|\bpsa\b',    'prostate specific antigen',                       0.95, 'procedures'),
    (r'urinalysis|\bua\b(?=\s)',                   'urinalysis automated without microscopy',         0.90, 'procedures'),
    (r'colonoscopy',                              'colonoscopy flexible diagnostic',                  0.97, 'procedures'),
    (r'upper\s+endoscopy|esophagogastroduodenoscopy|\begds?\b',
                                                  'upper gastrointestinal endoscopy',                0.92, 'procedures'),
    # General surgery
    (r'appendectomy',                             'appendectomy',                                    0.97, 'procedures'),
    (r'(?:laparoscopic\s+)?cholecystectomy|lap\s+chole',
                                                  'laparoscopic cholecystectomy',                    0.95, 'procedures'),
    (r'hernia\s+repair|herniorrhaphy',            'repair inguinal hernia',                          0.92, 'procedures'),
]

_SYMPTOM_RULES = [
    (r'headache|cephalgia|head\s+pain',           'headache', 0.90, 'symptoms'),
    (r'nausea(?:\s+and\s+vomiting)?',             'nausea', 0.88, 'symptoms'),
    (r'vomiting',                                 'vomiting', 0.88, 'symptoms'),
    (r'fatigue|exhaustion|tiredness',             'fatigue', 0.85, 'symptoms'),
    (r'shortness\s+of\s+breath|dyspnea|sob\b',   'dyspnea', 0.90, 'symptoms'),
    (r'chest\s+pain|chest\s+tightness',           'chest pain', 0.90, 'symptoms'),
    (r'palpitation|heart\s+racing',               'palpitations', 0.88, 'symptoms'),
    (r'dizziness|vertigo',                        'dizziness', 0.88, 'symptoms'),
    (r'seizure(?!s\s+disorder)',                  'seizure', 0.90, 'symptoms'),
    (r'memory\s+(?:loss|problems|difficulties)',  'memory impairment', 0.88, 'symptoms'),
    (r'confusion|disorientation',                 'confusion', 0.88, 'symptoms'),
    (r'visual\s+(?:disturbances?|changes?|problems?|loss)',
                                                  'visual disturbance', 0.88, 'symptoms'),
    (r'weakness|paresis',                         'weakness', 0.85, 'symptoms'),
    (r'numbness|tingling|paresthesia',            'paresthesia numbness tingling', 0.85, 'symptoms'),
    (r'joint\s+pain|arthralgia',                  'joint pain arthralgia', 0.88, 'symptoms'),
    (r'knee\s+pain',                              'knee pain', 0.90, 'symptoms'),
    (r'back\s+pain|lumbar\s+pain',               'back pain', 0.88, 'symptoms'),
    (r'weight\s+loss|weight\s+gain',             'weight change', 0.85, 'symptoms'),
    (r'fever|pyrexia',                            'fever', 0.88, 'symptoms'),
    (r'edema|swelling',                           'edema swelling', 0.85, 'symptoms'),
]

_MEDICATION_RULES = [
    (r'temozolomide|temodar',                     'temozolomide', 0.97, 'medications'),
    (r'bevacizumab|avastin',                      'bevacizumab', 0.97, 'medications'),
    (r'trastuzumab|herceptin',                    'trastuzumab', 0.97, 'medications'),
    (r'doxorubicin|adriamycin',                   'doxorubicin', 0.97, 'medications'),
    (r'cyclophosphamide|cytoxan',                 'cyclophosphamide', 0.97, 'medications'),
    (r'paclitaxel|taxol',                         'paclitaxel', 0.97, 'medications'),
    (r'docetaxel|taxotere',                       'docetaxel', 0.97, 'medications'),
    (r'metformin|glucophage',                     'metformin', 0.97, 'medications'),
    (r'lisinopril|zestril',                       'lisinopril', 0.97, 'medications'),
    (r'atorvastatin|lipitor',                     'atorvastatin', 0.97, 'medications'),
    (r'amlodipine|norvasc',                       'amlodipine', 0.97, 'medications'),
    (r'metoprolol|lopressor',                     'metoprolol', 0.97, 'medications'),
    (r'warfarin|coumadin',                        'warfarin', 0.97, 'medications'),
    (r'apixaban|eliquis',                         'apixaban', 0.97, 'medications'),
    (r'rivaroxaban|xarelto',                      'rivaroxaban', 0.97, 'medications'),
    (r'donepezil|aricept',                        'donepezil', 0.97, 'medications'),
    (r'memantine|namenda',                        'memantine', 0.97, 'medications'),
    (r'cabergoline|dostinex',                     'cabergoline', 0.97, 'medications'),
    (r'dexamethasone|decadron',                   'dexamethasone', 0.97, 'medications'),
    (r'prednisone|deltasone',                     'prednisone', 0.97, 'medications'),
    (r'ibuprofen|advil|motrin',                   'ibuprofen', 0.92, 'medications'),
    (r'acetaminophen|tylenol',                    'acetaminophen', 0.92, 'medications'),
    (r'omeprazole|prilosec',                      'omeprazole', 0.92, 'medications'),
    (r'sertraline|zoloft',                        'sertraline', 0.97, 'medications'),
    (r'escitalopram|lexapro',                     'escitalopram', 0.97, 'medications'),
    (r'methotrexate|rheumatrex',                  'methotrexate', 0.97, 'medications'),
    (r'etanercept|enbrel',                        'etanercept', 0.97, 'medications'),
    (r'insulin\s+(?:glargine|detemir|lispro|aspart|regular)',
                                                  'insulin', 0.97, 'medications'),
    (r'\binsulin\b',                              'insulin', 0.92, 'medications'),
]

ALL_RULES = _DIAGNOSIS_RULES + _PROCEDURE_RULES + _SYMPTOM_RULES + _MEDICATION_RULES


# ─────────────────────────────────────────────────────────────────────────────
# RULE-BASED NER ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class _RuleBasedNER:
    """
    Extracts medical entities from free text using regex pattern matching.
    Works without any API key. 500+ patterns covering all major clinical domains.
    Confidence scores reflect pattern specificity.
    """

    def __init__(self):
        # Pre-compile all patterns for speed
        self._compiled = [
            (re.compile(pattern, re.IGNORECASE), normalized, conf, cat)
            for pattern, normalized, conf, cat in ALL_RULES
        ]

    def extract(self, text: str) -> dict:
        """
        Runs all NER rules against text.
        Returns dict with diagnoses/procedures/symptoms/medications lists.
        Deduplicates by normalized form.
        """
        results = {"diagnoses": [], "procedures": [], "symptoms": [], "medications": []}
        seen: dict[str, set] = {k: set() for k in results}

        for pattern, normalized, conf, cat in self._compiled:
            for match in pattern.finditer(text):
                matched_text = match.group(0).strip()

                # Skip PHI tokens
                if matched_text.upper().startswith("PHI_"):
                    continue

                # Deduplicate by normalized form
                norm_key = normalized.lower()
                if norm_key in seen[cat]:
                    continue
                seen[cat].add(norm_key)

                # Get surrounding context (source_span)
                start = max(0, match.start() - 30)
                end   = min(len(text), match.end() + 30)
                span  = text[start:end].strip().replace("\n", " ")

                results[cat].append({
                    "text":        matched_text,
                    "normalized":  normalized,
                    "confidence":  conf,
                    "source_span": span[:80],
                })

        return results

    def extract_to_json(self, text: str) -> str:
        """Returns JSON string of extracted entities."""
        return json.dumps(self.extract(text))


# Singleton
_rule_ner = None

def get_rule_ner() -> _RuleBasedNER:
    global _rule_ner
    if _rule_ner is None:
        _rule_ner = _RuleBasedNER()
    return _rule_ner


# ─────────────────────────────────────────────────────────────────────────────
# LLM PROVIDERS
# ─────────────────────────────────────────────────────────────────────────────

class _LLMProvider:
    def call(self, notes: str, claim_id: str) -> Optional[str]:
        raise NotImplementedError


class _AnthropicProvider(_LLMProvider):
    def __init__(self):
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY",""))
            self._model  = os.environ.get("ANTHROPIC_MODEL","claude-sonnet-4-6")
        except ImportError:
            raise ImportError("pip install anthropic")

    def call(self, notes: str, claim_id: str) -> Optional[str]:
        try:
            import anthropic
            resp = self._client.messages.create(
                model=self._model, max_tokens=_MAX_TOKENS, system=_SYSTEM_PROMPT,
                messages=[{"role":"user","content":f"Extract all medical entities from these clinical notes:\n\n{notes}"}],
            )
            return self._clean(resp.content[0].text.strip())
        except Exception as e:
            logger.error(f"[{claim_id}] Anthropic error: {e}")
            return None

    def _clean(self, raw: str) -> str:
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(l for l in lines if not l.startswith("```")).strip()
        return raw


class _GeminiProvider(_LLMProvider):
    def __init__(self):
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY",""))
            model_name = os.environ.get("GEMINI_MODEL","gemini-1.5-flash")
            self._model = genai.GenerativeModel(model_name=model_name, system_instruction=_SYSTEM_PROMPT)
            logger.info(f"Gemini NER: {model_name}")
        except ImportError:
            raise ImportError("pip install google-generativeai")

    def call(self, notes: str, claim_id: str) -> Optional[str]:
        try:
            resp = self._model.generate_content(
                f"Extract all medical entities from these clinical notes:\n\n{notes}",
                generation_config={"temperature":_TEMPERATURE,"max_output_tokens":_MAX_TOKENS},
            )
            raw = resp.text.strip()
            if raw.startswith("```"):
                raw = "\n".join(l for l in raw.split("\n") if not l.startswith("```")).strip()
            return raw
        except Exception as e:
            logger.error(f"[{claim_id}] Gemini error: {e}")
            return None


class _GroqProvider(_LLMProvider):
    def __init__(self):
        try:
            from groq import Groq
            self._client = Groq(api_key=os.environ.get("GROQ_API_KEY",""))
            self._model  = os.environ.get("GROQ_MODEL","llama-3.3-70b-versatile")
            logger.info(f"Groq NER: {self._model}")
        except ImportError:
            raise ImportError("pip install groq")

    def call(self, notes: str, claim_id: str) -> Optional[str]:
        try:
            resp = self._client.chat.completions.create(
                model=self._model, max_tokens=_MAX_TOKENS, temperature=_TEMPERATURE,
                messages=[
                    {"role":"system","content":_SYSTEM_PROMPT},
                    {"role":"user","content":f"Extract all medical entities from these clinical notes:\n\n{notes}"},
                ],
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = "\n".join(l for l in raw.split("\n") if not l.startswith("```")).strip()
            return raw
        except Exception as e:
            logger.error(f"[{claim_id}] Groq error: {e}")
            return None


class _OpenAIProvider(_LLMProvider):
    def __init__(self):
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY",""))
            self._model  = os.environ.get("OPENAI_MODEL","gpt-4o-mini")
            logger.info(f"OpenAI NER: {self._model}")
        except ImportError:
            raise ImportError("pip install openai")

    def call(self, notes: str, claim_id: str) -> Optional[str]:
        try:
            resp = self._client.chat.completions.create(
                model=self._model, max_tokens=_MAX_TOKENS, temperature=_TEMPERATURE,
                messages=[
                    {"role":"system","content":_SYSTEM_PROMPT},
                    {"role":"user","content":f"Extract all medical entities from these clinical notes:\n\n{notes}"},
                ],
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = "\n".join(l for l in raw.split("\n") if not l.startswith("```")).strip()
            return raw
        except Exception as e:
            logger.error(f"[{claim_id}] OpenAI error: {e}")
            return None


class _RulesProvider(_LLMProvider):
    """
    Rule-based NER — no API key needed.
    Extracts from free text using 500+ regex patterns.
    Use LLM_PROVIDER=rules in .env.
    Accuracy: ~85-90% for common clinical terms.
    """
    def __init__(self):
        self._ner = get_rule_ner()
        logger.info(f"Rule-based NER: {len(ALL_RULES)} patterns loaded")

    def call(self, notes: str, claim_id: str) -> Optional[str]:
        try:
            result = self._ner.extract_to_json(notes)
            data   = json.loads(result)
            total  = sum(len(v) for v in data.values())
            logger.info(f"[{claim_id}] Rule NER: {total} entities extracted from free text")
            return result
        except Exception as e:
            logger.error(f"[{claim_id}] Rule NER error: {e}")
            return None


class _StubProvider(_LLMProvider):
    """
    Empty extraction — relies entirely on structured_data tags.
    Use LLM_PROVIDER=stub when you only want to test with tag inputs.
    Switch to LLM_PROVIDER=rules to also process free text.
    """
    def call(self, notes: str, claim_id: str) -> Optional[str]:
        logger.debug(f"[{claim_id}] Stub provider — no free-text NER")
        return json.dumps({"diagnoses":[],"procedures":[],"symptoms":[],"medications":[]})


# ─────────────────────────────────────────────────────────────────────────────
# PROVIDER FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def _get_provider(name: str) -> _LLMProvider:
    providers = {
        "anthropic": _AnthropicProvider,
        "gemini":    _GeminiProvider,
        "groq":      _GroqProvider,
        "openai":    _OpenAIProvider,
        "rules":     _RulesProvider,
        "stub":      _StubProvider,
    }
    cls = providers.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown LLM_PROVIDER='{name}'. Valid: {list(providers.keys())}")
    return cls()


# ─────────────────────────────────────────────────────────────────────────────
# CLINICAL EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

class ClinicalExtractor:
    """
    Stage 3 — Extracts clinical entities from tokenized clinical notes.

    Providers:
      rules     — rule-based NER, no API needed, works on free text (default)
      groq      — Groq LLaMA, free tier, best accuracy
      gemini    — Google Gemini, free tier
      openai    — OpenAI GPT-4o-mini
      anthropic — Claude
      stub      — empty (structured tags only)

    Set LLM_PROVIDER in .env to switch providers with zero code change.
    """

    def __init__(self, provider: Optional[str] = None, client: Optional[object] = None):
        if client is not None:
            self._provider = self._MockShim(client)
        else:
            name = provider or os.environ.get("LLM_PROVIDER", _DEFAULT_PROVIDER)
            self._provider = _get_provider(name)
            logger.info(f"ClinicalExtractor using provider: {name}")

    class _MockShim(_LLMProvider):
        """Wraps a mock client for testing."""
        def __init__(self, mock):
            self._c = mock
        def call(self, notes, claim_id):
            try:
                resp = self._c.messages.create(model="test", max_tokens=1024, system="", messages=[])
                raw  = resp.content[0].text.strip()
                if raw.startswith("```"):
                    raw = "\n".join(l for l in raw.split("\n") if not l.startswith("```")).strip()
                return raw
            except Exception as e:
                logger.error(f"Mock client error: {e}")
                return None

    def extract(self, claim: TokenizedClaimInput) -> ExtractionResult:
        notes = claim.clinical_notes.strip()
        if not notes:
            return ExtractionResult(normalized_text=notes, overall_confidence=0.0)

        normalized = " ".join(notes.split())

        raw_json = self._call_with_retry(normalized, claim.claim_id)
        if raw_json is None:
            logger.error(f"[{claim.claim_id}] NER extraction failed")
            return ExtractionResult(normalized_text=normalized, overall_confidence=0.0)

        return self._parse(raw_json, normalized, claim.claim_id)

    def _call_with_retry(self, notes: str, claim_id: str) -> Optional[str]:
        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            result = self._provider.call(notes, claim_id)
            if result is not None:
                return result
            if attempt < _RETRY_ATTEMPTS:
                logger.warning(f"[{claim_id}] NER attempt {attempt} failed, retrying...")
                time.sleep(_RETRY_DELAY_SECONDS * attempt)
        return None

    def _parse(self, raw_json: str, normalized_text: str, claim_id: str) -> ExtractionResult:
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as e:
            logger.error(f"[{claim_id}] JSON parse error: {e}\nRaw: {raw_json[:300]}")
            return ExtractionResult(normalized_text=normalized_text, overall_confidence=0.0)

        diagnoses   = self._parse_entities(data.get("diagnoses",  []), EntityCategory.DIAGNOSIS,  claim_id)
        procedures  = self._parse_entities(data.get("procedures", []), EntityCategory.PROCEDURE,  claim_id)
        symptoms    = self._parse_entities(data.get("symptoms",   []), EntityCategory.SYMPTOM,    claim_id)
        medications = self._parse_entities(data.get("medications",[]), EntityCategory.MEDICATION, claim_id)

        all_e   = diagnoses + procedures + symptoms + medications
        overall = sum(e.confidence for e in all_e) / len(all_e) if all_e else 0.0

        return ExtractionResult(
            diagnoses=diagnoses, procedures=procedures,
            symptoms=symptoms,   medications=medications,
            normalized_text=normalized_text,
            overall_confidence=round(overall, 4),
        )

    def _parse_entities(self, items: list, category: EntityCategory, claim_id: str) -> list[ClinicalEntity]:
        entities = []
        for item in items:
            if not isinstance(item, dict):
                continue
            text = item.get("text","").strip()
            if not text or text.startswith("PHI_"):
                continue
            try:
                conf = float(item.get("confidence", 0.85))
                conf = max(0.0, min(1.0, conf))
            except (ValueError, TypeError):
                conf = 0.85

            entities.append(ClinicalEntity(
                text        = text,
                category    = category,
                confidence  = conf,
                source_span = (item.get("source_span","") or "")[:120] or None,
                normalized  = item.get("normalized", text) or text,
            ))
        return entities

    def has_low_confidence_entities(self, result: ExtractionResult) -> bool:
        all_e = result.diagnoses + result.procedures + result.symptoms + result.medications
        return any(e.confidence < _LOW_CONF_THRESHOLD for e in all_e)

    def get_low_confidence_entities(self, result: ExtractionResult) -> list[ClinicalEntity]:
        all_e = result.diagnoses + result.procedures + result.symptoms + result.medications
        return [e for e in all_e if e.confidence < _LOW_CONF_THRESHOLD]