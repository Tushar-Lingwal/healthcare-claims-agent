"""
medical_coder.py — Stage 4: ICD-10-CM and CPT coding via SQLite FTS5 full-text search.

The hardcoded dict is replaced with a SQLite database supporting:
  1. Exact code lookup  (confidence 0.97) — when entity text IS a code
  2. FTS5 full-text search (0.78-0.92)   — searches all descriptions
  3. Fuzzy rerank (0.70+)                — SequenceMatcher blended with FTS rank

Ships with ~300 codes covering all major specialties.
Load full CMS 70k dataset: python scripts/build_code_db.py --icd icd10cm_2024.txt
"""

import logging
import os
import re
import sqlite3
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from agent.models.enums import CodeSystem
from agent.models.schemas import ClinicalEntity, CodeMapping, CodingResult, ExtractionResult

logger = logging.getLogger(__name__)

_CONF_EXACT    = 0.97
_CONF_FTS_HIGH = 0.92
_CONF_FTS_MED  = 0.85
_CONF_FTS_LOW  = 0.78
_CONF_MIN      = 0.65

_DB_PATH = Path(__file__).parent.parent.parent / "data" / "codes" / "medical_codes.sqlite"



# ─────────────────────────────────────────────
# SYNONYM / ALIAS MAP
# Translates common clinical shorthand to
# canonical terms that match DB descriptions
# ─────────────────────────────────────────────
_SYNONYMS: dict[str, str] = {
    # Neoplasms
    'glioma':              'malignant neoplasm brain',
    'glioma stage 1':      'malignant neoplasm brain',
    'glioma stage 2':      'malignant neoplasm brain temporal lobe',
    'glioma stage 3':      'malignant neoplasm brain',
    'glioma stage 4':      'malignant neoplasm brain glioblastoma',
    'glioblastoma':        'malignant neoplasm brain glioblastoma',
    'gbm':                 'malignant neoplasm brain glioblastoma',
    'brain tumor':         'malignant neoplasm brain',
    'brain tumour':        'malignant neoplasm brain',
    'meningioma':          'benign neoplasm meninges',
    'pituitary tumor':     'benign neoplasm pituitary gland',
    'pituitary adenoma':   'benign neoplasm pituitary gland',
    'brain met':           'secondary malignant neoplasm brain',
    'brain metastasis':    'secondary malignant neoplasm brain',
    'brain mets':          'secondary malignant neoplasm brain',
    'breast cancer':       'malignant neoplasm breast',
    'breast carcinoma':    'malignant neoplasm breast',
    'breast ca':           'malignant neoplasm breast',
    'lung cancer':         'malignant neoplasm bronchus lung',
    'colon cancer':        'malignant neoplasm colon',
    'colorectal cancer':   'malignant neoplasm colon',
    'prostate cancer':     'malignant neoplasm prostate',
    # Dementia
    'alzheimer':           'alzheimer disease',
    'alzheimers':          'alzheimer disease',
    'mci':                 'mild cognitive impairment',
    'mild dementia':       'unspecified dementia',
    'vascular dementia':   'vascular dementia',
    'lewy body':           'dementia lewy bodies',
    # Metabolic
    'type 2 diabetes':     'E11.9',
    't2dm':                'E11.9',
    'type 2 dm':           'E11.9',
    'type 1 diabetes':     'type 1 diabetes mellitus',
    'diabetes':            'E11.9',
    'high cholesterol':    'hyperlipidemia',
    'high blood pressure': 'essential hypertension',
    'htn':                 'essential hypertension',
    'obesity':             'obesity excess calories',
    # Cardiac
    'heart attack':        'acute myocardial infarction',
    'mi':                  'acute myocardial infarction',
    'stemi':               'st elevation myocardial infarction',
    'nstemi':              'non-st elevation myocardial infarction',
    'afib':                'atrial fibrillation',
    'a fib':               'atrial fibrillation',
    'chf':                 'heart failure unspecified',
    'congestive heart failure': 'heart failure unspecified',
    'cad':                 'atherosclerotic heart disease coronary artery',
    'coronary artery disease': 'atherosclerotic heart disease coronary artery',
    'stroke':              'cerebral infarction',
    'cva':                 'cerebral infarction',
    'tia':                 'transient cerebral ischemic attack',
    'pe':                  'pulmonary embolism',
    'pulmonary embolism':  'pulmonary embolism',
    'dvt':                 'deep vein thrombosis',
    # Musculoskeletal
    'knee oa':             'osteoarthritis knee',
    'knee osteoarthritis': 'osteoarthritis knee',
    'hip oa':              'osteoarthritis hip',
    'hip osteoarthritis':  'osteoarthritis hip',
    'ra':                  'rheumatoid arthritis',
    'rheumatoid arthritis':'rheumatoid arthritis factor',
    'back pain':           'low back pain',
    'lbp':                 'low back pain',
    'acl tear':            'sprain anterior cruciate ligament',
    'acl injury':          'sprain anterior cruciate ligament',
    'osteoporosis':        'age-related osteoporosis',
    # Mental health
    'depression':          'major depressive disorder single episode',
    'mdd':                 'major depressive disorder single episode',
    'anxiety':             'anxiety disorder unspecified',
    'gad':                 'generalized anxiety disorder',
    'adhd':                'attention-deficit hyperactivity disorder',
    'ptsd':                'post-traumatic stress disorder',
    # Respiratory
    'asthma':              'asthma uncomplicated',
    'copd':                'chronic obstructive pulmonary disease',
    'pneumonia':           'pneumonia unspecified organism',
    'sleep apnea':         'obstructive sleep apnea',
    'osa':                 'obstructive sleep apnea',
    # Renal
    'ckd':                 'chronic kidney disease',
    'esrd':                'end stage renal disease',
    'aki':                 'acute kidney failure',
    'kidney stones':       'calculus kidney',
    'uti':                 'urinary tract infection',
    # Procedures — Imaging
    'mri brain':           'MRI brain without contrast 70551',
    'brain mri':           'MRI brain without contrast 70551',
    'mri head':            'MRI brain without contrast 70551',
    'ct brain':            'computed tomography head brain without contrast',
    'ct head':             'computed tomography head brain without contrast',
    'chest ct':            'computed tomography thorax',
    'chest x-ray':         'radiologic examination chest',
    'cxr':                 'radiologic examination chest',
    'echo':                'echocardiography transthoracic real time',
    'echocardiogram':      'echocardiography transthoracic real time',
    'tte':                 'echocardiography transthoracic real time',
    'tee':                 'echocardiography transesophageal',
    'ekg':                 'electrocardiogram routine',
    'ecg':                 'electrocardiogram routine',
    'eeg':                 'electroencephalogram',
    'pet scan':            'brain imaging positron emission tomography',
    'pet brain':           'brain imaging positron emission tomography',
    'mri knee':            'magnetic resonance imaging joint lower extremity',
    'mri spine':           'magnetic resonance imaging spinal canal lumbar',
    'mri lumbar':          'magnetic resonance imaging spinal canal lumbar',
    # Procedures — Surgery
    'knee replacement':    'total knee arthroplasty',
    'tka':                 'total knee arthroplasty',
    'tkr':                 'total knee arthroplasty',
    'hip replacement':     'total hip arthroplasty',
    'tha':                 'total hip arthroplasty',
    'thr':                 'total hip arthroplasty',
    'acl reconstruction':  'arthroscopically aided anterior cruciate ligament repair',
    'acl repair':          'arthroscopically aided anterior cruciate ligament repair',
    'mastectomy':          'mastectomy simple complete',
    'simple mastectomy':   'mastectomy simple complete',
    'lap chole':           'laparoscopic cholecystectomy',
    'cholecystectomy':     'laparoscopic cholecystectomy',
    'appendectomy':        'appendectomy',
    'colonoscopy':         'colonoscopy flexible diagnostic',
    'cabg':                'coronary artery bypass arterial graft',
    'bypass surgery':      'coronary artery bypass arterial graft',
    # Procedures — Oncology
    'chemo':               'chemotherapy administration intravenous infusion',
    'chemotherapy':        'chemotherapy administration intravenous infusion',
    'radiation':           'intensity modulated radiation treatment delivery complex',
    'radiation therapy':   'intensity modulated radiation treatment delivery complex',
    'imrt':                'intensity modulated radiation treatment delivery complex',
    'radiosurgery':        'stereotactic radiosurgery cranial lesion',
    'gamma knife':         'stereotactic radiosurgery cranial lesion',
    'stereotactic radiosurgery': 'stereotactic radiosurgery cranial lesion',
    'brain biopsy':        'stereotactic biopsy aspiration intracranial',
    'craniotomy':          'craniectomy excision brain tumor',
    # Procedures — Lab
    'cbc':                 'blood count complete automated',
    'bmp':                 'basic metabolic panel',
    'cmp':                 'comprehensive metabolic panel',
    'lipid panel':         'lipid panel',
    'hba1c':               'hemoglobin a1c',
    'a1c':                 'hemoglobin a1c',
    'tsh':                 'thyroid stimulating hormone',
    'psa':                 'prostate specific antigen',
    'urinalysis':          'urinalysis automated without microscopy',
    'biopsy':              'surgical pathology gross microscopic examination',
    # Procedures — Physical therapy
    'physical therapy':    'therapeutic procedure therapeutic exercises',
    'pt':                  'therapeutic procedure therapeutic exercises',
    'occupational therapy':'therapeutic activities direct patient contact',
}

class CodeDatabase:
    """SQLite-backed code lookup with FTS5 full-text search."""

    def __init__(self, db_path: Optional[Path] = None):
        self._path  = db_path or _DB_PATH
        self._conn  = None
        self._ready = False
        self._init()

    def _init(self):
        if not self._path.exists():
            logger.info(f"Code DB not found — building...")
            self._build()
        try:
            self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            tables = {r[0] for r in self._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "icd10" not in tables:
                self._conn.close(); self._build()
                self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
                self._conn.row_factory = sqlite3.Row
            self._ready = True
            n_icd = self._conn.execute("SELECT COUNT(*) FROM icd10").fetchone()[0]
            n_cpt = self._conn.execute("SELECT COUNT(*) FROM cpt").fetchone()[0]
            logger.info(f"CodeDB: {n_icd:,} ICD-10 + {n_cpt:,} CPT codes")
        except Exception as e:
            logger.error(f"CodeDB init failed: {e}")

    def _build(self):
        import subprocess, sys
        script = Path(__file__).parent.parent.parent / "scripts" / "build_code_db.py"
        if script.exists():
            subprocess.run([sys.executable, str(script)], check=False)

    def lookup_icd10(self, code: str) -> Optional[tuple]:
        if not self._ready: return None
        r = self._conn.execute("SELECT code,description FROM icd10 WHERE code=?", (code.upper(),)).fetchone()
        return (r["code"], r["description"]) if r else None

    def lookup_cpt(self, code: str) -> Optional[tuple]:
        if not self._ready: return None
        r = self._conn.execute("SELECT code,description FROM cpt WHERE code=?", (code.strip(),)).fetchone()
        return (r["code"], r["description"]) if r else None

    def search_icd10(self, query: str, limit: int = 8) -> list:
        if not self._ready or not query.strip(): return []
        results = self._fts_search("icd10_fts", query, limit)
        if not results:
            results = self._like_search("icd10", query, limit)
        return results

    def search_cpt(self, query: str, limit: int = 8) -> list:
        if not self._ready or not query.strip(): return []
        results = self._fts_search("cpt_fts", query, limit)
        if not results:
            results = self._like_search("cpt", query, limit)
        return results

    def _fts_search(self, table: str, query: str, limit: int) -> list:
        try:
            clean = " OR ".join(w for w in re.sub(r'[^\w\s]',' ',query.lower()).split() if len(w)>2)
            if not clean: return []
            rows = self._conn.execute(
                f"SELECT code,description,rank FROM {table} WHERE {table} MATCH ? ORDER BY rank LIMIT ?",
                (clean, limit)
            ).fetchall()
            return [(r["code"], r["description"], max(_CONF_FTS_LOW, _CONF_FTS_HIGH - i*0.03))
                    for i, r in enumerate(rows)]
        except:
            return []

    def _like_search(self, table: str, query: str, limit: int) -> list:
        try:
            rows = self._conn.execute(
                f"SELECT code,description FROM {table} WHERE LOWER(description) LIKE ? LIMIT ?",
                (f"%{query.lower()}%", limit)
            ).fetchall()
            return [(r["code"], r["description"], _CONF_FTS_LOW) for r in rows]
        except:
            return []

    def stats(self) -> dict:
        if not self._ready: return {"icd10":0,"cpt":0,"ready":False}
        return {
            "icd10": self._conn.execute("SELECT COUNT(*) FROM icd10").fetchone()[0],
            "cpt":   self._conn.execute("SELECT COUNT(*) FROM cpt").fetchone()[0],
            "ready": True,
        }

    def all_icd10_codes(self) -> set:
        if not self._ready: return set()
        return {r[0] for r in self._conn.execute("SELECT code FROM icd10")}

    def all_cpt_codes(self) -> set:
        if not self._ready: return set()
        return {r[0] for r in self._conn.execute("SELECT code FROM cpt")}


_db_singleton: Optional[CodeDatabase] = None

def get_code_db() -> CodeDatabase:
    global _db_singleton
    if _db_singleton is None:
        _db_singleton = CodeDatabase()
    return _db_singleton


class MedicalCoder:
    """Stage 4 — Maps ExtractionResult entities to ICD-10-CM and CPT codes."""

    def __init__(self, db: Optional[CodeDatabase] = None):
        self._db = db or get_code_db()

    def code(self, extraction: ExtractionResult) -> CodingResult:
        icd10, cpt, unm_dx, unm_proc = [], [], [], []

        for e in extraction.diagnoses:
            m = self._map(e, "icd10")
            if m: icd10.append(m)
            else:  unm_dx.append(e.text)

        for e in extraction.procedures:
            m = self._map(e, "cpt")
            if m: cpt.append(m)
            else:  unm_proc.append(e.text)

        icd10 = self._dedup(icd10)
        cpt   = self._dedup(cpt)
        all_c = icd10 + cpt
        overall = sum(c.confidence for c in all_c) / len(all_c) if all_c else 0.0

        logger.info(f"Coding: {len(icd10)} ICD-10, {len(cpt)} CPT, {len(unm_dx)} unmapped dx, {len(unm_proc)} unmapped proc")
        return CodingResult(
            icd10_codes=icd10, cpt_codes=cpt,
            unmapped_diagnoses=unm_dx, unmapped_procedures=unm_proc,
            overall_confidence=round(overall, 4),
        )

    def _map(self, entity: ClinicalEntity, system: str) -> Optional[CodeMapping]:
        raw  = (entity.normalized or entity.text).strip()
        # Apply synonym translation
        text = _SYNONYMS.get(raw.lower(), _SYNONYMS.get(entity.text.lower(), raw))

        # Exact code lookup
        if system == "icd10" and re.match(r'^[A-Z]\d{2}', text.upper()):
            r = self._db.lookup_icd10(text)
            if r: return self._cm(entity, r[0], r[1], CodeSystem.ICD10_CM, _CONF_EXACT, True)
        if system == "cpt" and re.match(r'^\d{5}', text):
            r = self._db.lookup_cpt(text)
            if r: return self._cm(entity, r[0], r[1], CodeSystem.CPT, _CONF_EXACT, True)

        # FTS search
        fn = self._db.search_icd10 if system == "icd10" else self._db.search_cpt
        csys = CodeSystem.ICD10_CM if system == "icd10" else CodeSystem.CPT
        candidates = fn(text, 8)
        if entity.text.lower() != text.lower():
            candidates = self._merge(candidates, fn(entity.text, 4))

        if not candidates: return None

        best = self._rerank(text, candidates)
        if best and best[2] >= _CONF_MIN:
            return self._cm(entity, best[0], best[1], csys, min(best[2], entity.confidence), False)
        return None

    def _rerank(self, query: str, candidates: list) -> Optional[tuple]:
        q = query.lower()
        scored = [(c, d, round(0.6*s + 0.4*SequenceMatcher(None,q,d.lower()).ratio(), 4))
                  for c, d, s in candidates]
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[0] if scored else None

    def _merge(self, a: list, b: list) -> list:
        seen = {x[0] for x in a}
        return a + [x for x in b if x[0] not in seen]

    def _cm(self, entity, code, desc, csys, conf, exact) -> CodeMapping:
        return CodeMapping(
            original_text=entity.text, code=code, code_system=csys,
            description=desc, confidence=round(min(conf,1.0),4), is_exact_match=exact,
        )

    def _dedup(self, codes: list) -> list:
        best = {}
        for cm in codes:
            if cm.code not in best or cm.confidence > best[cm.code].confidence:
                best[cm.code] = cm
        return list(best.values())

    # Compatibility with code_validator
    def is_valid_icd10(self, code: str) -> bool:
        return self._db.lookup_icd10(code) is not None

    def is_valid_cpt(self, code: str) -> bool:
        return self._db.lookup_cpt(code) is not None

    def get_all_icd10_codes(self) -> set:
        return self._db.all_icd10_codes()

    def get_all_cpt_codes(self) -> set:
        return self._db.all_cpt_codes()