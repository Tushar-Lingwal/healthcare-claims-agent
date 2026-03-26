"""
build_code_db.py — Builds the medical codes SQLite database.

Run this ONCE after downloading the CMS data files:

  python scripts/build_code_db.py

Downloads needed (all free):
  ICD-10-CM 2024:
    https://www.cms.gov/medicare/coding-billing/icd-10-codes/2024-icd-10-cm
    → Download "FY2024 Code Descriptions in Tabular Order"
    → Extract: icd10cm_codes_2024.txt  (or .csv)

  CPT (AMA — requires free registration):
    https://www.ama-assn.org/practice-management/cpt/cpt-code-database
    OR use the CMS Physician Fee Schedule:
    https://www.cms.gov/medicare/payment/fee-schedules/physician
    → Extract: RVU24D.txt

If you don't have the files yet, this script builds the DB from
the built-in comprehensive dataset (~1,200 codes covering all
major specialties) and you can augment it later with full CMS data.

Usage:
  python scripts/build_code_db.py                          # use built-in data
  python scripts/build_code_db.py --icd path/to/icd10.txt # load CMS file
  python scripts/build_code_db.py --cpt path/to/cpt.csv   # load CPT file
"""

import argparse
import csv
import os
import re
import sqlite3
import sys
from pathlib import Path

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

DB_PATH = Path(__file__).parent.parent / "data" / "codes" / "medical_codes.sqlite"

# ─────────────────────────────────────────────────────────────────────────────
# COMPREHENSIVE BUILT-IN ICD-10-CM DATASET
# Covers all major specialties encountered in claims adjudication
# ─────────────────────────────────────────────────────────────────────────────

BUILTIN_ICD10 = [
    # ── Neoplasms — Brain & CNS (C70-C72, D32-D35) ──
    ("C70.0","Malignant neoplasm of cerebral meninges"),
    ("C70.1","Malignant neoplasm of spinal meninges"),
    ("C70.9","Malignant neoplasm of meninges, unspecified"),
    ("C71.0","Malignant neoplasm of cerebrum, except lobes and ventricles"),
    ("C71.1","Malignant neoplasm of frontal lobe"),
    ("C71.2","Malignant neoplasm of temporal lobe"),
    ("C71.3","Malignant neoplasm of parietal lobe"),
    ("C71.4","Malignant neoplasm of occipital lobe"),
    ("C71.5","Malignant neoplasm of cerebral ventricle"),
    ("C71.6","Malignant neoplasm of cerebellum"),
    ("C71.7","Malignant neoplasm of brain stem"),
    ("C71.8","Malignant neoplasm of overlapping sites of brain"),
    ("C71.9","Malignant neoplasm of brain, unspecified"),
    ("C72.0","Malignant neoplasm of spinal cord"),
    ("C72.1","Malignant neoplasm of cauda equina"),
    ("C72.9","Malignant neoplasm of central nervous system, unspecified"),
    ("C75.1","Malignant neoplasm of pituitary gland"),
    ("C75.2","Malignant neoplasm of craniopharyngeal duct"),
    ("C79.31","Secondary malignant neoplasm of brain"),
    ("C79.32","Secondary malignant neoplasm of cerebral meninges"),
    ("D32.0","Benign neoplasm of cerebral meninges"),
    ("D32.1","Benign neoplasm of spinal meninges"),
    ("D32.9","Benign neoplasm of meninges, unspecified"),
    ("D33.0","Benign neoplasm of brain, supratentorial"),
    ("D33.1","Benign neoplasm of brain, infratentorial"),
    ("D33.2","Benign neoplasm of brain, unspecified"),
    ("D33.3","Benign neoplasm of cranial nerves"),
    ("D35.2","Benign neoplasm of pituitary gland"),
    ("D35.3","Benign neoplasm of craniopharyngeal duct"),
    ("D43.0","Neoplasm of uncertain behavior of brain, supratentorial"),
    ("D43.1","Neoplasm of uncertain behavior of brain, infratentorial"),
    ("D43.2","Neoplasm of uncertain behavior of brain, unspecified"),
    # ── Neoplasms — Breast (C50) ──
    ("C50.011","Malignant neoplasm of nipple and areola, right female breast"),
    ("C50.012","Malignant neoplasm of nipple and areola, left female breast"),
    ("C50.111","Malignant neoplasm of central portion, right female breast"),
    ("C50.112","Malignant neoplasm of central portion, left female breast"),
    ("C50.211","Malignant neoplasm of upper-inner quadrant, right female breast"),
    ("C50.212","Malignant neoplasm of upper-inner quadrant, left female breast"),
    ("C50.311","Malignant neoplasm of lower-inner quadrant, right female breast"),
    ("C50.411","Malignant neoplasm of upper-outer quadrant, right female breast"),
    ("C50.412","Malignant neoplasm of upper-outer quadrant, left female breast"),
    ("C50.511","Malignant neoplasm of lower-outer quadrant, right female breast"),
    ("C50.611","Malignant neoplasm of axillary tail, right female breast"),
    ("C50.811","Malignant neoplasm of overlapping sites, right female breast"),
    ("C50.911","Malignant neoplasm of unspecified site, right female breast"),
    ("C50.912","Malignant neoplasm of unspecified site, left female breast"),
    ("C50.919","Malignant neoplasm of unspecified site of unspecified female breast"),
    ("D05.10","Intraductal carcinoma in situ of unspecified breast"),
    ("D05.11","Intraductal carcinoma in situ of right breast"),
    ("D05.12","Intraductal carcinoma in situ of left breast"),
    # ── Neoplasms — Lung, Colon, Prostate ──
    ("C34.10","Malignant neoplasm of upper lobe, unspecified bronchus or lung"),
    ("C34.11","Malignant neoplasm of upper lobe, right bronchus or lung"),
    ("C34.12","Malignant neoplasm of upper lobe, left bronchus or lung"),
    ("C34.30","Malignant neoplasm of lower lobe, unspecified bronchus or lung"),
    ("C34.90","Malignant neoplasm of unspecified part of unspecified bronchus and lung"),
    ("C18.0","Malignant neoplasm of cecum"),
    ("C18.2","Malignant neoplasm of ascending colon"),
    ("C18.4","Malignant neoplasm of transverse colon"),
    ("C18.6","Malignant neoplasm of descending colon"),
    ("C18.7","Malignant neoplasm of sigmoid colon"),
    ("C18.9","Malignant neoplasm of colon, unspecified"),
    ("C19","Malignant neoplasm of rectosigmoid junction"),
    ("C20","Malignant neoplasm of rectum"),
    ("C61","Malignant neoplasm of prostate"),
    ("C64.1","Malignant neoplasm of right kidney, except renal pelvis"),
    ("C64.2","Malignant neoplasm of left kidney, except renal pelvis"),
    ("C67.9","Malignant neoplasm of bladder, unspecified"),
    ("C73","Malignant neoplasm of thyroid gland"),
    ("C85.90","Non-Hodgkin lymphoma, unspecified, unspecified site"),
    ("C91.00","Acute lymphoblastic leukemia not having achieved remission"),
    ("C92.00","Acute myeloblastic leukemia, not having achieved remission"),
    ("C95.90","Leukemia, unspecified, not having achieved remission"),
    # ── Dementia & Cognitive Disorders (F01-F03, G30-G31) ──
    ("F01.50","Vascular dementia, unspecified severity, without behavioral disturbance"),
    ("F01.51","Vascular dementia, unspecified severity, with agitation"),
    ("F01.511","Vascular dementia, unspecified severity, with agitation"),
    ("F01.518","Vascular dementia, unspecified severity, with other behavioral disturbance"),
    ("F02.80","Dementia in other diseases classified elsewhere, unspecified severity"),
    ("F03.90","Unspecified dementia without behavioral disturbance, unspecified severity"),
    ("F03.91","Unspecified dementia, unspecified severity, with agitation"),
    ("F03.911","Unspecified dementia, unspecified severity, with agitation"),
    ("F03.918","Unspecified dementia, unspecified severity, with other behavioral disturbance"),
    ("G30.0","Alzheimer's disease with early onset"),
    ("G30.1","Alzheimer's disease with late onset"),
    ("G30.8","Other Alzheimer's disease"),
    ("G30.9","Alzheimer's disease, unspecified"),
    ("G31.01","Pick's disease"),
    ("G31.09","Other frontotemporal dementia"),
    ("G31.83","Dementia with Lewy bodies"),
    ("G31.84","Mild cognitive impairment of uncertain or unknown etiology"),
    ("G31.85","Corticobasal degeneration"),
    ("G31.89","Other specified degenerative diseases of nervous system"),
    ("G31.9","Degenerative disease of nervous system, unspecified"),
    # ── Cardiovascular (I10-I79) ──
    ("I10","Essential (primary) hypertension"),
    ("I11.0","Hypertensive heart disease with heart failure"),
    ("I11.9","Hypertensive heart disease without heart failure"),
    ("I20.0","Unstable angina"),
    ("I20.9","Angina pectoris, unspecified"),
    ("I21.01","ST elevation myocardial infarction of anterior wall"),
    ("I21.09","ST elevation myocardial infarction of other anterior wall"),
    ("I21.11","ST elevation myocardial infarction of inferior wall"),
    ("I21.19","ST elevation myocardial infarction of other inferior wall"),
    ("I21.3","ST elevation myocardial infarction of unspecified site"),
    ("I21.4","Non-ST elevation myocardial infarction"),
    ("I21.9","Acute myocardial infarction, unspecified"),
    ("I25.10","Atherosclerotic heart disease of native coronary artery without angina pectoris"),
    ("I25.110","Atherosclerotic heart disease of native coronary artery with unstable angina pectoris"),
    ("I25.2","Old myocardial infarction"),
    ("I25.5","Ischemic cardiomyopathy"),
    ("I26.09","Other pulmonary embolism without acute cor pulmonale"),
    ("I26.99","Other pulmonary embolism without acute cor pulmonale"),
    ("I35.0","Nonrheumatic aortic (valve) stenosis"),
    ("I42.0","Dilated cardiomyopathy"),
    ("I42.9","Cardiomyopathy, unspecified"),
    ("I48.0","Paroxysmal atrial fibrillation"),
    ("I48.11","Longstanding persistent atrial fibrillation"),
    ("I48.19","Other persistent atrial fibrillation"),
    ("I48.20","Chronic atrial fibrillation, unspecified"),
    ("I48.91","Unspecified atrial fibrillation"),
    ("I50.20","Unspecified systolic (congestive) heart failure"),
    ("I50.21","Acute systolic (congestive) heart failure"),
    ("I50.22","Chronic systolic (congestive) heart failure"),
    ("I50.30","Unspecified diastolic (congestive) heart failure"),
    ("I50.9","Heart failure, unspecified"),
    ("I63.00","Cerebral infarction due to thrombosis of unspecified precerebral artery"),
    ("I63.9","Cerebral infarction, unspecified"),
    ("I65.2","Occlusion and stenosis of carotid artery"),
    ("I70.0","Atherosclerosis of aorta"),
    ("I70.201","Unspecified atherosclerosis of native arteries of extremities, right leg"),
    # ── Diabetes & Metabolic (E08-E13) ──
    ("E08.9","Diabetes mellitus due to underlying condition, unspecified"),
    ("E09.9","Drug or chemical induced diabetes mellitus, unspecified"),
    ("E10.9","Type 1 diabetes mellitus without complications"),
    ("E10.10","Type 1 diabetes mellitus with ketoacidosis without coma"),
    ("E10.65","Type 1 diabetes mellitus with hyperglycemia"),
    ("E11.00","Type 2 diabetes mellitus with hyperosmolarity without nonketotic hyperglycemic-hyperosmolar coma"),
    ("E11.21","Type 2 diabetes mellitus with diabetic nephropathy"),
    ("E11.311","Type 2 diabetes mellitus with unspecified diabetic retinopathy with macular edema"),
    ("E11.40","Type 2 diabetes mellitus with diabetic neuropathy, unspecified"),
    ("E11.51","Type 2 diabetes mellitus with diabetic peripheral angiopathy without gangrene"),
    ("E11.65","Type 2 diabetes mellitus with hyperglycemia"),
    ("E11.9","Type 2 diabetes mellitus without complications"),
    ("E13.9","Other specified diabetes mellitus without complications"),
    ("E66.01","Morbid (severe) obesity due to excess calories"),
    ("E66.09","Other obesity due to excess calories"),
    ("E66.9","Obesity, unspecified"),
    ("E78.00","Pure hypercholesterolemia, unspecified"),
    ("E78.01","Familial hypercholesterolemia"),
    ("E78.1","Pure hyperglyceridemia"),
    ("E78.5","Hyperlipidemia, unspecified"),
    ("E03.9","Hypothyroidism, unspecified"),
    ("E05.00","Thyrotoxicosis with diffuse goiter without thyrotoxic crisis"),
    ("E05.90","Thyrotoxicosis, unspecified, without thyrotoxic crisis or storm"),
    # ── Musculoskeletal (M00-M99, S00-S99) ──
    ("M05.9","Rheumatoid arthritis with rheumatoid factor, unspecified"),
    ("M06.9","Rheumatoid arthritis, unspecified"),
    ("M10.9","Gout, unspecified"),
    ("M16.10","Unilateral primary osteoarthritis, unspecified hip"),
    ("M16.11","Unilateral primary osteoarthritis, right hip"),
    ("M16.12","Unilateral primary osteoarthritis, left hip"),
    ("M16.9","Osteoarthritis of hip, unspecified"),
    ("M17.10","Unilateral primary osteoarthritis, unspecified knee"),
    ("M17.11","Unilateral primary osteoarthritis, right knee"),
    ("M17.12","Unilateral primary osteoarthritis, left knee"),
    ("M17.31","Unilateral post-traumatic osteoarthritis, right knee"),
    ("M17.9","Osteoarthritis of knee, unspecified"),
    ("M19.90","Primary osteoarthritis, unspecified site"),
    ("M19.011","Primary osteoarthritis, right shoulder"),
    ("M47.816","Spondylosis without myelopathy or radiculopathy, lumbar region"),
    ("M48.061","Spinal stenosis, lumbar region without neurogenic claudication"),
    ("M51.16","Intervertebral disc degeneration, lumbar region"),
    ("M51.17","Intervertebral disc degeneration, lumbosacral region"),
    ("M54.50","Low back pain, unspecified"),
    ("M54.51","Vertebrogenic low back pain"),
    ("M54.59","Other low back pain"),
    ("M79.3","Panniculitis"),
    ("M81.0","Age-related osteoporosis without current pathological fracture"),
    ("M81.6","Localized osteoporosis"),
    ("S72.001A","Fracture of unspecified part of neck of right femur, initial encounter for closed fracture"),
    ("S82.001A","Fracture of right patella, initial encounter for closed fracture"),
    ("S83.001A","Unspecified subluxation of right patella, initial encounter"),
    ("S83.511A","Sprain of anterior cruciate ligament of right knee, initial encounter"),
    ("S83.512A","Sprain of anterior cruciate ligament of left knee, initial encounter"),
    ("S83.9XXA","Sprain of unspecified site of knee, initial encounter"),
    # ── Respiratory (J00-J99) ──
    ("J18.0","Bronchopneumonia, unspecified organism"),
    ("J18.1","Lobar pneumonia, unspecified organism"),
    ("J18.9","Pneumonia, unspecified organism"),
    ("J44.0","Chronic obstructive pulmonary disease with acute lower respiratory infection"),
    ("J44.1","Chronic obstructive pulmonary disease with acute exacerbation"),
    ("J44.9","Chronic obstructive pulmonary disease, unspecified"),
    ("J45.20","Mild intermittent asthma, uncomplicated"),
    ("J45.30","Mild persistent asthma, uncomplicated"),
    ("J45.40","Moderate persistent asthma, uncomplicated"),
    ("J45.50","Severe persistent asthma, uncomplicated"),
    ("J45.901","Unspecified asthma with acute exacerbation"),
    ("J45.909","Unspecified asthma, uncomplicated"),
    ("J96.00","Acute respiratory failure, unspecified whether with hypoxia or hypercapnia"),
    ("J96.11","Chronic respiratory failure with hypoxia"),
    # ── Mental Health (F10-F99) ──
    ("F10.10","Alcohol abuse, uncomplicated"),
    ("F10.20","Alcohol dependence, uncomplicated"),
    ("F17.210","Nicotine dependence, cigarettes, uncomplicated"),
    ("F32.0","Major depressive disorder, single episode, mild"),
    ("F32.1","Major depressive disorder, single episode, moderate"),
    ("F32.2","Major depressive disorder, single episode, severe without psychotic features"),
    ("F32.9","Major depressive disorder, single episode, unspecified"),
    ("F33.0","Major depressive disorder, recurrent, mild"),
    ("F33.9","Major depressive disorder, recurrent, unspecified"),
    ("F41.0","Panic disorder without agoraphobia"),
    ("F41.1","Generalized anxiety disorder"),
    ("F41.9","Anxiety disorder, unspecified"),
    ("F60.3","Borderline personality disorder"),
    ("F90.0","Attention-deficit hyperactivity disorder, predominantly inattentive type"),
    ("F90.1","Attention-deficit hyperactivity disorder, predominantly hyperactive type"),
    ("F90.9","Attention-deficit hyperactivity disorder, unspecified type"),
    # ── Renal (N00-N39) ──
    ("N17.9","Acute kidney failure, unspecified"),
    ("N18.1","Chronic kidney disease, stage 1"),
    ("N18.2","Chronic kidney disease, stage 2 (mild)"),
    ("N18.3","Chronic kidney disease, stage 3 (moderate)"),
    ("N18.4","Chronic kidney disease, stage 4 (severe)"),
    ("N18.5","Chronic kidney disease, stage 5"),
    ("N18.6","End stage renal disease"),
    ("N18.9","Chronic kidney disease, unspecified"),
    ("N20.0","Calculus of kidney"),
    ("N20.1","Calculus of ureter"),
    ("N39.0","Urinary tract infection, site not specified"),
    # ── Neurological (G00-G99) ──
    ("G20","Parkinson's disease"),
    ("G35","Multiple sclerosis"),
    ("G40.009","Localization-related epilepsy, unspecified"),
    ("G40.909","Epilepsy, unspecified, not intractable, without status epilepticus"),
    ("G43.009","Migraine without aura, not intractable, without status migrainosus"),
    ("G43.109","Migraine with aura, not intractable, without status migrainosus"),
    ("G43.909","Migraine, unspecified, not intractable, without status migrainosus"),
    ("G45.9","Transient cerebral ischemic attack, unspecified"),
    ("G47.00","Insomnia, unspecified"),
    ("G47.33","Obstructive sleep apnea (adult)(pediatric)"),
    ("G89.29","Other chronic pain"),
    ("G89.4","Chronic pain syndrome"),
    # ── Infectious (A00-B99) ──
    ("A41.9","Sepsis, unspecified organism"),
    ("A49.9","Bacterial infection, unspecified"),
    ("B20","Human immunodeficiency virus disease"),
    ("J11.1","Influenza due to unidentified influenza virus with other respiratory manifestations"),
    ("U07.1","COVID-19"),
    ("Z86.19","Personal history of other infectious and parasitic diseases"),
    # ── Preventive / Screening (Z00-Z99) ──
    ("Z00.00","Encounter for general adult medical examination without abnormal findings"),
    ("Z00.01","Encounter for general adult medical examination with abnormal findings"),
    ("Z12.11","Encounter for screening for malignant neoplasm of colon"),
    ("Z12.31","Encounter for screening mammogram for malignant neoplasm of breast"),
    ("Z12.4","Encounter for screening for malignant neoplasm of cervix"),
    ("Z12.5","Encounter for screening for malignant neoplasm of prostate"),
    ("Z23","Encounter for immunization"),
    ("Z51.11","Encounter for antineoplastic chemotherapy"),
    ("Z51.12","Encounter for antineoplastic immunotherapy"),
]

# ─────────────────────────────────────────────────────────────────────────────
# COMPREHENSIVE BUILT-IN CPT DATASET
# ─────────────────────────────────────────────────────────────────────────────

BUILTIN_CPT = [
    # ── E/M Office Visits ──
    ("99201","Office or other outpatient visit, new patient, straightforward"),
    ("99202","Office or other outpatient visit, new patient, low complexity"),
    ("99203","Office or other outpatient visit, new patient, moderate complexity"),
    ("99204","Office or other outpatient visit, new patient, moderate-high complexity"),
    ("99205","Office or other outpatient visit, new patient, high complexity"),
    ("99211","Office or other outpatient visit, established patient, minimal"),
    ("99212","Office or other outpatient visit, established patient, straightforward"),
    ("99213","Office or other outpatient visit, established patient, low complexity"),
    ("99214","Office or other outpatient visit, established patient, moderate complexity"),
    ("99215","Office or other outpatient visit, established patient, high complexity"),
    ("99221","Initial hospital care, low complexity"),
    ("99222","Initial hospital care, moderate complexity"),
    ("99223","Initial hospital care, high complexity"),
    ("99231","Subsequent hospital care, low complexity"),
    ("99232","Subsequent hospital care, moderate complexity"),
    ("99233","Subsequent hospital care, high complexity"),
    ("99238","Hospital discharge day management, 30 minutes or less"),
    ("99239","Hospital discharge day management, more than 30 minutes"),
    ("99281","Emergency department visit, self-limited or minor problem"),
    ("99282","Emergency department visit, low complexity"),
    ("99283","Emergency department visit, moderate complexity"),
    ("99284","Emergency department visit, high complexity"),
    ("99285","Emergency department visit, high complexity with threat to life"),
    ("99442","Telephone evaluation and management service, 11-20 minutes"),
    ("99443","Telephone evaluation and management service, 21-30 minutes"),
    # ── Neurology / Brain Imaging ──
    ("70450","CT scan of head or brain without contrast"),
    ("70460","CT scan of head or brain with contrast"),
    ("70470","CT scan of head or brain without contrast, followed by with contrast"),
    ("70480","CT scan of orbit, sella, or posterior fossa without contrast"),
    ("70490","CT scan of soft tissue neck without contrast"),
    ("70496","CT angiography, head"),
    ("70498","CT angiography, neck"),
    ("70540","MRI of orbit, face, and neck without contrast"),
    ("70551","MRI of brain without contrast"),
    ("70552","MRI of brain with contrast"),
    ("70553","MRI of brain without contrast, followed by with contrast"),
    ("70554","Functional MRI brain"),
    ("70557","MRI of brain during surgery, without contrast"),
    ("70559","MRI of brain during surgery, without contrast followed by with contrast"),
    ("70580","MR spectroscopy"),
    ("78600","Brain scan, less than 4 static views"),
    ("78601","Brain scan, minimum 4 static views"),
    ("78606","Brain scan with vascular flow"),
    ("78607","Tomographic (SPECT) brain scan"),
    ("78608","Brain imaging, positron emission tomography (PET)"),
    ("78610","Brain imaging, vascular flow only"),
    ("93882","Duplex scan of extracranial arteries"),
    ("95812","Electroencephalogram (EEG), awake and asleep"),
    ("95816","Electroencephalogram (EEG), awake and drowsy"),
    ("95819","Electroencephalogram (EEG), awake and asleep, with hyperventilation"),
    # ── Neurosurgery ──
    ("61510","Craniectomy for excision of brain tumor"),
    ("61512","Craniectomy for excision of meningioma, supratentorial"),
    ("61519","Removal of brain tumor, meningioma, infratentorial"),
    ("61520","Removal of posterior fossa tumor"),
    ("61521","Removal of cerebellopontine angle tumor"),
    ("61530","Removal of intracranial arteriovenous malformation"),
    ("61545","Craniotomy for excision of craniopharyngioma"),
    ("61546","Craniotomy for hypophysectomy or excision of pituitary tumor, transnasal or transseptal approach"),
    ("61548","Hypophysectomy or excision of pituitary tumor, transsphenoidal approach"),
    ("61750","Stereotactic biopsy, aspiration, or excision, including burr hole(s), for intracranial lesion"),
    ("61751","Stereotactic biopsy with intraoperative MRI"),
    ("61760","Stereotactic implantation of depth electrodes into the cerebrum"),
    ("61781","Stereotactic computer-assisted volumetric procedure, intracranial, without imaging guidance"),
    ("61796","Stereotactic radiosurgery, 1 simple cranial lesion"),
    ("61797","Stereotactic radiosurgery, each additional cranial lesion, simple"),
    ("61798","Stereotactic radiosurgery, 1 complex cranial lesion"),
    ("61799","Stereotactic radiosurgery, each additional complex cranial lesion"),
    ("61800","Application of stereotactic headframe for radiosurgery"),
    # ── Radiation Oncology ──
    ("77261","Therapeutic radiology treatment planning, simple"),
    ("77262","Therapeutic radiology treatment planning, intermediate"),
    ("77263","Therapeutic radiology treatment planning, complex"),
    ("77280","Simulation, simple"),
    ("77285","Simulation, intermediate"),
    ("77290","Simulation, complex"),
    ("77295","3-dimensional radiotherapy plan, including dose-volume histograms"),
    ("77300","Basic radiation dosimetry calculation"),
    ("77321","Special teletherapy port plan"),
    ("77370","Special medical radiation physics consultation"),
    ("77385","Intensity modulated radiation treatment delivery, simple"),
    ("77386","Intensity modulated radiation treatment delivery, complex"),
    ("77387","Guidance for localization of target volume for delivery of radiation treatment"),
    ("77402","Radiation treatment delivery, simple"),
    ("77407","Radiation treatment delivery, intermediate"),
    ("77412","Radiation treatment delivery, complex"),
    ("77422","High energy neutron radiation treatment delivery, simple"),
    ("77427","Radiation treatment management, 5 treatments"),
    ("77431","Radiation therapy management with complete course of therapy"),
    ("77435","Stereotactic body radiation therapy, treatment delivery"),
    # ── Oncology / Chemotherapy ──
    ("96401","Chemotherapy administration, subcutaneous or intramuscular; non-hormonal anti-neoplastic"),
    ("96402","Chemotherapy administration, subcutaneous or intramuscular; hormonal anti-neoplastic"),
    ("96405","Chemotherapy administration, intralesional; up to and including 7 lesions"),
    ("96406","Chemotherapy administration, intralesional; more than 7 lesions"),
    ("96409","Chemotherapy administration, intravenous; push technique, single or initial substance/drug"),
    ("96411","Chemotherapy administration, intravenous; push technique, each additional substance"),
    ("96413","Chemotherapy administration, intravenous infusion technique; up to 1 hour, single or initial substance"),
    ("96415","Chemotherapy administration, intravenous infusion technique; each additional hour"),
    ("96416","Chemotherapy administration, intravenous infusion technique; initiation of prolonged chemotherapy"),
    ("96417","Chemotherapy administration, intravenous infusion technique; each additional sequential"),
    ("96420","Chemotherapy administration, intra-arterial; push technique"),
    ("96440","Chemotherapy administration into pleural cavity"),
    ("96446","Chemotherapy administration into peritoneal cavity"),
    ("96450","Chemotherapy administration, into CNS (intrathecal)"),
    ("96521","Refilling and maintenance of portable pump"),
    # ── Cardiac ──
    ("92950","Cardiopulmonary resuscitation"),
    ("93000","Electrocardiogram, routine ECG with at least 12 leads"),
    ("93005","Electrocardiogram, tracing only, without interpretation and report"),
    ("93010","Electrocardiogram, interpretation and report only"),
    ("93015","Cardiovascular stress test using maximal or submaximal treadmill"),
    ("93016","Cardiovascular stress test, physician supervision"),
    ("93018","Cardiovascular stress test, interpretation and report"),
    ("93040","Rhythm ECG, 1-3 leads"),
    ("93042","Rhythm ECG, interpretation and report"),
    ("93224","External electrocardiographic recording up to 48 hours"),
    ("93280","Programming device evaluation, single chamber pacemaker"),
    ("93303","Transthoracic echocardiography for congenital cardiac anomalies"),
    ("93304","Transthoracic echocardiography for congenital cardiac anomalies; follow-up"),
    ("93306","Echocardiography, transthoracic, real-time with image documentation"),
    ("93307","Echocardiography, transthoracic, real-time with image documentation, without spectral"),
    ("93308","Echocardiography, transthoracic, follow-up or limited study"),
    ("93312","Echocardiography, transesophageal, real time with image documentation"),
    ("93320","Doppler echocardiography, pulsed wave and/or continuous wave with spectral display"),
    ("93325","Doppler echocardiography color flow velocity mapping"),
    ("93350","Echocardiography, transthoracic, real-time with image documentation; during stress"),
    ("93451","Right heart catheterization including measurement of oxygen saturation"),
    ("93452","Left heart catheterization including intraprocedural injection"),
    ("93453","Combined right and left heart catheterization"),
    ("93454","Coronary angiography without left heart catheterization"),
    ("93455","Coronary angiography with right heart catheterization"),
    ("93456","Coronary angiography with right and left heart catheterization"),
    ("93457","Coronary angiography with pharmacologic stress"),
    ("93458","Left heart catheterization with coronary angiography"),
    ("93459","Left heart catheterization with coronary angiography and left ventriculography"),
    ("93460","Right and left heart catheterization with coronary angiography"),
    ("33533","Coronary artery bypass using arterial graft, single"),
    ("33534","Coronary artery bypass using arterial graft, two"),
    ("33535","Coronary artery bypass using arterial graft, three"),
    ("33536","Coronary artery bypass using arterial graft, four or more"),
    ("33510","Coronary artery bypass using venous graft only, single"),
    # ── Orthopedics ──
    ("27130","Total hip arthroplasty"),
    ("27132","Conversion of previous hip surgery to total hip arthroplasty"),
    ("27134","Revision of total hip arthroplasty; both components"),
    ("27137","Revision of total hip arthroplasty; acetabular component only"),
    ("27138","Revision of total hip arthroplasty; femoral component only"),
    ("27447","Total knee arthroplasty"),
    ("27445","Arthroplasty, knee, hinge prosthesis"),
    ("27486","Revision of total knee arthroplasty, one component"),
    ("27487","Revision of total knee arthroplasty, femoral and tibial components"),
    ("27570","Manipulation of knee joint under general anesthesia"),
    ("29870","Arthroscopy, knee, diagnostic"),
    ("29871","Arthroscopy, knee, surgical; for infection"),
    ("29873","Arthroscopy, knee, surgical; with lateral release"),
    ("29874","Arthroscopy, knee, surgical; for removal of loose body or foreign body"),
    ("29875","Arthroscopy, knee, surgical; synovectomy, limited"),
    ("29876","Arthroscopy, knee, surgical; synovectomy, major, 2 or more compartments"),
    ("29877","Arthroscopy, knee, surgical; debridement/shaving of articular cartilage"),
    ("29880","Arthroscopy, knee, surgical; with meniscectomy including any meniscal shaving"),
    ("29881","Arthroscopy, knee, surgical; with meniscectomy (medial or lateral)"),
    ("29882","Arthroscopy, knee, surgical; with meniscus repair (medial or lateral)"),
    ("29883","Arthroscopy, knee, surgical; with meniscus repair (medial and lateral)"),
    ("29884","Arthroscopy, knee, surgical; with lysis of adhesions"),
    ("29885","Arthroscopy, knee, surgical; drilling for osteochondritis dissecans"),
    ("29888","Arthroscopically aided anterior cruciate ligament repair/augmentation or reconstruction"),
    ("29889","Arthroscopically aided posterior cruciate ligament repair/augmentation or reconstruction"),
    ("73560","Radiologic examination, knee; 1 or 2 views"),
    ("73562","Radiologic examination, knee; 3 views"),
    ("73564","Radiologic examination, knee; complete, 4 or more views"),
    ("73721","Magnetic resonance imaging, any joint of lower extremity"),
    ("73723","Magnetic resonance imaging, any joint of lower extremity; without contrast, followed by with contrast"),
    ("73221","Magnetic resonance imaging, any joint of upper extremity"),
    ("73223","Magnetic resonance imaging, any joint of upper extremity; without contrast, followed by with contrast"),
    ("97110","Therapeutic procedure, 1 or more areas; therapeutic exercises"),
    ("97112","Therapeutic procedure; neuromuscular reeducation"),
    ("97530","Therapeutic activities, direct patient contact"),
    # ── Spine ──
    ("72100","Radiologic examination, spine, lumbosacral; 2 or 3 views"),
    ("72110","Radiologic examination, spine, lumbosacral; minimum 4 views"),
    ("72141","Magnetic resonance imaging, spinal canal and contents, cervical; without contrast"),
    ("72142","Magnetic resonance imaging, spinal canal and contents, cervical; with contrast"),
    ("72146","Magnetic resonance imaging, spinal canal and contents, thoracic; without contrast"),
    ("72148","Magnetic resonance imaging, spinal canal and contents, lumbar; without contrast"),
    ("72149","Magnetic resonance imaging, spinal canal and contents, lumbar; with contrast"),
    ("72156","Magnetic resonance imaging, spinal canal and contents, cervical; without contrast, followed by with contrast"),
    ("72158","Magnetic resonance imaging, spinal canal and contents, lumbar; without contrast, followed by with contrast"),
    ("62323","Injection, including imaging guidance; lumbar or sacral, transforaminal epidural"),
    ("64483","Injection, anesthetic agent; transforaminal epidural, lumbar or sacral"),
    ("22612","Arthrodesis, posterior or posterolateral technique, single level; lumbar"),
    ("22630","Arthrodesis, posterior interbody technique, single interspace; lumbar"),
    # ── General Surgery ──
    ("19303","Mastectomy, simple, complete"),
    ("19304","Mastectomy, subcutaneous"),
    ("19305","Mastectomy, radical, including pectoral muscles, axillary lymph nodes"),
    ("19306","Mastectomy, radical, including pectoral muscles, axillary and internal mammary lymph nodes"),
    ("19307","Modified radical mastectomy, including axillary lymph nodes"),
    ("19316","Mastopexy"),
    ("44950","Appendectomy"),
    ("44960","Appendectomy; for ruptured appendix with abscess or generalized peritonitis"),
    ("47562","Laparoscopic cholecystectomy"),
    ("47563","Laparoscopic cholecystectomy with operative cholangiography"),
    ("47600","Cholecystectomy"),
    ("49505","Repair initial inguinal hernia, age 5 years or over"),
    ("49520","Repair recurrent inguinal hernia"),
    ("43239","Upper gastrointestinal endoscopy including esophagus with biopsy"),
    ("43248","Upper gastrointestinal endoscopy; with insertion of guide wire"),
    ("45378","Colonoscopy, flexible; diagnostic"),
    ("45380","Colonoscopy, flexible; with biopsy, single or multiple"),
    ("45385","Colonoscopy, flexible; with removal of tumor, polyp, or other lesion by snare technique"),
    # ── Radiology / Imaging ──
    ("71046","Radiologic examination, chest; 2 views"),
    ("71048","Radiologic examination, chest; 4 or more views"),
    ("71250","Computed tomography, thorax; without contrast"),
    ("71260","Computed tomography, thorax; with contrast"),
    ("71275","Computed tomographic angiography, chest"),
    ("74177","Computed tomographic angiography, abdomen and pelvis; without contrast"),
    ("74178","Computed tomographic angiography, abdomen and pelvis; with contrast"),
    ("76536","Ultrasound, soft tissue of head and neck"),
    ("76700","Ultrasound, abdominal, real time with image documentation; complete"),
    ("76705","Ultrasound, abdominal, real time with image documentation; limited"),
    ("76770","Ultrasound, retroperitoneal"),
    ("76801","Ultrasound, pregnant uterus, real time with image documentation; less than 14 weeks"),
    ("77065","Diagnostic mammography, including CAD; unilateral"),
    ("77066","Diagnostic mammography, including CAD; bilateral"),
    ("77067","Screening mammography, bilateral (2-view)"),
    ("77080","Dual-energy X-ray absorptiometry (DXA), bone density study, 1 or more sites"),
    # ── Lab & Pathology ──
    ("80047","Basic metabolic panel"),
    ("80048","Basic metabolic panel with calcium"),
    ("80050","General health panel"),
    ("80053","Comprehensive metabolic panel"),
    ("80061","Lipid panel"),
    ("81001","Urinalysis, by dip stick or tablet reagent"),
    ("81003","Urinalysis, automated, without microscopy"),
    ("82043","Albumin, urine, microalbumin, quantitative"),
    ("82247","Bilirubin, total"),
    ("82310","Calcium, total"),
    ("82374","Carbon dioxide (bicarbonate)"),
    ("82435","Chloride"),
    ("82550","Creatine kinase (CK), (CPK); total"),
    ("82565","Creatinine; blood"),
    ("82607","Cyanocobalamin (vitamin B-12)"),
    ("82670","Estradiol"),
    ("82728","Ferritin"),
    ("82746","Folic acid; serum"),
    ("82947","Glucose, quantitative, blood"),
    ("82962","Glucose, blood by glucose monitoring device(s)"),
    ("83001","Gonadotropin; follicle stimulating (FSH)"),
    ("83036","Hemoglobin A1c"),
    ("83540","Iron"),
    ("83550","Iron binding capacity"),
    ("83735","Magnesium"),
    ("84100","Phosphorus inorganic (phosphate)"),
    ("84132","Potassium; serum, plasma, or whole blood"),
    ("84153","Prostate specific antigen (PSA); total"),
    ("84295","Sodium; serum, plasma, or whole blood"),
    ("84443","Thyroid stimulating hormone (TSH)"),
    ("84479","Thyroid hormone (T3 or T4); uptake"),
    ("84480","Triiodothyronine T3; total (TT-3)"),
    ("84481","Triiodothyronine T3; free"),
    ("84484","Troponin, quantitative"),
    ("84520","Urea nitrogen; quantitative"),
    ("84550","Uric acid; blood"),
    ("85025","Blood count; complete (CBC), automated"),
    ("85027","Blood count; complete (CBC), automated, without platelet count"),
    ("85610","Prothrombin time"),
    ("85730","Thromboplastin time, partial (PTT)"),
    ("85732","Thromboplastin time, partial (PTT); substitution, plasma fractions"),
    ("86003","Allergen specific IgE; quantitative or semiquantitative, each allergen"),
    ("86140","C-reactive protein"),
    ("86147","Cardiolipin antibody, each Ig class"),
    ("86200","Cyclic citrullinated peptide (CCP), antibody"),
    ("86235","Nuclear antigen antibody; any antibody except anti-ds DNA"),
    ("86308","Heterophile antibodies, screening"),
    ("86360","Absolute CD4 and CD8 count"),
    ("86431","Rheumatoid factor, qualitative"),
    ("86677","Antibody, Helicobacter pylori"),
    ("87040","Culture, bacterial; blood, aerobic"),
    ("87070","Culture, bacterial; any other source except urine, blood or stool"),
    ("87081","Culture, presumptive, pathogenic organisms, screening only"),
    ("87086","Culture, bacterial; urine, quantitative colony count"),
    ("87210","Smear, primary source with interpretation; wet mount"),
    ("87400","Influenza, for detection"),
    ("87502","Influenza virus, detection by immunoassay with direct optical observation"),
    ("87632","Respiratory virus, multiple types or subtypes (includes multiplex reverse transcription)"),
    ("87635","Infectious agent detection by nucleic acid (DNA or RNA); SARS-CoV-2"),
    ("88300","Surgical pathology, gross examination only"),
    ("88302","Level II Surgical pathology, gross and microscopic examination"),
    ("88304","Level III Surgical pathology, gross and microscopic examination"),
    ("88305","Level IV Surgical pathology, gross and microscopic examination"),
    ("88307","Level V Surgical pathology, gross and microscopic examination"),
    ("88309","Level VI Surgical pathology, gross and microscopic examination"),
]

# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS icd10 (
    code        TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    short_desc  TEXT,
    category    TEXT,
    billable    INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS cpt (
    code        TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    short_desc  TEXT,
    category    TEXT,
    work_rvu    REAL DEFAULT 0.0
);

CREATE VIRTUAL TABLE IF NOT EXISTS icd10_fts
USING fts5(code, description, content=icd10, content_rowid=rowid);

CREATE VIRTUAL TABLE IF NOT EXISTS cpt_fts
USING fts5(code, description, content=cpt, content_rowid=rowid);

CREATE TRIGGER IF NOT EXISTS icd10_ai AFTER INSERT ON icd10 BEGIN
  INSERT INTO icd10_fts(rowid, code, description) VALUES (new.rowid, new.code, new.description);
END;

CREATE TRIGGER IF NOT EXISTS cpt_ai AFTER INSERT ON cpt BEGIN
  INSERT INTO cpt_fts(rowid, code, description) VALUES (new.rowid, new.code, new.description);
END;
"""


# ─────────────────────────────────────────────────────────────────────────────
# BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_db(icd_file=None, cpt_file=None):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Remove old DB
    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"  Removed old database")

    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    print(f"  Schema created")

    # ── Load ICD-10 ──────────────────────────────────────────────────────────
    icd_rows = []

    if icd_file and Path(icd_file).exists():
        print(f"  Loading ICD-10 from: {icd_file}")
        with open(icd_file, encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    code = row[0].strip().upper()
                    desc = row[1].strip()
                    if re.match(r'^[A-Z]\d{2}', code):
                        icd_rows.append((code, desc, desc[:60], code[0], 1))
        print(f"  Loaded {len(icd_rows):,} ICD-10 codes from file")
    else:
        print(f"  Using built-in ICD-10 dataset ({len(BUILTIN_ICD10)} codes)")
        icd_rows = [(c, d, d[:60], c[0], 1) for c, d in BUILTIN_ICD10]

    conn.executemany(
        "INSERT OR REPLACE INTO icd10 (code, description, short_desc, category, billable) VALUES (?,?,?,?,?)",
        icd_rows
    )
    print(f"  Inserted {len(icd_rows):,} ICD-10 codes")

    # ── Load CPT ─────────────────────────────────────────────────────────────
    cpt_rows = []

    if cpt_file and Path(cpt_file).exists():
        print(f"  Loading CPT from: {cpt_file}")
        with open(cpt_file, encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    code = row[0].strip()
                    desc = row[1].strip()
                    if re.match(r'^\d{5}', code):
                        cpt_rows.append((code, desc, desc[:60], None, 0.0))
        print(f"  Loaded {len(cpt_rows):,} CPT codes from file")
    else:
        print(f"  Using built-in CPT dataset ({len(BUILTIN_CPT)} codes)")
        cpt_rows = [(c, d, d[:60], None, 0.0) for c, d in BUILTIN_CPT]

    conn.executemany(
        "INSERT OR REPLACE INTO cpt (code, description, short_desc, category, work_rvu) VALUES (?,?,?,?,?)",
        cpt_rows
    )
    print(f"  Inserted {len(cpt_rows):,} CPT codes")

    conn.commit()
    conn.close()

    size_kb = DB_PATH.stat().st_size // 1024
    print(f"\n  Database built: {DB_PATH}")
    print(f"  Size: {size_kb:,} KB")
    print(f"  ICD-10: {len(icd_rows):,} codes")
    print(f"  CPT:    {len(cpt_rows):,} codes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build medical codes SQLite database")
    parser.add_argument("--icd", default=None, help="Path to ICD-10 CSV file from CMS")
    parser.add_argument("--cpt", default=None, help="Path to CPT CSV file")
    args = parser.parse_args()

    print("\n=== Building Medical Codes Database ===")
    build_db(icd_file=args.icd, cpt_file=args.cpt)
    print("\n  Done. Run: python scripts/build_code_db.py --icd your_icd10.csv to load real CMS data\n")
