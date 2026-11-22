"""Konfigurasi bersama pipeline preprocessing dan pemodelan.

Berkas ini menjadi satu-satunya sumber nilai konstanta (seed, pemetaan label,
pengelompokan kolom, dan hiperparameter preprocessing) agar seluruh anggota
dan seluruh notebook memakai definisi yang identik.
"""
from pathlib import Path

# ----------------------------------------------------------------------
# Umum
# ----------------------------------------------------------------------
SEED = 42
VAL_SIZE = 0.20
TARGET = "Target"
ID_COL = "Student_ID"

# Pemetaan label tunggal untuk seluruh model.
LABEL_MAP = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}
INV_LABEL = {v: k for k, v in LABEL_MAP.items()}

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR_CANDIDATES = ("data", "src")
ARTIFACT_DIR = REPO_ROOT / "artifacts"
FIGURE_DIR = REPO_ROOT / "doc" / "figures"

# ----------------------------------------------------------------------
# Pengelompokan kolom fitur (36 kolom setelah Student_ID dan Target dikeluarkan)
# ----------------------------------------------------------------------
CU = "Curricular units"

NOMINAL_COLS = [
    "Marital status", "Application mode", "Course", "Previous qualification",
    "Nationality", "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation",
]
BINARY_COLS = [
    "Daytime/evening attendance", "Displaced", "Educational special needs",
    "Debtor", "Tuition fees up to date", "Gender", "Scholarship holder", "International",
]
ORDINAL_COLS = ["Application order"]
NUMERIC_COLS = [
    "Previous qualification (grade)", "Admission grade", "Age at enrollment",
    f"{CU} 1st sem (credited)", f"{CU} 1st sem (enrolled)", f"{CU} 1st sem (evaluations)",
    f"{CU} 1st sem (approved)", f"{CU} 1st sem (grade)", f"{CU} 1st sem (without evaluations)",
    f"{CU} 2nd sem (credited)", f"{CU} 2nd sem (enrolled)", f"{CU} 2nd sem (evaluations)",
    f"{CU} 2nd sem (approved)", f"{CU} 2nd sem (grade)", f"{CU} 2nd sem (without evaluations)",
    "Unemployment rate", "Inflation rate", "GDP",
]
FEATURE_COLS = NOMINAL_COLS + BINARY_COLS + ORDINAL_COLS + NUMERIC_COLS

# ----------------------------------------------------------------------
# Penanganan outlier
# ----------------------------------------------------------------------
# Kolom dengan IQR positif dan tanpa nol struktural: batas dihitung dari seluruh nilai.
CLIP_STANDARD = [
    "Previous qualification (grade)", "Admission grade", "Age at enrollment",
    f"{CU} 1st sem (enrolled)", f"{CU} 2nd sem (enrolled)",
    f"{CU} 1st sem (evaluations)", f"{CU} 2nd sem (evaluations)",
    f"{CU} 1st sem (approved)", f"{CU} 2nd sem (approved)",
]
# Kolom nilai rata-rata: nol bersifat struktural (tidak ada mata kuliah lulus),
# sehingga batas dihitung dari nilai tidak nol dan nilai nol dibiarkan.
CLIP_ZERO_AWARE = [f"{CU} 1st sem (grade)", f"{CU} 2nd sem (grade)"]
# Kolom lain tidak dipotong (IQR bernilai nol atau tidak ada outlier).
CLIP_SKIPPED = [c for c in NUMERIC_COLS if c not in CLIP_STANDARD + CLIP_ZERO_AWARE]
OUTLIER_IQR_MULTIPLIER = 3.0

# ----------------------------------------------------------------------
# Seleksi fitur
# ----------------------------------------------------------------------
FEATURE_SELECTION_METHOD = "mi"
N_FEATURES = 20
MIN_SUPPORT = 0.01
CORR_THRESHOLD = 0.95

# ----------------------------------------------------------------------
# Validasi silang
# ----------------------------------------------------------------------
CV_FOLDS = 5
