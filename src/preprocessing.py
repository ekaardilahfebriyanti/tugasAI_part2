"""Komponen preprocessing: pembersihan, transformasi, seleksi fitur, dan resampling.

Seluruh transformer mengikuti antarmuka scikit-learn (fit dan transform) dan
statistiknya dipelajari hanya dari data train.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as cfg


# ----------------------------------------------------------------------
# Pemuatan dan pembersihan data
# ----------------------------------------------------------------------
def clean_columns(df):
    """Merapikan nama kolom: spasi dan tab berlebih dihapus, 'Nacionality' menjadi 'Nationality'."""
    df = df.copy()
    df.columns = (df.columns.str.replace(r"\s+", " ", regex=True)
                  .str.strip().str.replace("Nacionality", "Nationality"))
    return df


def find_data_dir(root=None):
    """Mengembalikan folder yang memuat train.csv (diperiksa pada data/ lalu src/)."""
    root = Path(root) if root is not None else cfg.REPO_ROOT
    for name in cfg.DATA_DIR_CANDIDATES:
        if (root / name / "train.csv").exists():
            return root / name
    raise FileNotFoundError("train.csv tidak ditemukan pada folder data/ maupun src/.")


def load_datasets(data_dir):
    """Memuat train, test, dan sample_submission dengan nama kolom yang sudah dirapikan."""
    data_dir = Path(data_dir)
    train = clean_columns(pd.read_csv(data_dir / "train.csv"))
    test = clean_columns(pd.read_csv(data_dir / "test.csv"))
    sub = pd.read_csv(data_dir / "sample_submission.csv")
    return train, test, sub


def audit_data(df):
    """Audit kualitas data berbasis aturan domain; setiap nilai adalah jumlah temuan."""
    u = cfg.CU
    unit_cols = [c for c in df.columns if c.startswith(u)]
    return pd.Series({
        "Missing value": int(df.isna().sum().sum()),
        "Baris duplikat (tanpa ID)": int(df.drop(columns=cfg.ID_COL).duplicated().sum()),
        "Approved > enrolled + credited (sem 1)": int((df[f"{u} 1st sem (approved)"] > df[f"{u} 1st sem (enrolled)"] + df[f"{u} 1st sem (credited)"]).sum()),
        "Approved > enrolled + credited (sem 2)": int((df[f"{u} 2nd sem (approved)"] > df[f"{u} 2nd sem (enrolled)"] + df[f"{u} 2nd sem (credited)"]).sum()),
        "Nilai negatif pada kolom hitungan": int((df[unit_cols] < 0).sum().sum()),
        "Nilai biner di luar {0, 1}": int((~df[cfg.BINARY_COLS].isin([0, 1])).sum().sum()),
        "Usia di luar 17-70": int(((df["Age at enrollment"] < 17) | (df["Age at enrollment"] > 70)).sum()),
    })


# ----------------------------------------------------------------------
# Penanganan outlier
# ----------------------------------------------------------------------
class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clipping berbasis IQR dengan batas yang dipelajari hanya dari data train.

    standard_cols   : batas dihitung dari seluruh nilai kolom.
    zero_aware_cols : batas dihitung dari nilai tidak nol saja dan nilai nol dibiarkan,
                      karena nol pada kolom ini bersifat struktural.
    k               : pengali IQR (np.inf berarti tanpa clipping).
    """

    def __init__(self, standard_cols=None, zero_aware_cols=None, k=3.0):
        self.standard_cols = standard_cols
        self.zero_aware_cols = zero_aware_cols
        self.k = k

    def fit(self, X, y=None):
        self.bounds_ = {}
        for c in (self.standard_cols or []):
            q1, q3 = X[c].quantile([0.25, 0.75])
            self.bounds_[c] = (q1 - self.k * (q3 - q1), q3 + self.k * (q3 - q1))
        for c in (self.zero_aware_cols or []):
            nz = X.loc[X[c] != 0, c]
            q1, q3 = nz.quantile([0.25, 0.75])
            self.bounds_[c] = (q1 - self.k * (q3 - q1), q3 + self.k * (q3 - q1))
        return self

    def transform(self, X):
        X = X.copy()
        for c, (lo, hi) in self.bounds_.items():
            if c in (self.zero_aware_cols or []):
                m = X[c] != 0
                X.loc[m, c] = X.loc[m, c].clip(lo, hi)
            else:
                X[c] = X[c].clip(lo, hi)
        return X

    def count_clipped(self, X):
        """Jumlah sel yang terpotong per kolom."""
        out = {}
        for c, (lo, hi) in self.bounds_.items():
            s = X[c]
            if c in (self.zero_aware_cols or []):
                s = s[s != 0]
            out[c] = int(((s < lo) | (s > hi)).sum())
        return pd.Series(out)


# ----------------------------------------------------------------------
# Encoding dan scaling
# ----------------------------------------------------------------------
def make_encoder_scaler():
    """One-hot untuk kolom nominal, standardisasi untuk kolom numerik dan ordinal,
    kolom biner diteruskan tanpa perubahan. Keluaran berupa DataFrame."""
    return ColumnTransformer([
        ("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cfg.NOMINAL_COLS),
        ("num", StandardScaler(), cfg.NUMERIC_COLS + cfg.ORDINAL_COLS),
        ("bin", "passthrough", cfg.BINARY_COLS),
    ]).set_output(transform="pandas")


# ----------------------------------------------------------------------
# Seleksi fitur
# ----------------------------------------------------------------------
class HybridFeatureSelector(BaseEstimator, TransformerMixin):
    """Seleksi fitur bertahap, seluruhnya dipelajari hanya dari data train.

    Tahap 0 (dukungan)   : membuang fitur biner/dummy yang aktif pada kurang dari
                           min_support baris (kategori terlalu langka untuk digeneralisasi).
    Tahap A (filter)     : membuang fitur dengan mutual information (MI) <= mi_min.
    Tahap B (redundansi) : berurutan menurut MI, membuang fitur yang |korelasi|-nya dengan
                           fitur yang sudah dipertahankan >= corr_threshold.
    Tahap C (pemangkasan) hingga n_features fitur:
        method="mi"  : mengambil n fitur dengan MI tertinggi.
        method="rfe" : RFE dengan Logistic Regression pada fitur yang distandarisasi seragam.
    n_features=None berarti seleksi berhenti setelah tahap B.
    """

    def __init__(self, n_features=cfg.N_FEATURES, method=cfg.FEATURE_SELECTION_METHOD,
                 min_support=cfg.MIN_SUPPORT, mi_min=0.0,
                 corr_threshold=cfg.CORR_THRESHOLD, random_state=cfg.SEED):
        self.n_features = n_features
        self.method = method
        self.min_support = min_support
        self.mi_min = mi_min
        self.corr_threshold = corr_threshold
        self.random_state = random_state

    def fit(self, X, y):
        cols = list(X.columns)
        discrete = (X.nunique() <= 2).values
        support = (X != 0).mean()
        rare = [c for c, d in zip(cols, discrete) if d and support[c] < self.min_support]
        base = [c for c in cols if c not in rare]
        Xb = X[base]
        discrete_b = (Xb.nunique() <= 2).values
        mi = pd.Series(mutual_info_classif(Xb, y, discrete_features=discrete_b,
                                           random_state=self.random_state), index=base)
        self.mi_ = mi.sort_values(ascending=False)
        keep_a = list(self.mi_[self.mi_ > self.mi_min].index)
        corr = Xb[keep_a].corr().abs()
        keep_b = []
        for c in keep_a:
            if all(corr.loc[c, k] < self.corr_threshold for k in keep_b):
                keep_b.append(c)

        if self.n_features is None or self.n_features >= len(keep_b):
            final = keep_b
        elif self.method == "mi":
            final = keep_b[:self.n_features]
        else:
            Z = StandardScaler().fit_transform(Xb[keep_b])
            rfe = RFE(LogisticRegression(max_iter=3000, class_weight="balanced",
                                         random_state=self.random_state),
                      n_features_to_select=self.n_features, step=0.1)
            rfe.fit(Z, y)
            final = [c for c, s in zip(keep_b, rfe.support_) if s]

        self.stage_counts_ = {"awal": len(cols), "setelah 0 (dukungan)": len(base),
                              "setelah A (MI)": len(keep_a), "setelah B (redundansi)": len(keep_b),
                              "akhir (C)": len(final)}
        self.selected_ = final
        return self

    def transform(self, X):
        return X[self.selected_]


# ----------------------------------------------------------------------
# Resampling
# ----------------------------------------------------------------------
def random_oversample_indices(y, seed=cfg.SEED):
    """Indeks hasil random oversampling: setiap kelas minoritas diambil ulang (dengan
    pengembalian) hingga jumlahnya sama dengan kelas mayoritas. Dipakai pada data train saja."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    counts = pd.Series(y).value_counts()
    idx = [np.arange(len(y))]
    for cls, n in counts.items():
        if n < counts.max():
            pool = np.where(y == cls)[0]
            idx.append(rng.choice(pool, size=counts.max() - n, replace=True))
    return rng.permutation(np.concatenate(idx))


# ----------------------------------------------------------------------
# Pipeline lengkap
# ----------------------------------------------------------------------
@dataclass
class PreparedData:
    """Keluaran pipeline preprocessing. Seluruh transformer di-fit pada data train."""
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    id_train: pd.Series
    id_val: pd.Series
    id_test: pd.Series
    resample_idx: np.ndarray
    clipper: OutlierClipper
    encoder: ColumnTransformer
    selector: HybridFeatureSelector

    @property
    def feature_names(self):
        return list(self.X_train.columns)

    @property
    def X_train_resampled(self):
        return self.X_train.values[self.resample_idx]

    @property
    def y_train_resampled(self):
        return self.y_train.values[self.resample_idx]


def prepare_data(data_dir=None, iqr_multiplier=cfg.OUTLIER_IQR_MULTIPLIER,
                 n_features=cfg.N_FEATURES, method=cfg.FEATURE_SELECTION_METHOD):
    """Menjalankan seluruh pipeline: split stratified, clipping, encoding dan scaling,
    seleksi fitur, serta indeks oversampling untuk data train."""
    data_dir = find_data_dir() if data_dir is None else Path(data_dir)
    train_raw, test_raw, _ = load_datasets(data_dir)

    y_all = train_raw[cfg.TARGET].map(cfg.LABEL_MAP).astype(int).reset_index(drop=True)
    X_all = train_raw.drop(columns=[cfg.ID_COL, cfg.TARGET]).reset_index(drop=True)
    id_all = train_raw[cfg.ID_COL].reset_index(drop=True)
    X_test = test_raw.drop(columns=[cfg.ID_COL]).reset_index(drop=True)
    id_test = test_raw[cfg.ID_COL].reset_index(drop=True)

    idx_tr, idx_val = train_test_split(np.arange(len(X_all)), test_size=cfg.VAL_SIZE,
                                       stratify=y_all, random_state=cfg.SEED)
    X_tr, X_va = X_all.iloc[idx_tr].reset_index(drop=True), X_all.iloc[idx_val].reset_index(drop=True)
    y_tr, y_va = y_all.iloc[idx_tr].reset_index(drop=True), y_all.iloc[idx_val].reset_index(drop=True)

    clipper = OutlierClipper(cfg.CLIP_STANDARD, cfg.CLIP_ZERO_AWARE, k=iqr_multiplier).fit(X_tr)
    X_tr, X_va, X_te = clipper.transform(X_tr), clipper.transform(X_va), clipper.transform(X_test)

    encoder = make_encoder_scaler().fit(X_tr)
    X_tr, X_va, X_te = (encoder.transform(d) for d in (X_tr, X_va, X_te))
    for d in (X_tr, X_va, X_te):
        d.columns = [c.split("__", 1)[1] for c in d.columns]

    selector = HybridFeatureSelector(n_features=n_features, method=method).fit(X_tr, y_tr)
    X_tr, X_va, X_te = (selector.transform(d) for d in (X_tr, X_va, X_te))

    return PreparedData(
        X_train=X_tr, X_val=X_va, X_test=X_te, y_train=y_tr, y_val=y_va,
        id_train=id_all.iloc[idx_tr].reset_index(drop=True),
        id_val=id_all.iloc[idx_val].reset_index(drop=True), id_test=id_test,
        resample_idx=random_oversample_indices(y_tr.values),
        clipper=clipper, encoder=encoder, selector=selector)


def load_preprocessed(path):
    """Memuat berkas preprocessed_*.npz menjadi dictionary larik beserta data train
    hasil oversampling (X_train_res dan y_train_res)."""
    z = np.load(path, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    d["X_train_res"] = d["X_train"][d["resample_idx"]]
    d["y_train_res"] = d["y_train"][d["resample_idx"]]
    return d
