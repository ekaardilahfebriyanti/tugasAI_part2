import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
TARGET = "Target"
ID_COL = "Student_ID"
CLASS_NAMES = ["Dropout", "Enrolled", "Graduate"]
LABEL_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}
CU = "Curricular units"

NOMINAL_COLS = [
    "Marital status", "Application mode", "Course", "Previous qualification",
    "Nationality", "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation",
]
NUMERIC_COLS = [
    "Previous qualification (grade)", "Admission grade", "Age at enrollment",
    f"{CU} 1st sem (credited)", f"{CU} 1st sem (enrolled)",
    f"{CU} 1st sem (evaluations)", f"{CU} 1st sem (approved)",
    f"{CU} 1st sem (grade)", f"{CU} 1st sem (without evaluations)",
    f"{CU} 2nd sem (credited)", f"{CU} 2nd sem (enrolled)",
    f"{CU} 2nd sem (evaluations)", f"{CU} 2nd sem (approved)",
    f"{CU} 2nd sem (grade)", f"{CU} 2nd sem (without evaluations)",
    "Unemployment rate", "Inflation rate", "GDP", "Application order",
]
BINARY_COLS = [
    "Daytime/evening attendance", "Displaced", "Educational special needs",
    "Debtor", "Tuition fees up to date", "Gender", "Scholarship holder", "International",
]
CLIP_STANDARD = [
    "Previous qualification (grade)", "Admission grade", "Age at enrollment",
    f"{CU} 1st sem (enrolled)", f"{CU} 2nd sem (enrolled)",
    f"{CU} 1st sem (evaluations)", f"{CU} 2nd sem (evaluations)",
    f"{CU} 1st sem (approved)", f"{CU} 2nd sem (approved)",
]
CLIP_ZERO_AWARE = [f"{CU} 1st sem (grade)", f"{CU} 2nd sem (grade)"]
PLACEHOLDERS = {"?", "-", "na", "n/a", "nan", "null", "none", ""}


def clean_column_names(df):
    df = df.copy()
    df.columns = (
        df.columns.str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.replace("Nacionality", "Nationality")
    )
    return df


def clean_values(df):
    df = df.copy()
    flagged = {}
    for col in df.columns:
        if col in (TARGET, ID_COL):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            text = df[col].astype("string").str.strip()
            text = text.mask(text.str.lower().isin(PLACEHOLDERS))
            df[col] = pd.to_numeric(text, errors="coerce")

    def flag(rule, mask):
        mask = mask.fillna(False)
        flagged[rule] = int(mask.sum())
        return mask

    for col in ["Previous qualification (grade)", "Admission grade"]:
        mask = flag(f"{col} di luar 0-200", (df[col] < 0) | (df[col] > 200))
        df.loc[mask, col] = np.nan
    for sem in ["1st sem", "2nd sem"]:
        grade_col = f"{CU} {sem} (grade)"
        mask = flag(f"{grade_col} di luar 0-20", (df[grade_col] < 0) | (df[grade_col] > 20))
        df.loc[mask, grade_col] = np.nan
        approved = f"{CU} {sem} (approved)"
        limit = df[f"{CU} {sem} (enrolled)"] + df[f"{CU} {sem} (credited)"]
        mask = flag(f"{approved} melebihi enrolled + credited", df[approved] > limit)
        df.loc[mask, approved] = np.nan
    count_cols = [c for c in NUMERIC_COLS if c.startswith(CU) and "(grade)" not in c]
    for col in count_cols:
        mask = flag(f"{col} negatif", df[col] < 0)
        df.loc[mask, col] = np.nan
    for col in BINARY_COLS:
        mask = flag(f"{col} di luar 0/1", ~df[col].isin([0, 1]) & df[col].notna())
        df.loc[mask, col] = np.nan
    mask = flag("Age at enrollment di luar 17-70", (df["Age at enrollment"] < 17) | (df["Age at enrollment"] > 70))
    df.loc[mask, "Age at enrollment"] = np.nan
    mask = flag("Application order di luar 0-9", (df["Application order"] < 0) | (df["Application order"] > 9))
    df.loc[mask, "Application order"] = np.nan
    return df, flagged


class TrainImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.values_ = {}
        for col in X.columns:
            if col in NUMERIC_COLS:
                self.values_[col] = X[col].median()
            else:
                self.values_[col] = X[col].mode().iloc[0]
        return self

    def transform(self, X):
        return X.fillna(self.values_)

    def count_missing(self, X):
        return int(X.isna().sum().sum())


class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, standard_cols=None, zero_aware_cols=None, k=3.0):
        self.standard_cols = standard_cols
        self.zero_aware_cols = zero_aware_cols
        self.k = k

    def fit(self, X, y=None):
        self.standard_cols_ = list(CLIP_STANDARD if self.standard_cols is None else self.standard_cols)
        self.zero_aware_cols_ = list(CLIP_ZERO_AWARE if self.zero_aware_cols is None else self.zero_aware_cols)
        self.bounds_ = {}
        for col in self.standard_cols_:
            q1, q3 = X[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            self.bounds_[col] = (q1 - self.k * iqr, q3 + self.k * iqr)
        for col in self.zero_aware_cols_:
            nonzero = X.loc[X[col] != 0, col]
            q1, q3 = nonzero.quantile([0.25, 0.75])
            iqr = q3 - q1
            self.bounds_[col] = (q1 - self.k * iqr, q3 + self.k * iqr)
        return self

    def transform(self, X):
        X = X.copy()
        for col, (low, high) in self.bounds_.items():
            if col in self.zero_aware_cols_:
                mask = X[col] != 0
                X.loc[mask, col] = X.loc[mask, col].clip(low, high)
            else:
                X[col] = X[col].clip(low, high)
        return X

    def count_clipped(self, X):
        total = 0
        for col, (low, high) in self.bounds_.items():
            values = X[col]
            if col in self.zero_aware_cols_:
                values = values[values != 0]
            total += int(((values < low) | (values > high)).sum())
        return total


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_features=20, min_support=0.01, mi_min=0.0, corr_threshold=0.95, random_state=SEED):
        self.n_features = n_features
        self.min_support = min_support
        self.mi_min = mi_min
        self.corr_threshold = corr_threshold
        self.random_state = random_state

    def fit(self, X, y):
        columns = list(X.columns)
        is_discrete = (X.nunique() <= 2).values
        support = (X != 0).mean()
        rare = [c for c, d in zip(columns, is_discrete) if d and support[c] < self.min_support]
        base = [c for c in columns if c not in rare]
        X_base = X[base]
        discrete_base = (X_base.nunique() <= 2).values
        scores = mutual_info_classif(X_base, y, discrete_features=discrete_base, random_state=self.random_state)
        self.mi_ = pd.Series(scores, index=base).sort_values(ascending=False)
        after_mi = list(self.mi_[self.mi_ > self.mi_min].index)
        corr = X_base[after_mi].corr().abs()
        after_redundancy = []
        for col in after_mi:
            if all(corr.loc[col, kept] < self.corr_threshold for kept in after_redundancy):
                after_redundancy.append(col)
        if self.n_features is None or self.n_features >= len(after_redundancy):
            final = after_redundancy
        else:
            final = after_redundancy[:self.n_features]
        self.stage_counts_ = {
            "awal": len(columns),
            "setelah dukungan minimum": len(base),
            "setelah filter MI": len(after_mi),
            "setelah redundansi": len(after_redundancy),
            "akhir": len(final),
        }
        self.selected_ = final
        return self

    def transform(self, X):
        return X[self.selected_]


def oversample_indices(y, seed=SEED):
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    counts = pd.Series(y).value_counts()
    parts = [np.arange(len(y))]
    for label, n in counts.items():
        if n < counts.max():
            pool = np.where(y == label)[0]
            parts.append(rng.choice(pool, size=counts.max() - n, replace=True))
    return rng.permutation(np.concatenate(parts))


class Dataset:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def build_dataset(data_dir="datasets", n_features=20, oversample=True, val_size=0.20, clip_k=3.0, seed=SEED):
    train_df = clean_column_names(pd.read_csv(os.path.join(data_dir, "train.csv")))
    test_df = clean_column_names(pd.read_csv(os.path.join(data_dir, "test.csv")))
    report = {"baris train awal": len(train_df), "baris test": len(test_df)}

    feature_cols = [c for c in train_df.columns if c not in (TARGET, ID_COL)]
    before = len(train_df)
    train_df = train_df.drop_duplicates(subset=feature_cols + [TARGET]).reset_index(drop=True)
    report["duplikat dibuang (train)"] = before - len(train_df)

    y_all = train_df[TARGET].map(LABEL_MAP).astype(int).values
    X_all, flagged_train = clean_values(train_df[feature_cols])
    test_ids = test_df[ID_COL].values
    X_test_raw, flagged_test = clean_values(test_df[feature_cols])
    report["nilai tidak valid dijadikan kosong (train)"] = {k: v for k, v in flagged_train.items() if v}
    report["nilai tidak valid dijadikan kosong (test)"] = {k: v for k, v in flagged_test.items() if v}

    train_idx, val_idx = train_test_split(np.arange(len(y_all)), test_size=val_size, stratify=y_all, random_state=seed)
    X_train_raw = X_all.iloc[train_idx].reset_index(drop=True)
    X_val_raw = X_all.iloc[val_idx].reset_index(drop=True)
    y_train = y_all[train_idx]
    y_val = y_all[val_idx]

    imputer = TrainImputer().fit(X_train_raw)
    report["sel kosong diisi (train, val, test)"] = (
        imputer.count_missing(X_train_raw), imputer.count_missing(X_val_raw), imputer.count_missing(X_test_raw)
    )
    X_train_raw = imputer.transform(X_train_raw)
    X_val_raw = imputer.transform(X_val_raw)
    X_test_raw = imputer.transform(X_test_raw)

    clipper = OutlierClipper(k=clip_k).fit(X_train_raw)
    report["sel terpotong oleh clipping (train)"] = clipper.count_clipped(X_train_raw)
    X_train_raw = clipper.transform(X_train_raw)
    X_val_raw = clipper.transform(X_val_raw)
    X_test_raw = clipper.transform(X_test_raw)

    encoder = ColumnTransformer([
        ("nominal", OneHotEncoder(handle_unknown="ignore", sparse_output=False), NOMINAL_COLS),
        ("numeric", StandardScaler(), NUMERIC_COLS),
        ("binary", "passthrough", BINARY_COLS),
    ])
    encoder.set_output(transform="pandas")
    encoder.fit(X_train_raw)
    frames = []
    for raw in (X_train_raw, X_val_raw, X_test_raw):
        frame = encoder.transform(raw)
        frame.columns = [name.split("__", 1)[1] for name in frame.columns]
        frames.append(frame)
    X_train_enc, X_val_enc, X_test_enc = frames
    report["kolom setelah encoding"] = X_train_enc.shape[1]

    if n_features is None:
        feature_names = list(X_train_enc.columns)
        report["seleksi fitur"] = "tidak dipakai"
        mi_scores = None
    else:
        selector = FeatureSelector(n_features=n_features).fit(X_train_enc, y_train)
        feature_names = list(selector.selected_)
        report["seleksi fitur"] = selector.stage_counts_
        mi_scores = selector.mi_

    X_train = X_train_enc[feature_names].values
    X_val = X_val_enc[feature_names].values
    X_test = X_test_enc[feature_names].values
    report["distribusi kelas train sebelum oversampling"] = np.bincount(y_train).tolist()

    if oversample:
        resample = oversample_indices(y_train, seed)
        X_train = X_train[resample]
        y_train = y_train[resample]
    report["distribusi kelas train akhir"] = np.bincount(y_train).tolist()
    report["distribusi kelas validasi"] = np.bincount(y_val).tolist()
    report["ukuran akhir (train, val, test)"] = (X_train.shape, X_val.shape, X_test.shape)

    return Dataset(
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test,
        test_ids=test_ids, feature_names=feature_names, class_names=CLASS_NAMES,
        mi_scores=mi_scores, report=report,
    )


def write_submission(test_ids, predictions, path):
    labels = [CLASS_NAMES[int(p)] for p in predictions]
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    pd.DataFrame({ID_COL: test_ids, TARGET: labels}).to_csv(path, index=False)
