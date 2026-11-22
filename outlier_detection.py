import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

CU = "Curricular units"

CLIP_STANDARD = [
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    f"{CU} 1st sem (enrolled)",
    f"{CU} 2nd sem (enrolled)",
    f"{CU} 1st sem (evaluations)",
    f"{CU} 2nd sem (evaluations)",
    f"{CU} 1st sem (approved)",
    f"{CU} 2nd sem (approved)",
]

CLIP_ZERO_AWARE = [
    f"{CU} 1st sem (grade)",
    f"{CU} 2nd sem (grade)",
]


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
        counts = {}
        for col, (low, high) in self.bounds_.items():
            values = X[col]
            if col in self.zero_aware_cols_:
                values = values[values != 0]
            counts[col] = int(((values < low) | (values > high)).sum())
        return pd.Series(counts)
