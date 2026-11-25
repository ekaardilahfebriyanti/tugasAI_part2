"""Normalisasi kolom numerik: z-score atau min-max"""
import pandas as pd

NUM_COLS = ["Previous qualification (grade)", 
            "Admission grade", 
            "Age at enrollment",
            "Application order", 
            "Unemployment rate", 
            "Inflation rate", 
            "GDP"
]


class Normalizer:
    def __init__(self, method="zscore", columns=None):
        assert method in ("zscore", "minmax")
        self.method, self.columns = method, columns

    def fit(self, df):
        if self.columns is None:   # + semua kolom 'Curricular units ...'
            self.columns = NUM_COLS + [c for c in df.columns if c.startswith("Curricular units")]
        x = df[self.columns]
        # statistik dari TRAIN saja; pembagi 0 diganti 1
        self.mean_, self.std_ = x.mean(), x.std(ddof=0).replace(0, 1)
        self.min_, self.rng_ = x.min(), (x.max() - x.min()).replace(0, 1)
        return self

    def transform(self, df):
        out = df.copy()
        x = df[self.columns]
        out[self.columns] = (x - self.mean_) / self.std_ if self.method == "zscore" else (x - self.min_) / self.rng_
        return out

    def fit_transform(self, df):
        return self.fit(df).transform(df)
