# One-hot encoding untuk kolom nominal (kode = label, tanpa urutan).
import pandas as pd

ONEHOT_COLS = ["Marital status", 
               "Application mode", 
               "Course", 
               "Nacionality",
               "Mother's occupation", 
               "Father's occupation"
]


class OneHotEncoder:
    def __init__(self, columns=ONEHOT_COLS):
        self.columns = columns

    def fit(self, df):
        # daftar kategori dari TRAIN saja
        self.categories_ = {c: sorted(df[c].dropna().unique()) for c in self.columns}
        return self

    def transform(self, df):
        parts = [df.drop(columns=self.columns)]
        for c, cats in self.categories_.items():
            # kategori yang tidak ada di train -> semua kolom 0
            parts.append(pd.DataFrame({f"{c}_{v}": (df[c] == v).astype(float) for v in cats}, index=df.index))
        return pd.concat(parts, axis=1)

    def fit_transform(self, df):
        return self.fit(df).transform(df)

