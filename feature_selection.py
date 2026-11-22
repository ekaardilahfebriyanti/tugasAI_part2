import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class HybridFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_features=20, method="mi", min_support=0.01,
                 mi_min=0.0, corr_threshold=0.95, random_state=42):
        self.n_features = n_features
        self.method = method
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
        scores = mutual_info_classif(X_base, y, discrete_features=discrete_base,
                                     random_state=self.random_state)
        self.mi_ = pd.Series(scores, index=base).sort_values(ascending=False)
        after_mi = list(self.mi_[self.mi_ > self.mi_min].index)

        corr = X_base[after_mi].corr().abs()
        after_redundancy = []
        for col in after_mi:
            if all(corr.loc[col, kept] < self.corr_threshold for kept in after_redundancy):
                after_redundancy.append(col)

        if self.n_features is None or self.n_features >= len(after_redundancy):
            final = after_redundancy
        elif self.method == "mi":
            final = after_redundancy[:self.n_features]
        else:
            Z = StandardScaler().fit_transform(X_base[after_redundancy])
            rfe = RFE(
                LogisticRegression(max_iter=3000, class_weight="balanced",
                                   random_state=self.random_state),
                n_features_to_select=self.n_features,
                step=0.1,
            )
            rfe.fit(Z, y)
            final = [c for c, keep in zip(after_redundancy, rfe.support_) if keep]

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
