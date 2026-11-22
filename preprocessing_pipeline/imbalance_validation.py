import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def stratified_split_indices(y, val_size=0.20, seed=42):
    y = np.asarray(y)
    train_idx, val_idx = train_test_split(
        np.arange(len(y)), test_size=val_size, stratify=y, random_state=seed
    )
    return train_idx, val_idx


def stratified_kfold_indices(y, n_splits=5, seed=42):
    y = np.asarray(y)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(folds.split(np.zeros(len(y)), y))


def random_oversample_indices(y, seed=42):
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    counts = pd.Series(y).value_counts()
    parts = [np.arange(len(y))]
    for label, n in counts.items():
        if n < counts.max():
            pool = np.where(y == label)[0]
            parts.append(rng.choice(pool, size=counts.max() - n, replace=True))
    return rng.permutation(np.concatenate(parts))


def class_distribution(y):
    counts = pd.Series(np.asarray(y)).value_counts().sort_index()
    return pd.DataFrame({"jumlah": counts, "persen": (counts / counts.sum() * 100).round(1)})
