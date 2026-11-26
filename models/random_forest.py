import numpy as np
import pickle
from collections import Counter
from models.dtl import DecisionTree


class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 max_features="sqrt", random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features  # "sqrt" = akar jumlah kolom, atau angka
        self.random_state = random_state

    def fit(self, X, y, categorical=()):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_fitur = int(np.sqrt(X.shape[1])) if self.max_features == "sqrt" else self.max_features
        rng = np.random.default_rng(self.random_state)
        self.trees = []
        for _ in range(self.n_estimators):
            idx = rng.integers(0, len(y), len(y))  # bootstrap: ambil n baris dengan pengembalian
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                max_features=n_fitur, random_state=int(rng.integers(1_000_000)))
            tree.fit(X[idx], y[idx], categorical)
            self.trees.append(tree)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        votes = np.array([tree.predict(X) for tree in self.trees])  # baris = pohon, kolom = data uji
        return np.array([Counter(votes[:, i]).most_common(1)[0][0] for i in range(len(X))])  # suara terbanyak

    def save_model_rf(self, filepath):
        # menyimpan seluruh atribut model ke file .pkl
        with open(filepath, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load_model_rf(self, filepath):
        # memuat atribut model dari file .pkl ke dalam objek saat ini
        with open(filepath, 'rb') as f:
            self.__dict__.update(pickle.load(f))
