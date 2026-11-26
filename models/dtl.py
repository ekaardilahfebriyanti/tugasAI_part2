import numpy as np
import pickle
from collections import Counter


class DecisionTree:
    # CART: split biner, numerik "<= t", kategorikal "== t", impurity gini.
    # node dalam = (kolom, t, null_kiri, kiri, kanan), daun = nomor kelas
    def __init__(self, max_depth=None, min_samples_split=2, max_features=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features  # jumlah kolom yg dicoba tiap split (dipakai random forest)
        self.random_state = random_state

    def gini(self, y):
        p = np.bincount(y) / len(y)
        return 1 - np.sum(p ** 2)

    def best_split(self, X, y):
        n = X.shape[1]
        if self.max_features is None:
            kolom = range(n)
        else:
            kolom = self.rng.choice(n, min(self.max_features, n), replace=False)

        best, best_gain = None, -1  # gain 0 tetap boleh (contoh XOR)
        for j in kolom:
            ada = ~np.isnan(X[:, j])  # nilai null diabaikan saat mencari split
            nilai, label = X[ada, j], y[ada]
            if len(nilai) < 2:
                continue
            if j in self.categorical:
                kandidat = np.unique(nilai)
            else:
                kandidat = np.unique(np.percentile(nilai, range(5, 100, 5)))  # maks 19 kandidat threshold

            for t in kandidat:
                kiri = (nilai == t) if j in self.categorical else (nilai <= t)
                if kiri.all() or not kiri.any():
                    continue
                w = kiri.mean()
                gain = self.gini(label) - w * self.gini(label[kiri]) - (1 - w) * self.gini(label[~kiri])
                if gain > best_gain:
                    best_gain, best = gain, (j, t)
        return best

    def build(self, X, y, depth):
        mayoritas = Counter(y).most_common(1)[0][0]
        if len(set(y)) == 1 or len(y) < self.min_samples_split:
            return mayoritas
        if self.max_depth is not None and depth >= self.max_depth:
            return mayoritas

        best = self.best_split(X, y)
        if best is None:
            return mayoritas

        j, t = best
        kosong = np.isnan(X[:, j])
        kiri = (X[:, j] == t) if j in self.categorical else (X[:, j] <= t)
        null_kiri = kiri.sum() >= (~kiri & ~kosong).sum()  # nilai null ikut cabang yg datanya lebih banyak
        kiri = kiri | (kosong & null_kiri)
        return (j, t, null_kiri, self.build(X[kiri], y[kiri], depth + 1), self.build(X[~kiri], y[~kiri], depth + 1))

    def fit(self, X, y, categorical=()):
        # categorical = nomor kolom yg bertipe kategori, nilai null = NaN
        X = np.asarray(X, dtype=float)
        self.classes, y = np.unique(y, return_inverse=True)
        self.categorical = list(categorical)
        self.rng = np.random.default_rng(self.random_state)
        self.tree = self.build(X, y, 0)

    def predict_one(self, x):
        node = self.tree
        while isinstance(node, tuple):
            j, t, null_kiri, kiri, kanan = node
            if np.isnan(x[j]):
                ke_kiri = null_kiri
            elif j in self.categorical:
                ke_kiri = x[j] == t
            else:
                ke_kiri = x[j] <= t
            node = kiri if ke_kiri else kanan
        return node

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.classes[np.array([self.predict_one(x) for x in X])]

    def save_model_dtl(self, filepath):
        # menyimpan atribut model ke file .pkl
        with open(filepath, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load_model_dtl(self, filepath):
        # memuat atribut model dari file .pkl ke dalam objek saat ini
        with open(filepath, 'rb') as f:
            self.__dict__.update(pickle.load(f))
