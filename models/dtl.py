import numpy as np
from collections import Counter


class DecisionTree:
    # CART: tiap split biner. Numerik: "<= t", kategorikal: "== t". Ukuran impurity: gini.
    # Node dalam = (kolom, t, null_ke_kiri, kiri, kanan), daun = nomor kelas.
    def __init__(self, max_depth=None, min_samples_split=2, max_features=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features  # jumlah fitur yg dicoba tiap split (dipakai random forest)
        self.random_state = random_state

    def fit(self, X, y, categorical=()):
        # categorical = nomor kolom yg bertipe kategori, nilai null = NaN
        X = np.asarray(X, dtype=float)
        self.classes, y = np.unique(y, return_inverse=True)
        self.categorical = list(categorical)
        self.rng = np.random.default_rng(self.random_state)
        self.tree = self._build_tree(X, y, 0)

    def _get_gini(self, y):
        p = np.bincount(y, minlength=len(self.classes)) / len(y)
        return 1 - np.sum(p ** 2)

    def _get_best_split(self, X, y):
        n_fitur = X.shape[1]
        if self.max_features is None:
            kolom = range(n_fitur)
        else:
            kolom = self.rng.choice(n_fitur, min(self.max_features, n_fitur), replace=False)

        best = None
        best_gain = -1  # gain 0 tetap boleh (contoh XOR)
        for j in kolom:
            ada = ~np.isnan(X[:, j])  # null diabaikan saat mencari split
            if ada.sum() < 2:
                continue
            nilai, label = X[ada, j], y[ada]
            u = np.unique(nilai)
            if j in self.categorical:
                kandidat = u
            else:
                kandidat = (u[:-1] + u[1:]) / 2  # titik tengah antar nilai
                if len(kandidat) > 30:  # terlalu banyak -> ambil persentil supaya tidak lambat
                    kandidat = np.unique(np.percentile(nilai, np.linspace(0, 100, 31)))[:-1]

            for t in kandidat:
                kiri = (nilai == t) if j in self.categorical else (nilai <= t)
                if kiri.all() or not kiri.any():
                    continue
                w = kiri.mean()
                gain = self._get_gini(label) - w * self._get_gini(label[kiri]) - (1 - w) * self._get_gini(label[~kiri])
                gain = gain * ada.mean()  # kalau banyak null, gain dikecilkan
                if gain > best_gain:
                    best_gain = gain
                    best = (j, t, kiri.sum() >= (~kiri).sum())
        return best

    def _build_tree(self, X, y, depth):
        mayoritas = Counter(y).most_common(1)[0][0]
        if len(set(y)) == 1 or len(y) < self.min_samples_split:
            return mayoritas
        if self.max_depth is not None and depth >= self.max_depth:
            return mayoritas

        best = self._get_best_split(X, y)
        if best is None:
            return mayoritas

        j, t, null_kiri = best
        if j in self.categorical:
            kiri = X[:, j] == t
        else:
            kiri = X[:, j] <= t
        kiri = kiri | (np.isnan(X[:, j]) & null_kiri)  # null ikut cabang yg datanya lebih banyak

        return (j, t, null_kiri,
                self._build_tree(X[kiri], y[kiri], depth + 1),
                self._build_tree(X[~kiri], y[~kiri], depth + 1))

    def _get_single_prediction(self, x_datatest):
        node = self.tree
        while isinstance(node, tuple):
            j, t, null_kiri, kiri, kanan = node
            if np.isnan(x_datatest[j]):
                ke_kiri = null_kiri
            elif j in self.categorical:
                ke_kiri = x_datatest[j] == t
            else:
                ke_kiri = x_datatest[j] <= t
            node = kiri if ke_kiri else kanan
        return node

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.classes[np.array([self._get_single_prediction(x) for x in X])]
