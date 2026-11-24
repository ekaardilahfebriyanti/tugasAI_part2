import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3, metric="euclidean"):
        self.k = k
        self.metric = metric.lower()

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)

    def _get_distances(self, x_single):
        # menghitung jarak antara 1 baris data uji (x_single) dengan seluruh baris data latih (self.X_train) secara bersamaan.
        diff = self.X_train - x_single
        
        if self.metric == "euclidean":
            return np.sqrt(np.sum(diff ** 2, axis=1))
        elif self.metric == "manhattan":
            return np.sum(np.abs(diff), axis=1)
        else:
            raise ValueError("Metric harus 'euclidean' atau 'manhattan'")

    def _get_single_prediction(self, x_datatest):
        distances = self._get_distances(x_datatest)
        k_idx = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_idx]
        return Counter(k_labels).most_common(1)[0][0]

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._get_single_prediction(x) for x in X])