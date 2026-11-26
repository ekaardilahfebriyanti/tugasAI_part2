import pickle
import numpy as np


def macro_f1(y_true, y_pred, n_classes=3):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    scores = []
    for c in range(n_classes):
        true_positive = np.sum((y_true == c) & (y_pred == c))
        false_positive = np.sum((y_true != c) & (y_pred == c))
        false_negative = np.sum((y_true == c) & (y_pred != c))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0)
    return float(np.mean(scores))


class Ensemble:
    def __init__(self, models, weights=None, voting="hard", n_classes=3):
        if voting != "hard":
            raise ValueError("voting yang tersedia saat ini hanya 'hard'")
        if len(models) == 0:
            raise ValueError("models tidak boleh kosong")
        self.models = dict(models)
        self.names = list(self.models.keys())
        self.voting = voting
        self.n_classes = n_classes
        if weights is None:
            self.weights = {name: 1.0 for name in self.names}
        else:
            self.weights = {name: float(weights[name]) for name in self.names}

    def _inputs(self, X):
        if isinstance(X, dict):
            return {name: np.asarray(X[name], dtype=float) for name in self.names}
        shared = np.asarray(X, dtype=float)
        return {name: shared for name in self.names}

    def member_predictions(self, X):
        data = self._inputs(X)
        return {name: np.asarray(self.models[name].predict(data[name])).astype(int) for name in self.names}

    def set_weights_from_validation(self, X_val, y_val):
        predictions = self.member_predictions(X_val)
        self.weights = {name: macro_f1(y_val, predictions[name], self.n_classes) for name in self.names}
        return dict(self.weights)

    def predict(self, X):
        predictions = self.member_predictions(X)
        n_samples = len(predictions[self.names[0]])
        scores = np.zeros((n_samples, self.n_classes))
        for rank, name in enumerate(self.names):
            vote_weight = self.weights[name] + 1e-9 * (len(self.names) - rank)
            scores[np.arange(n_samples), predictions[name]] += vote_weight
        return np.argmax(scores, axis=1)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)
        with open(path, "rb") as f:
            obj.__dict__.update(pickle.load(f))
        return obj
