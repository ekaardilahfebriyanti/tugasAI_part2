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


def member_proba(model, X, n_classes=3):
    X = np.asarray(X, dtype=float)
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X), dtype=float)
        total = proba.sum(axis=1, keepdims=True)
        total[total == 0] = 1.0
        return proba / total
    if hasattr(model, "trees"):
        votes = np.array([tree.predict(X) for tree in model.trees]).astype(int)
        counts = np.stack([(votes == c).sum(axis=0) for c in range(n_classes)], axis=1)
        return counts / counts.sum(axis=1, keepdims=True)
    if hasattr(model, "X_train") and hasattr(model, "y_train") and hasattr(model, "k"):
        proba = np.zeros((len(X), n_classes))
        labels = np.asarray(model.y_train).astype(int)
        for i, x in enumerate(X):
            nearest = np.argsort(model._get_distances(x))[:model.k]
            proba[i] = np.bincount(labels[nearest], minlength=n_classes) / len(nearest)
        return proba
    predicted = np.asarray(model.predict(X)).astype(int)
    proba = np.zeros((len(X), n_classes))
    proba[np.arange(len(X)), predicted] = 1.0
    return proba


class Ensemble:
    def __init__(self, models, weights=None, voting="hard", n_classes=3):
        if voting not in ("hard", "soft"):
            raise ValueError("voting harus 'hard' atau 'soft'")
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

    def member_probabilities(self, X):
        data = self._inputs(X)
        return {name: member_proba(self.models[name], data[name], self.n_classes) for name in self.names}

    def set_weights_from_validation(self, X_val, y_val):
        predictions = self.member_predictions(X_val)
        self.weights = {name: macro_f1(y_val, predictions[name], self.n_classes) for name in self.names}
        return dict(self.weights)

    def predict_proba(self, X):
        probabilities = self.member_probabilities(X)
        total_weight = sum(self.weights[name] for name in self.names)
        combined = sum(self.weights[name] * probabilities[name] for name in self.names)
        return combined / total_weight

    def predict(self, X):
        if self.voting == "soft":
            return np.argmax(self.predict_proba(X), axis=1)
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
