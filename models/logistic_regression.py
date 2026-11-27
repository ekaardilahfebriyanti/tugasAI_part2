import numpy as np
import pickle

class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.classes = None
        self.cost_history = []

    def softmax(self, z):
        # fungsi aktivasi Softmax yang dapat mengubah angka skor mentah hasil hitungan model menjadi nilai peluang (probabilitas) untuk setiap kelas (0, 1, dan 2).
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def cost(self, h, y_one_hot):
        # menghitung Categorical Cross-Entropy Loss
        m = y_one_hot.shape[0]
        # mencegah log(0)
        eps = 1e-15  
        return -(1 / m) * np.sum(y_one_hot * np.log(np.clip(h, eps, 1 - eps)))

    def fit(self, X, y):
        # training menggunakan gradient descent (Multinomial / Softmax)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes = np.unique(y)
        n_classes = len(self.classes)
        m, n = X.shape

        # One-hot encoding untuk label target (misal: 0, 1, 2)
        y_indices = np.searchsorted(self.classes, y)
        y_one_hot = np.eye(n_classes)[y_indices]

        self.weights = np.zeros((n, n_classes))
        self.bias = np.zeros(n_classes)
        self.cost_history = []

        for _ in range(self.iterations):
            z = np.dot(X, self.weights) + self.bias
            h = self.softmax(z)

            dw = (1 / m) * np.dot(X.T, (h - y_one_hot))
            db = (1 / m) * np.sum(h - y_one_hot, axis=0)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            self.cost_history.append(self.cost(h, y_one_hot))

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        z = np.dot(X, self.weights) + self.bias
        return self.softmax(z)

    def predict(self, X):
        # mengembalikan label kelas dengan probabilitas tertinggi (0, 1, atau 2)
        probs = self.predict_proba(X)
        class_indices = np.argmax(probs, axis=1)
        return self.classes[class_indices]

    def save_model(self, filepath):
        # menyimpan bobot (weights), bias, dan kelas ke file .pkl
        with open(filepath, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load_model(self, filepath):
        # memuat parameter model dari file .pkl ke dalam objek saat ini
        with open(filepath, 'rb') as f:
            self.__dict__.update(pickle.load(f))