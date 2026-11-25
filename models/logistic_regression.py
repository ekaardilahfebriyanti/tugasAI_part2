import numpy as np
import pickle

class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = 0
        self.cost_history = []

    def sigmoid(self, z):
        #Sigmoid activation function
        return 1 / (1 + np.exp(-z))

    def cost(self, h, y):
        """Cross-entropy loss"""
        m = len(y)
        return - (1/m) * np.sum(y*np.log(h) + (1-y)*np.log(1-h))

    def fit(self, X, y):
        #training menggunakan gradient descent
        m, n = X.shape
        self.weights = np.zeros(n)

        for _ in range(self.iterations):
            z = np.dot(X, self.weights) + self.bias
            h = self.sigmoid(z)

            dw = (1/m) * np.dot(X.T, (h - y))
            db = (1/m) * np.sum(h - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            self.cost_history.append(self.cost(h, y))

    def predict(self, X):
        # membuat prediksi
        return (self.sigmoid(np.dot(X, self.weights) + self.bias) >= 0.5).astype(int)

    def save_model(self, filepath):
        # menyimpan bobot (weights), bias, dan kelas ke file .pkl
        with open(filepath, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load_model(self, filepath):
        # memuat parameter model dari file .pkl ke dalam objek saat ini
        with open(filepath, 'rb') as f:
            self.__dict__.update(pickle.load(f))