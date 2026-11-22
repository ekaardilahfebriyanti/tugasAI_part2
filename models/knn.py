import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from preprocessing_pipeline.preproccessing import Preproccessing
from collections import Counter

class KNN:
    def __init__(self, k=3):
        # define hyperparameter k
        self.k = k
    
    def fit(self, X, y):
        # store complete training data
        self.X_train = X
        self.y_train = y

    def _get_euclidean_distance(self, x1, x2):
        # calculate the euclidean distance for a row pair
        sum_squared_distance = np.sum((x1 - x2)**2)
        return np.sqrt(sum_squared_distance)
    
    def _get_single_prediction(self, x_datatest):
        # mendapatkan jarak antara baris uji dan semua baris pelatihan
        distances = [self._get_euclidean_distance(x_datatest, x_datatrain) 
                    for x_datatrain in self.X_train]
        # mendapatkan indeks dari k-tetangga terdekat -> k dengan jarak terkecil
        k_idx = np.argsort(distances)[:self.k]

        # mendapatkan label-y yang bersesuaian dari data pelatihan
        k_labels = [self.y_train[idx] for idx in k_idx]
        
        # return label yang paling sering muncul
        return Counter(k_labels).most_common(1)[0][0]
    
    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._get_single_prediction(x) for x in X])

prep = Preproccessing('src/train.csv')
student_clean = prep.cleaning_student_dataset()
X, y = prep.sparating_dataset(student_clean)
X_train, X_test, y_train, y_test = prep.splitting_dataset(X,y)

# Standarisasi fitur (fit hanya pada X_train untuk mencegah data leakage)
X_train_scaled, X_test_scaled = prep.scaling_dataset(X_train, X_test)

# Inisialisasi, latih, dan uji model KNN
model = KNN(k=5)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Evaluasi hasil prediksi
print(f"\nAkurasi Model KNN (k={model.k}): {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))