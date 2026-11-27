import time
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocessing.pipeline import build_dataset

# 1. mempersiapkan data untuk training
prep = build_dataset()
X_train_scaled, y_train = prep.X_train, prep.y_train
_, X_test_scaled, _, y_test = train_test_split(
    prep.X_val, prep.y_val, test_size=0.5, stratify=prep.y_val, random_state=42
)
nama_kelas = prep.class_names

# 2. men-training & evaluasi model knn sklearn dengan euclidean
sklearn_knn_eu = KNeighborsClassifier(n_neighbors=39, metric="euclidean")
t0 = time.time()
sklearn_knn_eu.fit(X_train_scaled, y_train)
train_time_eu = time.time() - t0

t0 = time.time()
pred_test_eu = sklearn_knn_eu.predict(X_test_scaled)
infer_time_eu = time.time() - t0

print("\n--- EVALUASI SKLEARN KNN EUCLIDEAN (k=39) ---")
print(f"Training Time  : {train_time_eu:.6f} detik")
print(f"Inference Time : {infer_time_eu:.6f} detik")
print(f"Accuracy       : {accuracy_score(y_test, pred_test_eu):.4f}")
print(classification_report(y_test, pred_test_eu, target_names=nama_kelas, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_test_eu))

# 3. men-training & evaluasi model knn sklearn dengan jarak manhattan
sklearn_knn_man = KNeighborsClassifier(n_neighbors=21, metric="manhattan")
t0 = time.time()
sklearn_knn_man.fit(X_train_scaled, y_train)
train_time_man = time.time() - t0

t0 = time.time()
pred_test_man = sklearn_knn_man.predict(X_test_scaled)
infer_time_man = time.time() - t0

print("\n--- EVALUASI SKLEARN KNN MANHATTAN (k=21) ---")
print(f"Training Time  : {train_time_man:.6f} detik")
print(f"Inference Time : {infer_time_man:.6f} detik")
print(f"Accuracy       : {accuracy_score(y_test, pred_test_man):.4f}")
print(classification_report(y_test, pred_test_man, target_names=nama_kelas, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_test_man))