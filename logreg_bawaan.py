import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocessing.pipeline import build_dataset

# 1. mempersiapkan data untuk training
prep = build_dataset()
X_train_scaled, y_train = prep.X_train, prep.y_train
_, X_test_scaled, _, y_test = train_test_split(
    prep.X_val, prep.y_val, test_size=0.5, stratify=prep.y_val, random_state=42
)
nama_kelas = prep.class_names

# 2. men-training & evaluasi model logistic regression sklearn
itr = 1000
final_model_lr = LogisticRegression(max_iter=itr, random_state=42)

t0 = time.time()
final_model_lr.fit(X_train_scaled, y_train)
train_time_lr = time.time() - t0

t0 = time.time()
pred_test_lr = final_model_lr.predict(X_test_scaled)
infer_time_lr = time.time() - t0

print("\n--- EVALUASI FINAL SKLEARN LOGISTIC REGRESSION ---")
print(f"Fitur Terpilih     : {X_train_scaled.shape[1]} fitur (seleksi MI bawaan pipeline)")
print(f"Hyperparameter     : max_iter={itr}")
print(f"Training Time      : {train_time_lr:.6f} detik")
print(f"Inference Time     : {infer_time_lr:.6f} detik")
print(f"Accuracy           : {accuracy_score(y_test, pred_test_lr):.4f}")
print(classification_report(y_test, pred_test_lr, target_names=nama_kelas, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_test_lr))