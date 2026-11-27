import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocessing.pipeline import build_dataset, write_submission
from models.k_nearest_neighbors import KNN

# 1. mempersiapkan data untuk training
prep = build_dataset()
X_train_scaled, y_train = prep.X_train, prep.y_train
_, X_test_scaled, _, y_test = train_test_split(
    prep.X_val, prep.y_val, test_size=0.5, stratify=prep.y_val, random_state=42
)
nama_kelas = prep.class_names

# 2. training model knn dengan euclidean
final_modelknn_eu = KNN(k=39, metric="euclidean")
t0 = time.time()
final_modelknn_eu.fit(X_train_scaled, y_train)
train_time_eu = time.time() - t0

t0 = time.time()
pred_test_eu = final_modelknn_eu.predict(X_test_scaled)
infer_time_eu = time.time() - t0

print("\n--- EVALUASI KNN EUCLIDEAN (k=39) ---")
print(f"Training Time  : {train_time_eu:.6f} detik")
print(f"Inference Time : {infer_time_eu:.6f} detik")
print(f"Accuracy       : {accuracy_score(y_test, pred_test_eu):.4f}")
print(classification_report(y_test, pred_test_eu, target_names=nama_kelas, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_test_eu))

# 3. training model knn dengan jarak manhattan
final_modelknn_man = KNN(k=21, metric="manhattan")
t0 = time.time()
final_modelknn_man.fit(X_train_scaled, y_train)
train_time_man = time.time() - t0

t0 = time.time()
pred_test_man = final_modelknn_man.predict(X_test_scaled)
infer_time_man = time.time() - t0

print("\n--- EVALUASI KNN MANHATTAN (k=21) ---")
print(f"Training Time  : {train_time_man:.6f} detik")
print(f"Inference Time : {infer_time_man:.6f} detik")
print(f"Accuracy       : {accuracy_score(y_test, pred_test_man):.4f}")
print(classification_report(y_test, pred_test_man, target_names=nama_kelas, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_test_man))

# 4. simpan dan load model
os.makedirs("results/saved_models", exist_ok=True)
final_modelknn_eu.save_model_knn("results/saved_models/knn_euclidean.pkl")
final_modelknn_man.save_model_knn("results/saved_models/knn_manhattan.pkl")

loaded_knn_eu = KNN()
loaded_knn_eu.load_model_knn("results/saved_models/knn_euclidean.pkl")
assert np.array_equal(pred_test_eu, loaded_knn_eu.predict(X_test_scaled)), "Prediksi model loaded berbeda!"
print("\n[OK] Verifikasi Save & Load model KNN Euclidean berhasil (prediksi identik 100%).")

loaded_knn_man = KNN()
loaded_knn_man.load_model_knn("results/saved_models/knn_manhattan.pkl")
assert np.array_equal(pred_test_man, loaded_knn_man.predict(X_test_scaled)), "Prediksi model loaded berbeda!"
print("[OK] Verifikasi Save & Load model KNN Manhattan berhasil (prediksi identik 100%).")

# 5. prediksi test.csv
student_ids = prep.test_ids
X_submission_scaled = prep.X_test

hasil_prediksi_knn_eu = final_modelknn_eu.predict(X_submission_scaled)
hasil_prediksi_knn_man = final_modelknn_man.predict(X_submission_scaled)

pred_knn_eu_teks = [nama_kelas[int(p)] for p in hasil_prediksi_knn_eu]
pred_knn_man_teks = [nama_kelas[int(p)] for p in hasil_prediksi_knn_man]

df_hasil = pd.DataFrame({
    "Student_ID": student_ids,
    "Target_kNN_Euclidean": pred_knn_eu_teks,
    "Target_kNN_Manhattan": pred_knn_man_teks,
})
print("\n5 hasil prediksi pada test.csv:")
print(df_hasil.head())

df_terbaik = pd.DataFrame({
    "Student_ID": student_ids,
    "Target": pred_knn_man_teks
})
df_terbaik.to_csv("results/submission_knn_manhattan.csv", index=False)
write_submission(student_ids, hasil_prediksi_knn_man, "results/submission_knn_manhattan.csv")
print("File results/submission_knn_manhattan.csv berhasil disimpan!")