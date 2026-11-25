import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocessing_pipeline.preprocessing import Preprocessing
from models.knn import KNN

# 1. melakukan data pre-processing dan pemilihan fitur
prep = Preprocessing("src/train.csv")
student_clean = prep.cleaning_student_dataset()
X, y = prep.sparating_dataset(student_clean)
X_train, X_test, y_train, y_test = prep.splitting_dataset(X, y)

# menyeleksi fitur berdasarkan korelasi dg threshold = 0.15
X_train_sel, X_test_sel, corr_scores = prep.select_features_by_correlation(
    X_train, X_test, y_train, threshold=0.15
)

# menstandarisasi hanya pada fitur yg sudah terpilih
X_train_scaled, X_test_scaled = prep.scaling_dataset(X_train_sel, X_test_sel)
nama_kelas = prep.label_encoder.classes_


# 2. melakukan training knn euclidean
final_modelknn_eu = KNN(k=9, metric="euclidean")

t0 = time.time()
final_modelknn_eu.fit(X_train_scaled, y_train)
train_time_eu = time.time() - t0

t0 = time.time()
pred_test_eu = final_modelknn_eu.predict(X_test_scaled)
infer_time_eu = time.time() - t0

print("--- EVALUASI KNN EUCLIDEAN (k=11) ---")
print(f"Training Time : {train_time_eu:.6f} detik")
print(f"Inference Time: {infer_time_eu:.6f} detik")
print(f"Accuracy      : {accuracy_score(y_test, pred_test_eu):.4f}")
print(classification_report(y_test, pred_test_eu, target_names=nama_kelas, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_test_eu))


# 3. melakukan training dg jarak manhattan
final_modelknn_man = KNN(k=13, metric="manhattan")

t0 = time.time()
final_modelknn_man.fit(X_train_scaled, y_train)
train_time_man = time.time() - t0

t0 = time.time()
pred_test_man = final_modelknn_man.predict(X_test_scaled)
infer_time_man = time.time() - t0

print("\n--- EVALUASI KNN MANHATTAN (k=13) ---")
print(f"Training Time : {train_time_man:.6f} detik")
print(f"Inference Time: {infer_time_man:.6f} detik")
print(f"Accuracy      : {accuracy_score(y_test, pred_test_man):.4f}")
print(classification_report(y_test, pred_test_man, target_names=nama_kelas, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_test_man))


# 4. menyimpan dan load model
final_modelknn_eu.save_model_knn("saved_models/knn_euclidean.pkl")
final_modelknn_man.save_model_knn("saved_models/knn_manhattan.pkl")

loaded_knn_man = KNN()
loaded_knn_man.load_model_knn("saved_models/knn_manhattan.pkl")
assert np.array_equal(pred_test_man, loaded_knn_man.predict(X_test_scaled)), "Prediksi model loaded berbeda!"
print("\n[OK] Verifikasi Save & Load model KNN berhasil (prediksi identik 100%).")

loaded_knn_eu = KNN()
loaded_knn_eu.load_model_knn("saved_models/knn_manhattan.pkl")
assert np.array_equal(pred_test_man, loaded_knn_eu.predict(X_test_scaled)), "Prediksi model loaded berbeda!"
print("\n[OK] Verifikasi Save & Load model KNN berhasil (prediksi identik 100%).")


# 5. memprediksi test.csv dan simpan output
student_ids, X_submission_scaled = prep.prepare_test_data("src/test.csv")

hasil_prediksi_knn_eu = final_modelknn_eu.predict(X_submission_scaled)
hasil_prediksi_knn_man = final_modelknn_man.predict(X_submission_scaled)

pred_knn_eu_teks = prep.label_encoder.inverse_transform(hasil_prediksi_knn_eu)
pred_knn_man_teks = prep.label_encoder.inverse_transform(hasil_prediksi_knn_man)

# DataFrame perbandingan kedua metrik
df_hasil = pd.DataFrame({
    "Student_ID": student_ids,
    "Target_kNN_Euclidean": pred_knn_eu_teks,
    "Target_kNN_Manhattan": pred_knn_man_teks,
})
print("\n5 hasil prediksi pada test.csv:")
print(df_hasil.head())


df_submission_kaggle = pd.DataFrame({
    "Student_ID": student_ids,
        pilih model dengan akurasi terbaik
    "Target": pred_knn_man_teks  
})
df_submission_kaggle.to_csv("submission_knn_manhattan.csv", index=False)
print("\nFile submission_knn_manhattan.csv berhasil disimpan!")