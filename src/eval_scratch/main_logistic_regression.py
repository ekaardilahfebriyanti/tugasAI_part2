import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocessing.pipeline import build_dataset, write_submission
from models.logistic_regression import LogisticRegression

# 1. mempersiapkan data untuk training
prep = build_dataset()
X_train_scaled, y_train = prep.X_train, prep.y_train
_, X_test_scaled, _, y_test = train_test_split(
    prep.X_val, prep.y_val, test_size=0.5, stratify=prep.y_val, random_state=42
)
nama_kelas = prep.class_names

# 2. men-training model
itr = 1000
l_r = 0.1
final_model_lr = LogisticRegression(learning_rate=l_r, iterations=itr)

t0 = time.time()
final_model_lr.fit(X_train_scaled, y_train)
train_time_lr = time.time() - t0

t0 = time.time()
pred_test_lr = final_model_lr.predict(X_test_scaled)
infer_time_lr = time.time() - t0

print("\n--- EVALUASI FINAL LOGISTIC REGRESSION (FROM SCRATCH) ---")
print(f"Fitur Terpilih     : {X_train_scaled.shape[1]} fitur (seleksi MI bawaan pipeline)")
print(f"Hyperparameter     : learning_rate={l_r}, iterations={itr}")
print(f"Training Time      : {train_time_lr:.6f} detik")
print(f"Inference Time     : {infer_time_lr:.6f} detik")
print(f"Accuracy           : {accuracy_score(y_test, pred_test_lr):.4f}")
print(classification_report(y_test, pred_test_lr, target_names=nama_kelas, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_test_lr))

# 3. simpan dan load model
folder_path = "results/saved_models"
os.makedirs(folder_path, exist_ok=True)

model_save_path = f"{folder_path}/logreg_model.pkl"
final_model_lr.save_model(model_save_path)

loaded_lr = LogisticRegression()
loaded_lr.load_model(model_save_path)
assert np.array_equal(pred_test_lr, loaded_lr.predict(X_test_scaled)), "Prediksi model loaded berbeda!"
print(f"\n[OK] Verifikasi Save & Load model Logistic Regression berhasil di {model_save_path} (prediksi identik 100%).")

# 4. prediksi test.csv
student_ids = prep.test_ids
X_submission_scaled = prep.X_test

hasil_prediksi_lr = final_model_lr.predict(X_submission_scaled)
pred_lr_teks = [nama_kelas[int(p)] for p in hasil_prediksi_lr]

df_submission_lr = pd.DataFrame({
    "Student_ID": student_ids,
    "Target": pred_lr_teks
})
print("\n5 hasil prediksi Logistic Regression pada test.csv:")
print(df_submission_lr.head())

submission_path = "results/submission_logreg.csv"
write_submission(student_ids, hasil_prediksi_lr, submission_path)
print(f"File {submission_path} berhasil disimpan!")