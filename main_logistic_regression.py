import time
import os
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocessing_pipeline.preprocessing import Preprocessing
from models.logistic_regression import LogisticRegression

prep = Preprocessing("src/train.csv")
student_clean = prep.cleaning_student_dataset()
X, y = prep.sparating_dataset(student_clean)
X_train, X_test, y_train, y_test = prep.splitting_dataset(X, y)

# menyeleksi fitur berdasarkan korelasi (threshold = 0.15)
X_train_sel, X_test_sel, corr_scores = prep.select_features_by_correlation(
    X_train, X_test, y_train, threshold=0.0
)

itr = 1000
l_r =0.15
# 1. mentraining model LOGISTIC REGRESSION final & melakukan evaluasi
final_model_lr = LogisticRegression(learning_rate=l_r, iterations=itr)

# mengukur waktu training
t0 = time.time()
final_model_lr.fit(X_train_scaled, y_train)
train_time_lr = time.time() - t0

# mengukur waktu inference (prediksi)
t0 = time.time()
pred_test_lr = final_model_lr.predict(X_test_scaled)
infer_time_lr = time.time() - t0

nama_kelas = prep.label_encoder.classes_

print("\n=== EVALUASI FINAL LOGISTIC REGRESSION (FROM SCRATCH) ===")
print(f"Threshold Terpilih : {best_threshold} ({X_train_scaled.shape[1]} fitur)")
print(f"Hyperparameter     : learning_rate={l_r}, iterations={iter}")
print(f"Training Time      : {train_time_lr:.6f} detik")
print(f"Inference Time     : {infer_time_lr:.6f} detik")
print(f"Accuracy           : {accuracy_score(y_test, pred_test_lr):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, pred_test_lr, target_names=nama_kelas, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred_test_lr))


# 2. menyimpan dan load model logreg
folder_path = '/content/drive/My Drive/project2AILanjut/saved_models'
os.makedirs(folder_path, exist_ok=True)

model_save_path = f"{folder_path}/logreg_model.pkl"
final_model_lr.save_model(model_save_path)

# memverifikasi load model menghasilkan prediksi yang sama persis
loaded_lr = LogisticRegression()
loaded_lr.load_model(model_save_path)
assert np.array_equal(pred_test_lr, loaded_lr.predict(X_test_scaled)), "Prediksi model loaded berbeda!"
print(f"\nModel berhasil disimpan di {model_save_path} dan terverifikasi identik saat di-load.")


# 3. membuat prediksi pada data test.csv dan simpan prediksinya

student_ids, X_submission_scaled = prep.prepare_test_data(
    '/content/drive/My Drive/project2AILanjut/test.csv'
)

hasil_prediksi_lr = final_model_lr.predict(X_submission_scaled)
pred_lr_teks = prep.label_encoder.inverse_transform(hasil_prediksi_lr)

df_submission_lr = pd.DataFrame({
    "Student_ID": student_ids,
    "Target": pred_lr_teks
})

print("\n5 hasil prediksi Logistic Regression pada test.csv:")
print(df_submission_lr.head())

submission_path = '/content/drive/My Drive/project2AILanjut/submission_logreg.csv'
df_submission_lr.to_csv(submission_path, index=False)
print(f"\nFile {submission_path} berhasil disimpan!")