from sklearn.metrics import accuracy_score, classification_report
from preprocessing_pipeline.preprocessing import Preprocessing
from models.knn import KNN

prep = Preprocessing("src/train.csv")
student_clean = prep.cleaning_student_dataset()
X, y = prep.sparating_dataset(student_clean)
X_train, X_test, y_train, y_test = prep.splitting_dataset(X, y)

# mengecekan threshold terbaik sebelum standarisasi final
kandidat_threshold = [0.0, 0.03, 0.05, 0.10, 0.15, 0.20]
best_threshold = 0.05
best_acc = 0.0

print("-------------------------------------")
print("Evaluasi Kandidat Threshold")
print("-------------------------------------")
for thresh in kandidat_threshold:
    X_tr_tmp, X_te_tmp, _ = prep.select_features_by_correlation(
        X_train, X_test, y_train, threshold=thresh
    )
    # standarisasi sementara di dlm loop agar perhitungan jarak KNN tetap adil
    X_tr_tmp_scaled, X_te_tmp_scaled = prep.scaling_dataset(X_tr_tmp, X_te_tmp)

    model_tmp = KNN(k=5, metric="euclidean")
    model_tmp.fit(X_tr_tmp_scaled, y_train)
    acc = accuracy_score(y_test, model_tmp.predict(X_te_tmp_scaled))

    print(f"-> Threshold {thresh:.2f} | Fitur: {X_tr_tmp.shape[1]:2d} | Akurasi: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        best_threshold = thresh

print(f"\nThreshold terbaik terpilih: {best_threshold} (Akurasi: {best_acc:.4f})")

# menyeleksi fitur berdasarkan threshold terbaik
X_train_sel, X_test_sel, corr_scores = prep.select_features_by_correlation(
    X_train, X_test, y_train, threshold=best_threshold
)

# melihat 5 fitur dengan korelasi tertinggi
print("\n5 fitur paling berkorelasi:")
print(corr_scores.sort_values(ascending=False).head(5))

# standarisasi hanya pada fitur yg sudah terpilih
X_train_scaled, X_test_scaled = prep.scaling_dataset(X_train_sel, X_test_sel)

