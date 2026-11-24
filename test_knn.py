from sklearn.metrics import accuracy_score, classification_report
from preprocessing_pipeline.preprocessing import Preprocessing
from models.knn import KNN

prep = Preprocessing("src/train.csv")
student_clean = prep.cleaning_student_dataset()
X, y = prep.sparating_dataset(student_clean)
X_train, X_test, y_train, y_test = prep.splitting_dataset(X, y)

# menyeleksi fitur berdasarkan korelasi (threshold = 0.15)
X_train_sel, X_test_sel, corr_scores = prep.select_features_by_correlation(
    X_train, X_test, y_train, threshold=0.15
)

# menstandarisasi hanya pada fitur yang sudah terpilih
X_train_scaled, X_test_scaled = prep.scaling_dataset(X_train_sel, X_test_sel)

# model KNN
# melatih dengan euclidean
model = KNN(k=5, metric="euclidean")
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print(f"\nAkurasi Model KNN dengan Euclidean (k={model.k}): {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# melatih model KNN menggunakan jarak Manhattan
model_man = KNN(k=5, metric="manhattan")
model_man.fit(X_train_scaled, y_train)
pred_man = model_man.predict(X_test_scaled)

print(f"\nAkurasi Model KNN dengan jarak Manhattan (k={model_man.k}): {accuracy_score(y_test, pred_man):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, pred_man))