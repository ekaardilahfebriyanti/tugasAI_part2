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

# model KNN k-n
# melatih dengan euclidean

model = KNN(k=3, metric="euclidean")
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print(f"\nAkurasi Model KNN dengan Euclidean (k={model.k}): {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

model5 = KNN(k=5, metric="euclidean")
model5.fit(X_train_scaled, y_train)
y_pred5 = model5.predict(X_test_scaled)

print(f"\nAkurasi Model KNN dengan Euclidean (k={model5.k}): {accuracy_score(y_test, y_pred5):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred5))

model7 = KNN(k=7, metric="euclidean")
model7.fit(X_train_scaled, y_train)
y_pred7 = model7.predict(X_test_scaled)

print(f"\nAkurasi Model KNN dengan Euclidean (k={model7.k}): {accuracy_score(y_test, y_pred7):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred7))

model9 = KNN(k=9, metric="euclidean")
model9.fit(X_train_scaled, y_train)
y_pred9 = model9.predict(X_test_scaled)

print(f"\nAkurasi Model KNN dengan Euclidean (k={model9.k}): {accuracy_score(y_test, y_pred9):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred9))

# melatih model KNN dengan k-n menggunakan jarak Manhattan
model_man = KNN(k=3, metric="manhattan")
model_man.fit(X_train_scaled, y_train)
pred_man = model_man.predict(X_test_scaled)

print(f"\nAkurasi Model KNN dengan jarak Manhattan (k={model_man.k}): {accuracy_score(y_test, pred_man):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, pred_man))

model_man5 = KNN(k=5, metric="manhattan")
model_man5.fit(X_train_scaled, y_train)
pred_man5 = model_man5.predict(X_test_scaled)

print(f"\nAkurasi Model KNN dengan jarak Manhattan (k={model_man5.k}): {accuracy_score(y_test, pred_man5):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, pred_man5))

model_man7 = KNN(k=7, metric="manhattan")
model_man7.fit(X_train_scaled, y_train)
pred_man7 = model_man7.predict(X_test_scaled)

print(f"\nAkurasi Model KNN dengan jarak Manhattan (k={model_man7.k}): {accuracy_score(y_test, pred_man7):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, pred_man7))

model_man9 = KNN(k=9, metric="manhattan")
model_man9.fit(X_train_scaled, y_train)
pred_man9 = model_man9.predict(X_test_scaled)

print(f"\nAkurasi Model KNN dengan jarak Manhattan (k={model_man9.k}): {accuracy_score(y_test, pred_man9):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, pred_man9))

model_man11 = KNN(k=11, metric="manhattan")
model_man11.fit(X_train_scaled, y_train)
pred_man11 = model_man11.predict(X_test_scaled)

print(f"\nAkurasi Model KNN dengan jarak Manhattan (k={model_man11.k}): {accuracy_score(y_test, pred_man11):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, pred_man5))

model_man13 = KNN(k=13, metric="manhattan")
model_man13.fit(X_train_scaled, y_train)
pred_man13 = model_man13.predict(X_test_scaled)

print(f"\nAkurasi Model KNN dengan jarak Manhattan (k={model_man13.k}): {accuracy_score(y_test, pred_man13):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, pred_man13))

