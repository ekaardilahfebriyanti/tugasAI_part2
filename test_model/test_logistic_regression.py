from preprocessing_pipeline.preprocessing import Preprocessing
from models.logistic_regression import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

prep = Preprocessing("src/train.csv")
student_clean = prep.cleaning_student_dataset()
X, y = prep.sparating_dataset(student_clean)
X_train, X_test, y_train, y_test = prep.splitting_dataset(X, y)

# menyeleksi fitur berdasarkan korelasi (threshold = 0.15)
X_train_sel, X_test_sel, corr_scores = prep.select_features_by_correlation(
    X_train, X_test, y_train, threshold=0.0
)

# menstandarisasi hanya pada fitur yang sudah terpilih
X_train_scaled, X_test_scaled = prep.scaling_dataset(X_train_sel, X_test_sel)
# melatih dan mengevaluasi model Logistic Regression
model_lr = LogisticRegression(learning_rate=0.15, iterations=1000)
model_lr.fit(X_train_scaled, y_train)

# menggunakan X_test_scaled dan model_lr
predictions = model_lr.predict(X_test_scaled)

acc = accuracy_score(y_test, predictions)

print(f"\nAkurasi Model Logistic Regression: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions))