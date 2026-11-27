import time
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocessing.pipeline import build_dataset

# 1. mempersiapkan data untuk training
prep = build_dataset()
X_train_scaled, y_train = prep.X_train, prep.y_train
_, X_test_scaled, _, y_test = train_test_split(
    prep.X_val, prep.y_val, test_size=0.5, stratify=prep.y_val, random_state=42
)
nama_kelas = prep.class_names

# 2. men-training & evaluasi model decision tree sklearn (CART, criterion gini)
# hyperparameter sama dengan model from scratch (hasil tuning di main_decision_tree_learning.py)
max_depth = 8
min_samples_split = 30
final_model_dtl = DecisionTreeClassifier(criterion="gini", max_depth=max_depth,
                                         min_samples_split=min_samples_split, random_state=42)

t0 = time.time()
final_model_dtl.fit(X_train_scaled, y_train)
train_time_dtl = time.time() - t0

t0 = time.time()
pred_test_dtl = final_model_dtl.predict(X_test_scaled)
infer_time_dtl = time.time() - t0

print("\n--- EVALUASI SKLEARN DECISION TREE (CART, criterion=gini) ---")
print(f"Hyperparameter : max_depth={max_depth}, min_samples_split={min_samples_split}")
print(f"Training Time  : {train_time_dtl:.6f} detik")
print(f"Inference Time : {infer_time_dtl:.6f} detik")
print(f"Accuracy       : {accuracy_score(y_test, pred_test_dtl):.4f}")
print(classification_report(y_test, pred_test_dtl, target_names=nama_kelas, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_test_dtl))
