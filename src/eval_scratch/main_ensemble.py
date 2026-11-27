import os
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from preprocessing.pipeline import build_dataset, write_submission
from models.artificial_neural_network import ANN
from models.decision_tree_learning import DecisionTree
from models.random_forest import RandomForest
from models.k_nearest_neighbors import KNN
from models.logistic_regression import LogisticRegression
from models.ensemble import Ensemble

# 1. data (pembagian validasi sama dengan skrip model lain: setengah untuk tuning, setengah untuk laporan)
ds = build_dataset()
X_submission = ds.X_test
X_val_all, y_val_all = ds.X_val, ds.y_val
_, X_report, _, y_report = train_test_split(X_val_all, y_val_all, test_size=0.5, stratify=y_val_all, random_state=42)

# 2. memuat lima anggota yang sudah dilatih dan disimpan oleh skrip masing-masing
folder = "results/saved_models"
ann = ANN.load(f"{folder}/ann.npz")
lr = LogisticRegression(); lr.load_model(f"{folder}/logreg_model.pkl")
rf = RandomForest(); rf.load_model_rf(f"{folder}/random_forest.pkl")
dtl = DecisionTree(); dtl.load_model_dtl(f"{folder}/decision_tree_learning.pkl")
knn = KNN(); knn.load_model_knn(f"{folder}/knn_manhattan.pkl")
members = {"ANN": ann, "LR": lr, "RF": rf, "DTL": dtl, "KNN": knn}

print("--- ANGGOTA ENSEMBLE (data laporan: setengah validasi) ---")
for name, m in members.items():
    t0 = time.time()
    pred = m.predict(X_report)
    infer = time.time() - t0
    print(f"{name:4s} Accuracy {accuracy_score(y_report, pred):.4f} | Macro F1 {f1_score(y_report, pred, average='macro'):.4f} | Inference {infer:.6f} detik")

# 3. ensemble hard voting dengan bobot sama
ensemble = Ensemble(members, voting="hard")
t0 = time.time()
pred_report = ensemble.predict(X_report)
infer_time = time.time() - t0
pred_all = ensemble.predict(X_val_all)

print("\n--- EVALUASI ENSEMBLE HARD VOTING (BOBOT SAMA) ---")
print(f"Anggota        : {ensemble.names}")
print(f"Inference Time : {infer_time:.6f} detik")
print(f"Accuracy       : {accuracy_score(y_report, pred_report):.4f}")
print(f"Macro F1       : {f1_score(y_report, pred_report, average='macro'):.4f}")
print(classification_report(y_report, pred_report, target_names=ds.class_names, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_report, pred_report))
print(f"\nSeluruh validasi (620 data): Accuracy {accuracy_score(y_val_all, pred_all):.4f} | Macro F1 {f1_score(y_val_all, pred_all, average='macro'):.4f}")

# 4. simpan dan load model
os.makedirs(folder, exist_ok=True)
ensemble.save(f"{folder}/ensemble.pkl")
loaded = Ensemble.load(f"{folder}/ensemble.pkl")
assert np.array_equal(pred_report, loaded.predict(X_report)), "Prediksi model loaded berbeda!"
print("\n[OK] Verifikasi Save & Load Ensemble berhasil (prediksi identik 100%).")

# 5. prediksi test.csv
write_submission(ds.test_ids, ensemble.predict(X_submission), "results/submission_ensemble.csv")
print("File results/submission_ensemble.csv berhasil disimpan!")
