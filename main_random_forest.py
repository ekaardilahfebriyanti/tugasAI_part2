import os
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from preprocessing.pipeline import build_dataset, write_submission
from models.random_forest import RandomForest

# 1. data 
ds = build_dataset()
X_train, y_train = ds.X_train, ds.y_train
X_submission = ds.X_test  

X_val, X_test, y_val, y_test = train_test_split(ds.X_val, ds.y_val, test_size=0.5, stratify=ds.y_val, random_state=42)

# 2. tuning hyperparameter
print("Evaluasi Kandidat Hyperparameter Random Forest (data validasi)")
best_f1, best_param = -1, None
for n_est in [10, 25, 50]:
    for depth in [6, 10, None]:
        m = RandomForest(n_estimators=n_est, max_depth=depth, min_samples_split=10, random_state=42)
        t0 = time.time()
        m.fit(X_train, y_train)
        waktu = time.time() - t0
        pred = m.predict(X_val)
        acc, f1 = accuracy_score(y_val, pred), f1_score(y_val, pred, average="macro")
        print(f"-> n_estimators={n_est:2d} max_depth={str(depth):4s} | Akurasi: {acc:.4f} | Macro F1: {f1:.4f} | fit {waktu:.1f}s")
        if f1 > best_f1:
            best_f1, best_param = f1, (n_est, depth)
n_estimators, max_depth = best_param
print(f"\nHyperparameter terbaik: n_estimators={n_estimators}, max_depth={max_depth} (Macro F1: {best_f1:.4f})")

# 3. training model
model = RandomForest(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=10, random_state=42)
t0 = time.time()
model.fit(X_train, y_train)
train_time = time.time() - t0

t0 = time.time()
pred_test = model.predict(X_test)
infer_time = time.time() - t0

print("\n--- EVALUASI RANDOM FOREST (FROM SCRATCH) ---")
print(f"Training Time  : {train_time:.6f} detik")
print(f"Inference Time : {infer_time:.6f} detik")
print(f"Accuracy       : {accuracy_score(y_test, pred_test):.4f}")
print(classification_report(y_test, pred_test, target_names=ds.class_names, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_test))

# 4. simpan dan load model
os.makedirs("results/saved_models", exist_ok=True)
model.save_model_rf("results/saved_models/random_forest.pkl")
loaded = RandomForest()
loaded.load_model_rf("results/saved_models/random_forest.pkl")
assert np.array_equal(pred_test, loaded.predict(X_test)), "Prediksi model loaded berbeda!"
print("\n[OK] Verifikasi Save & Load model Random Forest berhasil (prediksi identik 100%).")

# 5. prediksi test.csv
write_submission(ds.test_ids, model.predict(X_submission), "results/submission_random_forest.csv")
print("File results/submission_random_forest.csv berhasil disimpan!")
