import os
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from preprocessing.pipeline import build_dataset, write_submission
from models.artificial_neural_network import ANN

# 1. data (validasi dibagi dua: setengah untuk tuning, setengah untuk laporan, sama dengan skrip model lain)
ds = build_dataset()
X_train, y_train = ds.X_train, ds.y_train
X_val_all, y_val_all = ds.X_val, ds.y_val
_, X_report, _, y_report = train_test_split(X_val_all, y_val_all, test_size=0.5, stratify=y_val_all, random_state=42)

# 2. training ANN buatan sendiri (arsitektur 20-64-32-3, ReLU, tanh, softmax)
# early stopping memakai seluruh data validasi (620 data)
model = ANN(hidden_sizes=(64, 32), hidden_activations=("relu", "tanh"), learning_rate=0.01,
            l2_lambda=0.001, max_epochs=100, patience=15, seed=42)
t0 = time.time()
model.fit(X_train, y_train, X_val_all, y_val_all)
train_time = time.time() - t0

# 3. evaluasi pada setengah validasi untuk laporan
t0 = time.time()
pred = model.predict(X_report)
infer_time = time.time() - t0

print("--- EVALUASI ANN BUATAN SENDIRI ---")
print(f"Training Time  : {train_time:.6f} detik")
print(f"Inference Time : {infer_time:.6f} detik")
print(f"Accuracy       : {accuracy_score(y_report, pred):.4f}")
print(f"Macro F1       : {f1_score(y_report, pred, average='macro'):.4f}")
print(classification_report(y_report, pred, target_names=ds.class_names, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_report, pred))

# 4. simpan dan load model
folder = "results/saved_models"
os.makedirs(folder, exist_ok=True)
model.save(f"{folder}/ann.npz")
loaded = ANN.load(f"{folder}/ann.npz")
assert np.array_equal(pred, loaded.predict(X_report)), "Prediksi model loaded berbeda!"
print("\n[OK] Verifikasi Save & Load ANN berhasil (prediksi identik 100%).")

# 5. prediksi test.csv
write_submission(ds.test_ids, model.predict(ds.X_test), "results/submission_ann.csv")
print("File results/submission_ann.csv berhasil disimpan!")
