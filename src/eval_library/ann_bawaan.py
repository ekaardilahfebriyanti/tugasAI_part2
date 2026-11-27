import copy
import time
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from preprocessing.pipeline import build_dataset

# 1. data (pembagian sama dengan model buatan sendiri)
ds = build_dataset()
X_train, y_train = ds.X_train, ds.y_train
X_val_all, y_val_all = ds.X_val, ds.y_val
_, X_report, _, y_report = train_test_split(X_val_all, y_val_all, test_size=0.5, stratify=y_val_all, random_state=42)

# 2. MLPClassifier scikit-learn dengan pengaturan disamakan (64-32, SGD momentum 0,9, batch 64, lr 0,01)
# scikit-learn hanya menerima satu aktivasi untuk semua lapisan (tanh) dan alpha harus dikali jumlah sampel agar setara L2 = 0,001
# early stopping dibuat manual: pantau Macro F1 pada seluruh validasi, kesabaran 15 epoch
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation="tanh", solver="sgd", momentum=0.9,
                    nesterovs_momentum=False, learning_rate_init=0.01, alpha=0.001 * len(X_train),
                    batch_size=64, random_state=42)
t0 = time.time()
best_f1, best_model, wait = -1.0, None, 0
for epoch in range(100):
    mlp.partial_fit(X_train, y_train, classes=[0, 1, 2])
    f1 = f1_score(y_val_all, mlp.predict(X_val_all), average="macro")
    if f1 > best_f1:
        best_f1, best_model, wait = f1, copy.deepcopy(mlp), 0
    else:
        wait += 1
        if wait >= 15:
            break
train_time = time.time() - t0

# 3. evaluasi pada setengah validasi untuk laporan
t0 = time.time()
pred = best_model.predict(X_report)
infer_time = time.time() - t0

print("--- EVALUASI MLPClassifier SCIKIT-LEARN ---")
print(f"Training Time  : {train_time:.6f} detik")
print(f"Inference Time : {infer_time:.6f} detik")
print(f"Accuracy       : {accuracy_score(y_report, pred):.4f}")
print(f"Macro F1       : {f1_score(y_report, pred, average='macro'):.4f}")
print(classification_report(y_report, pred, target_names=ds.class_names, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_report, pred))
