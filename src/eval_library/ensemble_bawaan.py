import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import copy
from preprocessing.pipeline import build_dataset

# 1. data (pembagian sama dengan model buatan sendiri)
ds = build_dataset()
X_train, y_train = ds.X_train, ds.y_train
X_val_all, y_val_all = ds.X_val, ds.y_val
_, X_report, _, y_report = train_test_split(X_val_all, y_val_all, test_size=0.5, stratify=y_val_all, random_state=42)

# 2. lima anggota scikit-learn dengan konfigurasi disamakan dengan anggota buatan sendiri
def fit_mlp():
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation="tanh", solver="sgd", momentum=0.9,
                        nesterovs_momentum=False, learning_rate_init=0.01, alpha=0.001 * len(X_train),
                        batch_size=64, random_state=42)
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
    return best_model

builders = {
    "ANN": fit_mlp,
    "LR": lambda: LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train),
    "RF": lambda: RandomForestClassifier(n_estimators=25, min_samples_split=10, random_state=42).fit(X_train, y_train),
    "DTL": lambda: DecisionTreeClassifier(max_depth=8, min_samples_split=30, random_state=42).fit(X_train, y_train),
    "KNN": lambda: KNeighborsClassifier(n_neighbors=21, metric="manhattan").fit(X_train, y_train),
}
members = {}
t0 = time.time()
for name, build in builders.items():
    members[name] = build()
train_time = time.time() - t0

print("--- ANGGOTA ENSEMBLE SCIKIT-LEARN (data laporan: setengah validasi) ---")
for name, m in members.items():
    pred = m.predict(X_report)
    print(f"{name:4s} Accuracy {accuracy_score(y_report, pred):.4f} | Macro F1 {f1_score(y_report, pred, average='macro'):.4f}")

# 3. hard voting bobot sama, dibuat manual (setara VotingClassifier) karena MLP dengan early stopping manual
# tidak bisa dibungkus VotingClassifier tanpa dilatih ulang. Bila seri, anggota dengan urutan lebih awal menang.
def hard_vote(X):
    preds = np.array([m.predict(X) for m in members.values()])
    return np.array([np.bincount(preds[:, i], minlength=3).argmax() for i in range(preds.shape[1])])

t0 = time.time()
pred = hard_vote(X_report)
infer_time = time.time() - t0

print("\n--- EVALUASI ENSEMBLE HARD VOTING SCIKIT-LEARN ---")
print(f"Training Time  : {train_time:.6f} detik (jumlah kelima anggota)")
print(f"Inference Time : {infer_time:.6f} detik")
print(f"Accuracy       : {accuracy_score(y_report, pred):.4f}")
print(f"Macro F1       : {f1_score(y_report, pred, average='macro'):.4f}")
print(classification_report(y_report, pred, target_names=ds.class_names, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_report, pred))
