import time
import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.neural_network import MLPClassifier
from preprocessing.pipeline import build_dataset
from models.artificial_neural_network import ANN
from models.decision_tree_learning import DecisionTree
from models.random_forest import RandomForest
from models.k_nearest_neighbors import KNN
from models.logistic_regression import LogisticRegression
from models.ensemble import Ensemble

SEED = 42


def section(title, width=70):
    print("\n" + "=" * width + f"\n{title}\n" + "=" * width)


def acc_f1(y, p):
    return accuracy_score(y, p), f1_score(y, p, average="macro")


ds = build_dataset()
X_train, y_train = ds.X_train, ds.y_train
X_val_all, y_val_all = ds.X_val, ds.y_val
X_tune, X_rep, y_tune, y_rep = train_test_split(X_val_all, y_val_all, test_size=0.5, stratify=y_val_all, random_state=SEED)
# X_tune: setengah validasi untuk memilih bobot, X_rep: setengah lain khusus untuk melaporkan hasil

section("1.1 Lima anggota final (dilatih ulang dengan hyperparameter dari skrip masing-masing)")
builders = {
    "ANN": (lambda: ANN(hidden_sizes=(64, 32), hidden_activations=("relu", "tanh"), learning_rate=0.01,
                        l2_lambda=0.001, max_epochs=100, patience=15, seed=SEED),
            lambda m: m.fit(X_train, y_train, X_val_all, y_val_all), "relu/tanh 64-32, lr 0.01"),
    "LR": (lambda: LogisticRegression(learning_rate=0.1, iterations=1000),
           lambda m: m.fit(X_train, y_train), "softmax, lr 0.1, 1000 iterasi"),
    "RF": (lambda: RandomForest(n_estimators=25, max_depth=None, min_samples_split=10, random_state=SEED),
           lambda m: m.fit(X_train, y_train), "25 pohon, split 10"),
    "DTL": (lambda: DecisionTree(max_depth=8, min_samples_split=30),
            lambda m: m.fit(X_train, y_train), "CART gini, depth 8, split 30"),
    "KNN": (lambda: KNN(k=21, metric="manhattan"),
            lambda m: m.fit(X_train, y_train), "k 21, manhattan"),
}
members, fit_s, rows = {}, {}, []
for name, (make, fit, config) in builders.items():
    model = make()
    t0 = time.time()
    fit(model)
    fit_s[name] = time.time() - t0
    members[name] = model
    t0 = time.time()
    pred = model.predict(X_rep)
    infer_ms = (time.time() - t0) * 1000
    a, f = acc_f1(y_rep, pred)
    rows.append({"Model": name, "Konfigurasi": config, "Accuracy": a, "Macro F1": f,
                 "Waktu latih (s)": fit_s[name], "Inferensi (ms)": infer_ms})
members_df = pd.DataFrame(rows).set_index("Model")
print(members_df.round(4).to_string())
print(f"\nTotal waktu latih lima anggota : {sum(fit_s.values()):.2f} detik")

section("1.2 Ensemble hard voting")
ens_equal = Ensemble(members, voting="hard")
ens_weighted = Ensemble(members, voting="hard")
weights = ens_weighted.set_weights_from_validation(X_tune, y_tune)
print("Urutan anggota (tie-break)           :", ens_equal.names)
print("Bobot dari Macro F1 (setengah tuning) :", {k: round(v, 4) for k, v in weights.items()})
ens_results = {}
for label, ens in [("Bobot sama", ens_equal), ("Bobot Macro F1", ens_weighted)]:
    t0 = time.time()
    pred = ens.predict(X_rep)
    infer_ms = (time.time() - t0) * 1000
    a, f = acc_f1(y_rep, pred)
    ens_results[label] = {"pred": pred, "acc": a, "f1": f, "ms": infer_ms}
    print(f"{label:15s} Accuracy {a:.4f} | Macro F1 {f:.4f} | Inferensi {infer_ms:.1f} ms")
chosen = ens_results["Bobot sama"]
print("\nBobot sama (varian terpilih):")
print(classification_report(y_rep, chosen["pred"], target_names=ds.class_names, digits=4))
print("Confusion Matrix:\n", confusion_matrix(y_rep, chosen["pred"]))

section("1.3 Perbandingan dengan tiap anggota")
compare = members_df[["Accuracy", "Macro F1"]].copy()
for label, r in ens_results.items():
    compare.loc[f"Ensemble {label.lower()}"] = [r["acc"], r["f1"]]
print(compare.sort_values("Macro F1", ascending=False).round(4).to_string())
best_member = members_df["Macro F1"].idxmax()
print(f"\nAnggota terbaik : {best_member} ({members_df['Macro F1'].max():.4f})")
print(f"Ensemble bobot sama dikurangi anggota terbaik : {chosen['f1'] - members_df['Macro F1'].max():+.4f}")

section("1.4 Keragaman anggota")
member_pred = ens_equal.member_predictions(X_rep)
names = ens_equal.names
agree = pd.DataFrame([[np.mean(member_pred[a] == member_pred[b]) for b in names] for a in names], index=names, columns=names)
print("Kesepakatan prediksi antar anggota (proporsi data):")
print(agree.round(3).to_string())
correct = np.array([member_pred[n] == y_rep for n in names])
print(f"\nSetidaknya satu anggota benar : {correct.any(axis=0).mean():.4f}")
print(f"Seluruh anggota benar         : {correct.all(axis=0).mean():.4f}")
print(f"Seluruh anggota salah         : {(~correct).all(axis=0).mean():.4f}")

section("1.5 Ablasi: Ensemble tanpa satu anggota (bobot sama)")
full_f1 = chosen["f1"]
ab = []
for out in names:
    rest = {n: members[n] for n in names if n != out}
    f1_wo = f1_score(y_rep, Ensemble(rest, voting="hard").predict(X_rep), average="macro")
    ab.append({"Tanpa": out, "Macro F1": f1_wo, "Selisih terhadap Ensemble penuh": f1_wo - full_f1})
print(pd.DataFrame(ab).set_index("Tanpa").round(4).to_string())

section("1.6 Soft voting lawan hard voting")
sv = []
for label, ens_w in [("bobot sama", None), ("bobot Macro F1", weights)]:
    hard = Ensemble(members, weights=ens_w, voting="hard").predict(X_rep)
    soft = Ensemble(members, weights=ens_w, voting="soft").predict(X_rep)
    sv.append({"Bobot": label, "Hard Macro F1": f1_score(y_rep, hard, average="macro"),
               "Soft Macro F1": f1_score(y_rep, soft, average="macro"),
               "Hard Accuracy": accuracy_score(y_rep, hard), "Soft Accuracy": accuracy_score(y_rep, soft)})
print(pd.DataFrame(sv).set_index("Bobot").round(4).to_string())

section("1.7 Ensemble scikit-learn (anggota dengan konfigurasi yang disamakan)")


def fit_sklearn_mlp():
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation="tanh", solver="sgd", momentum=0.9,
                        nesterovs_momentum=False, learning_rate_init=0.01, alpha=0.001 * len(X_train),
                        batch_size=64, random_state=SEED)
    best_f1, best_model, wait = -1.0, None, 0
    for _ in range(100):
        mlp.partial_fit(X_train, y_train, classes=[0, 1, 2])
        f1 = f1_score(y_val_all, mlp.predict(X_val_all), average="macro")
        if f1 > best_f1:
            best_f1, best_model, wait = f1, copy.deepcopy(mlp), 0
        else:
            wait += 1
            if wait >= 15:
                break
    return best_model


sk_builders = {
    "ANN": fit_sklearn_mlp,
    "LR": lambda: SkLogisticRegression(max_iter=1000, random_state=SEED).fit(X_train, y_train),
    "RF": lambda: RandomForestClassifier(n_estimators=25, min_samples_split=10, random_state=SEED).fit(X_train, y_train),
    "DTL": lambda: DecisionTreeClassifier(max_depth=8, min_samples_split=30, random_state=SEED).fit(X_train, y_train),
    "KNN": lambda: KNeighborsClassifier(n_neighbors=21, metric="manhattan").fit(X_train, y_train),
}
sk_members, sk_fit, sk_rows = {}, {}, []
for name, build in sk_builders.items():
    t0 = time.time()
    sk_members[name] = build()
    sk_fit[name] = time.time() - t0
    a, f = acc_f1(y_rep, sk_members[name].predict(X_rep))
    sk_rows.append({"Model": name, "Accuracy sklearn": a, "Macro F1 sklearn": f,
                    "Macro F1 scratch": members_df.loc[name, "Macro F1"], "Waktu latih sklearn (s)": sk_fit[name]})
print(pd.DataFrame(sk_rows).set_index("Model").round(4).to_string())


def sk_vote(models, X, voting):
    if voting == "hard":
        preds = np.array([m.predict(X) for m in models.values()])
        return np.array([np.bincount(preds[:, i], minlength=3).argmax() for i in range(preds.shape[1])])
    return np.mean([m.predict_proba(X) for m in models.values()], axis=0).argmax(axis=1)


vote_rows = []
for voting in ["hard", "soft"]:
    t0 = time.time()
    pred = sk_vote(sk_members, X_rep, voting)
    ms = (time.time() - t0) * 1000
    a, f = acc_f1(y_rep, pred)
    vote_rows.append({"Ensemble": f"sklearn {voting} voting", "Accuracy": a, "Macro F1": f, "Inferensi (ms)": ms})
vote_rows.append({"Ensemble": "scratch hard, bobot sama", "Accuracy": chosen["acc"], "Macro F1": chosen["f1"], "Inferensi (ms)": chosen["ms"]})
print()
print(pd.DataFrame(vote_rows).set_index("Ensemble").round(4).to_string())
print(f"\nTotal waktu latih anggota sklearn : {sum(sk_fit.values()):.2f} detik")
sk_pred = sk_vote(sk_members, X_rep, "hard")
print("\nConfusion matrix Ensemble sklearn (hard):\n", confusion_matrix(y_rep, sk_pred))
