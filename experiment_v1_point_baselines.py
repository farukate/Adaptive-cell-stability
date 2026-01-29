"""
Experiment V1
Point-based neighborhood baselines:
- Euclidean KNN
- Angular (cosine) KNN

This script evaluates decision stability under perturbations
using identical decision rules and neighborhood sizes.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score

# =========================
# CONFIGURATION
# =========================
DATA_PATH = "data.csv"   # dataset must be provided by the user
N_SPLITS = 5
KNN_K = 7
N_PERTURB = 50
SIGMA = 0.05
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["id", "diagnosis"]).values
y = (df["diagnosis"] == "M").astype(int).values

# =========================
# HELPER FUNCTIONS
# =========================
def l2_normalize(X, eps=1e-12):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < eps, eps, norms)
    return X / norms

def knn_predict(X_train, y_train, X_test, metric):
    nn = NearestNeighbors(n_neighbors=KNN_K, metric=metric)
    nn.fit(X_train)
    idx = nn.kneighbors(X_test, return_distance=False)
    return np.array([int(np.round(y_train[i].mean())) for i in idx])

def decision_stability(model_fn, Xtr, ytr, x):
    base = model_fn(Xtr, ytr, x.reshape(1, -1))[0]
    same = 0
    for _ in range(N_PERTURB):
        xp = x + np.random.normal(0, SIGMA, size=x.shape)
        xp = xp / max(np.linalg.norm(xp), 1e-12)
        yp = model_fn(Xtr, ytr, xp.reshape(1, -1))[0]
        same += int(yp == base)
    return same / N_PERTURB

# =========================
# EXPERIMENT
# =========================
results = []

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

for fold, (tr, te) in enumerate(skf.split(X, y), 1):
    print(f"\nFold {fold}")

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[tr])
    Xte = scaler.transform(X[te])
    ytr, yte = y[tr], y[te]

    # remove zero-variance features
    var = np.nanvar(Xtr, axis=0)
    keep = var > 1e-12
    Xtr = Xtr[:, keep]
    Xte = Xte[:, keep]

    # representations
    Xtr_l2 = l2_normalize(Xtr)
    Xte_l2 = l2_normalize(Xte)

    configs = {
        "Point_Euclidean": (Xtr, Xte, "euclidean"),
        "Point_Angular": (Xtr_l2, Xte_l2, "cosine"),
    }

    for name, (A_tr, A_te, metric) in configs.items():
        preds = knn_predict(A_tr, ytr, A_te, metric)
        acc = accuracy_score(yte, preds)

        stabilities = [
            decision_stability(
                lambda Xt, yt, Xq: knn_predict(Xt, yt, Xq, metric),
                A_tr,
                ytr,
                x
            )
            for x in A_te[:30]
        ]

        results.append({
            "Fold": fold,
            "Model": name,
            "Accuracy": acc,
            "Stability_Mean": np.mean(stabilities),
            "Stability_Std": np.std(stabilities),
        })

# =========================
# SUMMARY
# =========================
res_df = pd.DataFrame(results)

print("\n=== SUMMARY ===")
print(
    res_df
    .groupby("Model")[["Accuracy", "Stability_Mean", "Stability_Std"]]
    .mean()
)
