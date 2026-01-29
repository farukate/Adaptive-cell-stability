import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score

# =========================
# CONFIG
# =========================
DATA_PATH = "data.csv"
N_SPLITS = 5
K = 7
N_PERTURB = 50
SIGMA = 0.05
RANDOM_STATE = 42

OUTPUT_CSV = "results_summary.csv"

np.random.seed(RANDOM_STATE)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["id", "diagnosis"]).values
y = (df["diagnosis"] == "M").astype(int).values

# =========================
# HELPERS
# =========================
def l2_normalize(X, eps=1e-12):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < eps, eps, norms)
    return X / norms


def knn_predict(X_train, y_train, X_test, metric="euclidean"):
    nn = NearestNeighbors(n_neighbors=K, metric=metric)
    nn.fit(X_train)
    neigh_idx = nn.kneighbors(X_test, return_distance=False)
    return np.array([np.round(y_train[idx].mean()).astype(int) for idx in neigh_idx])


def decision_stability(model_fn, X_train, y_train, x):
    y0 = model_fn(X_train, y_train, x.reshape(1, -1))[0]
    same = 0
    for _ in range(N_PERTURB):
        eps = np.random.normal(0, SIGMA, size=x.shape)
        x_p = x + eps
        x_p = x_p / max(np.linalg.norm(x_p), 1e-12)
        yp = model_fn(X_train, y_train, x_p.reshape(1, -1))[0]
        same += int(yp == y0)
    return same / N_PERTURB

# =========================
# EXPERIMENT
# =========================
results = []

skf = StratifiedKFold(
    n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
)

for fold, (tr, te) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold}")

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[tr])
    Xte = scaler.transform(X[te])

    ytr, yte = y[tr], y[te]

    # drop zero-variance features
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

        stabilities = []
        for x in A_te[:30]:  # sample 30 points for speed
            s = decision_stability(
                lambda Xtr, ytr, xt: knn_predict(Xtr, ytr, xt, metric),
                A_tr, ytr, x
            )
            stabilities.append(s)

        results.append({
            "fold": fold,
            "representation": name,
            "accuracy_mean": acc,
            "accuracy_std": 0.0,  # single estimate per fold
            "stability_mean": float(np.mean(stabilities)),
            "stability_std": float(np.std(stabilities)),
            "num_cells": np.nan  # not applicable for point-based models
        })

# =========================
# SAVE RESULTS
# =========================
res_df = pd.DataFrame(results)
res_df.to_csv(OUTPUT_CSV, index=False)

print("\n=== SUMMARY (mean over folds) ===")
print(
    res_df
    .groupby("representation")[[
        "accuracy_mean",
        "stability_mean",
        "stability_std"
    ]]
    .mean()
)

print(f"\nSaved detailed results to: {OUTPUT_CSV}")
