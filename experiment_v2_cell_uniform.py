"""
Experiment V2
Uniform cell-based spherical representation (k-means, k=50), refinement OFF.

This script evaluates decision stability when neighborhood inference
is performed over cell representatives rather than individual samples,
using an identical angular distance metric and decision rule.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# =========================
# CONFIGURATION
# =========================
DATA_PATH = "data.csv"   # dataset must be provided by the user
N_SPLITS = 5
KNN_K = 7
N_PERTURB = 50
SIGMA = 0.05

N_CELLS = 50
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

def decision_stability_cell(Xc, yc, x, metric):
    base = knn_predict(Xc, yc, x.reshape(1, -1), metric)[0]
    same = 0
    for _ in range(N_PERTURB):
        xp = x + np.random.normal(0, SIGMA, size=x.shape)
        xp = xp / max(np.linalg.norm(xp), 1e-12)
        yp = knn_predict(Xc, yc, xp.reshape(1, -1), metric)[0]
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

    # spherical embedding
    Xtr_l2 = l2_normalize(Xtr)
    Xte_l2 = l2_normalize(Xte)

    # =========================
    # CELL CONSTRUCTION
    # =========================
    kmeans = KMeans(
        n_clusters=N_CELLS,
        random_state=RANDOM_STATE,
        n_init=10
    )
    cell_id = kmeans.fit_predict(Xtr_l2)
    cell_centers = kmeans.cluster_centers_

    # cell labels via majority vote
    cell_labels = np.array([
        int(np.round(ytr[cell_id == i].mean()))
        if np.any(cell_id == i) else 0
        for i in range(N_CELLS)
    ])

    # =========================
    # TEST
    # =========================
    preds = knn_predict(
        cell_centers,
        cell_labels,
        Xte_l2,
        metric="cosine"
    )
    acc = accuracy_score(yte, preds)

    stabilities = [
        decision_stability_cell(
            cell_centers,
            cell_labels,
            x,
            metric="cosine"
        )
        for x in Xte_l2[:30]
    ]

    results.append({
        "Fold": fold,
        "Model": "Cell_Angular_k50",
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
