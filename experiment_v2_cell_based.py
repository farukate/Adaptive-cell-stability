import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# =========================
# CONFIG
# =========================
DATA_PATH = "data.csv"
N_SPLITS = 5
KNN_K = 7
N_PERTURB = 50
SIGMA = 0.05

N_CELLS_INIT = 50
TAU_N = 8        # occupancy threshold
TAU_H = 0.05     # entropy threshold

RANDOM_STATE = 42
OUTPUT_CSV = "results_cell_based.csv"

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
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < eps, eps, n)
    return X / n

def entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))

def knn_predict(Xtr, ytr, Xte, metric):
    nn = NearestNeighbors(n_neighbors=KNN_K, metric=metric)
    nn.fit(Xtr)
    idx = nn.kneighbors(Xte, return_distance=False)
    return np.array([int(np.round(ytr[i].mean())) for i in idx])

def decision_stability(Xc, yc, x, metric):
    base = knn_predict(Xc, yc, x.reshape(1, -1), metric)[0]
    same = 0
    for _ in range(N_PERTURB):
        xp = x + np.random.normal(0, SIGMA, size=x.shape)
        xp = xp / max(np.linalg.norm(xp), 1e-12)
        yp = knn_predict(Xc, yc, xp.reshape(1, -1), metric)[0]
        same += int(yp == base)
    return same / N_PERTURB

# =========================
# ADAPTIVE REFINEMENT
# =========================
def refine_cells(X, y, centers, labels, cid):
    new_centers = []
    new_labels = []

    for i, (c, lab) in enumerate(zip(centers, labels)):
        idx = np.where(cid == i)[0]

        if len(idx) == 0:
            new_centers.append(c)
            new_labels.append(lab)
            continue

        occ = len(idx)
        p = y[idx].mean()
        H = entropy(p)

        if occ > TAU_N or H > TAU_H:
            pca = PCA(n_components=1)
            Xp = pca.fit_transform(X[idx])
            split = Xp[:, 0] > np.median(Xp[:, 0])

            for mask in [split, ~split]:
                if mask.sum() == 0:
                    continue
                center = X[idx][mask].mean(axis=0)
                center = center / max(np.linalg.norm(center), 1e-12)
                new_centers.append(center)
                new_labels.append(int(np.round(y[idx][mask].mean())))
        else:
            new_centers.append(c)
            new_labels.append(lab)

    return np.array(new_centers), np.array(new_labels)

# =========================
# EXPERIMENT
# =========================
results = []

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

for fold, (tr, te) in enumerate(skf.split(X, y), 1):
    print(f"\nFold {fold}")

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[tr])
    Xte = scaler.transform(X[te])
    ytr, yte = y[tr], y[te]

    var = np.nanvar(Xtr, axis=0)
    keep = var > 1e-12
    Xtr, Xte = Xtr[:, keep], Xte[:, keep]

    Xtr = l2_normalize(Xtr)
    Xte = l2_normalize(Xte)

    # ---------- INITIAL CELLS (Uniform) ----------
    km = KMeans(n_clusters=N_CELLS_INIT, random_state=RANDOM_STATE, n_init=10)
    cid = km.fit_predict(Xtr)
    centers = km.cluster_centers_
    labels = np.array([
        int(np.round(ytr[cid == i].mean())) if np.any(cid == i) else 0
        for i in range(N_CELLS_INIT)
    ])

    preds_u = knn_predict(centers, labels, Xte, metric="cosine")
    acc_u = accuracy_score(yte, preds_u)
    stab_u = np.mean([
        decision_stability(centers, labels, x, "cosine") for x in Xte[:30]
    ])

    results.append({
        "fold": fold,
        "representation": "Cell_Uniform",
        "num_cells": len(centers),
        "accuracy_mean": acc_u,
        "stability_mean": stab_u
    })

    # ---------- ADAPTIVE REFINEMENT ----------
    rc, rl = refine_cells(Xtr, ytr, centers, labels, cid)

    preds_a = knn_predict(rc, rl, Xte, metric="cosine")
    acc_a = accuracy_score(yte, preds_a)
    stab_a = np.mean([
        decision_stability(rc, rl, x, "cosine") for x in Xte[:30]
    ])

    results.append({
        "fold": fold,
        "representation": "Cell_Adaptive",
        "num_cells": len(rc),
        "accuracy_mean": acc_a,
        "stability_mean": stab_a
    })

# =========================
# SAVE RESULTS
# =========================
df_res = pd.DataFrame(results)
df_res.to_csv(OUTPUT_CSV, index=False)

print("\n=== SUMMARY (mean over folds) ===")
print(
    df_res.groupby("representation")[[
        "num_cells",
        "accuracy_mean",
        "stability_mean"
    ]].mean()
)

print(f"\nSaved detailed results to: {OUTPUT_CSV}")
