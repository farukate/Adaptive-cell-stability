"""
Experiment V3
Cell-based spherical representation with adaptive refinement (OFF vs ON).

This script compares uniform cell-based inference with ambiguity-triggered
adaptive refinement, measuring decision stability and accuracy under
identical inference rules.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# =========================
# CONFIGURATION
# =========================
DATA_PATH = "data.csv"   # dataset must be provided by the user
N_SPLITS = 5
KNN_K = 7
N_PERTURB = 50
SIGMA = 0.05

N_CELLS_INIT = 50
TAU_N = 8        # occupancy threshold
TAU_H = 0.05     # entropy threshold

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

def entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))

def knn_predict(X_train, y_train, X_test, metric):
    nn = NearestNeighbors(n_neighbors=KNN_K, metric=metric)
    nn.fit(X_train)
    idx = nn.kneighbors(X_test, return_distance=False)
    return np.array([int(np.round(y_train[i].mean())) for i in idx])

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
            # split cell along dominant variance direction
            pca = PCA(n_components=1)
            proj = pca.fit_transform(X[idx])
            split = proj[:, 0] > np.median(proj[:, 0])

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
    Xtr = l2_normalize(Xtr)
    Xte = l2_normalize(Xte)

    # =========================
    # INITIAL CELLS (OFF)
    # =========================
    km = KMeans(
        n_clusters=N_CELLS_INIT,
        random_state=RANDOM_STATE,
        n_init=10
    )
    cid = km.fit_predict(Xtr)
    centers = km.cluster_centers_

    labels = np.array([
        int(np.round(ytr[cid == i].mean()))
        if np.any(cid == i) else 0
        for i in range(N_CELLS_INIT)
    ])

    # OFF
    preds_off = knn_predict(centers, labels, Xte, metric="cosine")
    acc_off = accuracy_score(yte, preds_off)
    stab_off = np.mean([
        decision_stability(centers, labels, x, "cosine")
        for x in Xte[:30]
    ])

    # =========================
    # REFINEMENT (ON)
    # =========================
    r_centers, r_labels = refine_cells(Xtr, ytr, centers, labels, cid)

    preds_on = knn_predict(r_centers, r_labels, Xte, metric="cosine")
    acc_on = accuracy_score(yte, preds_on)
    stab_on = np.mean([
        decision_stability(r_centers, r_labels, x, "cosine")
        for x in Xte[:30]
    ])

    results.append({
        "Fold": fold,
        "Acc_OFF": acc_off,
        "Stab_OFF": stab_off,
        "Acc_ON": acc_on,
        "Stab_ON": stab_on,
        "Cells_OFF": len(centers),
        "Cells_ON": len(r_centers),
    })

# =========================
# SUMMARY
# =========================
df_res = pd.DataFrame(results)

print("\n=== SUMMARY (mean over folds) ===")
print(df_res.mean())
