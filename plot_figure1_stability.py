import pandas as pd
import matplotlib.pyplot as plt

# =========================
# LOAD RESULTS
# =========================
df = pd.read_csv("results_summary.csv")

# aggregate across folds
agg = (
    df.groupby("representation")[["stability_mean", "stability_std"]]
    .mean()
    .reset_index()
)

# order representations explicitly
order = [
    "Point_Euclidean",
    "Point_Angular",
]

agg["representation"] = pd.Categorical(
    agg["representation"], categories=order, ordered=True
)
agg = agg.sort_values("representation")

# =========================
# PLOT
# =========================
plt.figure(figsize=(6, 4))

plt.bar(
    agg["representation"],
    agg["stability_mean"],
    yerr=agg["stability_std"],
    capsize=6
)

plt.ylabel("Decision Stability")
plt.ylim(0, 1.05)
plt.xticks(rotation=20)

plt.tight_layout()
plt.savefig("figure1_stability_comparison.png", dpi=300)
plt.savefig("figure1_stability_comparison.pdf")
plt.show()
