import pandas as pd
import matplotlib.pyplot as plt

# =========================
# LOAD RESULTS
# =========================
df = pd.read_csv("results_summary.csv")

# aggregate across folds
agg = (
    df.groupby("representation")[["accuracy_mean"]]
    .mean()
    .reset_index()
)

# order explicitly
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
    agg["accuracy_mean"]
)

plt.ylabel("Classification Accuracy")
plt.ylim(0.90, 1.0)
plt.xticks(rotation=20)

plt.tight_layout()
plt.savefig("figure2_accuracy_comparison.png", dpi=300)
plt.savefig("figure2_accuracy_comparison.pdf")
plt.show()
