import pandas as pd
import matplotlib.pyplot as plt

# =========================
# LOAD RESULTS
# =========================
df = pd.read_csv("results_cell_based.csv")

# split representations
df_uniform = df[df["representation"] == "Cell_Uniform"]
df_adapt   = df[df["representation"] == "Cell_Adaptive"]

# =========================
# PLOT
# =========================
plt.figure(figsize=(6, 4))

plt.scatter(
    df_uniform["num_cells"],
    df_uniform["stability_mean"],
    label="Uniform discretization",
    s=60
)

plt.scatter(
    df_adapt["num_cells"],
    df_adapt["stability_mean"],
    label="Adaptive refinement",
    s=60
)

plt.xlabel("Number of Cells")
plt.ylabel("Decision Stability")
plt.ylim(0.97, 1.0)
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figure3_cells_vs_stability.png", dpi=300)
plt.savefig("figure3_cells_vs_stability.pdf")
plt.show()
