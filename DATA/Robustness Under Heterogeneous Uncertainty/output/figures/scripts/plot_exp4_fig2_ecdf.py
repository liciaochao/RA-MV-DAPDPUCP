import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

matplotlib.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

SCENARIO_ORDER = ["alpha", "beta", "gamma", "delta", "epsilon"]
SCENARIO_LABEL = {
    "alpha":   r"$\gamma_1$",
    "beta":    r"$\gamma_2$",
    "gamma":   r"$\gamma_3$",
    "delta":   r"$\gamma_4$",
    "epsilon": r"$\gamma_5$",
}
SCENARIO_COLOR = {
    "alpha": "#2166AC",
    "beta": "#74ADD1",
    "gamma": "#ABD9E9",
    "delta": "#F46D43",
    "epsilon": "#A50026",
}
SIZE_ORDER = ["Small", "Medium", "Large"]
SIZE_LABEL = {
    "Small": "Small (n=8-20)",
    "Medium": "Medium (n=30-100)",
    "Large": "Large (n=150-300)",
}

DATA_DIR = "D:/experiment/experiment_4/output/"
OUT_DIR = "D:/experiment/experiment_4/output/figures/"

df = pd.read_csv(f"{DATA_DIR}exp4_main_table.csv")

fig, axes = plt.subplots(1, 3, figsize=(11, 5.5))
plt.subplots_adjust(wspace=0.32, left=0.07, right=0.82, top=0.88, bottom=0.10)

for col, size in enumerate(SIZE_ORDER):
    ax = axes[col]
    sub = df[df["size"] == size]

    for sc in SCENARIO_ORDER:
        vals = sub[sub["scenario"] == sc]["avg_actual_cost"].dropna().values
        if len(vals) < 3:
            continue

        vals_s = np.sort(vals)
        n = len(vals_s)
        ecdf_y = np.arange(1, n + 1) / n

        main_lw = 2.0 if sc == "alpha" else 1.6
        step_handle = ax.step(
            vals_s,
            ecdf_y,
            color=SCENARIO_COLOR[sc],
            linewidth=main_lw,
            where="post",
            alpha=0.95,
            zorder=4,
            label=f"{SCENARIO_LABEL[sc]}  P90={np.percentile(vals, 90):.1f}",
        )[0]

        kde = gaussian_kde(vals, bw_method="silverman")
        x_range = np.linspace(vals.min() * 0.95, vals.max() * 1.05, 300)
        pdf = kde(x_range)
        cdf = np.cumsum(pdf) / np.sum(pdf)

        ax.plot(
            x_range,
            cdf,
            color=SCENARIO_COLOR[sc],
            linewidth=0.9,
            linestyle="-",
            alpha=0.40,
            zorder=2,
        )

    ax.axhline(0.50, color="#888888", lw=0.7, linestyle="--", alpha=0.7, zorder=1)
    ax.axhline(0.90, color="#888888", lw=0.7, linestyle=":", alpha=0.7, zorder=1)
    xmax = sub["avg_actual_cost"].max()
    ax.text(
        xmax * 1.01,
        0.50,
        "P50",
        fontsize=7.5,
        color="#888888",
        va="center",
        ha="left",
    )
    ax.text(
        xmax * 1.01,
        0.90,
        "P90",
        fontsize=7.5,
        color="#888888",
        va="center",
        ha="left",
    )

    ax.set_xlabel("Avg. actual cost", fontsize=10)
    if col == 0:
        ax.set_ylabel("Cumulative probability", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(False)
    ax.set_axisbelow(False)

# ---- 淇濆瓨鍓嶅畬鏁村鐞嗛『搴?----
# 鍏抽棴瀛愬浘鑷韩鍥句緥
for ax in axes.flat:
    lgd = ax.get_legend()
    if lgd:
        lgd.remove()

# 璋冩暣鏁翠綋甯冨眬
fig.tight_layout(rect=[0, 0.08, 1, 0.92])

# 椤剁妯帓鍥句緥
handles, labels = None, None
for ax in axes.flat:
    h, l = ax.get_legend_handles_labels()
    if h:
        handles, labels = h, l
        break
if handles:
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(labels),
        frameon=True,
        fontsize=10,
    )

# 搴曠鍒嗙粍鏍囬锛堝姩鎬佸眳涓級
fig.canvas.draw()
group_labels = ["Small (n=8–20)", "Medium (n=30–100)", "Large (n=150–300)"]
num_cols = 3
for col in range(num_cols):
    ax = axes[0, col] if getattr(axes, "ndim", 1) == 2 else axes[col]
    pos = ax.get_position()
    x_center = (pos.x0 + pos.x1) / 2
    fig.text(
        x_center, 0.02, group_labels[col],
        ha="center", va="bottom", fontsize=11, fontweight="bold"
    )

fig.savefig(r"D:\experiment\experiment_4\output\figures\exp4_fig2_ecdf.pdf",
            bbox_inches="tight", dpi=300)
fig.savefig(r"D:\experiment\experiment_4\output\figures\exp4_fig2_ecdf.png",
            bbox_inches="tight", dpi=300)
plt.close()
print("[Fig2 Done] exp4_fig2_ecdf.pdf / .png")
