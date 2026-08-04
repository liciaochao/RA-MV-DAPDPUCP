import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
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
    "alpha": r"$\alpha$",
    "beta": r"$\beta$",
    "gamma": r"$\gamma$",
    "delta": r"$\delta$",
    "epsilon": r"$\varepsilon$",
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


def plot_confidence_ellipse(ax, x, y, color, n_std=1.5, alpha=0.15):
    if len(x) < 3:
        return None, None
    if np.std(x) < 1e-9:
        y_mean = np.mean(y)
        y_std = np.std(y) * n_std
        ax.plot(
            [np.mean(x), np.mean(x)],
            [y_mean - y_std, y_mean + y_std],
            color=color,
            lw=1.5,
            linestyle="--",
            alpha=0.6,
            zorder=2,
        )
        return np.mean(x), y_mean
    if np.std(y) < 1e-9:
        x_mean = np.mean(x)
        x_std = np.std(x) * n_std
        ax.plot(
            [x_mean - x_std, x_mean + x_std],
            [np.mean(y), np.mean(y)],
            color=color,
            lw=1.5,
            linestyle="--",
            alpha=0.6,
            zorder=2,
        )
        return x_mean, np.mean(y)

    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(np.maximum(vals, 0.0))
    ellipse = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=w,
        height=h,
        angle=theta,
        facecolor=color,
        alpha=alpha,
        edgecolor=color,
        linewidth=1.2,
        linestyle="--",
        zorder=2,
    )
    ax.add_patch(ellipse)
    return np.mean(x), np.mean(y)


df = pd.read_csv(f"{DATA_DIR}exp4_main_table.csv")

fig, axes = plt.subplots(1, 3, figsize=(11, 4.5))
plt.subplots_adjust(wspace=0.35, left=0.07, right=0.97, top=0.88, bottom=0.15)

for col, size in enumerate(SIZE_ORDER):
    ax = axes[col]
    sub = df[df["size"] == size]

    for sc in SCENARIO_ORDER:
        sc_sub = sub[sub["scenario"] == sc]
        x = sc_sub["avg_fail"].values
        y = sc_sub["avg_actual_cost"].values
        sizes = 30 + sc_sub["cost_std"].values * 2
        ax.scatter(
            x,
            y,
            c=SCENARIO_COLOR[sc],
            s=sizes,
            alpha=0.70,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
            label=SCENARIO_LABEL[sc],
        )

        if len(x) >= 3:
            cx, cy = plot_confidence_ellipse(ax, x, y, SCENARIO_COLOR[sc])
            if cx is not None and cy is not None:
                ax.text(
                    cx,
                    cy,
                    SCENARIO_LABEL[sc],
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color=SCENARIO_COLOR[sc],
                    zorder=5,
                )

    ax.set_xlabel("Avg. failed customers", fontsize=10)
    if col == 0:
        ax.set_ylabel("Avg. actual cost", fontsize=10)
    ax.set_title(f"({'abc'[col]}) {SIZE_LABEL[size]}", fontsize=10, pad=5)
    ax.grid(False)
    ax.set_axisbelow(False)

legend_handles = [
    plt.scatter(
        [],
        [],
        c=SCENARIO_COLOR[sc],
        s=60,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.4,
        label=SCENARIO_LABEL[sc],
    )
    for sc in SCENARIO_ORDER
]

axes[2].legend(
    handles=legend_handles,
    labels=[h.get_label() for h in legend_handles],
    loc="lower right",
    frameon=True,
    framealpha=0.95,
    edgecolor="#cccccc",
    fontsize=9.5,
    handletextpad=0.4,
    borderpad=0.6,
    labelspacing=0.5,
    scatteryoffsets=[0.5],
)

for ext in ["pdf", "png"]:
    fig.savefig(f"{OUT_DIR}exp4_fig3_bicriteria_scatter.{ext}")
plt.close()
print("[Fig3 Done] 图例已移至第三子图右下角，右侧空白已回收")
