import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

DATA_DIR = r"D:\experiment\experiment_4\output"
OUT_DIR = r"D:\experiment\experiment_4\output\figures"
SCRIPT_DIR = r"D:\experiment\experiment_4\output\figures\scripts"

SCENARIO_ORDER = ["alpha", "beta", "gamma", "delta", "epsilon"]
SCENARIO_LABEL = {
    "alpha": r"$\alpha$",
    "beta": r"$\beta$",
    "gamma": r"$\gamma$",
    "delta": r"$\delta$",
    "epsilon": r"$\varepsilon$",
}
SIZE_ORDER = ["Small", "Medium", "Large"]
SIZE_LABEL = {
    "Small": r"Small ($n$=8$-$20)",
    "Medium": r"Medium ($n$=30$-$100)",
    "Large": r"Large ($n$=150$-$300)",
}
SIZE_COLOR = {
    "Small": "#2166AC",
    "Medium": "#4DAC26",
    "Large": "#D6604D",
}
SEED_STYLE = {
    1: {"ls": "-", "marker": "o", "lw": 1.8},
    2: {"ls": "--", "marker": "s", "lw": 1.6},
    3: {"ls": ":", "marker": "^", "lw": 1.6},
}

df = pd.read_csv(f"{DATA_DIR}/exp4_main_table.csv")

# Auto-select representative nc/nv per size
rep_instances = {}
for size in SIZE_ORDER:
    sub = df[df["size"] == size]
    nc_col = "n_customers" if "n_customers" in sub.columns else "customers"
    nv_col = "n_vehicles" if "n_vehicles" in sub.columns else "vehicle_pairs"
    uniq = sub[["instance", nc_col, nv_col]].drop_duplicates().copy()
    median_nc = uniq[nc_col].median()
    uniq["dist"] = (uniq[nc_col] - median_nc).abs()
    best_row = uniq.sort_values(["dist", nc_col, nv_col, "instance"]).iloc[0]
    best_nc = int(best_row[nc_col])
    best_nv = int(best_row[nv_col])
    rep_instances[size] = [
        f"{size.lower()}_{best_nc}c_{best_nv}v_seed{s}" for s in [1, 2, 3]
    ]

METRICS = [
    ("avg_actual_cost", "Avg. actual cost"),
    ("avg_fail", "Avg. failed customers"),
]

LABELS = {
    (0, 0): "a",
    (0, 1): "b",
    (1, 0): "c",
    (1, 1): "d",
    (2, 0): "e",
    (2, 1): "f",
}

fig, axes = plt.subplots(3, 2, figsize=(10, 12))
plt.subplots_adjust(
    hspace=0.35, wspace=0.28, left=0.10, right=0.96, top=0.94, bottom=0.06
)

x_pos = np.arange(len(SCENARIO_ORDER))

# Base instance name per size (without trailing _seed1)
inst_base_name = {}
for size in SIZE_ORDER:
    base_inst = rep_instances[size][0]
    inst_base_name[size] = "_".join(base_inst.split("_")[:-1])

for row, size in enumerate(SIZE_ORDER):
    color = SIZE_COLOR[size]
    insts = rep_instances[size]

    for col, (metric, col_title) in enumerate(METRICS):
        ax = axes[row][col]

        ax.grid(False)
        ax.set_axisbelow(False)

        for seed_idx, inst in enumerate(insts, start=1):
            style = SEED_STYLE[seed_idx]
            vals = []
            for sc in SCENARIO_ORDER:
                row_data = df[(df["instance"] == inst) & (df["scenario"] == sc)]
                vals.append(float(row_data[metric].iloc[0]) if len(row_data) > 0 else np.nan)
            vals = np.array(vals, dtype=float)

            ax.plot(
                x_pos,
                vals,
                color=color,
                linestyle=style["ls"],
                linewidth=style["lw"],
                marker=style["marker"],
                markersize=6,
                markerfacecolor="white",
                markeredgewidth=1.8,
                label=f"seed {seed_idx}",
                zorder=3,
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels([SCENARIO_LABEL[s] for s in SCENARIO_ORDER], fontsize=10)
        ax.set_xlabel(
            inst_base_name[size],
            fontsize=9.5,
            color=color,
            fontweight="500",
            labelpad=6,
            fontfamily="Times New Roman",
        )

        if metric == "avg_fail":
            ax.set_ylim(bottom=0)

        char = LABELS[(row, col)]
        if row == 0:
            ax.set_title(f"({char}) {col_title}", fontsize=11, pad=6)
        else:
            ax.set_title(f"({char})", fontsize=11, pad=6)

        if col != 0:
            ax.tick_params(axis="y", labelsize=9.5)

legend_handles = [
    Line2D(
        [0],
        [0],
        color="gray",
        ls="-",
        lw=1.8,
        marker="o",
        markerfacecolor="white",
        markeredgewidth=1.8,
        markersize=6,
        label="seed 1",
    ),
    Line2D(
        [0],
        [0],
        color="gray",
        ls="--",
        lw=1.6,
        marker="s",
        markerfacecolor="white",
        markeredgewidth=1.8,
        markersize=6,
        label="seed 2",
    ),
    Line2D(
        [0],
        [0],
        color="gray",
        ls=":",
        lw=1.6,
        marker="^",
        markerfacecolor="white",
        markeredgewidth=1.8,
        markersize=6,
        label="seed 3",
    ),
]

axes[0][0].legend(
    handles=legend_handles,
    loc="lower right",
    frameon=True,
    framealpha=0.95,
    edgecolor="#cccccc",
    fontsize=9,
    handlelength=2.0,
    handleheight=0.85,
    borderpad=0.5,
    labelspacing=0.45,
    title="Line style",
    title_fontsize=9.5,
)

os.makedirs(OUT_DIR, exist_ok=True)
for ext in ["pdf", "png"]:
    fig.savefig(f"{OUT_DIR}/exp4_fig1_degradation_gradient.{ext}")
plt.close()
print("[Fig1 Done] 横坐标标题已改为算例名，右侧注释已删除，图例已移至第一子图右下角")
