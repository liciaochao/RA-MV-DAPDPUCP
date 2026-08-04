import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

matplotlib.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
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

SCENARIO_ORDER = ["alpha", "beta", "gamma", "delta", "epsilon"]
SCENARIO_LABEL = {
    "alpha":   r"$\gamma_1$",
    "beta":    r"$\gamma_2$",
    "gamma":   r"$\gamma_3$",
    "delta":   r"$\gamma_4$",
    "epsilon": r"$\gamma_5$",
}
x_pos = np.arange(len(SCENARIO_ORDER))

df = pd.read_csv(f"{DATA_DIR}/exp4_main_table.csv")
med = df[df["size"] == "Medium"].copy()


def count_offline_ge_online(inst_name):
    sub = med[med["instance"] == inst_name]
    return int((sub["offline_obj"] >= sub["avg_actual_cost"]).sum())


seed1_insts = med[med["instance"].str.endswith("seed1")]["instance"].unique()
candidates = []
for inst in seed1_insts:
    n = count_offline_ge_online(inst)
    candidates.append({"instance": inst, "n_ge": n})

cands_df = pd.DataFrame(candidates).sort_values("n_ge", ascending=False)
top3 = cands_df.head(3)["instance"].tolist()

print("[Fig4 Instances Selected]")
for inst in top3:
    n_ge = count_offline_ge_online(inst)
    print(f"  {inst}  (offline>=online in {n_ge}/5 scenarios)")

inst_data = []
for inst in top3:
    sub = med[med["instance"] == inst]
    offline = []
    online = []
    for sc in SCENARIO_ORDER:
        row = sub[sub["scenario"] == sc]
        if len(row) > 0:
            offline.append(float(row["offline_obj"].iloc[0]))
            online.append(float(row["avg_actual_cost"].iloc[0]))
        else:
            offline.append(np.nan)
            online.append(np.nan)
    inst_data.append(
        {"instance": inst, "offline": np.array(offline), "online": np.array(online)}
    )

fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
plt.subplots_adjust(
    left=0.07,
    right=0.97,
    top=0.82,
    bottom=0.14,
    wspace=0.35,
)

for col, d in enumerate(inst_data):
    ax = axes[col]
    offline_vals = d["offline"]
    online_vals = d["online"]
    inst_name = d["instance"]
    net_vals = offline_vals - online_vals

    ax.plot(
        x_pos,
        offline_vals,
        "-s",
        color="#1F77B4",
        linewidth=2.0,
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=2.0,
        zorder=4,
        label="Offline planning cost",
    )
    ax.plot(
        x_pos,
        online_vals,
        "--D",
        color="#D62728",
        linewidth=2.0,
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=2.0,
        zorder=4,
        label="Avg. online cost",
    )

    ax.fill_between(
        x_pos,
        offline_vals,
        online_vals,
        where=(offline_vals >= online_vals),
        alpha=0.18,
        color="#2CA02C",
        interpolate=True,
    )
    ax.fill_between(
        x_pos,
        offline_vals,
        online_vals,
        where=(offline_vals < online_vals),
        alpha=0.18,
        color="#FF7F0E",
        interpolate=True,
    )

    y_range = max(np.nanmax(offline_vals), np.nanmax(online_vals)) - min(
        np.nanmin(offline_vals), np.nanmin(online_vals)
    )
    offset = y_range * 0.04 if y_range > 0 else 1.0

    for xi in range(len(x_pos)):
        ax.text(
            xi,
            offline_vals[xi] + offset,
            f"{offline_vals[xi]:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#1F77B4",
            fontweight="500",
        )
        ax.text(
            xi,
            online_vals[xi] - offset,
            f"{online_vals[xi]:.1f}",
            ha="center",
            va="top",
            fontsize=7.5,
            color="#D62728",
            fontweight="500",
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([SCENARIO_LABEL[s] for s in SCENARIO_ORDER], fontsize=10)
    ax.grid(False)
    ax.set_axisbelow(False)
    if col == 0:
        ax.set_ylabel("Cost", fontsize=10)

    parts = inst_name.split("_")
    short_name = f"{parts[1]}c-{parts[2]}v  seed {parts[3].replace('seed','')}"

    inset_ax = inset_axes(
        ax,
        width="32%",
        height="38%",
        loc="lower right",
        bbox_to_anchor=(-0.02, 0.06, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0.4,
    )

    bar_colors = ["#2CA02C" if v >= 0 else "#FF7F0E" for v in net_vals]
    inset_ax.bar(
        x_pos,
        net_vals,
        width=0.65,
        color=bar_colors,
        alpha=0.80,
        edgecolor="white",
        linewidth=0.7,
    )
    inset_ax.axhline(0, color="#333333", lw=0.7, linestyle="-", alpha=0.8)
    inset_ax.grid(False)
    inset_ax.set_axisbelow(False)

    for xi, v in enumerate(net_vals):
        if not np.isnan(v):
            va = "bottom" if v >= 0 else "top"
            ypos = v
            fmt = f"{v:+.1f}" if abs(v) < 1.0 else f"{v:+.0f}"
            inset_ax.text(
                xi, ypos, fmt, ha="center", va=va, fontsize=6.5, color="#222222"
            )

    inset_ax.set_xticks(x_pos)
    inset_ax.set_xticklabels([SCENARIO_LABEL[s] for s in SCENARIO_ORDER], fontsize=6)
    inset_ax.tick_params(axis="y", labelsize=6)
    inset_ax.set_title(
        r"offline $-$ online", fontsize=6.5, pad=2, style="italic", color="#444444"
    )
    inset_ax.spines["top"].set_visible(False)
    inset_ax.spines["right"].set_visible(False)
    inset_ax.set_facecolor("#F8F8F8")

legend_handles = [
    Line2D(
        [0],
        [0],
        color="#1F77B4",
        lw=2.0,
        ls="-",
        marker="s",
        markerfacecolor="white",
        markeredgewidth=2.0,
        markersize=7,
        label="Offline planning cost",
    ),
    Line2D(
        [0],
        [0],
        color="#D62728",
        lw=2.0,
        ls="--",
        marker="D",
        markerfacecolor="white",
        markeredgewidth=2.0,
        markersize=7,
        label="Avg. online exec. cost",
    ),
    mpatches.Patch(
        facecolor="#2CA02C", alpha=0.4, label="offline > online\n(defense investment)"
    ),
    mpatches.Patch(
        facecolor="#FF7F0E", alpha=0.4, label="online > offline\n(cost overrun)"
    ),
]
# ---- 保存前完整处理顺序 ----
# 关闭子图自身图例
for ax in axes.flat:
    lgd = ax.get_legend()
    if lgd:
        lgd.remove()

# 调整整体布局（顶部留图例，底部留分组标题）
fig.tight_layout(rect=[0, 0.08, 1, 0.88])

# 顶端横排图例
handles, labels = None, None
for ax in axes.flat:
    h, l = ax.get_legend_handles_labels()
    if h:
        handles, labels = h, l
        break
if not handles:
    handles = legend_handles
    labels = [h.get_label() for h in legend_handles]
fig.legend(
    handles, labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=len(labels),
    frameon=True,
    fontsize=10,
)

# 底端分组标题（动态居中）
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

os.makedirs(OUT_DIR, exist_ok=True)
fig.savefig(r"D:\experiment\experiment_4\output\figures\exp4_fig4_defense_evidence.pdf",
            bbox_inches="tight", dpi=300)
fig.savefig(r"D:\experiment\experiment_4\output\figures\exp4_fig4_defense_evidence.png",
            bbox_inches="tight", dpi=300)
plt.close()
print("[Fig4 Done] exp4_fig4_defense_evidence.pdf / .png")
