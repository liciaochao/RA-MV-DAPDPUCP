from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    'font.family':        'Times New Roman',
    'font.size':          11,
    'axes.titlesize':     12,
    'axes.labelsize':     11,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'legend.fontsize':    10,
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth':     0.8,
    'grid.linewidth':     0.4,
    'lines.linewidth':    1.2,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
})

# 三算法配色（色盲友好）
COLOR = {
    'Pure_ACO':  '#4878CF',   # 蓝
    'Pure_ALNS': '#6ACC65',   # 绿
    'ACO_ALNS':  '#D65F5F',   # 红（重点算法）
}
LABEL = {
    'Pure_ACO':  'Pure ACO',
    'Pure_ALNS': 'Pure ALNS',
    'ACO_ALNS':  'ACO-ALNS',
}
HATCH = {
    'Pure_ACO':  '//',
    'Pure_ALNS': '\\\\',
    'ACO_ALNS':  '',
}
SIZE_ORDER  = ['Small', 'Medium', 'Large']
ALG_ORDER   = ['Pure_ACO', 'Pure_ALNS', 'ACO_ALNS']

SCRIPT_NAME = "plot_fig2_delta_boxplot.py"
BASE_DIR = Path(r"D:\experiment\experiment_1\output")
IN_FILE = BASE_DIR / "exp1_main_table.csv"
FIG_DIR = BASE_DIR / "figures"
WARN_LOG = FIG_DIR / "warnings.log"
OUT_BASENAME = "fig2_delta_boxplot"


def log_warning(msg: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(WARN_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{SCRIPT_NAME}] {msg}\n")


def group_series(df, col):
    out = []
    for s in SIZE_ORDER:
        arr = pd.to_numeric(df.loc[df["size"] == s, col], errors="coerce").dropna().values
        out.append(arr)
    return out


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if not IN_FILE.exists():
        log_warning(f"missing input file: {IN_FILE}")
        return

    df = pd.read_csv(IN_FILE)
    required = {"size", "delta_vs_ACO_pct", "delta_vs_ALNS_pct"}
    missing = required - set(df.columns)
    if missing:
        log_warning(f"missing required columns in exp1_main_table.csv: {sorted(missing)}")
        return

    data_aco = group_series(df, "delta_vs_ACO_pct")
    data_alns = group_series(df, "delta_vs_ALNS_pct")

    all_vals = np.concatenate([np.concatenate([d for d in data_aco if len(d) > 0]),
                               np.concatenate([d for d in data_alns if len(d) > 0])]) if any(len(d) > 0 for d in data_aco + data_alns) else np.array([0.0])
    ymin = np.nanmin(all_vals)
    ymax = np.nanmax(all_vals)
    pad = max((ymax - ymin) * 0.08, 0.5)
    ylo, yhi = ymin - pad, ymax + pad

    fig, axes = plt.subplots(1, 2, figsize=(7, 4.5), sharey=True)
    box_colors = ['#AEC6E8', '#FFBB78', '#98DF8A']

    style = dict(
        patch_artist=True,
        whis=1.5,
        medianprops=dict(color='white', linewidth=1.5),
        boxprops=dict(linewidth=0.8, edgecolor='black'),
        whiskerprops=dict(linewidth=0.8, color='black'),
        capprops=dict(linewidth=0.8, color='black'),
        flierprops=dict(marker='o', markerfacecolor='none', markeredgecolor='black', markersize=4, linestyle='none')
    )

    bp1 = axes[0].boxplot(data_aco, tick_labels=SIZE_ORDER, **style)
    bp2 = axes[1].boxplot(data_alns, tick_labels=SIZE_ORDER, **style)

    for bp in [bp1, bp2]:
        for patch, c in zip(bp['boxes'], box_colors):
            patch.set_facecolor(c)

    axes[0].set_title("(a) ACO-ALNS vs. Pure ACO")
    axes[1].set_title("(b) ACO-ALNS vs. Pure ALNS")
    axes[0].set_ylabel("Δ (%)")
    axes[0].set_ylim(ylo, yhi)
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].grid(axis='y', alpha=0.35)
    axes[1].grid(axis='y', alpha=0.35)

    for ax, data in zip(axes, [data_aco, data_alns]):
        for i, arr in enumerate(data, start=1):
            if len(arr) == 0:
                txt = "n=0\nmean=NA"
                y = yhi - 0.05 * (yhi - ylo)
            else:
                mean_v = np.nanmean(arr)
                txt = f"n={len(arr)}\nmean={mean_v:+.1f}%"
                y = np.nanmax(arr) + 0.03 * (yhi - ylo)
            ax.text(i, y, txt, ha='center', va='bottom', fontsize=8)

    fig.text(0.5, 0.01, "Note: Positive Δ indicates ACO-ALNS achieves lower objective value.", ha='center', va='bottom', fontsize=9)
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    out_pdf = FIG_DIR / f"{OUT_BASENAME}.pdf"
    out_png = FIG_DIR / f"{OUT_BASENAME}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=600)
    plt.close(fig)

    print(f"[Task B Done] 脚本已保存: {Path(__file__).resolve()}")
    print(f"[Task B Done] 图片已保存: {out_pdf}  +  {out_png}")


if __name__ == "__main__":
    main()
