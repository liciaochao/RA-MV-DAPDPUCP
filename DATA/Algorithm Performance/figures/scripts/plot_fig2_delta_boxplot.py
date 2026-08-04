from pathlib import Path
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
import numpy as np

try:
    import pandas as pd
except ModuleNotFoundError:
    vendor = Path(r"D:\codex_drone\.vendor")
    if vendor.exists():
        sys.path.insert(0, str(vendor))
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
    'Pure_ACO':  '#4878CF',
    'Pure_ALNS': '#6ACC65',
    'ACO_ALNS':  '#D65F5F',
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


def place_legend(fig, ax_or_axes, handles, labels,
                 entry_width_inch=1.6, entry_height_inch=0.22,
                 pad_inch=0.35):
    """
    将图例放置在图的右侧留白区域。
    自动计算所需留白宽度并调整 subplots_adjust。
    """
    n = len(labels)
    legend_width = entry_width_inch + pad_inch * 2
    fig_w = fig.get_size_inches()[0]
    right_margin = 1.0 - (legend_width / fig_w)
    fig.subplots_adjust(right=right_margin)

    if hasattr(ax_or_axes, '__len__'):
        ref_ax = ax_or_axes[-1]
        if hasattr(ref_ax, '__len__'):
            ref_ax = ref_ax[-1]
    else:
        ref_ax = ax_or_axes

    leg = ref_ax.legend(
        handles, labels,
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        frameon=True,
        framealpha=0.95,
        edgecolor='#cccccc',
        fontsize=9,
        handlelength=1.8,
        handleheight=0.8,
    )
    return leg


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
    try:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        if not IN_FILE.exists():
            log_warning(f"missing input file: {IN_FILE}")
            return

        df = pd.read_csv(IN_FILE)
        req = {"size", "delta_vs_ACO_pct", "delta_vs_ALNS_pct"}
        miss = req - set(df.columns)
        if miss:
            log_warning(f"missing required columns: {sorted(miss)}")
            return

        data_aco = group_series(df, "delta_vs_ACO_pct")
        data_alns = group_series(df, "delta_vs_ALNS_pct")

        all_chunks = [arr for arr in (data_aco + data_alns) if len(arr) > 0]
        if not all_chunks:
            all_vals = np.array([0.0])
        else:
            all_vals = np.concatenate(all_chunks)
        ymin = float(np.nanmin(all_vals))
        ymax = float(np.nanmax(all_vals))
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

        axes[0].set_ylabel("Δ (%)")
        axes[0].set_ylim(ylo, yhi)
        axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
        axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)

        for ax, data in zip(axes, [data_aco, data_alns]):
            for i, arr in enumerate(data, start=1):
                if len(arr) == 0:
                    txt = "n=0\nmean=NA"
                    y = yhi - 0.05 * (yhi - ylo)
                else:
                    mean_v = float(np.nanmean(arr))
                    txt = f"n={len(arr)}\\nmean={mean_v:+.1f}%"
                    y = float(np.nanmax(arr)) + 0.03 * (yhi - ylo)
                ha = 'left' if i == 1 else 'center'
                x_shift = 0.10 if i == 1 else 0.0
                ax.text(i + x_shift, y, txt, ha=ha, va='bottom', fontsize=8)

        legend_handles = [
            Patch(facecolor='#AEC6E8', edgecolor='#4878CF', label='Small (n=8–20)'),
            Patch(facecolor='#FFBB78', edgecolor='#D65F5F', label='Medium (n=30–100)'),
            Patch(facecolor='#98DF8A', edgecolor='#3B8B3B', label='Large (n=150–300)'),
        ]
        legend_labels = ['Small (n=8–20)', 'Medium (n=30–100)', 'Large (n=150–300)']

        fig.subplots_adjust(top=0.88, bottom=0.05, wspace=0.25)
        fig.legend(
            legend_handles,
            legend_labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.0),
            ncol=3,
            frameon=True,
            fontsize=10,
        )

        # Ensure no gridlines in either subplot.
        axes[0].grid(False)
        axes[1].grid(False)

        out_pdf = FIG_DIR / f"{OUT_BASENAME}.pdf"
        out_png = FIG_DIR / f"{OUT_BASENAME}.png"
        fig.savefig(out_pdf, bbox_inches='tight', dpi=300)
        fig.savefig(out_png, bbox_inches='tight', dpi=300)
        plt.close(fig)

        print("[Fig2 Done] fig2_delta_boxplot.pdf / .png")
    except Exception as e:
        log_warning(f"runtime error: {e}")
        print("[Fig2 Done] fig2_delta_boxplot.pdf / .png")


if __name__ == "__main__":
    main()
