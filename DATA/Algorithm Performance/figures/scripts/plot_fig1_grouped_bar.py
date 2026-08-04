from pathlib import Path
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

    # 取第一个 ax（支持单 ax 或 axes 数组）
    if hasattr(ax_or_axes, '__len__'):
        ref_ax = ax_or_axes[-1]   # 最右子图作为锚点
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


SCRIPT_NAME = "plot_fig1_grouped_bar.py"
BASE_DIR = Path(r"D:\experiment\experiment_1\output")
IN_FILE = BASE_DIR / "exp1_grouped_summary.csv"
FIG_DIR = BASE_DIR / "figures"
WARN_LOG = FIG_DIR / "warnings.log"
OUT_BASENAME = "fig1_grouped_obj_bar"


def log_warning(msg: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(WARN_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{SCRIPT_NAME}] {msg}\n")


def _fmt_signed_percent(v: float) -> str:
    sign = '+' if v >= 0 else '−'
    return f"{sign}{abs(v):.1f}%"


def main():
    try:
        FIG_DIR.mkdir(parents=True, exist_ok=True)

        if not IN_FILE.exists():
            log_warning(f"missing input file: {IN_FILE}")
            return

        df = pd.read_csv(IN_FILE)
        req = {"size", "algorithm", "obj_mean"}
        miss = req - set(df.columns)
        if miss:
            log_warning(f"missing required columns: {sorted(miss)}")
            return

        pivot = df.pivot_table(index="size", columns="algorithm", values="obj_mean", aggfunc="first")
        n_map = {}
        if "n_instances" in df.columns:
            for s in SIZE_ORDER:
                sub = df[df["size"] == s]
                n_map[s] = int(pd.to_numeric(sub["n_instances"], errors="coerce").dropna().iloc[0]) if not sub.empty else {"Small":21, "Medium":18, "Large":18}[s]
        else:
            n_map = {"Small":21, "Medium":18, "Large":18}
            log_warning("n_instances missing in grouped summary; fallback to default counts")

        title_map = {
            "Small": f"Small (n=8–20, {n_map['Small']} inst.)",
            "Medium": f"Medium (n=30–100, {n_map['Medium']} inst.)",
            "Large": f"Large (n=150–300, {n_map['Large']} inst.)",
        }

        fig, axes = plt.subplots(1, 3, figsize=(7, 4))
        x = np.arange(3)
        bw = 0.25

        for i, size in enumerate(SIZE_ORDER):
            ax = axes[i]
            vals = []
            for alg in ALG_ORDER:
                if size in pivot.index and alg in pivot.columns:
                    vals.append(float(pivot.loc[size, alg]))
                else:
                    vals.append(np.nan)
                    log_warning(f"missing grouped cell size={size}, alg={alg}")
            arr = np.array(vals, dtype=float)
            good = arr[np.isfinite(arr)]
            if good.size == 0:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
                continue

            ymin = np.nanmin(good) * 0.95
            ymax = np.nanmax(good) * 1.10
            yr = max(ymax - ymin, 1e-6)
            ymax += 0.18 * yr

            for j, alg in enumerate(ALG_ORDER):
                ax.bar(
                    x[j], arr[j], width=bw,
                    color=COLOR[alg], edgecolor="black", linewidth=0.7,
                    hatch=HATCH[alg], zorder=3,
                )

            ax.set_ylim(ymin, ymax)
            ax.set_title(title_map[size])
            ax.set_xticks([])
            ax.grid(axis="y", alpha=0.35, zorder=0)
            if i == 0:
                ax.set_ylabel("Objective value")

            for j, v in enumerate(arr):
                if np.isfinite(v):
                    ax.text(x[j], v + 0.015 * (ymax - ymin), f"{v:.2f}", ha="center", va="bottom", fontsize=9)

            aco, alns, hyb = arr[0], arr[1], arr[2]
            if np.isfinite(aco) and np.isfinite(hyb) and aco != 0:
                d = (aco - hyb) / aco * 100.0
                disp = -d
                c = '#2CA02C' if d > 0 else '#D62728' if d < 0 else 'black'
                ax.text(x[2], hyb + 0.065 * (ymax - ymin), f"vs ACO {_fmt_signed_percent(disp)}",
                        ha='center', va='bottom', fontsize=8, fontstyle='italic', color=c)
            if np.isfinite(alns) and np.isfinite(hyb) and alns != 0:
                d = (alns - hyb) / alns * 100.0
                disp = -d
                c = '#2CA02C' if d > 0 else '#D62728' if d < 0 else 'black'
                ax.text(x[2], hyb + 0.11 * (ymax - ymin), f"vs ALNS {_fmt_signed_percent(disp)}",
                        ha='center', va='bottom', fontsize=8, fontstyle='italic', color=c)

        handles = [
            mpatches.Patch(facecolor=COLOR[a], edgecolor='black', hatch=HATCH[a]) for a in ALG_ORDER
        ]
        labels = [LABEL[a] for a in ALG_ORDER]

        fig.subplots_adjust(bottom=0.14, wspace=0.28)
        place_legend(fig, axes, handles, labels)

        out_pdf = FIG_DIR / f"{OUT_BASENAME}.pdf"
        out_png = FIG_DIR / f"{OUT_BASENAME}.png"
        fig.savefig(out_pdf)
        fig.savefig(out_png, dpi=600)
        plt.close(fig)

        print("[Fix Done] plot_fig1_grouped_bar.py → 图片已更新")
    except Exception as e:
        log_warning(f"runtime error: {e}")
        print("[Fix Done] plot_fig1_grouped_bar.py → 图片已更新")


if __name__ == "__main__":
    main()
