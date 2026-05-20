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


def safe_float(v):
    try:
        return float(v)
    except Exception:
        return np.nan


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not IN_FILE.exists():
        log_warning(f"missing input file: {IN_FILE}")
        return

    df = pd.read_csv(IN_FILE)
    required_cols = {"size", "algorithm", "obj_mean"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        log_warning(f"missing required columns in exp1_grouped_summary.csv: {sorted(missing_cols)}")
        return

    # n_instances can be missing in some versions; fallback to known counts.
    fallback_n = {"Small": 21, "Medium": 18, "Large": 18}
    n_map = {}
    if "n_instances" in df.columns:
        for s in SIZE_ORDER:
            sub = df[df["size"] == s]
            if not sub.empty:
                n_map[s] = int(pd.to_numeric(sub["n_instances"], errors="coerce").dropna().iloc[0]) if not pd.to_numeric(sub["n_instances"], errors="coerce").dropna().empty else fallback_n[s]
            else:
                n_map[s] = fallback_n[s]
    else:
        log_warning("column n_instances not found; fallback to predefined counts")
        n_map = fallback_n

    pivot = df.pivot_table(index="size", columns="algorithm", values="obj_mean", aggfunc="first")
    for s in SIZE_ORDER:
        for a in ALG_ORDER:
            if s not in pivot.index or a not in pivot.columns:
                log_warning(f"missing grouped value for size={s}, algorithm={a}")

    size_title = {
        "Small":  "Small (n=8–20, 21 inst.)",
        "Medium": "Medium (n=30–100, 18 inst.)",
        "Large":  "Large (n=150–300, 18 inst.)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(7, 4), constrained_layout=False)
    x = np.arange(len(ALG_ORDER))
    bar_width = 0.25

    for idx, size in enumerate(SIZE_ORDER):
        ax = axes[idx]
        vals = []
        for alg in ALG_ORDER:
            if (size in pivot.index) and (alg in pivot.columns):
                vals.append(safe_float(pivot.loc[size, alg]))
            else:
                vals.append(np.nan)

        vals_arr = np.array(vals, dtype=float)
        valid = vals_arr[~np.isnan(vals_arr)]
        if valid.size == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(size_title.get(size, size))
            ax.set_xticks([])
            continue

        ymin = np.nanmin(valid) * 0.95
        ymax = np.nanmax(valid) * 1.12
        yspan = max(ymax - ymin, 1e-6)
        ymax = ymax + 0.12 * yspan

        bars = []
        for i, alg in enumerate(ALG_ORDER):
            bar = ax.bar(
                x[i],
                vals_arr[i],
                width=bar_width,
                color=COLOR[alg],
                edgecolor="black",
                linewidth=0.7,
                hatch=HATCH[alg],
                zorder=3,
            )
            bars.append(bar[0])

        ax.set_ylim(ymin, ymax)
        ax.set_title(size_title.get(size, size))
        ax.set_xticks([])
        ax.grid(axis="y", alpha=0.35, zorder=0)

        if idx == 0:
            ax.set_ylabel("Objective value")

        for i, v in enumerate(vals_arr):
            if np.isnan(v):
                continue
            ax.text(x[i], v + 0.015 * (ax.get_ylim()[1] - ax.get_ylim()[0]), f"{v:.2f}", ha="center", va="bottom", fontsize=9)

        # ACO-ALNS improvement annotations
        try:
            aco = float(vals_arr[0])
            alns = float(vals_arr[1])
            hybrid = float(vals_arr[2])
            d_aco = (aco - hybrid) / aco * 100.0 if aco != 0 else np.nan
            d_alns = (alns - hybrid) / alns * 100.0 if alns != 0 else np.nan
        except Exception:
            d_aco, d_alns = np.nan, np.nan

        if not np.isnan(d_aco):
            # Requirement: improvement displayed with negative sign.
            disp = -d_aco
            c = "#2CA02C" if d_aco > 0 else "#D62728" if d_aco < 0 else "black"
            ax.text(
                x[2],
                vals_arr[2] + 0.065 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                f"vs ACO {disp:+.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                fontstyle="italic",
                color=c,
                clip_on=False,
            )
        if not np.isnan(d_alns):
            disp = -d_alns
            c = "#2CA02C" if d_alns > 0 else "#D62728" if d_alns < 0 else "black"
            ax.text(
                x[2],
                vals_arr[2] + 0.11 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                f"vs ALNS {disp:+.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                fontstyle="italic",
                color=c,
                clip_on=False,
            )

    handles = [
        mpatches.Patch(facecolor=COLOR[a], edgecolor="black", hatch=HATCH[a], label=LABEL[a])
        for a in ALG_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    out_pdf = FIG_DIR / f"{OUT_BASENAME}.pdf"
    out_png = FIG_DIR / f"{OUT_BASENAME}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=600)
    plt.close(fig)

    print(f"[Task A Done] 脚本已保存: {Path(__file__).resolve()}")
    print(f"[Task A Done] 图片已保存: {out_pdf}  +  {out_png}")


if __name__ == "__main__":
    main()

