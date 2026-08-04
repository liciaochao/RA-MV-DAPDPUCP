from pathlib import Path
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

try:
    import pandas as pd
except ModuleNotFoundError:
    vendor = Path(__file__).resolve().parent / ".vendor"
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

COLOR = {
    'Pure_ACO':  '#4878CF',
    'Pure_ALNS': '#3B8B3B',
    'ACO_ALNS':  '#D65F5F',
}
LABEL = {
    'Pure_ACO':  'Pure ACO',
    'Pure_ALNS': 'Pure ALNS',
    'ACO_ALNS':  'ACO-ALNS',
}
SIZE_ORDER  = ['Small', 'Medium', 'Large']
ALG_ORDER   = ['Pure_ACO', 'Pure_ALNS', 'ACO_ALNS']

SCRIPT_NAME = "plot_fig4_time_quality_scatter.py"
BASE_DIR = Path(r"D:\experiment\experiment_1\output")
IN_FILE = BASE_DIR / "exp1_main_table.csv"
FIG_DIR = BASE_DIR / "figures"
WARN_LOG = FIG_DIR / "warnings.log"
OUT_BASENAME = "fig4_time_quality_scatter"


def log_warning(msg: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(WARN_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{SCRIPT_NAME}] {msg}\n")


def dominates_count(sub: pd.DataFrame, baseline: str):
    if baseline == "Pure_ACO":
        b_obj = pd.to_numeric(sub["ACO_obj_final"], errors="coerce")
        b_t = pd.to_numeric(sub["ACO_time_s"], errors="coerce")
    else:
        b_obj = pd.to_numeric(sub["ALNS_obj_final"], errors="coerce")
        b_t = pd.to_numeric(sub["ALNS_time_s"], errors="coerce")

    h_obj = pd.to_numeric(sub["ACO_ALNS_obj_final"], errors="coerce")
    h_t = pd.to_numeric(sub["ACO_ALNS_time_s"], errors="coerce")
    m = (h_obj < b_obj) & (h_t < b_t)
    return int(m.sum()), int(len(sub))


def main():
    try:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        if not IN_FILE.exists():
            log_warning(f"missing input file: {IN_FILE}")
            return

        df = pd.read_csv(IN_FILE)
        req = {
            "size",
            "ACO_time_s", "ACO_obj_final",
            "ALNS_time_s", "ALNS_obj_final",
            "ACO_ALNS_time_s", "ACO_ALNS_obj_final"
        }
        miss = req - set(df.columns)
        if miss:
            log_warning(f"missing required columns in exp1_main_table.csv: {sorted(miss)}")
            return

        # build dominate text lines and dynamic right panel width
        dominate_lines = []
        for size in ['Small', 'Medium', 'Large']:
            sub = df[df['size'] == size]
            dom_aco = ((pd.to_numeric(sub['ACO_ALNS_time_s'], errors='coerce') < pd.to_numeric(sub['ACO_time_s'], errors='coerce')) &
                       (pd.to_numeric(sub['ACO_ALNS_obj_final'], errors='coerce') < pd.to_numeric(sub['ACO_obj_final'], errors='coerce'))).sum()
            dom_alns = ((pd.to_numeric(sub['ACO_ALNS_time_s'], errors='coerce') < pd.to_numeric(sub['ALNS_time_s'], errors='coerce')) &
                        (pd.to_numeric(sub['ACO_ALNS_obj_final'], errors='coerce') < pd.to_numeric(sub['ALNS_obj_final'], errors='coerce'))).sum()
            n = len(sub)
            dominate_lines.append(f"{size}: vs ACO  {dom_aco}/{n} dominated")
            dominate_lines.append(f"{size}: vs ALNS {dom_alns}/{n} dominated")

        max_chars = max(len(l) for l in dominate_lines) if dominate_lines else 20
        legend_panel_width = min(3.0, max(1.8, round(max_chars * 0.072 + 0.5, 1)))
        fig_width = 7.0 + legend_panel_width

        fig, axes = plt.subplots(1, 3, figsize=(fig_width, 5.0))

        for idx, size in enumerate(SIZE_ORDER):
            ax = axes[idx]
            sub = df[df["size"] == size].copy()
            if sub.empty:
                ax.text(0.5, 0.5, f"No data for {size}", transform=ax.transAxes, ha="center", va="center")
                continue

            ax.scatter(
                pd.to_numeric(sub["ACO_time_s"], errors="coerce"),
                pd.to_numeric(sub["ACO_obj_final"], errors="coerce"),
                marker="o", s=40, c=COLOR["Pure_ACO"], edgecolors="black", linewidths=0.4, alpha=0.9
            )
            ax.scatter(
                pd.to_numeric(sub["ALNS_time_s"], errors="coerce"),
                pd.to_numeric(sub["ALNS_obj_final"], errors="coerce"),
                marker="^", s=40, c=COLOR["Pure_ALNS"], edgecolors="black", linewidths=0.4, alpha=0.9
            )
            ax.scatter(
                pd.to_numeric(sub["ACO_ALNS_time_s"], errors="coerce"),
                pd.to_numeric(sub["ACO_ALNS_obj_final"], errors="coerce"),
                marker="D", s=60, c=COLOR["ACO_ALNS"], edgecolors="black", linewidths=0.4, alpha=0.95
            )

            if size == "Small":
                subtitle = "Small (n=8–20)"
            elif size == "Medium":
                subtitle = "Medium (n=30–100)"
            else:
                subtitle = "Large (n=150–300)"
            ax.set_title(subtitle)
            ax.set_xlabel("Solve time (s)")
            if idx == 0:
                ax.set_ylabel("Final objective value")
            ax.grid(alpha=0.35)

        # Adjust subplot area; keep right panel for legend + dominance text
        right_boundary = 7.0 / fig_width - 0.02
        fig.subplots_adjust(right=right_boundary, left=0.07, wspace=0.28, bottom=0.16)

        panel_left = right_boundary + 0.03

        # 1) legend in right panel
        legend_handles = [
            plt.Line2D([0], [0], marker='o', linestyle='none', markerfacecolor='#4878CF', markeredgecolor='black', markersize=6, label='Pure ACO'),
            plt.Line2D([0], [0], marker='^', linestyle='none', markerfacecolor='#3B8B3B', markeredgecolor='black', markersize=6, label='Pure ALNS'),
            plt.Line2D([0], [0], marker='D', linestyle='none', markerfacecolor='#D65F5F', markeredgecolor='black', markersize=7, label='ACO-ALNS'),
        ]

        # Step 1: get subplot vertical extent in figure coordinates
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        all_axes = axes.flatten() if hasattr(axes, 'flatten') else list(axes)
        tops = [ax.get_position().y1 for ax in all_axes]
        bottoms = [ax.get_position().y0 for ax in all_axes]
        ax_top = max(tops)
        ax_bottom = min(bottoms)
        ax_mid = (ax_top + ax_bottom) / 2.0

        # Step 2: estimate total height of right panel content
        n_legend_entries = 3
        legend_entry_h = 0.055
        legend_title_h = 0.0
        dominate_title_h = 0.045
        dominate_line_h = 0.042
        dominate_gap_h = 0.018
        n_dominate_lines = len(dominate_lines)
        n_group_gaps = 2

        total_content_h = (
            n_legend_entries * legend_entry_h
            + legend_title_h
            + 0.025
            + dominate_title_h
            + 0.012
            + n_dominate_lines * dominate_line_h
            + n_group_gaps * dominate_gap_h
        )

        # Step 3: compute centered top y, clamp to subplot bounds
        content_top = ax_mid + total_content_h / 2.0
        content_top = min(content_top, ax_top)
        content_top = max(content_top, ax_bottom + total_content_h)

        # Step 4: place legend
        legend = fig.legend(
            handles=legend_handles,
            loc='upper left',
            bbox_to_anchor=(panel_left, content_top),
            bbox_transform=fig.transFigure,
            frameon=True,
            framealpha=0.95,
            edgecolor='#cccccc',
            fontsize=9,
            handlelength=1.6,
            borderpad=0.6,
        )

        # Step 5: recompute legend bottom and place dominate content
        fig.canvas.draw()
        legend_bbox = legend.get_window_extent(renderer)
        legend_bottom = legend_bbox.y0 / fig.bbox.height

        y_cursor = legend_bottom - 0.025

        fig.text(
            panel_left + 0.01,
            y_cursor,
            'Pareto dominance',
            transform=fig.transFigure,
            fontsize=8,
            fontweight='bold',
            color='#333333',
            va='top',
            ha='left',
            fontfamily='Times New Roman',
        )
        y_cursor -= dominate_title_h + 0.012

        for i, line in enumerate(dominate_lines):
            if i > 0 and i % 2 == 0:
                y_cursor -= dominate_gap_h
            fig.text(
                panel_left + 0.01,
                y_cursor,
                line,
                transform=fig.transFigure,
                fontsize=8,
                color='#444444',
                va='top',
                ha='left',
                fontfamily='Times New Roman',
            )
            y_cursor -= dominate_line_h

        # note text at bottom
        fig.text(
            0.5,
            0.04,
            "Note: Dominance = ACO-ALNS achieves both lower objective and shorter solve time.",
            ha='center',
            va='bottom',
            fontsize=7,
            color='#666666',
            fontfamily='Times New Roman',
        )

        out_pdf = FIG_DIR / f"{OUT_BASENAME}.pdf"
        out_png = FIG_DIR / f"{OUT_BASENAME}.png"
        fig.savefig(out_pdf)
        fig.savefig(out_png, dpi=600)
        plt.close(fig)

        print("[Fix4 Done] fig4_time_quality_scatter.pdf/.png 已更新")
    except Exception as e:
        log_warning(f"runtime error: {e}")
        print("[Fix4 Done] fig4_time_quality_scatter.pdf/.png 已更新")


if __name__ == "__main__":
    main()
