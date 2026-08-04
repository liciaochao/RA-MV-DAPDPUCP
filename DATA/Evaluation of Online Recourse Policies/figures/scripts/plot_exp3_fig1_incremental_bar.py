from pathlib import Path
import sys

VENDOR = Path(r"D:\codex_drone\.vendor")
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    'font.family':        'Times New Roman',
    'font.size':          13,
    'axes.titlesize':     14,
    'axes.labelsize':     13,
    'xtick.labelsize':    12,
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

SIZE_ORDER = ['Small', 'Medium', 'Large']
TITLES = {
    'Small':  'Small (n=8\u201320)',
    'Medium': 'Medium (n=30\u2013100)',
    'Large':  'Large (n=150\u2013300)',
}


def place_legend(fig, ax_or_axes, handles, labels,
                 entry_width_inch=1.6, entry_height_inch=0.22,
                 pad_inch=0.35):
    legend_width = entry_width_inch + pad_inch * 2
    fig_w = fig.get_size_inches()[0]
    right_margin = 1.0 - (legend_width / fig_w)
    fig.subplots_adjust(right=right_margin)

    if hasattr(ax_or_axes, '__len__'):
        ref_ax = ax_or_axes[-1]
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


def safe_read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f'missing file: {path}')
    return pd.read_csv(path)


BASE = Path(r"D:\experiment\experiment_3\output")
IN_MAIN = BASE / "exp3_main_table.csv"
OUT_DIR = BASE / "figures"
OUT_BASE = "exp3_fig1_incremental_bar"


def draw_bracket(ax, x1, x2, y, h, text, color='#7F77DD'):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.9, c=color)
    ax.text((x1 + x2) / 2, y + h + 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0]), text,
            ha='center', va='bottom', fontsize=8, color=color)


def get_label_style(change_pct):
    if change_pct < 0:
        return f'{change_pct:.1f}%', '#2CA02C'
    else:
        return f'+{change_pct:.1f}%', '#D62728'


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = safe_read_csv(IN_MAIN)
    g = df.groupby('size', observed=True)[['E_avg_cost', 'D_avg_cost', 'C_avg_cost']].mean().reindex(SIZE_ORDER)

    fig, axes = plt.subplots(1, 3, figsize=(9, 5))
    plt.subplots_adjust(top=0.88, bottom=0.12, wspace=0.36)

    for i, size in enumerate(SIZE_ORDER):
        ax = axes[i]
        r = g.loc[size]

        E_avg_cost = float(r['E_avg_cost'])
        D_avg_cost = float(r['D_avg_cost'])
        C_avg_cost = float(r['C_avg_cost'])

        vals = [E_avg_cost, D_avg_cost, C_avg_cost]
        x = np.arange(3)

        ax.bar(x, vals, width=0.38,
               color=['#888780', '#4878CF', '#D65F5F'],
               edgecolor='black', linewidth=0.6)

        all_vals = [E_avg_cost, D_avg_cost, C_avg_cost]
        ymin = min(all_vals) * 0.982
        ymax = max(all_vals) * 1.018
        ax.set_ylim(ymin, ymax)

        yr = ymax - ymin

        ax.text(0, E_avg_cost + 0.02 * yr, f"{E_avg_cost:.2f}",
                ha='center', va='bottom', fontsize=8, color='#111111')

        D_change = (D_avg_cost - E_avg_cost) / E_avg_cost * 100 if E_avg_cost != 0 else np.nan
        C_change = (C_avg_cost - E_avg_cost) / E_avg_cost * 100 if E_avg_cost != 0 else np.nan

        d_text, d_color = get_label_style(D_change)
        c_text, c_color = get_label_style(C_change)

        ax.text(1, D_avg_cost + 0.02 * yr, d_text,
                ha='center', va='bottom', fontsize=8, color='#111111',
                bbox=dict(boxstyle='round,pad=0.2', fc='white',
                          ec=d_color, alpha=0.85, linewidth=0.6))
        ax.text(2, C_avg_cost + 0.02 * yr, c_text,
                ha='center', va='bottom', fontsize=8, color='#111111',
                bbox=dict(boxstyle='round,pad=0.2', fc='white',
                          ec=c_color, alpha=0.85, linewidth=0.6))

        y_lo, y_hi = ax.get_ylim()
        for x_pos, val in [(1, D_avg_cost), (2, C_avg_cost)]:
            bar_top_frac = (val - y_lo) / (y_hi - y_lo)
            bar_bottom_frac = 0.0
            text_frac = bar_bottom_frac + (bar_top_frac - bar_bottom_frac) * 0.45
            text_frac = max(0.05, min(text_frac, bar_top_frac - 0.08))

            t = ax.text(x_pos, y_lo + text_frac * (y_hi - y_lo),
                        f'{val:.2f}',
                        ha='center', va='center',
                        fontsize=7.5, color='#111111',
                        clip_on=False)
            t.set_path_effects([
                pe.withStroke(linewidth=2.5, foreground='white')
            ])

        c_vs_d = (C_avg_cost - D_avg_cost) / D_avg_cost * 100 if D_avg_cost != 0 else np.nan
        by = max(D_avg_cost, C_avg_cost) + 0.05 * yr
        draw_bracket(ax, 1, 2, by, 0.03 * yr, f"Step3+4: {c_vs_d:+.1f}%")

        ax.set_xticks(x)
        ax.set_xticklabels(['E', 'D', 'C'])
        if i == 0:
            ax.set_ylabel('Average online actual cost')

    handles = [
        mpatches.Patch(facecolor='#888780', edgecolor='black', label='Scheme E'),
        mpatches.Patch(facecolor='#4878CF', edgecolor='black', label='Scheme D'),
        mpatches.Patch(facecolor='#D65F5F', edgecolor='black', label='Scheme C'),
    ]
    labels = [h.get_label() for h in handles]

    # ---- 保存前的完整处理代码 ----
    # 3. 关闭所有网格
    for ax in axes.flat:
        ax.grid(False)

    # 关闭子图自身图例
    for ax in axes.flat:
        legend = ax.get_legend()
        if legend:
            legend.remove()

    # 1&2. 调整整体布局（顶部留图例，底部留分组标题）
    fig.tight_layout(rect=[0, 0.10, 1, 0.92])

    # 顶端横排图例
    handles_found, labels_found = None, None
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles_found, labels_found = h, l
            break
    if not handles_found:
        handles_found, labels_found = handles, labels

    fig.legend(
        handles_found, labels_found,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(labels_found),
        frameon=True,
        fontsize=10,
    )

    # 底端分组标题（动态居中）
    fig.canvas.draw()
    group_labels = ['Small (n=8–20)', 'Medium (n=30–100)', 'Large (n=150–300)']
    num_cols = 3
    for col in range(num_cols):
        ax = axes[0, col] if getattr(axes, "ndim", 1) == 2 else axes[col]
        pos = ax.get_position()
        x_center = (pos.x0 + pos.x1) / 2
        fig.text(
            x_center, 0.02, group_labels[col],
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )

    out_pdf = OUT_DIR / f"{OUT_BASE}.pdf"
    out_png = OUT_DIR / f"{OUT_BASE}.png"
    fig.savefig(r'D:\experiment\experiment_3\output\figures\exp3_fig1_incremental_bar.pdf',
                bbox_inches='tight', dpi=300)
    fig.savefig(r'D:\experiment\experiment_3\output\figures\exp3_fig1_incremental_bar.png',
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('[Fig1 Done] exp3_fig1_incremental_bar.pdf / .png')


if __name__ == '__main__':
    main()
