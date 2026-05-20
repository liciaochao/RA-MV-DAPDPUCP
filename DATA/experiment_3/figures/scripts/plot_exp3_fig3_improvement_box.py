from pathlib import Path
import sys

VENDOR = Path(r"D:\codex_drone\.vendor")
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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

SIZE_ORDER = ['Small', 'Medium', 'Large']
TITLES = {
    'Small':  'Small (n=8\u201320)',
    'Medium': 'Medium (n=30\u2013100)',
    'Large':  'Large (n=150\u2013300)',
}


def safe_read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f'missing file: {path}')
    return pd.read_csv(path)


BASE = Path(r"D:\experiment\experiment_3\output")
IN_MAIN = BASE / "exp3_main_table.csv"
OUT_DIR = BASE / "figures"
OUT_BASE = "exp3_fig3_improvement_box"


def make_box(ax, arrays, colors):
    bp = ax.boxplot(
        arrays,
        tick_labels=['avg_cost', 'worst_cost', 'fail_rate'],
        patch_artist=True,
        whis=1.5,
        medianprops=dict(color='white', linewidth=1.4),
        boxprops=dict(linewidth=0.8, color='#444444'),
        whiskerprops=dict(linewidth=0.8, color='#444444'),
        capprops=dict(linewidth=0.8, color='#444444'),
        flierprops=dict(marker='o', markerfacecolor='none', markeredgecolor='#666666', markersize=4),
    )
    for b, c in zip(bp['boxes'], colors):
        b.set_facecolor(c)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = safe_read_csv(IN_MAIN)

    # derived for plotting
    df['impr_CE_avg'] = (df['E_avg_cost'] - df['C_avg_cost']) / df['E_avg_cost'] * 100
    df['impr_CE_worst'] = (df['E_worst_cost'] - df['C_worst_cost']) / df['E_worst_cost'] * 100
    df['impr_CE_fail'] = (df['E_avg_fail'] - df['C_avg_fail']) / (df['E_avg_fail'] + 1e-6) * 100

    df['impr_DE_avg'] = (df['E_avg_cost'] - df['D_avg_cost']) / df['E_avg_cost'] * 100
    df['impr_DE_worst'] = (df['E_worst_cost'] - df['D_worst_cost']) / df['E_worst_cost'] * 100
    df['impr_DE_fail'] = (df['E_avg_fail'] - df['D_avg_fail']) / (df['E_avg_fail'] + 1e-6) * 100

    # required aliases
    df['delta_cost_C_vs_E_pct'] = df['impr_CE_avg']
    df['delta_fail_C_vs_E_pct'] = df['impr_CE_fail']
    df['delta_cost_D_vs_E_pct'] = df['impr_DE_avg']
    df['delta_fail_D_vs_E_pct'] = df['impr_DE_fail']

    fig, axes = plt.subplots(2, 3, figsize=(10, 5))
    fig.subplots_adjust(left=0.08, right=0.80, top=0.92, bottom=0.10, hspace=0.40, wspace=0.32)

    colors = ['#AEC6E8', '#FFBB78', '#98DF8A']

    for j, size in enumerate(SIZE_ORDER):
        sub = df[df['size'] == size]

        arr1 = [sub['impr_CE_avg'].dropna().values,
                sub['impr_CE_worst'].dropna().values,
                sub['impr_CE_fail'].dropna().values]
        make_box(axes[0, j], arr1, colors)
        axes[0, j].set_title(TITLES[size], fontsize=10)
        axes[0, j].grid(axis='y', alpha=0.28)

        arr2 = [sub['impr_DE_avg'].dropna().values,
                sub['impr_DE_worst'].dropna().values,
                sub['impr_DE_fail'].dropna().values]
        make_box(axes[1, j], arr2, colors)
        axes[1, j].grid(axis='y', alpha=0.28)

        if j == 0:
            axes[0, j].set_ylabel('C vs E improvement (%)')
            axes[1, j].set_ylabel('D vs E improvement (%)')

    # 修复一：统一两行y轴范围（放在所有boxplot之后）
    all_vals = pd.concat([
        df['delta_cost_C_vs_E_pct'],
        df['delta_fail_C_vs_E_pct'],
        df['delta_cost_D_vs_E_pct'],
        df['delta_fail_D_vs_E_pct'],
    ]).dropna()

    pad = 0.15
    global_min = all_vals.min() * (1 + pad)
    global_max = all_vals.max() * (1 + pad)

    for row_axes in axes:
        for ax in row_axes:
            ax.set_ylim(global_min, global_max)
            ax.axhline(0, color='#aaaaaa', linewidth=0.8, linestyle='--', zorder=0)

    # 下半行标注
    for ax in axes[1]:
        ax.text(0.03, 0.05,
                'D increases cost\n(repair adds travel)',
                transform=ax.transAxes,
                ha='left', va='bottom', fontsize=7,
                color='#D62728', style='italic',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8))

    # 修复二：图例右侧固定，移出子图
    legend_handles = [
        Patch(facecolor='#AEC6E8', edgecolor='#4878CF', label='Avg cost'),
        Patch(facecolor='#FFBB78', edgecolor='#D65F5F', label='Worst cost'),
        Patch(facecolor='#98DF8A', edgecolor='#3B8B3B', label='Fail rate'),
    ]

    fig.legend(
        handles=legend_handles,
        loc='center left',
        bbox_to_anchor=(0.815, 0.50),
        bbox_transform=fig.transFigure,
        frameon=True,
        framealpha=0.95,
        edgecolor='#cccccc',
        fontsize=9,
        handlelength=1.6,
        borderpad=0.7,
        labelspacing=0.55,
    )

    fig.text(0.42, 0.01,
             'Row 1: Scheme C vs. E improvement.  '
             'Row 2: Scheme D vs. E — '
             'negative avg/worst cost values indicate that basic repair '
             'incurs extra travel cost despite reducing failures.',
             ha='center', va='bottom', fontsize=7.5,
             color='#555555', fontfamily='Times New Roman',
             linespacing=1.5)
    plt.subplots_adjust(bottom=0.14)

    out_pdf = OUT_DIR / f"{OUT_BASE}.pdf"
    out_png = OUT_DIR / f"{OUT_BASE}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=600)
    plt.close(fig)


if __name__ == '__main__':
    main()
