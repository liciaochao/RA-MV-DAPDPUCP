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

SIZE_ORDER = ['Small', 'Medium', 'Large']


def place_legend(fig, ax_or_axes, handles, labels,
                 entry_width_inch=1.6, entry_height_inch=0.22,
                 pad_inch=0.35):
    """
    Place legend in right-side whitespace and auto-adjust right margin.
    """
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


def safe_read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f'missing file: {path}')
    return pd.read_csv(path)


BASE = Path(r"D:\experiment\experiment_2\output")
IN_MAIN = BASE / "exp2_main_table.csv"
OUT_DIR = BASE / "figures"
OUT_BASE = "exp2_fig4_winrate"


def styled_boxplot(ax, data_groups, labels, colors):
    bp = ax.boxplot(
        data_groups,
        tick_labels=labels,
        patch_artist=True,
        whis=1.5,
        medianprops=dict(color='white', linewidth=1.4),
        boxprops=dict(linewidth=0.8, color='#444444'),
        whiskerprops=dict(linewidth=0.8, color='#444444'),
        capprops=dict(linewidth=0.8, color='#444444'),
        flierprops=dict(marker='o', markerfacecolor='none', markeredgecolor='#666666', markersize=4)
    )
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
    return bp


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = safe_read_csv(IN_MAIN)
    df['paired_cost_delta_pct'] = df['paired_cost_delta'] / df['B_avg_cost'] * 100

    colors = ['#AEC6E8', '#FFBB78', '#98DF8A']

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))
    plt.subplots_adjust(left=0.08, right=0.82, wspace=0.48, top=0.88, bottom=0.18)

    win_data = [df[df['size'] == s]['A_win_rate'].dropna().values for s in SIZE_ORDER]
    styled_boxplot(axes[0], win_data, SIZE_ORDER, colors)
    axes[0].axhline(50, color='#888888', linestyle='--', linewidth=0.9)
    axes[0].set_ylabel('Win rate (%)', fontsize=9, labelpad=6)
    axes[0].set_title('Scheme A win-rate distribution', fontsize=11, pad=8)
    axes[0].grid(axis='y', alpha=0.3)

    for i, arr in enumerate(win_data, start=1):
        if len(arr) == 0:
            continue
        axes[0].text(i, np.max(arr) + 1.0, f'mean={np.mean(arr):.1f}%', ha='center', va='bottom', fontsize=8)

    delta_data = [df[df['size'] == s]['paired_cost_delta_pct'].dropna().values for s in SIZE_ORDER]
    styled_boxplot(axes[1], delta_data, SIZE_ORDER, colors)
    axes[1].axhline(0, color='#888888', linestyle='--', linewidth=0.9)
    axes[1].set_ylabel('Paired cost saving (%)', fontsize=9, labelpad=6)
    axes[1].yaxis.set_label_coords(-0.12, 0.5)
    axes[1].set_title('Paired cost delta distribution', fontsize=11, pad=8)
    axes[1].grid(axis='y', alpha=0.3)

    handles = [
        mpatches.Patch(facecolor='#AEC6E8', edgecolor='#444444', label='Small'),
        mpatches.Patch(facecolor='#FFBB78', edgecolor='#444444', label='Medium'),
        mpatches.Patch(facecolor='#98DF8A', edgecolor='#444444', label='Large'),
    ]
    labels = ['Small', 'Medium', 'Large']
    place_legend(fig, axes, handles, labels)

    fig.text(
        0.42, 0.01,
        'Win rate = fraction of 100 MC trials where Scheme A achieves lower actual cost.\n'
        'Paired delta = mean(costₐ − costᵦ) per instance across trials.',
        ha='center', va='bottom',
        fontsize=7, color='#666666',
        linespacing=1.5,
        fontfamily='Times New Roman'
    )

    out_pdf = OUT_DIR / f'{OUT_BASE}.pdf'
    out_png = OUT_DIR / f'{OUT_BASE}.png'
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=600)
    plt.close(fig)


if __name__ == '__main__':
    main()
