from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    'font.family':        'Times New Roman',
    'font.size':          13,
    'axes.titlesize':     14,
    'axes.labelsize':     13,
    'xtick.labelsize':    12,
    'ytick.labelsize':    12,
    'legend.fontsize':    12,
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
    'Small': 'Small\n(n=8\u201320)',
    'Medium': 'Medium\n(n=30\u2013100)',
    'Large': 'Large\n(n=150\u2013300)',
}


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
OUT_BASE = "exp2_fig3_offline_vs_online"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = safe_read_csv(IN_MAIN)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8))
    fig.subplots_adjust(left=0.07, right=0.97, wspace=0.48, top=0.88, bottom=0.08)

    for i, (ax, size) in enumerate(zip(axes, SIZE_ORDER)):
        sub = df[df['size'] == size].copy()
        if sub.empty:
            ax.text(0.5, 0.5, f'No data for {size}', transform=ax.transAxes, ha='center', va='center')
            continue

        for _, r in sub.iterrows():
            xa, ya = r['A_offline_obj'], r['A_avg_cost']
            xb, yb = r['B_offline_obj'], r['B_avg_cost']
            color = '#2CA02C' if r['delta_avg_cost_pct'] >= 0 else '#D62728'
            ax.plot([xb, xa], [yb, ya], color=color, alpha=0.4, linewidth=0.8, zorder=1)

        ax.scatter(sub['A_offline_obj'], sub['A_avg_cost'], marker='D', s=60,
                   c='#D65F5F', edgecolors='black', linewidths=0.4, zorder=3)
        ax.scatter(sub['B_offline_obj'], sub['B_avg_cost'], marker='o', s=40,
                   c='#888780', edgecolors='black', linewidths=0.4, zorder=2)

        all_x = pd.concat([sub['A_offline_obj'], sub['B_offline_obj']])
        all_y = pd.concat([sub['A_avg_cost'], sub['B_avg_cost']])
        all_vals = pd.concat([all_x, all_y])

        margin = (all_vals.max() - all_vals.min()) * 0.08
        vmin = all_vals.min() - margin
        vmax = all_vals.max() + margin

        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.set_aspect('equal', adjustable='box')
        ax.plot([vmin, vmax], [vmin, vmax], linestyle='--', color='#999999', linewidth=0.9)

        b_mean = sub['B_avg_cost'].mean()
        b_std = sub['B_avg_cost'].std()
        threshold = b_mean + 2.5 * b_std
        for _, row in sub.iterrows():
            if pd.notna(threshold) and row['B_avg_cost'] > threshold:
                ax.annotate(
                    f"outlier\n({row['instance'].split('_')[1]})",
                    xy=(row['B_offline_obj'], row['B_avg_cost']),
                    xytext=(8, -12),
                    textcoords='offset points',
                    fontsize=7,
                    color='#888888',
                    arrowprops=dict(arrowstyle='-', color='#cccccc', lw=0.5),
                )

        a_below = int((sub['A_avg_cost'] < sub['A_offline_obj']).sum())
        b_below = int((sub['B_avg_cost'] < sub['B_offline_obj']).sum())
        n = len(sub)
        ax.text(0.97, 0.03,
                f'A below diag: {a_below}/{n}\nB below diag: {b_below}/{n}',
                transform=ax.transAxes,
                ha='right', va='bottom',
                fontsize=10, color='#555555',
                linespacing=1.5,
                bbox=dict(boxstyle='round,pad=0.3',
                          fc='white', ec='none', alpha=0.85))

        ax.set_xlabel('Offline objective')
        if i == 0:
            ax.set_ylabel('Online average actual cost')

    handles = [
        plt.Line2D([0], [0], marker='D', linestyle='none', color='none', markerfacecolor='#D65F5F', markeredgecolor='black', markersize=7, label='Scheme A'),
        plt.Line2D([0], [0], marker='o', linestyle='none', color='none', markerfacecolor='#888780', markeredgecolor='black', markersize=6, label='Scheme B'),
        plt.Line2D([0], [0], color='#2CA02C', linewidth=1.2, alpha=0.5, label='A better online cost'),
        plt.Line2D([0], [0], color='#D62728', linewidth=1.2, alpha=0.5, label='B better online cost'),
    ]
    labels = ['Scheme A', 'Scheme B', 'A better online cost', 'B better online cost']
    fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(labels),
        frameon=True,
        fontsize=10,
    )

    # Move size-group labels to the bottom of the figure.
    fig.text(0.2, 0.01, 'Small (n=8–20)', ha='center', va='bottom', fontsize=11, fontweight='bold')
    fig.text(0.5, 0.01, 'Medium (n=30–100)', ha='center', va='bottom', fontsize=11, fontweight='bold')
    fig.text(0.8, 0.01, 'Large (n=150–300)', ha='center', va='bottom', fontsize=11, fontweight='bold')

    for ax in axes.flat:
        ax.grid(False)

    out_pdf = OUT_DIR / f'{OUT_BASE}.pdf'
    out_png = OUT_DIR / f'{OUT_BASE}.png'
    fig.savefig(r'D:\experiment\experiment_2\output\figures\exp2_fig3_offline_vs_online.pdf',
                bbox_inches='tight', dpi=300)
    fig.savefig(r'D:\experiment\experiment_2\output\figures\exp2_fig3_offline_vs_online.png',
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('[Fig3 Done] exp2_fig3_offline_vs_online.pdf / .png')


if __name__ == '__main__':
    main()
