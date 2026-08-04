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
TITLES = {
    'Small': 'Small (n=8\u201320)',
    'Medium': 'Medium (n=30\u2013100)',
    'Large': 'Large (n=150\u2013300)',
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
OUT_BASE = "exp2_fig1_cost_bar"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = safe_read_csv(IN_MAIN)
    grouped = (
        df.groupby('size', observed=True)[
            ['A_avg_cost', 'B_avg_cost', 'A_cost_std', 'B_cost_std', 'offline_overhead_pct']
        ]
        .mean()
        .reset_index()
    )

    fig, axes = plt.subplots(1, 3, figsize=(9, 5.5), sharey=False)
    plt.subplots_adjust(top=0.88, bottom=0.18, wspace=0.38)

    global_min = np.nanmin([
        (grouped['A_avg_cost'] - grouped['A_cost_std']).min(),
        (grouped['B_avg_cost'] - grouped['B_cost_std']).min(),
    ])
    y_start = global_min * 0.95 if np.isfinite(global_min) else 0.0

    for ax, size in zip(axes, SIZE_ORDER):
        sub = grouped[grouped['size'] == size]
        if sub.empty:
            ax.text(0.5, 0.5, f'No data for {size}', transform=ax.transAxes, ha='center', va='center')
            continue

        a_val = float(sub['A_avg_cost'].values[0])
        b_val = float(sub['B_avg_cost'].values[0])
        a_std = float(sub['A_cost_std'].values[0])
        b_std = float(sub['B_cost_std'].values[0])

        ax.bar(
            [0, 1],
            [a_val, b_val],
            yerr=[a_std, b_std],
            color=['#D65F5F', '#888780'],
            width=0.35,
            edgecolor='black',
            linewidth=0.6,
            capsize=3,
            error_kw={'elinewidth': 0.8},
        )

        max_top = max(a_val + a_std, b_val + b_std)
        span = max(max_top - y_start, 1e-6)
        ax.set_ylim(y_start, max_top + span * 0.38)

        ax.set_title(TITLES[size])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Scheme A', 'Scheme B'])
        ax.grid(axis='y', alpha=0.35)
        if size == 'Small':
            ax.set_ylabel('Average online actual cost')

    # layered annotations to avoid overlap
    for ax, size in zip(axes, SIZE_ORDER):
        sub = grouped[grouped['size'] == size]
        if sub.empty:
            continue

        a_val = float(sub['A_avg_cost'].values[0])
        b_val = float(sub['B_avg_cost'].values[0])
        a_std = float(sub['A_cost_std'].values[0])
        b_std = float(sub['B_cost_std'].values[0])
        overhead = float(sub['offline_overhead_pct'].values[0])

        if b_val != 0:
            cost_change_pct = (a_val - b_val) / b_val * 100.0
        else:
            cost_change_pct = np.nan

        label_text = f'{cost_change_pct:+.1f}%' if np.isfinite(cost_change_pct) else 'NA'
        label_color = '#2CA02C' if np.isfinite(cost_change_pct) and cost_change_pct < 0 else '#D62728'

        ymin, ymax = ax.get_ylim()
        yrange = ymax - ymin

        # Layer 1: bar value labels with anti-overlap logic for close values
        value_gap_pct = abs(a_val - b_val) / max(a_val, b_val) if max(a_val, b_val) > 0 else 0.0
        if value_gap_pct < 0.20:
            ax.text(-0.1, a_val - yrange * 0.03, f'{a_val:.2f}',
                    ha='right', va='top', fontsize=8,
                    color='#8B2020')
            ax.text(1.1, b_val - yrange * 0.03, f'{b_val:.2f}',
                    ha='left', va='top', fontsize=8,
                    color='#444444')
        else:
            ax.text(0, a_val + yrange * 0.02, f'{a_val:.2f}',
                    ha='center', va='bottom', fontsize=8,
                    color='#8B2020')
            ax.text(1, b_val + yrange * 0.02, f'{b_val:.2f}',
                    ha='center', va='bottom', fontsize=8,
                    color='#444444')

        # Layer 2: cost-change label centered above both bars
        mid_x = 0.5
        top_y = max(a_val + a_std, b_val + b_std) + yrange * 0.10
        ax.annotate(
            label_text,
            xy=(mid_x, top_y),
            ha='center', va='bottom',
            fontsize=8, fontstyle='italic',
            color=label_color,
            bbox=dict(boxstyle='round,pad=0.2',
                      fc='white',
                      ec='#2CA02C' if np.isfinite(cost_change_pct) and cost_change_pct < 0 else '#D62728',
                      alpha=0.85, linewidth=0.6),
        )
        ax.annotate('', xy=(0, a_val + a_std + yrange * 0.01),
                    xytext=(mid_x, top_y - yrange * 0.005),
                    arrowprops=dict(arrowstyle='-', color=label_color,
                                    lw=0.6, alpha=0.6))
        ax.annotate('', xy=(1, b_val + b_std + yrange * 0.01),
                    xytext=(mid_x, top_y - yrange * 0.005),
                    arrowprops=dict(arrowstyle='-', color=label_color,
                                    lw=0.6, alpha=0.6))

        # Layer 3: overhead in top-left corner
        ax.text(0.03, 0.97, f'Offline overhead:\n{overhead:+.1f}%',
                transform=ax.transAxes,
                ha='left', va='top',
                fontsize=7.5, color='#888888',
                linespacing=1.4,
                bbox=dict(boxstyle='round,pad=0.25',
                          fc='white', ec='none', alpha=0.8))

    handles = [
        mpatches.Patch(facecolor='#D65F5F', edgecolor='black', label='Scheme A (with defense)'),
        mpatches.Patch(facecolor='#888780', edgecolor='black', label='Scheme B (no defense)'),
    ]
    labels = ['Scheme A (with defense)', 'Scheme B (no defense)']
    place_legend(fig, axes, handles, labels)

    fig.text(
        0.42, 0.04,
        'Error bars indicate standard deviation across 100 Monte Carlo trials.\n'
        'Offline overhead = extra planning cost of Scheme A vs. Scheme B.\n'
        'Percentage label = relative cost change of Scheme A vs. Scheme B '
        '(negative = cost saving).',
        ha='center', va='top', fontsize=7.5, color='#666666',
        linespacing=1.5, fontfamily='Times New Roman'
    )

    out_pdf = OUT_DIR / f'{OUT_BASE}.pdf'
    out_png = OUT_DIR / f'{OUT_BASE}.png'
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=600)
    plt.close(fig)


if __name__ == '__main__':
    main()
