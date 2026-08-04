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
    'legend.fontsize':    14,
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
OUT_BASE = "exp2_fig2_multi_metric"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = safe_read_csv(IN_MAIN)

    grouped = (
        df.groupby('size', observed=True)[
            ['delta_avg_cost_pct', 'delta_worst_cost_pct', 'delta_fail_pct', 'delta_std_pct']
        ]
        .mean()
        .reindex(SIZE_ORDER)
    )

    metrics = ['Avg cost', 'Worst cost', 'Fail rate', 'Cost std']
    cols = ['delta_avg_cost_pct', 'delta_worst_cost_pct', 'delta_fail_pct', 'delta_std_pct']

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.5))
    fig.subplots_adjust(left=0.10, right=0.97, wspace=0.52, top=0.88, bottom=0.15)

    all_values = []
    for _, r in grouped.iterrows():
        all_values.extend([r[c] for c in cols if np.isfinite(r[c])])
    lim = max(5, max(abs(v) for v in all_values) * 1.25) if all_values else 10

    for i, size in enumerate(SIZE_ORDER):
        ax = axes[i]
        row = grouped.loc[size]
        vals = np.array([row[c] for c in cols], dtype=float)
        y = np.arange(len(metrics))
        colors = ['#2CA02C' if v >= 0 else '#D62728' for v in vals]

        bars = ax.barh(y, vals, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xlim(-lim, lim)

        for bar in bars:
            width = bar.get_width()
            y_pos = bar.get_y() + bar.get_height() / 2
            if width >= 0:
                x_pos = width + 0.5
                ha = 'left'
            else:
                x_pos = width - 0.5
                ha = 'right'

            text_color = 'black'
            ax_xmax = ax.get_xlim()[1]
            if width >= 0 and x_pos > ax_xmax * 0.92:
                x_pos = width - 0.8
                ha = 'right'
                text_color = 'white'

            ax.text(x_pos, y_pos, f'{width:.1f}%',
                    va='center', ha=ha, fontsize=14,
                    color=text_color, clip_on=False)

        ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
        ax.set_yticks(y)
        if i == 0:
            ax.set_yticklabels(['Avg cost', 'Worst cost', 'Fail rate', 'Cost std'], fontsize=9)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis='y', length=0)

        ax.set_xlabel('Improvement (%)')

    handles = [
        mpatches.Patch(facecolor='#2CA02C', edgecolor='black', label='Scheme A (defense) better'),
        mpatches.Patch(facecolor='#D62728', edgecolor='black', label='Scheme B (no defense) better'),
    ]
    labels = ['Scheme A (defense) better', 'Scheme B (no defense) better']
    fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(labels),
        frameon=True,
        fontsize=10,
    )

    fig.canvas.draw()
    for ax in axes.flat:
        ax.xaxis.set_label_coords(0.5, -0.12)
        ax.grid(False)

    # Reserve space for top legend and two-layer bottom text region.
    fig.tight_layout(rect=[0, 0.10, 1, 0.92])

    # Bottom group labels centered to actual subplot centers.
    group_labels = ['Small (n=8–20)', 'Medium (n=30–100)', 'Large (n=150–300)']
    num_cols = 3
    for col in range(num_cols):
        ax = axes[0, col] if getattr(axes, "ndim", 1) == 2 else axes[col]
        pos = ax.get_position()
        x_center = (pos.x0 + pos.x1) / 2
        fig.text(
            x_center, 0.02,
            group_labels[col],
            ha='center', va='bottom',
            fontsize=11, fontweight='bold'
        )

    out_pdf = OUT_DIR / f'{OUT_BASE}.pdf'
    out_png = OUT_DIR / f'{OUT_BASE}.png'
    fig.savefig(r'D:\experiment\experiment_2\output\figures\exp2_fig2_multi_metric.pdf',
                bbox_inches='tight', dpi=300)
    fig.savefig(r'D:\experiment\experiment_2\output\figures\exp2_fig2_multi_metric.png',
                bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('[Fig2 Done] exp2_fig2_multi_metric.pdf / .png')


if __name__ == '__main__':
    main()
