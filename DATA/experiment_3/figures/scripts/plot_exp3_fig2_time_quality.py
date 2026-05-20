from pathlib import Path
import sys

VENDOR = Path(r"D:\codex_drone\.vendor")
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

import matplotlib
import matplotlib.pyplot as plt
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
OUT_BASE = "exp3_fig2_time_quality"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = safe_read_csv(IN_MAIN)

    fig, axes = plt.subplots(1, 3, figsize=(9, 4))
    plt.subplots_adjust(left=0.08, right=0.82, wspace=0.36, top=0.88, bottom=0.14)

    rng = np.random.default_rng(seed=42)

    for i, size in enumerate(SIZE_ORDER):
        ax = axes[i]
        sub = df[df['size'] == size].copy().reset_index(drop=True)

        x_all = np.concatenate([
            sub['E_online_time'].values,
            sub['D_online_time'].values,
            sub['C_online_time'].values,
        ]) if len(sub) > 0 else np.array([0.0, 1.0])
        xmin, xmax = np.nanmin(x_all), np.nanmax(x_all)
        xpad = (xmax - xmin) * 0.12 if xmax > xmin else 0.1
        ax.set_xlim(xmin - xpad, xmax + xpad)

        y_all = np.concatenate([
            sub['E_avg_cost'].values,
            sub['D_avg_cost'].values,
            sub['C_avg_cost'].values,
        ]) if len(sub) > 0 else np.array([0.0, 1.0])
        y_range = float(np.nanmax(y_all) - np.nanmin(y_all))
        if not np.isfinite(y_range) or y_range <= 0:
            y_range = 1.0

        plot_vals = {}
        for scheme, marker, color, ec, s, y_jitter in [
            ('E', 'o', '#555555', '#222222', 60, 0.000),
            ('D', 's', '#4878CF', '#2E5FA3', 55, 0.008),
            ('C', 'D', '#D65F5F', '#A03030', 65, 0.000),
        ]:
            x_vals = sub[f'{scheme}_online_time'].values.astype(float, copy=True)
            y_vals = sub[f'{scheme}_avg_cost'].values.astype(float, copy=True)

            if y_jitter > 0 and len(y_vals) > 0:
                y_vals = y_vals + rng.uniform(
                    -y_jitter * y_range,
                    y_jitter * y_range,
                    size=len(y_vals),
                )

            plot_vals[scheme] = (x_vals, y_vals)

            ax.scatter(
                x_vals,
                y_vals,
                marker=marker,
                s=s,
                color=color,
                edgecolors=ec,
                linewidths=0.8,
                zorder=4,
                label=f'Scheme {scheme}',
                alpha=0.90,
            )

        for idx in range(len(sub)):
            x_e, y_e = plot_vals['E'][0][idx], plot_vals['E'][1][idx]
            x_d, y_d = plot_vals['D'][0][idx], plot_vals['D'][1][idx]
            x_c, y_c = plot_vals['C'][0][idx], plot_vals['C'][1][idx]

            ax.plot([x_e, x_d], [y_e, y_d], color='#BBBBBB', lw=0.7, alpha=0.7, zorder=2)
            c_is_right_down = (x_c >= x_d) and (y_c <= y_d)
            ax.plot(
                [x_d, x_c],
                [y_d, y_c],
                color='#999999',
                lw=0.8,
                alpha=0.8,
                linestyle='-' if c_is_right_down else '--',
                zorder=2,
            )

        e_mean = sub['E_avg_cost'].mean()
        c_mean = sub['C_avg_cost'].mean()
        saving = (e_mean - c_mean) / e_mean * 100 if e_mean else np.nan
        time_diff = sub['C_online_time'].mean() - sub['E_online_time'].mean()

        ax.text(0.03, 0.05,
                f'C saves {saving:.1f}% cost vs E\n'
                f'at +{time_diff:.2f}s online time',
                transform=ax.transAxes,
                ha='left', va='bottom', fontsize=8,
                color='#444444',
                bbox=dict(boxstyle='round,pad=0.3',
                          fc='white', ec='none', alpha=0.85))

        if size == 'Large':
            n_anomalous = sum(
                1 for _, row in sub.iterrows()
                if row['C_avg_cost'] > row['D_avg_cost'])
            if n_anomalous > 0:
                ax.text(0.97, 0.03,
                        f'{n_anomalous} instances:\nStep3+4 not beneficial',
                        transform=ax.transAxes,
                        ha='right', va='bottom', fontsize=7.5,
                        color='#888888', style='italic',
                        bbox=dict(boxstyle='round,pad=0.25',
                                  fc='white', ec='none', alpha=0.8))

        ax.set_title(TITLES[size])
        ax.set_xlabel('Online replanning time (s)')
        if i == 0:
            ax.set_ylabel('Average online actual cost')
        ax.grid(alpha=0.3)

    handles = [
        plt.Line2D([0], [0], marker='o', linestyle='none', color='none',
                   markerfacecolor='#555555', markeredgecolor='#222222',
                   markeredgewidth=0.8, markersize=7, label='Scheme E'),
        plt.Line2D([0], [0], marker='s', linestyle='none', color='none',
                   markerfacecolor='#4878CF', markeredgecolor='#2E5FA3',
                   markeredgewidth=0.8, markersize=7, label='Scheme D'),
        plt.Line2D([0], [0], marker='D', linestyle='none', color='none',
                   markerfacecolor='#D65F5F', markeredgecolor='#A03030',
                   markeredgewidth=0.8, markersize=7, label='Scheme C'),
        plt.Line2D([0], [0], color='#999999', lw=1.0, label='Per-instance trajectory'),
        plt.Line2D([0], [0], color='#999999', lw=1.0, linestyle='--', label='Anomalous C-vs-D step'),
    ]
    labels = [h.get_label() for h in handles]

    fig.canvas.draw()
    all_ax = axes.flatten() if hasattr(axes, 'flatten') else list(axes)
    tops = [a.get_position().y1 for a in all_ax]
    bottoms = [a.get_position().y0 for a in all_ax]
    ax_mid = (max(tops) + min(bottoms)) / 2.0
    panel_left = max(a.get_position().x1 for a in all_ax) + 0.015

    fig.legend(
        handles,
        labels,
        loc='center left',
        bbox_to_anchor=(panel_left, ax_mid),
        bbox_transform=fig.transFigure,
        frameon=True,
        framealpha=0.95,
        edgecolor='#cccccc',
        fontsize=9,
        handlelength=1.8,
        handleheight=0.8,
    )

    out_pdf = OUT_DIR / f"{OUT_BASE}.pdf"
    out_png = OUT_DIR / f"{OUT_BASE}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=600)
    plt.close(fig)


if __name__ == '__main__':
    main()
