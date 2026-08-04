import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

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
    'axes.spines.top':    False,
    'axes.spines.right':  False,
})

OUT_DIR    = r'D:\experiment\experiment_5\output\figures'
SCRIPT_DIR = r'D:\experiment\experiment_5\output\figures\scripts'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SCRIPT_DIR, exist_ok=True)

SIZE_ORDER = ['small', 'medium', 'large']
SIZE_LABEL = {'small': 'Small', 'medium': 'Medium', 'large': 'Large'}
SIZE_COLOR = {
    'small':  '#2166AC',
    'medium': '#4DAC26',
    'large':  '#D6604D',
}

df1 = pd.read_csv(r'D:\experiment\experiment_5\exp5_1_penalty.csv')

fig, axes = plt.subplots(1, 3, figsize=(12, 5.5))
plt.subplots_adjust(left=0.07, right=0.97,
                    top=0.88, bottom=0.30,
                    wspace=0.32)

lambda_vals = sorted(df1['parameter_value'].unique())
x_pos = np.arange(len(lambda_vals))
bar_w = 0.55

for col, size in enumerate(SIZE_ORDER):
    ax = axes[col]
    ax.grid(False)

    sub = df1[df1['instance_scale'] == size]

    routing_means = []
    fail_means    = []
    fail_raw_vals = []

    for lv in lambda_vals:
        grp = sub[sub['parameter_value'] == lv]
        zt  = grp['z_total_mean'].mean()
        zf  = grp['z_fail_expected_mean'].mean()
        routing_means.append(zt - zf)
        fail_means.append(zf)
        fail_raw_vals.append(zf / lv)

    routing_means = np.array(routing_means)
    fail_means    = np.array(fail_means)
    fail_raw_vals = np.array(fail_raw_vals)
    baseline_raw  = fail_raw_vals[0]

    ax.bar(x_pos, routing_means,
           width=bar_w, color=SIZE_COLOR[size],
           alpha=0.78, label='Routing cost',
           edgecolor='white', linewidth=0.8,
           zorder=3)
    ax.bar(x_pos, fail_means,
           bottom=routing_means,
           width=bar_w, color=SIZE_COLOR[size],
           alpha=0.32, label=r'$\lambda \cdot Z_{\mathrm{fail}}$ (weighted penalty)',
           edgecolor='white', linewidth=0.8,
           hatch='///', zorder=3)

    for xi in range(len(lambda_vals)):
        total = routing_means[xi] + fail_means[xi]
        ax.text(xi, total + total * 0.025,
                f'{total:.0f}',
                ha='center', va='bottom',
                fontsize=10, color='#222222',
                fontweight='500')

    for xi in range(len(lambda_vals)):
        total = routing_means[xi] + fail_means[xi]
        if fail_means[xi] > total * 0.15:
            pct = fail_means[xi] / total * 100
            ax.text(xi,
                    routing_means[xi] + fail_means[xi] * 0.50,
                    f'{pct:.0f}%',
                    ha='center', va='center',
                    fontsize=10, color='#333333',
                    style='italic')

    ax.plot(x_pos, routing_means, '-o',
            color='white', linewidth=0,
            markersize=0, zorder=5)

    for xi in range(len(lambda_vals)):
        ax.text(xi, routing_means[xi] * 0.50,
                f'{routing_means[xi]:.0f}',
                ha='center', va='center',
                fontsize=10, color='white',
                fontweight='500')

    ax.set_xticks(x_pos)
    xticklabels = [rf'${int(lv)}\!\times\!P_{{\rm fail}}$'
                   for lv in lambda_vals]
    ax.set_xticklabels(xticklabels, fontsize=9.5)
    size_xlabel = ['Small (n=8–20)', 'Medium (n=30–100)', 'Large (n=150–300)']
    ax.set_xlabel(size_xlabel[col], fontsize=10, labelpad=8)

    if col == 0:
        ax.set_ylabel(r'$Z_{\mathrm{total}}$ (mean)', fontsize=10)

    y_low = ax.get_ylim()[0]
    y_high = ax.get_ylim()[1]
    y_span = y_high - y_low

    for xi, (lv, fr) in enumerate(zip(lambda_vals, fail_raw_vals)):
        pct_reduction = (baseline_raw - fr) / baseline_raw * 100

        ax.annotate(
            f'{fr:.1f}',
            xy=(xi, y_low),
            xytext=(0, -38),
            textcoords='offset points',
            ha='center', va='top',
            fontsize=10, color='#666666',
            annotation_clip=False,
        )

        if pct_reduction < 0.5:
            reduction_label = '—'
            reduction_color = '#aaaaaa'
        else:
            reduction_label = f'\u2193{pct_reduction:.0f}%'
            reduction_color = '#2CA02C'

        ax.annotate(
            reduction_label,
            xy=(xi, y_low),
            xytext=(0, -52),
            textcoords='offset points',
            ha='center', va='top',
            fontsize=10, color=reduction_color,
            fontweight='500',
            annotation_clip=False,
        )

    if col == 0:
        ax.annotate(
            r'$Z_{\rm fail}^{\rm raw}$:',
            xy=(x_pos[0], y_low),
            xytext=(-44, -38),
            textcoords='offset points',
            ha='right', va='top',
            fontsize=10, color='#666666',
            style='italic',
            annotation_clip=False,
        )
        ax.annotate(
            'vs. \u03c6=1:',
            xy=(x_pos[0], y_low),
            xytext=(-44, -52),
            textcoords='offset points',
            ha='right', va='top',
            fontsize=10, color='#666666',
            style='italic',
            annotation_clip=False,
        )

legend_handles = [
    mpatches.Patch(facecolor='#666666', alpha=0.78,
                   edgecolor='white',
                   label='Routing cost'),
    mpatches.Patch(facecolor='#666666', alpha=0.32,
                   edgecolor='white', hatch='///',
                   label=r'Weighted failure penalty ($\varphi \cdot P_{\mathrm{fail}}$)'),
]
fig.legend(
    handles=legend_handles,
    loc='upper center',
    bbox_to_anchor=(0.50, 1.02),
    bbox_transform=fig.transFigure,
    ncol=2,
    frameon=True, framealpha=0.95,
    edgecolor='#cccccc',
    fontsize=9.5,
    handlelength=1.8,
    columnspacing=1.5,
    borderpad=0.5,
)

for ext in ['pdf', 'png']:
    fig.savefig(f'{OUT_DIR}/exp5_fig1_cost_decomposition.{ext}')
plt.close()
print('[Fig1 Done] exp5_fig1_cost_decomposition.pdf/.png')
