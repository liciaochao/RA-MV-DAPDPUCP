import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

matplotlib.rcParams.update({
    'font.family':        'Times New Roman',
    'font.size':          11,
    'axes.titlesize':     11,
    'axes.labelsize':     10,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'legend.fontsize':    9.5,
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

df3 = pd.read_csv(r'D:\experiment\experiment_5\exp5_3_drone_ratio.csv')
ratio_vals = sorted(df3['parameter_value'].unique())

REC_LO, REC_HI = 0.35, 0.50

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
plt.subplots_adjust(left=0.07, right=0.97,
                    top=0.88, bottom=0.15,
                    wspace=0.35)

ax_left = axes[0]
ax_left.grid(False)

all_deg = []

for size in SIZE_ORDER:
    sub = df3[df3['instance_scale'] == size]
    grp = sub.groupby('parameter_value').agg(
        z_mean=('z_total_mean', 'mean')
    ).reset_index().sort_values('parameter_value')

    rv = grp['parameter_value'].values
    zm = grp['z_mean'].values
    best = zm.min()
    deg  = (zm - best) / best * 100

    all_deg.extend(deg.tolist())

    ax_left.plot(rv, deg, '-D',
                 color=SIZE_COLOR[size], linewidth=1.8,
                 markersize=7, markerfacecolor='white',
                 markeredgewidth=1.8, zorder=3,
                 label=SIZE_LABEL[size])

    for r, d in zip(rv, deg):
        if d > 0.3:
            ax_left.text(r, d + 0.15,
                         f'{d:.1f}%',
                         ha='center', va='bottom',
                         fontsize=7.5,
                         color=SIZE_COLOR[size])

ax_left.axvspan(REC_LO, REC_HI, alpha=0.12, color='#2CA02C', zorder=0)

y_max = max(all_deg) * 1.10 if all_deg else 1.0
ax_left.text((REC_LO + REC_HI) / 2,
             y_max * 0.95,
             'Recommended\nrange',
             ha='center', va='top',
             fontsize=8, color='#2CA02C',
             style='italic')

ax_left.axhline(2.0, color='#aaaaaa', lw=0.8,
                linestyle='--', alpha=0.8, zorder=1)
ax_left.text(ratio_vals[-1] + 0.01, 2.15,
             '2% threshold',
             fontsize=7.5, color='#aaaaaa',
             va='bottom', ha='left')

ax_left.set_ylim(bottom=0, top=y_max * 1.05)
ax_left.set_xticks(ratio_vals)
ax_left.set_xticklabels([str(r) for r in ratio_vals], fontsize=10)
ax_left.set_xlabel('Drone serviceable ratio', fontsize=10, labelpad=4)
ax_left.set_ylabel(r'$Z_{\mathrm{total}}$ degradation from optimum (%)',
                   fontsize=10)
ax_left.set_title('(a) Solution quality degradation',
                  fontsize=10, pad=6)
ax_left.legend(loc='upper left', frameon=True, framealpha=0.95,
               edgecolor='#cccccc', fontsize=9,
               handlelength=1.8, borderpad=0.5)

ax_right = axes[1]
ax_right.grid(False)

for size in SIZE_ORDER:
    sub = df3[df3['instance_scale'] == size]
    grp = sub.groupby('parameter_value').agg(
        t_mean=('solve_seconds_mean', 'mean')
    ).reset_index().sort_values('parameter_value')

    rv = grp['parameter_value'].values
    tm = grp['t_mean'].values

    ax_right.plot(rv, tm, '-D',
                  color=SIZE_COLOR[size], linewidth=1.8,
                  markersize=7, markerfacecolor='white',
                  markeredgewidth=1.8, zorder=3,
                  label=SIZE_LABEL[size])

    for r, t in zip(rv, tm):
        if t >= 1000:
            label = f'{t/60:.0f}min'
        elif t >= 60:
            label = f'{t:.0f}s'
        else:
            label = f'{t:.1f}s'
        ax_right.text(r, t * 1.12,
                      label,
                      ha='center', va='bottom',
                      fontsize=7.5,
                      color=SIZE_COLOR[size])

ax_right.axvspan(REC_LO, REC_HI, alpha=0.12, color='#2CA02C', zorder=0)
ax_right.text((REC_LO + REC_HI) / 2,
              ax_right.get_ylim()[0] * 1.5
              if ax_right.get_ylim()[0] > 0 else 1,
              'Recommended\nrange',
              ha='center', va='bottom',
              fontsize=8, color='#2CA02C',
              style='italic',
              transform=ax_right.get_xaxis_transform())

ax_right.set_yscale('log')
ax_right.set_xticks(ratio_vals)
ax_right.set_xticklabels([str(r) for r in ratio_vals], fontsize=10)
ax_right.set_xlabel('Drone serviceable ratio', fontsize=10, labelpad=4)
ax_right.set_ylabel('Solve time (s, log scale)', fontsize=10)
ax_right.set_title('(b) Computational time (log scale)',
                   fontsize=10, pad=6)
ax_right.legend(loc='upper left', frameon=True, framealpha=0.95,
                edgecolor='#cccccc', fontsize=9,
                handlelength=1.8, borderpad=0.5)

fig.text(0.50, 0.01,
    'Shaded region (ratio = 0.35–0.50): solution quality degradation <2% from optimum,'
    ' while solve time is reduced by up to 75% vs. full coverage (ratio=1.0).'
    '\nThis range represents the recommended operating point for real-time deployment,'
    ' balancing coverage flexibility with computational efficiency.',
    ha='center', va='bottom', fontsize=8.5,
    color='#555555', fontfamily='Times New Roman',
    linespacing=1.5)

for ext in ['pdf', 'png']:
    fig.savefig(f'{OUT_DIR}/exp5_fig3_efficiency_frontier.{ext}')
plt.close()
print('[Fig3 Done] exp5_fig3_efficiency_frontier.pdf/.png')
