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

df2 = pd.read_csv(r'D:\experiment\experiment_5\exp5_2_speed_summary.csv')
speed_vals = sorted(df2['drone_speed_factor'].unique())

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
plt.subplots_adjust(left=0.08, right=0.97,
                    top=0.88, bottom=0.16,
                    wspace=0.35)

ax_left = axes[0]
ax_left.grid(False)

for size in SIZE_ORDER:
    sub = df2[df2['instance_scale'] == size]
    grp = sub.groupby('drone_speed_factor').agg(
        z_mean=('z_total_mean', 'mean'),
        z_std =('z_total_mean', 'std'),
    ).reset_index().sort_values('drone_speed_factor')

    sv = grp['drone_speed_factor'].values
    zm = grp['z_mean'].values
    zs = grp['z_std'].fillna(0).values

    ax_left.fill_between(sv, zm - zs, zm + zs,
                         alpha=0.12, color=SIZE_COLOR[size], zorder=1)
    ax_left.plot(sv, zm, '-D',
                 color=SIZE_COLOR[size], linewidth=1.8,
                 markersize=7, markerfacecolor='white',
                 markeredgewidth=1.8, zorder=3,
                 label=SIZE_LABEL[size])

y_lo, y_hi = ax_left.get_ylim()
ax_left.axvspan(0.75, 1.15, alpha=0.10, color='#888888', zorder=0)
ax_left.text(0.95, 0.97,
             'Typical commercial\ndrone speed range',
             ha='center', va='top',
             transform=ax_left.transAxes,
             fontsize=7.5, color='#888888',
             style='italic',
             bbox=dict(boxstyle='round,pad=0.2',
                       fc='white', ec='none', alpha=0.7))

ax_left.set_xlabel('Drone speed factor', fontsize=10, labelpad=4)
ax_left.set_ylabel(r'$Z_{\mathrm{total}}$ (mean)', fontsize=10)
ax_left.set_title('(a) Absolute planning cost vs. drone speed',
                  fontsize=10, pad=6)
ax_left.legend(loc='center right', frameon=True, framealpha=0.95,
               edgecolor='#cccccc', fontsize=9,
               handlelength=1.8, borderpad=0.5)

ax_right = axes[1]
ax_right.grid(False)

for size in SIZE_ORDER:
    sub = df2[df2['instance_scale'] == size]
    grp = sub.groupby('drone_speed_factor').agg(
        z_mean=('z_total_mean', 'mean')
    ).reset_index().sort_values('drone_speed_factor')

    sv = grp['drone_speed_factor'].values
    zm = grp['z_mean'].values

    marginal = []
    mid_speeds = []
    for i in range(len(sv) - 1):
        improvement = (zm[i] - zm[i + 1]) / zm[i] * 100
        marginal.append(improvement)
        mid_speeds.append((sv[i] + sv[i + 1]) / 2)

    ax_right.plot(mid_speeds, marginal, '-o',
                  color=SIZE_COLOR[size], linewidth=1.8,
                  markersize=7, markerfacecolor='white',
                  markeredgewidth=1.8, zorder=3,
                  label=SIZE_LABEL[size])

ax_right.axhline(0, color='#333333', lw=0.9,
                 linestyle='--', alpha=0.6, zorder=1)
ax_right.text(speed_vals[-2] - 0.05, 0.08,
              'No improvement',
              fontsize=7.5, color='#888888',
              style='italic', va='bottom')

y_lo_r, y_hi_r = ax_right.get_ylim()
ax_right.axhspan(-0.5, 0.5,
                 alpha=0.08, color='#D6604D', zorder=0)
ax_right.text(mid_speeds[len(mid_speeds)//2], 0.55,
              'Near-zero ROI zone',
              ha='center', va='bottom',
              fontsize=7.5, color='#D6604D',
              style='italic')

ax_right.set_xlabel('Drone speed factor (interval midpoint)',
                    fontsize=10, labelpad=4)
ax_right.set_ylabel('Marginal cost improvement (%)', fontsize=10)
ax_right.set_title('(b) Marginal return on speed improvement',
                   fontsize=10, pad=6)
ax_right.legend(loc='upper right', frameon=True, framealpha=0.95,
                edgecolor='#cccccc', fontsize=9,
                handlelength=1.8, borderpad=0.5)

fig.text(0.50, 0.01,
    'The marginal cost improvement from increasing drone speed remains near zero across all instance scales.'
    '\nProcurement should prioritize battery endurance and payload capacity over speed;'
    ' speed factor 0.75–1.15 covers the commercially available range with negligible performance difference.',
    ha='center', va='bottom', fontsize=8.5,
    color='#555555', fontfamily='Times New Roman',
    linespacing=1.5)

for ext in ['pdf', 'png']:
    fig.savefig(f'{OUT_DIR}/exp5_fig2_speed_investment.{ext}')
plt.close()
print('[Fig2 Done] exp5_fig2_speed_investment.pdf/.png')
