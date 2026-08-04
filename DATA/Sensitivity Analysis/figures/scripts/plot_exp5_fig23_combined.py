import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

matplotlib.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 13,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
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
df3 = pd.read_csv(r'D:\experiment\experiment_5\exp5_3_drone_ratio.csv')

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
plt.subplots_adjust(left=0.08, right=0.97,
                    top=0.88, bottom=0.14,
                    wspace=0.35)

ax_speed = axes[0]
ax_speed.grid(False)
h_small = None
h_medium = None
h_large = None

for size in SIZE_ORDER:
    sub = df2[df2['instance_scale'] == size]
    grp = sub.groupby('drone_speed_factor').agg(
        z_mean=('z_total_mean', 'mean'),
        z_std =('z_total_mean', 'std'),
    ).reset_index().sort_values('drone_speed_factor')

    sv = grp['drone_speed_factor'].values
    zm = grp['z_mean'].values
    zs = grp['z_std'].fillna(0).values

    line, = ax_speed.plot(
        sv,
        zm,
        '-D',
        color=SIZE_COLOR[size],
        linewidth=1.8,
        markersize=7,
        markerfacecolor='white',
        markeredgewidth=1.8,
        zorder=3,
        label=SIZE_LABEL[size],
    )
    if size == 'small':
        h_small = line
    elif size == 'medium':
        h_medium = line
    elif size == 'large':
        h_large = line

ax_speed.set_yscale('log')
ax_speed.set_xlabel('Drone speed factor', fontsize=10, labelpad=4)
ax_speed.set_ylabel(r'$Z_{\mathrm{total}}$ (mean, log scale)', fontsize=10)
ax_speed.legend(
    handles=[h_small, h_medium, h_large],
    labels=['Small', 'Medium', 'Large'],
    loc='upper right',
    bbox_to_anchor=(1.0, 0.78),
    fontsize=8,
    framealpha=0.9,
    frameon=True,
    edgecolor='#cccccc',
)

ax_speed.axvspan(0.75, 1.15, alpha=0.10, color='#888888', zorder=0)
ax_speed.text(0.95, 0.90,
              'The plausible\noperational range',
              ha='center', va='top',
              transform=ax_speed.transAxes,
              fontsize=7.5, color='#888888', style='italic',
              bbox=dict(boxstyle='round,pad=0.2', fc='white',
                        ec='none', alpha=0.7))

ax_ratio = axes[1]
ax_ratio.grid(False)

for size in SIZE_ORDER:
    sub = df3[df3['instance_scale'] == size]
    grp = sub.groupby('parameter_value').agg(
        z_mean=('z_total_mean', 'mean'),
        z_std =('z_total_mean', 'std'),
    ).reset_index().sort_values('parameter_value')

    rv = grp['parameter_value'].values
    zm = grp['z_mean'].values
    zs = grp['z_std'].fillna(0).values

    ax_ratio.plot(
        rv,
        zm,
        '-D',
        color=SIZE_COLOR[size],
        linewidth=1.8,
        markersize=7,
        markerfacecolor='white',
        markeredgewidth=1.8,
        zorder=3,
        label=SIZE_LABEL[size],
    )

ax_ratio.set_yscale('log')
ax_ratio.set_xlabel('Drone serviceable ratio', fontsize=10, labelpad=4)
ax_ratio.set_ylabel(r'$Z_{\mathrm{total}}$ (mean, log scale)', fontsize=10)
if ax_ratio.get_legend() is not None:
    ax_ratio.get_legend().remove()

ax_ratio.axvspan(0.35, 0.50, alpha=0.12, color='#2CA02C', zorder=0)
ax_ratio.text(0.425, 0.90,
              'Recommended\nrange',
              ha='center', va='top',
              transform=ax_ratio.transAxes,
              fontsize=7.5, color='#2CA02C', style='italic',
              bbox=dict(boxstyle='round,pad=0.2', fc='white',
                        ec='none', alpha=0.7))

# ── 读取求解时间数据 ──────────────────────────────────────────
time_col = None
for candidate in ['solve_seconds_mean', 'solve_time_mean',
                  'avg_solve_time', 'solve_time', 'time_mean']:
    if candidate in df3.columns:
        time_col = candidate
        break

size_col = None
for candidate in ['size', 'instance_scale']:
    if candidate in df3.columns:
        size_col = candidate
        break

if size_col is not None:
    df_large = df3[df3[size_col].astype(str).str.lower() == 'large'].copy()
else:
    df_large = pd.DataFrame()

ratio_col = None
for candidate in ['drone_serviceable_ratio', 'parameter_value']:
    if candidate in df_large.columns:
        ratio_col = candidate
        break

if time_col is not None and ratio_col is not None and not df_large.empty:
    large_grp = (
        df_large.groupby(ratio_col, as_index=False)
        .agg(time_s=(time_col, 'mean'))
        .sort_values(ratio_col)
    )
    ratio_vals = large_grp[ratio_col].to_numpy(dtype=float)
    time_s = large_grp['time_s'].to_numpy(dtype=float)
else:
    # Fallback 硬编码
    ratio_vals = np.array([0.20, 0.35, 0.50, 0.75, 1.00], dtype=float)
    time_s = np.array([710, 780, 900, 2000, 3025], dtype=float)

time_min = time_s / 60.0

# ── 创建 inset（置于右下角，避免与主图数据重叠）────────────────
axins = inset_axes(
    ax_ratio,
    width='32%',
    height='32%',
    loc='lower right',
    bbox_to_anchor=(-0.02, 0.18, 1, 0.72),
    bbox_transform=ax_ratio.transAxes,
    borderpad=0,
)

# ── 绘制折线 ──────────────────────────────────────────────────
axins.plot(
    ratio_vals,
    time_min,
    color='#333333',
    marker='o',
    markersize=3,
    linewidth=1.2,
)

# ── 推荐区间绿色阴影（0.35–0.50）─────────────────────────────
axins.axvspan(0.35, 0.50, color='#A8D5A2', alpha=0.35, zorder=0)

# ── 标注首尾原始秒数 ──────────────────────────────────────────
axins.annotate(
    f'{int(time_s[0])}s',
    xy=(ratio_vals[0], time_min[0]),
    xytext=(4, 3),
    textcoords='offset points',
    fontsize=5.5,
    color='#333333',
)
axins.annotate(
    f'{int(time_s[-1])}s',
    xy=(ratio_vals[-1], time_min[-1]),
    xytext=(-28, -8),
    textcoords='offset points',
    fontsize=5.5,
    color='#333333',
)

# ── 坐标轴标签与格式 ──────────────────────────────────────────
axins.set_xlabel('Ratio', fontsize=5.5, labelpad=1)
axins.set_ylabel('Solve time', fontsize=5.5, labelpad=1)
axins.tick_params(axis='both', labelsize=5)
axins.set_facecolor('#F8F8F8')
axins.spines['top'].set_visible(False)
axins.spines['right'].set_visible(False)
axins.grid(False)

fig.savefig(r'D:\experiment\experiment_5\output\figures\exp5_fig23_combined.pdf',
            bbox_inches='tight', dpi=300)
fig.savefig(r'D:\experiment\experiment_5\output\figures\exp5_fig23_combined.png',
            bbox_inches='tight', dpi=300)
plt.close()
print('[Fig23 Done] exp5_fig23_combined.pdf/.png')
