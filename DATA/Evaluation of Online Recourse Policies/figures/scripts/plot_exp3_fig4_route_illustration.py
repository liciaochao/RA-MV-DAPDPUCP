import os
import json
import random
from pathlib import Path
import sys

VENDOR = Path(r"D:\codex_drone\.vendor")
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

EXP1_BASE = r'D:\experiment\experiment_1\ACO_ALNS'
EXP3_BASE = r'D:\experiment\experiment_3'
OUT_DIR = r'D:\experiment\experiment_3\output\figures'

matplotlib.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def load_instance_data(instance_name):
    coord_path = os.path.join(EXP1_BASE, instance_name, 'customer_service_report.json')
    route_path = os.path.join(EXP1_BASE, instance_name, 'solution_detail.json')
    comp_path = os.path.join(EXP3_BASE, instance_name, 'experiment3_comparison.json')

    for p in [coord_path, route_path, comp_path]:
        if not os.path.exists(p):
            print(f'[WARN] 文件缺失：{p}，跳过 {instance_name}')
            return None

    with open(coord_path, 'r', encoding='utf-8') as f:
        csr = json.load(f)
    customers_dict = {int(c['customer_id']): (float(c['x']), float(c['y'])) for c in csr['customers']}

    with open(route_path, 'r', encoding='utf-8') as f:
        sol = json.load(f)
    vehicle_pairs = sol.get('vehicle_pairs', [])
    depot_raw = sol.get('depot', {})
    depot_pos = (float(depot_raw.get('x', 0.0)), float(depot_raw.get('y', 0.0)))

    with open(comp_path, 'r', encoding='utf-8') as f:
        comp = json.load(f)

    mc_e = comp['scheme_e']['monte_carlo']
    avg_fail_e = float(mc_e['average_failed_customers'])
    avg_fail_c = float(comp['scheme_c']['monte_carlo']['average_failed_customers'])
    avg_fail_d = float(comp['scheme_d']['monte_carlo']['average_failed_customers'])
    failed_samples = mc_e['failed_count_samples']

    diffs = [abs(v - avg_fail_e) for v in failed_samples]
    typical_idx = int(np.argmin(diffs))
    n_fail_scenario = int(failed_samples[typical_idx])

    random.seed(int(comp['scenario_seed']) + typical_idx)
    all_ids = list(customers_dict.keys())
    n_to_fail = min(int(round(n_fail_scenario)), len(all_ids))
    failed_ids = set(random.sample(all_ids, n_to_fail)) if n_to_fail > 0 else set()

    parts = instance_name.split('_')
    return {
        'instance_name': instance_name,
        'n_customers': int(parts[1].replace('c', '')),
        'seed': int(parts[3].replace('seed', '')),
        'customers_dict': customers_dict,
        'depot_pos': depot_pos,
        'vehicle_pairs': vehicle_pairs,
        'failed_ids': failed_ids,
        'n_fail_scenario': n_fail_scenario,
        'avg_fail_e': avg_fail_e,
        'avg_fail_d': avg_fail_d,
        'avg_fail_c': avg_fail_c,
    }


def draw_route_compact(ax, vehicle_pairs, customers_dict, depot_pos, failed_ids,
                       show_replan=False, is_leftmost_col=False):
    ax.set_aspect('equal', adjustable='datalim')
    ax.tick_params(labelsize=7.5)
    ax.grid(True, alpha=0.22, linewidth=0.35, linestyle='--')
    ax.set_xlabel('x', fontsize=8, labelpad=3)
    if is_leftmost_col:
        if not ax.get_ylabel():
            ax.set_ylabel('y', fontsize=8, labelpad=3)
    else:
        ax.set_yticklabels([])

    truck_colors = ['#4878CF', '#D85A30', '#3B8B3B', '#7F77DD']

    def get_pos(nid):
        if nid == 0 or nid not in customers_dict:
            return depot_pos
        return customers_dict[nid]

    for vi, vp in enumerate(vehicle_pairs):
        color = truck_colors[vi % len(truck_colors)]
        route = vp.get('truck_route', [])
        if len(route) < 2:
            continue

        xs = [get_pos(r)[0] for r in route]
        ys = [get_pos(r)[1] for r in route]
        ax.plot(xs, ys, '-', color=color, linewidth=1.3, alpha=0.70, zorder=2)
        ax.annotate('', xy=(xs[1], ys[1]), xytext=(xs[0], ys[0]),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.0))

        for sortie in vp.get('sorties', []):
            nodes = [sortie.get('launch_node', 0)] + sortie.get('customers', []) + [sortie.get('recovery_node', 0)]
            sx = [get_pos(n)[0] for n in nodes]
            sy = [get_pos(n)[1] for n in nodes]
            ax.plot(sx, sy, '--', color=color, linewidth=0.85, alpha=0.60, zorder=2)

    for cid, (cx, cy) in customers_dict.items():
        if cid in failed_ids:
            ax.plot(cx, cy, 'x', color='#D62728', markersize=9, markeredgewidth=2.2, zorder=5)
            if not show_replan:
                ax.text(cx + 0.06, cy + 0.06, 'skip', fontsize=6, color='#D62728',
                        va='bottom', ha='left', style='italic')
            else:
                ax.plot(cx, cy, 'o', color='#2CA02C', markersize=16, alpha=0.28,
                        markeredgecolor='#2CA02C', markeredgewidth=2.0, zorder=4)
        else:
            ax.plot(cx, cy, 'o', color='#333333', markersize=5.5, zorder=3)

    ax.plot(*depot_pos, 's', color='#111111', markersize=9, zorder=6)
    ax.annotate('Depot', xy=depot_pos, xytext=(5, 4), textcoords='offset points',
                fontsize=7.5, color='#333333')


def main():
    df_main = pd.read_csv(r'D:\experiment\experiment_3\output\exp3_main_table.csv')
    small_df = df_main[df_main['size'] == 'Small'].copy()
    small_df['fail_improvement'] = small_df['E_avg_fail'] - small_df['C_avg_fail']

    candidates = (small_df[small_df['fail_improvement'] > 0]
                  .sort_values('fail_improvement', ascending=False))
    if len(candidates) < 3:
        candidates = small_df.sort_values('delta_fail_C_vs_E_pct', ascending=False)

    selected_instances = []
    for _, row in candidates.iterrows():
        if len(selected_instances) >= 3:
            break
        data = load_instance_data(row['instance'])
        if data is not None:
            selected_instances.append(data)

    if len(selected_instances) < 3:
        raise RuntimeError('有效算例不足3个，请检查实验一ACO_ALNS目录')

    print('[Fig4] 选取算例：')
    for i, inst in enumerate(selected_instances):
        print(f'  Col {i+1}: {inst["instance_name"]}  improvement={inst["avg_fail_e"] - inst["avg_fail_c"]:.2f}')

    fig = plt.figure(figsize=(11, 7.5))
    gs = gridspec.GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 0.28], hspace=0.42, wspace=0.32)
    fig.subplots_adjust(left=0.07, right=0.83, top=0.88, bottom=0.11)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]

    row_cfg = [('Scheme E\n(no adjustment)', '#666666'),
               ('Scheme C\n(full framework)', '#C0392B')]
    for row, (label, color) in enumerate(row_cfg):
        axes[row][0].set_ylabel(label, fontsize=10,
                                fontweight='500', labelpad=10, color=color,
                                fontfamily='Times New Roman')

    for col, inst in enumerate(selected_instances):
        nc, seed = inst['n_customers'], inst['seed']
        fe, fd, fc = inst['avg_fail_e'], inst['avg_fail_d'], inst['avg_fail_c']
        delta = fe - fc
        n_sc = inst['n_fail_scenario']
        vp = inst['vehicle_pairs']
        cust = inst['customers_dict']
        depot = inst['depot_pos']
        failed = inst['failed_ids']

        axes[0][col].set_title(
            f'{nc}c-1v  seed {seed}\n'
            f'Avg failed — E:{fe:.1f} D:{fd:.1f} C:{fc:.1f} (↓{delta:.1f})',
            fontsize=8.5, pad=6, color='#333333',
            linespacing=1.55, fontfamily='Times New Roman')

        for row, replan in enumerate([False, True]):
            draw_route_compact(
                axes[row][col], vp, cust, depot, failed,
                show_replan=replan,
                is_leftmost_col=(col == 0))

            if not replan:
                label_txt = f'scenario: {n_sc} absent'
                label_col = '#D62728'
            else:
                label_txt = f'replanned≈{n_sc}\nfailed≈{fc:.1f}'
                label_col = '#2CA02C'

            corner_x = 0.03 if col == 2 else 0.97
            corner_ha = 'left' if col == 2 else 'right'

            axes[row][col].text(
                corner_x, 0.03, label_txt,
                transform=axes[row][col].transAxes,
                ha=corner_ha, va='bottom', fontsize=7,
                color=label_col, style='italic',
                bbox=dict(boxstyle='round,pad=0.2',
                          fc='white', ec='none', alpha=0.85))

    legend_ax = fig.add_subplot(gs[:, 3])
    legend_ax.set_visible(False)

    legend_handles = [
        Line2D([0], [0], color='#4878CF', lw=1.5, label='Truck route'),
        Line2D([0], [0], color='#4878CF', lw=0.9, linestyle='--', label='Drone sortie'),
        Line2D([0], [0], marker='s', color='#111111', markersize=8, lw=0, label='Depot'),
        Line2D([0], [0], marker='o', color='#333333', markersize=5, lw=0, label='Customer (home)'),
        Line2D([0], [0], marker='x', color='#D62728', markersize=8, markeredgewidth=2.0, lw=0, label='Not home'),
        mpatches.Patch(facecolor='#2CA02C', alpha=0.28, edgecolor='#2CA02C', linewidth=2.0, label='Scheme C:\nreplanned'),
    ]

    fig.legend(
        handles=legend_handles,
        loc='upper left',
        bbox_to_anchor=(0.842, 0.72),
        bbox_transform=fig.transFigure,
        frameon=True,
        framealpha=0.95,
        edgecolor='#cccccc',
        fontsize=8.5,
        handlelength=1.8,
        handleheight=0.9,
        borderpad=0.7,
        labelspacing=0.6,
        title='Legend',
        title_fontsize=9,
    )

    fig.suptitle(
        'Three Small instances with the largest failure reduction '
        'by Scheme C vs. Scheme E '
        '(offline plan is identical across both rows).',
        fontsize=9, color='#555555', y=0.97,
        style='italic', fontfamily='Times New Roman')

    fig.text(
        0.46, 0.015,
        'Red × = customer not home in representative '
        'MC scenario (closest to E’s average).  '
        'Green circle (row 2) = Scheme C replanning.  '
        'Column headers show per-scheme average failed customers '
        'over 100 MC trials.  '
        'Small absolute improvements (≈0.2–0.4) reflect '
        'that Small instances have few customers, '
        'limiting replanning gains.',
        ha='center', va='bottom',
        fontsize=7.5, color='#666666',
        linespacing=1.45,
        fontfamily='Times New Roman')

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(OUT_DIR, f'exp3_fig4_route_illustration.{ext}'))
    plt.close(fig)

    print('[Step 4 Done] exp3_fig4 2×3 路径对比图已生成')


if __name__ == '__main__':
    main()
