import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

EXP3_BASE = r'D:\experiment\experiment_3'
OUT_DIR = r'D:\experiment\experiment_3\output\figures'

SCHEME_FILES = {
    'C': 'scheme_c_solution_detail.json',
    'D': 'scheme_d_solution_detail.json',
    'E': 'scheme_e_solution_detail.json',
}


def route_signature(vehicle_pairs):
    """将路径结构序列化为字符串，用于比较是否相同"""
    sig_parts = []
    for vp in vehicle_pairs:
        tr = tuple(vp.get('truck_route', []))
        sorties = tuple(
            (s.get('launch_node'), s.get('recovery_node'), tuple(s.get('customers', [])))
            for s in vp.get('sorties', [])
        )
        sig_parts.append((tr, sorties))
    return str(sig_parts)


def load_instance(instance_name):
    data = {}
    for scheme, fname in SCHEME_FILES.items():
        fpath = os.path.join(EXP3_BASE, instance_name, fname)
        if not os.path.exists(fpath):
            return None
        with open(fpath, encoding='utf-8') as f:
            data[scheme] = json.load(f)
    return data


matplotlib.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


# Part A: 算例选取
# ── 直接指定三个算例（按客户数升序排列）──
SELECTED_INSTANCES = [
    'small_12c_1v_seed2',   # 12 customers
    'small_18c_2v_seed1',   # 18 customers
    'small_20c_2v_seed2',   # 20 customers
]

selected = []
for inst_name in SELECTED_INSTANCES:
    data = {}
    skip = False
    for scheme, fname in SCHEME_FILES.items():
        fpath = os.path.join(EXP3_BASE, inst_name, fname)
        if not os.path.exists(fpath):
            print(f'[WARN] 缺少文件：{fpath}，跳过算例 {inst_name}')
            skip = True
            break
        with open(fpath, encoding='utf-8') as f:
            data[scheme] = json.load(f)
    if skip:
        continue

    parts = inst_name.split('_')
    failed = {
        s: set(data[s]['online_execution']['failed_customers'])
        for s in ['E', 'D', 'C']
    }
    off_sig = {
        s: route_signature(data[s]['offline_plan']['vehicle_pairs'])
        for s in ['E', 'D', 'C']
    }
    on_sig = {
        s: route_signature(
            data[s].get('online_route', {}).get('vehicle_pairs', []))
        for s in ['E', 'D', 'C']
    }
    selected.append({
        'instance_name': inst_name,
        'n_customers': int(parts[1].replace('c', '')),
        'seed': int(parts[3].replace('seed', '')),
        'failed_e': failed['E'],
        'failed_d': failed['D'],
        'failed_c': failed['C'],
        'route_diffs': {s: off_sig[s] != on_sig[s]
                        for s in ['E', 'D', 'C']},
        'data': data,
    })

if len(selected) < 3:
    raise RuntimeError(
        f'只加载到 {len(selected)} 个算例，'
        '请检查 EXP3_BASE 目录下对应算例的 JSON 文件是否存在')

print('[Instance Selection] 使用指定算例：')
for i, s in enumerate(selected):
    print(f'  Row {i+1}: {s["instance_name"]}  '
          f'n_customers={s["n_customers"]}')
    for scheme in ['E', 'D', 'C']:
        print(f'    Scheme {scheme}: '
              f'route_changed={s["route_diffs"][scheme]}  '
              f'failed={sorted(s[f"failed_{scheme.lower()}"])}')


# Part B: 绘图函数
SCHEME_COLOR = {
    'E': '#D62728',
    'D': '#E8800A',
    'C': '#2CA02C',
}
TRUCK_COLORS = ['#4878CF', '#D85A30', '#3B8B3B', '#7F77DD']


def draw_vehicle_pairs(
    ax,
    vehicle_pairs,
    customers_dict,
    depot_pos,
    color,
    lw,
    alpha,
    zorder,
    linestyle='-',
    arrow=True,
):
    """在 ax 上绘制一套 vehicle_pairs 路径"""

    def gp(nid):
        return customers_dict.get(nid, depot_pos)

    for vi, vp in enumerate(vehicle_pairs):
        truck_color = color if color else TRUCK_COLORS[vi % len(TRUCK_COLORS)]
        route = vp.get('truck_route', [])
        if len(route) < 2:
            continue

        xs = [gp(r)[0] for r in route]
        ys = [gp(r)[1] for r in route]
        ax.plot(xs, ys, linestyle=linestyle, color=truck_color, lw=lw, alpha=alpha, zorder=zorder)
        if arrow and len(xs) >= 2:
            ax.annotate(
                '',
                xy=(xs[1], ys[1]),
                xytext=(xs[0], ys[0]),
                arrowprops=dict(arrowstyle='->', color=truck_color, lw=lw * 0.8),
            )

        for sortie in vp.get('sorties', []):
            nodes = [sortie.get('launch_node', 0)] + sortie.get('customers', []) + [sortie.get('recovery_node', 0)]
            sx = [gp(n)[0] for n in nodes]
            sy = [gp(n)[1] for n in nodes]
            ax.plot(sx, sy, linestyle='--', color=truck_color, lw=lw * 0.7, alpha=alpha * 0.85, zorder=zorder)


def draw_route(ax, scheme_data, scheme_label, failed_ids, is_leftmost=False):
    """
    绘制一个方案的路径子图。
    - 灰色细线：离线计划路径（参考）
    - 彩色粗线：在线执行路径（实际）
    - 两者不同时，彩色线会与灰色线在结构上有明显差异
    """
    cust_list = scheme_data['customers']
    cust_dict = {c['id']: (c['x'], c['y']) for c in cust_list}
    depot_pos = (
        scheme_data['offline_plan']['depot']['x'],
        scheme_data['offline_plan']['depot']['y'],
    )
    offline_vp = scheme_data['offline_plan']['vehicle_pairs']
    online_vp = scheme_data.get('online_route', {}).get('vehicle_pairs', [])

    ax.set_aspect('equal', adjustable='datalim')
    ax.tick_params(labelsize=9)
    ax.grid(False)
    ax.set_xlabel('x (km)', fontsize=10, labelpad=2)
    if is_leftmost:
        ax.set_ylabel('y (km)', fontsize=10, labelpad=2)
    else:
        ax.set_ylabel('')
        ax.set_yticklabels([])
        ax.tick_params(axis='y', length=0)

    draw_vehicle_pairs(
        ax,
        offline_vp,
        cust_dict,
        depot_pos,
        color='#C8C8C8',
        lw=0.9,
        alpha=0.60,
        zorder=2,
        linestyle='-',
        arrow=False,
    )

    scheme_color = SCHEME_COLOR[scheme_label]
    if online_vp:
        draw_vehicle_pairs(
            ax,
            online_vp,
            cust_dict,
            depot_pos,
            color=scheme_color,
            lw=1.6,
            alpha=0.85,
            zorder=4,
            linestyle='-',
            arrow=True,
        )

    for c in cust_list:
        cx, cy = c['x'], c['y']
        cid = c['id']
        if cid in failed_ids:
            ax.plot(cx, cy, 'x', color='#D62728', markersize=5, markeredgewidth=1.3, zorder=7)
            if scheme_label == 'E':
                ax.text(
                    cx + 0.04,
                    cy + 0.04,
                    'skip',
                    fontsize=7,
                    color='#D62728',
                    va='bottom',
                    ha='left',
                    style='italic',
                    zorder=8,
                )
        else:
            ax.plot(cx, cy, 'o', color='#333333', markersize=4.5, zorder=4)

    ax.plot(*depot_pos, 's', color='#111111', markersize=7, zorder=8)
    t = ax.text(depot_pos[0] + 0.04, depot_pos[1] + 0.04, 'Depot', fontsize=8, color='#333333')
    t.set_path_effects([pe.withStroke(linewidth=1.5, foreground='white')])

    ax.text(
        0.97,
        0.03,
        f'failed={len(failed_ids)}',
        transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=9,
        color=SCHEME_COLOR[scheme_label],
        style='italic',
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85),
    )


# Part C: 布局与绘制
FIG_W = 18.0
LEGEND_W_INCH = 2.4
PATH_W_FRAC = 1.0 - (LEGEND_W_INCH / FIG_W)
legend_ratio = LEGEND_W_INCH / (FIG_W * PATH_W_FRAC / 3)

fig = plt.figure(figsize=(FIG_W, 9.5))
gs = gridspec.GridSpec(
    3,
    4,
    figure=fig,
    width_ratios=[1, 1, 1, legend_ratio],
    hspace=0.32,
    wspace=0.3,
)
fig.subplots_adjust(left=0.05, right=PATH_W_FRAC - 0.01, top=0.93, bottom=0.09)

axes = [[fig.add_subplot(gs[row, col]) for col in range(3)] for row in range(3)]

col_cfg = [
    ('Scheme E', '#888888'),
    ('Scheme D', '#E8800A'),
    ('Scheme C', '#C0392B'),
]
for col, (title, color) in enumerate(col_cfg):
    axes[0][col].set_title(
        title,
        fontsize=12,
        pad=7,
        color=color,
        fontfamily='Times New Roman',
    )

SCHEME_ORDER = ['E', 'D', 'C']
for row, inst in enumerate(selected):
    for col, scheme in enumerate(SCHEME_ORDER):
        failed_ids = inst[f'failed_{scheme.lower()}']
        draw_route(
            axes[row][col],
            inst['data'][scheme],
            scheme,
            failed_ids=failed_ids,
            is_leftmost=(col == 0),
        )

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
fig_h_px = fig.get_size_inches()[1] * fig.dpi


def ax_tight_bottom(ax):
    bbox = ax.get_tightbbox(renderer)
    return (bbox.y0 / fig_h_px) if bbox else ax.get_position().y0


def ax_tight_top(ax):
    bbox = ax.get_tightbbox(renderer)
    return (bbox.y1 / fig_h_px) if bbox else ax.get_position().y1


for row, inst in enumerate(selected):
    row_bottom = min(ax_tight_bottom(axes[row][c]) for c in range(3))

    if row < 2:
        next_top = max(ax_tight_top(axes[row + 1][c]) for c in range(3))
        label_y = (row_bottom + next_top) / 2.0
    else:
        label_y = max(0.015, (row_bottom + 0.0) / 2.0)

    left_x = axes[row][0].get_position().x0
    right_x = axes[row][2].get_position().x1
    center_x = (left_x + right_x) / 2.0

    fe = sorted(inst['failed_e'])
    fd = sorted(inst['failed_d'])
    fc_s = sorted(inst['failed_c'])


    def format_instance_name(name):
        return name.replace('seed1', 'c').replace('seed2', 'r').replace('seed3', 'm')


    fig.text(
        center_x,
        label_y,
        f'{format_instance_name(inst["instance_name"])}    failed — E: {len(fe)}  D: {len(fd)}  C: {len(fc_s)}',
        ha='center',
        va='center',
        fontsize=12,
        color='#222222',
        fontfamily='Times New Roman',
    )


# Part D: 图例
legend_ax = fig.add_subplot(gs[:, 3])
legend_ax.set_visible(False)

all_pos = [axes[r][c].get_position() for r in range(3) for c in range(3)]
ax_top = max(p.y1 for p in all_pos)
ax_bottom = min(p.y0 for p in all_pos)
ax_mid = (ax_top + ax_bottom) / 2.0
right_x = axes[0][2].get_position().x1

legend_handles = [
    Line2D([0], [0], color='#C8C8C8', lw=1.2, label='Offline plan'),
    Line2D([0], [0], color='#888888', lw=1.6, label='E: online route'),
    Line2D([0], [0], color='#E8800A', lw=1.6, label='D: online route'),
    Line2D([0], [0], color='#2CA02C', lw=1.6, label='C: online route'),
    Line2D([0], [0], color='#888888', lw=1.0, linestyle='--', label='Drone sortie'),
    Line2D([0], [0], marker='s', color='#111111', markersize=8, lw=0, label='Depot'),
    Line2D([0], [0], marker='o', color='#333333', markersize=5, lw=0, label='Customer\n(home)'),
    Line2D([0], [0], marker='x', color='#D62728', markersize=5, markeredgewidth=1.3, lw=0, label='Not home'),
]

fig.legend(
    handles=legend_handles,
    loc='center left',
    bbox_to_anchor=(right_x + 0.018, ax_mid),
    bbox_transform=fig.transFigure,
    frameon=True,
    framealpha=0.95,
    edgecolor='#cccccc',
    fontsize=10,
    handlelength=2.0,
    handleheight=0.88,
    borderpad=0.8,
    labelspacing=0.58,
    title='Legend',
    title_fontsize=11,
)


# Part E: 保存
os.makedirs(OUT_DIR, exist_ok=True)
for ext in ['pdf', 'png']:
    fig.savefig(os.path.join(OUT_DIR, f'exp3_fig4_route_illustration.{ext}'))
plt.close(fig)

print('[Fig4 Done] exp3_fig4_route_illustration 已生成')
print(f'选取算例：{[s["instance_name"] for s in selected]}')
for s in selected:
    print(
        f'  {s["instance_name"]}: '
        f'route_diff C={s["route_diffs"]["C"]} '
        f'D={s["route_diffs"]["D"]} '
        f'E={s["route_diffs"]["E"]}'
    )
