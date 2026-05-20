import os, json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

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

EXP3_BASE = r'D:\experiment\experiment_3'

SCHEME_FILES = {
    'E': 'scheme_e_solution_detail.json',
    'D': 'scheme_d_solution_detail.json',
    'C': 'scheme_c_solution_detail.json',
}
SCHEME_COLOR = {
    'E': '#D62728',
    'D': '#E8800A',
    'C': '#2CA02C',
}
SCHEME_TITLE = {
    'E': ('Scheme E', 'no adjustment',  '#888888'),
    'D': ('Scheme D', 'basic repair',   '#E8800A'),
    'C': ('Scheme C', 'full framework', '#C0392B'),
}

# ────────────────────────────────────────────────────────────
# 工具函数
# ────────────────────────────────────────────────────────────

def route_signature(vehicle_pairs):
    """路径结构签名，用于判断在线路径是否与离线计划不同"""
    parts = []
    for vp in vehicle_pairs:
        tr = tuple(vp.get('truck_route', []))
        sorties = tuple(
            (s.get('launch_node'), s.get('recovery_node'),
             tuple(s.get('customers', [])))
            for s in vp.get('sorties', [])
        )
        parts.append((tr, sorties))
    return str(parts)


def draw_vehicle_pairs(ax, vehicle_pairs, cust_dict,
                       depot_pos, color, lw, alpha,
                       zorder, linestyle='-', arrow=True):
    """绘制一套 vehicle_pairs 的卡车路径和无人机 sorties"""
    def gp(nid):
        return cust_dict.get(nid, depot_pos)

    for vp in vehicle_pairs:
        route = vp.get('truck_route', [])
        if len(route) < 2:
            continue
        xs = [gp(r)[0] for r in route]
        ys = [gp(r)[1] for r in route]
        ax.plot(xs, ys, linestyle=linestyle, color=color,
                lw=lw, alpha=alpha, zorder=zorder)
        if arrow and len(xs) >= 2:
            ax.annotate('',
                xy=(xs[1], ys[1]), xytext=(xs[0], ys[0]),
                arrowprops=dict(arrowstyle='->',
                                color=color, lw=lw * 0.8))
        for sortie in vp.get('sorties', []):
            nodes = (
                [sortie.get('launch_node', 0)]
                + sortie.get('customers', [])
                + [sortie.get('recovery_node', 0)]
            )
            sx = [gp(n)[0] for n in nodes]
            sy = [gp(n)[1] for n in nodes]
            ax.plot(sx, sy, linestyle='--', color=color,
                    lw=lw * 0.70, alpha=alpha * 0.85,
                    zorder=zorder)


def draw_route(ax, scheme_data, scheme_label,
               failed_ids, is_leftmost=False):
    """在 ax 上绘制单方案路径图"""
    cust_list  = scheme_data['customers']
    cust_dict  = {c['id']: (c['x'], c['y']) for c in cust_list}
    depot_pos  = (
        scheme_data['offline_plan']['depot']['x'],
        scheme_data['offline_plan']['depot']['y'],
    )
    offline_vp = scheme_data['offline_plan']['vehicle_pairs']
    online_vp  = (scheme_data.get('online_route', {})
                  .get('vehicle_pairs', []))

    ax.set_aspect('equal', adjustable='datalim')
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.18, linewidth=0.28, linestyle='--')
    ax.set_xlabel('x (km)', fontsize=8, labelpad=2)
    if is_leftmost:
        ax.set_ylabel('y (km)', fontsize=8, labelpad=2)
    else:
        ax.set_ylabel('')
        ax.set_yticklabels([])
        ax.tick_params(axis='y', length=0)

    # 层1：离线计划路径（浅灰，参考）
    draw_vehicle_pairs(ax, offline_vp, cust_dict, depot_pos,
                       color='#C8C8C8', lw=0.9, alpha=0.60,
                       zorder=2, linestyle='-', arrow=False)

    # 层2：在线执行路径（方案色，粗线）
    if online_vp:
        draw_vehicle_pairs(ax, online_vp, cust_dict, depot_pos,
                           color=SCHEME_COLOR[scheme_label],
                           lw=1.6, alpha=0.85,
                           zorder=4, linestyle='-', arrow=True)

    # 层3：客户点
    for c in cust_list:
        cx, cy = c['x'], c['y']
        cid = c['id']
        if cid in failed_ids:
            ax.plot(cx, cy, 'x', color='#D62728',
                    markersize=5, markeredgewidth=1.3, zorder=7)
            if scheme_label == 'E':
                ax.text(cx + 0.04, cy + 0.04, 'skip',
                        fontsize=5.5, color='#D62728',
                        va='bottom', ha='left',
                        style='italic', zorder=8)
        else:
            ax.plot(cx, cy, 'o', color='#333333',
                    markersize=4.5, zorder=4)

    # 层4：仓库
    ax.plot(*depot_pos, 's', color='#111111',
            markersize=7, zorder=8)
    t = ax.text(depot_pos[0] + 0.04, depot_pos[1] + 0.04,
                'Depot', fontsize=6.5, color='#333333')
    t.set_path_effects([pe.withStroke(linewidth=1.5,
                                      foreground='white')])

    # 右下角：失败数
    ax.text(0.97, 0.03,
            f'failed={len(failed_ids)}',
            transform=ax.transAxes,
            ha='right', va='bottom', fontsize=7.5,
            color=SCHEME_COLOR[scheme_label],
            style='italic',
            bbox=dict(boxstyle='round,pad=0.2',
                      fc='white', ec='none', alpha=0.85))

    # 是否路径发生变化（右上角小标注）
    offline_sig = route_signature(offline_vp)
    online_sig  = route_signature(online_vp)
    if offline_sig != online_sig:
        ax.text(0.03, 0.97, 'route changed',
                transform=ax.transAxes,
                ha='left', va='top', fontsize=6.5,
                color=SCHEME_COLOR[scheme_label],
                style='italic',
                bbox=dict(boxstyle='round,pad=0.15',
                          fc='white', ec='none', alpha=0.80))


# ────────────────────────────────────────────────────────────
# 图例 handles（所有图共用）
# ────────────────────────────────────────────────────────────
LEGEND_HANDLES = [
    Line2D([0],[0], color='#C8C8C8', lw=1.2,
           label='Offline plan'),
    Line2D([0],[0], color='#D62728', lw=1.6,
           label='E: online route'),
    Line2D([0],[0], color='#E8800A', lw=1.6,
           label='D: online route'),
    Line2D([0],[0], color='#2CA02C', lw=1.6,
           label='C: online route'),
    Line2D([0],[0], color='#888888', lw=1.0,
           linestyle='--', label='Drone sortie'),
    Line2D([0],[0], marker='s', color='#111111',
           markersize=7, lw=0, label='Depot'),
    Line2D([0],[0], marker='o', color='#333333',
           markersize=5, lw=0, label='Customer (home)'),
    Line2D([0],[0], marker='x', color='#D62728',
           markersize=5, markeredgewidth=1.3,
           lw=0, label='Not home'),
]


# ────────────────────────────────────────────────────────────
# 主循环：为每个 Small 算例生成一张图
# ────────────────────────────────────────────────────────────
all_dirs = sorted([
    d for d in os.listdir(EXP3_BASE)
    if d.startswith('small_')
    and os.path.isdir(os.path.join(EXP3_BASE, d))
])

success_count = 0
skip_count    = 0

for inst_name in all_dirs:

    # 读取三个方案的 JSON
    data = {}
    skip = False
    for scheme, fname in SCHEME_FILES.items():
        fpath = os.path.join(EXP3_BASE, inst_name, fname)
        if not os.path.exists(fpath):
            print(f'[SKIP] {inst_name}: 缺少 {fname}')
            skip = True
            break
        with open(fpath, encoding='utf-8') as f:
            data[scheme] = json.load(f)
    if skip:
        skip_count += 1
        continue

    # 解析元数据
    parts = inst_name.split('_')
    nc    = int(parts[1].replace('c', ''))
    nv    = int(parts[2].replace('v', ''))
    seed  = int(parts[3].replace('seed', ''))

    failed = {
        s: set(data[s]['online_execution']['failed_customers'])
        for s in ['E', 'D', 'C']
    }

    # ── 创建图形（1行×3列 + 右侧图例列）──
    FIG_W        = 11.5
    LEGEND_W     = 2.2
    PATH_W_FRAC  = 1.0 - LEGEND_W / FIG_W
    legend_ratio = LEGEND_W / (FIG_W * PATH_W_FRAC / 3)

    fig = plt.figure(figsize=(FIG_W, 4.0))
    gs  = gridspec.GridSpec(
        1, 4, figure=fig,
        width_ratios=[1, 1, 1, legend_ratio],
        wspace=0.28,
    )
    fig.subplots_adjust(
        left=0.06, right=PATH_W_FRAC - 0.01,
        top=0.83, bottom=0.15,
    )
    axes = [fig.add_subplot(gs[0, col]) for col in range(3)]

    # ── 图标题 ──
    fe_n = len(failed['E'])
    fd_n = len(failed['D'])
    fc_n = len(failed['C'])

    # 判断路径变化情况
    def diff_flag(scheme):
        off_sig = route_signature(
            data[scheme]['offline_plan']['vehicle_pairs'])
        on_vp   = (data[scheme].get('online_route', {})
                   .get('vehicle_pairs', []))
        on_sig  = route_signature(on_vp)
        return '✔' if off_sig != on_sig else '–'
        # ✔ = 路径变化；– = 路径未变

    fig.suptitle(
        f'{inst_name}   ({nc}c-{nv}v)    '
        f'failed — E: {fe_n}  D: {fd_n}  C: {fc_n}    '
        f'route changed — E: {diff_flag("E")}  '
        f'D: {diff_flag("D")}  C: {diff_flag("C")}',
        fontsize=9, color='#333333', y=0.97,
        fontfamily='Times New Roman',
    )

    # ── 列标题 + 绘图 ──
    for col, scheme in enumerate(['E', 'D', 'C']):
        title, subtitle, color = SCHEME_TITLE[scheme]
        axes[col].set_title(
            f'{title}\n({subtitle})',
            fontsize=9, pad=5, color=color,
            fontfamily='Times New Roman', linespacing=1.4)

        draw_route(axes[col], data[scheme], scheme,
                   failed_ids=failed[scheme],
                   is_leftmost=(col == 0))

    # ── 图例 ──
    legend_ax = fig.add_subplot(gs[0, 3])
    legend_ax.set_visible(False)

    all_pos   = [axes[c].get_position() for c in range(3)]
    ax_mid    = (max(p.y1 for p in all_pos)
                 + min(p.y0 for p in all_pos)) / 2.0
    right_x   = axes[2].get_position().x1

    fig.legend(
        handles=LEGEND_HANDLES,
        loc='center left',
        bbox_to_anchor=(right_x + 0.015, ax_mid),
        bbox_transform=fig.transFigure,
        frameon=True, framealpha=0.95,
        edgecolor='#cccccc',
        fontsize=8,
        handlelength=1.8, handleheight=0.85,
        borderpad=0.7, labelspacing=0.55,
        title='Legend', title_fontsize=8.5,
    )

    # ── 保存到算例自身目录 ──
    out_pdf = os.path.join(EXP3_BASE, inst_name,
                           'route_comparison.pdf')
    out_png = os.path.join(EXP3_BASE, inst_name,
                           'route_comparison.png')
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)

    success_count += 1
    print(f'[Done] {inst_name} -> route_comparison.pdf/png')

print(f'\n========================================')
print(f'[Batch Done] 共生成 {success_count} 张图，'
      f'跳过 {skip_count} 个算例（缺少数据文件）')
print(f'每张图保存在各算例目录下：'
      f'route_comparison.pdf / route_comparison.png')
print(f'========================================')
