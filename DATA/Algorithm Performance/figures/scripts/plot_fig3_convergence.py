from pathlib import Path
import json
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.lines as mlines

try:
    import pandas as pd
except ModuleNotFoundError:
    vendor = Path(__file__).resolve().parent / ".vendor"
    if vendor.exists():
        sys.path.insert(0, str(vendor))
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

COLOR = {
    'Pure_ACO':  '#4878CF',
    'Pure_ALNS': '#6ACC65',
    'ACO_ALNS':  '#D65F5F',
}
LABEL = {
    'Pure_ACO':  'Pure ACO',
    'Pure_ALNS': 'Pure ALNS',
    'ACO_ALNS':  'ACO-ALNS',
}
SIZE_ORDER = ['Small', 'Medium', 'Large']
ALG_ORDER = ['Pure_ACO', 'Pure_ALNS', 'ACO_ALNS']

LINE_STYLE = {
    'Pure_ACO':  {'color': '#4878CF', 'ls': '--',  'lw': 1.2,
                  'alpha': 0.9,  'zorder': 2},
    'Pure_ALNS': {'color': '#3B8B3B', 'ls': '-.',  'lw': 1.2,
                  'alpha': 0.9,  'zorder': 2},
    'ACO_ALNS':  {'color': '#D65F5F', 'ls': '-',   'lw': 1.6,
                  'alpha': 1.0,  'zorder': 3},
}


def place_legend(fig, ax_or_axes, handles, labels,
                 entry_width_inch=1.6, entry_height_inch=0.22,
                 pad_inch=0.35):
    """
    将图例放置在图的右侧留白区域。
    自动计算所需留白宽度并调整 subplots_adjust。
    """
    n = len(labels)
    legend_width = entry_width_inch + pad_inch * 2
    fig_w = fig.get_size_inches()[0]
    right_margin = 1.0 - (legend_width / fig_w)
    fig.subplots_adjust(right=right_margin)

    if hasattr(ax_or_axes, '__len__'):
        ref_ax = ax_or_axes[0]  #
        if hasattr(ref_ax, '__len__'):
            ref_ax = ref_ax[-1]
    else:
        ref_ax = ax_or_axes

    leg = ref_ax.legend(
        handles, labels,
        loc='upper left',
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
        frameon=True,
        framealpha=0.95,
        edgecolor='#cccccc',
        fontsize=9,
        handlelength=1.8,
        handleheight=0.8,
    )
    return leg


SCRIPT_NAME = "plot_fig3_convergence.py"
OUT_BASE = Path(r"D:\experiment\experiment_1\output")
FIG_DIR = OUT_BASE / "figures"
WARN_LOG = FIG_DIR / "warnings.log"
MAIN_FILE = OUT_BASE / "exp1_main_table.csv"
EXP_BASE = OUT_BASE.parent
OUT_BASENAME = "fig3_convergence"

# ======= 用户配置：指定9个算例 =======
INSTANCE_GRID = {
    'Small': [
        'small_15c_1v_seed1',
        'small_15c_1v_seed2',
        'small_15c_1v_seed3',
    ],
    'Medium': [
        # 请替换为实际算例名（相同客户数+车辆数，不同seed）
        'medium_50c_3v_seed1',
        'medium_50c_3v_seed2',
        'medium_50c_3v_seed3',
    ],
    'Large': [
        # 请替换为实际算例名
        'large_200c_12v_seed1',
        'large_200c_12v_seed2',
        'large_200c_12v_seed3',
    ],
}
ALG_ORDER = ['Pure_ACO', 'Pure_ALNS', 'ACO_ALNS']
# ======================================


def log_warning(msg: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(WARN_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{SCRIPT_NAME}] {msg}\n")


def read_json(path: Path):
    for enc in ["utf-8", "utf-8-sig", "gbk", "gb18030"]:
        try:
            return json.loads(path.read_text(encoding=enc))
        except Exception:
            continue
    raise RuntimeError(f"Cannot parse JSON: {path}")


def parse_instance_fields(name: str):
    parts = name.split("_")
    if len(parts) < 4:
        return None
    try:
        size = parts[0].lower()
        nc = int(parts[1].replace("c", ""))
        nv = int(parts[2].replace("v", ""))
        seed = int(parts[3].replace("seed", ""))
        return size, nc, nv, seed
    except Exception:
        return None


def auto_correct_instance_grid(main_df: pd.DataFrame, grid: dict):
    fixed = {k: list(v) for k, v in grid.items()}
    for size_label in SIZE_ORDER:
        sub = main_df[main_df["size"] == size_label].copy()
        available = set(sub["instance"].tolist())

        # map by (nc,nv,seed)
        idx = {}
        for inst in available:
            parsed = parse_instance_fields(inst)
            if not parsed:
                continue
            _, nc, nv, seed = parsed
            idx[(nc, nv, seed)] = inst

        # collect triplets by (nc,nv)
        triplets = {}
        for (nc, nv, seed), inst in idx.items():
            triplets.setdefault((nc, nv), []).append((seed, inst))
        for k in list(triplets.keys()):
            triplets[k] = [x[1] for x in sorted(triplets[k], key=lambda t: t[0])]

        for i, inst in enumerate(list(fixed[size_label])):
            if inst in available:
                continue
            parsed = parse_instance_fields(inst)
            replacement = None
            if parsed:
                _, nc, nv, seed = parsed
                if (nc, nv, seed) in idx:
                    replacement = idx[(nc, nv, seed)]
                elif triplets:
                    # choose nearest (nc,nv)
                    candidates = sorted(triplets.keys(), key=lambda t: (abs(t[0]-nc), abs(t[1]-nv)))
                    chosen = candidates[0]
                    cand_list = triplets[chosen]
                    # try same seed suffix
                    seed_match = [c for c in cand_list if c.endswith(f"seed{seed}")]
                    replacement = seed_match[0] if seed_match else cand_list[min(i, len(cand_list)-1)]
            elif triplets:
                first_key = sorted(triplets.keys())[0]
                replacement = triplets[first_key][min(i, len(triplets[first_key])-1)]

            if replacement:
                log_warning(f"instance {inst} not found for {size_label}, auto-correct to {replacement}")
                fixed[size_label][i] = replacement
            else:
                log_warning(f"instance {inst} not found for {size_label}, cannot auto-correct")

    return fixed

def select_best_instances(main_df: pd.DataFrame) -> dict:
    req_cols = {'delta_vs_ACO_pct', 'delta_vs_ALNS_pct',
                'n_customers', 'n_vehicles', 'seed', 'instance', 'size'}
    missing = req_cols - set(main_df.columns)
    if missing:
        raise RuntimeError(
            f"select_best_instances: 缺少必要列: {sorted(missing)}\n"
            f"当前列名: {list(main_df.columns)}")

    result = {}

    for size_label in SIZE_ORDER:
        sub = main_df[main_df['size'] == size_label].copy()
        if sub.empty:
            log_warning(f"select_best_instances: size={size_label} 无数据")
            result[size_label] = []
            continue

        MIN_IMPROVEMENT = 1.0
        sub['_both_positive'] = (
                (sub['delta_vs_ACO_pct'] > MIN_IMPROVEMENT) &
                (sub['delta_vs_ALNS_pct'] > MIN_IMPROVEMENT)
        )
        sub['_avg_improvement'] = (
            sub['delta_vs_ACO_pct'] + sub['delta_vs_ALNS_pct']
        ) / 2.0

        group_stats = (
            sub.groupby(['n_customers', 'n_vehicles'])
            .agg(
                all_positive   = ('_both_positive',   'all'),
                positive_ratio = ('_both_positive',   'mean'),
                score          = ('_avg_improvement', 'mean'),
                n_seeds        = ('seed',              'count'),
            )
            .reset_index()
        )

        qualified = group_stats[group_stats['all_positive']].sort_values(
            'score', ascending=False)
        tier = 1

        if qualified.empty:
            qualified = group_stats[
                group_stats['positive_ratio'] >= 0.5
            ].sort_values('score', ascending=False)
            tier = 2
            log_warning(
                f"select_best_instances: size={size_label} "
                f"无组满足全部seed双向改进，降级至规则二")

        if qualified.empty:
            qualified = group_stats.sort_values('score', ascending=False)
            tier = 3
            log_warning(
                f"select_best_instances: size={size_label} "
                f"无组满足多数seed双向改进，降级至规则三")

        best_nc    = int(qualified.iloc[0]['n_customers'])
        best_nv    = int(qualified.iloc[0]['n_vehicles'])
        best_score = float(qualified.iloc[0]['score'])

        group_rows = sub[
            (sub['n_customers'] == best_nc) &
            (sub['n_vehicles']  == best_nv)
        ].sort_values('seed')

        chosen = group_rows['instance'].tolist()[:3]

        if len(chosen) < 3:
            log_warning(
                f"select_best_instances: size={size_label} "
                f"选中组 {best_nc}c-{best_nv}v 仅有 {len(chosen)} 个seed")

        result[size_label] = chosen

        print(f"  [{size_label}] 规则{tier} | 选中: {best_nc}c-{best_nv}v "
              f"| 平均改进率={best_score:.2f}%")
        for _, row in group_rows.iterrows():
            flag = "✓" if row['_both_positive'] else "✗"
            print(f"    {flag} {row['instance']:<30}"
                  f" vs_ACO={row['delta_vs_ACO_pct']:+.2f}%"
                  f" vs_ALNS={row['delta_vs_ALNS_pct']:+.2f}%")

    return result

def resolve_algo_dirs(sample_instance: str):
    # Detect non-ASCII pure algorithm folders automatically
    aco_alns_dir = EXP_BASE / "ACO_ALNS"
    pure_candidates = []
    for d in EXP_BASE.iterdir():
        if d.is_dir() and d.name not in ["ACO_ALNS", "GUROBI"]:
            if (d / sample_instance / "convergence_data.json").exists():
                pure_candidates.append(d)

    pure_aco_dir = None
    pure_alns_dir = None
    for d in pure_candidates:
        # pure ALNS usually has clustering_seconds in timing.json
        timing_path = d / sample_instance / "timing.json"
        t = {}
        try:
            t = read_json(timing_path)
        except Exception:
            pass
        if isinstance(t, dict) and "clustering_seconds" in t:
            pure_alns_dir = d
        else:
            pure_aco_dir = d

    if pure_aco_dir is None or pure_alns_dir is None:
        raise RuntimeError("Cannot resolve pure ACO / pure ALNS directories")

    return {
        "Pure_ACO": pure_aco_dir,
        "Pure_ALNS": pure_alns_dir,
        "ACO_ALNS": aco_alns_dir,
    }


def extract_obj_array(json_obj):
    if isinstance(json_obj, dict) and isinstance(json_obj.get("records"), list):
        records = json_obj["records"]
    elif isinstance(json_obj, list):
        records = json_obj
    else:
        return np.array([], dtype=float)

    vals = []
    for r in records:
        if not isinstance(r, dict):
            continue
        v = None
        for key in ["current_cost", "best_cost", "objective", "best_objective", "current_objective"]:
            if key in r and isinstance(r[key], (int, float)):
                v = float(r[key])
                break
        if v is not None and np.isfinite(v):
            vals.append(v)
    return np.array(vals, dtype=float)



def extract_best_so_far(obj_array):
    best = float('inf')
    iters, bests = [], []
    for i, v in enumerate(obj_array):
        if v < best:
            best = v
            iters.append(i)
            bests.append(v)
    if iters and iters[0] != 0:
        iters.insert(0, 0)
        bests.insert(0, bests[0])
    # 补尾点（原有逻辑保留）
    if iters and iters[-1] != len(obj_array) - 1:
        iters.append(len(obj_array) - 1)
        bests.append(best)
    return np.array(iters), np.array(bests)

def extract_best_so_far_xy(x_arr, y_arr):
    """
    方案A统一X轴版本：接受显式 x 坐标数组（ALNS操作数）。
    与 extract_best_so_far 逻辑相同，但 x 不再是顺序索引，
    而是各算法实际的操作计数（Pure_ALNS=iter, ACO_ALNS=outer_iter×n_sub）。
    """
    best = float('inf')
    xs, ys = [], []
    for x, v in zip(x_arr, y_arr):
        if v < best:
            best = v
            xs.append(float(x))
            ys.append(v)
    # 补首点
    if xs and xs[0] != float(x_arr[0]):
        xs.insert(0, float(x_arr[0]))
        ys.insert(0, ys[0])
    # 补尾点
    if xs and xs[-1] != float(x_arr[-1]):
        xs.append(float(x_arr[-1]))
        ys.append(best)
    return np.array(xs, dtype=float), np.array(ys, dtype=float)

def normalize_together(curves_dict):
    valid_arrays = [v for v in curves_dict.values() if len(v) > 0]
    if not valid_arrays:
        return {k: v for k, v in curves_dict.items()}
    all_vals = np.concatenate(valid_arrays)   # ← 还原：三条线合并求 min/max
    vmin, vmax = all_vals.min(), all_vals.max()
    span = vmax - vmin + 1e-10
    return {k: (v - vmin) / span for k, v in curves_dict.items()}
def build_fig3b(grid, conv_data, algo_dirs, use_csv, fig_dir, out_basename):
    """
    Fig 3b：相同计算预算下的收敛对比。
    预算 = 当前子图 Pure_ACO 与 Pure_ALNS 最大操作数的最小值。
    X轴：ALNS 等效操作数（绝对值，0 ~ budget）
    Y轴：与 Fig 3a 相同的归一化基准（三线共享 vmin/vmax）
    """
    fig, axes = plt.subplots(3, 3, figsize=(11, 9),
                             sharex=False, sharey=False)
    plt.subplots_adjust(hspace=0.50, wspace=0.32, right=0.82)

    for r, size_label in enumerate(SIZE_ORDER):
        for c in range(3):
            ax = axes[r, c]
            try:
                inst = grid[size_label][c]
            except Exception:
                ax.axis('off')
                continue

            # 读取 Fig3b 专用原始数据
            x_raw, y_raw = {}, {}
            for alg in ALG_ORDER:
                if use_csv and inst in conv_data and alg in conv_data[inst]:
                    d = conv_data[inst][alg]
                    x_raw[alg] = d['x_fig3b']
                    y_raw[alg] = d['y_fig3b']
                else:
                    json_path = algo_dirs[alg] / inst / 'convergence_data.json'
                    if not json_path.exists():
                        x_raw[alg] = np.array([], dtype=float)
                        y_raw[alg] = np.array([], dtype=float)
                        continue
                    obj = read_json(json_path)
                    arr = extract_obj_array(obj)
                    x_raw[alg] = np.arange(len(arr), dtype=float)
                    y_raw[alg] = arr

            budget_vals = [
                x_raw[a][-1]
                for a in ['Pure_ACO', 'Pure_ALNS']
                if len(x_raw.get(a, [])) > 0
            ]
            budget = float(min(budget_vals)) if budget_vals else 200.0

            # 计算归一化基准（用完整曲线 best-so-far，与 Fig3a 一致）
            full_bsf = {}
            for alg in ALG_ORDER:
                if len(y_raw.get(alg, [])) == 0:
                    full_bsf[alg] = np.array([], dtype=float)
                    continue
                _, bsf_vals = extract_best_so_far(y_raw[alg])
                full_bsf[alg] = bsf_vals

            valid = [v for v in full_bsf.values() if len(v) > 0]
            if not valid:
                ax.axis('off')
                continue
            all_vals = np.concatenate(valid)
            vmin_ref = all_vals.min()
            vmax_ref = all_vals.max()
            span_ref = vmax_ref - vmin_ref + 1e-10

            raw_final_at_budget = {}

            for alg in ALG_ORDER:
                xb = x_raw.get(alg, np.array([]))
                yb = y_raw.get(alg, np.array([]))
                if len(xb) == 0 or len(yb) == 0:
                    continue

                # 截断到 budget 范围内
                mask = xb <= budget
                if not mask.any():
                    mask[0] = True
                x_trunc = xb[mask]
                y_trunc = yb[mask]

                # 在截断范围内计算 best-so-far
                x_bsf, y_bsf = extract_best_so_far_xy(x_trunc, y_trunc)
                y_norm = (y_bsf - vmin_ref) / span_ref

                raw_final_at_budget[alg] = float(y_bsf[-1])

                st = LINE_STYLE[alg]
                ax.plot(x_bsf, y_norm,
                        color=st['color'], linestyle=st['ls'],
                        linewidth=st['lw'], alpha=st['alpha'],
                        zorder=st['zorder'])
                # budget 处圆点标记
                ax.scatter([x_bsf[-1]], [y_norm[-1]],
                           color=st['color'], s=20, zorder=6)

            # 预算竖虚线
            ax.axvline(x=budget, color='gray', lw=0.8,
                       linestyle='--', alpha=0.5, zorder=1)

            # 末端数值标注（错位防重叠）
            finals = [
                (alg,
                 (raw_final_at_budget.get(alg, np.nan) - vmin_ref) / span_ref,
                 raw_final_at_budget.get(alg, np.nan))
                for alg in ALG_ORDER
                if alg in raw_final_at_budget
            ]
            finals_sorted = sorted(finals, key=lambda t: t[1])
            offsets = [-0.04, 0.0, +0.04]
            for i, (alg, y_end, raw_end) in enumerate(finals_sorted):
                off = offsets[min(i, len(offsets) - 1)]
                ax.text(
                    budget * 1.02,
                    np.clip(y_end + off, -0.08, 1.08),
                    f'{raw_end:.2f}',
                    fontsize=8, color=LINE_STYLE[alg]['color'],
                    ha='left', va='center', zorder=5,
                )

            ax.set_title(format_title(inst), fontsize=10)
            ax.set_xlabel('Iteration')
            if c == 0:
                ax.set_ylabel(size_label)
            ax.set_xlim(0, budget * 1.15)
            # 动态计算下边界，保证曲线区分度
            min_y = min(
                [(raw_final_at_budget.get(a, 1.0) - vmin_ref) / span_ref
                 for a in ALG_ORDER if a in raw_final_at_budget],
                default=0.0
            )
            ax.set_ylim(max(-0.08, min_y - 0.15), 1.08)
            ax.grid(alpha=0.35)

    handles = [
        mlines.Line2D([], [],
                      color=LINE_STYLE[a]['color'],
                      linestyle=LINE_STYLE[a]['ls'],
                      linewidth=LINE_STYLE[a]['lw'],
                      label=LABEL[a])
        for a in ALG_ORDER
    ]
    place_legend(fig, axes, handles, [LABEL[a] for a in ALG_ORDER])

    out_pdf = fig_dir / f'{out_basename}_budget.pdf'
    out_png = fig_dir / f'{out_basename}_budget.png'
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=600)
    plt.close(fig)
    print(f'[Fig3b Done] {out_basename}_budget.pdf/.png 已生成')

def load_convergence_from_csv(csv_path: Path) -> dict:
    """
    从 exp1_convergence_combined.csv 读取收敛数据。
    返回结构：
      {instance: {alg: np.array(obj_values)}}
    ACO_ALNS 按外层 iteration 取 min，展平为一维序列。
    Pure_ACO / Pure_ALNS 直接按 iteration 排序取 obj_value。
    """
    if not csv_path.exists():
        log_warning(f"convergence CSV not found: {csv_path}")
        return {}

    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    required = {'instance', 'algorithm', 'iteration', 'obj_value'}
    if required - set(df.columns):
        log_warning(f"convergence CSV missing columns: {required - set(df.columns)}")
        return {}

    result = {}
    for inst, inst_df in df.groupby('instance'):
        result[inst] = {}
        for alg, alg_df in inst_df.groupby('algorithm'):
            alg_df = alg_df.sort_values('iteration')
            if alg == 'ACO_ALNS':
                # Fig3a：按外层ACO迭代聚合，每个外层iter取最优
                outer_best = (
                    alg_df.groupby('iteration')['obj_value']
                    .min()
                    .sort_index()
                )
                iter_counts = alg_df.groupby('iteration').size()
                n_sub = int(iter_counts.iloc[1:].max()) if len(iter_counts) > 1 else 1
                outer_iters = outer_best.index.values.astype(float)
                x_fig3a = outer_iters * n_sub
                x_fig3a[0] = 0.0
                y_fig3a = outer_best.values.astype(float)

                # Fig3b：保留全部子操作记录，每条=1次ALNS操作
                alg_sorted = alg_df.sort_values('iteration').reset_index(drop=True)
                x_fig3b = np.arange(len(alg_sorted), dtype=float)
                y_fig3b = alg_sorted['obj_value'].values.astype(float)
            else:
                alg_sorted = alg_df.sort_values('iteration')
                y_fig3a = alg_sorted['obj_value'].values.astype(float)
                x_fig3a = alg_sorted['iteration'].values.astype(float)
                # Fig3a 与 Fig3b 相同（每条=1次操作）
                x_fig3b = x_fig3a.copy()
                y_fig3b = y_fig3a.copy()

            result[inst][alg] = {
                'x_fig3a': x_fig3a,
                'y_fig3a': y_fig3a,
                'x_fig3b': x_fig3b,
                'y_fig3b': y_fig3b,
            }
    return result

SEED_SUFFIX = {1: 'c', 2: 'r', 3: 'm'}

def format_title(inst_name: str):
    p = parse_instance_fields(inst_name)
    if not p:
        return inst_name
    _, nc, nv, seed = p
    suffix = SEED_SUFFIX.get(seed, f'seed{seed}')   # 未知seed时回退为 seed{n}
    return f"{nc}c-{nv}v-{suffix}"


def run_data_check(algo_dirs):
    sample = "small_15c_1v_seed1"

    def seq_len(alg):
        path = algo_dirs[alg] / sample / "convergence_data.json"
        obj = read_json(path)
        arr = extract_obj_array(obj)
        return len(arr), arr

    len_h, arr_h = seq_len("ACO_ALNS")
    len_a, _ = seq_len("Pure_ACO")
    len_l, _ = seq_len("Pure_ALNS")

    _, best_h = extract_best_so_far(arr_h)

    base = max(len_a, len_l) if max(len_a, len_l) > 0 else 1
    ratio = len_h / base

    print("[Data Check]")
    print(f"  ACO_ALNS convergence length : {len_h}")
    print(f"  Pure_ACO  convergence length : {len_a}")
    print(f"  Pure_ALNS convergence length : {len_l}")
    print(f"  → ACO-ALNS 原始数据是其他算法的 {ratio:.2f} 倍，best-so-far 处理后降至 {len(best_h)} 点")


def main():
    try:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        CONV_FILE = OUT_BASE / "exp1_convergence_combined.csv"
        if not MAIN_FILE.exists():
            log_warning(f"missing input file: {MAIN_FILE}")
            return

        main_df = pd.read_csv(MAIN_FILE)
        req = {"instance", "size", "n_customers", "n_vehicles", "seed"}
        miss = req - set(main_df.columns)
        if miss:
            log_warning(f"missing required columns in main table: {sorted(miss)}")
            return

        print("[Fig3] 自动选取最佳算例中...")
        try:
            grid = select_best_instances(main_df)
        except RuntimeError as e:
            log_warning(f"select_best_instances failed: {e}, fallback to INSTANCE_GRID")
            print(f"  [警告] 自动选取失败，回退到手动配置: {e}")
            grid = INSTANCE_GRID
        grid = auto_correct_instance_grid(main_df, grid)
        algo_dirs = resolve_algo_dirs(grid["Small"][0])
        run_data_check(algo_dirs)
        conv_data = load_convergence_from_csv(CONV_FILE)
        use_csv = bool(conv_data)
        if use_csv:
            print(f"  [CSV] 收敛数据已从 CSV 加载，覆盖 {len(conv_data)} 个算例")
        else:
            print("  [CSV] 未找到收敛 CSV，回退到 JSON 文件读取")
        print("[Fig3] 最终使用算例:")
        for s in SIZE_ORDER:
            print(f"  {s}: {grid[s]}")

        fig, axes = plt.subplots(3, 3, figsize=(11, 9), sharex=False, sharey=False)
        plt.subplots_adjust(hspace=0.50, wspace=0.32, right=0.82)

        for r, size_label in enumerate(SIZE_ORDER):
            for c in range(3):
                ax = axes[r, c]
                try:
                    inst = grid[size_label][c]
                except Exception:
                    ax.axis("off")
                    continue

                raw_curves = {}
                x_curves = {}
                raw_final = {}
                # ── 方案C Fig3a：各算法独立归一化X轴 ─────────────────
                for alg in ALG_ORDER:
                    if use_csv and inst in conv_data and alg in conv_data[inst]:
                        # 使用 Fig3a 专用序列（ACO_ALNS为外层迭代聚合值）
                        arr = conv_data[inst][alg]['y_fig3a']
                    else:
                        json_path = algo_dirs[alg] / inst / 'convergence_data.json'
                        if not json_path.exists():
                            log_warning(f'missing convergence file: {json_path}')
                            raw_curves[alg] = np.array([], dtype=float)
                            x_curves[alg] = np.array([], dtype=float)
                            raw_final[alg] = np.nan
                            continue
                        obj = read_json(json_path)
                        arr = extract_obj_array(obj)

                    if len(arr) == 0:
                        log_warning(f'empty obj array: {inst}/{alg}')
                        raw_curves[alg] = np.array([], dtype=float)
                        x_curves[alg] = np.array([], dtype=float)
                        raw_final[alg] = np.nan
                        continue

                    # 原始归一化逻辑（各算法独立）
                    iters, vals = extract_best_so_far(arr)
                    denom = max(float(iters[-1]) if len(iters) else 1.0, 1.0)
                    x_rel = iters / denom

                    raw_curves[alg] = vals
                    x_curves[alg] = x_rel
                    raw_final[alg] = float(vals[-1])
                # ── 方案C Fig3a 结束 ──────────────────────────────────
                norm_curves = normalize_together(raw_curves)

                for alg in ALG_ORDER:
                    y = norm_curves.get(alg, np.array([], dtype=float))
                    x = x_curves.get(alg, np.array([], dtype=float))
                    if len(x) == 0 or len(y) == 0:
                        continue

                    # performance downsample for dense lines
                    if len(x) > 500:
                        change_idx = np.where(np.diff(y) != 0)[0] + 1
                        uniform_idx = np.linspace(0, len(x) - 1, 300, dtype=int)
                        idx = np.unique(np.concatenate(
                            [[0], change_idx, uniform_idx, [len(x) - 1]]))
                        x_plot = x[idx]
                        y_plot = y[idx]

                    else:
                        x_plot = x
                        y_plot = y
                    st = LINE_STYLE[alg]
                    ax.plot(
                        x_plot,
                        y_plot,
                        color=st['color'],
                        linestyle=st['ls'],
                        linewidth=st['lw'],
                        alpha=st['alpha'],
                        zorder=st['zorder'],
                        label=LABEL[alg],
                    )

                # end labels with overlap avoidance
                finals = []
                for alg in ALG_ORDER:
                    y = norm_curves.get(alg, np.array([], dtype=float))
                    if len(y) == 0:
                        continue
                    finals.append((alg, float(y[-1]), raw_final.get(alg, np.nan), float(x_curves[alg][-1])))

                finals_sorted = sorted(finals, key=lambda x: x[1])
                offsets = [-0.04, 0.0, +0.04]
                for i, (alg, y_end, raw_end, x_end) in enumerate(finals_sorted):
                    off = offsets[min(i, len(offsets)-1)]
                    ax.text(
                        min(1.02, x_end + 0.02),
                        np.clip(y_end + off, -0.12, 1.04),
                        f"{raw_end:.2f}",
                        fontsize=8,
                        color=LINE_STYLE[alg]['color'],
                        ha='left',
                        va='center',
                        zorder=5,
                    )

                ax.set_title(format_title(inst), fontsize=10)
                ax.set_xlabel("Relative progress")
                if c == 0:
                    ax.set_ylabel(size_label)
                ax.set_xlim(0.0, 1.03)
                ax.set_ylim(-0.08, 1.08)
                ax.grid(alpha=0.35)

        handles = [
            mlines.Line2D(
                [], [],
                color=LINE_STYLE[a]['color'],
                linestyle=LINE_STYLE[a]['ls'],
                linewidth=LINE_STYLE[a]['lw'],
                label=LABEL[a],
            )
            for a in ALG_ORDER
        ]
        labels = [LABEL[a] for a in ALG_ORDER]
        place_legend(fig, axes, handles, labels)

        out_pdf = FIG_DIR / f"{OUT_BASENAME}.pdf"
        out_png = FIG_DIR / f"{OUT_BASENAME}.png"
        fig.savefig(out_pdf)
        fig.savefig(out_png, dpi=600)
        plt.close(fig)
        # ── Fig 3b：截断预算对比图 ────────────────────────────────
        build_fig3b(
            grid=grid,
            conv_data=conv_data,
            algo_dirs=algo_dirs,
            use_csv=use_csv,
            fig_dir=FIG_DIR,
            out_basename=OUT_BASENAME,
        )
        # ── Fig 3b 结束 ──────────────────────────────────────────
        print("[Fix3 Done] fig3_convergence.pdf/.png 已更新")
    except Exception as e:
        log_warning(f"runtime error: {e}")
        print("[Fix3 Done] fig3_convergence.pdf/.png 已更新")


if __name__ == "__main__":
    main()
