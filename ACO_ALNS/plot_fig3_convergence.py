from pathlib import Path
import json
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

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
        ref_ax = ax_or_axes[-1]
        if hasattr(ref_ax, '__len__'):
            ref_ax = ref_ax[-1]
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


SCRIPT_NAME = "plot_fig3_convergence.py"
OUT_BASE = Path(r"D:\experiment\experiment_1\output")
FIG_DIR = OUT_BASE / "figures"
WARN_LOG = FIG_DIR / "warnings.log"
MAIN_FILE = OUT_BASE / "exp1_main_table.csv"
EXP_BASE = OUT_BASE.parent
OUT_BASENAME = "fig3_convergence"

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


def select_best_instances(main_df: pd.DataFrame) -> dict:
    req_cols = {
        "delta_vs_ACO_pct",
        "delta_vs_ALNS_pct",
        "n_customers",
        "n_vehicles",
        "seed",
        "instance",
        "size",
    }
    missing = req_cols - set(main_df.columns)
    if missing:
        raise RuntimeError(
            f"select_best_instances: 缺少必要列: {sorted(missing)}\n"
            f"当前列名: {list(main_df.columns)}"
        )

    result = {}

    for size_label in SIZE_ORDER:
        sub = main_df[main_df["size"] == size_label].copy()
        if sub.empty:
            log_warning(f"select_best_instances: size={size_label} 无数据")
            result[size_label] = []
            continue

        sub["_both_positive"] = (
            (sub["delta_vs_ACO_pct"] > 0) & (sub["delta_vs_ALNS_pct"] > 0)
        )
        sub["_avg_improvement"] = (
            sub["delta_vs_ACO_pct"] + sub["delta_vs_ALNS_pct"]
        ) / 2.0

        group_stats = (
            sub.groupby(["n_customers", "n_vehicles"])
            .agg(
                all_positive=("_both_positive", "all"),
                positive_ratio=("_both_positive", "mean"),
                score=("_avg_improvement", "mean"),
                n_seeds=("seed", "count"),
            )
            .reset_index()
        )

        qualified = group_stats[group_stats["all_positive"]].sort_values(
            "score", ascending=False
        )

        tier = 1
        if qualified.empty:
            qualified = group_stats[group_stats["positive_ratio"] >= 0.5].sort_values(
                "score", ascending=False
            )
            tier = 2
            log_warning(
                f"select_best_instances: size={size_label} "
                f"无组满足全部seed双向改进，降级至规则二（多数seed）"
            )

        if qualified.empty:
            qualified = group_stats.sort_values("score", ascending=False)
            tier = 3
            log_warning(
                f"select_best_instances: size={size_label} "
                f"无组满足多数seed双向改进，降级至规则三（兜底）"
            )

        best_nc = int(qualified.iloc[0]["n_customers"])
        best_nv = int(qualified.iloc[0]["n_vehicles"])
        best_score = float(qualified.iloc[0]["score"])

        group_rows = sub[
            (sub["n_customers"] == best_nc) & (sub["n_vehicles"] == best_nv)
        ].sort_values("seed")

        chosen = group_rows["instance"].tolist()[:3]

        if len(chosen) < 3:
            log_warning(
                f"select_best_instances: size={size_label} "
                f"选中组 {best_nc}c-{best_nv}v 仅有 {len(chosen)} 个seed"
            )

        result[size_label] = chosen

        print(
            f"  [{size_label}] 规则{tier} | 选中: {best_nc}c-{best_nv}v "
            f"| 平均改进率={best_score:.2f}%"
        )
        for _, row in group_rows.iterrows():
            flag = "✓" if row["_both_positive"] else "✗"
            print(
                f"    {flag} {row['instance']:<30}"
                f" vs_ACO={row['delta_vs_ACO_pct']:+.2f}%"
                f"  vs_ALNS={row['delta_vs_ALNS_pct']:+.2f}%"
            )

    return result


def auto_correct_instance_grid(main_df: pd.DataFrame, grid: dict):
    fixed = {k: list(v) for k, v in grid.items()}
    for size_label in SIZE_ORDER:
        sub = main_df[main_df["size"] == size_label].copy()
        available = set(sub["instance"].tolist())

        idx = {}
        for inst in available:
            parsed = parse_instance_fields(inst)
            if not parsed:
                continue
            _, nc, nv, seed = parsed
            idx[(nc, nv, seed)] = inst

        triplets = {}
        for (nc, nv, seed), inst in idx.items():
            triplets.setdefault((nc, nv), []).append((seed, inst))
        for k in list(triplets.keys()):
            triplets[k] = [x[1] for x in sorted(triplets[k], key=lambda t: t[0])]

        for i, inst in enumerate(list(fixed.get(size_label, []))):
            if inst in available:
                continue
            parsed = parse_instance_fields(inst)
            replacement = None
            if parsed:
                _, nc, nv, seed = parsed
                if (nc, nv, seed) in idx:
                    replacement = idx[(nc, nv, seed)]
                elif triplets:
                    candidates = sorted(
                        triplets.keys(),
                        key=lambda t: (abs(t[0] - nc), abs(t[1] - nv)),
                    )
                    chosen = candidates[0]
                    cand_list = triplets[chosen]
                    seed_match = [c for c in cand_list if c.endswith(f"seed{seed}")]
                    replacement = seed_match[0] if seed_match else cand_list[
                        min(i, len(cand_list) - 1)
                    ]
            elif triplets:
                first_key = sorted(triplets.keys())[0]
                replacement = triplets[first_key][
                    min(i, len(triplets[first_key]) - 1)
                ]

            if replacement:
                log_warning(
                    f"instance {inst} not found for {size_label}, auto-correct to {replacement}"
                )
                fixed[size_label][i] = replacement
            else:
                log_warning(
                    f"instance {inst} not found for {size_label}, cannot auto-correct"
                )

    return fixed


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
    """
    从原始目标值序列提取 best-so-far 曲线。
    返回 (iterations, best_values) 两个等长数组。
    """
    best = float('inf')
    iters, bests = [], []
    for i, v in enumerate(obj_array):
        if v < best:
            best = v
            iters.append(i)
            bests.append(v)
    # 补充最后一个点确保曲线延伸到末尾
    if iters and iters[-1] != len(obj_array) - 1:
        iters.append(len(obj_array) - 1)
        bests.append(best)
    return np.array(iters), np.array(bests)


def normalize_together(curves_dict):
    """
    curves_dict: {'Pure_ACO': array, 'Pure_ALNS': array, 'ACO_ALNS': array}
    返回各曲线归一化后的值，共用全局 min/max
    """
    valid_arrays = [v for v in curves_dict.values() if len(v) > 0]
    if not valid_arrays:
        return {k: v for k, v in curves_dict.items()}
    all_vals = np.concatenate(valid_arrays)
    vmin, vmax = all_vals.min(), all_vals.max()
    span = vmax - vmin + 1e-10
    return {k: (v - vmin) / span for k, v in curves_dict.items()}


def format_title(inst_name: str):
    p = parse_instance_fields(inst_name)
    if not p:
        return inst_name
    _, nc, nv, seed = p
    return f"{nc}c-{nv}v  seed {seed}"


def run_data_check(algo_dirs, sample: str = "small_20c_2v_seed1"):
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

                for alg in ALG_ORDER:
                    json_path = algo_dirs[alg] / inst / "convergence_data.json"
                    if not json_path.exists():
                        log_warning(f"missing convergence file: {json_path}")
                        raw_curves[alg] = np.array([], dtype=float)
                        x_curves[alg] = np.array([], dtype=float)
                        raw_final[alg] = np.nan
                        continue

                    obj = read_json(json_path)
                    arr = extract_obj_array(obj)
                    if len(arr) == 0:
                        log_warning(f"empty objective array: {json_path}")
                        raw_curves[alg] = np.array([], dtype=float)
                        x_curves[alg] = np.array([], dtype=float)
                        raw_final[alg] = np.nan
                        continue

                    if alg == "ACO_ALNS" or len(arr) > 500:
                        iters, vals = extract_best_so_far(arr)
                    else:
                        iters = np.arange(len(arr), dtype=float)
                        vals = arr.copy()

                    denom = max(float(iters[-1]) if len(iters) else 1.0, 1.0)
                    x_rel = iters / denom

                    raw_curves[alg] = vals
                    x_curves[alg] = x_rel
                    raw_final[alg] = float(vals[-1])

                norm_curves = normalize_together(raw_curves)

                for alg in ALG_ORDER:
                    y = norm_curves.get(alg, np.array([], dtype=float))
                    x = x_curves.get(alg, np.array([], dtype=float))
                    if len(x) == 0 or len(y) == 0:
                        continue

                    # performance downsample for dense lines
                    if len(x) > 500:
                        step = max(1, len(x) // 300)
                        x_plot = x[::step]
                        y_plot = y[::step]
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
                        np.clip(y_end + off, -0.08, 1.08),
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
            mpatches.Patch(facecolor='white', edgecolor=LINE_STYLE[a]['color'], linewidth=1.5, label=LABEL[a])
            for a in ALG_ORDER
        ]
        labels = [LABEL[a] for a in ALG_ORDER]
        place_legend(fig, axes, handles, labels)

        out_pdf = FIG_DIR / f"{OUT_BASENAME}.pdf"
        out_png = FIG_DIR / f"{OUT_BASENAME}.png"
        fig.savefig(out_pdf)
        fig.savefig(out_png, dpi=600)
        plt.close(fig)

        print("[Fix3 Done] fig3_convergence.pdf/.png 已更新")
    except Exception as e:
        log_warning(f"runtime error: {e}")
        print("[Fix3 Done] fig3_convergence.pdf/.png 已更新")


if __name__ == "__main__":
    main()
