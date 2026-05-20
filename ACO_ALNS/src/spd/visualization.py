from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Patch


def _configure_chinese_font() -> str | None:
    """配置 matplotlib 使用中文字体，按优先级尝试多种方案。"""

    def _refresh_font_manager_cache() -> None:
        try:
            fm._load_fontmanager(try_read_cache=False)
        except (AttributeError, TypeError):
            pass

    chinese_font_names = [
        "SimHei",
        "Microsoft YaHei",
        "SimSun",
        "STHeiti",
        "STSong",
        "PingFang SC",
        "WenQuanYi Micro Hei",
        "Noto Sans CJK SC",
        "AR PL UMing CN",
    ]

    _refresh_font_manager_cache()
    available_fonts = {font.name for font in fm.fontManager.ttflist}

    for font_name in chinese_font_names:
        if font_name in available_fonts:
            matplotlib.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            print(f"[visualization] 使用中文字体: {font_name}")
            _refresh_font_manager_cache()
            return font_name

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    font_dirs = [
        os.path.join(project_root, "fonts"),
        os.path.join(project_root, "resources", "fonts"),
        os.path.join(project_root, "assets", "fonts"),
        os.path.join(project_root, "src", "fonts"),
        os.path.join(project_root, "src", "resources", "fonts"),
        os.path.join(project_root, "src", "assets", "fonts"),
    ]

    for font_dir in font_dirs:
        if not os.path.isdir(font_dir):
            continue
        for font_file in os.listdir(font_dir):
            if not font_file.lower().endswith((".ttf", ".otf", ".ttc")):
                continue
            font_path = os.path.join(font_dir, font_file)
            try:
                fm.fontManager.addfont(font_path)
                _refresh_font_manager_cache()
                fm.fontManager.addfont(font_path)
                prop = fm.FontProperties(fname=font_path)
                font_name = prop.get_name()
                matplotlib.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
                matplotlib.rcParams["axes.unicode_minus"] = False
                print(f"[visualization] 使用项目字体: {font_name} ({font_path})")
                _refresh_font_manager_cache()
                return font_name
            except (OSError, RuntimeError, ValueError):
                continue

    print("[visualization] 警告：未找到中文字体，中文标签可能显示为方块。")
    print("[visualization] 建议：安装 SimHei 或 Microsoft YaHei 字体，或将 .ttf 文件放入项目 fonts/ 目录。")
    matplotlib.rcParams["axes.unicode_minus"] = False
    _refresh_font_manager_cache()
    return None


_CHINESE_FONT = _configure_chinese_font()


def _warn(message: str) -> None:
    print(f"[visualization] warning: {message}")


def _finalize_figure(fig: plt.Figure, save_path: str | Path | None) -> None:
    if save_path is None:
        plt.show()
    else:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)

def _get_instance_coords(instance: Any) -> dict[int, tuple[float, float]]:
    coords: dict[int, tuple[float, float]] = {0: (0.0, 0.0)}
    if instance is None or not hasattr(instance, "customers"):
        return coords
    for customer_id, customer in instance.customers.items():
        coords[int(customer_id)] = (float(customer.x), float(customer.y))
    return coords


def _normalize_groups(clustering_result: Any) -> tuple[list[list[int]], set[int], set[int]]:
    if clustering_result is None:
        return [], set(), set()

    if isinstance(clustering_result, dict):
        groups_raw = clustering_result.get("groups", [])
        truck = set(int(v) for v in clustering_result.get("truck_customers", []))
        drone = set(int(v) for v in clustering_result.get("drone_customers", []))
    else:
        groups_raw = clustering_result
        truck = set()
        drone = set()

    groups: list[list[int]] = []
    for group in groups_raw:
        groups.append([int(v) for v in group])
    return groups, truck, drone


def _unpack_solution_payload(payload: Any) -> tuple[Any, float | None, Any, Any, dict[int, bool] | None]:
    if isinstance(payload, dict) and "solution" in payload:
        solution = payload.get("solution")
        objective = payload.get("objective")
        instance = payload.get("instance")
        params = payload.get("params")
        home_status = payload.get("home_status")
        return solution, objective, instance, params, home_status
    return payload, None, None, None, None


def _deduplicate_legend_entries(handles: list[Any], labels: list[str]) -> tuple[list[Any], list[str]]:
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        if label and label not in unique:
            unique[label] = handle
    return list(unique.values()), list(unique.keys())


def _plot_solution_on_ax(
    ax: plt.Axes,
    instance: Any,
    solution: Any,
    title: str,
    objective_value: float | None = None,
    failed_customers: set[int] | list[int] | None = None,
    show_legend: bool = True,
) -> None:
    if solution is None or instance is None:
        _warn("solution/instance missing for route plot")
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("X (km)", fontsize=12)
        ax.set_ylabel("Y (km)", fontsize=12)
        return

    coords = _get_instance_coords(instance)
    tab10 = plt.cm.get_cmap("tab10", max(1, len(solution.vehicle_pairs)))

    depot_x, depot_y = coords.get(getattr(instance, "depot_id", 0), (0.0, 0.0))
    ax.scatter([depot_x], [depot_y], marker="*", s=200, c="red", label="Depot")

    for pair_idx, pair in enumerate(solution.vehicle_pairs):
        color = tab10(pair_idx)
        truck_label = f"Truck {pair.pair_id}"

        route = list(pair.truck_route)
        if len(route) >= 2:
            for idx in range(len(route) - 1):
                i = route[idx]
                j = route[idx + 1]
                xi, yi = coords.get(i, (0.0, 0.0))
                xj, yj = coords.get(j, (0.0, 0.0))
                ax.plot([xi, xj], [yi, yj], linestyle="-", linewidth=2, color=color, label=truck_label if idx == 0 else None)
                arrow = FancyArrowPatch((xi, yi), (xj, yj), arrowstyle="->", mutation_scale=8, color=color, linewidth=1, alpha=0.7)
                ax.add_patch(arrow)

        sortie_line_styles = ["--", ":", "-."]
        n_sorties = len(pair.sorties)
        for sortie_idx, sortie in enumerate(pair.sorties):
            nodes = [sortie.launch_node] + list(sortie.customers) + [sortie.recovery_node]
            xs = [coords.get(node, (0.0, 0.0))[0] for node in nodes]
            ys = [coords.get(node, (0.0, 0.0))[1] for node in nodes]
            ax.plot(
                xs,
                ys,
                linestyle=sortie_line_styles[sortie_idx % len(sortie_line_styles)],
                linewidth=1.8,
                color=color,
                alpha=0.6,
                label="Drone sortie" if pair_idx == 0 and sortie_idx == 0 else None,
            )
            jitter = (sortie_idx - n_sorties / 2.0) * 0.08
            lx, ly = coords.get(sortie.launch_node, (0.0, 0.0))
            rx, ry = coords.get(sortie.recovery_node, (0.0, 0.0))
            launch_x = lx + jitter
            launch_y = ly + jitter
            recovery_x = rx + jitter
            recovery_y = ry + jitter
            ax.scatter([launch_x], [launch_y], marker="^", s=100, color=color, alpha=0.9)
            ax.scatter([recovery_x], [recovery_y], marker="v", s=100, color=color, alpha=0.9)
            ax.annotate(
                f"S{sortie_idx}↑",
                (launch_x, launch_y),
                textcoords="offset points",
                xytext=(3, 3),
                fontsize=7,
                color=color,
            )
            ax.annotate(
                f"S{sortie_idx}↓",
                (recovery_x, recovery_y),
                textcoords="offset points",
                xytext=(3, -9),
                fontsize=7,
                color=color,
            )

    for customer_id, customer in instance.customers.items():
        x, y = customer.x, customer.y
        ax.scatter([x], [y], marker="o", s=30, c="black", alpha=0.85)
        ax.annotate(str(customer_id), (x, y), textcoords="offset points", xytext=(3, 3), fontsize=8)

    if failed_customers:
        failed_set = {int(customer_id) for customer_id in failed_customers}
        first_failed = True
        for customer_id in sorted(failed_set):
            customer = instance.customers.get(customer_id)
            if customer is None:
                continue
            ax.scatter(
                [customer.x],
                [customer.y],
                marker="x",
                s=110,
                c="red",
                linewidths=2,
                zorder=7,
                label="Failed customer" if first_failed else None,
            )
            first_failed = False

    final_title = title if objective_value is None else f"{title} | Objective={objective_value:.3f}"
    ax.set_title(final_title, fontsize=14)
    ax.set_xlabel("X (km)", fontsize=12)
    ax.set_ylabel("Y (km)", fontsize=12)
    if show_legend:
        handles, labels = _deduplicate_legend_entries(*ax.get_legend_handles_labels())
        if handles:
            ax.legend(handles, labels, fontsize=10)


def plot_clustering(instance: Any, clustering_result: Any, save_path: str | Path | None = None, initial_solution: Any = None) -> None:
    if instance is None or clustering_result is None:
        _warn("empty clustering input")
        return

    groups, truck_customers, drone_customers = _normalize_groups(clustering_result)
    if not truck_customers and not drone_customers and initial_solution is not None:
        inferred_solution, _, _, _, _ = _unpack_solution_payload(initial_solution)
        if inferred_solution is None:
            inferred_solution = initial_solution
        if hasattr(inferred_solution, "vehicle_pairs"):
            inferred_drone: set[int] = set()
            for pair in inferred_solution.vehicle_pairs:
                for sortie in getattr(pair, "sorties", []):
                    for customer_id in getattr(sortie, "customers", []):
                        inferred_drone.add(int(customer_id))
            drone_customers = inferred_drone
            depot_id = int(getattr(instance, "depot_id", 0))
            truck_customers = set(int(customer_id) for customer_id in instance.customers.keys()) - drone_customers - {depot_id}

    if not groups:
        _warn("clustering_result has no groups")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = plt.cm.get_cmap("tab10", max(1, len(groups)))

    coords = _get_instance_coords(instance)
    depot_x, depot_y = coords.get(getattr(instance, "depot_id", 0), (0.0, 0.0))
    ax.scatter([depot_x], [depot_y], marker="*", s=200, c="red", label="Depot")

    for idx, group in enumerate(groups):
        color = cmap(idx)
        gx = []
        gy = []
        for customer_id in group:
            customer = instance.customers.get(customer_id)
            if customer is None:
                continue
            gx.append(customer.x)
            gy.append(customer.y)
            marker = "o"
            if customer_id in drone_customers:
                marker = "^"
            ax.scatter([customer.x], [customer.y], marker=marker, s=60, color=color)
            ax.annotate(str(customer_id), (customer.x, customer.y), textcoords="offset points", xytext=(3, 3), fontsize=8)
        if gx and gy:
            ax.plot([], [], linestyle="", marker="o", color=color, label=f"Group {idx + 1}")

    if truck_customers:
        ax.plot([], [], linestyle="", marker="o", color="black", label="Truck-served")
    if drone_customers:
        ax.plot([], [], linestyle="", marker="^", color="black", label="Drone-served")

    ax.set_title("Customer Clustering Result", fontsize=14)
    ax.set_xlabel("X (km)", fontsize=12)
    ax.set_ylabel("Y (km)", fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique = {}
        for h, l in zip(handles, labels):
            if l not in unique:
                unique[l] = h
        ax.legend(unique.values(), unique.keys(), fontsize=10)

    _finalize_figure(fig, save_path)


def plot_initial_solution(instance: Any, initial_solution: Any, save_path: str | Path | None = None) -> None:
    solution, objective, _, _, _ = _unpack_solution_payload(initial_solution)
    if instance is None or solution is None:
        _warn("empty initial_solution input")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    _plot_solution_on_ax(ax, instance, solution, "Initial Solution (E-TPRC)", objective)
    _finalize_figure(fig, save_path)


def plot_optimized_solution(instance: Any, optimized_solution: Any, save_path: str | Path | None = None) -> None:
    solution, objective, _, _, _ = _unpack_solution_payload(optimized_solution)
    if instance is None or solution is None:
        _warn("empty optimized_solution input")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    _plot_solution_on_ax(ax, instance, solution, "Optimized Solution (ACO-ALNS)", objective)
    _finalize_figure(fig, save_path)



def _plot_online_comparison_legacy(
    instance: Any,
    planned_solution: Any,
    executed_solution: Any,
    failed_customers: set[int] | list[int] | None = None,
    save_path: str | Path | None = None,
) -> None:
    planned_sol, planned_obj, _, _, _ = _unpack_solution_payload(planned_solution)
    executed_sol, executed_obj, _, _, _ = _unpack_solution_payload(executed_solution)
    if instance is None or planned_sol is None or executed_sol is None:
        _warn("empty solution input for online comparison")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    _plot_solution_on_ax(ax1, instance, planned_sol, "Planned Solution (Offline)", planned_obj)
    _plot_solution_on_ax(
        ax2,
        instance,
        executed_sol,
        "Executed Solution (Online)",
        executed_obj,
        failed_customers=failed_customers,
    )

    failed_count = len(set(int(customer_id) for customer_id in failed_customers)) if failed_customers else 0
    fig.suptitle(f"Offline vs Online Execution | Failed: {failed_count} customers", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    _finalize_figure(fig, save_path)


def _load_cost_breakdown_from_save_path(save_path: str | Path | None) -> dict[str, Any] | None:
    if save_path is None:
        return None
    try:
        breakdown_path = Path(save_path).parent / "cost_breakdown.json"
        if not breakdown_path.exists():
            return None
        data = json.loads(breakdown_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_breakdown_components(breakdown: dict[str, Any]) -> dict[str, list[float]] | None:
    initial = breakdown.get("initial_solution")
    optimized = breakdown.get("optimized_solution")
    online = breakdown.get("online_trial_0")
    if not isinstance(initial, dict) or not isinstance(optimized, dict) or not isinstance(online, dict):
        return None

    init_oper = _to_float(
        initial.get("z_operational"),
        _to_float(initial.get("z_fixed")) + _to_float(initial.get("z_truck")) + _to_float(initial.get("z_drone")),
    )
    init_pen = _to_float(initial.get("z_fail_expected")) + _to_float(initial.get("z_tw_penalty")) + _to_float(initial.get("z_unserved_penalty"))
    init_total = _to_float(initial.get("z_total"), init_oper + init_pen)

    opt_oper = _to_float(
        optimized.get("z_operational"),
        _to_float(optimized.get("z_fixed")) + _to_float(optimized.get("z_truck")) + _to_float(optimized.get("z_drone")),
    )
    opt_pen = _to_float(optimized.get("z_fail_expected")) + _to_float(optimized.get("z_tw_penalty")) + _to_float(optimized.get("z_unserved_penalty"))
    opt_total = _to_float(optimized.get("z_total"), opt_oper + opt_pen)

    online_oper = _to_float(
        online.get("operational"),
        _to_float(online.get("fixed")) + _to_float(online.get("truck")) + _to_float(online.get("drone")),
    )
    online_pen = _to_float(online.get("fail_penalty"), _to_float(online.get("actual_total")) - online_oper)
    online_pen = max(0.0, online_pen)
    online_total = _to_float(online.get("actual_total"), online_oper + online_pen)

    return {
        "operational": [init_oper, opt_oper, online_oper],
        "penalty": [init_pen, opt_pen, online_pen],
        "total": [init_total, opt_total, online_total],
    }


def plot_cost_breakdown_table(cost_breakdown: Any, save_path: str | Path | None = None) -> None:
    """绘制确定性离线阶段的两列成本分解表（初始解 vs 优化解）。"""
    if not isinstance(cost_breakdown, dict):
        _warn("cost_breakdown payload invalid for table plot")
        return

    initial = cost_breakdown.get("initial_solution")
    optimized = cost_breakdown.get("optimized_solution")
    if not isinstance(initial, dict) or not isinstance(optimized, dict):
        _warn("cost_breakdown missing initial_solution or optimized_solution")
        return

    # 两阶段都使用与目标函数一致的六项口径，便于消融实验横向对比。
    rows = [
        ("z_fixed", "固定成本 z_fixed"),
        ("z_truck", "卡车行驶成本 z_truck"),
        ("z_drone", "无人机能耗成本 z_drone"),
        ("z_fail_expected", "期望失败防御项 z_fail_expected"),
        ("z_tw_penalty", "时间窗惩罚 z_tw_penalty"),
        ("z_unserved_penalty", "未服务惩罚 z_unserved_penalty"),
    ]
    table_values: list[list[str]] = []
    for key, _label in rows:
        init_value = _to_float(initial.get(key), 0.0)
        opt_value = _to_float(optimized.get(key), 0.0)
        delta_value = init_value - opt_value
        table_values.append([f"{init_value:.2f}", f"{opt_value:.2f}", f"{delta_value:.2f}"])

    init_total = _to_float(initial.get("z_total"), 0.0)
    opt_total = _to_float(optimized.get("z_total"), 0.0)
    table_values.append([f"{init_total:.2f}", f"{opt_total:.2f}", f"{(init_total - opt_total):.2f}"])
    row_labels = [label for _key, label in rows] + ["总目标 z_total"]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axis("off")
    ax.set_title("离线阶段成本分解表（初始解 vs 优化解）", fontsize=14, pad=12)
    table = ax.table(
        cellText=table_values,
        rowLabels=row_labels,
        colLabels=["初始解", "优化解", "改进量(初始-优化)"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.45)

    # 总目标行加粗，突出优化前后总体差异。
    total_row_idx = len(row_labels)
    for col_idx in range(3):
        table[(total_row_idx, col_idx)].set_text_props(weight="bold")
    table[(total_row_idx, -1)].set_text_props(weight="bold")

    note = "注: 该表仅展示离线两阶段，不包含在线执行与 Monte Carlo 统计。"
    fig.text(0.5, 0.04, note, ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=[0.02, 0.08, 0.98, 0.95])
    _finalize_figure(fig, save_path)


def _cost_breakdown_legend_items() -> list[Patch]:
    return [
        Patch(facecolor="#2b6cb0", edgecolor="black", label="运营成本（固定+卡车行驶+无人机能耗）"),
        Patch(facecolor="#90cdf4", edgecolor="black", hatch="///", label="离线惩罚项（期望失败+时间窗+未服务）"),
        Patch(facecolor="#fc8181", edgecolor="black", hatch="///", label="在线失败惩罚"),
    ]


def _draw_cost_breakdown_on_ax(
    ax: plt.Axes,
    breakdown: dict[str, Any],
    bar_width: float = 0.25,
    draw_legend: bool = True,
    bar_positions: list[float] | np.ndarray | None = None,
) -> bool:
    series = _extract_breakdown_components(breakdown)
    if series is None:
        return False

    operational_vals = series["operational"]
    penalty_vals = series["penalty"]
    total_vals = series["total"]
    if bar_positions is None:
        x = np.arange(3, dtype=float)
    else:
        x = np.asarray(bar_positions, dtype=float).reshape(-1)
        if x.size != 3:
            return False

    op_colors = ["#2b6cb0", "#2f855a", "#dd6b20"]
    pen_colors = ["#90cdf4", "#9ae6b4", "#fc8181"]

    for idx in range(3):
        ax.bar(x[idx], operational_vals[idx], width=bar_width, color=op_colors[idx], edgecolor="black", linewidth=0.8)
        ax.bar(
            x[idx],
            penalty_vals[idx],
            width=bar_width,
            bottom=operational_vals[idx],
            color=pen_colors[idx],
            edgecolor="black",
            linewidth=0.8,
            hatch="///",
        )

    max_total = max(total_vals) if total_vals else 1.0
    seg_threshold = max_total * 0.08
    for idx in range(3):
        if operational_vals[idx] >= seg_threshold:
            ax.text(
                x[idx],
                operational_vals[idx] * 0.5,
                f"{operational_vals[idx]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )
        if penalty_vals[idx] >= seg_threshold:
            ax.text(
                x[idx],
                operational_vals[idx] + penalty_vals[idx] * 0.5,
                f"{penalty_vals[idx]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )
        ax.text(
            x[idx],
            total_vals[idx] + max_total * 0.03,
            f"{total_vals[idx]:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["初始解", "离线优化解", "在线执行"])
    ax.set_ylabel("Cost ($)")
    ax.set_title("成本分解对比", fontsize=12, x=3, y=5)
    ax.set_ylim(0.0, max_total * 1.24 if max_total > 0.0 else 1.0)
    if bar_positions is None:
        ax.margins(x=0.08)
    else:
        pad = bar_width * 0.85
        ax.set_xlim(float(np.min(x) - pad), float(np.max(x) + pad))

    if draw_legend:
        ax.legend(handles=_cost_breakdown_legend_items(), fontsize=8, loc="upper right")
    return True


def _plot_online_comparison_stacked(
    breakdown: dict[str, Any],
    save_path: str | Path | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    if not _draw_cost_breakdown_on_ax(ax, breakdown, bar_width=0.5):
        raise ValueError("invalid cost_breakdown payload")
    ax.set_title("在线成本分项对比（同口径展示）", fontsize=14)

    note = (
        "注: 离线目标包含期望失败防御项(Z_fail_expected)，在线成本包含实际失败惩罚(fail_penalty)，"
        "二者口径不同。底部实色为运营成本（固定+行驶+能耗），顶部斜线为惩罚/防御项。"
    )
    fig.text(0.5, 0.02, note, ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    _finalize_figure(fig, save_path)


def plot_online_comparison(
    instance: Any,
    planned_solution: Any,
    executed_solution: Any,
    failed_customers: set[int] | list[int] | None = None,
    save_path: str | Path | None = None,
    cost_breakdown: dict[str, Any] | None = None,
) -> None:
    breakdown = cost_breakdown if isinstance(cost_breakdown, dict) else _load_cost_breakdown_from_save_path(save_path)
    if breakdown is not None:
        try:
            _plot_online_comparison_stacked(breakdown, save_path=save_path)
            return
        except (TypeError, ValueError, KeyError):
            _warn("cost_breakdown missing/invalid, fallback to legacy online comparison")

    _plot_online_comparison_legacy(
        instance=instance,
        planned_solution=planned_solution,
        executed_solution=executed_solution,
        failed_customers=failed_customers,
        save_path=save_path,
    )
def plot_solution_comparison(
    instance: Any,
    initial_solution: Any,
    optimized_solution: Any,
    save_path: str | Path | None = None,
    online_solution: Any = None,
    failed_customers: set[int] | list[int] | None = None,
    cost_breakdown: dict[str, Any] | None = None,
    case_name: str = "",
) -> None:
    init_solution, init_obj, _, _, _ = _unpack_solution_payload(initial_solution)
    opt_solution, opt_obj, _, _, _ = _unpack_solution_payload(optimized_solution)
    if instance is None or init_solution is None or opt_solution is None:
        _warn("empty solution input for comparison")
        return

    if online_solution is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        _plot_solution_on_ax(ax1, instance, init_solution, "Initial Solution", init_obj)
        _plot_solution_on_ax(ax2, instance, opt_solution, "Optimized Solution", opt_obj)

        if init_obj is not None and opt_obj is not None and abs(init_obj) > 1e-9:
            improve = (init_obj - opt_obj) / abs(init_obj) * 100.0
            suptitle = f"Initial vs Optimized Solution | Improvement: {improve:.1f}%"
        else:
            suptitle = "Initial vs Optimized Solution | Improvement: N/A"
        fig.suptitle(suptitle, fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        _finalize_figure(fig, save_path)
        return

    online_sol, online_obj, _, _, _ = _unpack_solution_payload(online_solution)
    if online_sol is None:
        _warn("empty online solution for combined comparison")
        return

    fig = plt.figure(figsize=(18, 10))
    gs_main = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.2], hspace=0.35)
    gs_top = gs_main[0].subgridspec(1, 3, wspace=0.25)
    ax_initial = fig.add_subplot(gs_top[0, 0])
    ax_offline = fig.add_subplot(gs_top[0, 1])
    ax_online = fig.add_subplot(gs_top[0, 2])

    gs_bottom = gs_main[1].subgridspec(1, 4, wspace=0.30)
    ax_cost_initial = fig.add_subplot(gs_bottom[0, 0])
    ax_cost_offline = fig.add_subplot(gs_bottom[0, 1], sharey=ax_cost_initial)
    ax_cost_online = fig.add_subplot(gs_bottom[0, 2], sharey=ax_cost_initial)
    ax_cost_legend = fig.add_subplot(gs_bottom[0, 3])
    ax_cost_legend.axis("off")
    cost_axes = [ax_cost_initial, ax_cost_offline, ax_cost_online]
    cost_stage_labels = ["初始解", "离线优化解", "在线执行"]

    _plot_solution_on_ax(ax_initial, instance, init_solution, "初始解 (E-TPRC)", init_obj, show_legend=False)
    _plot_solution_on_ax(ax_offline, instance, opt_solution, "离线优化解 (ACO-ALNS)", opt_obj, show_legend=False)

    failed_set = set(int(cid) for cid in failed_customers) if failed_customers else set()
    online_title = f"在线执行解 (Trial 0, Failed: {len(failed_set)} customers)"
    _plot_solution_on_ax(
        ax_online,
        instance,
        online_sol,
        online_title,
        online_obj,
        failed_customers=failed_set if failed_set else None,
        show_legend=False,
    )
    route_handles, route_labels = _deduplicate_legend_entries(*ax_online.get_legend_handles_labels())
    if route_handles:
        fig.legend(
            route_handles,
            route_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.945),
            ncol=max(1, len(route_labels)),
            fontsize=8,
            frameon=True,
            fancybox=True,
        )

    breakdown = cost_breakdown if isinstance(cost_breakdown, dict) else _load_cost_breakdown_from_save_path(save_path)
    series = _extract_breakdown_components(breakdown) if isinstance(breakdown, dict) else None
    if series is not None:
        operational_vals = [float(v) for v in series["operational"]]
        penalty_vals = [float(v) for v in series["penalty"]]
        total_vals = [float(v) for v in series["total"]]
        cost_title = "成本分解对比"
        cost_note = (
            "注: 离线目标包含期望失败防御项(Z_fail_expected)，在线成本包含实际失败惩罚(fail_penalty)，"
            "二者口径不同。底部实色为运营成本（固定+行驶+能耗），顶部斜线为惩罚/防御项。"
        )
        legend_items = _cost_breakdown_legend_items()
    else:
        total_vals = [
            _to_float(init_obj, np.nan),
            _to_float(opt_obj, np.nan),
            _to_float(online_obj, np.nan),
        ]
        operational_vals = [float(v) if np.isfinite(v) else 0.0 for v in total_vals]
        penalty_vals = [0.0, 0.0, 0.0]
        cost_title = "成本总值对比（缺少分项，显示总值）"
        cost_note = "注: 缺少 cost_breakdown.json，当前仅展示三阶段总成本。"
        legend_items = [
            Patch(facecolor="#2b6cb0", edgecolor="black", label="初始解总成本"),
            Patch(facecolor="#2f855a", edgecolor="black", label="离线优化总成本"),
            Patch(facecolor="#dd6b20", edgecolor="black", label="在线执行总成本"),
        ]

    finite_totals = [float(v) for v in total_vals if np.isfinite(v)]
    y_max = max(finite_totals) * 1.24 if finite_totals else 1.0
    if y_max <= 0.0:
        y_max = 1.0
    seg_threshold = y_max * 0.08

    op_colors = ["#2b6cb0", "#2f855a", "#dd6b20"]
    pen_colors = ["#90cdf4", "#9ae6b4", "#fc8181"]
    bar_width = 0.50
    for idx, ax in enumerate(cost_axes):
        op_val = max(0.0, operational_vals[idx]) if np.isfinite(operational_vals[idx]) else 0.0
        pen_val = max(0.0, penalty_vals[idx]) if np.isfinite(penalty_vals[idx]) else 0.0
        total_val = total_vals[idx]
        x_pos = 0.0

        ax.bar(x_pos, op_val, width=bar_width, color=op_colors[idx], edgecolor="black", linewidth=0.8)
        if pen_val > 0.0:
            ax.bar(
                x_pos,
                pen_val,
                width=bar_width,
                bottom=op_val,
                color=pen_colors[idx],
                edgecolor="black",
                linewidth=0.8,
                hatch="///",
            )

        if op_val >= seg_threshold:
            ax.text(x_pos, op_val * 0.5, f"{op_val:.2f}", ha="center", va="center", fontsize=8, color="white")
        if pen_val >= seg_threshold:
            ax.text(x_pos, op_val + pen_val * 0.5, f"{pen_val:.2f}", ha="center", va="center", fontsize=8)

        if np.isfinite(total_val):
            ax.text(
                x_pos,
                op_val + pen_val + y_max * 0.02,
                f"{float(total_val):.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        else:
            ax.text(x_pos, y_max * 0.02, "N/A", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_ylim(0.0, y_max)
        ax.set_xlim(-0.8, 0.8)
        ax.set_xticks([x_pos])
        ax.set_xticklabels([cost_stage_labels[idx]], fontsize=9)
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.35)
        ax.yaxis.set_visible(True)

        if idx == 0:
            ax.set_ylabel("Cost ($)", fontsize=10)
            ax.tick_params(axis="y", which="both", left=True, labelleft=True, labelsize=8)
        else:
            ax.tick_params(axis="y", which="both", left=True, labelleft=False)

    ax_cost_legend.legend(
        handles=legend_items,
        loc="center",
        fontsize=12,
        frameon=True,
        fancybox=True,
        edgecolor="lightgray",
        handlelength=1.5,
        handleheight=1.2,
    )
    fig.text(0.5, 0.13, cost_note, ha="center", va="top", fontsize=9)

    suptitle = "初始解-离线优化解-在线执行解综合对比"
    if case_name:
        suptitle = f"{case_name} | {suptitle}"
    fig.suptitle(suptitle, fontsize=16, y=0.985)
    fig.subplots_adjust(left=0.06, right=0.94, top=0.88, bottom=0.19)

    positions = [ax.get_position() for ax in [ax_cost_initial, ax_cost_offline, ax_cost_online, ax_cost_legend]]
    x_min = min(position.x0 for position in positions)
    x_max = max(position.x1 for position in positions)
    y_min = min(position.y0 for position in positions)
    y_max_box = max(position.y1 for position in positions)
    title_y = min(y_max_box + 0.016, 0.96)
    fig.text(0.5, title_y, cost_title, ha="center", va="bottom", fontsize=14, fontweight="bold")
    pad = 0.01
    rect = mpatches.Rectangle(
        (x_min - pad, y_min - pad),
        (x_max - x_min) + 2.0 * pad,
        (y_max_box - y_min) + 2.0 * pad,
        facecolor="none",
        edgecolor="black",
        linewidth=1.0,
        transform=fig.transFigure,
        clip_on=False,
    )
    fig.patches.append(rect)
    _finalize_figure(fig, save_path)


def plot_convergence_curve(iteration_history: Any, save_path: str | Path | None = None) -> None:
    if not iteration_history:
        _warn("iteration_history is empty")
        return

    rows = [row for row in iteration_history if isinstance(row, dict)]
    if not rows:
        _warn("iteration_history format invalid")
        return

    global_steps = np.arange(1, len(rows) + 1, dtype=float)
    current = np.array([float(row.get("current_cost", np.nan)) for row in rows], dtype=float)

    if not np.isfinite(current).any():
        _warn("iteration_history has no finite current_cost values")
        return

    best = np.full_like(current, np.nan, dtype=float)
    running_best = float("inf")
    for idx, value in enumerate(current):
        if np.isfinite(value):
            running_best = min(running_best, float(value))
        if np.isfinite(running_best):
            best[idx] = running_best

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(global_steps, best, color="#0b3d91", linewidth=2, label="Best", zorder=2)
    ax.scatter(
        global_steps,
        current,
        color="#ff6b6b",
        s=25,
        alpha=0.7,
        label="Current",
        zorder=3,
        edgecolors="white",
        linewidth=0.5,
    )

    # Optional ACO-iteration separators when metadata is available.
    aco_iters: list[int | None] = []
    for row in rows:
        raw = row.get("aco_iteration")
        if raw is None:
            aco_iters.append(None)
        else:
            try:
                aco_iters.append(int(raw))
            except (TypeError, ValueError):
                aco_iters.append(None)

    if any(value is not None for value in aco_iters):
        boundaries: list[tuple[float, int]] = []
        for idx in range(1, len(aco_iters)):
            prev_iter = aco_iters[idx - 1]
            curr_iter = aco_iters[idx]
            if prev_iter is None or curr_iter is None or prev_iter == curr_iter:
                continue
            boundary_x = float(idx + 0.5)
            ax.axvline(boundary_x, color="#cccccc", linestyle="--", linewidth=0.8, zorder=1)
            boundaries.append((boundary_x, int(curr_iter)))
        if boundaries:
            label_step = 1
            if len(boundaries) > 15:
                label_step = int(np.ceil(len(boundaries) / 15.0))
            for mark_idx, (boundary_x, curr_iter) in enumerate(boundaries):
                if mark_idx % label_step != 0 and mark_idx != len(boundaries) - 1:
                    continue
                ax.text(
                    boundary_x + 0.08,
                    0.985,
                    f"ACO {curr_iter}",
                    transform=ax.get_xaxis_transform(),
                    fontsize=7,
                    color="#777777",
                    va="top",
                )

    if np.isfinite(best).any():
        best_value = float(np.nanmin(best))
        first_idx = int(np.where(np.isclose(best, best_value, equal_nan=False))[0][0])
        ax.scatter([global_steps[first_idx]], [best[first_idx]], color="red", s=40, zorder=5)
        ax.text(
            0.02,
            0.95,
            f"Final Best: {best_value:.3f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    ax.set_title("Convergence Curve", fontsize=14)
    ax.set_xlabel("Global Step", fontsize=12)
    ax.set_ylabel("Objective Value", fontsize=12)
    ax.legend(fontsize=10, loc="upper right", bbox_to_anchor=(0.98, 0.88))

    _finalize_figure(fig, save_path)


def plot_timeline_gantt(solution: Any, save_path: str | Path | None = None) -> None:
    payload_solution, _, instance, params, home_status = _unpack_solution_payload(solution)
    if payload_solution is None:
        _warn("solution is empty for timeline gantt")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    state_colors = {
        "Travel": "#1f77b4",
        "Service": "#2ca02c",
        "Launch": "#ff7f0e",
        "Recovery": "#9467bd",
        "Drone Mission": "#7ec8e3",
    }

    y_labels: list[str] = []
    y_ticks: list[int] = []
    y_pos = 0
    simplified_used = False

    for pair in payload_solution.vehicle_pairs:
        truck_row = y_pos
        drone_row = y_pos + 1
        y_ticks.extend([truck_row, drone_row])
        y_labels.extend([f"Truck {pair.pair_id}", f"Drone {pair.pair_id}"])
        y_pos += 2

        route = list(pair.truck_route)

        if instance is not None and params is not None:
            try:
                hs = home_status or {cid: True for cid in instance.customers}
                core_mod = __import__("spd.core", fromlist=["compute_truck_timeline"])
                timeline = core_mod.compute_truck_timeline(instance, pair, hs, params)

                # Truck row: travel bars and on-node service/wait bars.
                for idx in range(len(route) - 1):
                    prev_node = route[idx]
                    next_node = route[idx + 1]
                    start = float(timeline.truck_departure.get(prev_node, timeline.truck_arrival.get(prev_node, 0.0)))
                    end = float(timeline.truck_arrival.get(next_node, start))
                    if end > start:
                        ax.barh(
                            truck_row,
                            end - start,
                            left=start,
                            color=state_colors["Travel"],
                            edgecolor="black",
                        )

                for node in route:
                    if node == instance.depot_id:
                        continue
                    arrive = float(timeline.truck_arrival.get(node, 0.0))
                    depart = float(timeline.truck_departure.get(node, arrive))
                    if depart > arrive:
                        ax.barh(
                            truck_row,
                            depart - arrive,
                            left=arrive,
                            color=state_colors["Service"],
                            edgecolor="black",
                        )
                        ax.text(arrive + max((depart - arrive) * 0.05, 0.1), truck_row, str(node), va="center", fontsize=8)

                # Drone row: launch marker + mission span + recovery marker for each sortie.
                event_width = 1.5
                for sortie in pair.sorties:
                    launch_time = float(
                        timeline.truck_departure.get(
                            sortie.launch_node,
                            timeline.truck_arrival.get(sortie.launch_node, 0.0),
                        )
                    )
                    recovery_time = float(timeline.drone_arrival.get(sortie.recovery_node, launch_time))

                    mission_start = min(launch_time, recovery_time)
                    mission_end = max(launch_time, recovery_time)

                    ax.barh(
                        drone_row,
                        event_width,
                        left=max(0.0, launch_time - event_width / 2.0),
                        color=state_colors["Launch"],
                        edgecolor="black",
                    )

                    if mission_end > mission_start:
                        ax.barh(
                            drone_row,
                            mission_end - mission_start,
                            left=mission_start,
                            color=state_colors["Drone Mission"],
                            edgecolor="black",
                        )
                        if sortie.customers:
                            customer_label = ",".join(str(customer_id) for customer_id in sortie.customers)
                            ax.text(
                                mission_start + max((mission_end - mission_start) * 0.05, 0.1),
                                drone_row,
                                customer_label,
                                va="center",
                                fontsize=8,
                            )

                    ax.barh(
                        drone_row,
                        event_width,
                        left=max(0.0, recovery_time - event_width / 2.0),
                        color=state_colors["Recovery"],
                        edgecolor="black",
                    )

            except Exception:
                simplified_used = True
                truck_cursor = 0.0
                for _ in range(1, len(route)):
                    ax.barh(truck_row, 8.0, left=truck_cursor, color=state_colors["Travel"], edgecolor="black")
                    truck_cursor += 8.0
                    ax.barh(truck_row, 3.0, left=truck_cursor, color=state_colors["Service"], edgecolor="black")
                    truck_cursor += 3.0

                drone_cursor = 0.0
                for sortie in pair.sorties:
                    ax.barh(drone_row, 1.2, left=drone_cursor, color=state_colors["Launch"], edgecolor="black")
                    ax.barh(drone_row, 4.0, left=drone_cursor + 1.2, color=state_colors["Drone Mission"], edgecolor="black")
                    ax.barh(drone_row, 1.2, left=drone_cursor + 5.2, color=state_colors["Recovery"], edgecolor="black")
                    if sortie.customers:
                        ax.text(drone_cursor + 1.4, drone_row, ",".join(str(c) for c in sortie.customers), va="center", fontsize=8)
                    drone_cursor += 7.0
        else:
            simplified_used = True
            truck_cursor = 0.0
            for _ in range(1, len(route)):
                ax.barh(truck_row, 8.0, left=truck_cursor, color=state_colors["Travel"], edgecolor="black")
                truck_cursor += 8.0
                ax.barh(truck_row, 3.0, left=truck_cursor, color=state_colors["Service"], edgecolor="black")
                truck_cursor += 3.0

            drone_cursor = 0.0
            for sortie in pair.sorties:
                ax.barh(drone_row, 1.2, left=drone_cursor, color=state_colors["Launch"], edgecolor="black")
                ax.barh(drone_row, 4.0, left=drone_cursor + 1.2, color=state_colors["Drone Mission"], edgecolor="black")
                ax.barh(drone_row, 1.2, left=drone_cursor + 5.2, color=state_colors["Recovery"], edgecolor="black")
                if sortie.customers:
                    ax.text(drone_cursor + 1.4, drone_row, ",".join(str(c) for c in sortie.customers), va="center", fontsize=8)
                drone_cursor += 7.0

    if simplified_used:
        ax.text(
            0.01,
            0.99,
            "[simplified view - timeline unavailable]",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_title("Vehicle Timeline (Gantt Chart)", fontsize=14)
    ax.set_xlabel("Time (min)", fontsize=12)
    ax.set_ylabel("Vehicle", fontsize=12)
    for name, color in state_colors.items():
        ax.barh(-10, 0, color=color, label=name)
    ax.legend(fontsize=10)

    _finalize_figure(fig, save_path)


def plot_energy_consumption(solution: Any, save_path: str | Path | None = None) -> None:
    payload_solution, _, instance, params, home_status = _unpack_solution_payload(solution)
    if payload_solution is None:
        _warn("solution is empty for energy plot")
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = plt.cm.get_cmap("tab10", 10)

    sortie_count = 0
    battery_capacity = float(getattr(getattr(params, "energy", None), "drone_battery_capacity", 100.0)) if params is not None else 100.0

    for pair in payload_solution.vehicle_pairs:
        for sortie in pair.sorties:
            phase_labels = ["launch"] + [f"customer_{cid}" for cid in sortie.customers] + ["recovery"]
            n = len(phase_labels)
            if n <= 1:
                continue

            energy_used = None
            if instance is not None and params is not None:
                try:
                    hs = home_status or {cid: True for cid in instance.customers}
                    timeline = __import__("spd.core", fromlist=["compute_truck_timeline"]).compute_truck_timeline(instance, pair, hs, params)
                    energy_used = __import__("spd.core", fromlist=["compute_sortie_energy"]).compute_sortie_energy(
                        instance,
                        sortie,
                        timeline,
                        hs,
                        params,
                    )
                except Exception:
                    energy_used = None

            if energy_used is None:
                energy_used = 5.0 * max(1, len(sortie.customers))

            drops = np.linspace(0.0, float(energy_used), n)
            levels = battery_capacity - drops
            levels = np.maximum(levels, 0.0)
            x = np.arange(n)
            color = cmap(sortie_count % 10)
            label = f"Pair {pair.pair_id} sortie {sortie_count + 1} ({','.join(str(c) for c in sortie.customers)})"
            ax.plot(x, levels, marker="o", linewidth=2, color=color, label=label)
            sortie_count += 1

    if sortie_count == 0:
        _warn("no sorties to plot energy consumption")
        plt.close(fig)
        return

    ax.axhline(0.0, color="red", linestyle="--", linewidth=1.5, label="Lower bound")
    ax.set_title("Drone Energy Consumption per Sortie", fontsize=14)
    ax.set_xlabel("Flight Phase", fontsize=12)
    ax.set_ylabel("Battery Level (Wh)", fontsize=12)
    ax.legend(fontsize=10)

    _finalize_figure(fig, save_path)


def plot_operator_statistics(operator_history: Any, save_path: str | Path | None = None) -> None:
    if not operator_history:
        _warn("operator_history is empty")
        return

    rows = [row for row in operator_history if isinstance(row, dict)]
    if not rows:
        _warn("operator_history format invalid")
        return

    op_counter: Counter[str] = Counter()
    weight_series: defaultdict[str, list[tuple[int, float]]] = defaultdict(list)

    for idx, row in enumerate(rows, start=1):
        op_name = str(row.get("operator_name", ""))
        if op_name:
            parts = [p for p in op_name.split("+") if p]
            for part in parts:
                if part == "initial_solution":
                    continue
                op_counter[part] += 1

        scores = row.get("operator_scores", {})
        if not isinstance(scores, dict):
            continue

        destroy_scores: dict[str, float] = {}
        repair_scores: dict[str, float] = {}

        for op, value in scores.items():
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue

            op_key = str(op)
            op_lower = op_key.lower()
            if op_key == "initial_solution":
                continue
            if op_key.startswith("destroy::") or "destroy" in op_lower:
                destroy_scores[op_key] = numeric_value
            elif op_key.startswith("repair::") or "repair" in op_lower:
                repair_scores[op_key] = numeric_value

        destroy_total = float(sum(destroy_scores.values()))
        if destroy_total > 0.0:
            for op_key, numeric_value in destroy_scores.items():
                weight_series[op_key].append((idx, numeric_value / destroy_total))

        repair_total = float(sum(repair_scores.values()))
        if repair_total > 0.0:
            for op_key, numeric_value in repair_scores.items():
                weight_series[op_key].append((idx, numeric_value / repair_total))

    if not op_counter and not weight_series:
        _warn("operator_history has no usable data")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    names = list(op_counter.keys())
    counts = [op_counter[name] for name in names]
    colors = ["#d62728" if "d" in name.lower() or "destroy" in name.lower() else "#1f77b4" for name in names]
    ax1.bar(names, counts, color=colors)
    ax1.set_title("ALNS Operator Statistics", fontsize=14)
    ax1.set_xlabel("Operator", fontsize=12)
    ax1.set_ylabel("Selected Count", fontsize=12)
    ax1.tick_params(axis="x", rotation=45)

    def _moving_average(values: list[float], window_size: int) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if window_size <= 1 or len(arr) <= window_size:
            return arr
        kernel = np.ones(window_size, dtype=float) / float(window_size)
        return np.convolve(arr, kernel, mode="valid")

    series_items = [(name, series) for name, series in weight_series.items() if name != "initial_solution"]
    cmap = plt.cm.get_cmap("tab20", max(1, len(series_items)))
    for color_idx, (name, series) in enumerate(series_items):
        if not series:
            continue
        xs = [item[0] for item in series]
        ys = [float(item[1]) for item in series]
        color = cmap(color_idx)

        # Background raw trajectory.
        ax2.plot(xs, ys, color=color, alpha=0.15, linewidth=0.5)

        # Foreground smoothed trend.
        window = max(1, len(ys) // 20)
        smoothed = _moving_average(ys, window)
        if window > 1 and len(ys) > window:
            x_smooth = xs[window - 1:]
        else:
            x_smooth = xs
        ax2.plot(x_smooth, smoothed, color=color, alpha=0.9, linewidth=1.5, label=name)

    ax2.set_xlabel("Iteration", fontsize=12)
    ax2.set_ylabel("Normalized Weight", fontsize=12)
    ax2.set_ylim(0.0, 1.0)
    if series_items:
        legend_cols = min(4, max(1, len(series_items)))
        ax2.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=legend_cols,
            fontsize=7,
            frameon=True,
            fancybox=True,
        )

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    _finalize_figure(fig, save_path)


def plot_monte_carlo_distribution(mc_results: Any, save_path: str | Path | None = None) -> None:
    deterministic_value = None

    if mc_results is None:
        _warn("mc_results is empty")
        return

    if isinstance(mc_results, dict):
        samples = mc_results.get("samples") or mc_results.get("mc_costs") or []
        deterministic_value = mc_results.get("deterministic")
    elif hasattr(mc_results, "samples"):
        samples = list(getattr(mc_results, "samples", []))
        deterministic_value = getattr(mc_results, "deterministic", None)
    else:
        samples = list(mc_results)

    if not samples:
        _warn("mc_results has no samples")
        return

    values = np.array([float(v) for v in samples], dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values))
    ci_low = float(np.percentile(values, 2.5))
    ci_high = float(np.percentile(values, 97.5))

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.hist(values, bins=30, color="#9ecae1", edgecolor="black")
    ax.axvline(mean, color="red", linestyle="--", linewidth=2, label="Mean")
    if deterministic_value is not None:
        ax.axvline(float(deterministic_value), color="green", linestyle="--", linewidth=2, label="Deterministic")

    stats_text = f"Mean: {mean:.2f}\nStd: {std:.2f}\n95% CI: [{ci_low:.2f}, {ci_high:.2f}]"
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_title(f"Monte Carlo Simulation Results (N={len(values)})", fontsize=14)
    ax.set_xlabel("Objective Value", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.legend(fontsize=10)

    _finalize_figure(fig, save_path)





def plot_replanning_snapshots(
    instance: Any,
    snapshots: list[dict],
    save_path: str | Path | None = None,
    filename_prefix: str = "replanning_visualization",
    max_subplots_per_page: int = 6,
) -> list[str]:
    """
    可视化在线重规划过程中每次服务失败后的解路径变化。

    参数:
        instance: 算例实例，包含客户坐标等信息
        snapshots: 快照列表，每个元素是 dict，包含:
            - 'label': str, 子图标题
            - 'solution': 解对象（vehicle_pairs 等）
            - 'failed_customer': int | None, 本次失败的客户ID
            - 'objective': float | None, 目标函数值（可选）
        save_path: 保存目录路径（注意：这里是目录，不是文件路径）
        filename_prefix: 文件名前缀
        max_subplots_per_page: 每页最大子图数，默认6

    返回:
        生成的图片文件路径列表
    """
    import math

    if not snapshots:
        _warn("no replanning snapshots to plot")
        return []

    if max_subplots_per_page <= 0:
        raise ValueError("max_subplots_per_page must be positive")

    total = len(snapshots)
    num_pages = math.ceil(total / max_subplots_per_page)
    output_files: list[str] = []
    save_dir = Path(save_path) if save_path is not None else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    for page in range(num_pages):
        start = page * max_subplots_per_page
        end = min(start + max_subplots_per_page, total)
        page_snapshots = snapshots[start:end]
        count = len(page_snapshots)

        if count <= 3:
            nrows, ncols = 1, count
        elif count == 4:
            nrows, ncols = 2, 2
        else:
            nrows, ncols = 2, 3

        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
        axes_list = np.atleast_1d(axes).ravel().tolist()

        for idx, snapshot in enumerate(page_snapshots):
            failed_set = None
            failed_customer = snapshot.get("failed_customer")
            if failed_customer is not None:
                failed_set = {int(failed_customer)}
            _plot_solution_on_ax(
                axes_list[idx],
                instance,
                snapshot.get("solution"),
                str(snapshot.get("label", f"重规划快照 {start + idx}")),
                snapshot.get("objective"),
                failed_customers=failed_set,
            )

        for idx in range(count, len(axes_list)):
            axes_list[idx].set_visible(False)

        if num_pages == 1:
            fig.suptitle("在线重规划路径变化", fontsize=16, fontweight="bold")
        else:
            fig.suptitle(f"在线重规划路径变化（第 {page + 1}/{num_pages} 页）", fontsize=16, fontweight="bold")
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])

        if save_dir is None:
            _finalize_figure(fig, None)
            continue

        if num_pages == 1:
            filepath = save_dir / f"{filename_prefix}.png"
        else:
            filepath = save_dir / f"{filename_prefix}_page{page + 1}.png"

        output_files.append(str(filepath))
        _finalize_figure(fig, filepath)

    return output_files


