from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from spd import core
from spd import main as spd_main
from spd.aco_alns import ACOALNSSolver, ALNSOptimizer
from spd.etprc import ETPRCBuilder
from spd.types import ProblemInstance, Solution, Sortie, VehiclePairSolution


@dataclass
class CheckResult:
    ok: bool
    message: str


def _fmt_bool(ok: bool) -> str:
    return "✓" if ok else "×"


def _sorted_set(values: set[int] | frozenset[int]) -> list[int]:
    return sorted(int(v) for v in values)


def _route_index(route: list[int], node: int) -> int | None:
    try:
        return route.index(node)
    except ValueError:
        return None


def _safe_call(func, *args, **kwargs):
    try:
        return True, func(*args, **kwargs), None
    except Exception as exc:  # noqa: BLE001
        return False, None, exc


def print_constraint_catalog() -> None:
    print("===== 约束清单（从 core.py 源码中提取）=====")
    print()
    print("--- 结构约束（Structural Constraints）---")
    print("S1: sortie 发射/回收点不能是 depot - launch_node/recovery_node != 0 - check_sortie_structure - launch==0 或 recovery==0 返回 False")
    print("S2: sortie 发射/回收点必须属于该 pair 的 truck_customers - check_sortie_structure - launch/recovery 不在 pair_solution.truck_customers 返回 False")
    print("S3: sortie 发射/回收点必须在 truck_route 且顺序正确 - check_sortie_structure - launch/recovery 不在 route 或 route.index(launch)>=route.index(recovery) 返回 False")
    print("S4: sortie 客户不能包含 launch/recovery，且不能与 truck_route 节点重叠 - check_sortie_structure - launch/recovery 出现在 customers 或 any(customer in route_nodes) 返回 False")
    print()
    print("--- 可行性约束（Feasibility Constraints）---")
    print("F1: sortie 电量约束 - compute_sortie_energy <= drone_battery_capacity - check_sortie_feasibility")
    print("F2: 无人机载重约束 - check_sortie_load_feasibility 为 True - check_sortie_feasibility")
    print("F3: 卡车等待约束 - max(0, drone_arrival[recovery]-truck_arrival[recovery]) <= max_truck_wait_time - check_sortie_feasibility")
    print("F4: 无人机时间窗上界约束 - drone_arrival[customer] <= customer.time_window[1] - check_sortie_feasibility")
    print("F5: 卡车回收点载重约束 - truck_load_at_node[recovery] <= truck_capacity - check_sortie_feasibility")
    print("F6: 卡车时间窗约束 - truck_arrival[node] <= customer.time_window[1] - is_truck_route_feasible")
    print("F7: 卡车载重约束 - 每个 route 节点 load_state.truck_load_at_node[node] <= truck_capacity - is_truck_route_feasible")
    print("F8: 返回 depot 时间约束 - truck_arrival[last_depot] <= depot_latest - is_truck_route_feasible")
    print("F9: sortie 客户数上限 - len(sortie.customers) <= max_customers_per_sortie - check_sortie_feasibility")
    print("F10: 全局硬约束 - 每个 pair 均通过 timeline/load、sortie 结构/可行性、truck_route 可行性 - all_hard_constraints_satisfied")
    print()
    print("--- 物理约束（Physical Constraints，可能未在代码中显式检查）---")
    print("P1: 无人机可用性 - 每个 vehicle pair 只有 1 架无人机，sortie_{i+1} 的 launch 不能早于 sortie_i 的 recovery")
    print("P2: 同节点连续回收再发射属于边界可行，若下一个 launch 的 route 索引小于上一个 recovery 索引则物理不可能")
    print()
    print("--- 完整性约束（Completeness Constraints）---")
    print("C1: 所有客户要么被卡车服务、要么被无人机服务、要么在 unserved 中，不能遗漏")
    print("C2: 同一客户不能同时被卡车和无人机服务")
    print("C3: 同一客户不能出现在多个 sortie 中")
    print("C4: 同一客户不能出现在多个 pair 中")
    print()
    print("=============================================")
    print()


def build_solutions() -> tuple[ProblemInstance, Any, dict[int, bool], Solution, Solution, list[set[int]]]:
    instance = spd_main._load_instance("examples/instance_small.json")
    params = spd_main._load_parameters("examples/config_default.json")
    home_status = {customer_id: True for customer_id in instance.customers}
    rng = random.Random(0)

    builder = ETPRCBuilder()

    # 与 main.py 带 --output-dir 时路径一致：先做一次 grouping
    groups = builder._customer_grouping(instance, params, rng)

    initial_solution = builder.build_initial_solution(instance, params, rng)

    alns_optimizer = ALNSOptimizer(spd_main._build_operator_set())
    solver = ACOALNSSolver(etprc_builder=builder, alns_optimizer=alns_optimizer)
    optimized_solution = solver.solve(instance, params, home_status, rng)

    return instance, params, home_status, initial_solution, optimized_solution, groups


def print_solution_detail(tag: str, instance: ProblemInstance, solution: Solution) -> None:
    print(f"===== {tag} SOLUTION DETAIL =====")
    print()
    print("总体信息：")
    print(f"  vehicle_pair 数量: {len(solution.vehicle_pairs)}")
    print(f"  all_customers（被服务的客户集合）: {sorted(solution.all_customers)}")
    print(f"  unserved_customers（未被服务的客户）: {sorted(solution.unserved_customers)}")
    print()

    for pair in solution.vehicle_pairs:
        print(f"[Pair {pair.pair_id}]")
        route = list(pair.truck_route)
        sortie_customers = {cid for s in pair.sorties for cid in s.customers}
        truck_direct = [node for node in route if node != instance.depot_id and node not in sortie_customers]

        print(f"  truck_route: {route}")
        print(f"  truck_route 长度: {len(route)}")
        print(f"  卡车直接服务的客户（在 truck_route 中且不在任何 sortie 中的非 depot 节点）: {sorted(truck_direct)}")
        print(f"  sortie 数量: {len(pair.sorties)}")
        print()

        for idx, sortie in enumerate(pair.sorties):
            launch_idx = _route_index(route, sortie.launch_node)
            recovery_idx = _route_index(route, sortie.recovery_node)
            print(f"  Sortie {idx}:")
            print(f"    launch_node = {sortie.launch_node} (在 truck_route 中的位置索引 = {launch_idx})")
            print(f"    customers = {list(sortie.customers)}")
            print(f"    recovery_node = {sortie.recovery_node} (在 truck_route 中的位置索引 = {recovery_idx})")
            print("    每个客户的详细信息:")
            for customer_id in sortie.customers:
                customer = instance.customers[customer_id]
                ctype = customer.customer_type.value
                print(f"      customer_id={customer_id}, type={ctype}, weight={customer.weight}")
            print()

    print("=============================================")
    print()


def check_solution_constraints(
    tag: str,
    instance: ProblemInstance,
    params: Any,
    home_status: dict[int, bool],
    solution: Solution,
) -> list[str]:
    violations: list[str] = []

    print(f"===== {tag} CONSTRAINT CHECK =====")
    print()

    for pair in solution.vehicle_pairs:
        pair_prefix = f"[{tag}][Pair {pair.pair_id}]"
        print(f"[Pair {pair.pair_id}]")
        print()

        # Precompute timeline/load with safe handling.
        timeline_ok, timeline, timeline_err = _safe_call(core.compute_truck_timeline, instance, pair, home_status, params)
        load_ok, load_state, load_err = _safe_call(core.compute_truck_load, instance, pair, home_status, params)

        print("  --- A. 整体约束检查（调用 core.py 中的函数）---")
        pair_solution = Solution(vehicle_pairs=[pair], unserved_customers=set(solution.unserved_customers))
        all_ok, all_hard, all_err = _safe_call(core.all_hard_constraints_satisfied, instance, pair_solution, home_status, params)
        all_hard_val = bool(all_hard) if all_ok else False
        print(f"  {_fmt_bool(all_hard_val)} all_hard_constraints_satisfied: {all_hard_val}")
        if not all_hard_val:
            if not all_ok:
                msg = f"{pair_prefix} A.all_hard_constraints_satisfied 调用异常: {all_err}"
                violations.append(msg)
                print(f"    × 调用异常: {all_err}")
            else:
                # Try to pinpoint sub-checks.
                if not timeline_ok:
                    msg = f"{pair_prefix} A.subcheck timeline 构建失败: {timeline_err}"
                    violations.append(msg)
                    print(f"    × 子检查失败: compute_truck_timeline 异常: {timeline_err}")
                if not load_ok:
                    msg = f"{pair_prefix} A.subcheck load 构建失败: {load_err}"
                    violations.append(msg)
                    print(f"    × 子检查失败: compute_truck_load 异常: {load_err}")
                if timeline_ok and load_ok:
                    for idx, sortie in enumerate(pair.sorties):
                        s_ok = core.check_sortie_structure(pair, sortie)
                        print(f"    {'✓' if s_ok else '×'} 子检查 sortie[{idx}] structure: {s_ok}")
                        if not s_ok:
                            violations.append(f"{pair_prefix} A.subcheck sortie[{idx}] structure=False")
                        sf_ok = core.check_sortie_feasibility(instance, pair, sortie, timeline, load_state, home_status, params)
                        print(f"    {'✓' if sf_ok else '×'} 子检查 sortie[{idx}] feasibility: {sf_ok}")
                        if not sf_ok:
                            violations.append(f"{pair_prefix} A.subcheck sortie[{idx}] feasibility=False")
                    route_ok = core.is_truck_route_feasible(instance, pair, timeline, load_state, params)
                    print(f"    {'✓' if route_ok else '×'} 子检查 route feasibility: {route_ok}")
                    if not route_ok:
                        violations.append(f"{pair_prefix} A.subcheck route_feasibility=False")

        route_feasible_val = False
        if timeline_ok and load_ok:
            rf_ok, route_feasible, rf_err = _safe_call(core.is_truck_route_feasible, instance, pair, timeline, load_state, params)
            route_feasible_val = bool(route_feasible) if rf_ok else False
            print(f"  {_fmt_bool(route_feasible_val)} is_truck_route_feasible: {route_feasible_val}")
            if not route_feasible_val:
                if not rf_ok:
                    msg = f"{pair_prefix} A.is_truck_route_feasible 调用异常: {rf_err}"
                    violations.append(msg)
                    print(f"    × 调用异常: {rf_err}")
                else:
                    print("    × 具体违反项将见 B/D/F 分项")
                    violations.append(f"{pair_prefix} A.is_truck_route_feasible=False")
        else:
            print("  × is_truck_route_feasible: False")
            print("    × 无法计算 timeline/load，跳过该函数调用")
            violations.append(f"{pair_prefix} A.is_truck_route_feasible=not_computable")
        print()

        print("  --- B. 卡车路径结构检查 ---")
        route = list(pair.truck_route)
        b1 = len(route) > 0 and route[0] == instance.depot_id
        print(f"  {_fmt_bool(b1)} B1: truck_route 以 depot 开始: {b1}（首节点={route[0] if route else None}, depot_id={instance.depot_id}）")
        if not b1:
            violations.append(f"{pair_prefix} B1 depot_start=False")

        b2 = len(route) > 0 and route[-1] == instance.depot_id
        print(f"  {_fmt_bool(b2)} B2: truck_route 以 depot 结束: {b2}（末节点={route[-1] if route else None}, depot_id={instance.depot_id}）")
        if not b2:
            violations.append(f"{pair_prefix} B2 depot_end=False")

        non_depot = [n for n in route if n != instance.depot_id]
        dup_counts = Counter(non_depot)
        dup_nodes = sorted([node for node, cnt in dup_counts.items() if cnt > 1])
        b3 = len(dup_nodes) == 0
        print(f"  {_fmt_bool(b3)} B3: truck_route 中无重复节点（depot 除外）: {b3}")
        if not b3:
            print(f"    × 重复节点: {dup_nodes}")
            violations.append(f"{pair_prefix} B3 duplicate_nodes={dup_nodes}")

        if timeline_ok and route:
            return_time = timeline.truck_arrival.get(route[-1], float('nan'))
            b4 = return_time <= params.time.depot_latest
            print(f"  {_fmt_bool(b4)} B4: 卡车返回 depot 时间 ≤ depot_latest: {b4}（返回时间={return_time}, 上限={params.time.depot_latest}）")
            if not b4:
                violations.append(f"{pair_prefix} B4 return_time={return_time} > depot_latest={params.time.depot_latest}")
        else:
            print("  × B4: 卡车返回 depot 时间 ≤ depot_latest: False（timeline 不可用）")
            violations.append(f"{pair_prefix} B4 timeline_unavailable")
        print()

        print("  --- C. 逐个 Sortie 结构检查 ---")
        for idx, sortie in enumerate(pair.sorties):
            launch_idx = _route_index(route, sortie.launch_node)
            recovery_idx = _route_index(route, sortie.recovery_node)

            c1 = launch_idx is not None
            print(f"  Sortie {idx}:")
            print(f"    {_fmt_bool(c1)} C1: launch_node 在 truck_route 中: {c1}")
            if not c1:
                violations.append(f"{pair_prefix} C1 sortie[{idx}] launch_not_in_route")

            c2 = recovery_idx is not None
            print(f"    {_fmt_bool(c2)} C2: recovery_node 在 truck_route 中: {c2}")
            if not c2:
                violations.append(f"{pair_prefix} C2 sortie[{idx}] recovery_not_in_route")

            c3 = c1 and c2 and launch_idx < recovery_idx
            print(
                f"    {_fmt_bool(c3)} C3: launch_node 在 recovery_node 之前（route 索引）: {c3} "
                f"（launch_idx={launch_idx}, recovery_idx={recovery_idx}）"
            )
            if not c3:
                violations.append(f"{pair_prefix} C3 sortie[{idx}] launch_idx={launch_idx}, recovery_idx={recovery_idx}")

            overlap = sorted([cid for cid in sortie.customers if cid in set(route)])
            c4 = len(overlap) == 0
            print(f"    {_fmt_bool(c4)} C4: sortie 客户不在 truck_route 的卡车服务节点中: {c4}")
            if not c4:
                print(f"      × 重叠客户: {overlap}")
                violations.append(f"{pair_prefix} C4 sortie[{idx}] overlap={overlap}")

            c5 = len(sortie.customers) <= params.constraints.max_customers_per_sortie
            print(
                f"    {_fmt_bool(c5)} C5: sortie 客户数量 ≤ max_customers_per_sortie: {c5} "
                f"（实际={len(sortie.customers)}, 上限={params.constraints.max_customers_per_sortie}）"
            )
            if not c5:
                violations.append(
                    f"{pair_prefix} C5 sortie[{idx}] count={len(sortie.customers)} > {params.constraints.max_customers_per_sortie}"
                )
        print()

        print("  --- D. 逐个 Sortie 可行性约束检查 ---")
        for idx, sortie in enumerate(pair.sorties):
            print(f"  Sortie {idx}:")
            if not timeline_ok:
                print("    × D1-D5 无法检查：timeline 不可用")
                violations.append(f"{pair_prefix} D sortie[{idx}] timeline_unavailable")
                continue

            # D1 energy
            d1_ok, energy_val, d1_err = _safe_call(core.compute_sortie_energy, instance, sortie, timeline, home_status, params)
            d1 = d1_ok and energy_val <= params.energy.drone_battery_capacity
            if d1_ok:
                print(
                    f"    {_fmt_bool(d1)} D1 电量约束: sortie 总能量消耗 ≤ 电池容量: {d1} "
                    f"（消耗={energy_val} Wh, 容量={params.energy.drone_battery_capacity} Wh）"
                )
            else:
                print(f"    × D1 电量约束计算失败: {d1_err}")
            if not d1:
                violations.append(f"{pair_prefix} D1 sortie[{idx}] energy={energy_val} err={d1_err}")

            # D2 load
            load_track = []
            load_val = core.initial_sortie_load(instance, sortie)
            load_track.append(load_val)
            for customer_id in sortie.customers:
                load_val = core.update_sortie_load(load_val, customer_id, home_status[customer_id], instance)
                load_track.append(load_val)
            max_load = max(load_track) if load_track else 0.0
            d2 = max_load <= params.vehicle.drone_capacity
            print(
                f"    {_fmt_bool(d2)} D2 无人机载重约束: 各飞行阶段载重 ≤ drone_capacity: {d2} "
                f"（最大载重={max_load}, 容量={params.vehicle.drone_capacity}）"
            )
            if not d2:
                violations.append(f"{pair_prefix} D2 sortie[{idx}] max_load={max_load} > {params.vehicle.drone_capacity}")

            # D3 wait
            recovery = sortie.recovery_node
            wait_time = None
            try:
                wait_time = max(0.0, timeline.drone_arrival[recovery] - timeline.truck_arrival[recovery])
                d3 = wait_time <= params.constraints.max_truck_wait_time
                print(
                    f"    {_fmt_bool(d3)} D3 卡车等待约束: 卡车在 recovery_node 等待无人机的时间 ≤ max_truck_wait_time: {d3} "
                    f"（等待时间={wait_time}, 上限={params.constraints.max_truck_wait_time}）"
                )
            except Exception as exc:  # noqa: BLE001
                d3 = False
                print(f"    × D3 卡车等待约束计算失败: {exc}")
            if not d3:
                violations.append(f"{pair_prefix} D3 sortie[{idx}] wait_time={wait_time}")

            # D4 time windows
            d4 = True
            d4_violations: list[str] = []
            for customer_id in sortie.customers:
                try:
                    arrival = timeline.drone_arrival[customer_id]
                    tw = instance.customers[customer_id].time_window
                    if not (tw[0] <= arrival <= tw[1]):
                        d4 = False
                        d4_violations.append(f"customer={customer_id}, arrival={arrival}, tw={tw}")
                except Exception as exc:  # noqa: BLE001
                    d4 = False
                    d4_violations.append(f"customer={customer_id}, error={exc}")
            print(f"    {_fmt_bool(d4)} D4 无人机时间窗约束: 无人机到达每个客户的时间在该客户的 time_window 内: {d4}")
            if not d4:
                for item in d4_violations:
                    print(f"      × {item}")
                violations.append(f"{pair_prefix} D4 sortie[{idx}] {d4_violations}")

            # D5 truck load
            if load_ok:
                max_truck_load = max(load_state.truck_load_at_node.values()) if load_state.truck_load_at_node else 0.0
                d5 = max_truck_load <= params.vehicle.truck_capacity
                print(
                    f"    {_fmt_bool(d5)} D5 卡车载重约束: 卡车在各节点的载重 ≤ truck_capacity: {d5} "
                    f"（最大载重={max_truck_load}, 容量={params.vehicle.truck_capacity}）"
                )
            else:
                d5 = False
                print(f"    × D5 卡车载重约束无法计算: {load_err}")
            if not d5:
                violations.append(f"{pair_prefix} D5 sortie[{idx}] truck_load_check_failed")
        print()

        print("  --- E. 无人机可用性约束（物理约束）---")
        if len(pair.sorties) <= 1:
            print("  ✓ sortie 数量 <= 1，无相邻 sortie 可检查")
        else:
            for idx in range(len(pair.sorties) - 1):
                s_cur = pair.sorties[idx]
                s_next = pair.sorties[idx + 1]
                launch_idx_next = _route_index(route, s_next.launch_node)
                recovery_idx_cur = _route_index(route, s_cur.recovery_node)
                print(f"  sortie_{idx} recovery_node = {s_cur.recovery_node} (route_idx = {recovery_idx_cur})")
                print(f"  sortie_{idx+1} launch_node = {s_next.launch_node} (route_idx = {launch_idx_next})")

                if launch_idx_next is None or recovery_idx_cur is None:
                    print("  × 违反：无法定位 launch/recovery 节点在 route 中的索引")
                    violations.append(f"{pair_prefix} E sortie_{idx}->{idx+1} missing_route_index")
                elif launch_idx_next > recovery_idx_cur:
                    print(
                        f"  ✓ 合法：无人机在 route_idx={recovery_idx_cur} 回收后，"
                        f"卡车前进到 route_idx={launch_idx_next} 再次发射"
                    )
                elif launch_idx_next == recovery_idx_cur:
                    print(f"  × 边界情况：无人机在 route_idx={launch_idx_next} 回收后在同一节点立即再次发射")
                else:
                    print(
                        f"  × 违反：无人机尚在执行 sortie_{idx}（route_idx={recovery_idx_cur} 才回收），"
                        f"但 sortie_{idx+1} 已在 route_idx={launch_idx_next} 发射，物理上不可能——卡车上此时没有无人机"
                    )
                    violations.append(
                        f"{pair_prefix} E sortie_{idx}->{idx+1} launch_idx={launch_idx_next} < recovery_idx={recovery_idx_cur}"
                    )
        print()

    print("  --- F. 完整性约束 ---")
    universe = set(instance.customers.keys())

    truck_served: set[int] = set()
    drone_served: set[int] = set()
    sortie_occurrence: defaultdict[int, list[str]] = defaultdict(list)
    pair_occurrence: defaultdict[int, list[int]] = defaultdict(list)

    for pair in solution.vehicle_pairs:
        route = list(pair.truck_route)
        sortie_customers = {cid for s in pair.sorties for cid in s.customers}
        truck_nodes = {n for n in route if n != instance.depot_id}
        truck_only = truck_nodes - sortie_customers
        truck_served |= truck_only

        for s_idx, sortie in enumerate(pair.sorties):
            for cid in sortie.customers:
                drone_served.add(cid)
                sortie_occurrence[cid].append(f"pair={pair.pair_id},sortie={s_idx}")

        for cid in pair.all_customers:
            pair_occurrence[cid].append(pair.pair_id)

    unserved = set(solution.unserved_customers)
    covered = truck_served | drone_served | unserved
    missing = sorted(universe - covered)

    f1 = len(missing) == 0
    print(
        f"  {_fmt_bool(f1)} F1: 所有 instance 中的客户，要么在卡车服务/无人机服务/unserved 中。是否有遗漏: {f1}"
    )
    if not f1:
        print(f"    × 遗漏客户: {missing}")
        violations.append(f"[{tag}] F1 missing={missing}")

    overlap_td = sorted(truck_served & drone_served)
    f2 = len(overlap_td) == 0
    print(f"  {_fmt_bool(f2)} F2: 没有客户同时被卡车和无人机服务: {f2}")
    if not f2:
        print(f"    × 重复服务客户: {overlap_td}")
        violations.append(f"[{tag}] F2 truck_drone_overlap={overlap_td}")

    multi_sortie = sorted([cid for cid, refs in sortie_occurrence.items() if len(refs) > 1])
    f3 = len(multi_sortie) == 0
    print(f"  {_fmt_bool(f3)} F3: 没有客户出现在多个 sortie 中: {f3}")
    if not f3:
        for cid in multi_sortie:
            print(f"    × customer {cid} 出现在: {sortie_occurrence[cid]}")
        violations.append(f"[{tag}] F3 multi_sortie={multi_sortie}")

    multi_pair = sorted([cid for cid, refs in pair_occurrence.items() if len(set(refs)) > 1])
    f4 = len(multi_pair) == 0
    print(f"  {_fmt_bool(f4)} F4: 没有客户出现在多个 pair 中: {f4}")
    if not f4:
        for cid in multi_pair:
            print(f"    × customer {cid} 出现在 pair: {sorted(set(pair_occurrence[cid]))}")
        violations.append(f"[{tag}] F4 multi_pair={multi_pair}")

    print()
    print("=============================================")
    print()

    return violations


def main() -> None:
    print_constraint_catalog()

    instance, params, home_status, initial_solution, optimized_solution, groups = build_solutions()

    print("===== 构建解信息 =====")
    print(f"instance 客户总数: {len(instance.customers)}")
    print(f"grouping 结果组数: {len(groups)}")
    for idx, g in enumerate(groups):
        print(f"  Group {idx + 1}: {sorted(list(g))}")
    print("====================")
    print()

    print_solution_detail("INITIAL", instance, initial_solution)
    initial_violations = check_solution_constraints("INITIAL", instance, params, home_status, initial_solution)

    print_solution_detail("OPTIMIZED", instance, optimized_solution)
    optimized_violations = check_solution_constraints("OPTIMIZED", instance, params, home_status, optimized_solution)

    print("===== 汇总 =====")
    print(f"初始解约束违反数量: {len(initial_violations)}")
    print(f"初始解具体违反项: {initial_violations}")
    print(f"优化解约束违反数量: {len(optimized_violations)}")
    print(f"优化解具体违反项: {optimized_violations}")
    print("================")


if __name__ == "__main__":
    main()
