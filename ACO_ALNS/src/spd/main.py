from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Protocol

from spd.aco_alns import ACOALNSSolver, PureACOOptimizer
from spd.aco_alns import ALNSOperatorSet, ALNSOptimizer
from spd.aco_alns import cross_pair_swap, cross_pair_transfer
from spd.aco_alns import d1_random_removal, d2_worst_removal, d3_related_removal, d4_low_home_probability_removal
from spd.aco_alns import r1_greedy_insertion, r2_regret_insertion, r3_timeslot_aware_insertion
from spd.config import (
    ACOParameters,
    ALNSParameters,
    BetaScheduleParameters,
    ConstraintParameters,
    CostParameters,
    ETPRCParameters,
    EnergyParameters,
    MonteCarloParameters,
    OnlineALNSParameters,
    ProblemParameters,
    TimeParameters,
    VehicleParameters,
)
from spd.core import (
    compute_actual_cost_breakdown,
    compute_objective,
    compute_objective_breakdown,
    compute_truck_timeline,
    get_arrival_time,
)
from spd.etprc import ETPRCBuilder
from spd.online import LightweightALNSOptimizer, OnlineReplanner
from spd.simulation import ExecutionEngine, MonteCarloSimulator
from spd.types import Customer, CustomerType, ExecutionResult, ProblemInstance, Solution


def _build_distance_matrix(coords: dict[int, tuple[float, float]], scale: float = 1.0) -> dict[tuple[int, int], float]:
    matrix: dict[tuple[int, int], float] = {}
    for i, (x1, y1) in coords.items():
        for j, (x2, y2) in coords.items():
            if i == j:
                matrix[(i, j)] = 0.0
            else:
                matrix[(i, j)] = scale * math.hypot(x2 - x1, y2 - y1)
    return matrix


def _load_instance(path: str | Path) -> ProblemInstance:
    payload = json.loads(Path(path).read_text(encoding='utf-8'))

    depot_id = int(payload.get('depot_id', 0))
    depot = payload.get('depot', {})
    depot_x = float(depot.get('x', 0.0))
    depot_y = float(depot.get('y', 0.0))

    raw_customers = payload.get('customers', [])
    if isinstance(raw_customers, dict):
        customer_items = list(raw_customers.values())
    elif isinstance(raw_customers, list):
        customer_items = raw_customers
    else:
        raise ValueError('customers must be a list or object')

    customers: dict[int, Customer] = {}
    for item in customer_items:
        customer_id = int(item['customer_id'])
        customer_type = CustomerType(str(item['customer_type']).lower())
        time_window = tuple(float(v) for v in item['time_window'])
        if len(time_window) != 2:
            raise ValueError(f'customer {customer_id} time_window must have 2 values')
        home_probabilities = tuple(float(v) for v in item['home_probabilities'])

        customers[customer_id] = Customer(
            customer_id=customer_id,
            x=float(item['x']),
            y=float(item['y']),
            customer_type=customer_type,
            weight=float(item['weight']),
            time_window=(time_window[0], time_window[1]),
            service_time=float(item['service_time']),
            home_probabilities=home_probabilities,
        )

    if not customers:
        raise ValueError('customers cannot be empty')

    if 'delivery_customers' in payload:
        delivery_customers = frozenset(int(v) for v in payload['delivery_customers'])
    else:
        delivery_customers = frozenset(
            customer_id
            for customer_id, customer in customers.items()
            if customer.customer_type == CustomerType.DELIVERY
        )

    if 'pickup_customers' in payload:
        pickup_customers = frozenset(int(v) for v in payload['pickup_customers'])
    else:
        pickup_customers = frozenset(
            customer_id
            for customer_id, customer in customers.items()
            if customer.customer_type == CustomerType.PICKUP
        )

    vehicle_pair_count = payload.get('vehicle_pair_count')
    if vehicle_pair_count is None:
        trucks = payload.get('trucks', [])
        drones = payload.get('drones', [])
        inferred = min(len(trucks), len(drones)) if trucks and drones else 1
        vehicle_pair_count = inferred
    vehicle_pair_count = int(vehicle_pair_count)

    coords: dict[int, tuple[float, float]] = {depot_id: (depot_x, depot_y)}
    for customer_id, customer in customers.items():
        coords[customer_id] = (customer.x, customer.y)

    truck_scale = float(payload.get('truck_distance_scale', 1.2))
    drone_scale = float(payload.get('drone_distance_scale', 1.0))

    truck_distance_km = _build_distance_matrix(coords, scale=truck_scale)
    drone_distance_km = _build_distance_matrix(coords, scale=drone_scale)

    return ProblemInstance(
        depot_id=depot_id,
        vehicle_pair_count=vehicle_pair_count,
        customers=customers,
        delivery_customers=delivery_customers,
        pickup_customers=pickup_customers,
        truck_distance_km=truck_distance_km,
        drone_distance_km=drone_distance_km,
    )


def _load_parameters(path: str | Path) -> ProblemParameters:
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    monte_carlo_payload = payload.get('monte_carlo', {})
    if not isinstance(monte_carlo_payload, dict):
        raise ValueError('monte_carlo must be an object if provided')

    # 允许纯 ACO 配置仅设置 alns.iterations，其余字段回填默认值以兼容既有参数对象。
    alns_payload = payload.get('alns', {})
    if not isinstance(alns_payload, dict):
        raise ValueError('alns must be an object if provided')
    alns_defaults = {
        'iterations': 50,
        'remove_count_min': 1,
        'remove_count_max': 3,
        'sa_initial_temperature': 100.0,
        'sa_cooling_rate': 0.98,
        'operator_weight_init': 1.0,
        'reward_global_best': 33.0,
        'reward_improve': 13.0,
        'reward_accept_worse': 9.0,
        'reaction_factor': 0.1,
        'cross_group_frequency': 10,
        'cross_group_remove_count': 1,
    }
    merged_alns_payload = {**alns_defaults, **alns_payload}

    return ProblemParameters(
        time=TimeParameters(**payload['time']),
        vehicle=VehicleParameters(**payload['vehicle']),
        energy=EnergyParameters(**payload['energy']),
        cost=CostParameters(**payload['cost']),
        beta_schedule=BetaScheduleParameters(**payload['beta_schedule']),
        constraints=ConstraintParameters(**payload['constraints']),
        aco=ACOParameters(**payload['aco']),
        alns=ALNSParameters(**merged_alns_payload),
        etprc=ETPRCParameters(**payload['etprc']),
        online_alns=OnlineALNSParameters(**payload['online_alns']),
        monte_carlo=MonteCarloParameters(**monte_carlo_payload),
    )


def _build_operator_set() -> ALNSOperatorSet:
    return ALNSOperatorSet(
        destroy_operators=[d1_random_removal, d2_worst_removal, d3_related_removal, d4_low_home_probability_removal],
        repair_operators=[r1_greedy_insertion, r2_regret_insertion, r3_timeslot_aware_insertion],
        cross_group_operators=[cross_pair_swap, cross_pair_transfer],
    )


def _build_offline_solver(optimizer_type: str):
    """根据配置选择离线优化器。"""
    normalized = str(optimizer_type).strip().lower()
    if normalized == 'pure_aco':
        return PureACOOptimizer(etprc_builder=ETPRCBuilder())
    if normalized == 'pure_alns':
        # 兼容历史分支：仅在工程中已有 PureALNSOptimizer 时启用。
        from spd.aco_alns import PureALNSOptimizer  # type: ignore[attr-defined]

        return PureALNSOptimizer()
    return ACOALNSSolver(etprc_builder=ETPRCBuilder(), alns_optimizer=ALNSOptimizer(_build_operator_set()))


# ============================================================
# 鍘熷鏂囦欢: spd/app/cli.py
# ============================================================

"""CLI contract for future command-line integration."""


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for offline solve and online simulation commands."""
    parser = argparse.ArgumentParser(prog='spd-planner')
    parser.add_argument('command', choices=['solve', 'simulate'])
    parser.add_argument('--instance', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', type=str, default=None, help='图片输出目录，不指定则不生成图片')
    parser.add_argument('--no-plot', action='store_true', help='跳过可视化')
    return parser


def main() -> int:
    """CLI entry point with JSON loading and command dispatch."""
    parser = build_parser()
    args = parser.parse_args()

    instance = _load_instance(args.instance)
    config_payload = json.loads(Path(args.config).read_text(encoding='utf-8'))
    params = _load_parameters(args.config)
    mc_trials = max(1, int(getattr(getattr(params, 'monte_carlo', None), 'trial_count', 30)))
    optimizer_type = str(config_payload.get('optimizer', {}).get('type', 'aco_alns')).strip().lower()
    is_deterministic = bool(config_payload.get('deterministic', False))

    offline_solver = _build_offline_solver(optimizer_type)
    planner = SPDPlanner(
        offline_solver=offline_solver,
        execution_engine=ExecutionEngine(OnlineReplanner(LightweightALNSOptimizer())),
    )

    plot_enabled = args.output_dir is not None and not args.no_plot
    output_dir = Path(args.output_dir) if args.output_dir is not None else None
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    if plot_enabled and output_dir is not None:
        from spd import visualization as viz

    rng = random.Random(0)

    def _write_json_file(filename: str, data: object) -> None:
        if output_dir is None:
            return
        (output_dir / filename).write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding='utf-8',
        )

    def _percentile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(float(v) for v in values)
        if len(ordered) == 1:
            return ordered[0]
        rank = (len(ordered) - 1) * (q / 100.0)
        low = int(math.floor(rank))
        high = int(math.ceil(rank))
        if low == high:
            return ordered[low]
        frac = rank - low
        return ordered[low] * (1.0 - frac) + ordered[high] * frac

    def _normalize_iteration_history(iteration_history: list[dict[str, object]]) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for row in iteration_history:
            if not isinstance(row, dict):
                continue
            normalized: dict[str, object] = {}
            for key, value in row.items():
                if key == 'operator_scores' and isinstance(value, dict):
                    normalized_scores: dict[str, object] = {}
                    for op_name, op_value in value.items():
                        try:
                            normalized_scores[str(op_name)] = float(op_value)
                        except (TypeError, ValueError):
                            normalized_scores[str(op_name)] = op_value
                    normalized[key] = normalized_scores
                else:
                    normalized[key] = value
            records.append(normalized)
        return records

    def _build_solution_detail(solution: Solution) -> dict[str, object]:
        pair_details: list[dict[str, object]] = []
        for pair in solution.vehicle_pairs:
            sortie_customers = {customer_id for sortie in pair.sorties for customer_id in sortie.customers}
            truck_direct = sorted(
                node
                for node in pair.truck_route
                if node != instance.depot_id and node not in sortie_customers
            )
            pair_details.append(
                {
                    'pair_id': pair.pair_id,
                    'truck_route': list(pair.truck_route),
                    'truck_customers': truck_direct,
                    'sorties': [
                        {
                            'sortie_index': sortie_index,
                            'launch_node': sortie.launch_node,
                            'recovery_node': sortie.recovery_node,
                            'customers': list(sortie.customers),
                        }
                        for sortie_index, sortie in enumerate(pair.sorties)
                    ],
                }
            )

        return {
            'command': 'solve',
            'instance_file': Path(args.instance).name,
            'config_file': Path(args.config).name,
            'vehicle_pairs': pair_details,
            'all_customers': sorted(solution.all_customers),
            'unserved_customers': sorted(solution.unserved_customers),
        }

    def _build_customer_service_report(solution: Solution, status_map: dict[int, bool]) -> dict[str, object]:
        assignment: dict[int, tuple[str, int | None, int | None]] = {}
        arrival_times: dict[int, float] = {}

        for pair in solution.vehicle_pairs:
            drone_assignment: dict[int, int] = {}
            for sortie_index, sortie in enumerate(pair.sorties):
                for customer_id in sortie.customers:
                    drone_assignment[customer_id] = sortie_index

            for customer_id in pair.all_customers:
                if customer_id in drone_assignment:
                    assignment[customer_id] = ('drone', pair.pair_id, drone_assignment[customer_id])
                else:
                    assignment[customer_id] = ('truck', pair.pair_id, None)

            try:
                timeline = compute_truck_timeline(instance, pair, status_map, params)
            except (AssertionError, KeyError, ValueError):
                continue

            for customer_id in pair.all_customers:
                try:
                    arrival_times[customer_id] = float(get_arrival_time(customer_id, pair, timeline))
                except (AssertionError, KeyError, ValueError):
                    continue

        customer_rows: list[dict[str, object]] = []
        for customer_id in sorted(instance.customers):
            customer = instance.customers[customer_id]

            if customer_id in solution.unserved_customers:
                served_by = 'unserved'
                served_pair = None
                served_sortie = None
            else:
                served_info = assignment.get(customer_id)
                if served_info is None:
                    served_by = 'unserved'
                    served_pair = None
                    served_sortie = None
                else:
                    served_by, served_pair, served_sortie = served_info

            arrival_time = arrival_times.get(customer_id)
            tw_start, tw_end = customer.time_window
            within_tw = None if arrival_time is None else bool(tw_start <= arrival_time <= tw_end)

            customer_rows.append(
                {
                    'customer_id': customer_id,
                    'customer_type': customer.customer_type.value,
                    'x': customer.x,
                    'y': customer.y,
                    'weight': customer.weight,
                    'time_window': [tw_start, tw_end],
                    'service_time': customer.service_time,
                    'home_probabilities': list(customer.home_probabilities),
                    'served_by': served_by,
                    'served_in_pair': served_pair,
                    'served_in_sortie': served_sortie,
                    'arrival_time': arrival_time,
                    'within_time_window': within_tw,
                }
            )

        return {'customers': customer_rows}

    def _build_online_execution_report(result: ExecutionResult) -> dict[str, object]:
        total_customers = len(instance.customers)
        failed_count = len(result.failed_customers)
        success_rate = ((total_customers - failed_count) / total_customers * 100.0) if total_customers > 0 else 0.0
        served_customers = sorted(result.solution.all_customers) if result.solution is not None else []
        return {
            'actual_cost': result.actual_cost,
            'failed_customers': sorted(result.failed_customers),
            'served_customers': served_customers,
            'total_customers': total_customers,
            'success_rate': success_rate,
        }

    def _build_mc_results_report(mc_result, deterministic_cost: float) -> dict[str, object]:
        samples = [float(sample) for sample in mc_result.samples]
        return {
            'trial_count': int(mc_result.trial_count),
            'samples': samples,
            'mean': float(mc_result.average_actual_cost),
            'std': float(mc_result.cost_std),
            'ci_95_low': _percentile(samples, 2.5),
            'ci_95_high': _percentile(samples, 97.5),
            'deterministic_cost': float(deterministic_cost),
        }

    if args.command == 'solve':
        # Backward-compatible path.
        if not plot_enabled:
            home_status = {customer_id: True for customer_id in instance.customers}
            t0 = time.perf_counter()
            solution = planner.solve_offline(instance, params, rng)
            solve_seconds = time.perf_counter() - t0
            optimized_objective = compute_objective(instance, solution, home_status, params)
            cost_breakdown = compute_objective_breakdown(instance, solution, home_status, params)
            objective_summary = {
                'initial_objective': None,
                'optimized_objective': float(optimized_objective),
                'improvement_percent': None,
                'note': 'no_plot_mode',
            }
            timing = {
                'solve_seconds': float(solve_seconds),
                'total_seconds': float(solve_seconds),
            }
            _write_json_file('solution_detail.json', _build_solution_detail(solution))
            _write_json_file('objective_summary.json', objective_summary)
            _write_json_file('cost_breakdown.json', cost_breakdown)
            _write_json_file('timing.json', timing)
            _write_json_file('timing_report.json', timing)
            print(
                json.dumps(
                    {
                        'command': 'solve',
                        'status': 'ok',
                        'vehicle_pairs': len(solution.vehicle_pairs),
                        'served_customers': len(solution.all_customers),
                        'unserved_customers': sorted(solution.unserved_customers),
                        'optimized_objective': float(optimized_objective),
                        'solve_seconds': float(solve_seconds),
                    },
                    ensure_ascii=False,
                )
            )
            return 0

        timing: dict[str, float] = {}

        etprc_builder = ETPRCBuilder()
        rng_initial = random.Random(0)
        t0 = time.perf_counter()
        initial_solution = etprc_builder.build_initial_solution(instance, params, rng_initial)
        timing['initial_solution_seconds'] = time.perf_counter() - t0

        # 确定性模式强制所有客户在家，保证离线算法对比口径一致。
        home_status = {customer_id: True for customer_id in instance.customers}
        initial_objective = compute_objective(instance, initial_solution, home_status, params)
        initial_breakdown = compute_objective_breakdown(instance, initial_solution, home_status, params)
        if abs(initial_breakdown['z_total'] - float(initial_objective)) > 1e-6:
            raise RuntimeError('initial objective and breakdown total mismatch')

        if not is_deterministic:
            rng_group = random.Random(0)
            t0 = time.perf_counter()
            groups = etprc_builder._customer_grouping(instance, params, rng_group)
            timing['clustering_seconds'] = time.perf_counter() - t0
            clustering_result = {
                'groups': [sorted(list(group)) for group in groups],
                'truck_customers': [],
                'drone_customers': [],
            }
            viz.plot_clustering(
                instance,
                clustering_result,
                save_path=output_dir / 'clustering.png',
                initial_solution=initial_solution,
            )
            viz.plot_initial_solution(
                instance,
                {'solution': initial_solution, 'objective': initial_objective},
                save_path=output_dir / 'initial_solution.png',
            )

        rng_solve = random.Random(0)
        t0 = time.perf_counter()
        if optimizer_type == 'pure_aco':
            if not isinstance(offline_solver, PureACOOptimizer):
                raise RuntimeError('optimizer_type is pure_aco but offline solver is not PureACOOptimizer')
            solution, optimized_objective, optimization_history = offline_solver.optimize(
                instance=instance,
                initial_solution=initial_solution,
                params=params,
                home_status=home_status,
                rng=rng_solve,
            )
        else:
            solution = planner.solve_offline(instance, params, rng_solve)
            optimized_objective = compute_objective(instance, solution, home_status, params)
            optimization_history = list(getattr(offline_solver, 'iteration_history', []))
        timing['optimization_seconds'] = time.perf_counter() - t0

        optimized_breakdown = compute_objective_breakdown(instance, solution, home_status, params)
        if abs(optimized_breakdown['z_total'] - float(optimized_objective)) > 1e-6:
            raise RuntimeError('optimized objective and breakdown total mismatch')

        # 非 pure_aco 的历史沿用旧格式，补一条初始解记录以兼容原有可视化语义。
        if optimizer_type != 'pure_aco':
            initial_history_row = {
                'iteration': 0,
                'current_cost': float(initial_objective),
                'best_cost': float(initial_objective),
                'operator_name': 'initial_solution',
                'operator_scores': {},
                'aco_iteration': 0,
                'ant_index': 0,
            }
            if optimization_history:
                first_cost = optimization_history[0].get('current_cost')
                if first_cost is None or abs(float(first_cost) - float(initial_objective)) > 1e-9:
                    optimization_history.insert(0, initial_history_row)
            else:
                optimization_history.append(initial_history_row)

        cost_breakdown: dict[str, object] = {
            'initial_solution': initial_breakdown,
            'optimized_solution': optimized_breakdown,
        }
        first_trial_result = None
        mc_result = None

        if is_deterministic:
            # 确定性模式只保留离线阶段图表，不生成 MC/在线重规划相关图片。
            viz.plot_solution_comparison(
                instance,
                {'solution': initial_solution, 'objective': initial_objective},
                {'solution': solution, 'objective': optimized_objective},
                save_path=output_dir / 'solution_comparison.png',
                cost_breakdown=cost_breakdown,
                case_name=Path(args.instance).stem,
            )
            viz.plot_convergence_curve(optimization_history, save_path=output_dir / 'convergence_curve.png')
            viz.plot_cost_breakdown_table(cost_breakdown, save_path=output_dir / 'cost_breakdown_table.png')
        else:
            viz.plot_optimized_solution(
                instance,
                {'solution': solution, 'objective': optimized_objective},
                save_path=output_dir / 'optimized_solution.png',
            )
            viz.plot_convergence_curve(optimization_history, save_path=output_dir / 'convergence_curve.png')

            solution_payload = {
                'solution': solution,
                'instance': instance,
                'params': params,
                'home_status': home_status,
            }
            viz.plot_timeline_gantt(solution_payload, save_path=output_dir / 'timeline_gantt.png')
            viz.plot_energy_consumption(solution_payload, save_path=output_dir / 'energy_consumption.png')
            if optimizer_type != 'pure_aco':
                viz.plot_operator_statistics(optimization_history, save_path=output_dir / 'operator_statistics.png')

            t0 = time.perf_counter()
            simulator = MonteCarloSimulator(ExecutionEngine(OnlineReplanner(LightweightALNSOptimizer())))
            mc_result = simulator.run(instance, solution, params, trial_count=mc_trials, rng=random.Random(2))
            timing['monte_carlo_seconds'] = time.perf_counter() - t0
            first_trial_result = simulator.first_trial_result
            if first_trial_result is None:
                raise RuntimeError('MC trial 0 result is unavailable')

            online_breakdown = compute_actual_cost_breakdown(first_trial_result, params)
            if abs(online_breakdown['actual_total'] - float(first_trial_result.actual_cost)) > 1e-6:
                raise RuntimeError('online actual cost and breakdown total mismatch')
            cost_breakdown['online_trial_0'] = online_breakdown

            viz.plot_solution_comparison(
                instance,
                {'solution': initial_solution, 'objective': initial_objective},
                {'solution': solution, 'objective': optimized_objective},
                save_path=output_dir / 'solution_comparison.png',
                online_solution={'solution': first_trial_result.solution, 'objective': first_trial_result.actual_cost},
                failed_customers=first_trial_result.failed_customers,
                cost_breakdown=cost_breakdown,
                case_name=Path(args.instance).stem,
            )
            viz.plot_monte_carlo_distribution(
                {'samples': mc_result.samples, 'deterministic': first_trial_result.actual_cost},
                save_path=output_dir / 'monte_carlo_distribution.png',
            )
            snapshots = getattr(first_trial_result, 'replanning_snapshots', [])
            if len(snapshots) > 1:
                viz.plot_replanning_snapshots(
                    instance=instance,
                    snapshots=snapshots,
                    save_path=output_dir,
                    filename_prefix=f"{Path(args.instance).stem}_replanning",
                )

        improvement_percent = (
            ((initial_objective - optimized_objective) / initial_objective * 100.0)
            if abs(initial_objective) > 1e-9
            else 0.0
        )
        objective_summary = {
            'initial_objective': float(initial_objective),
            'optimized_objective': float(optimized_objective),
            'improvement_percent': round(float(improvement_percent), 2),
            'note': (
                'no improvement found with current parameters'
                if abs(initial_objective - optimized_objective) <= 1e-9
                else ''
            ),
        }

        convergence_data = {'records': _normalize_iteration_history(optimization_history)}
        online_execution_report = (
            _build_online_execution_report(first_trial_result) if first_trial_result is not None else None
        )
        mc_results_report = (
            _build_mc_results_report(mc_result, first_trial_result.actual_cost)
            if mc_result is not None and first_trial_result is not None
            else None
        )

        timing['total_seconds'] = sum(value for value in timing.values() if isinstance(value, float))

        _write_json_file('solution_detail.json', _build_solution_detail(solution))
        _write_json_file('objective_summary.json', objective_summary)
        _write_json_file('customer_service_report.json', _build_customer_service_report(solution, home_status))
        _write_json_file('timing_report.json', timing)
        _write_json_file('timing.json', timing)
        _write_json_file('convergence_data.json', convergence_data)
        _write_json_file('cost_breakdown.json', cost_breakdown)
        if mc_results_report is not None:
            _write_json_file('mc_results.json', mc_results_report)
        if online_execution_report is not None:
            _write_json_file('online_execution_report.json', online_execution_report)

        print(
            json.dumps(
                {
                    'command': 'solve',
                    'status': 'ok',
                    'vehicle_pairs': len(solution.vehicle_pairs),
                    'served_customers': len(solution.all_customers),
                    'unserved_customers': sorted(solution.unserved_customers),
                },
                ensure_ascii=False,
            )
        )
        return 0

    # simulate
    if not plot_enabled:
        planned_solution = planner.solve_offline(instance, params, rng)
        result = planner.execute_online(instance, planned_solution, params, rng)
        print(
            json.dumps(
                {
                    'command': 'simulate',
                    'status': 'ok',
                    'actual_cost': result.actual_cost,
                    'failed_customers': sorted(result.failed_customers),
                    'served_customers': len(result.solution.all_customers),
                },
                ensure_ascii=False,
            )
        )
        return 0

    planned_solution = planner.solve_offline(instance, params, rng)
    planned_home_status = {customer_id: True for customer_id in instance.customers}
    planned_objective = compute_objective(instance, planned_solution, planned_home_status, params)
    planned_breakdown = compute_objective_breakdown(instance, planned_solution, planned_home_status, params)
    if abs(planned_breakdown['z_total'] - float(planned_objective)) > 1e-6:
        raise RuntimeError('planned objective and breakdown total mismatch')
    timing: dict[str, float] = {}

    t0 = time.perf_counter()
    simulator = MonteCarloSimulator(ExecutionEngine(OnlineReplanner(LightweightALNSOptimizer())))
    mc_result = simulator.run(
        instance,
        planned_solution,
        params,
        trial_count=mc_trials,
        rng=random.Random(1),
    )
    timing['monte_carlo_seconds'] = time.perf_counter() - t0
    first_trial_result = simulator.first_trial_result
    if first_trial_result is None:
        raise RuntimeError('MC trial 0 result is unavailable')
    online_breakdown = compute_actual_cost_breakdown(first_trial_result, params)
    if abs(online_breakdown['actual_total'] - float(first_trial_result.actual_cost)) > 1e-6:
        raise RuntimeError('online actual cost and breakdown total mismatch')
    cost_breakdown = {
        'initial_solution': planned_breakdown,
        'optimized_solution': planned_breakdown,
        'online_trial_0': online_breakdown,
    }

    viz.plot_solution_comparison(
        instance,
        {'solution': planned_solution, 'objective': planned_objective},
        {'solution': planned_solution, 'objective': planned_objective},
        save_path=output_dir / 'solution_comparison.png',
        online_solution={'solution': first_trial_result.solution, 'objective': first_trial_result.actual_cost},
        failed_customers=first_trial_result.failed_customers,
        cost_breakdown=cost_breakdown,
        case_name=Path(args.instance).stem,
    )
    viz.plot_monte_carlo_distribution(
        {'samples': mc_result.samples, 'deterministic': first_trial_result.actual_cost},
        save_path=output_dir / 'monte_carlo_distribution.png',
    )
    snapshots = getattr(first_trial_result, 'replanning_snapshots', [])
    if len(snapshots) > 1:
        viz.plot_replanning_snapshots(
            instance=instance,
            snapshots=snapshots,
            save_path=output_dir,
            filename_prefix=f"{Path(args.instance).stem}_replanning",
        )

    timing['total_seconds'] = sum(value for value in timing.values() if isinstance(value, float))
    _write_json_file('mc_results.json', _build_mc_results_report(mc_result, first_trial_result.actual_cost))
    _write_json_file('online_execution_report.json', _build_online_execution_report(first_trial_result))
    _write_json_file('cost_breakdown.json', cost_breakdown)
    _write_json_file('timing_report.json', timing)

    print(
        json.dumps(
            {
                'command': 'simulate',
                'status': 'ok',
                'actual_cost': first_trial_result.actual_cost,
                'failed_customers': sorted(first_trial_result.failed_customers),
                'served_customers': len(first_trial_result.solution.all_customers),
            },
            ensure_ascii=False,
        )
    )
    return 0


# ============================================================
# 鍘熷鏂囦欢: spd/app/planner.py
# ============================================================

"""High-level orchestrator that wires offline and online components."""


class OfflineSolverProtocol(Protocol):
    """离线求解器协议：只约束 solve 接口，便于切换不同优化器实现。"""

    def solve(
        self,
        instance: ProblemInstance,
        params: ProblemParameters,
        home_status: dict[int, bool],
        rng: random.Random | None = None,
    ) -> Solution:
        ...


class SPDPlanner:
    """Facade for solving and simulating the SPD problem."""

    def __init__(self, offline_solver: OfflineSolverProtocol, execution_engine: ExecutionEngine):
        self._offline_solver = offline_solver
        self._execution_engine = execution_engine

    def solve_offline(
        self,
        instance: ProblemInstance,
        params: ProblemParameters,
        rng: random.Random | None = None,
    ) -> Solution:
        """Return an offline plan assuming pre-defined home-status policy."""
        home_status = {customer_id: True for customer_id in instance.customers}
        return self._offline_solver.solve(instance, params, home_status, rng)

    def execute_online(
        self,
        instance: ProblemInstance,
        planned_solution: Solution,
        params: ProblemParameters,
        rng: random.Random | None = None,
    ) -> ExecutionResult:
        """Simulate online execution with event-triggered replanning."""
        return self._execution_engine.run(instance, planned_solution, params, rng)


if __name__ == '__main__':
    raise SystemExit(main())






