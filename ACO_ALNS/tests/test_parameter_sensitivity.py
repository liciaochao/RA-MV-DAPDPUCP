from __future__ import annotations

from dataclasses import replace
import random
import time

import pytest

from spd.config import ProblemParameters
from spd.core import all_hard_constraints_satisfied
from spd.core import compute_objective
from spd.aco_alns import ACOALNSSolver
from spd.aco_alns import ALNSOperatorSet, ALNSOptimizer
from spd.etprc import ETPRCBuilder
from spd.aco_alns import cross_pair_swap, cross_pair_transfer
from spd.aco_alns import (
    d1_random_removal,
    d2_worst_removal,
    d3_related_removal,
    d4_low_home_probability_removal,
)
from spd.aco_alns import r1_greedy_insertion, r2_regret_insertion, r3_timeslot_aware_insertion
from spd.online import LightweightALNSOptimizer
from spd.online import OnlineReplanner
from spd.simulation import ExecutionEngine
from spd.simulation import MonteCarloSimulator


def _build_operator_set() -> ALNSOperatorSet:
    """Build the standard ALNS operator set for sensitivity experiments."""
    return ALNSOperatorSet(
        destroy_operators=[d1_random_removal, d2_worst_removal, d3_related_removal, d4_low_home_probability_removal],
        repair_operators=[r1_greedy_insertion, r2_regret_insertion, r3_timeslot_aware_insertion],
        cross_group_operators=[cross_pair_swap, cross_pair_transfer],
    )


class _RecordingSolver(ACOALNSSolver):
    """Solver subclass that records beta progression for gamma-beta experiments."""

    def __init__(self, etprc_builder: ETPRCBuilder, alns_optimizer: ALNSOptimizer):
        super().__init__(etprc_builder=etprc_builder, alns_optimizer=alns_optimizer)
        self.beta_history: list[float] = []

    def advance_beta_schedule(self, state, params: ProblemParameters) -> float:
        beta = super().advance_beta_schedule(state, params)
        self.beta_history.append(beta)
        return beta


def _build_solver(record_beta: bool = False) -> ACOALNSSolver:
    """Create ACO-ALNS solver for experiments."""
    etprc_builder = ETPRCBuilder()
    alns_optimizer = ALNSOptimizer(_build_operator_set())
    if record_beta:
        return _RecordingSolver(etprc_builder=etprc_builder, alns_optimizer=alns_optimizer)
    return ACOALNSSolver(etprc_builder=etprc_builder, alns_optimizer=alns_optimizer)


def _small_params(sample_params: ProblemParameters) -> ProblemParameters:
    """Small ACO settings for fast sensitivity experiments."""
    return replace(
        sample_params,
        aco=replace(sample_params.aco, max_iterations=3, ant_count=2),
        alns=replace(sample_params.alns, iterations=30),
    )


def _print_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    """Print a markdown-like result table for experiment comparison."""
    print(f"\n=== {title} ===")
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["-" * max(8, len(header)) for header in headers]) + " |")
    for row in rows:
        print("| " + " | ".join(row) + " |")


def _run_offline_pipeline(
    instance,
    params: ProblemParameters,
    all_home_status: dict[int, bool],
    seed: int,
    record_beta: bool = False,
) -> dict[str, object]:
    """Run ETPRC + ACO-ALNS + objective evaluation with robust exception capture."""
    start = time.perf_counter()
    try:
        builder = ETPRCBuilder()
        _ = builder.build_initial_solution(instance, params, random.Random(seed))

        solver = _build_solver(record_beta=record_beta)
        solution = solver.solve(instance, params, all_home_status, random.Random(seed))
        objective = compute_objective(instance, solution, all_home_status, params)
        feasible = all_hard_constraints_satisfied(instance, solution, all_home_status, params)

        final_beta = params.beta_schedule.beta_init
        if record_beta and isinstance(solver, _RecordingSolver) and solver.beta_history:
            final_beta = solver.beta_history[-1]

        elapsed = time.perf_counter() - start
        return {
            "success": True,
            "solution": solution,
            "objective": objective,
            "feasible": feasible,
            "final_beta": final_beta,
            "elapsed": elapsed,
            "error": "",
        }
    except Exception as exc:  # pragma: no cover - exception capture path is for reporting stability
        elapsed = time.perf_counter() - start
        return {
            "success": False,
            "solution": None,
            "objective": float("inf"),
            "feasible": False,
            "final_beta": params.beta_schedule.beta_init,
            "elapsed": elapsed,
            "error": f"{type(exc).__name__}: {exc}",
        }


def test_sensitivity_1_aco_ant_count(sample_instance, sample_params, all_home_status) -> None:
    base_params = _small_params(sample_params)
    ant_counts = [1, 2, 3, 5]
    seed = 2026

    results: dict[int, dict[str, object]] = {}
    rows: list[list[str]] = []

    for ant_count in ant_counts:
        params = replace(base_params, aco=replace(base_params.aco, ant_count=ant_count, max_iterations=3))
        result = _run_offline_pipeline(sample_instance, params, all_home_status, seed)
        results[ant_count] = result

        if result["success"]:
            rows.append([str(ant_count), f"{result['objective']:.4f}", f"{result['elapsed']:.3f}"])
        else:
            rows.append([str(ant_count), "ERROR", str(result["error"])])

    _print_table("实验 1：ACO 蚂蚁数量", ["n_ants", "objective", "time(s)/error"], rows)

    assert results[1]["success"], f"n_ants=1 run failed: {results[1]['error']}"
    baseline = float(results[1]["objective"])

    for ant_count in [2, 3, 5]:
        assert results[ant_count]["success"], f"n_ants={ant_count} run failed: {results[ant_count]['error']}"
        assert float(results[ant_count]["objective"]) <= baseline + 1e-9


def test_sensitivity_2_alns_iterations(sample_instance, sample_params, all_home_status) -> None:
    base_params = _small_params(sample_params)
    alns_iterations = [10, 30, 50, 100]
    seed = 2026

    results: dict[int, dict[str, object]] = {}
    rows: list[list[str]] = []

    for iterations in alns_iterations:
        params = replace(base_params, alns=replace(base_params.alns, iterations=iterations))
        result = _run_offline_pipeline(sample_instance, params, all_home_status, seed)
        results[iterations] = result

        if result["success"]:
            rows.append([str(iterations), f"{result['objective']:.4f}", f"{result['elapsed']:.3f}"])
        else:
            rows.append([str(iterations), "ERROR", str(result["error"])])

    _print_table("实验 2：ALNS 迭代次数", ["alns_iter", "objective", "time(s)/error"], rows)

    for iterations in alns_iterations:
        assert results[iterations]["success"], f"alns_iterations={iterations} failed: {results[iterations]['error']}"

    objectives = [float(results[it]["objective"]) for it in alns_iterations]

    # Allow stochastic fluctuations, but require at least one quality improvement step as iterations increase.
    assert any(
        objectives[idx + 1] <= objectives[idx] + 1e-9
        for idx in range(len(objectives) - 1)
    )
    # Final quality should not be worse than the worst earlier configuration.
    assert objectives[-1] <= max(objectives[:-1]) + 1e-9


def test_sensitivity_3_sa_cooling_rate(sample_instance, sample_params, all_home_status) -> None:
    base_params = _small_params(sample_params)
    cooling_rates = [0.90, 0.95, 0.99]
    seed = 2026

    rows: list[list[str]] = []
    for rate in cooling_rates:
        params = replace(base_params, alns=replace(base_params.alns, sa_cooling_rate=rate))
        result = _run_offline_pipeline(sample_instance, params, all_home_status, seed)

        if result["success"]:
            rows.append([f"{rate:.2f}", f"{result['objective']:.4f}", str(result["feasible"])])
        else:
            rows.append([f"{rate:.2f}", "ERROR", str(result["error"])])

        assert result["success"], f"sa_cooling_rate={rate} failed: {result['error']}"
        assert bool(result["feasible"]) is True

    _print_table("实验 3：SA 冷却速率", ["cooling_rate", "objective", "feasible/error"], rows)


def test_sensitivity_4_gamma_beta(sample_instance, sample_params, all_home_status) -> None:
    base_params = _small_params(sample_params)
    gamma_values = [1.0, 1.1, 1.5, 2.0]
    seed = 2026

    rows: list[list[str]] = []
    for gamma_beta in gamma_values:
        params = replace(base_params, beta_schedule=replace(base_params.beta_schedule, gamma_beta=gamma_beta))
        result = _run_offline_pipeline(sample_instance, params, all_home_status, seed, record_beta=True)

        if result["success"]:
            rows.append(
                [
                    f"{gamma_beta:.2f}",
                    f"{result['final_beta']:.4f}",
                    f"{result['objective']:.4f}",
                ]
            )
        else:
            rows.append([f"{gamma_beta:.2f}", "ERROR", str(result["error"])])

        assert result["success"], f"gamma_beta={gamma_beta} failed: {result['error']}"
        if gamma_beta > 1.0:
            assert float(result["final_beta"]) > params.beta_schedule.beta_init

    _print_table("实验 4：beta 递增调度", ["gamma_beta", "final_beta", "objective/error"], rows)


def test_sensitivity_5_remove_count_range(sample_instance, sample_params, all_home_status) -> None:
    base_params = _small_params(sample_params)
    remove_ranges = [(1, 1), (1, 3), (2, 5)]
    seed = 2026

    rows: list[list[str]] = []
    for remove_min, remove_max in remove_ranges:
        params = replace(
            base_params,
            alns=replace(base_params.alns, remove_count_min=remove_min, remove_count_max=remove_max),
        )
        result = _run_offline_pipeline(sample_instance, params, all_home_status, seed)

        if result["success"]:
            rows.append([f"({remove_min},{remove_max})", f"{result['objective']:.4f}", str(result["feasible"])])
        else:
            rows.append([f"({remove_min},{remove_max})", "ERROR", str(result["error"])])

        assert result["success"], f"remove_count=({remove_min},{remove_max}) failed: {result['error']}"
        assert bool(result["feasible"]) is True

    _print_table("实验 5：remove_count 范围", ["remove_range", "objective", "feasible/error"], rows)


def test_sensitivity_6_online_iterations_vs_mc_cost(sample_instance, sample_params, all_home_status) -> None:
    base_params = _small_params(sample_params)
    iteration_values = [0, 5, 10, 20]
    seed = 2026

    rows: list[list[str]] = []
    results: dict[int, float] = {}

    for online_iterations in iteration_values:
        params = replace(
            base_params,
            online_alns=replace(base_params.online_alns, iterations=online_iterations),
        )

        try:
            solver = _build_solver()
            offline_solution = solver.solve(sample_instance, params, all_home_status, random.Random(seed))

            simulator = MonteCarloSimulator(ExecutionEngine(OnlineReplanner(LightweightALNSOptimizer())))
            mc_start = time.perf_counter()
            mc_result = simulator.run(
                sample_instance,
                offline_solution,
                params,
                trial_count=10,
                rng=random.Random(seed),
            )
            mc_elapsed = time.perf_counter() - mc_start

            results[online_iterations] = float(mc_result.average_actual_cost)
            rows.append([str(online_iterations), f"{mc_result.average_actual_cost:.4f}", f"{mc_elapsed:.3f}"])
        except Exception as exc:  # pragma: no cover - capture and report config failures
            rows.append([str(online_iterations), "ERROR", f"{type(exc).__name__}: {exc}"])

    _print_table("实验 6：在线ALNS迭代 vs 仿真成本", ["online_iter", "avg_actual_cost", "time(s)/error"], rows)

    assert 0 in results, "iterations=0 baseline run failed"
    baseline = results[0]

    for online_iterations in [5, 10, 20]:
        assert online_iterations in results, f"iterations={online_iterations} run failed"
        assert results[online_iterations] <= baseline + 1e-9


