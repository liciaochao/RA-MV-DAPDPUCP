from __future__ import annotations

from dataclasses import replace
import random
import time

import pytest

from spd.config import ProblemParameters
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
    """Build the standard ALNS operator set for benchmark runs."""
    return ALNSOperatorSet(
        destroy_operators=[d1_random_removal, d2_worst_removal, d3_related_removal, d4_low_home_probability_removal],
        repair_operators=[r1_greedy_insertion, r2_regret_insertion, r3_timeslot_aware_insertion],
        cross_group_operators=[cross_pair_swap, cross_pair_transfer],
    )


def _build_solver() -> ACOALNSSolver:
    """Create ACO-ALNS solver used in benchmark tests."""
    return ACOALNSSolver(
        etprc_builder=ETPRCBuilder(),
        alns_optimizer=ALNSOptimizer(_build_operator_set()),
    )


def _benchmark_params(sample_params: ProblemParameters) -> ProblemParameters:
    """Explicit benchmark parameters for ACO/ALNS timing comparisons."""
    return replace(
        sample_params,
        vehicle=replace(sample_params.vehicle, drone_capacity=0.1),
        aco=replace(sample_params.aco, max_iterations=5, ant_count=3),
        alns=replace(sample_params.alns, iterations=5),
    )


@pytest.mark.parametrize(
    ("instance_fixture", "label"),
    [
        ("medium_instance", "20"),
        ("large_instance", "50"),
    ],
)
def test_benchmark_etprc_build_time(request, sample_params, instance_fixture: str, label: str) -> None:
    instance = request.getfixturevalue(instance_fixture)
    builder = ETPRCBuilder()
    params = _benchmark_params(sample_params)

    start = time.perf_counter()
    _ = builder.build_initial_solution(instance, params, random.Random(42))
    elapsed = time.perf_counter() - start

    print(f"E-TPRC {label}客户: {elapsed:.2f}s")
    assert elapsed < 60.0, f"E-TPRC on {label} customers took {elapsed:.2f}s (>60s), module needs optimization"


def test_benchmark_aco_alns_solve_time(medium_instance, sample_params) -> None:
    params = _benchmark_params(sample_params)
    all_home = {customer_id: True for customer_id in medium_instance.customers}

    # Build an initial solution first (to mirror real pipeline setup cost separately).
    _ = ETPRCBuilder().build_initial_solution(medium_instance, params, random.Random(43))

    solver = _build_solver()
    start = time.perf_counter()
    _ = solver.solve(medium_instance, params, all_home, random.Random(44))
    elapsed = time.perf_counter() - start

    print(
        f"ACO-ALNS 20客户: {elapsed:.2f}s "
        f"(max_iterations={params.aco.max_iterations}, ant_count={params.aco.ant_count}, alns.iterations={params.alns.iterations})"
    )
    assert elapsed < 120.0, f"ACO-ALNS solve took {elapsed:.2f}s (>120s), module needs optimization"


def test_benchmark_single_execution_run_time(medium_instance, sample_params) -> None:
    params = _benchmark_params(sample_params)
    all_home = {customer_id: True for customer_id in medium_instance.customers}

    solver = _build_solver()
    optimized_solution = solver.solve(medium_instance, params, all_home, random.Random(45))

    engine = ExecutionEngine(OnlineReplanner(LightweightALNSOptimizer()))
    start = time.perf_counter()
    _ = engine.run(medium_instance, optimized_solution, params, rng=random.Random(46))
    elapsed = time.perf_counter() - start

    print(f"单次仿真 20客户: {elapsed:.2f}s")
    assert elapsed < 30.0, f"ExecutionEngine single run took {elapsed:.2f}s (>30s), module needs optimization"


def test_benchmark_monte_carlo_50_trials_time(medium_instance, sample_params) -> None:
    params = _benchmark_params(sample_params)
    all_home = {customer_id: True for customer_id in medium_instance.customers}

    solver = _build_solver()
    optimized_solution = solver.solve(medium_instance, params, all_home, random.Random(47))

    simulator = MonteCarloSimulator(ExecutionEngine(OnlineReplanner(LightweightALNSOptimizer())))
    start = time.perf_counter()
    _ = simulator.run(medium_instance, optimized_solution, params, trial_count=50, rng=random.Random(48))
    elapsed = time.perf_counter() - start

    print(f"MonteCarlo 50 trials 20客户: {elapsed:.2f}s, 平均: {elapsed/50:.2f}s/trial")
    assert elapsed < 300.0, f"MonteCarlo 50 trials took {elapsed:.2f}s (>300s), module needs optimization"
