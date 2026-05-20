from __future__ import annotations

from dataclasses import replace
import random

from spd.core import all_hard_constraints_satisfied
from spd.core import compute_objective
from spd.config import ProblemParameters
from spd.types import Solution
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
    """Build the standard offline ALNS operator set."""
    return ALNSOperatorSet(
        destroy_operators=[d1_random_removal, d2_worst_removal, d3_related_removal, d4_low_home_probability_removal],
        repair_operators=[r1_greedy_insertion, r2_regret_insertion, r3_timeslot_aware_insertion],
        cross_group_operators=[cross_pair_swap, cross_pair_transfer],
    )


def _build_solver() -> ACOALNSSolver:
    """Create ACO-ALNS solver with real ETPRC and ALNS components."""
    return ACOALNSSolver(
        etprc_builder=ETPRCBuilder(),
        alns_optimizer=ALNSOptimizer(_build_operator_set()),
    )


def _small_pipeline_params(sample_params: ProblemParameters) -> ProblemParameters:
    """Use small ACO/ALNS settings to keep E2E tests fast."""
    return replace(
        sample_params,
        aco=replace(sample_params.aco, max_iterations=2, ant_count=2),
        alns=replace(sample_params.alns, iterations=5),
    )


def _assert_solution_structure(solution: Solution, instance) -> None:
    """Shared structural assertions for offline optimized solutions."""
    assert solution.unserved_customers.issubset(set(instance.customers))
    assert solution.all_customers | solution.unserved_customers == set(instance.customers)
    for pair in solution.vehicle_pairs:
        assert pair.truck_route[0] == instance.depot_id
        assert pair.truck_route[-1] == instance.depot_id
        assert pair.truck_customers.isdisjoint(pair.drone_customers)
        assert pair.truck_customers | pair.drone_customers == pair.all_customers
        for sortie in pair.sorties:
            assert sortie.launch_node in pair.truck_customers
            assert sortie.recovery_node in pair.truck_customers


def test_offline_pipeline_end_to_end(sample_instance, sample_params, all_home_status) -> None:
    params = _small_pipeline_params(sample_params)

    builder = ETPRCBuilder()
    initial_solution = builder.build_initial_solution(sample_instance, params, random.Random(10))
    solver = _build_solver()
    optimized_solution = solver.solve(sample_instance, params, all_home_status, random.Random(11))

    initial_obj = compute_objective(sample_instance, initial_solution, all_home_status, params)
    optimized_obj = compute_objective(sample_instance, optimized_solution, all_home_status, params)

    assert all_hard_constraints_satisfied(sample_instance, optimized_solution, all_home_status, params) is True
    assert optimized_solution.unserved_customers.issubset(set(sample_instance.customers))
    assert optimized_solution.all_customers | optimized_solution.unserved_customers == set(sample_instance.customers)
    assert optimized_obj <= initial_obj + 1e-9
    _assert_solution_structure(optimized_solution, sample_instance)


def test_offline_online_simulation_pipeline(sample_instance, sample_params, all_home_status) -> None:
    params = _small_pipeline_params(sample_params)

    solver = _build_solver()
    offline_solution = solver.solve(sample_instance, params, all_home_status, random.Random(20))

    execution_engine = ExecutionEngine(OnlineReplanner(LightweightALNSOptimizer()))
    simulator = MonteCarloSimulator(execution_engine)

    mc_result = simulator.run(
        sample_instance,
        offline_solution,
        params,
        trial_count=20,
        rng=random.Random(21),
    )

    assert mc_result.trial_count == 20
    assert mc_result.average_actual_cost > 0.0
    assert mc_result.worst_actual_cost >= mc_result.average_actual_cost
    assert mc_result.cost_std >= 0.0
    assert mc_result.average_failed_customers >= 0.0
    assert all(sample > 0.0 for sample in mc_result.samples)


def test_pipeline_deterministic_reproducible(sample_instance, sample_params) -> None:
    params = _small_pipeline_params(sample_params)

    def run_once(seed: int) -> list[float]:
        rng = random.Random(seed)
        all_home = {customer_id: True for customer_id in sample_instance.customers}

        solver = _build_solver()
        solution = solver.solve(sample_instance, params, all_home, random.Random(rng.randint(0, 2**31 - 1)))

        simulator = MonteCarloSimulator(ExecutionEngine(OnlineReplanner(LightweightALNSOptimizer())))
        mc_result = simulator.run(
            sample_instance,
            solution,
            params,
            trial_count=10,
            rng=random.Random(rng.randint(0, 2**31 - 1)),
        )
        return mc_result.samples

    samples_1 = run_once(123456)
    samples_2 = run_once(123456)

    assert samples_1 == samples_2


def test_execution_customers_not_lost(sample_instance, sample_params, sample_pair_solution) -> None:
    execution_engine = ExecutionEngine(OnlineReplanner(LightweightALNSOptimizer()))
    solution = Solution(vehicle_pairs=[sample_pair_solution], unserved_customers=set())

    # 50% customers are marked not at home by deterministic parity rule.
    preset_home_status = {
        customer_id: (customer_id % 2 == 0)
        for customer_id in sample_instance.customers
    }

    result = execution_engine.run(
        sample_instance,
        solution,
        sample_params,
        rng=random.Random(30),
        preset_home_status=preset_home_status,
    )

    served_customers = {customer_id for customer_id, is_home in result.actual_is_home.items() if is_home}
    all_customers = set(sample_instance.customers)
    assert result.failed_customers | served_customers == all_customers


def test_multi_pair_large_instance_pipeline(large_instance, sample_params) -> None:
    params = _small_pipeline_params(sample_params)
    all_home_large = {customer_id: True for customer_id in large_instance.customers}

    builder = ETPRCBuilder()
    initial_solution = builder.build_initial_solution(large_instance, params, random.Random(40))

    solver = _build_solver()
    optimized_solution = solver.solve(large_instance, params, all_home_large, random.Random(41))

    # Full customer coverage in optimized solution.
    assert optimized_solution.unserved_customers.issubset(set(large_instance.customers))
    assert optimized_solution.all_customers | optimized_solution.unserved_customers == set(large_instance.customers)

    # Pair customer sets must not overlap.
    pair_customer_sets = [set(pair.all_customers) for pair in optimized_solution.vehicle_pairs]
    for idx in range(len(pair_customer_sets)):
        for jdx in range(idx + 1, len(pair_customer_sets)):
            assert pair_customer_sets[idx].isdisjoint(pair_customer_sets[jdx])

    simulator = MonteCarloSimulator(ExecutionEngine(OnlineReplanner(LightweightALNSOptimizer())))
    mc_result = simulator.run(
        large_instance,
        optimized_solution,
        params,
        trial_count=5,
        rng=random.Random(42),
    )

    assert mc_result.trial_count == 5
    assert mc_result.average_actual_cost > 0.0

    # Optimization should not degrade objective against initial solution.
    initial_obj = compute_objective(large_instance, initial_solution, all_home_large, params)
    optimized_obj = compute_objective(large_instance, optimized_solution, all_home_large, params)
    assert optimized_obj <= initial_obj + 1e-9


