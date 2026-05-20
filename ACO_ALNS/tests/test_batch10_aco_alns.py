from __future__ import annotations

import random
import sys

from spd.core import all_hard_constraints_satisfied, check_sortie_structure
from spd.core import compute_objective
from spd.config import ProblemParameters
from spd.types import ProblemInstance, Solution
from spd.aco_alns import ACOALNSSolver, ACOState
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


def _build_operator_set() -> ALNSOperatorSet:
    """Build the default ALNS operator set used by integration tests."""
    return ALNSOperatorSet(
        destroy_operators=[d1_random_removal, d2_worst_removal, d3_related_removal, d4_low_home_probability_removal],
        repair_operators=[r1_greedy_insertion, r2_regret_insertion, r3_timeslot_aware_insertion],
        cross_group_operators=[cross_pair_swap, cross_pair_transfer],
    )


def _build_solver() -> ACOALNSSolver:
    """Create solver with real ETPRC builder and ALNS optimizer."""
    etprc_builder = ETPRCBuilder()
    alns_optimizer = ALNSOptimizer(_build_operator_set())
    return ACOALNSSolver(etprc_builder=etprc_builder, alns_optimizer=alns_optimizer)


def _build_pheromone_matrix(instance: ProblemInstance, initial_value: float) -> dict[tuple[int, int], float]:
    """Initialize pheromone for every directed truck arc."""
    return {(i, j): initial_value for i, j in instance.truck_distance_km}


def _truck_edges(solution: Solution) -> set[tuple[int, int]]:
    """Extract all truck route arcs from a solution."""
    edges: set[tuple[int, int]] = set()
    for pair in solution.vehicle_pairs:
        for idx in range(len(pair.truck_route) - 1):
            edges.add((pair.truck_route[idx], pair.truck_route[idx + 1]))
    return edges


def _run_optimize_with_weight_snapshot(
    optimizer: ALNSOptimizer,
    initial_solution: Solution,
    sample_instance: ProblemInstance,
    all_home_status: dict[int, bool],
    small_aco_params: ProblemParameters,
    rng: random.Random,
) -> tuple[Solution, tuple[float, ...], tuple[float, ...]]:
    """Run optimize and capture final destroy/repair weight vectors via tracing."""
    target_code = ALNSOptimizer.optimize.__code__
    latest_weights: tuple[tuple[float, ...], tuple[float, ...]] | None = None

    def tracer(frame, event, arg):
        nonlocal latest_weights
        _ = event, arg

        # Only inspect locals inside ALNSOptimizer.optimize.
        if frame.f_code is not target_code:
            return tracer

        local_vars = frame.f_locals
        if "destroy_weights" in local_vars and "repair_weights" in local_vars:
            destroy_weights = tuple(float(weight) for weight in local_vars["destroy_weights"])
            repair_weights = tuple(float(weight) for weight in local_vars["repair_weights"])
            latest_weights = (destroy_weights, repair_weights)

        return tracer

    previous_trace = sys.gettrace()
    sys.settrace(tracer)
    try:
        result = optimizer.optimize(
            initial_solution=initial_solution,
            instance=sample_instance,
            home_status=all_home_status,
            params=small_aco_params,
            rng=rng,
        )
    finally:
        sys.settrace(previous_trace)

    assert latest_weights is not None
    return result, latest_weights[0], latest_weights[1]


def _all_equal(weights: tuple[float, ...], eps: float = 1e-9) -> bool:
    """Check whether all values in a weight vector are effectively equal."""
    if len(weights) <= 1:
        return True
    first = weights[0]
    return all(abs(weight - first) <= eps for weight in weights[1:])


def test_alns_optimize_improves_or_maintains(sample_instance, small_aco_params, all_home_status) -> None:
    builder = ETPRCBuilder()
    initial_solution = builder.build_initial_solution(sample_instance, small_aco_params, random.Random(10))

    optimizer = ALNSOptimizer(_build_operator_set())
    optimized_solution = optimizer.optimize(
        initial_solution=initial_solution,
        instance=sample_instance,
        home_status=all_home_status,
        params=small_aco_params,
        rng=random.Random(11),
    )

    initial_obj = compute_objective(sample_instance, initial_solution, all_home_status, small_aco_params)
    optimized_obj = compute_objective(sample_instance, optimized_solution, all_home_status, small_aco_params)

    # Returned solution should stay complete and non-deteriorating.
    assert optimized_solution.all_customers == set(sample_instance.customers)
    assert optimized_solution.unserved_customers == set()
    assert optimized_obj <= initial_obj + 1e-9


def test_alns_operator_weights_update(sample_instance, small_aco_params, all_home_status) -> None:
    builder = ETPRCBuilder()
    initial_solution = builder.build_initial_solution(sample_instance, small_aco_params, random.Random(20))

    optimizer = ALNSOptimizer(_build_operator_set())
    _, destroy_weights, repair_weights = _run_optimize_with_weight_snapshot(
        optimizer=optimizer,
        initial_solution=initial_solution,
        sample_instance=sample_instance,
        all_home_status=all_home_status,
        small_aco_params=small_aco_params,
        rng=random.Random(21),
    )

    init_weight = small_aco_params.alns.operator_weight_init
    all_weights = destroy_weights + repair_weights

    # Adaptive update should move at least one weight and break full equality.
    assert any(abs(weight - init_weight) > 1e-9 for weight in all_weights)
    assert (not _all_equal(destroy_weights)) or (not _all_equal(repair_weights))


def test_construct_ant_solution_valid(sample_instance, small_aco_params, all_home_status) -> None:
    solver = _build_solver()
    state = ACOState(
        pheromone=_build_pheromone_matrix(sample_instance, small_aco_params.aco.pheromone_init),
        current_beta=small_aco_params.beta_schedule.beta_init,
    )

    ant_solution = solver.construct_ant_solution(
        instance=sample_instance,
        params=small_aco_params,
        home_status=all_home_status,
        state=state,
        rng=random.Random(31),
    )

    assert ant_solution.all_customers == set(sample_instance.customers)
    assert ant_solution.unserved_customers == set()

    for pair in ant_solution.vehicle_pairs:
        for sortie in pair.sorties:
            assert check_sortie_structure(pair, sortie) is True


def test_update_pheromone(sample_instance, small_aco_params, all_home_status) -> None:
    solver = _build_solver()

    state = ACOState(
        pheromone=_build_pheromone_matrix(sample_instance, initial_value=2.0),
        current_beta=small_aco_params.beta_schedule.beta_init,
    )

    best_solution = ETPRCBuilder().build_initial_solution(sample_instance, small_aco_params, random.Random(40))
    best_objective = compute_objective(sample_instance, best_solution, all_home_status, small_aco_params)

    before = dict(state.pheromone)
    best_edges = _truck_edges(best_solution)
    edge_on_best_path = next(iter(best_edges))
    edge_off_best_path = next(edge for edge in before if edge not in best_edges)

    solver.update_pheromone(
        state=state,
        best_solution=best_solution,
        best_objective=best_objective,
        params=small_aco_params,
    )

    evaporated_on_best = before[edge_on_best_path] * (1.0 - small_aco_params.aco.evaporation_rate)

    # Non-best arcs should evaporate; best-path arc should receive additional deposit.
    assert state.pheromone[edge_off_best_path] < before[edge_off_best_path]
    assert state.pheromone[edge_on_best_path] > evaporated_on_best


def test_advance_beta_schedule(sample_instance, small_aco_params) -> None:
    solver = _build_solver()
    state = ACOState(
        pheromone=_build_pheromone_matrix(sample_instance, small_aco_params.aco.pheromone_init),
        current_beta=small_aco_params.beta_schedule.beta_init,
    )

    first_beta = solver.advance_beta_schedule(state, small_aco_params)

    history = [first_beta]
    for _ in range(32):
        history.append(solver.advance_beta_schedule(state, small_aco_params))

    assert first_beta > small_aco_params.beta_schedule.beta_init
    assert all(history[idx] <= history[idx + 1] + 1e-9 for idx in range(len(history) - 1))
    assert history[-1] <= small_aco_params.beta_schedule.beta_max + 1e-9


class _RecordingACOALNSSolver(ACOALNSSolver):
    """Test helper to observe beta progression during solve()."""

    def __init__(self, etprc_builder: ETPRCBuilder, alns_optimizer: ALNSOptimizer):
        super().__init__(etprc_builder=etprc_builder, alns_optimizer=alns_optimizer)
        self.beta_history: list[float] = []

    def advance_beta_schedule(self, state: ACOState, params: ProblemParameters) -> float:
        beta = super().advance_beta_schedule(state, params)
        self.beta_history.append(beta)
        return beta


def test_aco_alns_full_solve(sample_instance, small_aco_params, all_home_status) -> None:
    etprc_builder = ETPRCBuilder()
    initial_solution = etprc_builder.build_initial_solution(sample_instance, small_aco_params, random.Random(50))
    initial_obj = compute_objective(sample_instance, initial_solution, all_home_status, small_aco_params)

    solver = _RecordingACOALNSSolver(
        etprc_builder=etprc_builder,
        alns_optimizer=ALNSOptimizer(_build_operator_set()),
    )

    best_solution = solver.solve(
        instance=sample_instance,
        params=small_aco_params,
        home_status=all_home_status,
        rng=random.Random(51),
    )

    best_obj = compute_objective(sample_instance, best_solution, all_home_status, small_aco_params)

    assert all_hard_constraints_satisfied(sample_instance, best_solution, all_home_status, small_aco_params) is True
    assert best_obj <= initial_obj + 1e-9

    assert solver.beta_history
    assert solver.beta_history[0] > small_aco_params.beta_schedule.beta_init
    assert solver.beta_history[-1] <= small_aco_params.beta_schedule.beta_max + 1e-9




