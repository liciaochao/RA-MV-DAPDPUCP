from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import random

import pytest

from spd.config import ProblemParameters
from spd.core import (
    all_hard_constraints_satisfied,
    check_sortie_feasibility,
    check_sortie_structure,
    is_truck_route_feasible,
)
from spd.core import check_sortie_load_feasibility, compute_truck_load
from spd.core import failure_penalty
from spd.core import compute_truck_timeline
from spd.types import ProblemInstance, Solution, Sortie, VehiclePairSolution
from spd.types import ServiceMode
from spd.aco_alns import ALNSOperatorSet, ALNSOptimizer
from spd.etprc import ETPRCBuilder
from spd.aco_alns import d1_random_removal, d2_worst_removal, d3_related_removal, d4_low_home_probability_removal
from spd.aco_alns import r1_greedy_insertion, r2_regret_insertion, r3_timeslot_aware_insertion
from spd.types import CustomerNotHomeEvent
from spd.online import LightweightALNSOptimizer
from spd.online import apply_b1_skip_infeasible_customers, apply_b2_truncate_sortie, repair_sortie
from spd.online import OnlineReplanner
from spd.simulation import ExecutionEngine
from spd.simulation import MonteCarloSimulator


def _clone_instance_with_latest(base: ProblemInstance, latest_updates: dict[int, float]) -> ProblemInstance:
    customers = dict(base.customers)
    for customer_id, latest in latest_updates.items():
        customer = customers[customer_id]
        customers[customer_id] = replace(customer, time_window=(customer.time_window[0], latest))
    return ProblemInstance(
        depot_id=base.depot_id,
        vehicle_pair_count=base.vehicle_pair_count,
        customers=customers,
        delivery_customers=base.delivery_customers,
        pickup_customers=base.pickup_customers,
        truck_distance_km=base.truck_distance_km,
        drone_distance_km=base.drone_distance_km,
    )


def _clone_instance_all_home_probability(base: ProblemInstance, prob: float) -> ProblemInstance:
    customers = dict(base.customers)
    for customer_id, customer in customers.items():
        customers[customer_id] = replace(customer, home_probabilities=(prob,) * len(customer.home_probabilities))
    return ProblemInstance(
        depot_id=base.depot_id,
        vehicle_pair_count=base.vehicle_pair_count,
        customers=customers,
        delivery_customers=base.delivery_customers,
        pickup_customers=base.pickup_customers,
        truck_distance_km=base.truck_distance_km,
        drone_distance_km=base.drone_distance_km,
    )


def _single_sortie_pair(customers: list[int]) -> VehiclePairSolution:
    return VehiclePairSolution(
        pair_id=1,
        truck_route=[0, 1, 3, 5, 0],
        sorties=[Sortie(launch_node=1, recovery_node=5, customers=list(customers))],
        truck_customers={1, 3, 5},
        drone_customers=set(customers),
    )


class _NoOpOnlineALNS:
    """Deterministic no-op online ALNS used by inference edge tests."""

    def optimize_remaining_path(
        self,
        instance: ProblemInstance,
        pair_solution: VehiclePairSolution,
        home_status: dict[int, bool],
        params: ProblemParameters,
        rng: random.Random | None = None,
    ) -> VehiclePairSolution:
        _ = instance, home_status, params, rng
        return deepcopy(pair_solution)


def _build_replanner(use_noop: bool = False) -> OnlineReplanner:
    optimizer = _NoOpOnlineALNS() if use_noop else LightweightALNSOptimizer()
    return OnlineReplanner(optimizer)


def _build_execution_engine() -> ExecutionEngine:
    return ExecutionEngine(_build_replanner())


def _pair_snapshot(pair_solution: VehiclePairSolution) -> tuple[tuple[int, ...], frozenset[int], frozenset[int], tuple[tuple[int, int, tuple[int, ...]], ...]]:
    sorties = tuple((s.launch_node, s.recovery_node, tuple(s.customers)) for s in pair_solution.sorties)
    return (
        tuple(pair_solution.truck_route),
        frozenset(pair_solution.truck_customers),
        frozenset(pair_solution.drone_customers),
        sorties,
    )


def test_a1_consecutive_drone_not_home_events_inference(sample_instance, sample_params, all_home_status) -> None:
    replanner = _build_replanner(use_noop=True)
    solution = Solution(vehicle_pairs=[_single_sortie_pair([2, 4, 6])], unserved_customers=set())
    home_status = dict(all_home_status)

    # First event: customer A=2 not at home.
    event_a = CustomerNotHomeEvent(
        pair_id=1,
        customer_id=2,
        service_mode=ServiceMode.DRONE,
        event_time=10.0,
        sortie_index=0,
    )
    after_first = replanner.handle_event(
        sample_instance,
        solution,
        event_a,
        home_status,
        sample_params,
        random.Random(100),
    )

    # Tighten B=4 latest time before second repair so wrong from_customer inference can remove B.
    tight_instance = _clone_instance_with_latest(sample_instance, {4: 0.0})

    # Second event: customer B=4 not at home.
    event_b = CustomerNotHomeEvent(
        pair_id=1,
        customer_id=4,
        service_mode=ServiceMode.DRONE,
        event_time=20.0,
        sortie_index=0,
    )
    after_second = replanner.handle_event(
        tight_instance,
        after_first,
        event_b,
        home_status,
        sample_params,
        random.Random(101),
    )

    repaired_customers = after_second.vehicle_pairs[0].sorties[0].customers if after_second.vehicle_pairs[0].sorties else []

    assert 4 in repaired_customers





def test_a1_three_consecutive_drone_not_home_events(sample_instance, sample_params, all_home_status) -> None:
    replanner = _build_replanner(use_noop=True)
    solution = Solution(vehicle_pairs=[_single_sortie_pair([2, 4, 6])], unserved_customers=set())
    home_status = dict(all_home_status)

    # Event 1 on customer 2.
    after_first = replanner.handle_event(
        sample_instance,
        solution,
        CustomerNotHomeEvent(pair_id=1, customer_id=2, service_mode=ServiceMode.DRONE, event_time=10.0, sortie_index=0),
        home_status,
        sample_params,
        random.Random(102),
    )
    customers_after_first = after_first.vehicle_pairs[0].sorties[0].customers if after_first.vehicle_pairs[0].sorties else []
    assert 2 in customers_after_first

    # Event 2 on customer 4 with tight latest bound that would remove 4 under wrong inference.
    tight_second = _clone_instance_with_latest(sample_instance, {4: 0.0})
    after_second = replanner.handle_event(
        tight_second,
        after_first,
        CustomerNotHomeEvent(pair_id=1, customer_id=4, service_mode=ServiceMode.DRONE, event_time=20.0, sortie_index=0),
        home_status,
        sample_params,
        random.Random(103),
    )
    customers_after_second = after_second.vehicle_pairs[0].sorties[0].customers if after_second.vehicle_pairs[0].sorties else []
    assert 4 in customers_after_second

    # Event 3 on customer 6 with tight latest bound that would remove 6 under wrong inference.
    tight_third = _clone_instance_with_latest(sample_instance, {6: 0.0})
    after_third = replanner.handle_event(
        tight_third,
        after_second,
        CustomerNotHomeEvent(pair_id=1, customer_id=6, service_mode=ServiceMode.DRONE, event_time=30.0, sortie_index=0),
        home_status,
        sample_params,
        random.Random(104),
    )
    customers_after_third = after_third.vehicle_pairs[0].sorties[0].customers if after_third.vehicle_pairs[0].sorties else []
    assert 6 in customers_after_third

def test_a2_repair_sortie_all_customers_not_home(sample_instance, sample_params, all_home_status) -> None:
    pair_solution = _single_sortie_pair([2, 4])
    sortie = pair_solution.sorties[0]
    home_status = dict(all_home_status)
    home_status[2] = False
    home_status[4] = False

    timeline = compute_truck_timeline(sample_instance, pair_solution, home_status, sample_params)
    repaired = repair_sortie(sample_instance, sortie, pair_solution, home_status, sample_params, timeline)

    assert check_sortie_structure(pair_solution, repaired) is True


def test_b3_empty_sortie_for_b1_b2_and_repair(sample_instance, sample_params, all_home_status) -> None:
    empty_sortie = Sortie(launch_node=1, recovery_node=5, customers=[])
    pair_solution = VehiclePairSolution(
        pair_id=1,
        truck_route=[0, 1, 3, 5, 0],
        sorties=[empty_sortie],
        truck_customers={1, 3, 5},
        drone_customers=set(),
    )

    timeline = compute_truck_timeline(sample_instance, pair_solution, all_home_status, sample_params)

    out_b1 = apply_b1_skip_infeasible_customers(sample_instance, empty_sortie, pair_solution, all_home_status, sample_params, timeline)
    out_b2 = apply_b2_truncate_sortie(sample_instance, empty_sortie, pair_solution, all_home_status, sample_params, timeline)
    out_repair = repair_sortie(sample_instance, empty_sortie, pair_solution, all_home_status, sample_params, timeline)

    assert out_b1.customers == []
    assert out_b2.customers == []
    assert out_repair.customers == []


def test_b4_single_customer_sortie_not_home(sample_instance, sample_params, all_home_status) -> None:
    pair_solution = _single_sortie_pair([2])
    sortie = pair_solution.sorties[0]
    home_status = dict(all_home_status)
    home_status[2] = False

    timeline = compute_truck_timeline(sample_instance, pair_solution, home_status, sample_params)
    repaired = repair_sortie(sample_instance, sortie, pair_solution, home_status, sample_params, timeline)

    assert repaired.customers == [2]
    assert check_sortie_structure(pair_solution, repaired) is True


def test_b5_execution_engine_with_empty_route_solution(sample_instance, sample_params, all_home_status) -> None:
    engine = _build_execution_engine()
    empty_pair = VehiclePairSolution(
        pair_id=1,
        truck_route=[0, 0],
        sorties=[],
        truck_customers=set(),
        drone_customers=set(),
    )
    solution = Solution(vehicle_pairs=[empty_pair], unserved_customers=set())

    result = engine.run(
        sample_instance,
        solution,
        sample_params,
        rng=random.Random(1),
        preset_home_status=dict(all_home_status),
    )

    assert result.actual_cost >= 0.0


def test_b6_single_truck_customer_handle_event(sample_instance, sample_params, all_home_status) -> None:
    replanner = _build_replanner()
    pair_solution = VehiclePairSolution(
        pair_id=1,
        truck_route=[0, 1, 0],
        sorties=[],
        truck_customers={1},
        drone_customers=set(),
    )
    solution = Solution(vehicle_pairs=[pair_solution], unserved_customers=set())

    home_status = dict(all_home_status)
    event = CustomerNotHomeEvent(
        pair_id=1,
        customer_id=1,
        service_mode=ServiceMode.TRUCK,
        event_time=5.0,
        sortie_index=None,
    )

    updated = replanner.handle_event(sample_instance, solution, event, home_status, sample_params, random.Random(2))

    assert home_status[1] is False
    assert updated.vehicle_pairs[0].pair_id == 1


def test_c7_drone_load_equal_capacity_boundary(sample_instance, sample_params, all_home_status) -> None:
    sortie = Sortie(launch_node=1, recovery_node=5, customers=[2])

    params_equal = replace(sample_params, vehicle=replace(sample_params.vehicle, drone_capacity=0.8))
    params_over = replace(sample_params, vehicle=replace(sample_params.vehicle, drone_capacity=0.79))

    assert check_sortie_load_feasibility(sample_instance, sortie, all_home_status, params_equal) is True
    assert check_sortie_load_feasibility(sample_instance, sortie, all_home_status, params_over) is False


def test_c8_sortie_energy_equal_battery_boundary(sample_instance, sample_params, sample_pair_solution, all_home_status) -> None:
    pair_solution = deepcopy(sample_pair_solution)
    sortie = pair_solution.sorties[0]

    timeline = compute_truck_timeline(sample_instance, pair_solution, all_home_status, sample_params)
    load_state = compute_truck_load(sample_instance, pair_solution, all_home_status, sample_params)

    from spd.core import compute_sortie_energy

    energy = compute_sortie_energy(sample_instance, sortie, timeline, all_home_status, sample_params)
    params_equal = replace(sample_params, energy=replace(sample_params.energy, drone_battery_capacity=energy))
    params_low = replace(sample_params, energy=replace(sample_params.energy, drone_battery_capacity=energy - 0.01))

    assert check_sortie_feasibility(sample_instance, pair_solution, sortie, timeline, load_state, all_home_status, params_equal) is True
    assert check_sortie_feasibility(sample_instance, pair_solution, sortie, timeline, load_state, all_home_status, params_low) is False


def test_c9_truck_wait_equal_wmax_boundary(sample_instance, sample_params, sample_pair_solution, all_home_status) -> None:
    pair_solution = deepcopy(sample_pair_solution)
    sortie = pair_solution.sorties[0]

    timeline = compute_truck_timeline(sample_instance, pair_solution, all_home_status, sample_params)
    load_state = compute_truck_load(sample_instance, pair_solution, all_home_status, sample_params)

    recovery = sortie.recovery_node
    wait_target = 10.0
    timeline.drone_arrival[recovery] = timeline.truck_arrival[recovery] + wait_target

    params_equal = replace(sample_params, constraints=replace(sample_params.constraints, max_truck_wait_time=wait_target))
    params_low = replace(sample_params, constraints=replace(sample_params.constraints, max_truck_wait_time=wait_target - 0.01))

    assert check_sortie_feasibility(sample_instance, pair_solution, sortie, timeline, load_state, all_home_status, params_equal) is True
    assert check_sortie_feasibility(sample_instance, pair_solution, sortie, timeline, load_state, all_home_status, params_low) is False


def test_d10_drone_time_window_equal_and_plus_point01(sample_instance, sample_params, sample_pair_solution, all_home_status) -> None:
    pair_solution = deepcopy(sample_pair_solution)
    sortie = pair_solution.sorties[0]
    target = sortie.customers[0]

    timeline = compute_truck_timeline(sample_instance, pair_solution, all_home_status, sample_params)
    load_state = compute_truck_load(sample_instance, pair_solution, all_home_status, sample_params)
    arrival = timeline.drone_arrival[target]

    instance_equal = _clone_instance_with_latest(sample_instance, {target: arrival})
    instance_late = _clone_instance_with_latest(sample_instance, {target: arrival - 0.01})

    assert check_sortie_feasibility(instance_equal, pair_solution, sortie, timeline, load_state, all_home_status, sample_params) is True
    assert check_sortie_feasibility(instance_late, pair_solution, sortie, timeline, load_state, all_home_status, sample_params) is False


def test_d11_depot_return_equal_and_plus_point01(sample_instance, sample_params, sample_pair_solution, all_home_status) -> None:
    pair_solution = deepcopy(sample_pair_solution)
    timeline = compute_truck_timeline(sample_instance, pair_solution, all_home_status, sample_params)
    load_state = compute_truck_load(sample_instance, pair_solution, all_home_status, sample_params)

    return_time = timeline.truck_arrival[pair_solution.truck_route[-1]]
    params_equal = replace(sample_params, time=replace(sample_params.time, depot_latest=return_time))
    params_low = replace(sample_params, time=replace(sample_params.time, depot_latest=return_time - 0.01))

    assert is_truck_route_feasible(sample_instance, pair_solution, timeline, load_state, params_equal) is True
    assert is_truck_route_feasible(sample_instance, pair_solution, timeline, load_state, params_low) is False


def test_e12_monte_carlo_single_trial_std_zero(sample_instance, sample_params, sample_pair_solution) -> None:
    simulator = MonteCarloSimulator(_build_execution_engine())
    solution = Solution(vehicle_pairs=[deepcopy(sample_pair_solution)], unserved_customers=set())

    result = simulator.run(sample_instance, solution, sample_params, trial_count=1, rng=random.Random(10))

    assert result.trial_count == 1
    assert result.cost_std == pytest.approx(0.0)


def test_e13_monte_carlo_fifty_trials_statistics(sample_instance, sample_params, sample_pair_solution) -> None:
    simulator = MonteCarloSimulator(_build_execution_engine())
    solution = Solution(vehicle_pairs=[deepcopy(sample_pair_solution)], unserved_customers=set())

    result = simulator.run(sample_instance, solution, sample_params, trial_count=50, rng=random.Random(11))

    assert result.cost_std >= 0.0
    assert result.worst_actual_cost >= result.average_actual_cost
    assert min(result.samples) <= result.average_actual_cost <= max(result.samples)

def test_e14_all_customers_prob_one(sample_instance, sample_params, sample_pair_solution) -> None:
    simulator = MonteCarloSimulator(_build_execution_engine())
    solution = Solution(vehicle_pairs=[deepcopy(sample_pair_solution)], unserved_customers=set())
    instance_all_home = _clone_instance_all_home_probability(sample_instance, 1.0)

    result = simulator.run(instance_all_home, solution, sample_params, trial_count=10, rng=random.Random(12))

    assert result.average_failed_customers == pytest.approx(0.0)
    assert result.cost_std <= 1e-9


def test_e15_all_customers_prob_zero(sample_instance, sample_params, sample_pair_solution) -> None:
    simulator = MonteCarloSimulator(_build_execution_engine())
    solution = Solution(vehicle_pairs=[deepcopy(sample_pair_solution)], unserved_customers=set())
    instance_none_home = _clone_instance_all_home_probability(sample_instance, 0.0)

    result = simulator.run(instance_none_home, solution, sample_params, trial_count=10, rng=random.Random(13))
    expected_failed = float(len(instance_none_home.customers))
    expected_fail_penalty = sum(
        failure_penalty(instance_none_home, customer_id, sample_params)
        for customer_id in instance_none_home.customers
    )

    assert result.average_failed_customers == pytest.approx(expected_failed)
    assert all(sample >= expected_fail_penalty - 1e-9 for sample in result.samples)


def test_f16_destroy_remove_count_larger_than_customer_count(sample_instance, sample_params, all_home_status) -> None:
    solution = ETPRCBuilder().build_initial_solution(sample_instance, sample_params, random.Random(20))

    result_d1 = d1_random_removal(
        solution,
        sample_instance,
        sample_params,
        remove_count=100,
        home_status=all_home_status,
        rng=random.Random(21),
    )
    result_d2 = d2_worst_removal(
        solution,
        sample_instance,
        sample_params,
        remove_count=100,
        home_status=all_home_status,
        rng=random.Random(22),
    )

    total = len(solution.all_customers)
    assert len(result_d1.removed_customers) == total
    assert len(result_d2.removed_customers) == total


def test_f17_alns_optimize_single_customer_solution(sample_instance, sample_params, all_home_status) -> None:
    operators = ALNSOperatorSet(
        destroy_operators=[d1_random_removal, d2_worst_removal, d3_related_removal, d4_low_home_probability_removal],
        repair_operators=[r1_greedy_insertion, r2_regret_insertion, r3_timeslot_aware_insertion],
        cross_group_operators=[],
    )
    optimizer = ALNSOptimizer(operators)

    single_pair = VehiclePairSolution(
        pair_id=1,
        truck_route=[0, 1, 0],
        sorties=[],
        truck_customers={1},
        drone_customers=set(),
    )
    initial_solution = Solution(vehicle_pairs=[single_pair], unserved_customers=set())

    optimized = optimizer.optimize(
        initial_solution=initial_solution,
        instance=sample_instance,
        home_status=all_home_status,
        params=sample_params,
        rng=random.Random(23),
    )

    assert optimized.all_customers == {1}
    assert all_hard_constraints_satisfied(sample_instance, optimized, all_home_status, sample_params) is True


def test_f18_lightweight_alns_zero_iterations_returns_original_pair(
    sample_instance,
    sample_params,
    sample_pair_solution,
    all_home_status,
) -> None:
    optimizer = LightweightALNSOptimizer()
    params_zero = replace(sample_params, online_alns=replace(sample_params.online_alns, iterations=0))
    original_pair = deepcopy(sample_pair_solution)
    before = _pair_snapshot(original_pair)

    result_pair = optimizer.optimize_remaining_path(
        sample_instance,
        original_pair,
        all_home_status,
        params_zero,
        random.Random(24),
    )

    assert _pair_snapshot(result_pair) == before








