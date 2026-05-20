from __future__ import annotations

from dataclasses import replace

import pytest

from spd.core import compute_sortie_energy
from spd.core import evaluate_solution
from spd.core import (
    compute_actual_cost,
    compute_expected_failure_cost,
    compute_objective,
    failure_penalty,
)
from spd.core import compute_truck_timeline
from spd.core import get_arrival_time
from spd.types import ExecutionResult, ProblemInstance, Solution, Sortie, VehiclePairSolution


def _clone_instance_customer(
    base: ProblemInstance,
    customer_id: int,
    *,
    latest_time: float | None = None,
    home_probabilities: tuple[float, ...] | None = None,
) -> ProblemInstance:
    customers = dict(base.customers)
    customer = customers[customer_id]
    customers[customer_id] = replace(
        customer,
        time_window=(customer.time_window[0], latest_time if latest_time is not None else customer.time_window[1]),
        home_probabilities=home_probabilities if home_probabilities is not None else customer.home_probabilities,
    )
    return ProblemInstance(
        depot_id=base.depot_id,
        vehicle_pair_count=base.vehicle_pair_count,
        customers=customers,
        delivery_customers=base.delivery_customers,
        pickup_customers=base.pickup_customers,
        truck_distance_km=base.truck_distance_km,
        drone_distance_km=base.drone_distance_km,
    )


def _clone_instance_distances(base: ProblemInstance, updates: dict[tuple[int, int], float]) -> ProblemInstance:
    truck_dist = dict(base.truck_distance_km)
    truck_dist.update(updates)
    return ProblemInstance(
        depot_id=base.depot_id,
        vehicle_pair_count=base.vehicle_pair_count,
        customers=base.customers,
        delivery_customers=base.delivery_customers,
        pickup_customers=base.pickup_customers,
        truck_distance_km=truck_dist,
        drone_distance_km=base.drone_distance_km,
    )


def test_compute_objective_fixed_cost_only(sample_instance, sample_params, all_home_status) -> None:
    pair = VehiclePairSolution(
        pair_id=1,
        truck_route=[0, 1, 0],
        sorties=[],
        truck_customers={1},
        drone_customers=set(),
    )
    solution = Solution(vehicle_pairs=[pair], unserved_customers=set())
    params = replace(
        sample_params,
        cost=replace(sample_params.cost, truck_cost_per_km=0.0, drone_energy_cost_per_wh=0.0),
        beta_schedule=replace(sample_params.beta_schedule, beta_init=0.0),
    )
    instance = _clone_instance_customer(sample_instance, 1, home_probabilities=(1.0,) * 8)

    objective = compute_objective(instance, solution, all_home_status, params)
    assert objective == pytest.approx(params.cost.fixed_pair_cost)


def test_compute_objective_truck_cost_only(sample_instance, sample_params, all_home_status) -> None:
    pair = VehiclePairSolution(
        pair_id=1,
        truck_route=[0, 1, 0],
        sorties=[],
        truck_customers={1},
        drone_customers=set(),
    )
    solution = Solution(vehicle_pairs=[pair], unserved_customers=set())
    params = replace(
        sample_params,
        cost=replace(sample_params.cost, fixed_pair_cost=0.0, drone_energy_cost_per_wh=0.0),
        beta_schedule=replace(sample_params.beta_schedule, beta_init=0.0),
    )
    instance = _clone_instance_customer(sample_instance, 1, home_probabilities=(1.0,) * 8)

    expected = params.cost.truck_cost_per_km * (
        instance.truck_distance_km[(0, 1)] + instance.truck_distance_km[(1, 0)]
    )
    objective = compute_objective(instance, solution, all_home_status, params)
    assert objective == pytest.approx(expected)


def test_compute_objective_drone_cost_only(sample_instance, sample_params, sample_pair_solution, all_home_status) -> None:
    solution = Solution(vehicle_pairs=[sample_pair_solution], unserved_customers=set())
    params = replace(
        sample_params,
        cost=replace(sample_params.cost, fixed_pair_cost=0.0, truck_cost_per_km=0.0, drone_energy_cost_per_wh=0.1),
        beta_schedule=replace(sample_params.beta_schedule, beta_init=0.0),
    )

    timeline = compute_truck_timeline(sample_instance, sample_pair_solution, all_home_status, params)
    sortie = sample_pair_solution.sorties[0]
    expected = params.cost.drone_energy_cost_per_wh * compute_sortie_energy(
        sample_instance,
        sortie,
        timeline,
        all_home_status,
        params,
    )

    objective = compute_objective(sample_instance, solution, all_home_status, params)
    assert objective == pytest.approx(expected)


def test_compute_objective_expected_failure_only(sample_instance, sample_params, all_home_status) -> None:
    pair = VehiclePairSolution(
        pair_id=1,
        truck_route=[0, 1, 3, 0],
        sorties=[Sortie(launch_node=1, recovery_node=3, customers=[2])],
        truck_customers={1, 3},
        drone_customers={2},
    )
    solution = Solution(vehicle_pairs=[pair], unserved_customers=set())

    # Keep failure_penalty positive for customer 2, but make route travel cost zero.
    instance = _clone_instance_distances(
        sample_instance,
        {
            (0, 1): 0.0,
            (1, 3): 0.0,
            (3, 0): 0.0,
        },
    )
    instance = _clone_instance_customer(instance, 1, home_probabilities=(1.0,) * 8)
    instance = _clone_instance_customer(instance, 3, home_probabilities=(1.0,) * 8)
    instance = _clone_instance_customer(instance, 2, home_probabilities=(0.0,) * 8)

    params = replace(
        sample_params,
        cost=replace(sample_params.cost, fixed_pair_cost=0.0, drone_energy_cost_per_wh=0.0),
        beta_schedule=replace(sample_params.beta_schedule, beta_init=0.0),
    )

    expected = compute_expected_failure_cost(instance, solution, params)
    objective = compute_objective(instance, solution, all_home_status, params)

    assert expected > 0.0
    assert objective == pytest.approx(expected)


def test_compute_objective_time_window_penalty_only(sample_instance, sample_params, all_home_status) -> None:
    pair = VehiclePairSolution(
        pair_id=1,
        truck_route=[0, 1, 0],
        sorties=[],
        truck_customers={1},
        drone_customers=set(),
    )
    solution = Solution(vehicle_pairs=[pair], unserved_customers=set())
    instance = _clone_instance_customer(sample_instance, 1, latest_time=1.0, home_probabilities=(1.0,) * 8)

    params = replace(
        sample_params,
        cost=replace(sample_params.cost, fixed_pair_cost=0.0, truck_cost_per_km=0.0, drone_energy_cost_per_wh=0.0),
        beta_schedule=replace(sample_params.beta_schedule, beta_init=10.0),
    )

    timeline = compute_truck_timeline(instance, pair, all_home_status, params)
    arrival_1 = get_arrival_time(1, pair, timeline)
    expected = params.beta_schedule.beta_init * max(0.0, arrival_1 - instance.customers[1].time_window[1])

    objective = compute_objective(instance, solution, all_home_status, params)
    assert objective == pytest.approx(expected)


def test_compute_objective_sum_of_components(sample_instance, sample_params, sample_pair_solution, all_home_status) -> None:
    solution = Solution(vehicle_pairs=[sample_pair_solution], unserved_customers=set())
    beta = 7.0

    timeline = compute_truck_timeline(sample_instance, sample_pair_solution, all_home_status, sample_params)

    z_fixed = sample_params.cost.fixed_pair_cost * len(solution.used_vehicle_pairs)
    z_truck = 0.0
    for idx in range(len(sample_pair_solution.truck_route) - 1):
        i = sample_pair_solution.truck_route[idx]
        j = sample_pair_solution.truck_route[idx + 1]
        z_truck += sample_params.cost.truck_cost_per_km * sample_instance.truck_distance_km[(i, j)]

    z_drone = 0.0
    for sortie in sample_pair_solution.sorties:
        z_drone += sample_params.cost.drone_energy_cost_per_wh * compute_sortie_energy(
            sample_instance,
            sortie,
            timeline,
            all_home_status,
            sample_params,
        )

    z_fail = compute_expected_failure_cost(sample_instance, solution, sample_params)

    z_tw = 0.0
    for customer_id in sample_pair_solution.all_customers:
        arrival = get_arrival_time(customer_id, sample_pair_solution, timeline)
        latest = sample_instance.customers[customer_id].time_window[1]
        z_tw += max(0.0, arrival - latest)
    z_tw *= beta

    expected_total = z_fixed + z_truck + z_drone + z_fail + z_tw
    objective = compute_objective(sample_instance, solution, all_home_status, sample_params, time_window_penalty=beta)

    assert objective == pytest.approx(expected_total)


def test_compute_actual_cost_matches_manual_sum(sample_params, sample_pair_solution) -> None:
    solution = Solution(vehicle_pairs=[sample_pair_solution], unserved_customers=set())

    z_fixed = sample_params.cost.fixed_pair_cost * len(solution.used_vehicle_pairs)
    z_truck = sample_params.cost.truck_cost_per_km * (10.0 + 5.0)
    z_drone = sample_params.cost.drone_energy_cost_per_wh * 20.0
    z_fail = 30.0
    actual_cost = z_fixed + z_truck + z_drone + z_fail

    execution = ExecutionResult(
        solution=solution,
        truck_distances_km=[10.0, 5.0],
        drone_energies_wh=[20.0],
        failed_customers={2, 4},
        actual_cost=actual_cost,
    )

    assert compute_actual_cost(execution, sample_params) == pytest.approx(actual_cost)


def test_evaluate_solution_hard_feasible_true(sample_instance, sample_params, sample_pair_solution, all_home_status) -> None:
    solution = Solution(vehicle_pairs=[sample_pair_solution], unserved_customers=set())
    evaluation = evaluate_solution(sample_instance, solution, all_home_status, sample_params)

    assert evaluation.hard_feasible is True
    assert len(evaluation.pair_evaluations) == 1
    assert evaluation.pair_evaluations[0].hard_feasible is True
    assert evaluation.objective_value > 0.0


def test_evaluate_solution_hard_feasible_false(sample_instance, sample_params, all_home_status) -> None:
    bad_pair = VehiclePairSolution(
        pair_id=1,
        truck_route=[0, 1, 3, 5, 0],
        sorties=[Sortie(launch_node=5, recovery_node=1, customers=[2, 4])],
        truck_customers={1, 3, 5},
        drone_customers={2, 4},
    )
    solution = Solution(vehicle_pairs=[bad_pair], unserved_customers=set())

    evaluation = evaluate_solution(sample_instance, solution, all_home_status, sample_params)

    assert evaluation.hard_feasible is False
    assert len(evaluation.pair_evaluations) == 1
    assert evaluation.pair_evaluations[0].hard_feasible is False
