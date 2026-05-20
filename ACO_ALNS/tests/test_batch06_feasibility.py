from __future__ import annotations

from dataclasses import replace

import pytest

from spd.core import (
    all_hard_constraints_satisfied,
    check_sortie_feasibility,
    check_sortie_structure,
    is_truck_route_feasible,
)
from spd.core import compute_truck_load
from spd.core import compute_truck_timeline
from spd.types import LoadState, ProblemInstance, Solution, Sortie, VehiclePairSolution


def _clone_instance_with_customer(
    base: ProblemInstance,
    customer_id: int,
    *,
    latest_time: float,
) -> ProblemInstance:
    customers = dict(base.customers)
    customer = customers[customer_id]
    customers[customer_id] = replace(customer, time_window=(customer.time_window[0], latest_time))
    return ProblemInstance(
        depot_id=base.depot_id,
        vehicle_pair_count=base.vehicle_pair_count,
        customers=customers,
        delivery_customers=base.delivery_customers,
        pickup_customers=base.pickup_customers,
        truck_distance_km=base.truck_distance_km,
        drone_distance_km=base.drone_distance_km,
    )


def _build_states(instance, pair_solution, home_status, params):
    timeline = compute_truck_timeline(instance, pair_solution, home_status, params)
    load_state = compute_truck_load(instance, pair_solution, home_status, params)
    return timeline, load_state


def test_check_sortie_structure_valid(sample_pair_solution) -> None:
    sortie = sample_pair_solution.sorties[0]
    assert check_sortie_structure(sample_pair_solution, sortie) is True


@pytest.mark.parametrize(
    "sortie,truck_customers,truck_route",
    [
        (Sortie(launch_node=0, recovery_node=5, customers=[2, 4]), {1, 3, 5}, [0, 1, 3, 5, 0]),
        (Sortie(launch_node=1, recovery_node=0, customers=[2, 4]), {1, 3, 5}, [0, 1, 3, 5, 0]),
        (Sortie(launch_node=1, recovery_node=5, customers=[2, 4]), {3, 5}, [0, 1, 3, 5, 0]),
        (Sortie(launch_node=1, recovery_node=5, customers=[3, 4]), {1, 3, 5}, [0, 1, 3, 5, 0]),
        (Sortie(launch_node=1, recovery_node=5, customers=[1, 4]), {1, 3, 5}, [0, 1, 3, 5, 0]),
        (Sortie(launch_node=5, recovery_node=1, customers=[2, 4]), {1, 3, 5}, [0, 1, 3, 5, 0]),
    ],
)
def test_check_sortie_structure_violations(sortie, truck_customers, truck_route) -> None:
    pair = VehiclePairSolution(
        pair_id=1,
        truck_route=truck_route,
        sorties=[sortie],
        truck_customers=truck_customers,
        drone_customers={2, 4},
    )
    assert check_sortie_structure(pair, sortie) is False


def test_check_sortie_feasibility_f1_battery_violation(
    sample_instance,
    sample_params,
    sample_pair_solution,
    all_home_status,
) -> None:
    params = replace(sample_params, energy=replace(sample_params.energy, drone_battery_capacity=1.0))
    timeline, load_state = _build_states(sample_instance, sample_pair_solution, all_home_status, params)
    sortie = sample_pair_solution.sorties[0]

    assert check_sortie_feasibility(
        sample_instance,
        sample_pair_solution,
        sortie,
        timeline,
        load_state,
        all_home_status,
        params,
    ) is False


def test_check_sortie_feasibility_f2_drone_load_violation(
    sample_instance,
    sample_params,
    sample_pair_solution,
    all_home_status,
) -> None:
    params = replace(sample_params, vehicle=replace(sample_params.vehicle, drone_capacity=0.5))
    timeline, load_state = _build_states(sample_instance, sample_pair_solution, all_home_status, params)
    sortie = sample_pair_solution.sorties[0]

    assert check_sortie_feasibility(
        sample_instance,
        sample_pair_solution,
        sortie,
        timeline,
        load_state,
        all_home_status,
        params,
    ) is False


def test_check_sortie_feasibility_f3_truck_wait_violation(
    sample_instance,
    sample_params,
    sample_pair_solution,
    all_home_status,
) -> None:
    params = replace(
        sample_params,
        vehicle=replace(sample_params.vehicle, truck_speed=2.0, drone_speed=0.2),
        constraints=replace(sample_params.constraints, max_truck_wait_time=0.1),
    )
    timeline, load_state = _build_states(sample_instance, sample_pair_solution, all_home_status, params)
    sortie = sample_pair_solution.sorties[0]

    assert check_sortie_feasibility(
        sample_instance,
        sample_pair_solution,
        sortie,
        timeline,
        load_state,
        all_home_status,
        params,
    ) is False


def test_check_sortie_feasibility_f4_drone_time_window_violation(
    sample_instance,
    sample_params,
    sample_pair_solution,
    all_home_status,
) -> None:
    # Customer 2 drone arrival is around 13, so latest=5 forces violation.
    instance = _clone_instance_with_customer(sample_instance, 2, latest_time=5.0)
    timeline, load_state = _build_states(instance, sample_pair_solution, all_home_status, sample_params)
    sortie = sample_pair_solution.sorties[0]

    assert check_sortie_feasibility(
        instance,
        sample_pair_solution,
        sortie,
        timeline,
        load_state,
        all_home_status,
        sample_params,
    ) is False


def test_check_sortie_feasibility_f5_truck_load_violation(
    sample_instance,
    sample_params,
    sample_pair_solution,
    all_home_status,
) -> None:
    timeline, load_state = _build_states(sample_instance, sample_pair_solution, all_home_status, sample_params)
    sortie = sample_pair_solution.sorties[0]

    bad_load_state = LoadState(
        truck_load_at_node=dict(load_state.truck_load_at_node),
        drone_load_at_node=dict(load_state.drone_load_at_node),
    )
    bad_load_state.truck_load_at_node[sortie.recovery_node] = sample_params.vehicle.truck_capacity + 1.0

    assert check_sortie_feasibility(
        sample_instance,
        sample_pair_solution,
        sortie,
        timeline,
        bad_load_state,
        all_home_status,
        sample_params,
    ) is False


def test_is_truck_route_feasible_f6_truck_time_window_violation(
    sample_instance,
    sample_params,
    sample_pair_solution,
    all_home_status,
) -> None:
    # Customer 1 truck arrival is around 2.4, so latest=1 forces violation.
    instance = _clone_instance_with_customer(sample_instance, 1, latest_time=1.0)
    timeline, load_state = _build_states(instance, sample_pair_solution, all_home_status, sample_params)

    assert is_truck_route_feasible(instance, sample_pair_solution, timeline, load_state, sample_params) is False


def test_is_truck_route_feasible_f7_truck_load_violation(
    sample_instance,
    sample_params,
    sample_pair_solution,
    all_home_status,
) -> None:
    timeline, load_state = _build_states(sample_instance, sample_pair_solution, all_home_status, sample_params)

    bad_load_state = LoadState(
        truck_load_at_node=dict(load_state.truck_load_at_node),
        drone_load_at_node=dict(load_state.drone_load_at_node),
    )
    bad_load_state.truck_load_at_node[1] = sample_params.vehicle.truck_capacity + 0.1

    assert is_truck_route_feasible(sample_instance, sample_pair_solution, timeline, bad_load_state, sample_params) is False


def test_is_truck_route_feasible_f8_depot_latest_violation(
    sample_instance,
    sample_params,
    sample_pair_solution,
    all_home_status,
) -> None:
    params = replace(sample_params, time=replace(sample_params.time, depot_latest=1.0))
    timeline, load_state = _build_states(sample_instance, sample_pair_solution, all_home_status, params)

    assert is_truck_route_feasible(sample_instance, sample_pair_solution, timeline, load_state, params) is False


def test_all_hard_constraints_satisfied_true(
    sample_instance,
    sample_params,
    sample_pair_solution,
    all_home_status,
) -> None:
    solution = Solution(vehicle_pairs=[sample_pair_solution], unserved_customers=set())
    assert all_hard_constraints_satisfied(sample_instance, solution, all_home_status, sample_params) is True


def test_all_hard_constraints_satisfied_false_when_structure_invalid(
    sample_instance,
    sample_params,
    all_home_status,
) -> None:
    bad_pair = VehiclePairSolution(
        pair_id=1,
        truck_route=[0, 1, 3, 5, 0],
        sorties=[Sortie(launch_node=5, recovery_node=1, customers=[2, 4])],
        truck_customers={1, 3, 5},
        drone_customers={2, 4},
    )
    solution = Solution(vehicle_pairs=[bad_pair], unserved_customers=set())

    assert all_hard_constraints_satisfied(sample_instance, solution, all_home_status, sample_params) is False
