from __future__ import annotations

import pytest

from spd.core import compute_final_drone_load, compute_truck_load
from spd.types import VehiclePairSolution


def test_compute_truck_load_pure_truck_route_no_sortie(sample_instance, sample_params, all_home_status) -> None:
    pair_solution = VehiclePairSolution(
        pair_id=1,
        truck_route=[0, 1, 4, 3],
        sorties=[],
        truck_customers={1, 4, 3},
        drone_customers=set(),
    )

    load_state = compute_truck_load(sample_instance, pair_solution, all_home_status, sample_params)

    # Initial load: delivery customers {1,3} = 1.2 + 2.5 = 3.7
    assert load_state.truck_load_at_node[0] == pytest.approx(3.7)
    # Node 1 delivery -> 2.5
    assert load_state.truck_load_at_node[1] == pytest.approx(2.5)
    # Node 4 pickup -> 4.0
    assert load_state.truck_load_at_node[4] == pytest.approx(4.0)
    # Node 3 delivery -> 1.5
    assert load_state.truck_load_at_node[3] == pytest.approx(1.5)


def test_compute_truck_load_with_sortie_launch_and_recovery(
    sample_instance,
    sample_params,
    sample_pair_solution,
    all_home_status,
) -> None:
    pair_solution = VehiclePairSolution(
        pair_id=sample_pair_solution.pair_id,
        truck_route=[0, 1, 3, 5],
        sorties=sample_pair_solution.sorties,
        truck_customers=sample_pair_solution.truck_customers,
        drone_customers=sample_pair_solution.drone_customers,
    )

    load_state = compute_truck_load(sample_instance, pair_solution, all_home_status, sample_params)
    sortie = pair_solution.sorties[0]

    # Initial load includes drone-delivery parcel for customer 2.
    expected_initial = 1.2 + 0.8 + 2.5
    assert load_state.truck_load_at_node[0] == pytest.approx(expected_initial)

    # Launch node 1: service delivery(1.2) then launch delivery parcel(0.8) => 2.5
    assert load_state.truck_load_at_node[1] == pytest.approx(2.5)

    # Node 3: truck delivery(2.5) => 0.0
    assert load_state.truck_load_at_node[3] == pytest.approx(0.0)

    # Recovery at node 5 adds final drone load, then truck pickup at node 5.
    final_drone_load = compute_final_drone_load(sample_instance, sortie, all_home_status)
    expected_node5 = 0.0 + final_drone_load + sample_instance.customers[5].weight
    assert final_drone_load == pytest.approx(1.5)
    assert load_state.truck_load_at_node[5] == pytest.approx(expected_node5)


def test_compute_truck_load_not_home_customer_keeps_load(
    sample_instance,
    sample_params,
    sample_pair_solution,
    all_home_status,
) -> None:
    pair_solution = VehiclePairSolution(
        pair_id=sample_pair_solution.pair_id,
        truck_route=[0, 1, 3, 5],
        sorties=sample_pair_solution.sorties,
        truck_customers=sample_pair_solution.truck_customers,
        drone_customers=sample_pair_solution.drone_customers,
    )

    home_status = dict(all_home_status)
    home_status[3] = False

    load_state = compute_truck_load(sample_instance, pair_solution, home_status, sample_params)

    # Node 3 is delivery but not-home, so load stays same as after node 1.
    assert load_state.truck_load_at_node[3] == pytest.approx(load_state.truck_load_at_node[1])


def test_initial_truck_load_includes_drone_delivery_parcels(
    sample_instance,
    sample_params,
    sample_pair_solution,
    all_home_status,
) -> None:
    pair_solution = VehiclePairSolution(
        pair_id=sample_pair_solution.pair_id,
        truck_route=[0, 1, 3, 5],
        sorties=sample_pair_solution.sorties,
        truck_customers=sample_pair_solution.truck_customers,
        drone_customers=sample_pair_solution.drone_customers,
    )

    load_state = compute_truck_load(sample_instance, pair_solution, all_home_status, sample_params)

    truck_only_delivery = sum(
        sample_instance.customers[c].weight
        for c in pair_solution.truck_customers
        if c in sample_instance.delivery_customers
    )
    all_pair_delivery = sum(
        sample_instance.customers[c].weight
        for c in pair_solution.all_customers
        if c in sample_instance.delivery_customers
    )

    assert load_state.truck_load_at_node[0] == pytest.approx(all_pair_delivery)
    assert all_pair_delivery > truck_only_delivery
