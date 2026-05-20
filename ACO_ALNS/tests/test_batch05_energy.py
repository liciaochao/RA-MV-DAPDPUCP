from __future__ import annotations

from dataclasses import replace

import pytest

from spd.core import compute_sortie_energy, estimate_remaining_energy
from spd.core import compute_truck_timeline
from spd.types import Sortie, VehiclePairSolution


def test_compute_sortie_energy_single_customer_home_matches_stage_sum(
    sample_instance,
    sample_params,
    all_home_status,
) -> None:
    sortie = Sortie(launch_node=1, recovery_node=5, customers=[2])
    pair_solution = VehiclePairSolution(
        pair_id=1,
        truck_route=[0, 1, 5],
        sorties=[sortie],
        truck_customers={1, 5},
        drone_customers={2},
    )

    timeline = compute_truck_timeline(sample_instance, pair_solution, all_home_status, sample_params)
    energy = compute_sortie_energy(sample_instance, sortie, timeline, all_home_status, sample_params)

    eta = sample_params.energy.eta_wh_per_kg_min
    w_d = sample_params.vehicle.drone_empty_weight

    c2 = sample_instance.customers[2]
    initial_load = c2.weight
    final_load = 0.0

    # Stage 1 (launch and recovery flights).
    e_fly_launch = eta * (w_d + initial_load) * sample_instance.drone_travel_time(1, 2, sample_params.vehicle.drone_speed)
    e_fly_recovery = eta * (w_d + final_load) * sample_instance.drone_travel_time(2, 5, sample_params.vehicle.drone_speed)

    # Stage 2 waiting at customer.
    wait_2 = max(0.0, c2.time_window[0] - timeline.drone_arrival[2])
    e_wait = eta * (w_d + initial_load) * wait_2

    # Stage 3 service at customer (home=True).
    e_service = eta * (w_d + initial_load) * c2.service_time

    # Stage 4 hover at recovery.
    hover = max(0.0, timeline.truck_arrival[5] - timeline.drone_arrival[5])
    e_hover = eta * (w_d + final_load) * hover

    expected = e_fly_launch + e_wait + e_service + e_fly_recovery + e_hover
    assert energy == pytest.approx(expected)


def test_compute_sortie_energy_multi_customer_partial_not_home(
    sample_instance,
    sample_params,
    sample_pair_solution,
    all_home_status,
) -> None:
    home_status = dict(all_home_status)
    home_status[4] = False

    timeline = compute_truck_timeline(sample_instance, sample_pair_solution, home_status, sample_params)
    sortie = sample_pair_solution.sorties[0]
    energy = compute_sortie_energy(sample_instance, sortie, timeline, home_status, sample_params)

    eta = sample_params.energy.eta_wh_per_kg_min
    w_d = sample_params.vehicle.drone_empty_weight

    c2 = sample_instance.customers[2]
    c4 = sample_instance.customers[4]

    # Load evolution: initial 0.8 -> after c2 (delivery, home) => 0.0 -> c4 not-home keeps 0.0.
    load0 = c2.weight
    load_after_2 = 0.0
    load_after_4 = 0.0

    e_fly_1_2 = eta * (w_d + load0) * sample_instance.drone_travel_time(1, 2, sample_params.vehicle.drone_speed)
    e_wait_2 = eta * (w_d + load0) * max(0.0, c2.time_window[0] - timeline.drone_arrival[2])
    e_service_2 = eta * (w_d + load0) * c2.service_time

    e_fly_2_4 = eta * (w_d + load_after_2) * sample_instance.drone_travel_time(2, 4, sample_params.vehicle.drone_speed)
    e_wait_4 = eta * (w_d + load_after_2) * max(0.0, c4.time_window[0] - timeline.drone_arrival[4])
    e_service_4 = 0.0  # not-home customer has zero service energy

    e_fly_4_5 = eta * (w_d + load_after_4) * sample_instance.drone_travel_time(4, 5, sample_params.vehicle.drone_speed)
    hover = max(0.0, timeline.truck_arrival[5] - timeline.drone_arrival[5])
    e_hover = eta * (w_d + load_after_4) * hover

    expected = e_fly_1_2 + e_wait_2 + e_service_2 + e_fly_2_4 + e_wait_4 + e_service_4 + e_fly_4_5 + e_hover

    assert energy == pytest.approx(expected)
    assert e_wait_4 >= 0.0
    assert e_service_4 == 0.0


def test_compute_sortie_energy_hover_positive_when_drone_arrives_early(
    sample_instance,
    sample_params,
    sample_pair_solution,
    all_home_status,
) -> None:
    tuned_params = replace(
        sample_params,
        vehicle=replace(sample_params.vehicle, truck_speed=0.4, drone_speed=1.2),
    )

    timeline = compute_truck_timeline(sample_instance, sample_pair_solution, all_home_status, tuned_params)
    sortie = sample_pair_solution.sorties[0]

    hover_time = max(0.0, timeline.truck_arrival[sortie.recovery_node] - timeline.drone_arrival[sortie.recovery_node])
    energy = compute_sortie_energy(sample_instance, sortie, timeline, all_home_status, tuned_params)

    assert hover_time > 0.0
    assert energy > 0.0


def test_compute_sortie_energy_hover_zero_when_drone_not_earlier(
    sample_instance,
    sample_params,
    sample_pair_solution,
    all_home_status,
) -> None:
    tuned_params = replace(
        sample_params,
        vehicle=replace(sample_params.vehicle, truck_speed=2.0, drone_speed=0.2),
    )

    timeline = compute_truck_timeline(sample_instance, sample_pair_solution, all_home_status, tuned_params)
    sortie = sample_pair_solution.sorties[0]

    hover_time = max(0.0, timeline.truck_arrival[sortie.recovery_node] - timeline.drone_arrival[sortie.recovery_node])
    energy = compute_sortie_energy(sample_instance, sortie, timeline, all_home_status, tuned_params)

    assert hover_time == pytest.approx(0.0)
    assert energy > 0.0


def test_estimate_remaining_energy_from_middle_customer_is_smaller(
    sample_instance,
    sample_params,
    sample_pair_solution,
    all_home_status,
) -> None:
    timeline = compute_truck_timeline(sample_instance, sample_pair_solution, all_home_status, sample_params)
    sortie = sample_pair_solution.sorties[0]

    full_energy = compute_sortie_energy(sample_instance, sortie, timeline, all_home_status, sample_params)
    remaining_energy = estimate_remaining_energy(
        sample_instance,
        sortie,
        from_customer=2,
        timeline=timeline,
        home_status=all_home_status,
        params=sample_params,
    )

    assert remaining_energy > 0.0
    assert remaining_energy < full_energy
