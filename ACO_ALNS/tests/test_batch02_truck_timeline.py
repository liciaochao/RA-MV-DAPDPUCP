from __future__ import annotations

from dataclasses import replace

import pytest

from spd.core import compute_truck_timeline


def test_compute_truck_timeline_nominal_values(
    sample_instance,
    sample_params,
    sample_pair_solution,
    all_home_status,
) -> None:
    timeline = compute_truck_timeline(
        instance=sample_instance,
        pair_solution=sample_pair_solution,
        home_status=all_home_status,
        params=sample_params,
    )

    # Node 1 timeline.
    at_1 = sample_instance.truck_travel_time(0, 1, sample_params.vehicle.truck_speed)
    dt_1 = max(at_1, sample_instance.customers[1].time_window[0]) + sample_instance.customers[1].service_time
    assert timeline.truck_arrival[1] == pytest.approx(at_1)
    assert timeline.truck_departure[1] == pytest.approx(dt_1)

    # Node 3 timeline.
    at_3 = dt_1 + sample_instance.truck_travel_time(1, 3, sample_params.vehicle.truck_speed)
    dt_3 = max(at_3, sample_instance.customers[3].time_window[0]) + sample_instance.customers[3].service_time
    assert timeline.truck_arrival[3] == pytest.approx(at_3)
    assert timeline.truck_departure[3] == pytest.approx(dt_3)

    # Sortie timeline should be computed at launch node 1.
    at_d_2 = dt_1 + sample_instance.drone_travel_time(1, 2, sample_params.vehicle.drone_speed)
    dt_d_2 = max(at_d_2, sample_instance.customers[2].time_window[0]) + sample_instance.customers[2].service_time
    at_d_4 = dt_d_2 + sample_instance.drone_travel_time(2, 4, sample_params.vehicle.drone_speed)
    dt_d_4 = max(at_d_4, sample_instance.customers[4].time_window[0]) + sample_instance.customers[4].service_time
    at_d_5 = dt_d_4 + sample_instance.drone_travel_time(4, 5, sample_params.vehicle.drone_speed)

    assert timeline.drone_arrival[2] == pytest.approx(at_d_2)
    assert timeline.drone_departure[2] == pytest.approx(dt_d_2)
    assert timeline.drone_arrival[4] == pytest.approx(at_d_4)
    assert timeline.drone_departure[4] == pytest.approx(dt_d_4)
    assert timeline.drone_arrival[5] == pytest.approx(at_d_5)

    # Node 5 timeline (recovery + service).
    at_5 = dt_3 + sample_instance.truck_travel_time(3, 5, sample_params.vehicle.truck_speed)
    service_start_5 = max(at_5, at_d_5, sample_instance.customers[5].time_window[0])
    dt_5 = service_start_5 + sample_instance.customers[5].service_time
    assert timeline.truck_arrival[5] == pytest.approx(at_5)
    assert timeline.truck_departure[5] == pytest.approx(dt_5)

    # Return to depot timeline.
    at_depot = dt_5 + sample_instance.truck_travel_time(5, 0, sample_params.vehicle.truck_speed)
    assert timeline.truck_arrival[0] == pytest.approx(at_depot)
    assert timeline.truck_departure[0] == pytest.approx(at_depot)


def test_compute_truck_timeline_recovery_wait_is_applied(
    sample_instance,
    sample_params,
    sample_pair_solution,
    all_home_status,
) -> None:
    # Make truck fast and drone slow so recovery wait is clearly visible.
    tuned_params = replace(
        sample_params,
        vehicle=replace(sample_params.vehicle, truck_speed=2.0, drone_speed=0.2),
    )
    home_status = dict(all_home_status)
    home_status[3] = False

    timeline = compute_truck_timeline(
        instance=sample_instance,
        pair_solution=sample_pair_solution,
        home_status=home_status,
        params=tuned_params,
    )

    customer_5 = sample_instance.customers[5]
    no_recovery_wait_start = max(timeline.truck_arrival[5], customer_5.time_window[0])
    merged_start = max(timeline.truck_arrival[5], timeline.drone_arrival[5], customer_5.time_window[0])

    assert timeline.drone_arrival[5] > timeline.truck_arrival[5]
    assert merged_start > no_recovery_wait_start
    assert timeline.truck_departure[5] == pytest.approx(merged_start + customer_5.service_time)


def test_compute_truck_timeline_not_home_customer_skips_service_time(
    sample_instance,
    sample_params,
    sample_pair_solution,
    all_home_status,
) -> None:
    home_status = dict(all_home_status)
    home_status[3] = False

    timeline = compute_truck_timeline(
        instance=sample_instance,
        pair_solution=sample_pair_solution,
        home_status=home_status,
        params=sample_params,
    )

    customer_3 = sample_instance.customers[3]
    service_start_3 = max(timeline.truck_arrival[3], customer_3.time_window[0])

    assert timeline.truck_departure[3] == pytest.approx(service_start_3)
    assert timeline.truck_departure[3] != pytest.approx(service_start_3 + customer_3.service_time)
