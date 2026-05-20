from __future__ import annotations

import pytest

from spd.core import compute_drone_timeline
from spd.core import find_sortie_by_launch, find_sortie_by_recovery, get_arrival_time
from spd.types import TimelineState


def test_find_sortie_by_launch(sample_pair_solution) -> None:
    sortie = sample_pair_solution.sorties[0]

    assert find_sortie_by_launch(sample_pair_solution.sorties, 1) is sortie
    assert find_sortie_by_launch(sample_pair_solution.sorties, 99) is None


def test_find_sortie_by_recovery(sample_pair_solution) -> None:
    sortie = sample_pair_solution.sorties[0]

    assert find_sortie_by_recovery(sample_pair_solution.sorties, 5) is sortie
    assert find_sortie_by_recovery(sample_pair_solution.sorties, 100) is None


def test_get_arrival_time(sample_pair_solution) -> None:
    timeline = TimelineState(
        truck_arrival={1: 12.0, 3: 20.0, 5: 28.0},
        drone_arrival={2: 13.0, 4: 23.0},
    )

    assert get_arrival_time(1, sample_pair_solution, timeline) == pytest.approx(12.0)
    assert get_arrival_time(2, sample_pair_solution, timeline) == pytest.approx(13.0)

    with pytest.raises(KeyError):
        get_arrival_time(6, sample_pair_solution, timeline)


def test_compute_drone_timeline_with_home_and_not_home(
    sample_instance,
    sample_params,
    sample_pair_solution,
    all_home_status,
) -> None:
    sortie = sample_pair_solution.sorties[0]
    timeline = TimelineState(truck_departure={1: 10.0})

    # Customer 2 is at home, customer 4 is not at home.
    home_status = dict(all_home_status)
    home_status[4] = False

    compute_drone_timeline(
        instance=sample_instance,
        sortie=sortie,
        pair_solution=sample_pair_solution,
        home_status=home_status,
        params=sample_params,
        timeline=timeline,
    )

    # Launch node arrival equals truck departure at launch.
    assert timeline.drone_arrival[1] == pytest.approx(10.0)

    # Distances are 3 km and drone speed is 1 km/min.
    assert timeline.drone_arrival[2] == pytest.approx(13.0)
    assert timeline.drone_departure[2] == pytest.approx(20.0)  # max(13, e2=15) + s2=5

    assert timeline.drone_arrival[4] == pytest.approx(23.0)
    assert timeline.drone_departure[4] == pytest.approx(30.0)  # max(23, e4=30), no service

    # Recovery arrival: DT_D[4] + t_D(4,5) = 30 + 3.
    assert timeline.drone_arrival[5] == pytest.approx(33.0)

    # Not-at-home customer should not include service time in departure.
    c4 = sample_instance.customers[4]
    expected_without_service = max(timeline.drone_arrival[4], c4.time_window[0])
    expected_with_service = expected_without_service + c4.service_time
    assert timeline.drone_departure[4] == pytest.approx(expected_without_service)
    assert timeline.drone_departure[4] != pytest.approx(expected_with_service)
