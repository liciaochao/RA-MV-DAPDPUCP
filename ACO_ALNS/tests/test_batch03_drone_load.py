from __future__ import annotations

import pytest

from spd.core import (
    check_sortie_load_feasibility,
    compute_final_drone_load,
    initial_sortie_load,
    update_sortie_load,
)
from spd.types import Sortie


def test_pure_delivery_sortie_load_decreases(sample_instance, all_home_status) -> None:
    sortie = Sortie(launch_node=1, recovery_node=5, customers=[1, 2])

    load0 = initial_sortie_load(sample_instance, sortie)
    assert load0 == pytest.approx(2.0)

    load1 = update_sortie_load(load0, 1, all_home_status[1], sample_instance)
    load2 = update_sortie_load(load1, 2, all_home_status[2], sample_instance)

    assert load1 == pytest.approx(0.8)
    assert load2 == pytest.approx(0.0)


def test_pure_pickup_sortie_load_increases(sample_instance, all_home_status) -> None:
    sortie = Sortie(launch_node=1, recovery_node=5, customers=[4, 6])

    load0 = initial_sortie_load(sample_instance, sortie)
    assert load0 == pytest.approx(0.0)

    load1 = update_sortie_load(load0, 4, all_home_status[4], sample_instance)
    load2 = update_sortie_load(load1, 6, all_home_status[6], sample_instance)

    assert load1 == pytest.approx(1.5)
    assert load2 == pytest.approx(2.1)


def test_mixed_sortie_not_home_keeps_load(sample_instance, all_home_status) -> None:
    sortie = Sortie(launch_node=1, recovery_node=5, customers=[2, 4, 6])
    home_status = dict(all_home_status)
    home_status[4] = False

    load0 = initial_sortie_load(sample_instance, sortie)
    load1 = update_sortie_load(load0, 2, home_status[2], sample_instance)
    load2 = update_sortie_load(load1, 4, home_status[4], sample_instance)
    load3 = update_sortie_load(load2, 6, home_status[6], sample_instance)

    assert load0 == pytest.approx(0.8)
    assert load1 == pytest.approx(0.0)
    assert load2 == pytest.approx(0.0)  # customer 4 not home
    assert load3 == pytest.approx(0.6)


def test_check_sortie_load_feasibility_detects_overload(
    sample_instance,
    sample_params,
    all_home_status,
) -> None:
    # Pickup sequence exceeds Q_D=5.0 at the third customer: 1.5 + 3.0 + 0.6 = 5.1
    sortie = Sortie(launch_node=1, recovery_node=5, customers=[4, 5, 6])

    feasible = check_sortie_load_feasibility(sample_instance, sortie, all_home_status, sample_params)
    assert feasible is False


def test_compute_final_drone_load_matches_manual_calculation(sample_instance, all_home_status) -> None:
    sortie = Sortie(launch_node=1, recovery_node=5, customers=[2, 4, 6])
    home_status = dict(all_home_status)
    home_status[4] = False

    # Manual replay: initial 0.8 -> after 2: 0.0 -> after 4(not home): 0.0 -> after 6: 0.6
    final_load = compute_final_drone_load(sample_instance, sortie, home_status)
    assert final_load == pytest.approx(0.6)
