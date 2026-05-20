from __future__ import annotations

from copy import deepcopy
import random

import pytest

from spd.core import compute_truck_timeline
from spd.config import map_time_to_slot
from spd.core import get_arrival_time
from spd.types import Solution
from spd.aco_alns import (
    d1_random_removal,
    d2_worst_removal,
    d3_related_removal,
    d4_low_home_probability_removal,
)


def _build_solution(sample_pair_solution) -> Solution:
    return Solution(vehicle_pairs=[deepcopy(sample_pair_solution)], unserved_customers=set())


def _snapshot(solution: Solution) -> tuple[tuple[int, ...], frozenset[int], frozenset[int], tuple[tuple[int, int, tuple[int, ...]], ...]]:
    pair = solution.vehicle_pairs[0]
    sorties = tuple((s.launch_node, s.recovery_node, tuple(s.customers)) for s in pair.sorties)
    return (
        tuple(pair.truck_route),
        frozenset(pair.truck_customers),
        frozenset(pair.drone_customers),
        sorties,
    )


def _assert_removed_absent(partial_solution: Solution, removed_customers: list[int]) -> None:
    for pair in partial_solution.vehicle_pairs:
        for customer_id in removed_customers:
            assert customer_id not in pair.truck_customers
            assert customer_id not in pair.drone_customers
            assert customer_id not in pair.truck_route
            for sortie in pair.sorties:
                assert customer_id not in sortie.customers



def test_d1_random_removal_removes_count_and_keeps_input_immutable(
    sample_instance,
    sample_params,
    sample_pair_solution,
    all_home_status,
) -> None:
    solution = _build_solution(sample_pair_solution)
    before = _snapshot(solution)

    result = d1_random_removal(
        solution,
        sample_instance,
        sample_params,
        remove_count=2,
        home_status=all_home_status,
        rng=random.Random(1),
    )

    assert len(result.removed_customers) == 2
    _assert_removed_absent(result.partial_solution, result.removed_customers)
    assert _snapshot(solution) == before



def test_d2_worst_removal_removes_count_and_keeps_input_immutable(
    sample_instance,
    sample_params,
    sample_pair_solution,
    all_home_status,
) -> None:
    solution = _build_solution(sample_pair_solution)
    before = _snapshot(solution)

    result = d2_worst_removal(
        solution,
        sample_instance,
        sample_params,
        remove_count=2,
        home_status=all_home_status,
        rng=random.Random(2),
    )

    assert len(result.removed_customers) == 2
    _assert_removed_absent(result.partial_solution, result.removed_customers)
    assert _snapshot(solution) == before



def test_d3_related_removal_geographically_related_and_keeps_input_immutable(
    sample_instance,
    sample_params,
    sample_pair_solution,
    all_home_status,
) -> None:
    solution = _build_solution(sample_pair_solution)
    before = _snapshot(solution)

    remove_count = 3
    seed_rng = random.Random(5)

    # Reproduce expected seed and nearest neighbors using the same rule.
    all_customers = sorted(solution.all_customers)
    expected_seed = seed_rng.choice(all_customers)
    sx = sample_instance.customers[expected_seed].x
    sy = sample_instance.customers[expected_seed].y
    others = [customer_id for customer_id in all_customers if customer_id != expected_seed]
    others.sort(
        key=lambda customer_id: (
            ((sample_instance.customers[customer_id].x - sx) ** 2 + (sample_instance.customers[customer_id].y - sy) ** 2) ** 0.5,
            customer_id,
        )
    )
    expected_removed = {expected_seed, *others[: remove_count - 1]}

    result = d3_related_removal(
        solution,
        sample_instance,
        sample_params,
        remove_count=remove_count,
        home_status=all_home_status,
        rng=random.Random(5),
    )

    assert len(result.removed_customers) == remove_count
    assert set(result.removed_customers) == expected_removed
    _assert_removed_absent(result.partial_solution, result.removed_customers)
    assert _snapshot(solution) == before



def test_d4_low_home_probability_removal_lowest_probability_and_keeps_input_immutable(
    sample_instance,
    sample_params,
    sample_pair_solution,
    all_home_status,
) -> None:
    solution = _build_solution(sample_pair_solution)
    before = _snapshot(solution)

    # Compute expected low-probability customers by specification.
    pair = solution.vehicle_pairs[0]
    timeline = compute_truck_timeline(sample_instance, pair, all_home_status, sample_params)

    ranked = []
    for customer_id in sorted(pair.all_customers):
        arrival_time = get_arrival_time(customer_id, pair, timeline)
        time_slot = map_time_to_slot(arrival_time, sample_params.time)
        prob = sample_instance.customers[customer_id].home_probabilities[time_slot - 1]
        ranked.append((prob, customer_id))
    ranked.sort(key=lambda item: (item[0], item[1]))

    expected_removed = [customer_id for _, customer_id in ranked[:2]]

    result = d4_low_home_probability_removal(
        solution,
        sample_instance,
        sample_params,
        remove_count=2,
        home_status=all_home_status,
        rng=random.Random(3),
    )

    assert len(result.removed_customers) == 2
    assert result.removed_customers == expected_removed
    _assert_removed_absent(result.partial_solution, result.removed_customers)
    assert _snapshot(solution) == before
