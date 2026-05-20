from __future__ import annotations

from copy import deepcopy
import random

from spd.core import all_hard_constraints_satisfied, check_sortie_structure
from spd.types import Solution
from spd.aco_alns import d1_random_removal
from spd.aco_alns import r1_greedy_insertion, r2_regret_insertion, r3_timeslot_aware_insertion


def _build_solution(sample_pair_solution) -> Solution:
    return Solution(vehicle_pairs=[deepcopy(sample_pair_solution)], unserved_customers=set())


def _validate_repaired_solution(
    repaired: Solution,
    original: Solution,
    removed_customers: list[int],
    sample_instance,
    sample_params,
    all_home_status,
) -> None:
    # All originally assigned customers should be assigned again after repair.
    assert repaired.all_customers == original.all_customers

    # Removed customers must be reinserted and no longer unserved.
    for customer_id in removed_customers:
        assert customer_id in repaired.all_customers
        assert customer_id not in repaired.unserved_customers

    # Sortie structure should stay valid.
    for pair in repaired.vehicle_pairs:
        for sortie in pair.sorties:
            assert check_sortie_structure(pair, sortie) is True

    # Full hard-feasibility should hold.
    assert all_hard_constraints_satisfied(sample_instance, repaired, all_home_status, sample_params) is True



def test_r1_repairs_after_d1(sample_instance, sample_params, sample_pair_solution, all_home_status) -> None:
    original = _build_solution(sample_pair_solution)
    destroyed = d1_random_removal(
        original,
        sample_instance,
        sample_params,
        remove_count=2,
        home_status=all_home_status,
        rng=random.Random(11),
    )

    repaired = r1_greedy_insertion(
        destroyed.partial_solution,
        destroyed.removed_customers,
        sample_instance,
        sample_params,
        all_home_status,
        random.Random(101),
    )

    _validate_repaired_solution(
        repaired,
        original,
        destroyed.removed_customers,
        sample_instance,
        sample_params,
        all_home_status,
    )



def test_r2_repairs_after_d1(sample_instance, sample_params, sample_pair_solution, all_home_status) -> None:
    original = _build_solution(sample_pair_solution)
    destroyed = d1_random_removal(
        original,
        sample_instance,
        sample_params,
        remove_count=2,
        home_status=all_home_status,
        rng=random.Random(22),
    )

    repaired = r2_regret_insertion(
        destroyed.partial_solution,
        destroyed.removed_customers,
        sample_instance,
        sample_params,
        all_home_status,
        random.Random(202),
    )

    _validate_repaired_solution(
        repaired,
        original,
        destroyed.removed_customers,
        sample_instance,
        sample_params,
        all_home_status,
    )



def test_r3_repairs_after_d1(sample_instance, sample_params, sample_pair_solution, all_home_status) -> None:
    original = _build_solution(sample_pair_solution)
    destroyed = d1_random_removal(
        original,
        sample_instance,
        sample_params,
        remove_count=2,
        home_status=all_home_status,
        rng=random.Random(33),
    )

    repaired = r3_timeslot_aware_insertion(
        destroyed.partial_solution,
        destroyed.removed_customers,
        sample_instance,
        sample_params,
        all_home_status,
        random.Random(303),
    )

    _validate_repaired_solution(
        repaired,
        original,
        destroyed.removed_customers,
        sample_instance,
        sample_params,
        all_home_status,
    )
