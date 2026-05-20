from __future__ import annotations

from copy import deepcopy
import random

import spd.aco_alns as cross_group_mod


def _pair_by_id(solution, pair_id: int):
    for pair in solution.vehicle_pairs:
        if pair.pair_id == pair_id:
            return pair
    raise AssertionError(f"pair_id {pair_id} not found")



def test_cross_pair_swap_customer_sets_and_no_loss(
    sample_instance_two_pairs,
    sample_params,
    sample_solution_two_pairs,
    all_home_status,
    monkeypatch,
) -> None:
    solution = deepcopy(sample_solution_two_pairs)
    original_union = set(solution.all_customers)

    seed = 13
    exp_rng = random.Random(seed)
    pair_a, pair_b = exp_rng.sample(solution.used_vehicle_pairs, 2)

    q = max(1, sample_params.alns.cross_group_remove_count)
    sel_a = set(exp_rng.sample(sorted(pair_a.all_customers), min(q, len(pair_a.all_customers))))
    sel_b = set(exp_rng.sample(sorted(pair_b.all_customers), min(q, len(pair_b.all_customers))))

    expected_a = (set(pair_a.all_customers) - sel_a) | sel_b
    expected_b = (set(pair_b.all_customers) - sel_b) | sel_a

    monkeypatch.setattr(
        cross_group_mod,
        "compute_objective",
        lambda instance, sol, home_status, params: 1.0 if sol is solution else 0.0,
    )
    monkeypatch.setattr(cross_group_mod, "all_hard_constraints_satisfied", lambda *args, **kwargs: True)

    result = cross_group_mod.cross_pair_swap(
        solution,
        sample_instance_two_pairs,
        sample_params,
        all_home_status,
        random.Random(seed),
    )

    result_a = _pair_by_id(result, pair_a.pair_id)
    result_b = _pair_by_id(result, pair_b.pair_id)

    assert result_a.all_customers == expected_a
    assert result_b.all_customers == expected_b
    assert set(result.all_customers) == original_union
    assert result.unserved_customers == set()



def test_cross_pair_transfer_customer_sets_and_no_loss(
    sample_instance_two_pairs,
    sample_params,
    sample_solution_two_pairs,
    all_home_status,
    monkeypatch,
) -> None:
    solution = deepcopy(sample_solution_two_pairs)
    original_union = set(solution.all_customers)

    seed = 17
    exp_rng = random.Random(seed)
    pair_a, pair_b = exp_rng.sample(solution.used_vehicle_pairs, 2)
    moved = exp_rng.choice(sorted(pair_a.all_customers))

    expected_a = set(pair_a.all_customers)
    expected_b = set(pair_b.all_customers)
    expected_a.remove(moved)
    expected_b.add(moved)

    monkeypatch.setattr(
        cross_group_mod,
        "compute_objective",
        lambda instance, sol, home_status, params: 1.0 if sol is solution else 0.0,
    )
    monkeypatch.setattr(cross_group_mod, "all_hard_constraints_satisfied", lambda *args, **kwargs: True)

    result = cross_group_mod.cross_pair_transfer(
        solution,
        sample_instance_two_pairs,
        sample_params,
        all_home_status,
        random.Random(seed),
    )

    result_a = _pair_by_id(result, pair_a.pair_id)
    result_b = _pair_by_id(result, pair_b.pair_id)

    assert result_a.all_customers == expected_a
    assert result_b.all_customers == expected_b
    assert set(result.all_customers) == original_union
    assert result.unserved_customers == set()
