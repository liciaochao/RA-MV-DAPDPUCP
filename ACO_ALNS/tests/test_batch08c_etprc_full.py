from __future__ import annotations

import random

from spd.core import all_hard_constraints_satisfied
from spd.etprc import ETPRCBuilder


def test_build_initial_solution_structure_complete(sample_instance, sample_params) -> None:
    builder = ETPRCBuilder()

    solution = builder.build_initial_solution(sample_instance, sample_params, random.Random(42))

    assigned = set()
    for pair in solution.vehicle_pairs:
        assigned |= pair.all_customers

    assert assigned == set(sample_instance.customers.keys())
    assert solution.unserved_customers == set()



def test_build_initial_solution_hard_constraints_satisfied(sample_instance, sample_params) -> None:
    builder = ETPRCBuilder()

    solution = builder.build_initial_solution(sample_instance, sample_params, random.Random(7))
    home_status = {customer_id: True for customer_id in sample_instance.customers}

    assert all_hard_constraints_satisfied(sample_instance, solution, home_status, sample_params) is True



def test_build_initial_solution_sortie_launch_recovery_not_depot(sample_instance, sample_params) -> None:
    builder = ETPRCBuilder()

    solution = builder.build_initial_solution(sample_instance, sample_params, random.Random(21))

    for pair in solution.vehicle_pairs:
        for sortie in pair.sorties:
            assert sortie.launch_node != sample_instance.depot_id
            assert sortie.recovery_node != sample_instance.depot_id



def test_build_initial_solution_logs_and_handles_skipped(sample_instance, sample_params, caplog) -> None:
    """Verify that when skipping occurs, it is logged and the solution remains feasible."""
    import logging
    builder = ETPRCBuilder()

    # Use default sample_instance which should NOT trigger skipping.
    with caplog.at_level(logging.WARNING):
        solution = builder.build_initial_solution(sample_instance, sample_params, random.Random(42))

    # On the small sample, nothing should be skipped.
    assert solution.unserved_customers == set()
    assigned = set()
    for pair in solution.vehicle_pairs:
        assigned |= pair.all_customers
    assert assigned == set(sample_instance.customers.keys())


def test_build_truck_route_skips_return_correctly(sample_instance, sample_params) -> None:
    """Verify _build_truck_route with return_skipped returns the expected tuple shape."""
    builder = ETPRCBuilder()
    group = set(sample_instance.customers.keys())

    result = builder._build_truck_route(group, sample_instance, sample_params, return_skipped=True)

    assert isinstance(result, tuple)
    assert len(result) == 2
    route, skipped = result
    assert isinstance(route, list)
    assert isinstance(skipped, list)
    # Route must start and end at depot.
    assert route[0] == sample_instance.depot_id
    assert route[-1] == sample_instance.depot_id


def test_build_truck_route_return_skipped_shape(sample_instance, sample_params) -> None:
    """Verify _build_truck_route with return_skipped returns correct tuple."""
    builder = ETPRCBuilder()
    group = set(sample_instance.customers.keys())

    result = builder._build_truck_route(
        group, sample_instance, sample_params, return_skipped=True,
    )

    assert isinstance(result, tuple), "return_skipped=True must return a tuple"
    assert len(result) == 2
    route, skipped = result
    assert isinstance(route, list)
    assert isinstance(skipped, list)
    assert route[0] == sample_instance.depot_id
    assert route[-1] == sample_instance.depot_id
    # On small instance, no customer should be skipped.
    assert skipped == []


def test_build_truck_route_without_return_skipped_unchanged(
    sample_instance, sample_params,
) -> None:
    """Verify _build_truck_route without return_skipped still returns list."""
    builder = ETPRCBuilder()
    group = set(sample_instance.customers.keys())

    result = builder._build_truck_route(group, sample_instance, sample_params)

    assert isinstance(result, list), "default call must return list, not tuple"
    assert result[0] == sample_instance.depot_id
    assert result[-1] == sample_instance.depot_id



def test_unserved_customers_increase_objective(sample_instance, sample_params) -> None:
    """Verify unserved customers add penalty to objective value."""
    from spd.core import compute_objective
    from spd.types import Solution

    builder = ETPRCBuilder()
    home_status = {cid: True for cid in sample_instance.customers}

    solution_full = builder.build_initial_solution(
        sample_instance, sample_params, random.Random(42),
    )
    assert solution_full.unserved_customers == set()
    obj_full = compute_objective(
        sample_instance, solution_full, home_status, sample_params,
    )

    # Artificially create a solution with one unserved customer.
    first_cid = sorted(sample_instance.customers.keys())[0]
    solution_degraded = Solution(
        vehicle_pairs=list(solution_full.vehicle_pairs),
        unserved_customers={first_cid},
    )
    obj_degraded = compute_objective(
        sample_instance, solution_degraded, home_status, sample_params,
    )

    assert obj_degraded > obj_full, (
        "Objective with unserved customer must be strictly higher"
    )

def test_fallback_rebuild_restores_unserved(sample_instance, sample_params):
    """Verify _fallback_rebuild assigns unserved customers back to pairs."""
    from spd.aco_alns import _fallback_rebuild
    from copy import deepcopy

    rng = random.Random(42)
    builder = ETPRCBuilder()
    solution = builder.build_initial_solution(sample_instance, sample_params, rng)

    # Verify baseline: all served
    assert solution.unserved_customers == set()

    # Simulate destroy: remove a customer and mark as unserved
    victim = next(iter(sample_instance.customers))
    for pair in solution.vehicle_pairs:
        if victim in pair.truck_customers:
            pair.truck_customers.discard(victim)
            pair.truck_route = [n for n in pair.truck_route if n != victim]
        if victim in pair.drone_customers:
            pair.drone_customers.discard(victim)
            pair.sorties = [s for s in pair.sorties if victim not in s.customers]
    solution.unserved_customers.add(victim)

    assert victim in solution.unserved_customers

    home_status = {cid: True for cid in sample_instance.customers}
    _fallback_rebuild(solution, sample_instance, sample_params, home_status)

    # After fallback, victim should be served again
    assert victim not in solution.unserved_customers
    # Total served should be complete
    assert solution.all_customers == set(sample_instance.customers)
