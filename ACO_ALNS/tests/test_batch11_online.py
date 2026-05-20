from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from itertools import permutations
import random

from spd.core import compute_energy_cost, estimate_remaining_energy
from spd.core import all_hard_constraints_satisfied, check_sortie_feasibility
from spd.core import compute_truck_load
from spd.core import compute_objective, failure_penalty
from spd.core import compute_truck_timeline
from spd.config import map_time_to_slot
from spd.types import ProblemInstance, Solution, Sortie, VehiclePairSolution
from spd.types import ServiceMode
from spd.types import CustomerNotHomeEvent
from spd.online import LightweightALNSOptimizer
from spd.online import (
    apply_b1_skip_infeasible_customers,
    apply_b2_truncate_sortie,
    reorder_remaining_customers,
)
from spd.online import OnlineReplanner


def _clone_instance_customer(
    base: ProblemInstance,
    customer_id: int,
    *,
    home_probabilities: tuple[float, ...] | None = None,
) -> ProblemInstance:
    customers = dict(base.customers)
    customer = customers[customer_id]
    customers[customer_id] = replace(
        customer,
        home_probabilities=home_probabilities if home_probabilities is not None else customer.home_probabilities,
    )
    return ProblemInstance(
        depot_id=base.depot_id,
        vehicle_pair_count=base.vehicle_pair_count,
        customers=customers,
        delivery_customers=base.delivery_customers,
        pickup_customers=base.pickup_customers,
        truck_distance_km=base.truck_distance_km,
        drone_distance_km=base.drone_distance_km,
    )


def _replace_sortie(pair_solution: VehiclePairSolution, sortie_index: int, customers: list[int]) -> VehiclePairSolution:
    sorties = [
        Sortie(
            launch_node=sortie.launch_node,
            recovery_node=sortie.recovery_node,
            customers=list(sortie.customers),
        )
        for sortie in pair_solution.sorties
    ]
    base = sorties[sortie_index]
    sorties[sortie_index] = Sortie(base.launch_node, base.recovery_node, list(customers))

    drone_customers: set[int] = set()
    for sortie in sorties:
        drone_customers |= set(sortie.customers)

    return VehiclePairSolution(
        pair_id=pair_solution.pair_id,
        truck_route=list(pair_solution.truck_route),
        sorties=sorties,
        truck_customers=set(pair_solution.truck_customers),
        drone_customers=drone_customers,
    )


def test_apply_b1_skip_infeasible_customers_skips_overload_customer(
    sample_instance,
    sample_params,
    sample_pair_solution,
    all_home_status,
) -> None:
    pair_solution = deepcopy(sample_pair_solution)
    sortie = pair_solution.sorties[0]

    # Mark event customer as not home so B1 starts checking from the next customer.
    home_status = dict(all_home_status)
    home_status[2] = False

    # Shrink drone payload capacity to force skipping customer 4 (pickup load increase).
    params = replace(sample_params, vehicle=replace(sample_params.vehicle, drone_capacity=1.0))
    timeline = compute_truck_timeline(sample_instance, pair_solution, home_status, params)

    repaired = apply_b1_skip_infeasible_customers(
        sample_instance,
        sortie,
        pair_solution,
        home_status,
        params,
        timeline,
    )

    assert repaired.customers == [2]


def test_apply_b2_truncate_sortie_when_remaining_energy_insufficient(
    sample_instance,
    sample_params,
    sample_pair_solution,
    all_home_status,
) -> None:
    pair_solution = deepcopy(sample_pair_solution)
    sortie = pair_solution.sorties[0]

    # Mark event customer as not home and use a tiny battery to force truncation.
    home_status = dict(all_home_status)
    home_status[2] = False
    params = replace(sample_params, energy=replace(sample_params.energy, drone_battery_capacity=1.0))
    timeline = compute_truck_timeline(sample_instance, pair_solution, home_status, params)

    truncated = apply_b2_truncate_sortie(
        sample_instance,
        sortie,
        pair_solution,
        home_status,
        params,
        timeline,
    )

    assert truncated.customers[0] == 2
    assert 4 not in truncated.customers


def test_reorder_remaining_customers_selects_best_feasible_order(
    sample_instance,
    sample_params,
    all_home_status,
) -> None:
    # Build a sortie with two active remaining customers after event customer 2.
    pair_solution = VehiclePairSolution(
        pair_id=1,
        truck_route=[0, 1, 3, 5, 0],
        sorties=[Sortie(launch_node=1, recovery_node=5, customers=[2, 4, 6])],
        truck_customers={1, 3, 5},
        drone_customers={2, 4, 6},
    )
    sortie = pair_solution.sorties[0]

    # Bias home probabilities to make ordering effect visible in expected failure term.
    instance = _clone_instance_customer(
        sample_instance,
        4,
        home_probabilities=(0.20, 0.25, 0.30, 0.35, 0.50, 0.65, 0.75, 0.85),
    )
    instance = _clone_instance_customer(
        instance,
        6,
        home_probabilities=(0.90, 0.85, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30),
    )

    home_status = dict(all_home_status)
    home_status[2] = False

    timeline = compute_truck_timeline(instance, pair_solution, home_status, sample_params)
    reordered = reorder_remaining_customers(
        instance,
        sortie,
        from_customer=2,
        pair_solution=pair_solution,
        home_status=home_status,
        params=sample_params,
        timeline=timeline,
    )

    # Independently compute the best feasible suffix by the same objective definition.
    suffix = [4, 6]
    best_suffix = suffix
    best_cost = None

    for perm in permutations(suffix):
        candidate_pair = _replace_sortie(pair_solution, 0, [2, *perm])
        candidate_sortie = candidate_pair.sorties[0]

        candidate_timeline = compute_truck_timeline(instance, candidate_pair, home_status, sample_params)
        candidate_load_state = compute_truck_load(instance, candidate_pair, home_status, sample_params)
        if not check_sortie_feasibility(
            instance,
            candidate_pair,
            candidate_sortie,
            candidate_timeline,
            candidate_load_state,
            home_status,
            sample_params,
        ):
            continue

        failure_cost = 0.0
        for customer_id in perm:
            slot = map_time_to_slot(candidate_timeline.drone_arrival[customer_id], sample_params.time)
            prob = instance.customers[customer_id].home_probabilities[slot - 1]
            failure_cost += (1.0 - prob) * failure_penalty(instance, customer_id, sample_params)

        remaining_energy = estimate_remaining_energy(
            instance,
            candidate_sortie,
            2,
            candidate_timeline,
            home_status,
            sample_params,
        )
        total_cost = failure_cost + compute_energy_cost(remaining_energy, sample_params)

        if best_cost is None or total_cost < best_cost:
            best_cost = total_cost
            best_suffix = list(perm)

    assert reordered == [2, *best_suffix]


def test_lightweight_alns_optimize_remaining_path_keeps_feasible(
    sample_instance,
    sample_params,
    sample_pair_solution,
    all_home_status,
) -> None:
    optimizer = LightweightALNSOptimizer()
    initial_pair = deepcopy(sample_pair_solution)

    initial_solution = Solution(vehicle_pairs=[deepcopy(initial_pair)], unserved_customers=set())
    initial_obj = compute_objective(sample_instance, initial_solution, all_home_status, sample_params)

    optimized_pair = optimizer.optimize_remaining_path(
        sample_instance,
        initial_pair,
        all_home_status,
        sample_params,
        random.Random(123),
    )
    optimized_solution = Solution(vehicle_pairs=[optimized_pair], unserved_customers=set())
    optimized_obj = compute_objective(sample_instance, optimized_solution, all_home_status, sample_params)

    assert all_hard_constraints_satisfied(sample_instance, optimized_solution, all_home_status, sample_params) is True
    assert optimized_obj <= initial_obj + 1e-9


def test_resolve_event_sortie_index_direct_and_lookup(sample_pair_solution) -> None:
    replanner = OnlineReplanner(LightweightALNSOptimizer())
    solution = Solution(vehicle_pairs=[deepcopy(sample_pair_solution)], unserved_customers=set())

    direct_event = CustomerNotHomeEvent(
        pair_id=1,
        customer_id=2,
        service_mode=ServiceMode.DRONE,
        event_time=10.0,
        sortie_index=3,
    )
    assert replanner.resolve_event_sortie_index(solution, direct_event) == 3

    lookup_event = CustomerNotHomeEvent(
        pair_id=1,
        customer_id=4,
        service_mode=ServiceMode.DRONE,
        event_time=12.0,
        sortie_index=None,
    )
    assert replanner.resolve_event_sortie_index(solution, lookup_event) == 0


def test_handle_event_flow_drone_and_truck(
    sample_instance,
    sample_params,
    sample_pair_solution,
    all_home_status,
) -> None:
    replanner = OnlineReplanner(LightweightALNSOptimizer())
    base_solution = Solution(vehicle_pairs=[deepcopy(sample_pair_solution)], unserved_customers=set())

    # Drone event flow.
    drone_home_status = dict(all_home_status)
    drone_event = CustomerNotHomeEvent(
        pair_id=1,
        customer_id=2,
        service_mode=ServiceMode.DRONE,
        event_time=20.0,
        sortie_index=None,
    )
    after_drone = replanner.handle_event(
        sample_instance,
        base_solution,
        drone_event,
        drone_home_status,
        sample_params,
        random.Random(7),
    )

    assert drone_home_status[2] is False
    assert after_drone.vehicle_pairs[0].pair_id == 1

    # Truck event flow.
    truck_home_status = dict(all_home_status)
    truck_event = CustomerNotHomeEvent(
        pair_id=1,
        customer_id=1,
        service_mode=ServiceMode.TRUCK,
        event_time=25.0,
        sortie_index=None,
    )
    after_truck = replanner.handle_event(
        sample_instance,
        base_solution,
        truck_event,
        truck_home_status,
        sample_params,
        random.Random(9),
    )

    assert truck_home_status[1] is False
    assert after_truck.vehicle_pairs[0].pair_id == 1
