from __future__ import annotations

import inspect
import random
from copy import deepcopy
from itertools import permutations

from spd.aco_alns import d1_random_removal, d2_worst_removal, r1_greedy_insertion, r2_regret_insertion
from spd.config import ProblemParameters, map_time_to_slot
from spd.core import (
    all_hard_constraints_satisfied,
    check_sortie_feasibility,
    compute_energy_cost,
    compute_objective,
    compute_truck_load,
    compute_truck_timeline,
    estimate_remaining_energy,
    failure_penalty,
    initial_sortie_load,
    propagate_time_changes,
    update_sortie_load,
)
from spd.types import (
    CustomerNotHomeEvent,
    HomeStatusMap,
    ProblemInstance,
    ServiceMode,
    Solution,
    Sortie,
    TimelineState,
    VehiclePairSolution,
)

# ============================================================
# 原始文件: spd/online/repair.py
# ============================================================

"""Online repair strategy contracts (B1/B2 plus Step 3 reorder)."""




_EPS = 1e-9


def _resolve_from_customer(sortie: Sortie, home_status: HomeStatusMap) -> int | None:
    """Infer event customer in a sortie from home-status updates."""
    if not sortie.customers:
        return None

    for customer_id in sortie.customers:
        if not home_status[customer_id]:
            return customer_id

    # Fallback to the first customer when event customer is not explicitly marked.
    return sortie.customers[0]


def _find_sortie_index(pair_solution: VehiclePairSolution, sortie: Sortie) -> int:
    """Locate sortie index in a pair solution with identity-first lookup."""
    for idx, item in enumerate(pair_solution.sorties):
        if item is sortie:
            return idx

    for idx, item in enumerate(pair_solution.sorties):
        if (
            item.launch_node == sortie.launch_node
            and item.recovery_node == sortie.recovery_node
            and item.customers == sortie.customers
        ):
            return idx

    raise ValueError("sortie is not part of pair_solution.sorties")


def _replace_sortie(
    pair_solution: VehiclePairSolution,
    sortie: Sortie,
    new_sortie: Sortie,
) -> VehiclePairSolution:
    """Return a cloned pair with one sortie replaced and drone set refreshed."""
    sortie_index = _find_sortie_index(pair_solution, sortie)
    cloned_sorties = [
        Sortie(
            launch_node=item.launch_node,
            recovery_node=item.recovery_node,
            customers=list(item.customers),
        )
        for item in pair_solution.sorties
    ]
    cloned_sorties[sortie_index] = Sortie(
        launch_node=new_sortie.launch_node,
        recovery_node=new_sortie.recovery_node,
        customers=list(new_sortie.customers),
    )

    drone_customers: set[int] = set()
    for item in cloned_sorties:
        drone_customers |= set(item.customers)

    return VehiclePairSolution(
        pair_id=pair_solution.pair_id,
        truck_route=list(pair_solution.truck_route),
        sorties=cloned_sorties,
        truck_customers=set(pair_solution.truck_customers),
        drone_customers=drone_customers,
    )


def _estimate_consumed_energy_until(
    instance: ProblemInstance,
    sortie: Sortie,
    from_customer: int,
    home_status: HomeStatusMap,
    params: ProblemParameters,
    timeline: TimelineState,
) -> float:
    """Estimate consumed energy from launch through from_customer service."""
    if from_customer not in sortie.customers:
        return 0.0

    eta = params.energy.eta_wh_per_kg_min
    drone_empty_weight = params.vehicle.drone_empty_weight
    drone_speed = params.vehicle.drone_speed

    energy = 0.0
    load = initial_sortie_load(instance, sortie)
    prev_node = sortie.launch_node
    prev_depart = timeline.truck_departure.get(sortie.launch_node, params.time.depot_earliest)

    for customer_id in sortie.customers:
        total_weight = drone_empty_weight + load

        # Flight energy from previous node to this customer.
        fly_time = instance.drone_travel_time(prev_node, customer_id, drone_speed)
        energy += eta * total_weight * fly_time

        arrival_time = prev_depart + fly_time
        customer = instance.customers[customer_id]

        # Waiting and service energy at this customer.
        wait_time = max(0.0, customer.time_window[0] - arrival_time)
        energy += eta * total_weight * wait_time
        if home_status[customer_id]:
            energy += eta * total_weight * customer.service_time

        # Advance payload and timeline cursor.
        depart_time = max(arrival_time, customer.time_window[0])
        if home_status[customer_id]:
            depart_time += customer.service_time
        load = update_sortie_load(load, customer_id, home_status[customer_id], instance)

        prev_node = customer_id
        prev_depart = depart_time

        if customer_id == from_customer:
            break

    return energy


def apply_b1_skip_infeasible_customers(
    instance: ProblemInstance,
    sortie: Sortie,
    pair_solution: VehiclePairSolution,
    home_status: HomeStatusMap,
    params: ProblemParameters,
    timeline: TimelineState,
    from_customer: int | None = None,
) -> Sortie:
    """B1: skip customers causing hard-constraint violations."""
    _ = pair_solution

    if not sortie.customers:
        return Sortie(sortie.launch_node, sortie.recovery_node, [])

    trigger_customer = from_customer if from_customer is not None else _resolve_from_customer(sortie, home_status)
    if trigger_customer is None or trigger_customer not in sortie.customers:
        return Sortie(sortie.launch_node, sortie.recovery_node, list(sortie.customers))

    from_index = sortie.customers.index(trigger_customer)
    fixed_prefix = list(sortie.customers[: from_index + 1])
    remaining = list(sortie.customers[from_index + 1 :])
    if not remaining:
        return Sortie(sortie.launch_node, sortie.recovery_node, fixed_prefix)

    # Reconstruct load right after from_customer.
    current_load = initial_sortie_load(instance, sortie)
    for customer_id in fixed_prefix:
        current_load = update_sortie_load(current_load, customer_id, home_status[customer_id], instance)

    current_node = trigger_customer
    current_depart = timeline.drone_departure.get(trigger_customer, timeline.drone_arrival.get(trigger_customer, 0.0))
    kept_remaining: list[int] = []

    for customer_id in remaining:
        travel_time = instance.drone_travel_time(current_node, customer_id, params.vehicle.drone_speed)
        arrival_time = current_depart + travel_time

        customer = instance.customers[customer_id]

        # Skip if arrival breaks the time-window upper bound.
        if arrival_time > customer.time_window[1] + _EPS:
            continue

        next_load = update_sortie_load(current_load, customer_id, home_status[customer_id], instance)

        # Skip if drone payload exceeds capacity.
        if next_load > params.vehicle.drone_capacity + _EPS:
            continue

        kept_remaining.append(customer_id)
        current_load = next_load
        current_node = customer_id
        current_depart = max(arrival_time, customer.time_window[0])
        if home_status[customer_id]:
            current_depart += customer.service_time

    return Sortie(
        launch_node=sortie.launch_node,
        recovery_node=sortie.recovery_node,
        customers=fixed_prefix + kept_remaining,
    )


def apply_b2_truncate_sortie(
    instance: ProblemInstance,
    sortie: Sortie,
    pair_solution: VehiclePairSolution,
    home_status: HomeStatusMap,
    params: ProblemParameters,
    timeline: TimelineState,
    from_customer: int | None = None,
) -> Sortie:
    """B2: truncate sortie when remaining energy is insufficient."""
    if not sortie.customers:
        return Sortie(sortie.launch_node, sortie.recovery_node, [])

    trigger_customer = from_customer if from_customer is not None else _resolve_from_customer(sortie, home_status)
    if trigger_customer is None or trigger_customer not in sortie.customers:
        return Sortie(sortie.launch_node, sortie.recovery_node, list(sortie.customers))

    from_index = sortie.customers.index(trigger_customer)
    fixed_prefix = list(sortie.customers[: from_index + 1])
    remaining = list(sortie.customers[from_index + 1 :])
    if not remaining:
        return Sortie(sortie.launch_node, sortie.recovery_node, fixed_prefix)

    consumed_energy = _estimate_consumed_energy_until(
        instance,
        sortie,
        trigger_customer,
        home_status,
        params,
        timeline,
    )
    available_energy = params.energy.drone_battery_capacity - consumed_energy

    last_feasible_count = 0
    for keep_count in range(1, len(remaining) + 1):
        candidate_sortie = Sortie(
            launch_node=sortie.launch_node,
            recovery_node=sortie.recovery_node,
            customers=fixed_prefix + remaining[:keep_count],
        )
        candidate_pair = _replace_sortie(pair_solution, sortie, candidate_sortie)

        try:
            candidate_timeline = compute_truck_timeline(instance, candidate_pair, home_status, params)
            candidate_remaining_energy = estimate_remaining_energy(
                instance,
                candidate_sortie,
                trigger_customer,
                candidate_timeline,
                home_status,
                params,
            )
        except (AssertionError, KeyError, ValueError):
            break

        if candidate_remaining_energy <= available_energy + _EPS:
            last_feasible_count = keep_count
        else:
            break

    return Sortie(
        launch_node=sortie.launch_node,
        recovery_node=sortie.recovery_node,
        customers=fixed_prefix + remaining[:last_feasible_count],
    )


def reorder_remaining_customers(
    instance: ProblemInstance,
    sortie: Sortie,
    from_customer: int,
    pair_solution: VehiclePairSolution,
    home_status: HomeStatusMap,
    params: ProblemParameters,
    timeline: TimelineState,
) -> list[int]:
    """Step 3: reorder active remaining customers in current sortie.

    Expected strategy:
    - Enumerate permutations of customers after from_customer.
    - Filter by hard feasibility.
    - Select the order with minimum expected remaining cost.
    """
    _ = timeline

    if from_customer not in sortie.customers:
        return list(sortie.customers)

    from_index = sortie.customers.index(from_customer)
    fixed_prefix = list(sortie.customers[: from_index + 1])
    remaining = list(sortie.customers[from_index + 1 :])
    if len(remaining) <= 1:
        return list(sortie.customers)

    best_suffix: list[int] | None = None
    best_cost: float | None = None

    for perm in permutations(remaining):
        candidate_suffix = list(perm)
        candidate_sortie = Sortie(
            launch_node=sortie.launch_node,
            recovery_node=sortie.recovery_node,
            customers=fixed_prefix + candidate_suffix,
        )
        candidate_pair = _replace_sortie(pair_solution, sortie, candidate_sortie)

        try:
            candidate_timeline = compute_truck_timeline(instance, candidate_pair, home_status, params)
            candidate_load_state = compute_truck_load(instance, candidate_pair, home_status, params)
        except (AssertionError, KeyError, ValueError):
            continue

        if not check_sortie_feasibility(
            instance,
            candidate_pair,
            candidate_sortie,
            candidate_timeline,
            candidate_load_state,
            home_status,
            params,
        ):
            continue

        expected_failure_cost = 0.0
        for customer_id in candidate_suffix:
            arrival_time = candidate_timeline.drone_arrival[customer_id]
            slot = map_time_to_slot(arrival_time, params.time)
            home_prob = instance.customers[customer_id].home_probabilities[slot - 1]
            expected_failure_cost += (1.0 - home_prob) * failure_penalty(instance, customer_id, params)

        remaining_energy = estimate_remaining_energy(
            instance,
            candidate_sortie,
            from_customer,
            candidate_timeline,
            home_status,
            params,
        )
        expected_energy_cost = compute_energy_cost(remaining_energy, params)
        expected_cost = expected_failure_cost + expected_energy_cost

        if best_cost is None or expected_cost < best_cost - _EPS:
            best_cost = expected_cost
            best_suffix = candidate_suffix

    if best_suffix is None:
        return list(sortie.customers)
    return fixed_prefix + best_suffix


def repair_sortie(
    instance: ProblemInstance,
    sortie: Sortie,
    pair_solution: VehiclePairSolution,
    home_status: HomeStatusMap,
    params: ProblemParameters,
    timeline: TimelineState,
    from_customer: int | None = None,
) -> Sortie:
    """Composite online sortie repair for Step 2 and Step 3 of the workflow."""
    if not sortie.customers:
        return Sortie(sortie.launch_node, sortie.recovery_node, [])

    trigger_customer = from_customer if from_customer is not None else _resolve_from_customer(sortie, home_status)
    if trigger_customer is None:
        return Sortie(sortie.launch_node, sortie.recovery_node, list(sortie.customers))

    # Step B1: skip capacity/time-window violating remaining customers.
    b1_sortie = apply_b1_skip_infeasible_customers(
        instance,
        sortie,
        pair_solution,
        home_status,
        params,
        timeline,
        trigger_customer,
    )

    b1_pair = _replace_sortie(pair_solution, sortie, b1_sortie)
    try:
        b1_timeline = compute_truck_timeline(instance, b1_pair, home_status, params)
    except (AssertionError, KeyError, ValueError):
        b1_timeline = timeline

    # Step B2: truncate the suffix when remaining energy budget is not enough.
    b2_sortie = apply_b2_truncate_sortie(
        instance,
        b1_sortie,
        b1_pair,
        home_status,
        params,
        b1_timeline,
        trigger_customer,
    )

    b2_pair = _replace_sortie(b1_pair, b1_sortie, b2_sortie)
    try:
        b2_timeline = compute_truck_timeline(instance, b2_pair, home_status, params)
    except (AssertionError, KeyError, ValueError):
        b2_timeline = b1_timeline

    # Step 3: reorder active remaining customers by expected cost.
    reordered_customers = reorder_remaining_customers(
        instance,
        b2_sortie,
        trigger_customer,
        b2_pair,
        home_status,
        params,
        b2_timeline,
    )

    return Sortie(
        launch_node=b2_sortie.launch_node,
        recovery_node=b2_sortie.recovery_node,
        customers=reordered_customers,
    )

# ============================================================
# 原始文件: spd/online/lightweight_alns.py
# ============================================================

"""Lightweight ALNS contract for online re-optimization."""




_EPS = 1e-9


class LightweightALNSOptimizer:
    """Online optimizer for remaining path of one vehicle pair."""

    def optimize_remaining_path(
        self,
        instance: ProblemInstance,
        pair_solution: VehiclePairSolution,
        home_status: HomeStatusMap,
        params: ProblemParameters,
        rng: random.Random | None = None,
        frozen_customers: set[int] | None = None,
    ) -> VehiclePairSolution:
        """Run restricted online ALNS and return updated pair solution."""
        random_state = rng or random.Random(0)
        frozen = frozen_customers if frozen_customers is not None else set()

        # Work on a detached pair copy to avoid mutating caller-owned objects.
        current_pair = VehiclePairSolution(
            pair_id=pair_solution.pair_id,
            truck_route=list(pair_solution.truck_route),
            sorties=[
                Sortie(
                    launch_node=sortie.launch_node,
                    recovery_node=sortie.recovery_node,
                    customers=list(sortie.customers),
                )
                for sortie in pair_solution.sorties
            ],
            truck_customers=set(pair_solution.truck_customers),
            drone_customers=set(pair_solution.drone_customers),
        )
        current_solution = Solution(vehicle_pairs=[current_pair], unserved_customers=set())

        try:
            current_objective = compute_objective(instance, current_solution, home_status, params)
        except (AssertionError, KeyError, ValueError):
            return current_pair

        iterations = max(0, params.online_alns.iterations)
        remove_count = max(1, params.online_alns.remove_count)

        destroy_ops = [d1_random_removal, d2_worst_removal]
        repair_ops = [r1_greedy_insertion, r2_regret_insertion]

        for _ in range(iterations):
            destroy_op = random_state.choice(destroy_ops)
            repair_op = random_state.choice(repair_ops)

            try:
                destroy_result = destroy_op(
                    current_solution,
                    instance,
                    params,
                    remove_count,
                    home_status,
                    random_state,
                )
                # Skip this iteration if destroy touched frozen customers.
                if frozen and (set(destroy_result.removed_customers) & frozen):
                    continue
                candidate_solution = repair_op(
                    destroy_result.partial_solution,
                    destroy_result.removed_customers,
                    instance,
                    params,
                    home_status,
                    random_state,
                )
            except (AssertionError, KeyError, ValueError):
                continue

            # Online mode rejects any hard-constraint violation.
            if not all_hard_constraints_satisfied(instance, candidate_solution, home_status, params):
                continue

            try:
                candidate_objective = compute_objective(instance, candidate_solution, home_status, params)
            except (AssertionError, KeyError, ValueError):
                continue

            # Accept only strict improvements, without simulated annealing.
            if candidate_objective < current_objective - _EPS:
                current_solution = candidate_solution
                current_objective = candidate_objective

        return current_solution.vehicle_pairs[0]

# ============================================================
# 原始文件: spd/online/replanner.py
# ============================================================

"""Online replanning orchestrator for T5 events."""





class OnlineReplanner:
    """Handle online event processing without rerunning full offline optimization."""

    def __init__(self, online_alns: LightweightALNSOptimizer):
        self._online_alns = online_alns

    def handle_event(
        self,
        instance: ProblemInstance,
        solution: Solution,
        event: CustomerNotHomeEvent,
        is_home: dict[int, bool],
        params: ProblemParameters,
        rng: random.Random | None = None,
    ) -> Solution:
        """Dispatch a T5 event and update remaining plan for one pair."""
        if event.service_mode == ServiceMode.DRONE:
            return self._handle_drone_event(instance, solution, event, is_home, params, rng)
        return self._handle_truck_event(instance, solution, event, is_home, params, rng)

    def resolve_event_sortie_index(self, solution: Solution, event: CustomerNotHomeEvent) -> int:
        """Resolve event sortie index.

        Priority:
        1. Use event.sortie_index when provided by the execution engine.
        2. Otherwise locate the sortie containing event.customer_id in the pair.
        """
        if event.sortie_index is not None:
            return event.sortie_index

        pair_solution = self._find_pair(solution, event.pair_id)
        for idx, sortie in enumerate(pair_solution.sorties):
            if event.customer_id in sortie.customers:
                return idx

        raise ValueError(f"customer {event.customer_id} not found in any sortie of pair {event.pair_id}")

    def _handle_drone_event(
        self,
        instance: ProblemInstance,
        solution: Solution,
        event: CustomerNotHomeEvent,
        is_home: dict[int, bool],
        params: ProblemParameters,
        rng: random.Random | None,
    ) -> Solution:
        """Case (a): event occurs during drone service for a specific sortie."""
        random_state = rng or random.Random(0)
        updated_solution = deepcopy(solution)

        # Step 1: reveal customer is not at home.
        is_home[event.customer_id] = False

        pair_index = self._find_pair_index(updated_solution, event.pair_id)
        pair_solution = updated_solution.vehicle_pairs[pair_index]

        try:
            sortie_index = self.resolve_event_sortie_index(updated_solution, event)
        except ValueError:
            # Sortie may have been removed by previous replanning.
            return updated_solution

        if sortie_index < 0 or sortie_index >= len(pair_solution.sorties):
            sortie_index = next(
                (
                    idx
                    for idx, sortie in enumerate(pair_solution.sorties)
                    if event.customer_id in sortie.customers
                ),
                -1,
            )
            if sortie_index < 0:
                return updated_solution

        # If stale index points to a different sortie, relocate by customer membership.
        if event.customer_id not in pair_solution.sorties[sortie_index].customers:
            relocated = next(
                (
                    idx
                    for idx, sortie in enumerate(pair_solution.sorties)
                    if event.customer_id in sortie.customers
                ),
                -1,
            )
            if relocated >= 0:
                sortie_index = relocated

        try:
            timeline = compute_truck_timeline(instance, pair_solution, is_home, params)
        except (AssertionError, KeyError, ValueError):
            return updated_solution

        # Capture executing sortie before repair: this reflects already visited
        # customers and launch position in the real executed path.
        current_sortie_before = pair_solution.sorties[sortie_index]
        launch_node_before = current_sortie_before.launch_node
        visited_prefix_before_event: list[int] = []
        if event.customer_id in current_sortie_before.customers:
            fail_index_before = current_sortie_before.customers.index(event.customer_id)
            visited_prefix_before_event = list(current_sortie_before.customers[: fail_index_before + 1])

        # Step 2: repair current sortie using B1/B2 + reorder.
        repaired = repair_sortie(
            instance,
            current_sortie_before,
            pair_solution,
            is_home,
            params,
            timeline,
            from_customer=event.customer_id,
        )
        pair_solution.sorties[sortie_index] = repaired
        self._refresh_pair_assignment(pair_solution, instance)

        frozen_customers: set[int] = set()
        truck_route = pair_solution.truck_route

        try:
            launch_pos = truck_route.index(launch_node_before)
        except ValueError:
            launch_pos = 0

        visited_truck_nodes = truck_route[: launch_pos + 1]
        visited_truck_set = set(visited_truck_nodes)

        # (1) Freeze traversed truck nodes (excluding depot).
        for node in visited_truck_nodes:
            if node != instance.depot_id:
                frozen_customers.add(node)

        # (2) Freeze customers in completed/started sorties whose launch is already passed.
        for idx, sortie in enumerate(pair_solution.sorties):
            if idx == sortie_index:
                continue
            if sortie.launch_node in visited_truck_set:
                frozen_customers.update(sortie.customers)

        # (3) Freeze visited prefix of current sortie before failed customer.
        frozen_customers.update(visited_prefix_before_event)

        # Step 4: optimize remaining path with lightweight ALNS.
        optimized_pair = self._optimize_with_freeze(
            instance,
            pair_solution,
            is_home,
            params,
            random_state,
            frozen_customers,
        )
        self._refresh_pair_assignment(optimized_pair, instance)
        updated_solution.vehicle_pairs[pair_index] = optimized_pair

        # Step 5: propagate and validate subsequent sortie feasibility.
        try:
            timeline_after = compute_truck_timeline(instance, optimized_pair, is_home, params)
            try:
                timeline_after = propagate_time_changes(
                    instance,
                    optimized_pair,
                    sortie_index,
                    is_home,
                    params,
                    timeline_after,
                )
            except NotImplementedError:
                # Fallback to the recomputed timeline when propagation utility is unavailable.
                pass

            load_state = compute_truck_load(instance, optimized_pair, is_home, params)
            for idx in range(sortie_index + 1, len(optimized_pair.sorties)):
                sortie = optimized_pair.sorties[idx]
                if check_sortie_feasibility(
                    instance,
                    optimized_pair,
                    sortie,
                    timeline_after,
                    load_state,
                    is_home,
                    params,
                ):
                    continue

                repaired_following = repair_sortie(
                    instance,
                    sortie,
                    optimized_pair,
                    is_home,
                    params,
                    timeline_after,
                )
                optimized_pair.sorties[idx] = repaired_following
                self._refresh_pair_assignment(optimized_pair, instance)

                timeline_after = compute_truck_timeline(instance, optimized_pair, is_home, params)
                load_state = compute_truck_load(instance, optimized_pair, is_home, params)
        except (AssertionError, KeyError, ValueError):
            pass

        updated_solution.unserved_customers = set(instance.customers) - updated_solution.all_customers
        return updated_solution

    def _handle_truck_event(
        self,
        instance: ProblemInstance,
        solution: Solution,
        event: CustomerNotHomeEvent,
        is_home: dict[int, bool],
        params: ProblemParameters,
        rng: random.Random | None,
    ) -> Solution:
        """Case (b): event occurs during truck service."""
        random_state = rng or random.Random(0)
        updated_solution = deepcopy(solution)

        # Mark the triggered customer as absent.
        is_home[event.customer_id] = False

        pair_index = self._find_pair_index(updated_solution, event.pair_id)
        pair_solution = updated_solution.vehicle_pairs[pair_index]

        # Recompute timeline/load and repair any newly infeasible sorties.
        try:
            timeline = compute_truck_timeline(instance, pair_solution, is_home, params)
            load_state = compute_truck_load(instance, pair_solution, is_home, params)
        except (AssertionError, KeyError, ValueError):
            timeline = None
            load_state = None

        if timeline is not None and load_state is not None:
            for idx, sortie in enumerate(pair_solution.sorties):
                if check_sortie_feasibility(
                    instance,
                    pair_solution,
                    sortie,
                    timeline,
                    load_state,
                    is_home,
                    params,
                ):
                    continue

                repaired = repair_sortie(
                    instance,
                    sortie,
                    pair_solution,
                    is_home,
                    params,
                    timeline,
                )
                pair_solution.sorties[idx] = repaired
                self._refresh_pair_assignment(pair_solution, instance)
                timeline = compute_truck_timeline(instance, pair_solution, is_home, params)
                load_state = compute_truck_load(instance, pair_solution, is_home, params)

        frozen_customers: set[int] = set()
        truck_route = pair_solution.truck_route
        try:
            event_pos = truck_route.index(event.customer_id)
        except ValueError:
            event_pos = 0

        visited_truck_nodes = truck_route[: event_pos + 1]
        visited_truck_set = set(visited_truck_nodes)

        # (1) Freeze traversed truck nodes (excluding depot).
        for node in visited_truck_nodes:
            if node != instance.depot_id:
                frozen_customers.add(node)

        # (2) Freeze customers in sorties already launched
        # (recovery passed or in-flight are both protected conservatively).
        for sortie in pair_solution.sorties:
            if sortie.launch_node in visited_truck_set:
                frozen_customers.update(sortie.customers)

        optimized_pair = self._optimize_with_freeze(
            instance,
            pair_solution,
            is_home,
            params,
            random_state,
            frozen_customers,
        )
        self._refresh_pair_assignment(optimized_pair, instance)
        updated_solution.vehicle_pairs[pair_index] = optimized_pair

        updated_solution.unserved_customers = set(instance.customers) - updated_solution.all_customers
        return updated_solution

    def _optimize_with_freeze(
        self,
        instance: ProblemInstance,
        pair_solution: VehiclePairSolution,
        is_home: dict[int, bool],
        params: ProblemParameters,
        random_state: random.Random,
        frozen_customers: set[int],
    ) -> VehiclePairSolution:
        """Call online optimizer with backward-compatible frozen-customer support."""
        optimize_fn = self._online_alns.optimize_remaining_path
        if "frozen_customers" in inspect.signature(optimize_fn).parameters:
            return optimize_fn(
                instance,
                pair_solution,
                is_home,
                params,
                random_state,
                frozen_customers=frozen_customers,
            )
        return optimize_fn(
            instance,
            pair_solution,
            is_home,
            params,
            random_state,
        )

    @staticmethod
    def _find_pair(solution: Solution, pair_id: int) -> VehiclePairSolution:
        """Return the pair solution by pair_id."""
        for pair in solution.vehicle_pairs:
            if pair.pair_id == pair_id:
                return pair
        raise ValueError(f"pair_id {pair_id} not found")

    @staticmethod
    def _find_pair_index(solution: Solution, pair_id: int) -> int:
        """Return pair index by pair_id."""
        for idx, pair in enumerate(solution.vehicle_pairs):
            if pair.pair_id == pair_id:
                return idx
        raise ValueError(f"pair_id {pair_id} not found")

    @staticmethod
    def _refresh_pair_assignment(pair_solution: VehiclePairSolution, instance: ProblemInstance) -> None:
        """Recompute truck/drone customer assignment from route and sorties."""
        drone_customers: set[int] = set()
        for sortie in pair_solution.sorties:
            drone_customers |= set(sortie.customers)

        pair_solution.drone_customers = drone_customers
        pair_solution.truck_customers = {
            node
            for node in pair_solution.truck_route
            if node != instance.depot_id
        }





