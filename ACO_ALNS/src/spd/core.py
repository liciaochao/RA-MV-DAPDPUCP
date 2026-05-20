from __future__ import annotations

from dataclasses import dataclass, field

from spd.config import ProblemParameters, map_time_to_slot
from spd.types import (
    ExecutionResult,
    HomeStatusMap,
    LoadState,
    ProblemInstance,
    Solution,
    Sortie,
    TimelineState,
    VehiclePairSolution,
)

# ============================================================
# 鍘熷鏂囦欢: spd/core/utils.py
# ============================================================

"""Shared utility helpers for core computation modules."""




def find_sortie_by_launch(sorties: list[Sortie], node: int) -> Sortie | None:
    """Find the first sortie with launch_node == node, or None if not found."""
    for sortie in sorties:
        if sortie.launch_node == node:
            return sortie
    return None

def find_sortie_by_recovery(sorties: list[Sortie], node: int) -> Sortie | None:
    """Find the first sortie with recovery_node == node, or None if not found."""
    for sortie in sorties:
        if sortie.recovery_node == node:
            return sortie
    return None


def get_arrival_time(
    customer_id: int,
    pair_solution: VehiclePairSolution,
    timeline: TimelineState,
) -> float:
    """Return arrival time by service mode: truck uses AT_T, drone uses AT_D."""
    in_truck = customer_id in pair_solution.truck_customers
    in_drone = customer_id in pair_solution.drone_customers

    if in_truck and in_drone:
        raise ValueError(f"customer {customer_id} appears in both truck and drone sets")
    if in_truck:
        return timeline.truck_arrival[customer_id]
    if in_drone:
        return timeline.drone_arrival[customer_id]

    raise KeyError(f"customer {customer_id} is not assigned to the given vehicle pair")

# ============================================================
# 鍘熷鏂囦欢: spd/core/timeline.py
# ============================================================

"""Timeline interfaces for truck-drone synchronized scheduling."""




def compute_drone_timeline(
    instance: ProblemInstance,
    sortie: Sortie,
    pair_solution: VehiclePairSolution,
    home_status: HomeStatusMap,
    params: ProblemParameters,
    timeline: TimelineState,
) -> None:
    """Populate drone arrival/departure times for one sortie.

    Contract:
    - This function updates the provided timeline object in place.
    - It is intended to be called from within compute_truck_timeline() when the
      truck loop reaches a sortie launch node.
    - The caller should rely on in-place mutation and not on a return value.
    """
    # Keep this parameter for signature stability; timeline logic does not use it.
    _ = pair_solution

    launch_node = sortie.launch_node
    if launch_node not in timeline.truck_departure:
        raise KeyError(f"truck departure time at launch node {launch_node} is required")

    # Drone starts when truck departs from launch node (DT_T[l(s)]).
    launch_departure = timeline.truck_departure[launch_node]
    timeline.drone_arrival[launch_node] = launch_departure

    prev_node = launch_node
    prev_depart = launch_departure

    for customer_id in sortie.customers:
        # Travel to next customer.
        travel_time = instance.drone_travel_time(
            prev_node,
            customer_id,
            params.vehicle.drone_speed,
        )
        arrival_time = prev_depart + travel_time
        timeline.drone_arrival[customer_id] = arrival_time

        # Apply time-window waiting and optional service time.
        customer = instance.customers[customer_id]
        earliest_start = customer.time_window[0]
        depart_base = max(arrival_time, earliest_start)

        if home_status[customer_id]:
            departure_time = depart_base + customer.service_time
        else:
            departure_time = depart_base

        timeline.drone_departure[customer_id] = departure_time
        prev_node = customer_id
        prev_depart = departure_time

    # Fly to recovery node after last served customer (or launch node if empty).
    recovery_node = sortie.recovery_node
    recovery_travel_time = instance.drone_travel_time(
        prev_node,
        recovery_node,
        params.vehicle.drone_speed,
    )
    timeline.drone_arrival[recovery_node] = prev_depart + recovery_travel_time


def compute_truck_timeline(
    instance: ProblemInstance,
    pair_solution: VehiclePairSolution,
    home_status: HomeStatusMap,
    params: ProblemParameters,
) -> TimelineState:
    """Compute synchronized truck and drone timeline for one vehicle pair.

    Required execution model from the specification:
    - Run a single forward loop on truck_route.
    - At each node apply node-level order: recovery wait/merge, customer service,
      then drone launch.
    - When a launch node is reached, call compute_drone_timeline() immediately so
      subsequent recovery nodes can merge truck and drone timelines consistently.
    """
    route = pair_solution.truck_route
    if not route:
        raise ValueError("truck_route must contain at least one node")

    timeline = TimelineState()
    depot_earliest = params.time.depot_earliest

    # Initialize depot start time.
    timeline.truck_arrival[route[0]] = depot_earliest
    timeline.truck_departure[route[0]] = depot_earliest

    for idx in range(1, len(route)):
        prev = route[idx - 1]
        curr = route[idx]

        # Compute truck arrival by travel from previous node.
        travel_time = instance.truck_travel_time(prev, curr, params.vehicle.truck_speed)
        truck_arrival = timeline.truck_departure[prev] + travel_time
        timeline.truck_arrival[curr] = truck_arrival

        if curr == instance.depot_id:
            # Return to depot has no service processing.
            timeline.truck_departure[curr] = truck_arrival
            continue

        # Step 1: recovery wait/merge.
        sortie_recv = find_sortie_by_recovery(pair_solution.sorties, curr)
        if sortie_recv is not None:
            if curr not in timeline.drone_arrival:
                raise KeyError(f"drone arrival time at recovery node {curr} is required")
            actual_available = max(truck_arrival, timeline.drone_arrival[curr])
        else:
            actual_available = truck_arrival

        # Step 2: truck service.
        customer = instance.customers[curr]
        earliest_start = customer.time_window[0]
        service_start = max(actual_available, earliest_start)

        if home_status[curr]:
            truck_departure = service_start + customer.service_time
        else:
            truck_departure = service_start
        timeline.truck_departure[curr] = truck_departure

        # Step 3: drone launch.
        sortie_launch = find_sortie_by_launch(pair_solution.sorties, curr)
        if sortie_launch is not None:
            compute_drone_timeline(
                instance,
                sortie_launch,
                pair_solution,
                home_status,
                params,
                timeline,
            )

    return timeline


def recompute_truck_timeline_from_node(
    instance: ProblemInstance,
    pair_solution: VehiclePairSolution,
    from_node: int,
    home_status: HomeStatusMap,
    params: ProblemParameters,
    timeline: TimelineState,
) -> TimelineState:
    """Recompute truck timeline suffix after an online event."""
    raise NotImplementedError


def propagate_time_changes(
    instance: ProblemInstance,
    pair_solution: VehiclePairSolution,
    from_sortie_index: int,
    home_status: HomeStatusMap,
    params: ProblemParameters,
    timeline: TimelineState,
) -> TimelineState:
    """Propagate local timeline updates to subsequent sorties."""
    raise NotImplementedError

# ============================================================
# 鍘熷鏂囦欢: spd/core/load_tracking.py
# ============================================================

"""Load-tracking interfaces for drone sorties and truck routes."""




def initial_sortie_load(instance: ProblemInstance, sortie: Sortie) -> float:
    """Return initial drone payload at sortie launch (delivery parcels only)."""
    # Initial payload contains only delivery parcels in this sortie.
    return sum(
        instance.customers[customer_id].weight
        for customer_id in sortie.customers
        if customer_id in instance.delivery_customers
    )


def update_sortie_load(load: float, customer_id: int, is_home: bool, instance: ProblemInstance) -> float:
    """Update drone load after visiting one customer."""
    # Customer not at home does not change payload.
    if not is_home:
        return load

    weight = instance.customers[customer_id].weight

    # Delivery decreases payload, pickup increases payload.
    if customer_id in instance.delivery_customers:
        return load - weight
    if customer_id in instance.pickup_customers:
        return load + weight

    raise KeyError(f"customer {customer_id} is not in delivery or pickup sets")


def check_sortie_load_feasibility(
    instance: ProblemInstance,
    sortie: Sortie,
    home_status: HomeStatusMap,
    params: ProblemParameters,
) -> bool:
    """Check per-node drone payload feasibility for a sortie."""
    capacity = params.vehicle.drone_capacity
    load = initial_sortie_load(instance, sortie)

    # Check launch payload feasibility.
    if load > capacity:
        return False

    # Check payload right after each customer service.
    for customer_id in sortie.customers:
        load = update_sortie_load(load, customer_id, home_status[customer_id], instance)
        if load > capacity:
            return False

    return True


def compute_final_drone_load(
    instance: ProblemInstance,
    sortie: Sortie,
    home_status: HomeStatusMap,
) -> float:
    """Compute remaining drone payload at recovery node."""
    load = initial_sortie_load(instance, sortie)

    # Replay customer sequence to get recovery payload.
    for customer_id in sortie.customers:
        load = update_sortie_load(load, customer_id, home_status[customer_id], instance)

    return load


def compute_truck_load(
    instance: ProblemInstance,
    pair_solution: VehiclePairSolution,
    home_status: HomeStatusMap,
    params: ProblemParameters,
) -> LoadState:
    """Track truck load across route with launch/recovery transfer logic.

    Required per-node order from the specification:
    1. recovery transfer from drone to truck,
    2. truck service at current customer,
    3. drone launch transfer from truck to drone.
    """
    route = pair_solution.truck_route
    if not route:
        raise ValueError("truck_route must contain at least one node")

    capacity = params.vehicle.truck_capacity
    load_state = LoadState()

    # Initial truck load includes all delivery parcels assigned to this pair.
    all_delivery = [
        customer_id
        for customer_id in pair_solution.all_customers
        if customer_id in instance.delivery_customers
    ]
    load = sum(instance.customers[customer_id].weight for customer_id in all_delivery)
    load_state.truck_load_at_node[route[0]] = load
    assert load <= capacity

    for curr in route[1:]:
        if curr == instance.depot_id:
            # Returning to depot only records current load.
            load_state.truck_load_at_node[curr] = load
            continue

        # Step 1: recovery transfer from drone to truck.
        recovery_sortie = find_sortie_by_recovery(pair_solution.sorties, curr)
        if recovery_sortie is not None:
            load += compute_final_drone_load(instance, recovery_sortie, home_status)
            assert load <= capacity

        # Step 2: truck service at current customer.
        if home_status[curr]:
            customer_weight = instance.customers[curr].weight
            if curr in instance.delivery_customers:
                load -= customer_weight
            elif curr in instance.pickup_customers:
                load += customer_weight
            else:
                raise KeyError(f"customer {curr} is not in delivery or pickup sets")
        assert load <= capacity

        # Step 3: drone launch transfer from truck to drone.
        launch_sortie = find_sortie_by_launch(pair_solution.sorties, curr)
        if launch_sortie is not None:
            drone_delivery_weight = sum(
                instance.customers[customer_id].weight
                for customer_id in launch_sortie.customers
                if customer_id in instance.delivery_customers
            )
            load -= drone_delivery_weight

        load_state.truck_load_at_node[curr] = load

    return load_state

# ============================================================
# 鍘熷鏂囦欢: spd/core/energy.py
# ============================================================

"""Energy model interfaces for sortie-level and solution-level costs."""




def compute_sortie_energy(
    instance: ProblemInstance,
    sortie: Sortie,
    timeline: TimelineState,
    home_status: HomeStatusMap,
    params: ProblemParameters,
) -> float:
    """Compute total drone energy consumption for one sortie."""
    eta = params.energy.eta_wh_per_kg_min
    drone_empty_weight = params.vehicle.drone_empty_weight
    drone_speed = params.vehicle.drone_speed

    customers = sortie.customers
    load = initial_sortie_load(instance, sortie)
    energy_total = 0.0

    if customers:
        # Stage 1 (launch segment): launch -> first customer.
        first_customer = customers[0]
        launch_fly_time = instance.drone_travel_time(sortie.launch_node, first_customer, drone_speed)
        energy_total += eta * (drone_empty_weight + load) * launch_fly_time

        for idx, customer_id in enumerate(customers):
            customer = instance.customers[customer_id]
            total_weight_before_service = drone_empty_weight + load

            # Stage 2: waiting for customer time window, always counted.
            wait_time = max(0.0, customer.time_window[0] - timeline.drone_arrival[customer_id])
            energy_total += eta * total_weight_before_service * wait_time

            # Stage 3: service energy, only if customer is at home.
            if home_status[customer_id]:
                energy_total += eta * total_weight_before_service * customer.service_time

            # Update payload after customer interaction.
            load = update_sortie_load(load, customer_id, home_status[customer_id], instance)
            total_weight_after_service = drone_empty_weight + load

            # Stage 1 (middle/recovery segments): fly to next node.
            next_node = customers[idx + 1] if idx < len(customers) - 1 else sortie.recovery_node
            fly_time = instance.drone_travel_time(customer_id, next_node, drone_speed)
            energy_total += eta * total_weight_after_service * fly_time
    else:
        # Degenerate case: no customer in sortie, fly directly to recovery.
        direct_fly_time = instance.drone_travel_time(sortie.launch_node, sortie.recovery_node, drone_speed)
        energy_total += eta * (drone_empty_weight + load) * direct_fly_time

    # Stage 4: hover at recovery while waiting for truck.
    hover_time = max(
        0.0,
        timeline.truck_arrival[sortie.recovery_node] - timeline.drone_arrival[sortie.recovery_node],
    )
    final_total_weight = drone_empty_weight + load
    energy_total += eta * final_total_weight * hover_time

    return energy_total


def estimate_remaining_energy(
    instance: ProblemInstance,
    sortie: Sortie,
    from_customer: int,
    timeline: TimelineState,
    home_status: HomeStatusMap,
    params: ProblemParameters,
) -> float:
    """Estimate remaining energy demand from a partial sortie state."""
    customers = sortie.customers
    if from_customer not in customers:
        raise ValueError(f"from_customer {from_customer} is not in sortie customers")

    eta = params.energy.eta_wh_per_kg_min
    drone_empty_weight = params.vehicle.drone_empty_weight
    drone_speed = params.vehicle.drone_speed

    from_index = customers.index(from_customer)

    # Reconstruct payload right after servicing from_customer.
    load = initial_sortie_load(instance, sortie)
    for customer_id in customers[: from_index + 1]:
        load = update_sortie_load(load, customer_id, home_status[customer_id], instance)

    energy_remaining = 0.0
    prev_node = from_customer

    # Replay all remaining customers after from_customer.
    for customer_id in customers[from_index + 1 :]:
        total_weight_before_service = drone_empty_weight + load

        # Stage 1: fly from previous node to this remaining customer.
        fly_time = instance.drone_travel_time(prev_node, customer_id, drone_speed)
        energy_remaining += eta * total_weight_before_service * fly_time

        customer = instance.customers[customer_id]

        # Stage 2: waiting for time window, always counted.
        wait_time = max(0.0, customer.time_window[0] - timeline.drone_arrival[customer_id])
        energy_remaining += eta * total_weight_before_service * wait_time

        # Stage 3: service energy if customer is at home.
        if home_status[customer_id]:
            energy_remaining += eta * total_weight_before_service * customer.service_time

        # Update payload and move cursor.
        load = update_sortie_load(load, customer_id, home_status[customer_id], instance)
        prev_node = customer_id

    # Stage 1 (tail): fly from last considered node to recovery.
    total_weight_before_recovery = drone_empty_weight + load
    tail_fly_time = instance.drone_travel_time(prev_node, sortie.recovery_node, drone_speed)
    energy_remaining += eta * total_weight_before_recovery * tail_fly_time

    # Stage 4: recovery hover.
    hover_time = max(
        0.0,
        timeline.truck_arrival[sortie.recovery_node] - timeline.drone_arrival[sortie.recovery_node],
    )
    energy_remaining += eta * total_weight_before_recovery * hover_time

    return energy_remaining


def compute_energy_cost(energy_wh: float, params: ProblemParameters) -> float:
    """Convert energy amount (Wh) into monetary cost."""
    return params.cost.drone_energy_cost_per_wh * energy_wh

# ============================================================
# 鍘熷鏂囦欢: spd/core/feasibility.py
# ============================================================

"""Hard/soft constraint checking interfaces."""



_EPS = 1e-9


def check_sortie_structure(pair_solution: VehiclePairSolution, sortie: Sortie) -> bool:
    """Validate sortie structure constraints (launch/recovery/customers)."""
    launch = sortie.launch_node
    recovery = sortie.recovery_node
    route = pair_solution.truck_route

    # Launch/recovery cannot be depot node 0 by model definition.
    if launch == 0 or recovery == 0:
        return False

    # Launch/recovery must be truck-served customers in this pair.
    if launch not in pair_solution.truck_customers or recovery not in pair_solution.truck_customers:
        return False

    # Launch/recovery must exist on truck route and preserve order.
    if launch not in route or recovery not in route:
        return False
    if route.index(launch) >= route.index(recovery):
        return False

    # Sortie customer list cannot include launch/recovery or any truck-route node.
    if launch in sortie.customers or recovery in sortie.customers:
        return False
    route_nodes = set(route)
    if any(customer_id in route_nodes for customer_id in sortie.customers):
        return False

    return True


def check_sortie_feasibility(
    instance: ProblemInstance,
    pair_solution: VehiclePairSolution,
    sortie: Sortie,
    timeline: TimelineState,
    load_state: LoadState,
    home_status: HomeStatusMap,
    params: ProblemParameters,
) -> bool:
    """Validate sortie-level hard constraints (F1-F5).

    F1: sortie energy does not exceed E_D.
    F2: drone payload never exceeds Q_D during sortie execution.
    F3: truck wait at recovery does not exceed W_max.
    F4: customer time-window arrival constraints for drone-served customers.
    F5: truck payload remains within Q_T, including post-recovery transfer.
    """
    if not check_sortie_structure(pair_solution, sortie):
        return False

    # Enforce sortie cardinality upper bound from constraint parameters.
    if len(sortie.customers) > params.constraints.max_customers_per_sortie:
        return False

    try:
        # F1: drone battery feasibility.
        sortie_energy = compute_sortie_energy(instance, sortie, timeline, home_status, params)
        if sortie_energy > params.energy.drone_battery_capacity + _EPS:
            return False

        # F2: drone payload feasibility at each sortie node.
        if not check_sortie_load_feasibility(instance, sortie, home_status, params):
            return False

        recovery = sortie.recovery_node

        # F3: truck waiting time for drone at recovery node.
        wait_truck = max(0.0, timeline.drone_arrival[recovery] - timeline.truck_arrival[recovery])
        if wait_truck > params.constraints.max_truck_wait_time + _EPS:
            return False

        # F4: drone-customer arrival must satisfy time-window upper bounds.
        for customer_id in sortie.customers:
            latest = instance.customers[customer_id].time_window[1]
            if timeline.drone_arrival[customer_id] > latest + _EPS:
                return False

        # F5: truck load at recovery (after transfer) must respect Q_T.
        if load_state.truck_load_at_node[recovery] > params.vehicle.truck_capacity + _EPS:
            return False
    except (AssertionError, KeyError, ValueError):
        return False

    return True


def is_truck_route_feasible(
    instance: ProblemInstance,
    pair_solution: VehiclePairSolution,
    timeline: TimelineState,
    load_state: LoadState,
    params: ProblemParameters,
) -> bool:
    """Validate truck-route hard constraints (F6-F8).

    F6: truck-customer time-window feasibility, AT_T(c_j) <= l_j.
    F7: truck payload feasibility, Load_T <= Q_T at all route states.
    F8: depot return-time feasibility, AT_T(depot) <= L_0.
    """
    route = pair_solution.truck_route
    if not route:
        return False

    try:
        # F6: truck arrivals at customer nodes must satisfy latest time windows.
        for node in route:
            if node == instance.depot_id:
                continue
            latest = instance.customers[node].time_window[1]
            if timeline.truck_arrival[node] > latest + _EPS:
                return False

        # F7: truck load must not exceed truck capacity at all route nodes.
        for node in route:
            if node not in load_state.truck_load_at_node:
                return False
            if load_state.truck_load_at_node[node] > params.vehicle.truck_capacity + _EPS:
                return False

        # F8: return to depot must be before depot latest time.
        if timeline.truck_arrival[route[-1]] > params.time.depot_latest + _EPS:
            return False
    except (AssertionError, KeyError, ValueError):
        return False

    return True


def check_sortie_sequence(pair: VehiclePairSolution, strict: bool = True) -> bool:
    """Validate launch/recovery ordering across adjacent sorties in one pair."""
    if not strict:
        return True

    route = list(pair.truck_route)
    sorties = pair.sorties

    for idx in range(len(sorties) - 1):
        recovery_idx_i = route.index(sorties[idx].recovery_node)
        launch_idx_next = route.index(sorties[idx + 1].launch_node)
        if launch_idx_next < recovery_idx_i:
            return False

    return True


def all_hard_constraints_satisfied(
    instance: ProblemInstance,
    solution: Solution,
    home_status: HomeStatusMap,
    params: ProblemParameters,
    enforce_sortie_sequence: bool = False,
) -> bool:
    """Global hard-constraint checker used by optimization loops."""
    for pair in solution.vehicle_pairs:
        try:
            timeline = compute_truck_timeline(instance, pair, home_status, params)
            load_state = compute_truck_load(instance, pair, home_status, params)
        except (AssertionError, KeyError, ValueError):
            return False

        for sortie in pair.sorties:
            if not check_sortie_structure(pair, sortie):
                return False
            if not check_sortie_feasibility(instance, pair, sortie, timeline, load_state, home_status, params):
                return False
        if not check_sortie_sequence(pair, strict=enforce_sortie_sequence):
            return False

        if not is_truck_route_feasible(instance, pair, timeline, load_state, params):
            return False

    return True
# ============================================================
# 鍘熷鏂囦欢: spd/core/objective.py
# ============================================================

"""Objective-value interfaces for offline expected cost and online actual cost."""




def failure_penalty(instance: ProblemInstance, customer_id: int, params: ProblemParameters) -> float:
    """Return Z_fail(i) = 2 * d_T(0, i) * c_T."""
    truck_distance = instance.truck_distance_km[(instance.depot_id, customer_id)]
    return 2.0 * truck_distance * params.cost.truck_cost_per_km


def compute_expected_failure_cost(
    instance: ProblemInstance,
    solution: Solution,
    params: ProblemParameters,
) -> float:
    """Compute expected failure term from p_i(t) and Z_fail(i)."""
    expected_cost = 0.0

    for pair in solution.vehicle_pairs:
        if not pair.all_customers:
            continue

        # Expected-failure evaluation uses planned timeline under all-home assumption.
        pair_home_status: dict[int, bool] = {customer_id: True for customer_id in pair.all_customers}
        timeline = compute_truck_timeline(instance, pair, pair_home_status, params)

        for customer_id in pair.all_customers:
            arrival_time = get_arrival_time(customer_id, pair, timeline)
            time_slot = map_time_to_slot(arrival_time, params.time)
            prob_home = instance.customers[customer_id].home_probabilities[time_slot - 1]
            expected_cost += (1.0 - prob_home) * failure_penalty(instance, customer_id, params)

    return expected_cost


def compute_objective(
    instance: ProblemInstance,
    solution: Solution,
    home_status: HomeStatusMap,
    params: ProblemParameters,
    time_window_penalty: float | None = None,
) -> float:
    """Compute full offline objective value for a candidate solution.

    Args:
        time_window_penalty: Optional beta override. When None, implementations
            should use params.beta_schedule.beta_init.
    """
    beta = params.beta_schedule.beta_init if time_window_penalty is None else time_window_penalty


    z_fixed = params.cost.fixed_pair_cost * len(solution.used_vehicle_pairs)
    z_truck = 0.0
    z_drone = 0.0
    z_tw_penalty = 0.0

    for pair in solution.used_vehicle_pairs:
        # Precompute timeline once per pair and reuse across terms.
        timeline = compute_truck_timeline(instance, pair, home_status, params)

        # Truck travel cost.
        for idx in range(len(pair.truck_route) - 1):
            i = pair.truck_route[idx]
            j = pair.truck_route[idx + 1]
            z_truck += params.cost.truck_cost_per_km * instance.truck_distance_km[(i, j)]

        # Drone sortie energy cost.
        for sortie in pair.sorties:
            energy = compute_sortie_energy(instance, sortie, timeline, home_status, params)
            z_drone += compute_energy_cost(energy, params)

        # Time-window soft penalty for all customers in this pair.
        for customer_id in pair.all_customers:
            arrival_time = get_arrival_time(customer_id, pair, timeline)
            latest = instance.customers[customer_id].time_window[1]
            z_tw_penalty += max(0.0, arrival_time - latest)

    z_fail_expected = compute_expected_failure_cost(instance, solution, params)
    z_tw_penalty *= beta

    # Penalty for unserved customers: ensures ALNS does not treat
    # "dropping customers" as an improvement.
    z_unserved_penalty = 0.0
    for customer_id in solution.unserved_customers:
        z_unserved_penalty += 2.0 * failure_penalty(instance, customer_id, params)

    return z_fixed + z_truck + z_drone + z_fail_expected + z_tw_penalty + z_unserved_penalty


def compute_objective_breakdown(
    instance: ProblemInstance,
    solution: Solution,
    home_status: HomeStatusMap,
    params: ProblemParameters,
    beta: float | None = None,
) -> dict[str, float]:
    """Return offline objective breakdown terms for analysis/visualization."""
    beta_value = params.beta_schedule.beta_init if beta is None else beta

    z_fixed = params.cost.fixed_pair_cost * len(solution.used_vehicle_pairs)
    z_truck = 0.0
    z_drone = 0.0
    z_tw_penalty = 0.0

    for pair in solution.used_vehicle_pairs:
        timeline = compute_truck_timeline(instance, pair, home_status, params)

        for idx in range(len(pair.truck_route) - 1):
            i = pair.truck_route[idx]
            j = pair.truck_route[idx + 1]
            z_truck += params.cost.truck_cost_per_km * instance.truck_distance_km[(i, j)]

        for sortie in pair.sorties:
            energy = compute_sortie_energy(instance, sortie, timeline, home_status, params)
            z_drone += compute_energy_cost(energy, params)

        for customer_id in pair.all_customers:
            arrival_time = get_arrival_time(customer_id, pair, timeline)
            latest = instance.customers[customer_id].time_window[1]
            z_tw_penalty += max(0.0, arrival_time - latest)

    z_fail_expected = compute_expected_failure_cost(instance, solution, params)
    z_tw_penalty *= beta_value

    z_unserved_penalty = 0.0
    for customer_id in solution.unserved_customers:
        z_unserved_penalty += 2.0 * failure_penalty(instance, customer_id, params)

    z_operational = z_fixed + z_truck + z_drone
    z_total = z_operational + z_fail_expected + z_tw_penalty + z_unserved_penalty
    return {
        "z_fixed": z_fixed,
        "z_truck": z_truck,
        "z_drone": z_drone,
        "z_fail_expected": z_fail_expected,
        "z_tw_penalty": z_tw_penalty,
        "z_unserved_penalty": z_unserved_penalty,
        "z_total": z_total,
        "z_operational": z_operational,
    }


def compute_actual_cost(execution: ExecutionResult, params: ProblemParameters) -> float:
    """Compute online realized cost from execution result."""
    z_fixed = params.cost.fixed_pair_cost * len(execution.solution.used_vehicle_pairs)
    z_truck_actual = params.cost.truck_cost_per_km * sum(execution.truck_distances_km)
    z_drone_actual = params.cost.drone_energy_cost_per_wh * sum(execution.drone_energies_wh)

    # ExecutionResult does not include instance distances for exact Z_fail(i), so
    # infer failed-customer total from provided actual_cost when available.
    base = z_fixed + z_truck_actual + z_drone_actual
    if execution.failed_customers:
        z_fail_total = max(0.0, execution.actual_cost - base)
    else:
        z_fail_total = 0.0

    return base + z_fail_total


def compute_actual_cost_breakdown(execution_result: ExecutionResult, params: ProblemParameters) -> dict[str, float]:
    """Return realized online cost breakdown terms for analysis/visualization."""
    fixed = params.cost.fixed_pair_cost * len(execution_result.solution.used_vehicle_pairs)
    truck = params.cost.truck_cost_per_km * sum(execution_result.truck_distances_km)
    drone = params.cost.drone_energy_cost_per_wh * sum(execution_result.drone_energies_wh)
    operational = fixed + truck + drone
    actual_total = float(execution_result.actual_cost)
    fail_penalty = max(0.0, actual_total - operational) if execution_result.failed_customers else 0.0
    return {
        "fixed": fixed,
        "truck": truck,
        "drone": drone,
        "fail_penalty": fail_penalty,
        "actual_total": actual_total,
        "operational": operational,
    }

# ============================================================
# 鍘熷鏂囦欢: spd/core/evaluation.py
# ============================================================

"""Orchestration entry points for one-shot solution evaluation."""





@dataclass(slots=True)
class PairEvaluation:
    """Evaluation summary for one vehicle pair."""

    pair_id: int
    objective_value: float
    hard_feasible: bool


@dataclass(slots=True)
class SolutionEvaluation:
    """Evaluation summary for a full solution."""

    objective_value: float
    hard_feasible: bool
    pair_evaluations: list[PairEvaluation] = field(default_factory=list)


def evaluate_solution(
    instance: ProblemInstance,
    solution: Solution,
    home_status: HomeStatusMap,
    params: ProblemParameters,
) -> SolutionEvaluation:
    """Evaluate one complete solution under given home-status assumptions."""
    pair_evaluations: list[PairEvaluation] = []
    hard_feasible = True

    for pair in solution.vehicle_pairs:
        pair_feasible = True
        pair_objective = float("inf")

        try:
            # Build state timelines and loads for this pair.
            timeline = compute_truck_timeline(instance, pair, home_status, params)
            load_state = compute_truck_load(instance, pair, home_status, params)

            # Validate each sortie first, then truck-route constraints.
            for sortie in pair.sorties:
                if not check_sortie_structure(pair, sortie):
                    pair_feasible = False
                    break
                if not check_sortie_feasibility(instance, pair, sortie, timeline, load_state, home_status, params):
                    pair_feasible = False
                    break

            if pair_feasible and not is_truck_route_feasible(instance, pair, timeline, load_state, params):
                pair_feasible = False

            # Compute pair objective contribution via a single-pair wrapper solution.
            pair_solution = Solution(vehicle_pairs=[pair], unserved_customers=set())
            pair_objective = compute_objective(instance, pair_solution, home_status, params)
        except (AssertionError, KeyError, ValueError):
            pair_feasible = False
            pair_objective = float("inf")

        if not pair_feasible:
            hard_feasible = False

        pair_evaluations.append(
            PairEvaluation(
                pair_id=pair.pair_id,
                objective_value=pair_objective,
                hard_feasible=pair_feasible,
            )
        )

    # Compute full solution objective if possible.
    try:
        objective_value = compute_objective(instance, solution, home_status, params)
    except (AssertionError, KeyError, ValueError):
        objective_value = float("inf")
        hard_feasible = False

    return SolutionEvaluation(
        objective_value=objective_value,
        hard_feasible=hard_feasible,
        pair_evaluations=pair_evaluations,
    )














