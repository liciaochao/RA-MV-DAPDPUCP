"""Constructive heuristic contract (E-TPRC)."""

from __future__ import annotations

import logging
import random
from itertools import combinations, permutations
from math import hypot

from spd.config import ProblemParameters, map_time_to_slot
from spd.core import all_hard_constraints_satisfied, check_sortie_feasibility, check_sortie_structure
from spd.core import compute_truck_load
from spd.core import compute_objective
from spd.core import compute_truck_timeline
from spd.types import HomeStatusMap
from spd.core import get_arrival_time
from spd.types import ProblemInstance, Solution, Sortie, VehiclePairSolution

_EPS = 1e-9


class ETPRCBuilder:
    """Build an initial feasible solution for offline optimization."""

    def build_initial_solution(
        self,
        instance: ProblemInstance,
        params: ProblemParameters,
        rng: random.Random | None = None,
    ) -> Solution:
        """Create an initial solution candidate using E-TPRC rules."""
        random_state = rng or random.Random(0)

        # Step 1: customer grouping.
        groups = self._customer_grouping(instance, params, random_state)

        # Offline stage assumes all customers are at home.
        home_status: dict[int, bool] = {
            customer_id: True for customer_id in instance.customers
        }

        # Step 2-4: route construction + sortie extraction per group.
        vehicle_pairs: list[VehiclePairSolution] = []
        skipped_in_route_build: set[int] = set()

        for pair_idx, group in enumerate(groups):
            truck_route, skipped_customers = self._build_truck_route(
                group, instance, params, return_skipped=True,
            )
            skipped_in_route_build |= set(skipped_customers)

            (
                updated_route,
                sorties,
                truck_customers,
                drone_customers,
            ) = self._extract_sorties(truck_route, instance, params, home_status)

            vehicle_pairs.append(
                VehiclePairSolution(
                    pair_id=pair_idx + 1,
                    truck_route=updated_route,
                    sorties=sorties,
                    truck_customers=truck_customers,
                    drone_customers=drone_customers,
                )
            )

        # ---- ????: ????????????? pair ??? ----
        if skipped_in_route_build:
            rescued: set[int] = set()

            for cid in sorted(
                skipped_in_route_build,
                key=lambda c: (
                    instance.customers[c].time_window[1]
                    - instance.customers[c].time_window[0]
                ),
                reverse=True,
            ):
                best_pair_idx: int | None = None
                best_new_route: list[int] | None = None
                best_cost: float = float("inf")

                customer = instance.customers[cid]
                tw_tightness = max(
                    customer.time_window[1] - customer.time_window[0], 0.01
                )

                for pidx, pair in enumerate(vehicle_pairs):
                    # ?? pair ? truck_route ?????????
                    # truck_route ??: [depot, ..., depot]
                    tr = pair.truck_route
                    for pos in range(1, len(tr)):
                        # ??????? depot ??
                        prev_node = tr[pos - 1]
                        next_node = tr[pos]

                        delta_dist = (
                            instance.truck_distance_km[(prev_node, cid)]
                            + instance.truck_distance_km[(cid, next_node)]
                            - instance.truck_distance_km[(prev_node, next_node)]
                        )
                        cost = (
                            params.etprc.omega_distance * delta_dist
                            + params.etprc.omega_time_window * (1.0 / tw_tightness)
                        )

                        candidate_route = tr[:pos] + [cid] + tr[pos:]

                        # ?????: ??? + ????
                        current_time = params.time.depot_earliest
                        feasible = True
                        for i in range(1, len(candidate_route)):
                            p = candidate_route[i - 1]
                            c = candidate_route[i]
                            current_time += instance.truck_travel_time(
                                p, c, params.vehicle.truck_speed
                            )
                            if c == instance.depot_id:
                                continue
                            cust = instance.customers[c]
                            if current_time > cust.time_window[1] + _EPS:
                                feasible = False
                                break
                            current_time = max(
                                current_time, cust.time_window[0]
                            ) + cust.service_time
                        if not feasible:
                            continue
                        if current_time > params.time.depot_latest + _EPS:
                            continue

                        if cost < best_cost:
                            best_cost = cost
                            best_pair_idx = pidx
                            best_new_route = candidate_route

                if best_pair_idx is not None and best_new_route is not None:
                    old_pair = vehicle_pairs[best_pair_idx]
                    # ????: ??? sortie, ??? truck_route ? truck_customers
                    # ???? sorties ??, ??????????????
                    # ?????? sortie ? launch/recovery ???????????
                    sorties_valid = True
                    for s in old_pair.sorties:
                        if s.launch_node not in best_new_route:
                            sorties_valid = False
                            break
                        if s.recovery_node not in best_new_route:
                            sorties_valid = False
                            break
                        li = best_new_route.index(s.launch_node)
                        ri = best_new_route.index(s.recovery_node)
                        if li >= ri:
                            sorties_valid = False
                            break

                    if not sorties_valid:
                        # sortie ?????, ???????
                        continue

                    new_truck_customers = set(old_pair.truck_customers) | {cid}
                    vehicle_pairs[best_pair_idx] = VehiclePairSolution(
                        pair_id=old_pair.pair_id,
                        truck_route=best_new_route,
                        sorties=list(old_pair.sorties),
                        truck_customers=new_truck_customers,
                        drone_customers=set(old_pair.drone_customers),
                    )
                    rescued.add(cid)
                    logging.info(
                        "Customer %d rescued via cross-group insertion "
                        "into pair %d",
                        cid,
                        old_pair.pair_id,
                    )

            skipped_in_route_build -= rescued

        # ---- ???? solution ----
        assigned_customers: set[int] = set()
        for pair in vehicle_pairs:
            assigned_customers |= pair.all_customers
        unserved = (set(instance.customers) - assigned_customers) | skipped_in_route_build

        solution = Solution(
            vehicle_pairs=vehicle_pairs,
            unserved_customers=unserved,
        )

        # Step 5: probability-aware post adjustment.
        solution = self._adjust_by_probability(solution, instance, params)

        # ---- ?????: ??????????? ----
        if solution.unserved_customers:
            raise ValueError(
                f"ETPRC failed to serve all customers. "
                f"{len(solution.unserved_customers)} unserved: "
                f"{sorted(solution.unserved_customers)}"
            )

        return solution

    def _adjust_by_probability(
        self,
        solution: Solution,
        instance: ProblemInstance,
        params: ProblemParameters,
    ) -> Solution:
        """Step 5: probability-aware local adjustments with feasibility rollback."""
        threshold = 0.3
        home_status: dict[int, bool] = {customer_id: True for customer_id in instance.customers}

        # Work on a copy to avoid mutating the caller's object on failed attempts.
        adjusted_pairs: list[VehiclePairSolution] = []
        for pair in solution.vehicle_pairs:
            copied_sorties = [
                Sortie(
                    launch_node=sortie.launch_node,
                    recovery_node=sortie.recovery_node,
                    customers=list(sortie.customers),
                )
                for sortie in pair.sorties
            ]
            adjusted_pairs.append(
                VehiclePairSolution(
                    pair_id=pair.pair_id,
                    truck_route=list(pair.truck_route),
                    sorties=copied_sorties,
                    truck_customers=set(pair.truck_customers),
                    drone_customers=set(pair.drone_customers),
                )
            )
        adjusted_solution = Solution(
            vehicle_pairs=adjusted_pairs,
            unserved_customers=set(solution.unserved_customers),
        )

        def try_replace_pair(pair_index: int, candidate_pair: VehiclePairSolution) -> bool:
            """Apply candidate pair only if full-solution hard feasibility holds."""
            current_pair = adjusted_solution.vehicle_pairs[pair_index]
            adjusted_solution.vehicle_pairs[pair_index] = candidate_pair
            if all_hard_constraints_satisfied(instance, adjusted_solution, home_status, params):
                return True
            adjusted_solution.vehicle_pairs[pair_index] = current_pair
            return False

        for pair_index, pair in enumerate(list(adjusted_solution.vehicle_pairs)):
            try:
                timeline = compute_truck_timeline(instance, pair, home_status, params)
            except (AssertionError, KeyError, ValueError):
                continue

            # Order by lower home probability first.
            customer_probs: list[tuple[float, int]] = []
            for customer_id in pair.all_customers:
                arrival = get_arrival_time(customer_id, pair, timeline)
                slot = map_time_to_slot(arrival, params.time)
                prob = instance.customers[customer_id].home_probabilities[slot - 1]
                customer_probs.append((prob, customer_id))
            customer_probs.sort(key=lambda item: item[0])

            for prob, customer_id in customer_probs:
                if prob >= threshold:
                    continue

                # Case A: truck-served low-probability customer; try route reinsertion.
                if customer_id in pair.truck_customers:
                    base_route = list(pair.truck_route)
                    if customer_id not in base_route:
                        continue
                    remove_idx = base_route.index(customer_id)
                    route_without = base_route[:remove_idx] + base_route[remove_idx + 1 :]

                    best_pair: VehiclePairSolution | None = None
                    best_prob = prob

                    for insert_idx in range(1, len(route_without)):
                        candidate_route = route_without[:insert_idx] + [customer_id] + route_without[insert_idx:]
                        candidate_pair = VehiclePairSolution(
                            pair_id=pair.pair_id,
                            truck_route=candidate_route,
                            sorties=[
                                Sortie(
                                    launch_node=s.launch_node,
                                    recovery_node=s.recovery_node,
                                    customers=list(s.customers),
                                )
                                for s in pair.sorties
                            ],
                            truck_customers=set(pair.truck_customers),
                            drone_customers=set(pair.drone_customers),
                        )
                        try:
                            candidate_timeline = compute_truck_timeline(instance, candidate_pair, home_status, params)
                            arrival = get_arrival_time(customer_id, candidate_pair, candidate_timeline)
                        except (AssertionError, KeyError, ValueError):
                            continue

                        slot = map_time_to_slot(arrival, params.time)
                        candidate_prob = instance.customers[customer_id].home_probabilities[slot - 1]
                        if candidate_prob > best_prob + _EPS:
                            best_prob = candidate_prob
                            best_pair = candidate_pair

                    if best_pair is not None and try_replace_pair(pair_index, best_pair):
                        pair = adjusted_solution.vehicle_pairs[pair_index]
                        continue

                # Case B: drone-served low-probability customer; try sortie reordering.
                if customer_id in pair.drone_customers:
                    for sortie_idx, sortie in enumerate(pair.sorties):
                        if customer_id not in sortie.customers:
                            continue

                        optimized_order = self._optimize_sortie_order(
                            customers=list(sortie.customers),
                            launch=sortie.launch_node,
                            recovery=sortie.recovery_node,
                            instance=instance,
                            params=params,
                            home_status=home_status,
                        )
                        if optimized_order == sortie.customers:
                            continue

                        candidate_sorties = [
                            Sortie(
                                launch_node=s.launch_node,
                                recovery_node=s.recovery_node,
                                customers=list(s.customers),
                            )
                            for s in pair.sorties
                        ]
                        candidate_sorties[sortie_idx].customers = list(optimized_order)
                        candidate_pair = VehiclePairSolution(
                            pair_id=pair.pair_id,
                            truck_route=list(pair.truck_route),
                            sorties=candidate_sorties,
                            truck_customers=set(pair.truck_customers),
                            drone_customers=set(pair.drone_customers),
                        )

                        if try_replace_pair(pair_index, candidate_pair):
                            pair = adjusted_solution.vehicle_pairs[pair_index]
                            break

        # Keep unserved set coherent with adjusted assignment.
        assigned_customers: set[int] = set()
        for pair in adjusted_solution.vehicle_pairs:
            assigned_customers |= pair.all_customers
        adjusted_solution.unserved_customers = set(instance.customers) - assigned_customers

        return adjusted_solution

    def _customer_grouping(
        self,
        instance: ProblemInstance,
        params: ProblemParameters,
        rng: random.Random | None,
    ) -> list[set[int]]:
        """Step 1: K-means grouping with delivery-load balancing."""
        customer_ids = list(instance.customers.keys())
        k = instance.vehicle_pair_count
        if k <= 0:
            raise ValueError("vehicle_pair_count must be positive")
        if not customer_ids:
            return [set() for _ in range(k)]

        random_state = rng or random.Random(0)

        def customer_coord(customer_id: int) -> tuple[float, float]:
            customer = instance.customers[customer_id]
            return (customer.x, customer.y)

        # Initialize centroids from existing customers.
        if k <= len(customer_ids):
            centroid_ids = random_state.sample(customer_ids, k)
        else:
            centroid_ids = list(customer_ids)
            while len(centroid_ids) < k:
                centroid_ids.append(random_state.choice(customer_ids))
        centroids = [customer_coord(customer_id) for customer_id in centroid_ids]

        assignments: list[list[int]] = [[] for _ in range(k)]
        max_iter = max(1, params.etprc.max_iter_kmeans)

        for _ in range(max_iter):
            assignments = [[] for _ in range(k)]

            # Assign each customer to the nearest centroid.
            for customer_id in customer_ids:
                x, y = customer_coord(customer_id)
                best_idx = min(
                    range(k),
                    key=lambda idx: (x - centroids[idx][0]) ** 2 + (y - centroids[idx][1]) ** 2,
                )
                assignments[best_idx].append(customer_id)

            # Repair empty clusters by moving one point from the largest cluster.
            empty_indices = [idx for idx, cluster in enumerate(assignments) if not cluster]
            for empty_idx in empty_indices:
                donor_idx = max(range(k), key=lambda idx: len(assignments[idx]))
                if len(assignments[donor_idx]) <= 1:
                    continue
                donor_centroid = centroids[donor_idx]
                moved_customer = max(
                    assignments[donor_idx],
                    key=lambda customer_id: hypot(
                        customer_coord(customer_id)[0] - donor_centroid[0],
                        customer_coord(customer_id)[1] - donor_centroid[1],
                    ),
                )
                assignments[donor_idx].remove(moved_customer)
                assignments[empty_idx].append(moved_customer)

            # Update centroids and check convergence.
            new_centroids: list[tuple[float, float]] = []
            for idx, cluster in enumerate(assignments):
                if not cluster:
                    new_centroids.append(centroids[idx])
                    continue
                mean_x = sum(customer_coord(customer_id)[0] for customer_id in cluster) / len(cluster)
                mean_y = sum(customer_coord(customer_id)[1] for customer_id in cluster) / len(cluster)
                new_centroids.append((mean_x, mean_y))

            shift = sum(
                hypot(new_centroids[idx][0] - centroids[idx][0], new_centroids[idx][1] - centroids[idx][1])
                for idx in range(k)
            )
            centroids = new_centroids
            if shift <= _EPS:
                break

        groups: list[set[int]] = [set(cluster) for cluster in assignments]

        def group_delivery_weight(group: set[int]) -> float:
            return sum(
                instance.customers[customer_id].weight
                for customer_id in group
                if customer_id in instance.delivery_customers
            )

        def group_centroid(group: set[int], default: tuple[float, float]) -> tuple[float, float]:
            if not group:
                return default
            mean_x = sum(instance.customers[c].x for c in group) / len(group)
            mean_y = sum(instance.customers[c].y for c in group) / len(group)
            return (mean_x, mean_y)

        # Balance overloaded groups by moving farthest delivery customers.
        truck_capacity = params.vehicle.truck_capacity
        changed = True
        while changed:
            changed = False
            for src_idx, src_group in enumerate(groups):
                if group_delivery_weight(src_group) <= truck_capacity + _EPS:
                    continue

                src_center = group_centroid(src_group, centroids[src_idx])
                delivery_candidates = [
                    customer_id
                    for customer_id in src_group
                    if customer_id in instance.delivery_customers and len(src_group) > 1
                ]
                if not delivery_candidates:
                    continue

                # Try farthest first to match the specification.
                delivery_candidates.sort(
                    key=lambda customer_id: hypot(
                        instance.customers[customer_id].x - src_center[0],
                        instance.customers[customer_id].y - src_center[1],
                    ),
                    reverse=True,
                )

                moved = False
                for candidate in delivery_candidates:
                    candidate_weight = instance.customers[candidate].weight
                    recipient_indices = [
                        idx
                        for idx, group in enumerate(groups)
                        if idx != src_idx and group_delivery_weight(group) + candidate_weight <= truck_capacity + _EPS
                    ]
                    if not recipient_indices:
                        continue

                    # Move to the nearest feasible group centroid.
                    target_idx = min(
                        recipient_indices,
                        key=lambda idx: hypot(
                            instance.customers[candidate].x - group_centroid(groups[idx], centroids[idx])[0],
                            instance.customers[candidate].y - group_centroid(groups[idx], centroids[idx])[1],
                        ),
                    )

                    src_group.remove(candidate)
                    groups[target_idx].add(candidate)
                    changed = True
                    moved = True
                    break

                if moved:
                    break

        return groups

    def _build_truck_route(
        self,
        group: set[int],
        instance: ProblemInstance,
        params: ProblemParameters,
        return_skipped: bool = False,
    ) -> list[int] | tuple[list[int], list[int]]:
        """Step 2: nearest-insertion truck route with load/time-window feasibility.

        Insertion strategy:
        1. Normal mode: pick the cheapest feasible insertion across all candidates.
        2. Deadlock fallback: when no candidate has a feasible position, switch to
           priority-inversion mode ? try inserting the tightest time-window
           customer first (it needs the route position most urgently).
        3. Skip: if priority-inversion also fails, skip the tightest-window
           customer and continue.
        4. Retry pass: after the main loop, retry all skipped customers on the
           now-complete route.
        """
        depot_id = instance.depot_id
        if not group:
            empty_route = [depot_id, depot_id]
            return (empty_route, []) if return_skipped else empty_route

        route: list[int] = [depot_id]
        uninserted = set(group)
        skipped_customers: list[int] = []

        omega_dist = params.etprc.omega_distance
        omega_tw = params.etprc.omega_time_window

        def is_route_feasible(partial_route: list[int]) -> bool:
            full_route = partial_route + [depot_id]
            load = sum(
                instance.customers[cid].weight
                for cid in group
                if cid in instance.delivery_customers
            )
            if load > params.vehicle.truck_capacity + _EPS:
                return False
            current_time = params.time.depot_earliest
            for idx in range(1, len(full_route)):
                prev = full_route[idx - 1]
                curr = full_route[idx]
                current_time += instance.truck_travel_time(
                    prev, curr, params.vehicle.truck_speed
                )
                if curr == depot_id:
                    continue
                customer = instance.customers[curr]
                earliest, latest = customer.time_window
                if current_time > latest + _EPS:
                    return False
                current_time = max(current_time, earliest) + customer.service_time
                if curr in instance.delivery_customers:
                    load -= customer.weight
                elif curr in instance.pickup_customers:
                    load += customer.weight
                else:
                    return False
                if load > params.vehicle.truck_capacity + _EPS:
                    return False
            if current_time > params.time.depot_latest + _EPS:
                return False
            return True

        def _try_insert_one(
            cid: int, current_route: list[int],
        ) -> tuple[list[int] | None, float]:
            """Try all positions for one customer. Return (best_route, best_cost)."""
            customer = instance.customers[cid]
            tw_tightness = max(
                customer.time_window[1] - customer.time_window[0], 0.01
            )
            best_rt: list[int] | None = None
            best_cost = float("inf")
            for pos in range(1, len(current_route) + 1):
                prev = current_route[pos - 1]
                nxt = (
                    current_route[pos]
                    if pos < len(current_route)
                    else depot_id
                )
                delta_dist = (
                    instance.truck_distance_km[(prev, cid)]
                    + instance.truck_distance_km[(cid, nxt)]
                    - instance.truck_distance_km[(prev, nxt)]
                )
                cost = omega_dist * delta_dist + omega_tw * (1.0 / tw_tightness)
                candidate = (
                    current_route[:pos] + [cid] + current_route[pos:]
                )
                if not is_route_feasible(candidate):
                    continue
                if cost < best_cost:
                    best_cost = cost
                    best_rt = candidate
            return best_rt, best_cost

        # ---- ??? ----
        while uninserted:
            # Normal mode: find cheapest feasible insertion across all candidates.
            best_customer: int | None = None
            best_route: list[int] | None = None
            best_cost: float | None = None

            for cid in uninserted:
                rt, cost = _try_insert_one(cid, route)
                if rt is not None and (best_cost is None or cost < best_cost):
                    best_cost = cost
                    best_customer = cid
                    best_route = rt

            if best_customer is not None:
                route = best_route
                uninserted.remove(best_customer)
                continue

            # Deadlock: no candidate is feasible. Try priority-inversion ?
            # insert the tightest time-window customer first.
            priority_order = sorted(
                uninserted,
                key=lambda cid: (
                    instance.customers[cid].time_window[1]
                    - instance.customers[cid].time_window[0]
                ),
            )
            inserted_by_priority = False
            for cid in priority_order:
                rt, _ = _try_insert_one(cid, route)
                if rt is not None:
                    route = rt
                    uninserted.remove(cid)
                    inserted_by_priority = True
                    break

            if inserted_by_priority:
                continue

            # Priority-inversion also failed. Skip the tightest-window customer.
            tightest = priority_order[0]
            logging.warning(
                "Customer %d skipped in truck route building "
                "(tightest time-window, tw=%.1f): "
                "no feasible insertion found",
                tightest,
                instance.customers[tightest].time_window[1]
                - instance.customers[tightest].time_window[0],
            )
            skipped_customers.append(tightest)
            uninserted.remove(tightest)

        # ---- ???????????? ----
        if skipped_customers:
            still_skipped: list[int] = []
            retry_order = sorted(
                skipped_customers,
                key=lambda cid: (
                    instance.customers[cid].time_window[1]
                    - instance.customers[cid].time_window[0]
                ),
                reverse=True,
            )
            for cid in retry_order:
                rt, _ = _try_insert_one(cid, route)
                if rt is not None:
                    route = rt
                    logging.info("Customer %d re-inserted in retry pass", cid)
                else:
                    still_skipped.append(cid)
            skipped_customers = still_skipped

        route.append(depot_id)
        if return_skipped:
            return route, skipped_customers
        return route

    def _extract_sorties(
        self,
        truck_route: list[int],
        instance: ProblemInstance,
        params: ProblemParameters,
        home_status: HomeStatusMap,
    ) -> tuple[list[int], list[Sortie], set[int], set[int]]:
        """Step 3: greedily extract feasible sorties from an all-truck route."""
        depot_id = instance.depot_id
        route = list(truck_route)

        # Baseline pair with no sortie extraction.
        baseline_pair = VehiclePairSolution(
            pair_id=0,
            truck_route=list(route),
            sorties=[],
            truck_customers={node for node in route if node != depot_id},
            drone_customers=set(),
        )
        baseline_solution = Solution(vehicle_pairs=[baseline_pair], unserved_customers=set())
        baseline_objective = compute_objective(instance, baseline_solution, home_status, params)

        candidate_pool: list[dict[str, object]] = []
        max_sortie_size = params.constraints.max_customers_per_sortie
        drone_capacity = params.vehicle.drone_capacity

        # Enumerate all possible launch/recovery index pairs.
        for launch_idx in range(1, len(route) - 2):
            launch_node = route[launch_idx]
            if launch_node == depot_id:
                continue

            for recovery_idx in range(launch_idx + 1, len(route) - 1):
                recovery_node = route[recovery_idx]
                if recovery_node == depot_id:
                    continue

                # Candidate nodes between launch and recovery, excluding endpoints.
                between_nodes = [
                    node
                    for node in route[launch_idx + 1 : recovery_idx]
                    if node != depot_id
                ]
                feasible_nodes = [
                    node
                    for node in between_nodes
                    if instance.customers[node].weight <= drone_capacity + _EPS
                ]
                if not feasible_nodes:
                    continue

                for subset_size in range(1, min(len(feasible_nodes), max_sortie_size) + 1):
                    for subset in combinations(feasible_nodes, subset_size):
                        subset_set = set(subset)
                        ordered_customers = self._optimize_sortie_order(
                            list(subset),
                            launch_node,
                            recovery_node,
                            instance,
                            params,
                            home_status,
                        )
                        if set(ordered_customers) != subset_set:
                            continue

                        sortie = Sortie(
                            launch_node=launch_node,
                            recovery_node=recovery_node,
                            customers=list(ordered_customers),
                        )

                        # Remove selected drone customers from truck route.
                        candidate_route = [
                            node for node in route if node == depot_id or node not in subset_set
                        ]
                        candidate_pair = VehiclePairSolution(
                            pair_id=0,
                            truck_route=candidate_route,
                            sorties=[sortie],
                            truck_customers={node for node in candidate_route if node != depot_id},
                            drone_customers=set(subset_set),
                        )

                        if not check_sortie_structure(candidate_pair, sortie):
                            continue

                        try:
                            timeline = compute_truck_timeline(instance, candidate_pair, home_status, params)
                            load_state = compute_truck_load(instance, candidate_pair, home_status, params)
                        except (AssertionError, KeyError, ValueError):
                            continue

                        if not check_sortie_feasibility(
                            instance,
                            candidate_pair,
                            sortie,
                            timeline,
                            load_state,
                            home_status,
                            params,
                        ):
                            continue

                        candidate_solution = Solution(vehicle_pairs=[candidate_pair], unserved_customers=set())
                        try:
                            candidate_objective = compute_objective(
                                instance,
                                candidate_solution,
                                home_status,
                                params,
                            )
                        except (AssertionError, KeyError, ValueError):
                            continue

                        delta_z = baseline_objective - candidate_objective
                        candidate_pool.append(
                            {
                                "delta": delta_z,
                                "sortie": sortie,
                                "customers": subset_set,
                            }
                        )

        # Greedy select non-conflicting sorties by best delta first.
        candidate_pool.sort(key=lambda item: float(item["delta"]), reverse=True)

        selected_sorties: list[Sortie] = []
        selected_drone_customers: set[int] = set()
        selected_anchor_nodes: set[int] = set()
        selected_launch_nodes: set[int] = set()
        selected_recovery_nodes: set[int] = set()
        updated_route = list(route)

        for candidate in candidate_pool:
            sortie = candidate["sortie"]
            customers = set(candidate["customers"])
            if not isinstance(sortie, Sortie):
                continue
            if float(candidate["delta"]) <= _EPS:
                continue

            # Conflict rule: each customer can be extracted at most once.
            if customers & selected_drone_customers:
                continue

            # Keep launch/recovery anchors stable for already selected sorties.
            if customers & selected_anchor_nodes:
                continue
            if sortie.launch_node in selected_drone_customers or sortie.recovery_node in selected_drone_customers:
                continue

            # Timeline utilities assume one launch and one recovery sortie per node.
            if sortie.launch_node in selected_launch_nodes or sortie.recovery_node in selected_recovery_nodes:
                continue

            # Ensure launch/recovery still exist in current route order.
            if sortie.launch_node not in updated_route or sortie.recovery_node not in updated_route:
                continue
            if updated_route.index(sortie.launch_node) >= updated_route.index(sortie.recovery_node):
                continue

            selected_sorties.append(sortie)
            selected_drone_customers |= customers
            selected_anchor_nodes.add(sortie.launch_node)
            selected_anchor_nodes.add(sortie.recovery_node)
            selected_launch_nodes.add(sortie.launch_node)
            selected_recovery_nodes.add(sortie.recovery_node)
            updated_route = [
                node for node in updated_route if node == depot_id or node not in customers
            ]

        # Sort selected sorties by launch order on updated route.
        selected_sorties.sort(
            key=lambda sortie: (
                updated_route.index(sortie.launch_node),
                updated_route.index(sortie.recovery_node),
            )
        )
        truck_customers = {node for node in updated_route if node != depot_id}
        drone_customers = set(selected_drone_customers)

        return updated_route, selected_sorties, truck_customers, drone_customers

    def _optimize_sortie_order(
        self,
        customers: list[int],
        launch: int,
        recovery: int,
        instance: ProblemInstance,
        params: ProblemParameters,
        home_status: HomeStatusMap,
    ) -> list[int]:
        """Step 4: enumerate all customer permutations and pick the best one."""
        if len(customers) <= 1:
            return list(customers)

        depot_id = instance.depot_id
        best_order: list[int] | None = None
        best_objective: float | None = None

        for order in permutations(customers):
            sortie = Sortie(
                launch_node=launch,
                recovery_node=recovery,
                customers=list(order),
            )
            pair = VehiclePairSolution(
                pair_id=0,
                truck_route=[depot_id, launch, recovery, depot_id],
                sorties=[sortie],
                truck_customers={launch, recovery},
                drone_customers=set(customers),
            )

            if not check_sortie_structure(pair, sortie):
                continue

            try:
                timeline = compute_truck_timeline(instance, pair, home_status, params)
                load_state = compute_truck_load(instance, pair, home_status, params)
            except (AssertionError, KeyError, ValueError):
                continue

            if not check_sortie_feasibility(
                instance,
                pair,
                sortie,
                timeline,
                load_state,
                home_status,
                params,
            ):
                continue

            candidate_solution = Solution(vehicle_pairs=[pair], unserved_customers=set())
            try:
                objective = compute_objective(instance, candidate_solution, home_status, params)
            except (AssertionError, KeyError, ValueError):
                continue

            if best_objective is None or objective < best_objective:
                best_objective = objective
                best_order = list(order)

        # If no feasible permutation is found, keep deterministic original order.
        return best_order if best_order is not None else list(customers)










