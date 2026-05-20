from __future__ import annotations

import logging
import math
import multiprocessing as mp
import random
from copy import deepcopy
from dataclasses import dataclass, field
from math import hypot
from typing import Protocol

from spd.config import ACOParameters, ProblemParameters, map_time_to_slot
from spd.core import (
    all_hard_constraints_satisfied,
    compute_objective,
    compute_truck_timeline,
    compute_truck_load,
    compute_sortie_energy,
    check_sortie_structure,
    check_sortie_sequence,
    check_sortie_load_feasibility,
    get_arrival_time,
)
from spd.etprc import ETPRCBuilder
from spd.types import HomeStatusMap, ProblemInstance, Solution, Sortie, VehiclePairSolution

_ACO_PARALLEL_OVERRIDES: dict[int, bool] = {}
_ORIG_ACO_INIT = ACOParameters.__init__


def _patched_aco_init(self, *args, **kwargs):
    """Backward-compatible init wrapper that accepts optional aco.parallel_ants."""
    parallel_ants = kwargs.pop("parallel_ants", None)
    _ORIG_ACO_INIT(self, *args, **kwargs)
    if parallel_ants is not None:
        _ACO_PARALLEL_OVERRIDES[id(self)] = bool(parallel_ants)


if "parallel_ants" not in _ORIG_ACO_INIT.__code__.co_varnames:
    ACOParameters.__init__ = _patched_aco_init

# ============================================================
# 原始文件: spd/offline/operators/base.py
# ============================================================

"""Operator protocols and payload models for ALNS."""





@dataclass(slots=True)
class DestroyResult:
    """Output of a destroy step."""

    partial_solution: Solution
    removed_customers: list[int]


class DestroyOperator(Protocol):
    """Protocol for ALNS destroy operators."""

    name: str

    def __call__(
        self,
        solution: Solution,
        instance: ProblemInstance,
        params: ProblemParameters,
        remove_count: int,
        home_status: HomeStatusMap,
        rng: random.Random,
    ) -> DestroyResult:
        ...


class RepairOperator(Protocol):
    """Protocol for ALNS repair operators."""

    name: str

    def __call__(
        self,
        partial_solution: Solution,
        removed_customers: list[int],
        instance: ProblemInstance,
        params: ProblemParameters,
        home_status: HomeStatusMap,
        rng: random.Random,
    ) -> Solution:
        ...


class CrossGroupOperator(Protocol):
    """Protocol for low-frequency cross-pair operators."""

    name: str

    def __call__(
        self,
        solution: Solution,
        instance: ProblemInstance,
        params: ProblemParameters,
        home_status: HomeStatusMap,
        rng: random.Random,
    ) -> Solution:
        ...

# ============================================================
# 原始文件: spd/offline/operators/destroy.py
# ============================================================

"""Destroy operator implementations (D1-D4)."""





def _normalized_remove_count(total: int, requested: int) -> int:
    """Clamp requested removal count into [0, total]."""
    return max(0, min(total, requested))


def _remove_customer_in_place(solution: Solution, customer_id: int) -> None:
    """Remove one customer from truck/drone assignment in a mutable solution."""
    for pair in solution.vehicle_pairs:
        # Remove from truck assignment and route if present.
        if customer_id in pair.truck_customers:
            pair.truck_customers.remove(customer_id)
            pair.truck_route = [node for node in pair.truck_route if node != customer_id]

        # Remove from drone assignment and sortie customers if present.
        if customer_id in pair.drone_customers:
            pair.drone_customers.remove(customer_id)
            kept_sorties = []
            for sortie in pair.sorties:
                if customer_id in sortie.customers:
                    sortie.customers = [c for c in sortie.customers if c != customer_id]
                if sortie.customers:
                    kept_sorties.append(sortie)
            pair.sorties = kept_sorties

    # Removed customers become unserved in partial solution.
    solution.unserved_customers.add(customer_id)


def _build_partial_solution(solution: Solution, removed_customers: list[int]) -> Solution:
    """Create deep-copied partial solution after removing selected customers."""
    partial = deepcopy(solution)
    for customer_id in removed_customers:
        _remove_customer_in_place(partial, customer_id)
    return partial


def d1_random_removal(
    solution: Solution,
    instance: ProblemInstance,
    params: ProblemParameters,
    remove_count: int,
    home_status: HomeStatusMap,
    rng: random.Random,
) -> DestroyResult:
    """D1: remove random customers."""
    _ = instance, params, home_status

    customers = sorted(solution.all_customers)
    k = _normalized_remove_count(len(customers), remove_count)
    removed_customers = rng.sample(customers, k) if k > 0 else []

    partial_solution = _build_partial_solution(solution, removed_customers)
    return DestroyResult(partial_solution=partial_solution, removed_customers=removed_customers)


def d2_worst_removal(
    solution: Solution,
    instance: ProblemInstance,
    params: ProblemParameters,
    remove_count: int,
    home_status: HomeStatusMap,
    rng: random.Random,
) -> DestroyResult:
    """D2: remove customers with largest objective contribution."""
    _ = rng

    customers = sorted(solution.all_customers)
    k = _normalized_remove_count(len(customers), remove_count)
    if k == 0:
        return DestroyResult(partial_solution=deepcopy(solution), removed_customers=[])

    # Baseline objective under current state.
    baseline = compute_objective(instance, solution, home_status, params)

    improvements: list[tuple[float, int]] = []
    for customer_id in customers:
        candidate_solution = _build_partial_solution(solution, [customer_id])
        try:
            candidate_obj = compute_objective(instance, candidate_solution, home_status, params)
            improvement = baseline - candidate_obj
        except (AssertionError, KeyError, ValueError):
            # Infeasible candidate is treated as no improvement.
            improvement = float("-inf")
        improvements.append((improvement, customer_id))

    improvements.sort(key=lambda item: (-item[0], item[1]))
    removed_customers = [customer_id for _, customer_id in improvements[:k]]

    partial_solution = _build_partial_solution(solution, removed_customers)
    return DestroyResult(partial_solution=partial_solution, removed_customers=removed_customers)


def d3_related_removal(
    solution: Solution,
    instance: ProblemInstance,
    params: ProblemParameters,
    remove_count: int,
    home_status: HomeStatusMap,
    rng: random.Random,
) -> DestroyResult:
    """D3: related removal for geographically or structurally similar customers."""
    _ = params, home_status

    customers = sorted(solution.all_customers)
    k = _normalized_remove_count(len(customers), remove_count)
    if k == 0:
        return DestroyResult(partial_solution=deepcopy(solution), removed_customers=[])

    seed = rng.choice(customers)

    # Rank by geometric proximity to seed.
    sx = instance.customers[seed].x
    sy = instance.customers[seed].y
    others = [customer_id for customer_id in customers if customer_id != seed]
    others.sort(
        key=lambda customer_id: (
            hypot(instance.customers[customer_id].x - sx, instance.customers[customer_id].y - sy),
            customer_id,
        )
    )

    removed_customers = [seed] + others[: max(0, k - 1)]

    partial_solution = _build_partial_solution(solution, removed_customers)
    return DestroyResult(partial_solution=partial_solution, removed_customers=removed_customers)


def d4_low_home_probability_removal(
    solution: Solution,
    instance: ProblemInstance,
    params: ProblemParameters,
    remove_count: int,
    home_status: HomeStatusMap,
    rng: random.Random,
) -> DestroyResult:
    """D4: remove customers with low home probability at planned arrival slots."""
    _ = rng

    customers = sorted(solution.all_customers)
    k = _normalized_remove_count(len(customers), remove_count)
    if k == 0:
        return DestroyResult(partial_solution=deepcopy(solution), removed_customers=[])

    customer_probs: dict[int, float] = {}

    for pair in solution.vehicle_pairs:
        if not pair.all_customers:
            continue
        try:
            timeline = compute_truck_timeline(instance, pair, home_status, params)
        except (AssertionError, KeyError, ValueError):
            continue

        for customer_id in pair.all_customers:
            try:
                arrival_time = get_arrival_time(customer_id, pair, timeline)
                time_slot = map_time_to_slot(arrival_time, params.time)
                prob = instance.customers[customer_id].home_probabilities[time_slot - 1]
            except (AssertionError, KeyError, ValueError, IndexError):
                prob = 1.0
            customer_probs[customer_id] = prob

    # Fallback for any missing customers.
    for customer_id in customers:
        customer_probs.setdefault(customer_id, 1.0)

    ranked = sorted(customers, key=lambda customer_id: (customer_probs[customer_id], customer_id))
    removed_customers = ranked[:k]

    partial_solution = _build_partial_solution(solution, removed_customers)
    return DestroyResult(partial_solution=partial_solution, removed_customers=removed_customers)


def d3_route_cluster_removal(
    solution: Solution,
    instance: ProblemInstance,
    params: ProblemParameters,
    remove_count: int,
    home_status: HomeStatusMap,
    rng: random.Random,
) -> DestroyResult:
    """Backward-compatible alias for previous D3 naming."""
    return d3_related_removal(solution, instance, params, remove_count, home_status, rng)

# ============================================================
# 原始文件: spd/offline/operators/repair.py
# ============================================================

"""Repair operator implementations (R1-R3)."""





def _prepare_customer_assignment(pair: VehiclePairSolution, customer_id: int) -> None:
    """Remove customer from all current truck/drone assignments in one pair."""
    if customer_id in pair.truck_customers:
        pair.truck_customers.remove(customer_id)
    if customer_id in pair.drone_customers:
        pair.drone_customers.remove(customer_id)

    pair.truck_route = [node for node in pair.truck_route if node != customer_id]

    kept_sorties: list[Sortie] = []
    for sortie in pair.sorties:
        sortie.customers = [customer for customer in sortie.customers if customer != customer_id]
        if sortie.customers:
            kept_sorties.append(sortie)
    pair.sorties = kept_sorties


def _customer_probability_at_arrival(
    solution: Solution,
    customer_id: int,
    instance: ProblemInstance,
    params: ProblemParameters,
    home_status: HomeStatusMap,
) -> float:
    """Return p_i(t*) for the customer in a candidate solution."""
    for pair in solution.vehicle_pairs:
        if customer_id not in pair.all_customers:
            continue
        timeline = compute_truck_timeline(instance, pair, home_status, params)
        arrival_time = get_arrival_time(customer_id, pair, timeline)
        slot = map_time_to_slot(arrival_time, params.time)
        return instance.customers[customer_id].home_probabilities[slot - 1]
    return 0.0


def _candidate_solutions_for_customer(
    solution: Solution,
    customer_id: int,
    instance: ProblemInstance,
    params: ProblemParameters,
    home_status: HomeStatusMap,
) -> list[tuple[Solution, float, float]]:
    """Enumerate feasible insertion candidates.

    Returns tuples: (candidate_solution, objective_value, home_probability_at_arrival).
    """
    candidates: list[tuple[Solution, float, float]] = []
    depot_id = instance.depot_id

    for pair_idx, pair in enumerate(solution.vehicle_pairs):
        # Truck insertion candidates: all route insertion positions except before first depot.
        for insert_pos in range(1, len(pair.truck_route)):
            candidate = deepcopy(solution)
            cpair = candidate.vehicle_pairs[pair_idx]

            _prepare_customer_assignment(cpair, customer_id)
            cpair.truck_route = cpair.truck_route[:insert_pos] + [customer_id] + cpair.truck_route[insert_pos:]
            cpair.truck_customers.add(customer_id)
            cpair.drone_customers.discard(customer_id)
            candidate.unserved_customers.discard(customer_id)

            try:
                if not all_hard_constraints_satisfied(instance, candidate, home_status, params, enforce_sortie_sequence=True):
                    continue
                objective = compute_objective(instance, candidate, home_status, params)
                prob = _customer_probability_at_arrival(candidate, customer_id, instance, params, home_status)
            except (AssertionError, KeyError, ValueError):
                continue
            candidates.append((candidate, objective, prob))

        # Drone insertion into existing sorties.
        for sortie_idx, sortie in enumerate(pair.sorties):
            for insert_pos in range(len(sortie.customers) + 1):
                candidate = deepcopy(solution)
                cpair = candidate.vehicle_pairs[pair_idx]

                _prepare_customer_assignment(cpair, customer_id)
                cpair.drone_customers.add(customer_id)

                csortie = cpair.sorties[sortie_idx]
                csortie.customers = csortie.customers[:insert_pos] + [customer_id] + csortie.customers[insert_pos:]
                candidate.unserved_customers.discard(customer_id)

                try:
                    if not all_hard_constraints_satisfied(instance, candidate, home_status, params, enforce_sortie_sequence=True):
                        continue
                    objective = compute_objective(instance, candidate, home_status, params)
                    prob = _customer_probability_at_arrival(candidate, customer_id, instance, params, home_status)
                except (AssertionError, KeyError, ValueError):
                    continue
                candidates.append((candidate, objective, prob))

        # Drone insertion by creating a new sortie on existing truck route nodes.
        route_nodes = [node for node in pair.truck_route if node != depot_id]
        for launch_idx in range(len(route_nodes)):
            for recovery_idx in range(launch_idx + 1, len(route_nodes)):
                launch = route_nodes[launch_idx]
                recovery = route_nodes[recovery_idx]

                candidate = deepcopy(solution)
                cpair = candidate.vehicle_pairs[pair_idx]

                _prepare_customer_assignment(cpair, customer_id)
                cpair.drone_customers.add(customer_id)
                cpair.sorties.append(Sortie(launch_node=launch, recovery_node=recovery, customers=[customer_id]))
                candidate.unserved_customers.discard(customer_id)

                try:
                    if not all_hard_constraints_satisfied(instance, candidate, home_status, params, enforce_sortie_sequence=True):
                        continue
                    objective = compute_objective(instance, candidate, home_status, params)
                    prob = _customer_probability_at_arrival(candidate, customer_id, instance, params, home_status)
                except (AssertionError, KeyError, ValueError):
                    continue
                candidates.append((candidate, objective, prob))

    return candidates


def r1_greedy_insertion(
    partial_solution: Solution,
    removed_customers: list[int],
    instance: ProblemInstance,
    params: ProblemParameters,
    home_status: HomeStatusMap,
    rng: random.Random,
) -> Solution:
    """R1: greedily insert customers by minimum objective delta."""
    _ = rng

    repaired = deepcopy(partial_solution)

    # Retry queue: a customer may become insertable only after others are inserted.
    pending = list(removed_customers)
    no_progress_rounds = 0

    while pending and no_progress_rounds < len(pending):
        customer_id = pending.pop(0)
        candidates = _candidate_solutions_for_customer(repaired, customer_id, instance, params, home_status)
        if not candidates:
            pending.append(customer_id)
            no_progress_rounds += 1
            continue

        # Choose insertion with minimum objective (minimum delta from current state).
        best_solution, _, _ = min(candidates, key=lambda item: item[1])
        repaired = best_solution
        no_progress_rounds = 0

    return repaired


def r2_regret_insertion(
    partial_solution: Solution,
    removed_customers: list[int],
    instance: ProblemInstance,
    params: ProblemParameters,
    home_status: HomeStatusMap,
    rng: random.Random,
) -> Solution:
    """R2: regret-k insertion."""
    _ = rng

    repaired = deepcopy(partial_solution)
    pending = list(removed_customers)

    while pending:
        best_pick: tuple[float, int, Solution] | None = None

        for customer_id in pending:
            candidates = _candidate_solutions_for_customer(repaired, customer_id, instance, params, home_status)
            if not candidates:
                continue

            candidates.sort(key=lambda item: item[1])
            best_obj = candidates[0][1]
            second_obj = candidates[1][1] if len(candidates) > 1 else float("inf")
            regret = second_obj - best_obj

            if best_pick is None or regret > best_pick[0]:
                best_pick = (regret, customer_id, candidates[0][0])

        if best_pick is None:
            break

        _, chosen_customer, chosen_solution = best_pick
        repaired = chosen_solution
        pending.remove(chosen_customer)

    return repaired


def r3_timeslot_aware_insertion(
    partial_solution: Solution,
    removed_customers: list[int],
    instance: ProblemInstance,
    params: ProblemParameters,
    home_status: HomeStatusMap,
    rng: random.Random,
) -> Solution:
    """R3: insertions that favor higher p_i(t*) slots."""
    _ = rng

    repaired = deepcopy(partial_solution)

    for customer_id in removed_customers:
        candidates = _candidate_solutions_for_customer(repaired, customer_id, instance, params, home_status)
        if not candidates:
            continue

        # Prefer highest arrival-slot home probability; break ties by minimum objective.
        best_solution, _, _ = min(candidates, key=lambda item: (-item[2], item[1]))
        repaired = best_solution

    return repaired

# ============================================================
# 原始文件: spd/offline/operators/cross_group.py
# ============================================================

"""Cross-pair operator implementations (D5/D6 style)."""





def _rebuild_pair(
    pair: VehiclePairSolution,
    customers: set[int],
    instance: ProblemInstance,
    params: ProblemParameters,
    home_status: HomeStatusMap,
) -> VehiclePairSolution:
    """Rebuild one pair route/sorties from a customer set using E-TPRC building steps."""
    builder = ETPRCBuilder()

    if not customers:
        return VehiclePairSolution(
            pair_id=pair.pair_id,
            truck_route=[instance.depot_id, instance.depot_id],
            sorties=[],
            truck_customers=set(),
            drone_customers=set(),
        )

    truck_route = builder._build_truck_route(customers, instance, params)
    truck_route, sorties, truck_customers, drone_customers = builder._extract_sorties(
        truck_route,
        instance,
        params,
        home_status,
    )

    return VehiclePairSolution(
        pair_id=pair.pair_id,
        truck_route=truck_route,
        sorties=sorties,
        truck_customers=truck_customers,
        drone_customers=drone_customers,
    )


def _refresh_unserved(solution: Solution, instance: ProblemInstance) -> None:
    """Recompute unserved customer set from current pair assignments."""
    assigned = set()
    for pair in solution.vehicle_pairs:
        assigned |= pair.all_customers
    solution.unserved_customers = set(instance.customers) - assigned


def cross_pair_swap(
    solution: Solution,
    instance: ProblemInstance,
    params: ProblemParameters,
    home_status: HomeStatusMap,
    rng: random.Random,
) -> Solution:
    """Swap small customer subsets between two vehicle pairs."""
    used_pairs = solution.used_vehicle_pairs
    if len(used_pairs) < 2:
        return solution

    q_cross = max(1, params.alns.cross_group_remove_count)
    pair_a, pair_b = rng.sample(used_pairs, 2)

    customers_a = sorted(pair_a.all_customers)
    customers_b = sorted(pair_b.all_customers)
    if not customers_a or not customers_b:
        return solution

    q_a = min(q_cross, len(customers_a))
    q_b = min(q_cross, len(customers_b))
    selected_a = set(rng.sample(customers_a, q_a))
    selected_b = set(rng.sample(customers_b, q_b))

    target_a = (set(pair_a.all_customers) - selected_a) | selected_b
    target_b = (set(pair_b.all_customers) - selected_b) | selected_a

    candidate = deepcopy(solution)
    idx_a = next(idx for idx, pair in enumerate(candidate.vehicle_pairs) if pair.pair_id == pair_a.pair_id)
    idx_b = next(idx for idx, pair in enumerate(candidate.vehicle_pairs) if pair.pair_id == pair_b.pair_id)

    try:
        candidate.vehicle_pairs[idx_a] = _rebuild_pair(candidate.vehicle_pairs[idx_a], target_a, instance, params, home_status)
        candidate.vehicle_pairs[idx_b] = _rebuild_pair(candidate.vehicle_pairs[idx_b], target_b, instance, params, home_status)
        _refresh_unserved(candidate, instance)

        if not all_hard_constraints_satisfied(instance, candidate, home_status, params, enforce_sortie_sequence=True):
            return solution

        old_obj = compute_objective(instance, solution, home_status, params)
        new_obj = compute_objective(instance, candidate, home_status, params)
    except (AssertionError, KeyError, ValueError, StopIteration):
        return solution

    return candidate if new_obj < old_obj else solution


def cross_pair_transfer(
    solution: Solution,
    instance: ProblemInstance,
    params: ProblemParameters,
    home_status: HomeStatusMap,
    rng: random.Random,
) -> Solution:
    """Transfer one or more customers from one pair to another."""
    used_pairs = solution.used_vehicle_pairs
    if len(used_pairs) < 2:
        return solution

    pair_a, pair_b = rng.sample(used_pairs, 2)
    customers_a = sorted(pair_a.all_customers)
    if not customers_a:
        return solution

    moved_customer = rng.choice(customers_a)
    target_a = set(pair_a.all_customers)
    target_b = set(pair_b.all_customers)
    target_a.remove(moved_customer)
    target_b.add(moved_customer)

    candidate = deepcopy(solution)
    idx_a = next(idx for idx, pair in enumerate(candidate.vehicle_pairs) if pair.pair_id == pair_a.pair_id)
    idx_b = next(idx for idx, pair in enumerate(candidate.vehicle_pairs) if pair.pair_id == pair_b.pair_id)

    try:
        candidate.vehicle_pairs[idx_a] = _rebuild_pair(candidate.vehicle_pairs[idx_a], target_a, instance, params, home_status)
        candidate.vehicle_pairs[idx_b] = _rebuild_pair(candidate.vehicle_pairs[idx_b], target_b, instance, params, home_status)
        _refresh_unserved(candidate, instance)

        if not all_hard_constraints_satisfied(instance, candidate, home_status, params, enforce_sortie_sequence=True):
            return solution

        old_obj = compute_objective(instance, solution, home_status, params)
        new_obj = compute_objective(instance, candidate, home_status, params)
    except (AssertionError, KeyError, ValueError, StopIteration):
        return solution

    return candidate if new_obj < old_obj else solution

# ============================================================
# 原始文件: spd/offline/alns.py
# ============================================================

"""ALNS optimizer contract."""





@dataclass(slots=True)
class ALNSOperatorSet:
    """Collection of ALNS operators with dynamic weights."""

    destroy_operators: list[DestroyOperator]
    repair_operators: list[RepairOperator]
    cross_group_operators: list[CrossGroupOperator] = field(default_factory=list)


def _needs_sortie_rebuild(pair: VehiclePairSolution) -> bool:
    """Check if a pair's sortie structure has been damaged by destroy operations.

    Returns True if any sortie's launch_node or recovery_node is missing from
    the pair's truck_route or truck_customers, or if their ordering is wrong.
    """
    route = pair.truck_route
    route_set = set(route)
    sortie_spans: list[tuple[int, int]] = []
    for sortie in pair.sorties:
        # launch/recovery must still be in truck_customers and truck_route
        if sortie.launch_node not in pair.truck_customers:
            return True
        if sortie.recovery_node not in pair.truck_customers:
            return True
        if sortie.launch_node not in route_set:
            return True
        if sortie.recovery_node not in route_set:
            return True

        # launch must appear before recovery in route
        try:
            launch_idx = route.index(sortie.launch_node)
            recovery_idx = route.index(sortie.recovery_node)
            if launch_idx >= recovery_idx:
                return True
            sortie_spans.append((launch_idx, recovery_idx))
        except ValueError:
            return True

    # Adjacent sorties (sorted by launch order) must not overlap/cross.
    sortie_spans.sort(key=lambda span: span[0])
    for idx in range(len(sortie_spans) - 1):
        if sortie_spans[idx][1] > sortie_spans[idx + 1][0]:
            return True

    return False


def _fallback_rebuild(
    solution: Solution,
    instance: ProblemInstance,
    params: ProblemParameters,
    home_status: HomeStatusMap,
) -> None:
    """Rebuild pairs in-place when repair leaves structural issues.

    Triggers on TWO conditions (not just unserved customers):
    1. solution.unserved_customers is non-empty (repair didn't restore all customers)
    2. Any pair has damaged sortie structure (launch/recovery nodes missing from route)

    Strategy:
    - Identify ALL pairs needing rebuild (unserved assignment targets + structurally damaged)
    - Assign unserved customers to nearest pair
    - Rebuild affected pairs using E-TPRC's _build_truck_route + _extract_sorties
    - Recompute unserved set

    This mutates `solution` in-place.
    """
    # Step 0: identify pairs needing rebuild
    damaged_pair_indices: set[int] = set()
    for idx, pair in enumerate(solution.vehicle_pairs):
        if _needs_sortie_rebuild(pair):
            damaged_pair_indices.add(idx)

    has_unserved = bool(solution.unserved_customers)

    if not damaged_pair_indices and not has_unserved:
        return

    builder = ETPRCBuilder()

    # Step 1: assign unserved customers to nearest pair
    if has_unserved:
        unserved = list(solution.unserved_customers)
        for cid in unserved:
            cx = instance.customers[cid].x
            cy = instance.customers[cid].y
            best_pair_idx = 0
            best_dist = float('inf')
            for idx, pair in enumerate(solution.vehicle_pairs):
                if not pair.all_customers:
                    dist = float('inf') - 1.0
                else:
                    dist = sum(
                        hypot(cx - instance.customers[c].x, cy - instance.customers[c].y)
                        for c in pair.all_customers
                    ) / len(pair.all_customers)
                if dist < best_dist:
                    best_dist = dist
                    best_pair_idx = idx
            solution.vehicle_pairs[best_pair_idx].truck_customers.add(cid)
            damaged_pair_indices.add(best_pair_idx)

    def _clear_pair_sorties_to_truck(pair: VehiclePairSolution, pair_idx: int, reason: str) -> None:
        moved_customers: set[int] = set()
        if not pair.truck_route:
            pair.truck_route = [instance.depot_id, instance.depot_id]
        elif len(pair.truck_route) == 1:
            pair.truck_route.append(instance.depot_id)

        for sortie in pair.sorties:
            for cust in sortie.customers:
                moved_customers.add(cust)
                pair.truck_customers.add(cust)
                if cust not in pair.truck_route:
                    pair.truck_route.insert(-1, cust)
                pair.drone_customers.discard(cust)

        extra_drone_customers = set(pair.drone_customers)
        for cust in extra_drone_customers:
            moved_customers.add(cust)
            pair.truck_customers.add(cust)
            if cust not in pair.truck_route:
                pair.truck_route.insert(-1, cust)
        pair.drone_customers.clear()
        pair.sorties = []
        logging.debug(
            "FALLBACK pair %d: %s, cleared all sorties, moved %d drone customers to truck",
            pair_idx,
            reason,
            len(moved_customers),
        )

    # Step 2: rebuild ALL damaged pairs
    for idx in damaged_pair_indices:
        pair = solution.vehicle_pairs[idx]
        all_custs = pair.all_customers
        if not all_custs:
            continue
        try:
            truck_route = builder._build_truck_route(all_custs, instance, params)
            updated_route, sorties, truck_customers, drone_customers = builder._extract_sorties(
                truck_route, instance, params, home_status,
            )
            pair.truck_route = updated_route
            pair.sorties = sorties
            pair.truck_customers = truck_customers
            pair.drone_customers = drone_customers

            if _needs_sortie_rebuild(pair):
                _clear_pair_sorties_to_truck(pair, idx, "rebuild succeeded but sequence invalid")
        except (AssertionError, KeyError, ValueError):
            _clear_pair_sorties_to_truck(pair, idx, "rebuild failed")

    # Step 3: recompute unserved
    assigned: set[int] = set()
    for pair in solution.vehicle_pairs:
        assigned |= pair.all_customers
    solution.unserved_customers = set(instance.customers) - assigned


def _structural_and_physical_check(
    instance: ProblemInstance,
    solution: Solution,
    home_status: HomeStatusMap,
    params: ProblemParameters,
) -> bool:
    """Check only structural and physical constraints, skip time-window checks.

    This is used during ALNS search to allow the objective function's
    z_tw_penalty (with adaptive beta) to handle time-window violations
    as soft penalties, instead of hard-rejecting every candidate.

    Checks performed (must pass):
    - Timeline computability (structural prerequisite for compute_objective)
    - Sortie structure: launch/recovery in route and customers
    - Sortie sequence ordering
    - F1: drone energy <= battery capacity
    - F2: drone payload <= drone capacity
    - F3: truck wait time <= max_wait
    - F5: truck load at recovery <= truck capacity
    - F7: truck payload at all nodes <= truck capacity

    Checks SKIPPED (handled by z_tw_penalty in compute_objective):
    - F4: drone customer time-window upper bounds
    - F6: truck customer time-window upper bounds
    - F8: depot return-time upper bound
    """
    _EPS = 1e-9

    for pair_idx, pair in enumerate(solution.vehicle_pairs):
        try:
            timeline = compute_truck_timeline(instance, pair, home_status, params)
        except (AssertionError, KeyError, ValueError) as e:
            logging.debug("CONSTRAINT FAIL: pair %d timeline computation error: %s", pair_idx, e)
            return False

        # Sortie structure and sequence (structural)
        for sortie_idx, sortie in enumerate(pair.sorties):
            if not check_sortie_structure(pair, sortie):
                logging.debug("CONSTRAINT FAIL: pair %d sortie %d structure invalid", pair_idx, sortie_idx)
                return False

            # Sortie cardinality
            if len(sortie.customers) > params.constraints.max_customers_per_sortie:
                logging.debug(
                    "CONSTRAINT FAIL: pair %d sortie %d cardinality %d > max %d",
                    pair_idx,
                    sortie_idx,
                    len(sortie.customers),
                    params.constraints.max_customers_per_sortie,
                )
                return False

            try:
                # F1: drone energy feasibility
                sortie_energy = compute_sortie_energy(
                    instance, sortie, timeline, home_status, params,
                )
                if sortie_energy > params.energy.drone_battery_capacity + _EPS:
                    logging.debug(
                        "CONSTRAINT FAIL: pair %d sortie %d F1 energy %.2f > capacity %.2f",
                        pair_idx,
                        sortie_idx,
                        sortie_energy,
                        params.energy.drone_battery_capacity,
                    )
                    return False

                # F2: drone payload feasibility
                if not check_sortie_load_feasibility(
                    instance, sortie, home_status, params,
                ):
                    logging.debug(
                        "CONSTRAINT FAIL: pair %d sortie %d F2 drone payload exceeded",
                        pair_idx,
                        sortie_idx,
                    )
                    return False

                # F3: truck wait time for drone at recovery
                recovery = sortie.recovery_node
                wait_truck = max(
                    0.0,
                    timeline.drone_arrival[recovery] - timeline.truck_arrival[recovery],
                )
                if wait_truck > params.constraints.max_truck_wait_time + _EPS:
                    logging.debug(
                        "CONSTRAINT SOFT VIOLATION: pair %d sortie %d F3 truck wait %.2f > max %.2f (allowed in search)",
                        pair_idx,
                        sortie_idx,
                        wait_truck,
                        params.constraints.max_truck_wait_time,
                    )

                # F5: truck load at recovery
                load_state = compute_truck_load(instance, pair, home_status, params)
                if load_state.truck_load_at_node[recovery] > params.vehicle.truck_capacity + _EPS:
                    logging.debug(
                        "CONSTRAINT FAIL: pair %d sortie %d F5 truck load at recovery %.2f > capacity %.2f",
                        pair_idx,
                        sortie_idx,
                        load_state.truck_load_at_node[recovery],
                        params.vehicle.truck_capacity,
                    )
                    return False

            except (AssertionError, KeyError, ValueError) as e:
                logging.debug(
                    "CONSTRAINT FAIL: pair %d sortie %d physical check exception: %s",
                    pair_idx,
                    sortie_idx,
                    e,
                )
                return False

        if not check_sortie_sequence(pair, strict=True):
            logging.debug("CONSTRAINT FAIL: pair %d sortie sequence ordering invalid", pair_idx)
            return False

        # F7: truck payload at all nodes (physical)
        try:
            load_state = compute_truck_load(instance, pair, home_status, params)
            for node in pair.truck_route:
                if node not in load_state.truck_load_at_node:
                    logging.debug("CONSTRAINT FAIL: pair %d F7 missing load for node %d", pair_idx, node)
                    return False
                if load_state.truck_load_at_node[node] > params.vehicle.truck_capacity + _EPS:
                    logging.debug(
                        "CONSTRAINT FAIL: pair %d F7 truck load at node %d: %.2f > capacity %.2f",
                        pair_idx,
                        node,
                        load_state.truck_load_at_node[node],
                        params.vehicle.truck_capacity,
                    )
                    return False
        except (AssertionError, KeyError, ValueError) as e:
            logging.debug("CONSTRAINT FAIL: pair %d F7 truck load check exception: %s", pair_idx, e)
            return False

        # NOTE: F4, F6, F8 (time-window checks) are intentionally OMITTED here.
        # They are handled by z_tw_penalty in compute_objective, with adaptive
        # beta that increases over ACO iterations (beta_schedule).

    return True

class ALNSOptimizer:
    """Offline ALNS optimizer wrapper."""

    def __init__(self, operators: ALNSOperatorSet):
        self._operators = operators
        self.iteration_history: list[dict[str, object]] = []
        self.last_accepted_count: int = 0

    def optimize(
        self,
        initial_solution: Solution,
        instance: ProblemInstance,
        home_status: HomeStatusMap,
        params: ProblemParameters,
        global_best_objective: float | None = None,
        rng: random.Random | None = None,
    ) -> Solution:
        """Run ALNS local search from an initial solution.

        Args:
            global_best_objective: Global best value from ACO outer loop. Use it
                to distinguish sigma_1 (new global best) vs sigma_2 rewards.
        """
        if not self._operators.destroy_operators or not self._operators.repair_operators:
            self.last_accepted_count = 0
            return deepcopy(initial_solution)

        random_state = rng or random.Random(0)
        current_beta = params.beta_schedule.beta_init
        self.iteration_history = []

        current_solution = deepcopy(initial_solution)
        current_objective = compute_objective(
            instance,
            current_solution,
            home_status,
            params,
            time_window_penalty=current_beta,
        )

        best_local = deepcopy(current_solution)
        best_local_objective = current_objective

        destroy_weights = [max(params.alns.operator_weight_init, 1e-9) for _ in self._operators.destroy_operators]
        repair_weights = [max(params.alns.operator_weight_init, 1e-9) for _ in self._operators.repair_operators]

        temperature = max(params.alns.sa_initial_temperature, 1e-9)
        accepted_count = 0

        def roulette_select(weights: list[float]) -> int:
            """Select one index by roulette-wheel sampling."""
            total = sum(max(weight, 0.0) for weight in weights)
            if total <= 0.0:
                return random_state.randrange(len(weights))

            draw = random_state.random() * total
            cumulative = 0.0
            for idx, weight in enumerate(weights):
                cumulative += max(weight, 0.0)
                if draw <= cumulative:
                    return idx
            return len(weights) - 1

        for iteration in range(params.alns.iterations):
            destroy_idx = roulette_select(destroy_weights)
            repair_idx = roulette_select(repair_weights)

            remove_low = min(params.alns.remove_count_min, params.alns.remove_count_max)
            remove_high = max(params.alns.remove_count_min, params.alns.remove_count_max)
            remove_count = random_state.randint(remove_low, remove_high)

            destroy_op = self._operators.destroy_operators[destroy_idx]
            repair_op = self._operators.repair_operators[repair_idx]
            destroy_name = getattr(destroy_op, 'name', getattr(destroy_op, '__name__', f'destroy_{destroy_idx}'))
            repair_name = getattr(repair_op, 'name', getattr(repair_op, '__name__', f'repair_{repair_idx}'))

            destroy_result = destroy_op(
                current_solution,
                instance,
                params,
                remove_count,
                home_status,
                random_state,
            )
            new_solution = repair_op(
                destroy_result.partial_solution,
                destroy_result.removed_customers,
                instance,
                params,
                home_status,
                random_state,
            )

            # --- Fallback: rebuild pairs with structural issues ---
            has_unserved = bool(new_solution.unserved_customers)
            has_damaged = any(
                _needs_sortie_rebuild(pair) for pair in new_solution.vehicle_pairs
            )
            if has_unserved or has_damaged:
                unserved_before = len(new_solution.unserved_customers)
                damaged_before = sum(1 for p in new_solution.vehicle_pairs if _needs_sortie_rebuild(p))
                _fallback_rebuild(new_solution, instance, params, home_status)
                unserved_after = len(new_solution.unserved_customers)
                damaged_after = sum(1 for p in new_solution.vehicle_pairs if _needs_sortie_rebuild(p))
                logging.debug(
                    "ALNS iter %d: FALLBACK triggered, unserved %d->%d, damaged_pairs %d->%d",
                    iteration + 1,
                    unserved_before,
                    unserved_after,
                    damaged_before,
                    damaged_after,
                )

            # Periodic cross-pair operation.
            if (
                self._operators.cross_group_operators
                and params.alns.cross_group_frequency > 0
                and (iteration + 1) % params.alns.cross_group_frequency == 0
            ):
                cross_op = random_state.choice(self._operators.cross_group_operators)
                new_solution = cross_op(new_solution, instance, params, home_status, random_state)

            try:
                if not _structural_and_physical_check(
                    instance,
                    new_solution,
                    home_status,
                    params,
                ):
                    raise ValueError("structural or physical constraint violated")

                logging.debug(
                    "ALNS iter %d: candidate PASSED structural_and_physical_check, computing objective...",
                    iteration + 1,
                )
                new_objective = compute_objective(
                    instance,
                    new_solution,
                    home_status,
                    params,
                    time_window_penalty=current_beta,
                )
                logging.debug(
                    "ALNS iter %d: candidate objective=%.4f, current best=%.4f",
                    iteration + 1,
                    new_objective,
                    best_local_objective,
                )
                if (iteration + 1) % 50 == 0:
                    logging.debug(
                        "ALNS iter %d: candidate_obj=%.4f, current_obj=%.4f, best_obj=%.4f",
                        iteration + 1,
                        new_objective,
                        current_objective,
                        best_local_objective,
                    )
            except (AssertionError, KeyError, ValueError) as e:
                logging.debug(
                    "ALNS iter %d: candidate SKIPPED due to exception: %s",
                    iteration + 1,
                    str(e),
                )
                # Invalid candidate is treated as rejected move.
                destroy_score = 0.0
                repair_score = 0.0
                destroy_weights[destroy_idx] = (
                    (1.0 - params.alns.reaction_factor) * destroy_weights[destroy_idx]
                    + params.alns.reaction_factor * destroy_score
                )
                repair_weights[repair_idx] = (
                    (1.0 - params.alns.reaction_factor) * repair_weights[repair_idx]
                    + params.alns.reaction_factor * repair_score
                )
                temperature = max(temperature * params.alns.sa_cooling_rate, 1e-9)
                operator_scores = {
                    **{
                        f"destroy::{getattr(op, 'name', getattr(op, '__name__', f'destroy_{idx}'))}": float(destroy_weights[idx])
                        for idx, op in enumerate(self._operators.destroy_operators)
                    },
                    **{
                        f"repair::{getattr(op, 'name', getattr(op, '__name__', f'repair_{idx}'))}": float(repair_weights[idx])
                        for idx, op in enumerate(self._operators.repair_operators)
                    },
                }
                self.iteration_history.append(
                    {
                        'iteration': int(iteration + 1),
                        'current_cost': float(current_objective),
                        'best_cost': float(best_local_objective),
                        'operator_name': f"{destroy_name}+{repair_name}",
                        'operator_scores': operator_scores,
                    }
                )
                continue

            delta = new_objective - current_objective
            accept = False
            reward = 0.0
            improved = delta < 0.0
            sa_probability = 1.0

            if delta < 0.0:
                accept = True
                if global_best_objective is not None and new_objective < global_best_objective:
                    reward = params.alns.reward_global_best
                else:
                    reward = params.alns.reward_improve
            else:
                sa_probability = math.exp(-delta / max(temperature, 1e-9))
                if random_state.random() < sa_probability:
                    accept = True
                    reward = params.alns.reward_accept_worse

            if accept:
                accepted_count += 1
                current_solution = new_solution
                current_objective = new_objective

                if (iteration + 1) % 50 == 0:
                    logging.debug(
                        "ALNS iter %d: ACCEPTED (improved=%s, sa_prob=%.4f, temperature=%.4f)",
                        iteration + 1,
                        improved,
                        sa_probability,
                        temperature,
                    )

                if current_objective < best_local_objective or not all_hard_constraints_satisfied(
                    instance, best_local, home_status, params,
                    enforce_sortie_sequence=True,
                ):
                    # Only promote to best if hard constraints are satisfied.
                    # This prevents ALNS from returning infeasible solutions
                    # (e.g. solutions with unserved customers) as the "best".
                    # current_solution is still accepted by SA for exploration,
                    # but best_local must always be feasible.
                    if all_hard_constraints_satisfied(
                        instance, current_solution, home_status, params,
                        enforce_sortie_sequence=True,
                    ):
                        previous_best = best_local_objective
                        best_local = deepcopy(current_solution)
                        best_local_objective = current_objective
                        logging.debug(
                            "ALNS iter %d: NEW BEST found, obj=%.4f (improved from %.4f)",
                            iteration + 1,
                            current_objective,
                            previous_best,
                        )
                    else:
                        logging.debug(
                            "ALNS iter %d: better obj=%.4f but INFEASIBLE, best_local unchanged",
                            iteration + 1,
                            current_objective,
                        )
            else:
                if (iteration + 1) % 50 == 0:
                    logging.debug(
                        "ALNS iter %d: REJECTED (delta=%.4f, sa_prob=%.4f, temperature=%.4f)",
                        iteration + 1,
                        delta,
                        sa_probability,
                        temperature,
                    )

            # Adaptive weight update for selected operators.
            destroy_weights[destroy_idx] = (
                (1.0 - params.alns.reaction_factor) * destroy_weights[destroy_idx]
                + params.alns.reaction_factor * reward
            )
            repair_weights[repair_idx] = (
                (1.0 - params.alns.reaction_factor) * repair_weights[repair_idx]
                + params.alns.reaction_factor * reward
            )

            temperature = max(temperature * params.alns.sa_cooling_rate, 1e-9)
            operator_scores = {
                **{
                    f"destroy::{getattr(op, 'name', getattr(op, '__name__', f'destroy_{idx}'))}": float(destroy_weights[idx])
                    for idx, op in enumerate(self._operators.destroy_operators)
                },
                **{
                    f"repair::{getattr(op, 'name', getattr(op, '__name__', f'repair_{idx}'))}": float(repair_weights[idx])
                    for idx, op in enumerate(self._operators.repair_operators)
                },
            }
            self.iteration_history.append(
                {
                    'iteration': int(iteration + 1),
                    'current_cost': float(current_objective),
                    'best_cost': float(best_local_objective),
                    'operator_name': f"{destroy_name}+{repair_name}",
                    'operator_scores': operator_scores,
                }
            )

        self.last_accepted_count = accepted_count
        return best_local


def _build_default_alns_operator_set() -> ALNSOperatorSet:
    """Build the default ALNS operator set used by each ant worker."""
    return ALNSOperatorSet(
        destroy_operators=[
            d1_random_removal,
            d2_worst_removal,
            d3_related_removal,
            d4_low_home_probability_removal,
        ],
        repair_operators=[
            r1_greedy_insertion,
            r2_regret_insertion,
            r3_timeslot_aware_insertion,
        ],
        cross_group_operators=[
            cross_pair_swap,
            cross_pair_transfer,
        ],
    )


def _derive_ant_seed(base_seed: int, aco_iteration: int, ant_index: int) -> int:
    """Derive deterministic per-ant seed from base seed, ACO iteration, and ant index."""
    return int(base_seed) * 10000 + int(aco_iteration) * 100 + int(ant_index)


def _run_single_ant(
    args: tuple[
        ProblemInstance,
        ProblemParameters,
        dict[tuple[int, int], float],
        float,
        int,
        int,
        int,
        dict[int, bool],
        float | None,
        bool,
    ],
) -> dict[str, object]:
    """Run one ant end-to-end: construction + ALNS refinement.

    Designed as a top-level function to remain pickle-compatible under
    multiprocessing on Windows (spawn).
    """
    (
        instance,
        params,
        pheromone_matrix,
        current_beta,
        ant_seed,
        aco_iteration,
        ant_index,
        home_status,
        best_global_objective,
        suppress_debug_logs,
    ) = args

    if suppress_debug_logs:
        logging.getLogger().setLevel(logging.WARNING)
        logging.getLogger(__name__).setLevel(logging.WARNING)

    random_state = random.Random(int(ant_seed))

    try:
        etprc_builder = ETPRCBuilder()
        alns_optimizer = ALNSOptimizer(_build_default_alns_operator_set())
        solver = ACOALNSSolver(etprc_builder=etprc_builder, alns_optimizer=alns_optimizer)
        state = ACOState(
            pheromone=dict(pheromone_matrix),
            current_beta=float(current_beta),
        )

        ant_solution = solver.construct_ant_solution(
            instance=instance,
            params=params,
            home_status=home_status,
            state=state,
            rng=random_state,
        )
        refined_solution = alns_optimizer.optimize(
            initial_solution=ant_solution,
            instance=instance,
            home_status=home_status,
            params=params,
            global_best_objective=best_global_objective,
            rng=random_state,
        )
        refined_objective = compute_objective(
            instance,
            refined_solution,
            home_status,
            params,
            time_window_penalty=float(current_beta),
        )

        history_rows: list[dict[str, object]] = []
        for alns_row in alns_optimizer.iteration_history:
            row = dict(alns_row)
            row["aco_iteration"] = int(aco_iteration)
            row["ant_index"] = int(ant_index + 1)
            history_rows.append(row)

        return {
            "ok": True,
            "ant_index": int(ant_index),
            "refined_solution": refined_solution,
            "refined_objective": float(refined_objective),
            "iteration_history": history_rows,
            "accepted_count": int(getattr(alns_optimizer, "last_accepted_count", 0)),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "ant_index": int(ant_index),
            "error": str(exc),
            "iteration_history": [],
            "accepted_count": 0,
        }

# ============================================================
# 原始文件: spd/offline/aco_alns.py
# ============================================================

def _run_single_pure_aco_ant(
    args: tuple[
        ProblemInstance,
        ProblemParameters,
        dict[tuple[int, int], float],
        int,
        int,
        int,
        dict[int, bool],
        bool,
    ],
) -> dict[str, object]:
    """运行一只纯 ACO 蚂蚁：只做构造与评估，不触发 ALNS。"""
    (
        instance,
        params,
        pheromone_matrix,
        ant_seed,
        aco_iteration,
        ant_index,
        home_status,
        suppress_debug_logs,
    ) = args

    if suppress_debug_logs:
        logging.getLogger().setLevel(logging.WARNING)
        logging.getLogger(__name__).setLevel(logging.WARNING)

    random_state = random.Random(int(ant_seed))

    try:
        # 复用 ACOALNSSolver 的构造逻辑，确保纯 ACO 与混合算法使用同一套 E-TPRC+信息素规则。
        etprc_builder = ETPRCBuilder()
        helper_solver = ACOALNSSolver(
            etprc_builder=etprc_builder,
            alns_optimizer=ALNSOptimizer(_build_default_alns_operator_set()),
        )
        state = ACOState(
            pheromone=dict(pheromone_matrix),
            current_beta=float(params.beta_schedule.beta_init),
        )
        ant_solution = helper_solver.construct_ant_solution(
            instance=instance,
            params=params,
            home_status=home_status,
            state=state,
            rng=random_state,
        )
        ant_objective = compute_objective(
            instance,
            ant_solution,
            home_status,
            params,
        )
        return {
            "ok": True,
            "ant_index": int(ant_index),
            "aco_iteration": int(aco_iteration),
            "solution": ant_solution,
            "objective": float(ant_objective),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "ant_index": int(ant_index),
            "aco_iteration": int(aco_iteration),
            "error": str(exc),
        }


"""ACO + ALNS hybrid solver contract."""





@dataclass(slots=True)
class ACOState:
    """State holder for ACO iterations, pheromone matrix, and dynamic beta."""

    pheromone: dict[tuple[int, int], float]
    current_beta: float
    best_solution: Solution | None = None
    best_objective: float | None = None


class ACOALNSSolver:
    """Main offline solver implementing the ACO-ALNS flow."""

    def __init__(self, etprc_builder: ETPRCBuilder, alns_optimizer: ALNSOptimizer):
        self._etprc_builder = etprc_builder
        self._alns_optimizer = alns_optimizer
        self.iteration_history: list[dict[str, object]] = []

    @staticmethod
    def _clip01(value: float) -> float:
        return min(1.0, max(0.0, float(value)))

    @staticmethod
    def _truck_edges(solution: Solution) -> set[tuple[int, int]]:
        edges: set[tuple[int, int]] = set()
        for pair in solution.vehicle_pairs:
            route = pair.truck_route
            for idx in range(len(route) - 1):
                edges.add((route[idx], route[idx + 1]))
        return edges

    def _resolve_candidate_list_size(self, params: ProblemParameters, remaining_count: int) -> int:
        configured = getattr(params.aco, "candidate_list_size", None)
        if configured is None:
            dynamic_default = int(round(math.sqrt(max(1, remaining_count)) * 4.0))
            configured = min(25, max(10, dynamic_default))
        try:
            candidate_size = int(configured)
        except (TypeError, ValueError):
            candidate_size = 15
        candidate_size = max(1, candidate_size)
        return min(remaining_count, candidate_size)

    def _resolve_tau_bounds(self, params: ProblemParameters) -> tuple[float, float]:
        initial_pheromone = float(
            getattr(params.aco, "pheromone_init", getattr(params.aco, "initial_pheromone", 1.0))
        )
        configured_min = getattr(params.aco, "tau_min", None)
        configured_max = getattr(params.aco, "tau_max", None)

        if configured_min is None:
            tau_min = max(1e-9, 0.1 * initial_pheromone)
        else:
            tau_min = float(configured_min)
        if configured_max is None:
            tau_max = max(tau_min + 1e-9, 10.0 * initial_pheromone)
        else:
            tau_max = float(configured_max)

        tau_min = max(1e-9, tau_min)
        tau_max = max(tau_min + 1e-9, tau_max)
        return tau_min, tau_max

    def _partial_pheromone_restart(
        self,
        state: ACOState,
        params: ProblemParameters,
        rng: random.Random,
        protected_edges: set[tuple[int, int]] | None = None,
    ) -> None:
        restart_ratio = float(getattr(params.aco, "stagnation_restart_ratio", 0.35))
        restart_ratio = min(max(restart_ratio, 0.0), 1.0)
        if restart_ratio <= 0.0 or not state.pheromone:
            return

        tau_min, tau_max = self._resolve_tau_bounds(params)
        initial_pheromone = float(
            getattr(params.aco, "pheromone_init", getattr(params.aco, "initial_pheromone", 1.0))
        )
        reset_value = min(max(initial_pheromone, tau_min), tau_max)

        edges = list(state.pheromone.keys())
        mutable_edges = edges
        if protected_edges:
            filtered = [edge for edge in edges if edge not in protected_edges]
            if filtered:
                mutable_edges = filtered

        restart_count = max(1, int(math.ceil(len(mutable_edges) * restart_ratio)))
        restart_count = min(restart_count, len(mutable_edges))
        for edge in rng.sample(mutable_edges, restart_count):
            state.pheromone[edge] = reset_value

    def solve(
        self,
        instance: ProblemInstance,
        params: ProblemParameters,
        home_status: HomeStatusMap,
        rng: random.Random | None = None,
    ) -> Solution:
        """Run full offline search and return the best solution.

        Implementations should initialize and update ACOState.current_beta using:
            beta <- min(beta * gamma_beta, beta_max)
        once per ACO outer iteration.
        """
        random_state = rng or random.Random(0)
        self.iteration_history = []

        initial_solution = self._etprc_builder.build_initial_solution(instance, params, random_state)
        initial_objective = compute_objective(
            instance,
            initial_solution,
            home_status,
            params,
            time_window_penalty=params.beta_schedule.beta_init,
        )

        # Keep a deterministic fallback start to reduce seed sensitivity.
        try:
            fallback_solution = self._etprc_builder.build_initial_solution(instance, params, random.Random(0))
            fallback_objective = compute_objective(
                instance,
                fallback_solution,
                home_status,
                params,
                time_window_penalty=params.beta_schedule.beta_init,
            )
            if fallback_objective < initial_objective:
                initial_solution = fallback_solution
                initial_objective = fallback_objective
        except (AssertionError, KeyError, ValueError):
            pass

        initial_pheromone = float(
            getattr(params.aco, "pheromone_init", getattr(params.aco, "initial_pheromone", 1.0))
        )

        # Initialize directed pheromone matrix for truck arcs.
        pheromone = {arc: initial_pheromone for arc in instance.truck_distance_km}
        if not pheromone:
            nodes = [instance.depot_id, *instance.customers.keys()]
            pheromone = {(i, j): initial_pheromone for i in nodes for j in nodes}

        state = ACOState(
            pheromone=pheromone,
            current_beta=params.beta_schedule.beta_init,
        )

        state.best_solution = initial_solution
        state.best_objective = initial_objective

        max_iterations = max(0, int(getattr(params.aco, "max_iterations", getattr(params.aco, "MaxIter", 0))))
        ant_count = max(1, int(getattr(params.aco, "ant_count", 1)))
        no_improve_max = max(1, int(getattr(params.aco, "no_improve_max", getattr(params.aco, "NoImprove_max", 1))))
        no_improve_count = 0
        aco_iteration = 0
        parallel_ants = _ACO_PARALLEL_OVERRIDES.get(id(params.aco), bool(getattr(params.aco, "parallel_ants", True)))
        try:
            cpu_count = int(mp.cpu_count())
        except (NotImplementedError, ValueError):
            cpu_count = 1
        n_workers = min(ant_count, max(1, cpu_count - 1))
        use_parallel = parallel_ants and ant_count > 1 and n_workers > 1

        base_seed_raw = getattr(params.aco, "seed", getattr(params.aco, "random_seed", None))
        if base_seed_raw is None:
            base_seed = random_state.getrandbits(31)
        else:
            try:
                base_seed = int(base_seed_raw)
            except (TypeError, ValueError):
                base_seed = random_state.getrandbits(31)

        pool = None
        if use_parallel:
            try:
                ctx = mp.get_context("spawn")
                pool = ctx.Pool(processes=n_workers)
            except (OSError, RuntimeError, ValueError) as exc:
                logging.warning("ACO ant parallel disabled, fallback to serial: %s", exc)
                pool = None
                use_parallel = False

        try:
            for _ in range(max_iterations):
                aco_iteration += 1
                iteration_best_solution: Solution | None = None
                iteration_best_objective: float | None = None

                ant_args_list: list[
                    tuple[
                        ProblemInstance,
                        ProblemParameters,
                        dict[tuple[int, int], float],
                        float,
                        int,
                        int,
                        int,
                        dict[int, bool],
                        float | None,
                        bool,
                    ]
                ] = []
                for _ant in range(ant_count):
                    ant_seed = _derive_ant_seed(base_seed, aco_iteration, _ant)
                    ant_args_list.append(
                        (
                            instance,
                            params,
                            state.pheromone,
                            state.current_beta,
                            ant_seed,
                            aco_iteration,
                            _ant,
                            dict(home_status),
                            state.best_objective,
                            pool is not None,
                        )
                    )

                if pool is not None:
                    try:
                        ant_results = pool.map(_run_single_ant, ant_args_list)
                    except Exception as exc:  # noqa: BLE001
                        logging.warning("ACO ant parallel run failed, fallback to serial this iteration: %s", exc)
                        serial_args = [(*args[:-1], False) for args in ant_args_list]
                        ant_results = [_run_single_ant(args) for args in serial_args]
                else:
                    ant_results = [_run_single_ant(args) for args in ant_args_list]

                for ant_result in ant_results:
                    if not ant_result.get("ok", False):
                        continue

                    refined_solution = ant_result["refined_solution"]
                    refined_objective = float(ant_result["refined_objective"])
                    ant_index = int(ant_result.get("ant_index", -1))

                    for alns_row in ant_result.get("iteration_history", []):
                        self.iteration_history.append(dict(alns_row))

                    logging.info(
                        "ACO iter %d, ant %d: best_obj=%.4f, iterations_accepted=%d",
                        aco_iteration,
                        ant_index + 1,
                        refined_objective,
                        int(ant_result.get("accepted_count", 0)),
                    )

                    if iteration_best_objective is None or refined_objective < iteration_best_objective:
                        iteration_best_solution = refined_solution
                        iteration_best_objective = refined_objective

                if (
                    iteration_best_solution is not None
                    and iteration_best_objective is not None
                    and (state.best_objective is None or iteration_best_objective < state.best_objective)
                ):
                    state.best_solution = iteration_best_solution
                    state.best_objective = iteration_best_objective
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if state.best_solution is not None and state.best_objective is not None:
                    self.update_pheromone(
                        state=state,
                        best_solution=state.best_solution,
                        best_objective=state.best_objective,
                        params=params,
                        iteration_best_solution=iteration_best_solution,
                        iteration_best_objective=iteration_best_objective,
                        global_best_solution=state.best_solution,
                        global_best_objective=state.best_objective,
                    )

                self.advance_beta_schedule(state, params)

                if no_improve_count >= no_improve_max:
                    break
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        return state.best_solution if state.best_solution is not None else initial_solution

    def construct_ant_solution(
        self,
        instance: ProblemInstance,
        params: ProblemParameters,
        home_status: HomeStatusMap,
        state: ACOState,
        rng: random.Random,
    ) -> Solution:
        """Construct one candidate solution for an ant."""
        depot_id = instance.depot_id
        initial_pheromone = float(
            getattr(params.aco, "pheromone_init", getattr(params.aco, "initial_pheromone", 1.0))
        )
        alpha_aco = max(0.0, float(getattr(params.aco, "alpha_aco", getattr(params.aco, "alpha", 1.0))))
        beta_aco = max(0.0, float(getattr(params.aco, "beta_aco", getattr(params.aco, "beta", 1.0))))
        w_dist = max(0.0, float(getattr(params.aco, "eta_weight_distance", 1.0)))
        w_tw = max(0.0, float(getattr(params.aco, "eta_weight_time_window", 0.8)))
        w_cap = max(0.0, float(getattr(params.aco, "eta_weight_capacity", 0.6)))

        truck_speed = max(float(params.vehicle.truck_speed), 1e-9)
        truck_capacity = max(float(params.vehicle.truck_capacity), 1e-9)
        time_horizon = max(1.0, float(params.time.depot_latest - params.time.depot_earliest))

        def roulette_select(candidates: list[int], weights: list[float]) -> int:
            """Select one candidate by roulette-wheel sampling."""
            total = sum(max(weight, 0.0) for weight in weights)
            if total <= 0.0:
                return rng.choice(candidates)

            draw = rng.random() * total
            cumulative = 0.0
            for candidate, weight in zip(candidates, weights):
                cumulative += max(weight, 0.0)
                if draw <= cumulative:
                    return candidate
            return candidates[-1]

        def projected_truck_load_after_service(current_load: float, customer_id: int) -> float:
            weight = float(instance.customers[customer_id].weight)
            if customer_id in instance.delivery_customers:
                return current_load - weight
            if customer_id in instance.pickup_customers:
                return current_load + weight
            return current_load

        def transition_stats(
            current_node: int,
            current_time: float,
            current_load: float,
            candidate: int,
            dist_scale: float,
        ) -> tuple[float, bool, float, float]:
            distance = max(float(instance.truck_distance_km.get((current_node, candidate), 0.0)), 1e-9)
            travel_time = distance / truck_speed
            arrival_time = current_time + travel_time
            latest_time = float(instance.customers[candidate].time_window[1])
            projected_load = projected_truck_load_after_service(current_load, candidate)

            tw_slack = latest_time - arrival_time
            capacity_slack = truck_capacity - projected_load
            feasible = tw_slack >= -1e-9 and capacity_slack >= -1e-9

            dist_term = 1.0 / (1.0 + distance / max(dist_scale, 1e-9))
            tw_slack_term = self._clip01(tw_slack / time_horizon)
            capacity_slack_term = self._clip01(capacity_slack / truck_capacity)
            eta = max(1e-9, w_dist * dist_term + w_tw * tw_slack_term + w_cap * capacity_slack_term)

            tau = max(state.pheromone.get((current_node, candidate), initial_pheromone), 1e-9)
            weight = (tau**alpha_aco) * (eta**beta_aco)
            return weight, feasible, arrival_time, projected_load

        try:
            groups = self._etprc_builder._customer_grouping(instance, params, rng)
        except (AssertionError, KeyError, ValueError):
            return self._etprc_builder.build_initial_solution(instance, params, rng)

        vehicle_pairs: list[VehiclePairSolution] = []
        assigned_customers: set[int] = set()

        for pair_idx, group in enumerate(groups):
            customer_group = set(group)
            if not customer_group:
                vehicle_pairs.append(
                    VehiclePairSolution(
                        pair_id=pair_idx + 1,
                        truck_route=[depot_id, depot_id],
                        sorties=[],
                        truck_customers=set(),
                        drone_customers=set(),
                    )
                )
                continue

            route = [depot_id]
            remaining = set(customer_group)
            current = depot_id
            current_time = float(params.time.depot_earliest)
            current_load = sum(
                float(instance.customers[customer_id].weight)
                for customer_id in customer_group
                if customer_id in instance.delivery_customers
            )

            # Build sequence with candidate list + feasibility-aware eta.
            while remaining:
                sorted_candidates = sorted(
                    remaining,
                    key=lambda candidate: (
                        float(instance.truck_distance_km.get((current, candidate), 0.0)),
                        candidate,
                    ),
                )
                candidate_size = self._resolve_candidate_list_size(params, len(sorted_candidates))
                near_candidates = sorted_candidates[:candidate_size]

                near_dist_scale = sum(
                    max(float(instance.truck_distance_km.get((current, candidate), 0.0)), 1e-9)
                    for candidate in near_candidates
                ) / max(1, len(near_candidates))

                transition_cache: dict[int, tuple[float, bool, float, float]] = {}
                feasible_candidates: list[int] = []
                feasible_weights: list[float] = []
                for candidate in near_candidates:
                    stats = transition_stats(current, current_time, current_load, candidate, near_dist_scale)
                    transition_cache[candidate] = stats
                    weight, feasible, _arrival, _load = stats
                    if feasible:
                        feasible_candidates.append(candidate)
                        feasible_weights.append(weight)

                if feasible_candidates:
                    selection_pool = feasible_candidates
                    selection_weights = feasible_weights
                else:
                    full_dist_scale = sum(
                        max(float(instance.truck_distance_km.get((current, candidate), 0.0)), 1e-9)
                        for candidate in sorted_candidates
                    ) / max(1, len(sorted_candidates))
                    expanded_feasible: list[int] = []
                    expanded_feasible_weights: list[float] = []
                    for candidate in sorted_candidates:
                        if candidate in transition_cache:
                            stats = transition_cache[candidate]
                        else:
                            stats = transition_stats(current, current_time, current_load, candidate, full_dist_scale)
                            transition_cache[candidate] = stats
                        weight, feasible, _arrival, _load = stats
                        if feasible:
                            expanded_feasible.append(candidate)
                            expanded_feasible_weights.append(weight)

                    if expanded_feasible:
                        selection_pool = expanded_feasible
                        selection_weights = expanded_feasible_weights
                    else:
                        selection_pool = near_candidates if near_candidates else sorted_candidates
                        selection_weights = [
                            transition_cache[candidate][0] if candidate in transition_cache else 1.0
                            for candidate in selection_pool
                        ]

                next_customer = roulette_select(selection_pool, selection_weights)
                _weight, _feasible, arrival_time, projected_load = transition_cache[next_customer]

                route.append(next_customer)
                remaining.remove(next_customer)
                current = next_customer

                customer = instance.customers[next_customer]
                service_start = max(arrival_time, float(customer.time_window[0]))
                current_time = service_start + float(customer.service_time)
                current_load = min(max(projected_load, 0.0), truck_capacity)

            route.append(depot_id)

            try:
                updated_route, sorties, truck_customers, drone_customers = self._etprc_builder._extract_sorties(
                    route,
                    instance,
                    params,
                    home_status,
                )
            except (AssertionError, KeyError, ValueError):
                # Fallback to deterministic route builder when ant route is invalid.
                fallback_route = self._etprc_builder._build_truck_route(customer_group, instance, params)
                updated_route, sorties, truck_customers, drone_customers = self._etprc_builder._extract_sorties(
                    fallback_route,
                    instance,
                    params,
                    home_status,
                )

            pair_solution = VehiclePairSolution(
                pair_id=pair_idx + 1,
                truck_route=updated_route,
                sorties=sorties,
                truck_customers=truck_customers,
                drone_customers=drone_customers,
            )
            vehicle_pairs.append(pair_solution)
            assigned_customers |= pair_solution.all_customers

        # Keep pair count stable even if grouping returns fewer groups.
        while len(vehicle_pairs) < instance.vehicle_pair_count:
            vehicle_pairs.append(
                VehiclePairSolution(
                    pair_id=len(vehicle_pairs) + 1,
                    truck_route=[depot_id, depot_id],
                    sorties=[],
                    truck_customers=set(),
                    drone_customers=set(),
                )
            )

        unserved = set(instance.customers) - assigned_customers
        candidate_solution = Solution(vehicle_pairs=vehicle_pairs, unserved_customers=unserved)

        # Guarantee complete assignment for downstream ALNS calls.
        if candidate_solution.unserved_customers:
            return self._etprc_builder.build_initial_solution(instance, params, rng)
        return candidate_solution

    def update_pheromone(
        self,
        state: ACOState,
        best_solution: Solution | None,
        best_objective: float | None,
        params: ProblemParameters,
        iteration_best_solution: Solution | None = None,
        iteration_best_objective: float | None = None,
        global_best_solution: Solution | None = None,
        global_best_objective: float | None = None,
    ) -> None:
        """Apply evaporation + MMAS bounds + mixed iteration/global deposition."""
        evaporation_rate = float(getattr(params.aco, "evaporation_rate", 0.0))
        evaporation_rate = min(max(evaporation_rate, 0.0), 1.0)
        tau_min, tau_max = self._resolve_tau_bounds(params)

        # Global evaporation.
        for edge in list(state.pheromone.keys()):
            evaporated = state.pheromone[edge] * (1.0 - evaporation_rate)
            state.pheromone[edge] = min(max(evaporated, tau_min), tau_max)

        if iteration_best_solution is None:
            iteration_best_solution = best_solution
        if iteration_best_objective is None:
            iteration_best_objective = best_objective
        if global_best_solution is None:
            global_best_solution = best_solution
        if global_best_objective is None:
            global_best_objective = best_objective

        pheromone_q = float(getattr(params.aco, "pheromone_q", getattr(params.aco, "pheromone_deposit", 1.0)))
        global_best_weight = max(0.0, float(getattr(params.aco, "global_best_weight", 0.6)))
        initial_pheromone = float(
            getattr(params.aco, "pheromone_init", getattr(params.aco, "initial_pheromone", 1.0))
        )

        def _deposit(solution: Solution | None, objective: float | None, scale: float) -> None:
            if solution is None or objective is None or objective <= 0.0 or scale <= 0.0:
                return
            delta = scale * pheromone_q / max(objective, 1e-9)
            for edge in self._truck_edges(solution):
                if edge not in state.pheromone:
                    seeded = initial_pheromone * (1.0 - evaporation_rate)
                    state.pheromone[edge] = min(max(seeded, tau_min), tau_max)
                updated = state.pheromone[edge] + delta
                state.pheromone[edge] = min(max(updated, tau_min), tau_max)

        _deposit(iteration_best_solution, iteration_best_objective, 1.0)
        _deposit(global_best_solution, global_best_objective, global_best_weight)

    def advance_beta_schedule(self, state: ACOState, params: ProblemParameters) -> float:
        """Update dynamic beta and return the new value for objective evaluation."""
        gamma_beta = max(0.0, params.beta_schedule.gamma_beta)
        beta_max = params.beta_schedule.beta_max
        state.current_beta = min(state.current_beta * gamma_beta, beta_max)
        return state.current_beta



class PureACOOptimizer:
    """
    纯 ACO 优化器：保留蚁群框架（多轮迭代、多蚂蚁构造、信息素更新），
    去掉 ALNS 局部搜索。每只蚂蚁只做“E-TPRC 构造 → 评估目标值”，不做改进。
    用于实验一中与 ACO-ALNS 的消融对比，验证 ALNS 局部搜索的贡献。
    """

    def __init__(self, etprc_builder: ETPRCBuilder | None = None):
        # 统一保存迭代历史，供收敛曲线和实验报告使用。
        self.iteration_history: list[dict[str, object]] = []
        # 暴露最终信息素矩阵，便于测试确认每轮都发生更新。
        self.last_pheromone: dict[tuple[int, int], float] = {}

        # 复用现有 ACOALNSSolver 的构造与信息素更新实现，避免纯 ACO 与基线口径漂移。
        self._etprc_builder = etprc_builder or ETPRCBuilder()
        self._helper_solver = ACOALNSSolver(
            etprc_builder=self._etprc_builder,
            alns_optimizer=ALNSOptimizer(_build_default_alns_operator_set()),
        )

    def _initialize_pheromone(
        self,
        instance: ProblemInstance,
        params: ProblemParameters,
    ) -> dict[tuple[int, int], float]:
        """初始化信息素矩阵，规则与 ACOALNSSolver.solve 完全一致。"""
        initial_pheromone = float(
            getattr(params.aco, "pheromone_init", getattr(params.aco, "initial_pheromone", 1.0)),
        )
        pheromone = {arc: initial_pheromone for arc in instance.truck_distance_km}
        if not pheromone:
            nodes = [instance.depot_id, *instance.customers.keys()]
            pheromone = {(i, j): initial_pheromone for i in nodes for j in nodes}
        return pheromone

    def _construct_solution(
        self,
        instance: ProblemInstance,
        params: ProblemParameters,
        home_status: HomeStatusMap,
        pheromone: dict[tuple[int, int], float],
        rng: random.Random,
    ) -> Solution:
        """复用 ACOALNSSolver.construct_ant_solution，确保信息素引导构造保持一致。"""
        state = ACOState(
            pheromone=pheromone,
            current_beta=float(params.beta_schedule.beta_init),
        )
        return self._helper_solver.construct_ant_solution(
            instance=instance,
            params=params,
            home_status=home_status,
            state=state,
            rng=rng,
        )

    def _update_pheromone(
        self,
        pheromone: dict[tuple[int, int], float],
        best_solution: Solution | None,
        best_objective: float | None,
        params: ProblemParameters,
        iteration_best_solution: Solution | None = None,
        iteration_best_objective: float | None = None,
        global_best_solution: Solution | None = None,
        global_best_objective: float | None = None,
    ) -> None:
        """复用 ACOALNSSolver.update_pheromone，保证蒸发/强化公式完全一致。"""
        state = ACOState(
            pheromone=pheromone,
            current_beta=float(params.beta_schedule.beta_init),
        )
        self._helper_solver.update_pheromone(
            state=state,
            best_solution=best_solution,
            best_objective=best_objective,
            params=params,
            iteration_best_solution=iteration_best_solution,
            iteration_best_objective=iteration_best_objective,
            global_best_solution=global_best_solution,
            global_best_objective=global_best_objective,
        )

    def solve(
        self,
        instance: ProblemInstance,
        params: ProblemParameters,
        home_status: HomeStatusMap,
        rng: random.Random | None = None,
    ) -> Solution:
        """提供与现有离线求解器一致的入口，便于 main.py 统一调度。"""
        random_state = rng or random.Random(0)
        # 先构造统一的 E-TPRC 初始解，再进入纯 ACO 迭代。
        initial_solution = self._etprc_builder.build_initial_solution(instance, params, random_state)
        best_solution, _, history = self.optimize(
            instance=instance,
            initial_solution=initial_solution,
            params=params,
            home_status=home_status,
            rng=random_state,
        )
        self.iteration_history = history
        return best_solution

    def optimize(
        self,
        instance: ProblemInstance,
        initial_solution: Solution,
        params: ProblemParameters,
        home_status: HomeStatusMap,
        rng: random.Random | None = None,
    ) -> tuple[Solution, float, list[dict[str, object]]]:
        """执行纯 ACO：多轮迭代 + 多蚂蚁构造 + 信息素更新（无 ALNS）。"""
        random_state = rng or random.Random(0)
        pheromone = self._initialize_pheromone(instance, params)

        best_solution = deepcopy(initial_solution)
        best_objective = compute_objective(
            instance,
            best_solution,
            home_status,
            params,
        )

        iteration_history: list[dict[str, object]] = []
        max_iterations = max(0, int(getattr(params.aco, "max_iterations", getattr(params.aco, "MaxIter", 0))))
        ant_count = max(1, int(getattr(params.aco, "ant_count", 1)))
        restart_threshold_raw = getattr(params.aco, "stagnation_restart_threshold", None)
        if restart_threshold_raw is None:
            restart_threshold = max(1, int(getattr(params.aco, "no_improve_max", getattr(params.aco, "NoImprove_max", 10))))
        else:
            try:
                restart_threshold = max(1, int(restart_threshold_raw))
            except (TypeError, ValueError):
                restart_threshold = 10
        no_improve_count = 0

        # 纯 ACO 也沿用与 ACO-ALNS 相同的并行蚂蚁机制，确保性能对比公平。
        parallel_ants = _ACO_PARALLEL_OVERRIDES.get(id(params.aco), bool(getattr(params.aco, "parallel_ants", True)))
        try:
            cpu_count = int(mp.cpu_count())
        except (NotImplementedError, ValueError):
            cpu_count = 1
        n_workers = min(ant_count, max(1, cpu_count - 1))
        use_parallel = parallel_ants and ant_count > 1 and n_workers > 1

        base_seed_raw = getattr(params.aco, "seed", getattr(params.aco, "random_seed", None))
        if base_seed_raw is None:
            base_seed = random_state.getrandbits(31)
        else:
            try:
                base_seed = int(base_seed_raw)
            except (TypeError, ValueError):
                base_seed = random_state.getrandbits(31)

        pool = None
        if use_parallel:
            try:
                ctx = mp.get_context("spawn")
                pool = ctx.Pool(processes=n_workers)
            except (OSError, RuntimeError, ValueError) as exc:
                logging.warning("Pure ACO ant parallel disabled, fallback to serial: %s", exc)
                pool = None

        try:
            for aco_iter in range(max_iterations):
                aco_iteration = aco_iter + 1
                iteration_best_solution: Solution | None = None
                iteration_best_objective: float | None = None

                ant_args_list: list[
                    tuple[
                        ProblemInstance,
                        ProblemParameters,
                        dict[tuple[int, int], float],
                        int,
                        int,
                        int,
                        dict[int, bool],
                        bool,
                    ]
                ] = []
                for ant in range(ant_count):
                    ant_seed = _derive_ant_seed(base_seed, aco_iteration, ant)
                    ant_args_list.append(
                        (
                            instance,
                            params,
                            pheromone,
                            ant_seed,
                            aco_iteration,
                            ant,
                            dict(home_status),
                            pool is not None,
                        )
                    )

                if pool is not None:
                    try:
                        ant_results = pool.map(_run_single_pure_aco_ant, ant_args_list)
                    except Exception as exc:  # noqa: BLE001
                        logging.warning("Pure ACO ant parallel run failed, fallback to serial this iteration: %s", exc)
                        serial_args = [(*args[:-1], False) for args in ant_args_list]
                        ant_results = [_run_single_pure_aco_ant(args) for args in serial_args]
                else:
                    ant_results = []
                    for ant in range(ant_count):
                        ant_seed = _derive_ant_seed(base_seed, aco_iteration, ant)
                        ant_solution = self._construct_solution(
                            instance=instance,
                            params=params,
                            home_status=home_status,
                            pheromone=pheromone,
                            rng=random.Random(int(ant_seed)),
                        )
                        ant_objective = compute_objective(
                            instance,
                            ant_solution,
                            home_status,
                            params,
                        )
                        ant_results.append(
                            {
                                "ok": True,
                                "ant_index": int(ant),
                                "aco_iteration": int(aco_iteration),
                                "solution": ant_solution,
                                "objective": float(ant_objective),
                            }
                        )

                for ant_result in ant_results:
                    if not ant_result.get("ok", False):
                        continue

                    ant_solution = ant_result["solution"]
                    ant_objective = float(ant_result["objective"])
                    ant_index = int(ant_result.get("ant_index", 0))
                    global_step = aco_iter * ant_count + ant_index

                    if iteration_best_objective is None or ant_objective < iteration_best_objective:
                        iteration_best_objective = ant_objective
                        iteration_best_solution = deepcopy(ant_solution)

                    best_so_far = min(best_objective, ant_objective)
                    iteration_history.append(
                        {
                            "global_step": int(global_step),
                            "aco_iter": int(aco_iteration),
                            "ant": int(ant_index),
                            "objective": float(ant_objective),
                            "best_objective": float(best_so_far),
                            # 兼容现有可视化函数需要的字段命名。
                            "current_cost": float(ant_objective),
                            "best_cost": float(best_so_far),
                        }
                    )

                if iteration_best_solution is None or iteration_best_objective is None:
                    continue

                if iteration_best_objective < best_objective:
                    best_objective = float(iteration_best_objective)
                    best_solution = deepcopy(iteration_best_solution)
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                self._update_pheromone(
                    pheromone=pheromone,
                    best_solution=best_solution,
                    best_objective=float(best_objective),
                    params=params,
                    iteration_best_solution=iteration_best_solution,
                    iteration_best_objective=float(iteration_best_objective),
                    global_best_solution=best_solution,
                    global_best_objective=float(best_objective),
                )

                if no_improve_count >= restart_threshold:
                    restart_state = ACOState(
                        pheromone=pheromone,
                        current_beta=float(params.beta_schedule.beta_init),
                    )
                    self._helper_solver._partial_pheromone_restart(
                        state=restart_state,
                        params=params,
                        rng=random_state,
                        protected_edges=self._helper_solver._truck_edges(best_solution),
                    )
                    no_improve_count = 0
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        self.iteration_history = iteration_history
        self.last_pheromone = dict(pheromone)
        return best_solution, float(best_objective), iteration_history

















