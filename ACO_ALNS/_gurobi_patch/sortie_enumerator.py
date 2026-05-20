from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Optional

from model_types import ProblemData, SortieCandidate


@dataclass
class EnumerationStats:
    generated_feasible: int
    selected_for_model: int
    pair_limit_per_order: int | None
    global_cap: int | None


def _initial_sortie_load(data: ProblemData, ordered_customers: tuple[int, ...]) -> float:
    """Initial drone payload: sum of delivery parcels carried at launch."""
    load = 0.0
    for customer_id in ordered_customers:
        customer_type = str(data.customer_type.get(customer_id, 'delivery')).lower()
        if customer_type == 'delivery':
            load += data.weights[customer_id]
    return load


def _update_sortie_load(data: ProblemData, load: float, customer_id: int) -> float:
    """Update payload after servicing one customer (deterministic home=1)."""
    customer_type = str(data.customer_type.get(customer_id, 'delivery')).lower()
    weight = data.weights[customer_id]
    if customer_type == 'delivery':
        return load - weight
    if customer_type == 'pickup':
        return load + weight
    return load


def _compute_dynamic_base_energy(
    data: ProblemData,
    launch: int,
    ordered_customers: tuple[int, ...],
    recovery: int,
) -> tuple[float, float]:
    """Compute sortie base energy (without recovery hover) with dynamic load."""
    eta = data.eta_wh_per_kg_min
    drone_speed = data.drone_speed
    drone_empty_weight = data.drone_empty_weight

    load = _initial_sortie_load(data, ordered_customers)
    energy_wh = 0.0

    current_node = launch
    current_time = data.drone_launch_time

    for customer_id in ordered_customers:
        travel_time = data.distance[(current_node, customer_id)] / drone_speed
        total_weight = drone_empty_weight + load
        energy_wh += eta * total_weight * travel_time

        arrival_time = current_time + travel_time
        ready_time = data.time_windows[customer_id][0]
        wait_time = max(0.0, ready_time - arrival_time)
        energy_wh += eta * total_weight * wait_time

        service_time = data.service_times[customer_id]
        energy_wh += eta * total_weight * service_time

        current_time = arrival_time + wait_time + service_time
        load = _update_sortie_load(data, load, customer_id)
        current_node = customer_id

    tail_travel_time = data.distance[(current_node, recovery)] / drone_speed
    energy_wh += eta * (drone_empty_weight + load) * tail_travel_time

    return energy_wh, load


def _build_candidate(
    data: ProblemData,
    launch: int,
    ordered_customers: tuple[int, ...],
    recovery: int,
    payload_weight: float,
    candidate_id: int,
) -> SortieCandidate:
    if launch == recovery:
        raise ValueError('launch and recovery must differ')

    service_offsets: dict[int, float] = {}
    distance = data.distance

    first = ordered_customers[0]
    flight_distance = distance[(launch, first)]
    offset = data.drone_launch_time + distance[(launch, first)] / data.drone_speed
    service_offsets[first] = offset

    for idx in range(len(ordered_customers) - 1):
        i = ordered_customers[idx]
        j = ordered_customers[idx + 1]
        flight_distance += distance[(i, j)]
        offset += data.service_times[i] + distance[(i, j)] / data.drone_speed
        service_offsets[j] = offset

    last = ordered_customers[-1]
    flight_distance += distance[(last, recovery)]

    flight_time = flight_distance / data.drone_speed
    duration = (
        data.drone_launch_time
        + flight_time
        + sum(data.service_times[cid] for cid in ordered_customers)
        + data.drone_recovery_time
    )

    energy_wh, _ = _compute_dynamic_base_energy(
        data=data,
        launch=launch,
        ordered_customers=ordered_customers,
        recovery=recovery,
    )

    score = energy_wh + 0.1 * flight_distance + 0.05 * (
        data.distance[(launch, first)] + data.distance[(last, recovery)]
    )

    return SortieCandidate(
        candidate_id=candidate_id,
        launch=launch,
        customers=ordered_customers,
        recovery=recovery,
        payload_weight=payload_weight,
        flight_distance=flight_distance,
        flight_time=flight_time,
        duration=duration,
        energy_wh=energy_wh,
        service_offsets=service_offsets,
        score=score,
    )


def enumerate_sorties(
    data: ProblemData,
    pair_limit_per_order: int | None = None,
    model_candidate_cap: int | None = None,
    min_candidates_per_customer: int = 8,
) -> tuple[list[SortieCandidate], EnumerationStats]:
    all_customers = list(data.customers)
    drone_customers = list(data.drone_customers)

    raw_candidates: list[SortieCandidate] = []
    temp_id = 0

    for size in range(1, data.max_per_sortie + 1):
        for customer_group in combinations(drone_customers, size):
            payload_weight = sum(data.weights[cid] for cid in customer_group)
            if payload_weight > data.drone_capacity:
                continue

            group_set = set(customer_group)
            launch_nodes = [0] + [cid for cid in all_customers if cid not in group_set]
            recovery_nodes = [cid for cid in all_customers if cid not in group_set] + [data.depot_end]

            for ordered in permutations(customer_group):
                ordered_customers = tuple(int(x) for x in ordered)
                local_candidates: list[SortieCandidate] = []

                for launch in launch_nodes:
                    for recovery in recovery_nodes:
                        if launch == recovery:
                            continue
                        candidate = _build_candidate(
                            data=data,
                            launch=launch,
                            ordered_customers=ordered_customers,
                            recovery=recovery,
                            payload_weight=payload_weight,
                            candidate_id=temp_id,
                        )
                        temp_id += 1

                        if candidate.energy_wh <= data.battery_capacity:
                            local_candidates.append(candidate)

                local_candidates.sort(key=lambda c: c.score)
                if pair_limit_per_order is None:
                    raw_candidates.extend(local_candidates)
                else:
                    raw_candidates.extend(local_candidates[:pair_limit_per_order])

    raw_candidates.sort(key=lambda c: c.score)

    selected: list[SortieCandidate] = []
    selected_keys: set[tuple[int, tuple[int, ...], int]] = set()
    target_cap: Optional[int] = model_candidate_cap if model_candidate_cap and model_candidate_cap > 0 else None

    by_customer: dict[int, list[SortieCandidate]] = defaultdict(list)
    for cand in raw_candidates:
        for cid in cand.customers:
            by_customer[cid].append(cand)

    for cid in drone_customers:
        for cand in by_customer.get(cid, [])[:min_candidates_per_customer]:
            key = (cand.launch, cand.customers, cand.recovery)
            if key not in selected_keys:
                selected.append(cand)
                selected_keys.add(key)
            if target_cap is not None and len(selected) >= target_cap:
                break
        if target_cap is not None and len(selected) >= target_cap:
            break

    if target_cap is None or len(selected) < target_cap:
        for cand in raw_candidates:
            key = (cand.launch, cand.customers, cand.recovery)
            if key in selected_keys:
                continue
            selected.append(cand)
            selected_keys.add(key)
            if target_cap is not None and len(selected) >= target_cap:
                break

    reindexed: list[SortieCandidate] = []
    for new_id, cand in enumerate(selected):
        reindexed.append(
            SortieCandidate(
                candidate_id=new_id,
                launch=cand.launch,
                customers=cand.customers,
                recovery=cand.recovery,
                payload_weight=cand.payload_weight,
                flight_distance=cand.flight_distance,
                flight_time=cand.flight_time,
                duration=cand.duration,
                energy_wh=cand.energy_wh,
                service_offsets=dict(cand.service_offsets),
                score=cand.score,
            )
        )

    stats = EnumerationStats(
        generated_feasible=len(raw_candidates),
        selected_for_model=len(reindexed),
        pair_limit_per_order=pair_limit_per_order,
        global_cap=model_candidate_cap,
    )
    return reindexed, stats
