from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import random

import spd.online as online_module
from spd.aco_alns import DestroyResult
from spd.online import LightweightALNSOptimizer, OnlineReplanner
from spd.types import CustomerNotHomeEvent, ServiceMode, Solution, Sortie, VehiclePairSolution


def _pair_snapshot(pair: VehiclePairSolution) -> tuple:
    return (
        tuple(pair.truck_route),
        tuple((s.launch_node, s.recovery_node, tuple(s.customers)) for s in pair.sorties),
        tuple(sorted(pair.truck_customers)),
        tuple(sorted(pair.drone_customers)),
    )


def test_online_alns_respects_frozen_customers(
    sample_instance,
    sample_params,
    sample_pair_solution,
    all_home_status,
    monkeypatch,
) -> None:
    """Frozen customers should never be removed by online ALNS destroy steps."""
    optimizer = LightweightALNSOptimizer()
    initial_pair = deepcopy(sample_pair_solution)
    params = replace(sample_params, online_alns=replace(sample_params.online_alns, iterations=5, remove_count=1))

    repair_called = {"flag": False}

    def fake_destroy(solution, instance, params, remove_count, home_status, rng):
        return DestroyResult(partial_solution=deepcopy(solution), removed_customers=[1, 2])

    def fake_repair(*args, **kwargs):
        repair_called["flag"] = True
        raise AssertionError("repair should not be called when frozen customers are removed")

    monkeypatch.setattr(online_module, "d1_random_removal", fake_destroy)
    monkeypatch.setattr(online_module, "d2_worst_removal", fake_destroy)
    monkeypatch.setattr(online_module, "r1_greedy_insertion", fake_repair)
    monkeypatch.setattr(online_module, "r2_regret_insertion", fake_repair)

    optimized_pair = optimizer.optimize_remaining_path(
        sample_instance,
        initial_pair,
        all_home_status,
        params,
        random.Random(123),
        frozen_customers={1, 2},
    )

    assert repair_called["flag"] is False
    assert _pair_snapshot(optimized_pair) == _pair_snapshot(initial_pair)


def test_optimize_remaining_path_default_matches_empty_frozen(
    sample_instance,
    sample_params,
    sample_pair_solution,
    all_home_status,
) -> None:
    """Backward compatibility: omitting frozen_customers matches an empty freeze set."""
    optimizer = LightweightALNSOptimizer()
    params = replace(sample_params, online_alns=replace(sample_params.online_alns, iterations=8, remove_count=1))

    out_default = optimizer.optimize_remaining_path(
        sample_instance,
        deepcopy(sample_pair_solution),
        all_home_status,
        params,
        random.Random(42),
    )
    out_empty = optimizer.optimize_remaining_path(
        sample_instance,
        deepcopy(sample_pair_solution),
        all_home_status,
        params,
        random.Random(42),
        frozen_customers=set(),
    )

    assert _pair_snapshot(out_default) == _pair_snapshot(out_empty)


class _FreezeAwareOnlineALNS:
    def __init__(self) -> None:
        self.last_frozen: set[int] = set()

    def optimize_remaining_path(
        self,
        instance,
        pair_solution,
        home_status,
        params,
        rng=None,
        frozen_customers: set[int] | None = None,
    ):
        self.last_frozen = set(frozen_customers or set())
        candidate = deepcopy(pair_solution)
        if candidate.sorties and len(candidate.sorties[0].customers) >= 2:
            first = candidate.sorties[0].customers[0]
            second = candidate.sorties[0].customers[1]
            if not ({first, second} <= self.last_frozen):
                candidate.sorties[0].customers[0], candidate.sorties[0].customers[1] = (
                    second,
                    first,
                )
        return candidate


def test_drone_event_freezes_visited_prefix(
    sample_instance,
    sample_params,
    all_home_status,
) -> None:
    """Drone event should freeze customers visited before the failed customer."""
    pair = VehiclePairSolution(
        pair_id=1,
        truck_route=[0, 1, 3, 5, 0],
        sorties=[Sortie(launch_node=1, recovery_node=5, customers=[2, 4, 6])],
        truck_customers={1, 3, 5},
        drone_customers={2, 4, 6},
    )
    solution = Solution(vehicle_pairs=[pair], unserved_customers=set())

    spy_alns = _FreezeAwareOnlineALNS()
    replanner = OnlineReplanner(spy_alns)
    home_status = dict(all_home_status)

    event = CustomerNotHomeEvent(
        pair_id=1,
        customer_id=6,
        service_mode=ServiceMode.DRONE,
        event_time=20.0,
        sortie_index=0,
    )

    updated = replanner.handle_event(
        sample_instance,
        solution,
        event,
        home_status,
        sample_params,
        random.Random(7),
    )

    assert {2, 4}.issubset(spy_alns.last_frozen)

    result_sortie = next(s for s in updated.vehicle_pairs[0].sorties if 2 in s.customers and 4 in s.customers)
    assert result_sortie.customers.index(2) < result_sortie.customers.index(4)
