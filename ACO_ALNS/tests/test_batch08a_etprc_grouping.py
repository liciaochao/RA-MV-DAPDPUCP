from __future__ import annotations

import random
from dataclasses import replace

from spd.etprc import ETPRCBuilder
from spd.types import ProblemInstance


def test_customer_grouping_returns_k_nonempty_sets_and_covers_all_customers(
    sample_instance,
    sample_params,
) -> None:
    # Force multiple groups to validate K-way partition behavior.
    instance = ProblemInstance(
        depot_id=sample_instance.depot_id,
        vehicle_pair_count=2,
        customers=sample_instance.customers,
        delivery_customers=sample_instance.delivery_customers,
        pickup_customers=sample_instance.pickup_customers,
        truck_distance_km=sample_instance.truck_distance_km,
        drone_distance_km=sample_instance.drone_distance_km,
    )
    params = replace(
        sample_params,
        vehicle=replace(sample_params.vehicle, truck_capacity=3.0),
    )

    builder = ETPRCBuilder()
    groups = builder._customer_grouping(instance, params, random.Random(42))

    assert len(groups) == instance.vehicle_pair_count
    assert all(len(group) > 0 for group in groups)

    covered = set().union(*groups)
    assert covered == set(instance.customers.keys())



def test_customer_grouping_delivery_weight_within_truck_capacity(
    sample_instance,
    sample_params,
) -> None:
    instance = ProblemInstance(
        depot_id=sample_instance.depot_id,
        vehicle_pair_count=2,
        customers=sample_instance.customers,
        delivery_customers=sample_instance.delivery_customers,
        pickup_customers=sample_instance.pickup_customers,
        truck_distance_km=sample_instance.truck_distance_km,
        drone_distance_km=sample_instance.drone_distance_km,
    )
    params = replace(
        sample_params,
        vehicle=replace(sample_params.vehicle, truck_capacity=3.0),
    )

    builder = ETPRCBuilder()
    groups = builder._customer_grouping(instance, params, random.Random(7))

    for group in groups:
        delivery_weight = sum(
            instance.customers[c].weight for c in group if c in instance.delivery_customers
        )
        assert delivery_weight <= params.vehicle.truck_capacity + 1e-9



def test_build_truck_route_starts_and_ends_at_depot_and_contains_group(
    sample_instance,
    sample_params,
) -> None:
    builder = ETPRCBuilder()
    group = {1, 2, 4}

    route = builder._build_truck_route(group, sample_instance, sample_params)

    assert route[0] == sample_instance.depot_id
    assert route[-1] == sample_instance.depot_id
    assert set(route[1:-1]) == group
    assert len(route) == len(group) + 2
