from __future__ import annotations

from dataclasses import replace
import random

from spd.etprc import ETPRCBuilder
from spd.types import ProblemInstance


def _clone_instance_all_home_prob_one(base: ProblemInstance) -> ProblemInstance:
    customers = {
        customer_id: replace(customer, home_probabilities=(1.0,) * 8)
        for customer_id, customer in base.customers.items()
    }
    return ProblemInstance(
        depot_id=base.depot_id,
        vehicle_pair_count=base.vehicle_pair_count,
        customers=customers,
        delivery_customers=base.delivery_customers,
        pickup_customers=base.pickup_customers,
        truck_distance_km=base.truck_distance_km,
        drone_distance_km=base.drone_distance_km,
    )


def test_extract_sorties_launch_recovery_not_depot_and_customer_sets(
    sample_instance,
    sample_params,
    all_home_status,
) -> None:
    builder = ETPRCBuilder()

    instance = _clone_instance_all_home_prob_one(sample_instance)
    params = replace(
        sample_params,
        cost=replace(sample_params.cost, fixed_pair_cost=0.0, truck_cost_per_km=5.0, drone_energy_cost_per_wh=0.01),
        beta_schedule=replace(sample_params.beta_schedule, beta_init=0.0),
    )

    route = [0, 1, 2, 4, 5, 0]
    updated_route, sorties, truck_customers, drone_customers = builder._extract_sorties(
        route,
        instance,
        params,
        all_home_status,
    )

    for sortie in sorties:
        assert sortie.launch_node != instance.depot_id
        assert sortie.recovery_node != instance.depot_id

    # Drone customers must be removed from the updated truck route.
    assert drone_customers.isdisjoint(set(updated_route))

    # Returned customer sets must match the updated route / removed set.
    assert truck_customers == {node for node in updated_route if node != instance.depot_id}
    assert drone_customers.issubset({1, 2, 4, 5})



def test_optimize_sortie_order_returns_permutation_covering_input(
    sample_instance,
    sample_params,
    all_home_status,
) -> None:
    builder = ETPRCBuilder()

    customers = [2, 4]
    order = builder._optimize_sortie_order(
        customers=customers,
        launch=1,
        recovery=5,
        instance=sample_instance,
        params=sample_params,
        home_status=all_home_status,
    )

    assert set(order) == set(customers)
    assert len(order) == len(customers)



def test_extract_sorties_output_shape_stable(
    sample_instance,
    sample_params,
    all_home_status,
) -> None:
    builder = ETPRCBuilder()

    route = [0, 1, 3, 5, 0]
    updated_route, sorties, truck_customers, drone_customers = builder._extract_sorties(
        route,
        sample_instance,
        sample_params,
        all_home_status,
    )

    assert updated_route[0] == sample_instance.depot_id
    assert updated_route[-1] == sample_instance.depot_id
    assert isinstance(sorties, list)
    assert isinstance(truck_customers, set)
    assert isinstance(drone_customers, set)
