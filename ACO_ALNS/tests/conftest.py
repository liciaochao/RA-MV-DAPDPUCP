from __future__ import annotations

from dataclasses import replace
import math
import random

import pytest

from spd.config import (
    ACOParameters,
    ALNSParameters,
    BetaScheduleParameters,
    ConstraintParameters,
    CostParameters,
    ETPRCParameters,
    EnergyParameters,
    OnlineALNSParameters,
    ProblemParameters,
    TimeParameters,
    VehicleParameters,
)
from spd.types import Customer, ProblemInstance, Sortie, VehiclePairSolution
from spd.types import CustomerType


def _build_distance_matrix(coords: dict[int, tuple[float, float]], scale: float = 1.0) -> dict[tuple[int, int], float]:
    """Build a full directed distance matrix for all node pairs."""
    matrix: dict[tuple[int, int], float] = {}
    for i, (x1, y1) in coords.items():
        for j, (x2, y2) in coords.items():
            if i == j:
                matrix[(i, j)] = 0.0
            else:
                matrix[(i, j)] = scale * math.hypot(x2 - x1, y2 - y1)
    return matrix


@pytest.fixture
def sample_customers() -> dict[int, Customer]:
    """Six customers: 1-3 delivery, 4-6 pickup."""
    return {
        1: Customer(
            customer_id=1,
            x=1.0,
            y=0.0,
            customer_type=CustomerType.DELIVERY,
            weight=1.2,
            time_window=(0.0, 120.0),
            service_time=6.0,
            home_probabilities=(0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60),
        ),
        2: Customer(
            customer_id=2,
            x=4.0,
            y=0.0,
            customer_type=CustomerType.DELIVERY,
            weight=0.8,
            time_window=(15.0, 140.0),
            service_time=5.0,
            home_probabilities=(0.70, 0.75, 0.80, 0.82, 0.78, 0.74, 0.70, 0.68),
        ),
        3: Customer(
            customer_id=3,
            x=8.0,
            y=0.0,
            customer_type=CustomerType.DELIVERY,
            weight=2.5,
            time_window=(30.0, 220.0),
            service_time=7.0,
            home_probabilities=(0.90, 0.88, 0.86, 0.84, 0.82, 0.80, 0.78, 0.76),
        ),
        4: Customer(
            customer_id=4,
            x=4.0,
            y=3.0,
            customer_type=CustomerType.PICKUP,
            weight=1.5,
            time_window=(30.0, 200.0),
            service_time=4.0,
            home_probabilities=(0.60, 0.55, 0.50, 0.45, 0.50, 0.55, 0.60, 0.65),
        ),
        5: Customer(
            customer_id=5,
            x=7.0,
            y=3.0,
            customer_type=CustomerType.PICKUP,
            weight=3.0,
            time_window=(60.0, 300.0),
            service_time=8.0,
            home_probabilities=(0.80, 0.78, 0.76, 0.74, 0.72, 0.70, 0.68, 0.66),
        ),
        6: Customer(
            customer_id=6,
            x=1.0,
            y=4.0,
            customer_type=CustomerType.PICKUP,
            weight=0.6,
            time_window=(10.0, 160.0),
            service_time=6.0,
            home_probabilities=(0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64),
        ),
    }


@pytest.fixture
def sample_instance(sample_customers: dict[int, Customer]) -> ProblemInstance:
    """Problem instance with full depot/customer distance matrices."""
    coords: dict[int, tuple[float, float]] = {0: (0.0, 0.0)}
    for customer_id, customer in sample_customers.items():
        coords[customer_id] = (customer.x, customer.y)

    truck_distance_km = _build_distance_matrix(coords, scale=1.2)
    drone_distance_km = _build_distance_matrix(coords, scale=1.0)

    return ProblemInstance(
        depot_id=0,
        vehicle_pair_count=1,
        customers=sample_customers,
        delivery_customers=frozenset({1, 2, 3}),
        pickup_customers=frozenset({4, 5, 6}),
        truck_distance_km=truck_distance_km,
        drone_distance_km=drone_distance_km,
    )


@pytest.fixture
def sample_params() -> ProblemParameters:
    """Default parameter bundle used across tests."""
    return ProblemParameters(
        time=TimeParameters(depot_earliest=0.0, depot_latest=480.0),
        vehicle=VehicleParameters(
            truck_capacity=100.0,
            drone_capacity=5.0,
            drone_empty_weight=2.0,
            truck_speed=0.5,
            drone_speed=1.0,
        ),
        energy=EnergyParameters(
            drone_battery_capacity=500.0,
            eta_wh_per_kg_min=0.5,
        ),
        cost=CostParameters(
            truck_cost_per_km=1.0,
            fixed_pair_cost=50.0,
            drone_energy_cost_per_wh=0.1,
        ),
        beta_schedule=BetaScheduleParameters(
            beta_init=10.0,
            beta_max=1000.0,
            gamma_beta=1.1,
        ),
        constraints=ConstraintParameters(
            max_truck_wait_time=30.0,
            max_customers_per_sortie=3,
        ),
        aco=ACOParameters(
            max_iterations=100,
            ant_count=20,
            alpha_aco=1.0,
            beta_aco=2.0,
            evaporation_rate=0.2,
            pheromone_q=10.0,
            pheromone_init=1.0,
            no_improve_max=30,
        ),
        alns=ALNSParameters(
            iterations=80,
            remove_count_min=1,
            remove_count_max=3,
            sa_initial_temperature=100.0,
            sa_cooling_rate=0.98,
            operator_weight_init=1.0,
            reward_global_best=33.0,
            reward_improve=13.0,
            reward_accept_worse=9.0,
            reaction_factor=0.2,
            cross_group_frequency=10,
            cross_group_remove_count=1,
        ),
        etprc=ETPRCParameters(
            k_cluster=1,
            max_iter_kmeans=100,
            omega_distance=1.0,
            omega_time_window=1.0,
            omega_balance=1.0,
        ),
        online_alns=OnlineALNSParameters(
            iterations=20,
            remove_count=2,
        ),
    )


@pytest.fixture
def small_aco_params(sample_params: ProblemParameters) -> ProblemParameters:
    """Small ACO/ALNS iteration counts for fast integration tests."""
    return replace(
        sample_params,
        aco=replace(sample_params.aco, max_iterations=3, ant_count=2),
        alns=replace(sample_params.alns, iterations=5),
    )

@pytest.fixture
def all_home_status(sample_customers: dict[int, Customer]) -> dict[int, bool]:
    """Offline default: all customers are at home."""
    return {customer_id: True for customer_id in sample_customers}


@pytest.fixture
def sample_pair_solution() -> VehiclePairSolution:
    """Single pair with one sortie used by timeline tests."""
    return VehiclePairSolution(
        pair_id=1,
        truck_route=[0, 1, 3, 5, 0],
        sorties=[Sortie(launch_node=1, recovery_node=5, customers=[2, 4])],
        truck_customers={1, 3, 5},
        drone_customers={2, 4},
    )

@pytest.fixture
def sample_instance_two_pairs(sample_instance: ProblemInstance) -> ProblemInstance:
    """Variant of sample instance configured with two vehicle pairs."""
    return ProblemInstance(
        depot_id=sample_instance.depot_id,
        vehicle_pair_count=2,
        customers=sample_instance.customers,
        delivery_customers=sample_instance.delivery_customers,
        pickup_customers=sample_instance.pickup_customers,
        truck_distance_km=sample_instance.truck_distance_km,
        drone_distance_km=sample_instance.drone_distance_km,
    )


@pytest.fixture
def sample_solution_two_pairs() -> "Solution":
    """Two-pair seed solution used for cross-group operator tests."""
    from spd.types import Solution

    pair_1 = VehiclePairSolution(
        pair_id=1,
        truck_route=[0, 1, 4, 6, 0],
        sorties=[],
        truck_customers={1, 4, 6},
        drone_customers=set(),
    )
    pair_2 = VehiclePairSolution(
        pair_id=2,
        truck_route=[0, 2, 3, 5, 0],
        sorties=[],
        truck_customers={2, 3, 5},
        drone_customers=set(),
    )
    return Solution(vehicle_pairs=[pair_1, pair_2], unserved_customers=set())




@pytest.fixture
def large_customers() -> dict[int, Customer]:
    """Twelve customers for two-pair end-to-end integration tests."""
    customers: dict[int, Customer] = {}

    # Delivery customers: 1-6.
    for idx, customer_id in enumerate(range(1, 7), start=1):
        customers[customer_id] = Customer(
            customer_id=customer_id,
            x=float(idx * 2),
            y=0.5 if idx % 2 == 0 else 0.0,
            customer_type=CustomerType.DELIVERY,
            weight=1.0 + 0.3 * (idx % 3),
            time_window=(0.0 + 10.0 * idx, 220.0 + 10.0 * idx),
            service_time=5.0 + (idx % 3),
            home_probabilities=(0.90, 0.88, 0.86, 0.84, 0.82, 0.80, 0.78, 0.76),
        )

    # Pickup customers: 7-12.
    for idx, customer_id in enumerate(range(7, 13), start=1):
        customers[customer_id] = Customer(
            customer_id=customer_id,
            x=float(idx * 2),
            y=4.0 if idx % 2 == 0 else 3.5,
            customer_type=CustomerType.PICKUP,
            weight=0.8 + 0.4 * (idx % 4),
            time_window=(20.0 + 8.0 * idx, 300.0 + 12.0 * idx),
            service_time=4.0 + (idx % 4),
            home_probabilities=(0.75, 0.73, 0.71, 0.69, 0.67, 0.65, 0.63, 0.61),
        )

    return customers


@pytest.fixture
def large_instance(large_customers: dict[int, Customer]) -> ProblemInstance:
    """Two-pair instance with 12 customers for pipeline stress checks."""
    coords: dict[int, tuple[float, float]] = {0: (0.0, 0.0)}
    for customer_id, customer in large_customers.items():
        coords[customer_id] = (customer.x, customer.y)

    truck_distance_km = _build_distance_matrix(coords, scale=1.2)
    drone_distance_km = _build_distance_matrix(coords, scale=1.0)

    return ProblemInstance(
        depot_id=0,
        vehicle_pair_count=2,
        customers=large_customers,
        delivery_customers=frozenset({1, 2, 3, 4, 5, 6}),
        pickup_customers=frozenset({7, 8, 9, 10, 11, 12}),
        truck_distance_km=truck_distance_km,
        drone_distance_km=drone_distance_km,
    )


def _build_random_customers(
    count: int,
    delivery_count: int,
    rng: random.Random,
) -> dict[int, Customer]:
    """Build reproducible random customers for performance fixtures."""
    customers: dict[int, Customer] = {}
    for customer_id in range(1, count + 1):
        is_delivery = customer_id <= delivery_count
        customers[customer_id] = Customer(
            customer_id=customer_id,
            x=rng.uniform(0.0, 20.0),
            y=rng.uniform(0.0, 20.0),
            customer_type=CustomerType.DELIVERY if is_delivery else CustomerType.PICKUP,
            weight=rng.uniform(0.5, 3.0),
            time_window=(0.0, 480.0),
            service_time=5.0,
            home_probabilities=(0.80, 0.78, 0.76, 0.74, 0.72, 0.70, 0.68, 0.66),
        )
    return customers


@pytest.fixture
def medium_instance() -> ProblemInstance:
    """Performance medium-scale instance (20 customers, 2 pairs)."""
    rng = random.Random(42)
    customers = _build_random_customers(count=20, delivery_count=12, rng=rng)

    coords: dict[int, tuple[float, float]] = {0: (0.0, 0.0)}
    for customer_id, customer in customers.items():
        coords[customer_id] = (customer.x, customer.y)

    distance_matrix = _build_distance_matrix(coords, scale=1.0)
    return ProblemInstance(
        depot_id=0,
        vehicle_pair_count=2,
        customers=customers,
        delivery_customers=frozenset(range(1, 13)),
        pickup_customers=frozenset(range(13, 21)),
        truck_distance_km=distance_matrix,
        drone_distance_km=distance_matrix,
    )


@pytest.fixture
def large_instance() -> ProblemInstance:
    """Performance large-scale instance (50 customers, 3 pairs)."""
    rng = random.Random(42)
    customers = _build_random_customers(count=50, delivery_count=30, rng=rng)

    coords: dict[int, tuple[float, float]] = {0: (0.0, 0.0)}
    for customer_id, customer in customers.items():
        coords[customer_id] = (customer.x, customer.y)

    distance_matrix = _build_distance_matrix(coords, scale=1.0)
    return ProblemInstance(
        depot_id=0,
        vehicle_pair_count=3,
        customers=customers,
        delivery_customers=frozenset(range(1, 31)),
        pickup_customers=frozenset(range(31, 51)),
        truck_distance_km=distance_matrix,
        drone_distance_km=distance_matrix,
    )
