from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

# ============================================================
# 原始文件: spd/domain/enums.py
# ============================================================

"""Domain enums for SPD entities and events."""

class CustomerType(str, Enum):
    """Customer demand type."""

    DELIVERY = "delivery"
    PICKUP = "pickup"


class ServiceMode(str, Enum):
    """Service performer for a customer."""

    TRUCK = "truck"
    DRONE = "drone"


class OnlineEventType(str, Enum):
    """Online events considered by the model."""

    CUSTOMER_NOT_HOME = "T5_customer_not_home"

# ============================================================
# 原始文件: spd/core/types.py
# ============================================================

"""Common typing aliases used by core computation modules."""



HomeStatusMap = Mapping[int, bool]

# ============================================================
# 原始文件: spd/domain/entities.py
# ============================================================

"""Core domain entities shared by offline and online modules."""





@dataclass(slots=True, frozen=True)
class Customer:
    """Customer model from spec appendix."""

    customer_id: int
    x: float
    y: float
    customer_type: CustomerType
    weight: float
    time_window: tuple[float, float]
    service_time: float
    home_probabilities: tuple[float, ...]


@dataclass(slots=True)
class Sortie:
    """One drone sortie launched and recovered on truck route nodes.

    Structural constraints expected by feasibility checks:
    - launch_node and recovery_node are customer nodes, not depot node 0.
    - launch_node and recovery_node must belong to pair_solution.truck_customers.
    - customers are drone-served customers and should not appear on truck_route.
    - customers should not contain launch_node or recovery_node.
    - len(customers) must be <= m_max from constraint parameters.
    """

    launch_node: int
    recovery_node: int
    customers: list[int]


@dataclass(slots=True)
class VehiclePairSolution:
    """Solution fragment for one truck-drone pair."""

    pair_id: int
    truck_route: list[int]
    sorties: list[Sortie] = field(default_factory=list)
    truck_customers: set[int] = field(default_factory=set)
    drone_customers: set[int] = field(default_factory=set)

    @property
    def all_customers(self) -> set[int]:
        return set(self.truck_customers) | set(self.drone_customers)


@dataclass(slots=True)
class Solution:
    """Global solution object."""

    vehicle_pairs: list[VehiclePairSolution] = field(default_factory=list)
    unserved_customers: set[int] = field(default_factory=set)

    @property
    def all_customers(self) -> set[int]:
        result: set[int] = set()
        for pair in self.vehicle_pairs:
            result |= pair.all_customers
        return result

    @property
    def used_vehicle_pairs(self) -> list[VehiclePairSolution]:
        return [p for p in self.vehicle_pairs if p.truck_customers or p.drone_customers]


@dataclass(slots=True)
class TimelineState:
    """Arrival and departure times for truck and drone."""

    truck_arrival: dict[int, float] = field(default_factory=dict)
    truck_departure: dict[int, float] = field(default_factory=dict)
    drone_arrival: dict[int, float] = field(default_factory=dict)
    drone_departure: dict[int, float] = field(default_factory=dict)


@dataclass(slots=True)
class LoadState:
    """Tracked loads for truck and drone."""

    truck_load_at_node: dict[int, float] = field(default_factory=dict)
    drone_load_at_node: dict[int, float] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ProblemInstance:
    """Static input instance for one optimization run."""

    depot_id: int
    vehicle_pair_count: int
    customers: Mapping[int, Customer]
    delivery_customers: frozenset[int]
    pickup_customers: frozenset[int]
    truck_distance_km: Mapping[tuple[int, int], float]
    drone_distance_km: Mapping[tuple[int, int], float]

    def truck_distance(self, from_node: int, to_node: int) -> float:
        """Return d_T(i, j) in km."""
        return self.truck_distance_km[(from_node, to_node)]

    def drone_distance(self, from_node: int, to_node: int) -> float:
        """Return d_D(i, j) in km."""
        return self.drone_distance_km[(from_node, to_node)]

    def truck_travel_time(self, from_node: int, to_node: int, truck_speed: float) -> float:
        """Return t_T(i, j) = d_T(i, j) / v_T in minutes."""
        if truck_speed <= 0:
            raise ValueError("truck_speed must be positive")
        return self.truck_distance(from_node, to_node) / truck_speed

    def drone_travel_time(self, from_node: int, to_node: int, drone_speed: float) -> float:
        """Return t_D(i, j) = d_D(i, j) / v_D in minutes."""
        if drone_speed <= 0:
            raise ValueError("drone_speed must be positive")
        return self.drone_distance(from_node, to_node) / drone_speed


@dataclass(slots=True)
class ExecutionResult:
    """Online execution output with realized states and cost."""

    solution: Solution
    actual_timeline: dict[int, tuple[float, float]] = field(default_factory=dict)
    actual_is_home: dict[int, bool] = field(default_factory=dict)
    failed_customers: set[int] = field(default_factory=set)
    truck_distances_km: list[float] = field(default_factory=list)
    drone_energies_wh: list[float] = field(default_factory=list)
    actual_cost: float = 0.0
    replanning_snapshots: list[dict[str, object]] = field(default_factory=list)

# ============================================================
# 原始文件: spd/online/events.py
# ============================================================

"""Online event data models."""





@dataclass(slots=True, frozen=True)
class CustomerNotHomeEvent:
    """T5 event: customer is not at home when service attempt happens."""

    pair_id: int
    customer_id: int
    service_mode: ServiceMode
    event_time: float
    sortie_index: int | None = None
    event_type: OnlineEventType = OnlineEventType.CUSTOMER_NOT_HOME

