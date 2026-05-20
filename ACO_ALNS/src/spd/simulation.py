from __future__ import annotations

import math
import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Protocol

from spd.config import ProblemParameters, map_time_to_slot
from spd.core import compute_actual_cost, compute_sortie_energy, compute_truck_timeline, failure_penalty, get_arrival_time
from spd.online import OnlineReplanner
from spd.types import CustomerNotHomeEvent, ExecutionResult, ProblemInstance, ServiceMode, Solution, VehiclePairSolution

# ============================================================
# 原始文件: spd/simulation/execution.py
# ============================================================

"""Execution engine contract for online simulation."""





class HomeStatusSampler(Protocol):
    """Generate realized home status h_i from p_i(t*) under a random seed policy."""

    def sample(
        self,
        instance: ProblemInstance,
        planned_solution: Solution,
        planned_arrival_slots: dict[int, int],
        params: ProblemParameters,
        rng: random.Random,
    ) -> dict[int, bool]:
        ...


class ExecutionEngine:
    """Run one realization of online execution for a fixed offline plan."""

    def __init__(self, replanner: OnlineReplanner, home_status_sampler: HomeStatusSampler | None = None):
        self._replanner = replanner
        self._home_status_sampler = home_status_sampler

    def generate_home_status(
        self,
        instance: ProblemInstance,
        planned_solution: Solution,
        planned_arrival_slots: dict[int, int],
        params: ProblemParameters,
        rng: random.Random,
    ) -> dict[int, bool]:
        """Generate h_i using customer home probabilities and random sampling."""
        if self._home_status_sampler is not None:
            return self._home_status_sampler.sample(
                instance,
                planned_solution,
                planned_arrival_slots,
                params,
                rng,
            )

        home_status: dict[int, bool] = {}
        for customer_id in sorted(instance.customers):
            slot = planned_arrival_slots.get(customer_id, 1)
            slot = max(1, min(params.time.num_time_slots, slot))
            prob = instance.customers[customer_id].home_probabilities[slot - 1]
            home_status[customer_id] = rng.random() < prob

        return home_status

    def run(
        self,
        instance: ProblemInstance,
        planned_solution: Solution,
        params: ProblemParameters,
        rng: random.Random | None = None,
        preset_home_status: dict[int, bool] | None = None,
        planned_arrival_slots: dict[int, int] | None = None,
    ) -> ExecutionResult:
        """Execute route plan, reveal T5 events, and return realized result.

        Args:
            preset_home_status: Optional externally provided h_i map.
            planned_arrival_slots: Optional planned t* slots used for p_i(t*)
                lookup during h_i generation.
        """
        random_state = rng or random.Random(0)

        if preset_home_status is None:
            slots = dict(planned_arrival_slots) if planned_arrival_slots is not None else self._compute_planned_arrival_slots(
                instance,
                planned_solution,
                params,
            )
            home_status = self.generate_home_status(
                instance,
                planned_solution,
                slots,
                params,
                random_state,
            )
        else:
            home_status = {
                customer_id: bool(preset_home_status.get(customer_id, True))
                for customer_id in instance.customers
            }

        current_solution = deepcopy(planned_solution)
        replanning_snapshots: list[dict[str, object]] = [
            {
                "label": "离线优化解 (初始计划)",
                "solution": deepcopy(current_solution),
                "failed_customer": None,
                "objective": None,
            }
        ]
        failed_customers: set[int] = {customer_id for customer_id, is_home in home_status.items() if not is_home}
        truck_distances_km: list[float] = []
        drone_energies_wh: list[float] = []
        _snapshot_recorded_customers: set[int] = set()

        def _record_replanning_snapshot(failed_customer_id: int) -> None:
            if int(failed_customer_id) in _snapshot_recorded_customers:
                print(f"[snapshot] SKIP duplicate: customer {failed_customer_id}")
                return
            _snapshot_recorded_customers.add(int(failed_customer_id))
            replan_count = len(replanning_snapshots)
            replanning_snapshots.append(
                {
                    "label": f"重规划 #{replan_count}: 客户 {int(failed_customer_id)} 失败",
                    "solution": deepcopy(current_solution),
                    "failed_customer": int(failed_customer_id),
                    "objective": None,
                }
            )

        pair_ids = [pair.pair_id for pair in planned_solution.vehicle_pairs]
        for pair_id in pair_ids:
            route_pos = 0
            while True:
                pair_solution = self._find_pair_by_id(current_solution, pair_id)
                route = pair_solution.truck_route
                if route_pos >= len(route) - 1:
                    break

                from_node = route[route_pos]
                to_node = route[route_pos + 1]
                truck_distances_km.append(instance.truck_distance_km[(from_node, to_node)])
                route_pos += 1

                # Truck-side event handling at current visited node.
                pair_solution = self._find_pair_by_id(current_solution, pair_id)
                if to_node in pair_solution.truck_customers and not home_status.get(to_node, True):
                    event = CustomerNotHomeEvent(
                        pair_id=pair_id,
                        customer_id=to_node,
                        service_mode=ServiceMode.TRUCK,
                        event_time=float(route_pos),
                        sortie_index=None,
                    )
                    current_solution = self._replanner.handle_event(
                        instance,
                        current_solution,
                        event,
                        home_status,
                        params,
                        random_state,
                    )
                    _record_replanning_snapshot(to_node)

                # Drone-side events for sorties launched at this node.
                launch_node = to_node
                launch_rank = 0
                while True:
                    pair_solution = self._find_pair_by_id(current_solution, pair_id)
                    launched_indices = [
                        idx
                        for idx, sortie in enumerate(pair_solution.sorties)
                        if sortie.launch_node == launch_node
                    ]
                    if launch_rank >= len(launched_indices):
                        break

                    sortie_index = launched_indices[launch_rank]
                    handled_customers: set[int] = set()

                    while True:
                        pair_solution = self._find_pair_by_id(current_solution, pair_id)
                        launched_indices = [
                            idx
                            for idx, sortie in enumerate(pair_solution.sorties)
                            if sortie.launch_node == launch_node
                        ]
                        if launch_rank >= len(launched_indices):
                            break

                        sortie_index = launched_indices[launch_rank]
                        sortie = pair_solution.sorties[sortie_index]
                        next_customer = next(
                            (
                                customer_id
                                for customer_id in sortie.customers
                                if (not home_status.get(customer_id, True)) and customer_id not in handled_customers
                            ),
                            None,
                        )
                        if next_customer is None:
                            break

                        event = CustomerNotHomeEvent(
                            pair_id=pair_id,
                            customer_id=next_customer,
                            service_mode=ServiceMode.DRONE,
                            event_time=float(route_pos),
                            sortie_index=sortie_index,
                        )
                        current_solution = self._replanner.handle_event(
                            instance,
                            current_solution,
                            event,
                            home_status,
                            params,
                            random_state,
                        )
                        _record_replanning_snapshot(next_customer)
                        handled_customers.add(next_customer)

                    pair_solution = self._find_pair_by_id(current_solution, pair_id)
                    launched_indices = [
                        idx
                        for idx, sortie in enumerate(pair_solution.sorties)
                        if sortie.launch_node == launch_node
                    ]
                    if launch_rank < len(launched_indices):
                        relocated_index = launched_indices[launch_rank]
                        updated_sortie = pair_solution.sorties[relocated_index]
                        try:
                            timeline = compute_truck_timeline(instance, pair_solution, home_status, params)
                            drone_energies_wh.append(
                                compute_sortie_energy(instance, updated_sortie, timeline, home_status, params)
                            )
                        except (AssertionError, KeyError, ValueError):
                            pass

                    launch_rank += 1

        actual_timeline: dict[int, tuple[float, float]] = {}
        for pair_solution in current_solution.vehicle_pairs:
            try:
                timeline = compute_truck_timeline(instance, pair_solution, home_status, params)
            except (AssertionError, KeyError, ValueError):
                continue

            for node, arrival in timeline.truck_arrival.items():
                departure = timeline.truck_departure.get(node, arrival)
                actual_timeline[node] = (arrival, departure)

            for node, arrival in timeline.drone_arrival.items():
                departure = timeline.drone_departure.get(node, arrival)
                actual_timeline.setdefault(node, (arrival, departure))

        fail_penalty = sum(failure_penalty(instance, customer_id, params) for customer_id in failed_customers)
        fixed_cost = params.cost.fixed_pair_cost * len(current_solution.used_vehicle_pairs)
        truck_cost = params.cost.truck_cost_per_km * sum(truck_distances_km)
        drone_cost = params.cost.drone_energy_cost_per_wh * sum(drone_energies_wh)
        actual_total_cost = fixed_cost + truck_cost + drone_cost + fail_penalty

        result = ExecutionResult(
            solution=current_solution,
            actual_timeline=actual_timeline,
            actual_is_home=dict(home_status),
            failed_customers=set(failed_customers),
            truck_distances_km=truck_distances_km,
            drone_energies_wh=drone_energies_wh,
            actual_cost=actual_total_cost,
            replanning_snapshots=replanning_snapshots,
        )
        result.actual_cost = compute_actual_cost(result, params)
        return result

    @staticmethod
    def _find_pair_by_id(solution: Solution, pair_id: int) -> VehiclePairSolution:
        """Lookup pair solution by pair_id."""
        for pair_solution in solution.vehicle_pairs:
            if pair_solution.pair_id == pair_id:
                return pair_solution
        raise ValueError(f"pair_id {pair_id} not found")

    @staticmethod
    def _compute_planned_arrival_slots(
        instance: ProblemInstance,
        planned_solution: Solution,
        params: ProblemParameters,
    ) -> dict[int, int]:
        """Compute planned t* slots from offline timeline under all-home assumption."""
        all_home = {customer_id: True for customer_id in instance.customers}
        slots: dict[int, int] = {}

        for pair_solution in planned_solution.vehicle_pairs:
            try:
                timeline = compute_truck_timeline(instance, pair_solution, all_home, params)
            except (AssertionError, KeyError, ValueError):
                continue

            for customer_id in pair_solution.all_customers:
                try:
                    arrival_time = get_arrival_time(customer_id, pair_solution, timeline)
                    slots[customer_id] = map_time_to_slot(arrival_time, params.time)
                except (AssertionError, KeyError, ValueError):
                    continue

        default_slot = map_time_to_slot(params.time.depot_earliest, params.time)
        for customer_id in instance.customers:
            slots.setdefault(customer_id, default_slot)

        return slots

# ============================================================
# 原始文件: spd/simulation/monte_carlo.py
# ============================================================

"""Monte Carlo simulation contracts used by experiment suites."""





@dataclass(slots=True)
class MonteCarloResult:
    """Aggregated metrics over multiple simulation trials."""

    trial_count: int
    average_actual_cost: float
    worst_actual_cost: float
    average_failed_customers: float
    cost_std: float
    samples: list[float] = field(default_factory=list)


class MonteCarloSimulator:
    """Run repeated online simulations with fixed random seed policy."""

    def __init__(self, execution_engine: ExecutionEngine):
        self._execution_engine = execution_engine
        self.mc_results: list[float] = []
        self.first_trial_result: ExecutionResult | None = None

    def run(
        self,
        instance: ProblemInstance,
        planned_solution: Solution,
        params: ProblemParameters,
        trial_count: int,
        rng: random.Random | None = None,
    ) -> MonteCarloResult:
        """Run M trials and aggregate statistics for experiment analysis."""
        if trial_count <= 0:
            raise ValueError("trial_count must be positive")

        random_state = rng or random.Random(0)
        base_seed = random_state.randint(0, 2**31 - 1)
        self.mc_results = []
        self.first_trial_result = None

        cost_samples: list[float] = []
        failed_counts: list[int] = []

        for trial_index in range(trial_count):
            trial_rng = random.Random(base_seed + trial_index)
            result = self._execution_engine.run(
                instance,
                planned_solution,
                params,
                rng=trial_rng,
            )
            if trial_index == 0:
                self.first_trial_result = result
            cost_samples.append(result.actual_cost)
            self.mc_results.append(result.actual_cost)
            failed_counts.append(len(result.failed_customers))

        average_actual_cost = sum(cost_samples) / trial_count
        worst_actual_cost = max(cost_samples)
        average_failed_customers = sum(failed_counts) / trial_count

        variance = sum((sample - average_actual_cost) ** 2 for sample in cost_samples) / trial_count
        cost_std = math.sqrt(variance)

        return MonteCarloResult(
            trial_count=trial_count,
            average_actual_cost=average_actual_cost,
            worst_actual_cost=worst_actual_cost,
            average_failed_customers=average_failed_customers,
            cost_std=cost_std,
            samples=cost_samples,
        )

