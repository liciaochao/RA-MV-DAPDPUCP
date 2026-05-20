from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import random

from spd.types import ProblemInstance, Solution
from spd.online import LightweightALNSOptimizer
from spd.online import OnlineReplanner
from spd.simulation import ExecutionEngine
from spd.simulation import MonteCarloSimulator


def _clone_instance_customer(
    base: ProblemInstance,
    customer_id: int,
    *,
    home_probabilities: tuple[float, ...] | None = None,
) -> ProblemInstance:
    customers = dict(base.customers)
    customer = customers[customer_id]
    customers[customer_id] = replace(
        customer,
        home_probabilities=home_probabilities if home_probabilities is not None else customer.home_probabilities,
    )
    return ProblemInstance(
        depot_id=base.depot_id,
        vehicle_pair_count=base.vehicle_pair_count,
        customers=customers,
        delivery_customers=base.delivery_customers,
        pickup_customers=base.pickup_customers,
        truck_distance_km=base.truck_distance_km,
        drone_distance_km=base.drone_distance_km,
    )


def test_generate_home_status_deterministic_and_probability_extremes(
    sample_instance,
    sample_params,
    sample_pair_solution,
) -> None:
    replanner = OnlineReplanner(LightweightALNSOptimizer())
    engine = ExecutionEngine(replanner)
    planned_solution = Solution(vehicle_pairs=[deepcopy(sample_pair_solution)], unserved_customers=set())

    # Force extreme probabilities for two customers.
    instance = _clone_instance_customer(sample_instance, 1, home_probabilities=(1.0,) * 8)
    instance = _clone_instance_customer(instance, 2, home_probabilities=(0.0,) * 8)

    planned_arrival_slots = {customer_id: 1 for customer_id in instance.customers}

    first = engine.generate_home_status(
        instance,
        planned_solution,
        planned_arrival_slots,
        sample_params,
        random.Random(1234),
    )
    second = engine.generate_home_status(
        instance,
        planned_solution,
        planned_arrival_slots,
        sample_params,
        random.Random(1234),
    )

    assert first == second
    assert first[1] is True
    assert first[2] is False


def test_execution_engine_run_full_flow(
    sample_instance,
    sample_params,
    sample_pair_solution,
    all_home_status,
) -> None:
    replanner = OnlineReplanner(LightweightALNSOptimizer())
    engine = ExecutionEngine(replanner)
    planned_solution = Solution(vehicle_pairs=[deepcopy(sample_pair_solution)], unserved_customers=set())

    preset_home_status = dict(all_home_status)
    preset_home_status[1] = False  # truck customer
    preset_home_status[2] = False  # drone customer

    result = engine.run(
        sample_instance,
        planned_solution,
        sample_params,
        rng=random.Random(77),
        preset_home_status=preset_home_status,
    )

    assert result.actual_is_home[1] is False
    assert result.actual_is_home[2] is False
    assert {1, 2}.issubset(result.failed_customers)
    assert len(result.truck_distances_km) > 0
    assert all(distance >= 0.0 for distance in result.truck_distances_km)
    assert all(energy >= 0.0 for energy in result.drone_energies_wh)
    assert result.actual_cost > 0.0


def test_monte_carlo_simulator_statistics_reasonable(
    sample_instance,
    sample_params,
    sample_pair_solution,
) -> None:
    replanner = OnlineReplanner(LightweightALNSOptimizer())
    engine = ExecutionEngine(replanner)
    simulator = MonteCarloSimulator(engine)
    planned_solution = Solution(vehicle_pairs=[deepcopy(sample_pair_solution)], unserved_customers=set())

    result = simulator.run(
        sample_instance,
        planned_solution,
        sample_params,
        trial_count=10,
        rng=random.Random(2026),
    )

    assert result.trial_count == 10
    assert len(result.samples) == 10
    assert min(result.samples) <= result.average_actual_cost <= max(result.samples)
    assert result.worst_actual_cost == max(result.samples)
    assert result.cost_std >= 0.0
    assert 0.0 <= result.average_failed_customers <= float(len(sample_instance.customers))

    # Same seed should reproduce the same trial sequence.
    result_2 = simulator.run(
        sample_instance,
        planned_solution,
        sample_params,
        trial_count=10,
        rng=random.Random(2026),
    )
    assert result.samples == result_2.samples
