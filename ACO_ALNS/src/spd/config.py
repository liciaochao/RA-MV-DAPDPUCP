from __future__ import annotations

from dataclasses import dataclass, field

# ============================================================
# 原始文件: spd/core/timeslots.py
# ============================================================

"""Time-slot mapping utilities from Spec Sec 2.7."""




def map_time_to_slot(arrival_time: float, time: TimeParameters) -> int:
    """Map an arrival timestamp to a bounded slot index in [1, num_time_slots]."""
    if time.num_time_slots <= 0:
        raise ValueError("num_time_slots must be positive")
    slot = int((arrival_time - time.depot_earliest) // time.slot_length) + 1
    return max(1, min(time.num_time_slots, slot))

# ============================================================
# 原始文件: spd/config/parameters.py
# ============================================================

"""Typed configuration objects derived from the SPD specification."""




@dataclass(slots=True, frozen=True)
class TimeParameters:
    """Time and time-slot controls (Spec Sec 2.7)."""

    depot_earliest: float
    depot_latest: float
    num_time_slots: int = 8

    @property
    def slot_length(self) -> float:
        return (self.depot_latest - self.depot_earliest) / self.num_time_slots


@dataclass(slots=True, frozen=True)
class VehicleParameters:
    """Vehicle capacities and speeds (Spec Sec 2.3)."""

    truck_capacity: float  # Q_T
    drone_capacity: float  # Q_D
    drone_empty_weight: float  # W_D
    truck_speed: float  # v_T
    drone_speed: float  # v_D


@dataclass(slots=True, frozen=True)
class EnergyParameters:
    """Drone physical energy parameters (Spec Sec 2.4)."""

    drone_battery_capacity: float  # E_D
    eta_wh_per_kg_min: float  # eta


@dataclass(slots=True, frozen=True)
class CostParameters:
    """Cost terms in objective (Spec Sec 2.5 and Sec 6)."""

    truck_cost_per_km: float  # c_T
    fixed_pair_cost: float  # alpha
    drone_energy_cost_per_wh: float  # c_E


@dataclass(slots=True, frozen=True)
class BetaScheduleParameters:
    """Dynamic time-window penalty schedule (Spec Sec 10.6)."""

    beta_init: float
    beta_max: float
    gamma_beta: float


@dataclass(slots=True, frozen=True)
class ConstraintParameters:
    """Hard constraints from the model (Spec Sec 2.9)."""

    max_truck_wait_time: float  # W_max
    max_customers_per_sortie: int  # m_max


@dataclass(slots=True, frozen=True)
class ACOParameters:
    """ACO outer-loop parameters (Spec Sec 10.7)."""

    max_iterations: int
    ant_count: int
    alpha_aco: float
    beta_aco: float
    evaporation_rate: float  # rho
    pheromone_q: float  # Q_aco
    pheromone_init: float  # tau_0
    no_improve_max: int
    eta_weight_distance: float = 1.0
    eta_weight_time_window: float = 0.8
    eta_weight_capacity: float = 0.6
    candidate_list_size: int | None = None
    tau_min: float | None = None
    tau_max: float | None = None
    stagnation_restart_threshold: int | None = None
    stagnation_restart_ratio: float = 0.35
    global_best_weight: float = 0.6


@dataclass(slots=True, frozen=True)
class ALNSParameters:
    """Offline ALNS parameters (Spec Sec 10.8 and Sec 10.9)."""

    iterations: int  # N_alns
    remove_count_min: int
    remove_count_max: int
    sa_initial_temperature: float
    sa_cooling_rate: float
    operator_weight_init: float
    reward_global_best: float  # sigma_1
    reward_improve: float  # sigma_2
    reward_accept_worse: float  # sigma_3
    reaction_factor: float  # lambda_w
    cross_group_frequency: int  # Freq_cross
    cross_group_remove_count: int  # q_cross


@dataclass(slots=True, frozen=True)
class ETPRCParameters:
    """Constructive heuristic parameters for E-TPRC (Spec Sec 10.10)."""

    k_cluster: int
    max_iter_kmeans: int
    omega_distance: float
    omega_time_window: float
    omega_balance: float


@dataclass(slots=True, frozen=True)
class OnlineALNSParameters:
    """Online lightweight ALNS parameters (Spec Sec 9.3 and Sec 10.11)."""

    iterations: int  # N_alns_online
    remove_count: int  # q_remove_online




@dataclass(slots=True, frozen=True)
class MonteCarloParameters:
    """Monte Carlo simulation controls."""

    trial_count: int = 30


@dataclass(slots=True, frozen=True)
class ProblemParameters:
    """Top-level configuration object used across modules."""

    time: TimeParameters
    vehicle: VehicleParameters
    energy: EnergyParameters
    cost: CostParameters
    beta_schedule: BetaScheduleParameters
    constraints: ConstraintParameters
    aco: ACOParameters
    alns: ALNSParameters
    etprc: ETPRCParameters
    online_alns: OnlineALNSParameters
    monte_carlo: MonteCarloParameters = field(default_factory=MonteCarloParameters)

