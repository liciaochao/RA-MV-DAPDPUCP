"""SPD Planner - Stochastic Package Delivery with Drone-Truck Coordination."""

from spd.types import *
from spd.config import *
from spd.main import SPDPlanner

__all__ = ["SPDPlanner"]

try:
    from spd.visualization import (
        plot_clustering,
        plot_initial_solution,
        plot_optimized_solution,
        plot_solution_comparison,
        plot_convergence_curve,
        plot_timeline_gantt,
        plot_energy_consumption,
        plot_operator_statistics,
        plot_monte_carlo_distribution,
    )

    __all__.extend(
        [
            "plot_clustering",
            "plot_initial_solution",
            "plot_optimized_solution",
            "plot_solution_comparison",
            "plot_convergence_curve",
            "plot_timeline_gantt",
            "plot_energy_consumption",
            "plot_operator_statistics",
            "plot_monte_carlo_distribution",
        ]
    )
except ModuleNotFoundError:
    # Visualization dependencies (for example matplotlib) are optional.
    pass
