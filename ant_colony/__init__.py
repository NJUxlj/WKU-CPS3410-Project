"""
Ant Colony Optimization for the Traveling Salesman Problem (TSP)

This package implements Ant Colony Optimization (both Ant System AS and
Ant Colony System ACS variants) with:
- Configurable α (pheromone), β (heuristic), ρ (evaporation)
- ACS pseudo-random proportional rule with local pheromone update
- Best-so-far or all-ants pheromone deposit
- Comprehensive experiment framework & visualization
"""

from .aco import AntColony, ACOParams, ACOResult
from .experiment import (
    RunStats,
    run_experiment,
    full_experiment,
    compare_alpha_beta,
)
from .visualize import (
    plot_convergence,
    plot_iteration_best,
    plot_pheromone_heatmap,
    plot_tour,
    plot_quality_comparison,
    plot_stability_analysis,
    generate_all_plots,
)
