"""
Tabu Search for the Traveling Salesman Problem (TSP)

This package implements the Tabu Search metaheuristic with:
- 1-opt (swap) neighborhood (matching the TS report's design)
- Aspiration criterion (tabu override on global best)
- Configurable tabu tenure and iteration count
- Comprehensive experiment framework & visualization
"""

from .tabu_search import TabuSearch, TSParams, TSResult
from .experiment import (
    RunStats,
    run_experiment,
    full_experiment,
    compare_tabu_tenures,
)
from .visualize import (
    plot_convergence,
    plot_tabu_size,
    plot_tour,
    plot_quality_comparison,
    plot_stability_analysis,
    generate_all_plots,
)
