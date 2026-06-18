"""
Local Search for the Traveling Salesman Problem (TSP)

This package implements the classic Local Search metaheuristic with:
- Multiple neighborhood structures (1-opt/swap, 2-opt, 3-opt, or-opt)
- First-improvement and best-improvement hill climbing strategies
- Multi-restart mechanism to escape local optima
- Comprehensive experiment framework & visualization
"""

from .local_search import (
    LocalSearch,
    LSResult,
    LSParams,
    NEIGHBORHOOD_FUNCTIONS,
)
from .experiment import (
    RunStats,
    run_experiment,
    full_experiment,
    compare_neighborhoods,
    compare_strategies,
)
from .visualize import (
    plot_convergence,
    plot_tour,
    plot_quality_comparison,
    plot_stability_analysis,
    generate_all_plots,
)
