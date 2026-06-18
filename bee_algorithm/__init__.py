"""
Artificial Bee Colony (ABC) Algorithm for the Traveling Salesman Problem (TSP)

This package implements the ABC swarm metaheuristic with:
- Employed / Onlooker / Scout bee phases
- Multiple neighborhood operators (swap, 2-opt, or-opt)
- Trial-based abandonment mechanism (limit parameter)
- Comprehensive experiment framework & visualization
"""

from .abc import (
    ArtificialBeeColony,
    ABCParams,
    ABCResult,
    FoodSource,
)
from .experiment import (
    RunStats,
    run_experiment,
    full_experiment,
    compare_neighborhoods,
)
from .visualize import (
    plot_convergence,
    plot_abandoned_sources,
    plot_tour,
    plot_quality_comparison,
    plot_stability_analysis,
    generate_all_plots,
)
