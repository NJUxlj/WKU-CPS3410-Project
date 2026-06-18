"""
Genetic Algorithm for the Traveling Salesman Problem (TSP)

This package implements the Genetic Algorithm metaheuristic with:
- Roulette / Tournament / SUS selection
- OX and PMX crossover
- Multiple mutation operators (swap, inversion, insert, 2-opt, 3-opt)
- Elitism (matching the GA report's design)
- Comprehensive experiment framework & visualization
"""

from .genetic_algorithm import (
    GeneticAlgorithm,
    GAParams,
    GAResult,
    Individual,
    MUTATION_FUNCTIONS,
)
from .experiment import (
    RunStats,
    run_experiment,
    full_experiment,
    compare_mutation_methods,
    compare_selection_methods,
)
from .visualize import (
    plot_convergence,
    plot_diversity,
    plot_tour,
    plot_quality_comparison,
    plot_stability_analysis,
    generate_all_plots,
)
