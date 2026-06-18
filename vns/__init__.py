"""
Variable Neighborhood Search (VNS) and Reduced VNS (RVNS) for the TSP.

This package implements:
- Basic VNS with full local search descent
- Reduced VNS (RVNS) without local search (higher diversification)
- Configurable neighborhood family N_k
- Comprehensive experiment framework & visualization
"""

from .vns import VNS, RVNS, VNSParams, VNSResult
from .neighborhoods import NEIGHBORHOOD_FAMILY
from .experiment import (
    RunStats,
    run_vns_experiment,
    run_rvns_experiment,
    full_experiment,
    compare_vns_rvns,
)
from .visualize import (
    plot_convergence,
    plot_neighborhood_usage,
    plot_tour,
    plot_quality_comparison,
    plot_stability_analysis,
    generate_all_plots,
)
