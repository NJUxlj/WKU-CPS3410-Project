"""
Simulated Annealing for the Traveling Salesman Problem (TSP)

This package implements the Simulated Annealing metaheuristic algorithm
for solving the Traveling Salesman Problem, including:
- Multiple cooling schedules (geometric, linear, logarithmic)
- Multiple neighborhood structures (2-opt, swap, insert)
- Comprehensive experiment framework
- Visualization tools
"""

from .tsp_instance import TSPInstance, find_optimal_brute_force
from .simulated_annealing import SimulatedAnnealing, SAResult, SAParams
from .experiment import run_experiment, full_experiment, compare_cooling_rates
