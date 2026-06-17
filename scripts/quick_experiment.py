#!/usr/bin/env python3
"""Quick experiment to collect data for the SA report tables."""
import sys, os, time
sys.path.insert(0, '/Users/xiniuyiliao/Desktop/application_code/WKU-CPS3410-Project')

import numpy as np
from simulated_annealing.tsp_instance import TSPInstance
from simulated_annealing.simulated_annealing import SimulatedAnnealing

def load_or_create(name, n_cities, seed):
    """Load instance or use known parameters."""
    path = f'simulated_annealing/data/distances_between_cities_{n_cities}.txt'
    inst = TSPInstance.load(path)
    return inst

def run_quality_stability(inst, cooling_rate, n_runs, max_iter_per_temp, seed):
    """Run SA multiple times and collect quality stats."""
    costs = []
    times = []
    for s in range(n_runs):
        sa = SimulatedAnnealing(
            distance_matrix=inst.distance_matrix,
            initial_temp=10000.0,
            cooling_rate=cooling_rate,
            min_temp=0.01,
            max_iterations_per_temp=max_iter_per_temp,
            neighborhood_type='2-opt',
            seed=seed + s
        )
        result = sa.run(verbose=False)
        costs.append(result.best_cost)
        times.append(result.elapsed_time)
    costs = np.array(costs)
    return {
        'mean': np.mean(costs),
        'std': np.std(costs, ddof=1),
        'cv': (np.std(costs, ddof=1) / np.mean(costs)) * 100 if np.mean(costs) > 0 else 0,
        'min': np.min(costs),
        'max': np.max(costs),
        'time_mean': np.mean(times),
        'time_std': np.std(times, ddof=1),
    }

def run_convergence_time(inst, cooling_rates, max_iter_per_temp, seed):
    """Measure convergence time for different cooling rates."""
    results = {}
    for cr in cooling_rates:
        sa = SimulatedAnnealing(
            distance_matrix=inst.distance_matrix,
            initial_temp=10000.0,
            cooling_rate=cr,
            min_temp=0.01,
            max_iterations_per_temp=max_iter_per_temp,
            neighborhood_type='2-opt',
            seed=seed
        )
        result = sa.run(verbose=False)
        results[cr] = {
            'time': result.elapsed_time,
            'iterations': result.iteration_count,
            'cost': result.best_cost
        }
    return results

print("=" * 70)
print("Running Quick SA Experiment for Report Data")
print("=" * 70)

# Load instances
inst10 = load_or_create('cities_10', 10, 42)
inst50 = load_or_create('cities_50', 50, 123)

optimal_10 = 3473.0   # From project description
optimal_50 = 5644.0   # From PPT slides

print(f"\n10-city instance: {inst10.num_cities} cities, optimal = {optimal_10}")
print(f"50-city instance: {inst50.num_cities} cities, optimal = {optimal_50}")

# =====================================================
# 1. Cooling rate convergence time comparison
# =====================================================
print("\n--- Cooling Rate vs Time ---")
cooling_rates = [0.90, 0.95, 0.99, 0.995, 0.999, 0.9995]
time_results_10 = run_convergence_time(inst10, cooling_rates, 100, 42)
time_results_50 = run_convergence_time(inst50, cooling_rates, 200, 42)

print(f"\n{'Cooling Rate':<15} {'10-City Time(s)':<18} {'50-City Time(s)':<18}")
print("-" * 51)
for cr in cooling_rates:
    t10 = time_results_10[cr]['time']
    t50 = time_results_50[cr]['time']
    print(f"{cr:<15.4f} {t10:<18.4f} {t50:<18.4f}")

# =====================================================
# 2. Quality Comparison (best cooling rate = 0.999)
# =====================================================
print("\n--- Quality & Stability ---")
best_cr = 0.999
stats_10 = run_quality_stability(inst10, best_cr, 50, 100, 100)
stats_50 = run_quality_stability(inst50, best_cr, 30, 200, 200)

print(f"\n10-city (50 runs, cr={best_cr}):")
print(f"  Mean: {stats_10['mean']:.2f}, Std: {stats_10['std']:.2f}, CV: {stats_10['cv']:.2f}%")
print(f"  Min: {stats_10['min']:.2f}, Max: {stats_10['max']:.2f}")
print(f"  Time mean: {stats_10['time_mean']:.4f}s")

print(f"\n50-city (30 runs, cr={best_cr}):")
print(f"  Mean: {stats_50['mean']:.2f}, Std: {stats_50['std']:.2f}, CV: {stats_50['cv']:.2f}%")
print(f"  Min: {stats_50['min']:.2f}, Max: {stats_50['max']:.2f}")
print(f"  Time mean: {stats_50['time_mean']:.4f}s")

# =====================================================
# 3. Single best run for convergence trace
# =====================================================
print("\n--- Best Single Run ---")
sa10 = SimulatedAnnealing(
    inst10.distance_matrix, initial_temp=10000.0, cooling_rate=0.999,
    min_temp=0.01, max_iterations_per_temp=100, neighborhood_type='2-opt', seed=42
)
result10 = sa10.run(verbose=False)
print(f"10-city best: {result10.best_cost:.2f}, gap: {(result10.best_cost - optimal_10)/optimal_10*100:.2f}%")

sa50 = SimulatedAnnealing(
    inst50.distance_matrix, initial_temp=10000.0, cooling_rate=0.999,
    min_temp=0.01, max_iterations_per_temp=200, neighborhood_type='2-opt', seed=200
)
result50 = sa50.run(verbose=False)
print(f"50-city best: {result50.best_cost:.2f}, gap: {(result50.best_cost - optimal_50)/optimal_50*100:.2f}%")

print("\n" + "=" * 70)
print("Experiment Complete!")
print("=" * 70)
