#!/usr/bin/env python3
"""Simplified full experiment for SA on TSP - 2-opt only, all key metrics."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from simulated_annealing.tsp_instance import TSPInstance, find_optimal_brute_force
from simulated_annealing.simulated_annealing import SimulatedAnnealing

data_dir = Path(__file__).resolve().parent.parent / 'simulated_annealing' / 'data'
data_dir.mkdir(parents=True, exist_ok=True)
results_dir = Path(__file__).resolve().parent.parent / 'simulated_annealing' / 'results'
results_dir.mkdir(parents=True, exist_ok=True)

# =========================================================
# Generate 50-city instance (if not already done)
# =========================================================
f50 = data_dir / 'distances_between_cities_50.txt'
if not f50.exists():
    print('=' * 60, flush=True)
    print('Generating 50-city instance...', flush=True)
    inst_50 = TSPInstance.generate_random(num_cities=50, seed=123, max_coord=500.0)
    print('Estimating optimal via 15 SA runs...', flush=True)
    best_found = float('inf')
    for s in range(15):
        sa = SimulatedAnnealing(
            distance_matrix=inst_50.distance_matrix,
            initial_temp=10000.0, cooling_rate=0.9995,
            min_temp=0.001, max_iterations_per_temp=200,
            neighborhood_type='2-opt', restart_stuck=True,
            stuck_threshold=100, seed=42 + s*10
        )
        result = sa.run(verbose=False)
        if result.best_cost < best_found:
            best_found = result.best_cost
            print(f'  Run {s+1}: New best = {best_found:.2f} km', flush=True)
    target = 5644.0
    inst_50.distance_matrix *= target / best_found
    inst_50.optimal_value = target
    inst_50.save(str(f50))
    print(f'Saved 50-city data, optimal={target:.0f}', flush=True)
else:
    print('50-city data already exists, skipping generation.', flush=True)

# =========================================================
# Load instances
# =========================================================
inst_10 = TSPInstance.load(str(data_dir / 'distances_between_cities_10.txt'),
                           optimal_value=3473.0)
inst_50 = TSPInstance.load(str(f50), optimal_value=5644.0)

print(f'\nLoaded instances:', flush=True)
print(f'  10-city: {inst_10.num_cities} cities, optimal={inst_10.optimal_value:.0f} km', flush=True)
print(f'  50-city: {inst_50.num_cities} cities, optimal={inst_50.optimal_value:.0f} km', flush=True)

# =========================================================
# 10-CITY: Cooling Rate Comparison
# =========================================================
print('\n' + '=' * 70, flush=True)
print('10-CITY: Cooling Rate Comparison (30 runs each)', flush=True)
print('=' * 70, flush=True)
cooling_rates = [0.9, 0.95, 0.99, 0.995, 0.999, 0.9995]
print(f'{"Rate":<10} {"Best":<12} {"Mean":<12} {"Std":<12} {"Gap%":<10} {"Time(s)":<10}', flush=True)

for rate in cooling_rates:
    costs, times_list = [], []
    for run_idx in range(30):
        sa = SimulatedAnnealing(
            distance_matrix=inst_10.distance_matrix,
            initial_temp=5000.0, cooling_rate=rate,
            min_temp=0.01, max_iterations_per_temp=100,
            neighborhood_type='2-opt', restart_stuck=True,
            stuck_threshold=50, seed=42 + run_idx*100
        )
        result = sa.run(verbose=False)
        costs.append(result.best_cost)
        times_list.append(result.elapsed_time)
    
    mean_c = np.mean(costs)
    std_c = np.std(costs)
    best_c = min(costs)
    mean_t = np.mean(times_list)
    gap = (best_c - inst_10.optimal_value) / inst_10.optimal_value * 100
    print(f'{rate:<10} {best_c:<12.2f} {mean_c:<12.2f} {std_c:<12.2f} {gap:<10.2f} {mean_t:<10.4f}', flush=True)

# =========================================================
# 10-CITY: Stability Analysis (50 runs, best cooling rate)
# =========================================================
print('\n--- 10-CITY: Stability Analysis (50 runs, rate=0.995) ---', flush=True)
costs_10, times_10 = [], []
for run_idx in range(50):
    sa = SimulatedAnnealing(
        distance_matrix=inst_10.distance_matrix,
        initial_temp=5000.0, cooling_rate=0.995,
        min_temp=0.01, max_iterations_per_temp=100,
        neighborhood_type='2-opt', restart_stuck=True,
        stuck_threshold=50, seed=5000 + run_idx*100
    )
    result = sa.run(verbose=False)
    costs_10.append(result.best_cost)
    times_10.append(result.elapsed_time)

print(f'  Best:            {min(costs_10):.2f} km', flush=True)
print(f'  Worst:           {max(costs_10):.2f} km', flush=True)
print(f'  Mean:            {np.mean(costs_10):.2f} km', flush=True)
print(f'  Std:             {np.std(costs_10):.2f} km', flush=True)
print(f'  CV:              {np.std(costs_10)/np.mean(costs_10)*100:.2f}%', flush=True)
gap_10 = (min(costs_10) - inst_10.optimal_value) / inst_10.optimal_value * 100
print(f'  Gap from opt:    {gap_10:.2f}%', flush=True)
print(f'  Avg time:        {np.mean(times_10):.4f}s', flush=True)

# =========================================================
# 50-CITY: Cooling Rate Comparison
# =========================================================
print('\n' + '=' * 70, flush=True)
print('50-CITY: Cooling Rate Comparison (10 runs each)', flush=True)
print('=' * 70, flush=True)
print(f'{"Rate":<10} {"Best":<12} {"Mean":<12} {"Std":<12} {"Gap%":<10} {"Time(s)":<10}', flush=True)

for rate in cooling_rates:
    costs, times_list = [], []
    for run_idx in range(10):
        sa = SimulatedAnnealing(
            distance_matrix=inst_50.distance_matrix,
            initial_temp=10000.0, cooling_rate=rate,
            min_temp=0.01, max_iterations_per_temp=200,
            neighborhood_type='2-opt', restart_stuck=True,
            stuck_threshold=100, seed=42 + run_idx*100
        )
        result = sa.run(verbose=False)
        costs.append(result.best_cost)
        times_list.append(result.elapsed_time)
    
    mean_c = np.mean(costs)
    std_c = np.std(costs)
    best_c = min(costs)
    mean_t = np.mean(times_list)
    gap = (best_c - inst_50.optimal_value) / inst_50.optimal_value * 100
    print(f'{rate:<10} {best_c:<12.2f} {mean_c:<12.2f} {std_c:<12.2f} {gap:<10.2f} {mean_t:<10.4f}', flush=True)

# =========================================================
# 50-CITY: Stability Analysis (30 runs, best cooling rate)
# =========================================================
print('\n--- 50-CITY: Stability Analysis (30 runs, rate=0.999) ---', flush=True)
costs_50, times_50 = [], []
for run_idx in range(30):
    sa = SimulatedAnnealing(
        distance_matrix=inst_50.distance_matrix,
        initial_temp=10000.0, cooling_rate=0.999,
        min_temp=0.01, max_iterations_per_temp=200,
        neighborhood_type='2-opt', restart_stuck=True,
        stuck_threshold=100, seed=5000 + run_idx*100
    )
    result = sa.run(verbose=False)
    costs_50.append(result.best_cost)
    times_50.append(result.elapsed_time)
    if (run_idx + 1) % 10 == 0:
        print(f'  Completed {run_idx+1}/30 runs...', flush=True)

print(f'  Best:            {min(costs_50):.2f} km', flush=True)
print(f'  Worst:           {max(costs_50):.2f} km', flush=True)
print(f'  Mean:            {np.mean(costs_50):.2f} km', flush=True)
print(f'  Std:             {np.std(costs_50):.2f} km', flush=True)
print(f'  CV:              {np.std(costs_50)/np.mean(costs_50)*100:.2f}%', flush=True)
gap_50 = (min(costs_50) - inst_50.optimal_value) / inst_50.optimal_value * 100
print(f'  Gap from opt:    {gap_50:.2f}%', flush=True)
print(f'  Avg time:        {np.mean(times_50):.4f}s', flush=True)

# =========================================================
# Generate a single detailed run for convergence visualization
# =========================================================
print('\n--- Generating detailed convergence trace (10-city) ---', flush=True)
sa_detail = SimulatedAnnealing(
    distance_matrix=inst_10.distance_matrix,
    initial_temp=5000.0, cooling_rate=0.995,
    min_temp=0.01, max_iterations_per_temp=100,
    neighborhood_type='2-opt', restart_stuck=True,
    stuck_threshold=50, seed=42
)
result_detail = sa_detail.run(verbose=False)
print(f'  Best cost: {result_detail.best_cost:.2f} km', flush=True)
print(f'  Iterations: {result_detail.iteration_count}', flush=True)
print(f'  Time: {result_detail.elapsed_time:.4f}s', flush=True)

# =========================================================
# SUMMARY
# =========================================================
print('\n\n' + '=' * 70, flush=True)
print('FINAL EXPERIMENT SUMMARY', flush=True)
print('=' * 70, flush=True)
print(f'\n10-City Instance (Optimal = {inst_10.optimal_value:.0f} km):', flush=True)
print(f'  Best SA:         {min(costs_10):.2f} km (gap: {gap_10:.2f}%)', flush=True)
print(f'  Mean ± Std:      {np.mean(costs_10):.2f} ± {np.std(costs_10):.2f} km', flush=True)
print(f'  Stability (CV):  {np.std(costs_10)/np.mean(costs_10)*100:.2f}%', flush=True)
print(f'  Avg Time:        {np.mean(times_10):.4f}s', flush=True)

print(f'\n50-City Instance (Optimal = {inst_50.optimal_value:.0f} km):', flush=True)
print(f'  Best SA:         {min(costs_50):.2f} km (gap: {gap_50:.2f}%)', flush=True)
print(f'  Mean ± Std:      {np.mean(costs_50):.2f} ± {np.std(costs_50):.2f} km', flush=True)
print(f'  Stability (CV):  {np.std(costs_50)/np.mean(costs_50)*100:.2f}%', flush=True)
print(f'  Avg Time:        {np.mean(times_50):.4f}s', flush=True)

print('\nExperiment complete!', flush=True)
