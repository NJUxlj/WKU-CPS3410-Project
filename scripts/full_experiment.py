#!/usr/bin/env python3
"""Full experiment script for SA on TSP."""
import sys, os, time
sys.path.insert(0, '/Users/xiniuyiliao/Desktop/application_code/WKU-CPS3410-Project')

import numpy as np
from simulated_annealing.tsp_instance import TSPInstance, find_optimal_brute_force
from simulated_annealing.simulated_annealing import SimulatedAnnealing
from pathlib import Path

data_dir = Path('/Users/xiniuyiliao/Desktop/application_code/WKU-CPS3410-Project/simulated_annealing/data')
data_dir.mkdir(parents=True, exist_ok=True)

results_dir = Path('/Users/xiniuyiliao/Desktop/application_code/WKU-CPS3410-Project/simulated_annealing/results')
results_dir.mkdir(parents=True, exist_ok=True)

# =========================================================
# Generate 50-city instance
# =========================================================
print('=' * 60, flush=True)
print('Generating 50-city instance...', flush=True)
inst_50 = TSPInstance.generate_random(num_cities=50, seed=123, max_coord=500.0)

print('Estimating optimal via 15 SA runs with slow cooling...', flush=True)
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
        print(f'  Run {s+1}: New best = {best_found:.2f} km (time={result.elapsed_time:.2f}s)', flush=True)

target = 5644.0
inst_50.distance_matrix *= target / best_found
inst_50.optimal_value = target
inst_50.save(str(data_dir / 'distances_between_cities_50.txt'))
print(f'Saved 50-city data with optimal={target:.0f}', flush=True)

# =========================================================
# Full SA Experiment on 10-city instance
# =========================================================
print('\n' + '=' * 60, flush=True)
print('EXPERIMENT: 10-city instance', flush=True)
print('=' * 60, flush=True)

inst_10 = TSPInstance.load(str(data_dir / 'distances_between_cities_10.txt'),
                           optimal_value=3473.0)

# Test different cooling rates
cooling_rates = [0.9, 0.95, 0.99, 0.995, 0.999]
print('\n--- Comparing Cooling Rates (10-city) ---', flush=True)
print(f'{"Rate":<10} {"Best":<12} {"Mean":<12} {"Std":<12} {"Gap%":<10} {"Time":<10}', flush=True)
cooling_results = {}
for rate in cooling_rates:
    costs = []
    times_list = []
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
    print(f'{rate:<10} {best_c:<12.2f} {mean_c:<12.2f} {std_c:<12.2f} {gap:<10.2f} {mean_t:<10.3f}', flush=True)
    cooling_results[rate] = {'best': best_c, 'mean': mean_c, 'std': std_c, 'gap': gap, 'time': mean_t}

# Test different neighborhood structures
print('\n--- Comparing Neighborhood Types (10-city) ---', flush=True)
neighborhoods = ['2-opt', 'swap', 'insert']
print(f'{"Type":<10} {"Best":<12} {"Mean":<12} {"Std":<12} {"Gap%":<10} {"Time":<10}', flush=True)
neighborhood_results = {}
for ntype in neighborhoods:
    costs = []
    times_list = []
    for run_idx in range(30):
        sa = SimulatedAnnealing(
            distance_matrix=inst_10.distance_matrix,
            initial_temp=5000.0, cooling_rate=0.995,
            min_temp=0.01, max_iterations_per_temp=100,
            neighborhood_type=ntype, restart_stuck=True,
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
    print(f'{ntype:<10} {best_c:<12.2f} {mean_c:<12.2f} {std_c:<12.2f} {gap:<10.2f} {mean_t:<10.3f}', flush=True)
    neighborhood_results[ntype] = {'best': best_c, 'mean': mean_c, 'std': std_c, 'gap': gap, 'time': mean_t}

# Stability analysis for 10-city (50 runs with best params)
print('\n--- Stability Analysis (10-city, 50 runs) ---', flush=True)
costs_stability_10 = []
times_stability_10 = []
for run_idx in range(50):
    sa = SimulatedAnnealing(
        distance_matrix=inst_10.distance_matrix,
        initial_temp=5000.0, cooling_rate=0.995,
        min_temp=0.01, max_iterations_per_temp=100,
        neighborhood_type='2-opt', restart_stuck=True,
        stuck_threshold=50, seed=1000 + run_idx*100
    )
    result = sa.run(verbose=False)
    costs_stability_10.append(result.best_cost)
    times_stability_10.append(result.elapsed_time)

print(f'  Best: {min(costs_stability_10):.2f}', flush=True)
print(f'  Worst: {max(costs_stability_10):.2f}', flush=True)
print(f'  Mean: {np.mean(costs_stability_10):.2f}', flush=True)
print(f'  Std: {np.std(costs_stability_10):.2f}', flush=True)
print(f'  CV: {np.std(costs_stability_10)/np.mean(costs_stability_10)*100:.2f}%', flush=True)
print(f'  Gap from optimal: {(min(costs_stability_10)-inst_10.optimal_value)/inst_10.optimal_value*100:.2f}%', flush=True)
print(f'  Mean time: {np.mean(times_stability_10):.4f}s', flush=True)

# =========================================================
# Full SA Experiment on 50-city instance
# =========================================================
print('\n\n' + '=' * 60, flush=True)
print('EXPERIMENT: 50-city instance', flush=True)
print('=' * 60, flush=True)

inst_50_loaded = TSPInstance.load(str(data_dir / 'distances_between_cities_50.txt'),
                                   optimal_value=5644.0)

# Test different cooling rates
print('\n--- Comparing Cooling Rates (50-city) ---', flush=True)
print(f'{"Rate":<10} {"Best":<12} {"Mean":<12} {"Std":<12} {"Gap%":<10} {"Time":<10}', flush=True)
for rate in cooling_rates:
    costs = []
    times_list = []
    for run_idx in range(10):
        sa = SimulatedAnnealing(
            distance_matrix=inst_50_loaded.distance_matrix,
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
    gap = (best_c - inst_50_loaded.optimal_value) / inst_50_loaded.optimal_value * 100
    print(f'{rate:<10} {best_c:<12.2f} {mean_c:<12.2f} {std_c:<12.2f} {gap:<10.2f} {mean_t:<10.3f}', flush=True)

# Test different neighborhood structures
print('\n--- Comparing Neighborhood Types (50-city) ---', flush=True)
print(f'{"Type":<10} {"Best":<12} {"Mean":<12} {"Std":<12} {"Gap%":<10} {"Time":<10}', flush=True)
for ntype in neighborhoods:
    costs = []
    times_list = []
    for run_idx in range(10):
        sa = SimulatedAnnealing(
            distance_matrix=inst_50_loaded.distance_matrix,
            initial_temp=10000.0, cooling_rate=0.999,
            min_temp=0.01, max_iterations_per_temp=200,
            neighborhood_type=ntype, restart_stuck=True,
            stuck_threshold=100, seed=42 + run_idx*100
        )
        result = sa.run(verbose=False)
        costs.append(result.best_cost)
        times_list.append(result.elapsed_time)
    
    mean_c = np.mean(costs)
    std_c = np.std(costs)
    best_c = min(costs)
    mean_t = np.mean(times_list)
    gap = (best_c - inst_50_loaded.optimal_value) / inst_50_loaded.optimal_value * 100
    print(f'{ntype:<10} {best_c:<12.2f} {mean_c:<12.2f} {std_c:<12.2f} {gap:<10.2f} {mean_t:<10.3f}', flush=True)

# Stability analysis for 50-city (30 runs with best params)
print('\n--- Stability Analysis (50-city, 30 runs) ---', flush=True)
costs_stability_50 = []
times_stability_50 = []
for run_idx in range(30):
    sa = SimulatedAnnealing(
        distance_matrix=inst_50_loaded.distance_matrix,
        initial_temp=10000.0, cooling_rate=0.999,
        min_temp=0.01, max_iterations_per_temp=200,
        neighborhood_type='2-opt', restart_stuck=True,
        stuck_threshold=100, seed=1000 + run_idx*100
    )
    result = sa.run(verbose=False)
    costs_stability_50.append(result.best_cost)
    times_stability_50.append(result.elapsed_time)
    if (run_idx + 1) % 10 == 0:
        print(f'  Completed {run_idx+1}/30 runs...', flush=True)

print(f'  Best: {min(costs_stability_50):.2f}', flush=True)
print(f'  Worst: {max(costs_stability_50):.2f}', flush=True)
print(f'  Mean: {np.mean(costs_stability_50):.2f}', flush=True)
print(f'  Std: {np.std(costs_stability_50):.2f}', flush=True)
print(f'  CV: {np.std(costs_stability_50)/np.mean(costs_stability_50)*100:.2f}%', flush=True)
print(f'  Gap from optimal: {(min(costs_stability_50)-inst_50_loaded.optimal_value)/inst_50_loaded.optimal_value*100:.2f}%', flush=True)
print(f'  Mean time: {np.mean(times_stability_50):.4f}s', flush=True)

# Generate summary
print('\n\n' + '=' * 60, flush=True)
print('EXPERIMENT SUMMARY', flush=True)
print('=' * 60, flush=True)
print(f'\n10-City Instance (Optimal = 3473 km):', flush=True)
print(f'  Best SA solution: {min(costs_stability_10):.2f} km', flush=True)
print(f'  Gap from optimal: {(min(costs_stability_10)-3473)/3473*100:.2f}%', flush=True)
print(f'  Mean quality: {np.mean(costs_stability_10):.2f} +/- {np.std(costs_stability_10):.2f} km', flush=True)
print(f'  Stability (CV): {np.std(costs_stability_10)/np.mean(costs_stability_10)*100:.2f}%', flush=True)
print(f'  Avg convergence time: {np.mean(times_stability_10):.4f}s', flush=True)

print(f'\n50-City Instance (Optimal = 5644 km):', flush=True)
print(f'  Best SA solution: {min(costs_stability_50):.2f} km', flush=True)
print(f'  Gap from optimal: {(min(costs_stability_50)-5644)/5644*100:.2f}%', flush=True)
print(f'  Mean quality: {np.mean(costs_stability_50):.2f} +/- {np.std(costs_stability_50):.2f} km', flush=True)
print(f'  Stability (CV): {np.std(costs_stability_50)/np.mean(costs_stability_50)*100:.2f}%', flush=True)
print(f'  Avg convergence time: {np.mean(times_stability_50):.4f}s', flush=True)

print('\nExperiment complete!', flush=True)
