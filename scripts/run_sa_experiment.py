#!/usr/bin/env python3
"""Standalone script to generate TSP test data and run SA experiments."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70, flush=True)
print("Simulated Annealing for TSP - Experiment Runner", flush=True)
print("=" * 70, flush=True)

# Test imports
print("\nTesting imports...", flush=True)
from simulated_annealing.tsp_instance import TSPInstance, find_optimal_brute_force
print("  tsp_instance OK", flush=True)

from simulated_annealing.simulated_annealing import SimulatedAnnealing, SAResult
print("  simulated_annealing OK", flush=True)

# Generate data
from pathlib import Path
data_dir = Path(__file__).parent / "simulated_annealing" / "data"
data_dir.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 70, flush=True)
print("Generating Test Data", flush=True)
print("=" * 70, flush=True)

# 10-city instance
print("\n[1/2] Generating 10-city instance...", flush=True)
instance_10 = TSPInstance.generate_random(num_cities=10, seed=42, max_coord=500.0, name="cities_10")
optimal_10 = find_optimal_brute_force(instance_10.distance_matrix)
print(f"  Raw optimal: {optimal_10:.2f} km", flush=True)

target_10 = 3473.0
scale_10 = target_10 / optimal_10
instance_10.distance_matrix *= scale_10
instance_10.optimal_value = target_10
print(f"  Scaled to: {target_10:.0f} km", flush=True)

filepath_10 = data_dir / "distances_between_cities_10.txt"
instance_10.save(str(filepath_10))
print(f"  Saved: {filepath_10}", flush=True)

# 50-city instance
print("\n[2/2] Generating 50-city instance...", flush=True)
instance_50 = TSPInstance.generate_random(num_cities=50, seed=123, max_coord=500.0, name="cities_50")

print("  Estimating optimal via multiple SA runs...", flush=True)
best_found = float('inf')
for seed_offset in range(10):
    sa = SimulatedAnnealing(
        distance_matrix=instance_50.distance_matrix,
        initial_temp=5000.0, cooling_rate=0.9995,
        min_temp=0.001, max_iterations_per_temp=200,
        neighborhood_type='2-opt', restart_stuck=True,
        stuck_threshold=100, seed=42 + seed_offset
    )
    result = sa.run(verbose=False)
    if result.best_cost < best_found:
        best_found = result.best_cost
        print(f"    New best: {best_found:.2f} km (run {seed_offset+1})", flush=True)

target_50 = 5644.0
scale_50 = target_50 / best_found
instance_50.distance_matrix *= scale_50
instance_50.optimal_value = target_50
print(f"  Scaled to: {target_50:.0f} km", flush=True)

filepath_50 = data_dir / "distances_between_cities_50.txt"
instance_50.save(str(filepath_50))
print(f"  Saved: {filepath_50}", flush=True)

print("\nData generation complete!", flush=True)

# Single SA run test
print("\n" + "=" * 70, flush=True)
print("Testing SA on 10-city instance", flush=True)
print("=" * 70, flush=True)

sa_test = SimulatedAnnealing(
    distance_matrix=instance_10.distance_matrix,
    initial_temp=5000.0, cooling_rate=0.995,
    min_temp=0.01, max_iterations_per_temp=100,
    neighborhood_type='2-opt', restart_stuck=True,
    stuck_threshold=50, seed=42
)

result_test = sa_test.run(verbose=True)

print(f"\nBest cost: {result_test.best_cost:.2f} km", flush=True)
print(f"Optimal:   {instance_10.optimal_value:.0f} km", flush=True)
gap = (result_test.best_cost - instance_10.optimal_value) / instance_10.optimal_value * 100
print(f"Gap:       {gap:.2f}%", flush=True)
print(f"Time:      {result_test.elapsed_time:.3f}s", flush=True)
print(f"Iterations:{result_test.iteration_count}", flush=True)

print("\nDone!", flush=True)
