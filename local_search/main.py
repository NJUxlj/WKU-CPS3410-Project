"""
Main entry point for the Local Search TSP experiment.

Usage:
    python -m local_search.main [--mode MODE] [--cities 10|50]
"""

import sys
import argparse
from pathlib import Path

from simulated_annealing.tsp_instance import TSPInstance
from .local_search import LocalSearch
from .experiment import full_experiment, run_experiment
from .visualize import (
    generate_all_plots, plot_stability_analysis, plot_quality_comparison,
)


DATA_DIR = Path(__file__).resolve().parent.parent / "simulated_annealing" / "data"


def load_instance(num_cities: int) -> TSPInstance:
    file_map = {10: ("distances_between_cities_10.txt", 3473.0),
                50: ("distances_between_cities_50.txt", 5644.0)}
    fname, opt = file_map[num_cities]
    fp = DATA_DIR / fname
    if not fp.exists():
        raise FileNotFoundError(f"Data file not found: {fp}")
    return TSPInstance.load(str(fp), name=f"cities_{num_cities}",
                            optimal_value=opt)


def run_single(instance: TSPInstance):
    print("=" * 70)
    print(f"Running LS on {instance.name} ({instance.num_cities} cities)")
    if instance.optimal_value:
        print(f"Known optimal: {instance.optimal_value:.0f} km")
    print("=" * 70)

    ls = LocalSearch(
        distance_matrix=instance.distance_matrix,
        strategy='best',
        neighborhood='2-opt',
        max_iterations=10000,
        max_no_improve=1000,
        seed=42,
    )
    result = ls.run(num_restarts=30, verbose=True)

    print(f"\nResults:")
    print(f"  Best cost:    {result.best_cost:.2f} km")
    print(f"  Time:         {result.elapsed_time:.3f} seconds")
    print(f"  Unique tours: {result.unique_tours_explored}")
    print(f"  Best tour:    {result.best_tour}")

    if instance.optimal_value:
        gap = (result.best_cost - instance.optimal_value) / instance.optimal_value * 100
        print(f"  Gap from opt: {gap:.2f}%")

    results_dir = Path(__file__).parent / "results" / instance.name
    generate_all_plots(instance, result,
                       output_dir=str(results_dir),
                       prefix=instance.name)


def run_experiment_suite():
    print("=" * 70)
    print("FULL LOCAL SEARCH EXPERIMENT SUITE")
    print("=" * 70)

    results_root = Path(__file__).parent / "results"

    for n_cities in (10, 50):
        instance = load_instance(n_cities)
        print(f"\n>>> {n_cities}-CITY INSTANCE <<<")
        results = full_experiment(
            instance,
            num_runs=30,
            output_dir=str(results_root / instance.name),
            verbose=True,
        )

        # Stability plot
        stats = run_experiment(instance, num_runs=50,
                               strategy='best', neighborhood='2-opt',
                               verbose=False)
        plot_stability_analysis(
            stats.costs,
            title=f"LS Stability - {instance.name}",
            save_path=str(results_root / instance.name / "stability.png"),
        )

    print("\nExperiment suite complete!")


def main():
    parser = argparse.ArgumentParser(description="Local Search for TSP")
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'experiment'],
                        help='Execution mode')
    parser.add_argument('--cities', type=int, default=10, choices=[10, 50],
                        help='Number of cities for single mode')
    args = parser.parse_args()

    if args.mode == 'single':
        instance = load_instance(args.cities)
        run_single(instance)
    elif args.mode == 'experiment':
        run_experiment_suite()


if __name__ == '__main__':
    main()
