"""
Main entry point for Tabu Search TSP experiment.

Usage:
    python -m tabu_search.main [--mode MODE] [--cities 10|50]
"""

import argparse
from pathlib import Path

from simulated_annealing.tsp_instance import TSPInstance
from .tabu_search import TabuSearch
from .experiment import full_experiment, run_experiment
from .visualize import (
    generate_all_plots, plot_stability_analysis,
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
    print(f"Running TS on {instance.name} ({instance.num_cities} cities)")
    if instance.optimal_value:
        print(f"Known optimal: {instance.optimal_value:.0f} km")
    print("=" * 70)

    ts = TabuSearch(
        distance_matrix=instance.distance_matrix,
        tabu_tenure=7,
        max_iterations=5000,
        max_no_improve=200,
        region_iterations=50,
        neighborhood_iterations=100,
        use_aspiration=True,
        seed=42,
    )
    result = ts.run(verbose=True)

    print(f"\nResults:")
    print(f"  Best cost:        {result.best_cost:.2f} km")
    print(f"  Time:             {result.elapsed_time:.3f} seconds")
    print(f"  Iterations:       {result.iterations_run}")
    print(f"  Tabu overrides:   {result.tabu_overrides}")
    print(f"  Converged:        {result.converged}")
    print(f"  Best tour:        {result.best_tour}")

    if instance.optimal_value:
        gap = (result.best_cost - instance.optimal_value) / instance.optimal_value * 100
        print(f"  Gap from opt:     {gap:.2f}%")

    results_dir = Path(__file__).parent / "results" / instance.name
    generate_all_plots(instance, result,
                       output_dir=str(results_dir),
                       prefix=instance.name)


def run_experiment_suite():
    print("=" * 70)
    print("FULL TABU SEARCH EXPERIMENT SUITE")
    print("=" * 70)

    results_root = Path(__file__).parent / "results"

    for n_cities in (10, 50):
        instance = load_instance(n_cities)
        print(f"\n>>> {n_cities}-CITY INSTANCE <<<")
        full_experiment(
            instance,
            num_runs=20,
            output_dir=str(results_root / instance.name),
            verbose=True,
        )

        stats = run_experiment(instance, num_runs=30, tabu_tenure=7,
                               verbose=False)
        plot_stability_analysis(
            stats.costs,
            title=f"TS Stability - {instance.name}",
            save_path=str(results_root / instance.name / "stability.png"),
        )

    print("\nExperiment suite complete!")


def main():
    parser = argparse.ArgumentParser(description="Tabu Search for TSP")
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'experiment'])
    parser.add_argument('--cities', type=int, default=10, choices=[10, 50])
    args = parser.parse_args()

    if args.mode == 'single':
        instance = load_instance(args.cities)
        run_single(instance)
    elif args.mode == 'experiment':
        run_experiment_suite()


if __name__ == '__main__':
    main()
