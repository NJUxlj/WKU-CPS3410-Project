"""
Main entry point for VNS / RVNS TSP experiment.

Usage:
    python -m vns.main [--mode MODE] [--cities 10|50] [--method VNS|RVNS]
"""

import argparse
from pathlib import Path

from simulated_annealing.tsp_instance import TSPInstance
from .vns import VNS, RVNS
from .experiment import (
    full_experiment, run_vns_experiment, run_rvns_experiment, compare_vns_rvns,
)
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


def run_single(instance: TSPInstance, method: str = 'VNS'):
    print("=" * 70)
    print(f"Running {method} on {instance.name} ({instance.num_cities} cities)")
    if instance.optimal_value:
        print(f"Known optimal: {instance.optimal_value:.0f} km")
    print("=" * 70)

    if method == 'VNS':
        solver = VNS(distance_matrix=instance.distance_matrix,
                     k_max=5, max_iterations=50, max_no_improve=20,
                     local_search_iterations=2000, local_search_no_improve=500,
                     seed=42)
    else:
        solver = RVNS(distance_matrix=instance.distance_matrix,
                      k_max=5, max_iterations=300, max_no_improve=80,
                      seed=42)

    result = solver.run(verbose=True)

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
                       prefix=f"{method.lower()}_{instance.name}")


def run_experiment_suite():
    print("=" * 70)
    print("FULL VNS / RVNS EXPERIMENT SUITE")
    print("=" * 70)

    results_root = Path(__file__).parent / "results"

    for n_cities in (10, 50):
        instance = load_instance(n_cities)
        print(f"\n>>> {n_cities}-CITY INSTANCE <<<")
        full_experiment(
            instance,
            num_runs=15,
            output_dir=str(results_root / instance.name),
            verbose=True,
        )

        # Stability plots
        for method, runner in [('VNS', run_vns_experiment),
                               ('RVNS', run_rvns_experiment)]:
            stats = runner(instance, num_runs=30, verbose=False)
            plot_stability_analysis(
                stats.costs,
                title=f"{method} Stability - {instance.name}",
                save_path=str(results_root / instance.name /
                              f"stability_{method.lower()}.png"),
            )

    print("\nExperiment suite complete!")


def main():
    parser = argparse.ArgumentParser(description="VNS / RVNS for TSP")
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'experiment'])
    parser.add_argument('--cities', type=int, default=10, choices=[10, 50])
    parser.add_argument('--method', type=str, default='VNS',
                        choices=['VNS', 'RVNS'])
    args = parser.parse_args()

    if args.mode == 'single':
        instance = load_instance(args.cities)
        run_single(instance, method=args.method)
    elif args.mode == 'experiment':
        run_experiment_suite()


if __name__ == '__main__':
    main()
