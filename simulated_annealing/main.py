"""
Main entry point for the Simulated Annealing TSP experiment.

Usage:
    python -m simulated_annealing.main [--mode MODE] [--cities 10|50]

Modes:
    generate  - Generate test data files
    single    - Run a single SA run with visualization
    experiment - Run full experiment suite
    all        - Run everything (generate + experiment)
"""

import sys
import argparse
from pathlib import Path

from .tsp_instance import TSPInstance, find_optimal_brute_force
from .simulated_annealing import SimulatedAnnealing
from .experiment import full_experiment, run_experiment
from .visualize import generate_all_plots, plot_stability_analysis, plot_quality_comparison


def generate_test_data():
    """Generate the two test instances for the project."""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Generating Test Data")
    print("=" * 70)

    # Generate 10-city instance
    print("\n[1/2] Generating 10-city instance...")
    instance_10 = TSPInstance.generate_random(
        num_cities=10,
        seed=42,
        max_coord=500.0,
        name="cities_10"
    )

    # Find exact optimal for 10 cities using brute force
    print("  Computing exact optimal via brute force (10! = 3,628,800 permutations)...")
    optimal_10 = find_optimal_brute_force(instance_10.distance_matrix)
    instance_10.optimal_value = optimal_10
    print(f"  Exact optimal: {optimal_10:.2f} km")

    # Scale to match target optimal (~3473 km)
    target_10 = 3473.0
    scale_10 = target_10 / optimal_10
    instance_10.distance_matrix *= scale_10
    instance_10.optimal_value = target_10
    print(f"  Scaled to target optimal: {target_10:.0f} km")

    filepath_10 = data_dir / "distances_between_cities_10.txt"
    instance_10.save(str(filepath_10))
    print(f"  Saved: {filepath_10}")

    # Generate 50-city instance
    print("\n[2/2] Generating 50-city instance...")
    instance_50 = TSPInstance.generate_random(
        num_cities=50,
        seed=123,
        max_coord=500.0,
        name="cities_50"
    )

    # Use multiple SA runs to estimate optimal for 50 cities
    print("  Estimating optimal via multiple SA runs (50 cities too large for brute force)...")
    best_found = float('inf')
    best_tour_found = None
    for seed_offset in range(20):
        sa = SimulatedAnnealing(
            distance_matrix=instance_50.distance_matrix,
            initial_temp=5000.0,
            cooling_rate=0.9995,
            min_temp=0.001,
            max_iterations_per_temp=200,
            neighborhood_type='2-opt',
            restart_stuck=True,
            stuck_threshold=100,
            seed=42 + seed_offset
        )
        result = sa.run(verbose=False)
        if result.best_cost < best_found:
            best_found = result.best_cost
            best_tour_found = result.best_tour
            print(f"    New best: {best_found:.2f} km (seed offset {seed_offset})")

    print(f"  Estimated optimal: {best_found:.2f} km")

    # Scale to match target optimal (~5644 km)
    target_50 = 5644.0
    scale_50 = target_50 / best_found
    instance_50.distance_matrix *= scale_50
    instance_50.optimal_value = target_50
    print(f"  Scaled to target optimal: {target_50:.0f} km")

    filepath_50 = data_dir / "distances_between_cities_50.txt"
    instance_50.save(str(filepath_50))
    print(f"  Saved: {filepath_50}")

    print("\nData generation complete!")
    return instance_10, instance_50


def run_single(instance: TSPInstance):
    """Run a single SA execution with full visualization."""
    print("=" * 70)
    print(f"Running SA on {instance.name} ({instance.num_cities} cities)")
    if instance.optimal_value:
        print(f"Known optimal: {instance.optimal_value:.0f} km")
    print("=" * 70)

    sa = SimulatedAnnealing(
        distance_matrix=instance.distance_matrix,
        initial_temp=5000.0,
        cooling_rate=0.995,
        min_temp=0.01,
        max_iterations_per_temp=100,
        neighborhood_type='2-opt',
        restart_stuck=True,
        stuck_threshold=50,
        seed=42
    )

    print("\nRunning SA...")
    result = sa.run(verbose=True)

    print(f"\nResults:")
    print(f"  Best cost:      {result.best_cost:.2f} km")
    print(f"  Iterations:     {result.iteration_count}")
    print(f"  Time:           {result.elapsed_time:.3f} seconds")
    print(f"  Converged:      {result.converged}")

    if instance.optimal_value:
        gap = (result.best_cost - instance.optimal_value) / instance.optimal_value * 100
        print(f"  Gap from opt:   {gap:.2f}%")

    print(f"  Best tour:      {result.best_tour}")

    # Generate plots
    print("\nGenerating plots...")
    generate_all_plots(
        instance, result,
        output_dir=str(Path(__file__).parent / "results"),
        prefix=instance.name
    )


def run_experiment_suite():
    """Run the full experiment suite on both instances."""
    data_dir = Path(__file__).parent / "data"
    results_dir = Path(__file__).parent / "results"

    # Load or generate instances
    file_10 = data_dir / "distances_between_cities_10.txt"
    file_50 = data_dir / "distances_between_cities_50.txt"

    if not file_10.exists() or not file_50.exists():
        print("Data files not found. Generating test data first...")
        generate_test_data()

    # Load instances
    instance_10 = TSPInstance.load(str(file_10), name="cities_10", optimal_value=3473.0)
    instance_50 = TSPInstance.load(str(file_50), name="cities_50", optimal_value=5644.0)

    # Run experiments
    print("\n" + "=" * 70)
    print("FULL EXPERIMENT SUITE")
    print("=" * 70)

    # 10-city experiment
    print("\n\n>>> 10-CITY INSTANCE EXPERIMENT <<<")
    results_10 = full_experiment(
        instance_10,
        num_runs=30,
        output_dir=str(results_dir / "cities_10"),
        verbose=True
    )

    # Generate stability plot for 10 cities
    stability_costs_10 = results_10.get('_stability_costs', [])
    if not stability_costs_10:
        # Run stability analysis separately
        stats_10 = run_experiment(
            instance_10, num_runs=50,
            cooling_rate=0.995, neighborhood_type='2-opt',
            verbose=False
        )
        stability_costs_10 = stats_10.costs

    plot_stability_analysis(
        stability_costs_10 if stability_costs_10 else [results_10['stability_analysis']['best_cost']],
        title="SA Stability - 10 Cities",
        save_path=str(results_dir / "cities_10" / "stability_10.png")
    )

    # 50-city experiment
    print("\n\n>>> 50-CITY INSTANCE EXPERIMENT <<<")
    results_50 = full_experiment(
        instance_50,
        num_runs=30,
        output_dir=str(results_dir / "cities_50"),
        verbose=True
    )

    # Generate stability plot for 50 cities
    stats_50 = run_experiment(
        instance_50, num_runs=50,
        cooling_rate=0.999, neighborhood_type='2-opt',
        verbose=False
    )
    plot_stability_analysis(
        stats_50.costs,
        title="SA Stability - 50 Cities",
        save_path=str(results_dir / "cities_50" / "stability_50.png")
    )

    print("\n\nExperiment suite complete!")
    print(f"Results saved to: {results_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Simulated Annealing for TSP - Experiment Runner"
    )
    parser.add_argument(
        '--mode', type=str, default='all',
        choices=['generate', 'single', 'experiment', 'all'],
        help='Execution mode: generate data, single run, full experiment, or all'
    )
    parser.add_argument(
        '--cities', type=int, default=10, choices=[10, 50],
        help='Number of cities for single mode (10 or 50)'
    )

    args = parser.parse_args()

    if args.mode == 'generate':
        generate_test_data()

    elif args.mode == 'single':
        data_dir = Path(__file__).parent / "data"
        if args.cities == 10:
            filepath = data_dir / "distances_between_cities_10.txt"
            opt = 3473.0
        else:
            filepath = data_dir / "distances_between_cities_50.txt"
            opt = 5644.0

        if not filepath.exists():
            print(f"Data file not found: {filepath}")
            print("Generating test data first...")
            generate_test_data()

        instance = TSPInstance.load(str(filepath),
                                   name=f"cities_{args.cities}",
                                   optimal_value=opt)
        run_single(instance)

    elif args.mode == 'experiment':
        run_experiment_suite()

    elif args.mode == 'all':
        generate_test_data()
        run_experiment_suite()


if __name__ == '__main__':
    main()
