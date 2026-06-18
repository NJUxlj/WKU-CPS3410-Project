"""
Experiment Runner for Genetic Algorithm on TSP
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from simulated_annealing.tsp_instance import TSPInstance
from .genetic_algorithm import GeneticAlgorithm, GAResult, MUTATION_FUNCTIONS


@dataclass
class RunStats:
    """Statistics for a set of GA runs."""
    costs: List[float] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    generations: List[int] = field(default_factory=list)
    best_cost: float = float('inf')
    worst_cost: float = 0.0
    mean_cost: float = 0.0
    std_cost: float = 0.0
    mean_time: float = 0.0
    std_time: float = 0.0
    mean_generations: float = 0.0
    gap_from_optimal: float = 0.0
    optimal_hit_rate: float = 0.0
    mean_final_diversity: float = 0.0

    def compute(self, optimal_value: Optional[float] = None):
        if not self.costs:
            return
        self.best_cost = float(np.min(self.costs))
        self.worst_cost = float(np.max(self.costs))
        self.mean_cost = float(np.mean(self.costs))
        self.std_cost = float(np.std(self.costs))
        self.mean_time = float(np.mean(self.times))
        self.std_time = float(np.std(self.times))
        self.mean_generations = float(np.mean(self.generations))
        if optimal_value and optimal_value > 0:
            self.gap_from_optimal = (
                (self.best_cost - optimal_value) / optimal_value * 100
            )
            self.optimal_hit_rate = sum(
                1 for c in self.costs if abs(c - optimal_value) < 0.01
            ) / len(self.costs)

    def to_dict(self) -> Dict:
        return {
            'best_cost': self.best_cost,
            'worst_cost': self.worst_cost,
            'mean_cost': self.mean_cost,
            'std_cost': self.std_cost,
            'mean_time': self.mean_time,
            'std_time': self.std_time,
            'mean_generations': self.mean_generations,
            'gap_from_optimal_pct': self.gap_from_optimal,
            'optimal_hit_rate': self.optimal_hit_rate,
            'mean_final_diversity': self.mean_final_diversity,
            'num_runs': len(self.costs),
        }


def run_experiment(instance: TSPInstance,
                   num_runs: int = 20,
                   population_size: int = 100,
                   num_generations: int = 500,
                   elite_count: int = 20,
                   crossover_rate: float = 0.9,
                   mutation_rate: float = 0.2,
                   selection_method: str = 'roulette',
                   crossover_method: str = 'ox',
                   mutation_method: str = 'swap',
                   base_seed: int = 42,
                   verbose: bool = False) -> RunStats:
    """Run the GA multiple times and collect statistics."""
    stats = RunStats()

    for run_idx in range(num_runs):
        seed = base_seed + run_idx * 100
        ga = GeneticAlgorithm(
            distance_matrix=instance.distance_matrix,
            population_size=population_size,
            num_generations=num_generations,
            elite_count=elite_count,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            selection_method=selection_method,
            crossover_method=crossover_method,
            mutation_method=mutation_method,
            seed=seed,
        )
        result = ga.run(verbose=False)

        stats.costs.append(result.best_cost)
        stats.times.append(result.elapsed_time)
        stats.generations.append(result.generations_run)
        stats.mean_final_diversity = (
            (stats.mean_final_diversity * (len(stats.costs) - 1)
             + result.diversity_history[-1])
            / len(stats.costs)
        )

        if verbose and (run_idx + 1) % max(1, num_runs // 5) == 0:
            print(f"  Run {run_idx + 1}/{num_runs}: "
                  f"Best={result.best_cost:.2f}, "
                  f"Time={result.elapsed_time:.3f}s")

    stats.compute(instance.optimal_value)
    if verbose:
        print(f"\n  Results: Best={stats.best_cost:.2f}, "
              f"Mean={stats.mean_cost:.2f}±{stats.std_cost:.2f}, "
              f"Gap={stats.gap_from_optimal:.2f}%, "
              f"Time={stats.mean_time:.3f}±{stats.std_time:.3f}s")
    return stats


def compare_mutation_methods(instance: TSPInstance, num_runs: int = 10,
                             verbose: bool = True) -> Dict[str, RunStats]:
    """Compare the 5 mutation neighborhood variants (mirrors GA report)."""
    methods = ['swap', 'inversion', 'insert', '2-opt', '3-opt']
    results: Dict[str, RunStats] = {}

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Comparing Mutation Methods on {instance.name}")
        print(f"{'=' * 70}")
        print(f"{'Method':<12} {'Best':<12} {'Mean±Std':<22} {'Gap%':<10} {'Time(s)':<10}")

    for m in methods:
        if verbose:
            print(f"Testing mutation={m}...")
        stats = run_experiment(
            instance=instance,
            num_runs=num_runs,
            mutation_method=m,
            verbose=False,
        )
        results[m] = stats
        if verbose:
            print(f"  {m:<10} {stats.best_cost:<12.2f} "
                  f"{stats.mean_cost:<8.2f}±{stats.std_cost:<8.2f} "
                  f"{stats.gap_from_optimal:<10.2f} {stats.mean_time:<10.3f}")
    return results


def compare_selection_methods(instance: TSPInstance, num_runs: int = 10,
                              verbose: bool = True) -> Dict[str, RunStats]:
    """Compare selection methods: roulette / tournament / sus."""
    methods = ['roulette', 'tournament', 'sus']
    results: Dict[str, RunStats] = {}

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Comparing Selection Methods on {instance.name}")
        print(f"{'=' * 70}")

    for m in methods:
        if verbose:
            print(f"Testing selection={m}...")
        stats = run_experiment(
            instance=instance,
            num_runs=num_runs,
            selection_method=m,
            verbose=False,
        )
        results[m] = stats

    if verbose:
        print(f"\n{'Method':<12} {'Best':<12} {'Mean±Std':<22} {'Time(s)':<10}")
        for s, stats in results.items():
            print(f"  {s:<10} {stats.best_cost:<12.2f} "
                  f"{stats.mean_cost:<8.2f}±{stats.std_cost:<8.2f} "
                  f"{stats.mean_time:<10.3f}")
    return results


def full_experiment(instance: TSPInstance, num_runs: int = 20,
                    output_dir: str = "results",
                    verbose: bool = True) -> Dict:
    """Comprehensive GA experiment: mutation comparison + selection + stability."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results: Dict = {
        'instance_name': instance.name,
        'num_cities': instance.num_cities,
        'optimal_value': instance.optimal_value,
        'num_runs': num_runs,
    }

    # Phase 1: Mutation comparison (matches GA report)
    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 1: Comparing Mutation Neighborhood Variants")
        print("=" * 70)
    mutation_results = compare_mutation_methods(instance, num_runs=10, verbose=verbose)
    all_results['mutation_comparison'] = {
        m: stats.to_dict() for m, stats in mutation_results.items()
    }
    best_mutation = min(mutation_results.items(),
                        key=lambda x: x[1].mean_cost)[0]

    # Phase 2: Selection comparison
    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 2: Comparing Selection Methods")
        print("=" * 70)
    selection_results = compare_selection_methods(instance, num_runs=10, verbose=verbose)
    all_results['selection_comparison'] = {
        m: stats.to_dict() for m, stats in selection_results.items()
    }
    best_selection = min(selection_results.items(),
                         key=lambda x: x[1].mean_cost)[0]

    # Phase 3: Stability analysis
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"PHASE 3: Stability Analysis (best={best_mutation}/{best_selection})")
        print(f"{'=' * 70}")
    stability_stats = run_experiment(
        instance=instance,
        num_runs=num_runs,
        mutation_method=best_mutation,
        selection_method=best_selection,
        verbose=verbose,
    )
    all_results['stability_analysis'] = stability_stats.to_dict()

    results_file = output_path / f"{instance.name}_experiment_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    if verbose:
        print(f"\nResults saved to {results_file}")
    return all_results
