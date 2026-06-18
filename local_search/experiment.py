"""
Experiment Runner for Local Search on TSP

Runs Local Search multiple times with different parameter configurations and
collects statistics for quality, convergence time, and stability analysis.
"""

import numpy as np
import time
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from simulated_annealing.tsp_instance import TSPInstance
from .local_search import LocalSearch, LSResult


@dataclass
class RunStats:
    """Statistics for a set of Local Search runs (each run = multi-restart)."""
    costs: List[float] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    iterations: List[int] = field(default_factory=list)
    best_cost: float = float('inf')
    worst_cost: float = 0.0
    mean_cost: float = 0.0
    std_cost: float = 0.0
    mean_time: float = 0.0
    std_time: float = 0.0
    mean_iterations: float = 0.0
    gap_from_optimal: float = 0.0
    optimal_hit_rate: float = 0.0
    mean_unique_tours: float = 0.0

    def compute(self, optimal_value: Optional[float] = None):
        if not self.costs:
            return
        self.best_cost = float(np.min(self.costs))
        self.worst_cost = float(np.max(self.costs))
        self.mean_cost = float(np.mean(self.costs))
        self.std_cost = float(np.std(self.costs))
        self.mean_time = float(np.mean(self.times))
        self.std_time = float(np.std(self.times))
        self.mean_iterations = float(np.mean(self.iterations))

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
            'mean_iterations': self.mean_iterations,
            'gap_from_optimal_pct': self.gap_from_optimal,
            'optimal_hit_rate': self.optimal_hit_rate,
            'mean_unique_tours': self.mean_unique_tours,
            'num_runs': len(self.costs),
        }


def run_experiment(instance: TSPInstance,
                   num_runs: int = 30,
                   strategy: str = 'best',
                   neighborhood: str = '2-opt',
                   max_iterations: int = 10000,
                   max_no_improve: int = 1000,
                   num_restarts: int = 30,
                   base_seed: int = 42,
                   verbose: bool = False) -> RunStats:
    """
    Run Local Search (multi-restart) multiple times.

    Args:
        instance: TSP problem instance
        num_runs: Number of independent LS experiment rounds
        strategy: 'best' or 'first'
        neighborhood: 'swap' / '2-opt' / 'or-opt' / '3-opt'
        max_iterations: Max neighbor evaluations per restart
        max_no_improve: Stop restart early after this many non-improving iters
        num_restarts: Number of random restarts within each LS run
        base_seed: Starting random seed
        verbose: Print progress

    Returns:
        RunStats with aggregated statistics
    """
    stats = RunStats()

    for run_idx in range(num_runs):
        seed = base_seed + run_idx * 100

        ls = LocalSearch(
            distance_matrix=instance.distance_matrix,
            strategy=strategy,
            neighborhood=neighborhood,
            max_iterations=max_iterations,
            max_no_improve=max_no_improve,
            seed=seed,
        )
        result = ls.run(num_restarts=num_restarts, verbose=False)

        stats.costs.append(result.best_cost)
        stats.times.append(result.elapsed_time)
        stats.iterations.append(sum(result.iterations_per_restart))
        stats.mean_unique_tours = (
            (stats.mean_unique_tours * len(stats.costs) + result.unique_tours_explored)
            / (len(stats.costs) + 1)
        )

        if verbose and (run_idx + 1) % max(1, num_runs // 5) == 0:
            print(f"  Run {run_idx + 1}/{num_runs}: "
                  f"Best={result.best_cost:.2f}, "
                  f"Time={result.elapsed_time:.3f}s, "
                  f"Unique tours={result.unique_tours_explored}")

    stats.compute(instance.optimal_value)

    if verbose:
        print(f"\n  Results: Best={stats.best_cost:.2f}, "
              f"Mean={stats.mean_cost:.2f}±{stats.std_cost:.2f}, "
              f"Gap={stats.gap_from_optimal:.2f}%, "
              f"Time={stats.mean_time:.3f}±{stats.std_time:.3f}s")

    return stats


def compare_neighborhoods(instance: TSPInstance, num_runs: int = 10,
                          verbose: bool = True) -> Dict[str, RunStats]:
    """Compare different neighborhood structures on the same instance."""
    neighborhoods = ['swap', '2-opt', 'or-opt', '3-opt']
    results: Dict[str, RunStats] = {}

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Comparing Neighborhood Structures on {instance.name}")
        print(f"{'=' * 70}")
        print(f"{'Type':<10} {'Best':<12} {'Mean±Std':<22} {'Gap%':<10} {'Time(s)':<10}")

    for ntype in neighborhoods:
        if verbose:
            print(f"Testing neighborhood={ntype}...")
        stats = run_experiment(
            instance=instance,
            num_runs=num_runs,
            neighborhood=ntype,
            verbose=False,
        )
        results[ntype] = stats
        if verbose:
            print(f"  {ntype:<8} {stats.best_cost:<12.2f} "
                  f"{stats.mean_cost:<8.2f}±{stats.std_cost:<8.2f} "
                  f"{stats.gap_from_optimal:<10.2f} {stats.mean_time:<10.3f}")

    return results


def compare_strategies(instance: TSPInstance, num_runs: int = 10,
                        verbose: bool = True) -> Dict[str, RunStats]:
    """Compare best-improvement vs first-improvement."""
    strategies = ['best', 'first']
    results: Dict[str, RunStats] = {}

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Comparing Improvement Strategies on {instance.name}")
        print(f"{'=' * 70}")

    for s in strategies:
        if verbose:
            print(f"Testing strategy={s}...")
        stats = run_experiment(
            instance=instance,
            num_runs=num_runs,
            strategy=s,
            neighborhood='2-opt',
            verbose=False,
        )
        results[s] = stats

    if verbose:
        print(f"\n{'Strategy':<10} {'Best':<12} {'Mean±Std':<22} {'Time(s)':<10}")
        for s, stats in results.items():
            print(f"  {s:<8} {stats.best_cost:<12.2f} "
                  f"{stats.mean_cost:<8.2f}±{stats.std_cost:<8.2f} "
                  f"{stats.mean_time:<10.3f}")

    return results


def full_experiment(instance: TSPInstance, num_runs: int = 30,
                    output_dir: str = "results",
                    verbose: bool = True) -> Dict:
    """Run comprehensive LS experiment: neighborhood comparison + stability."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results: Dict = {
        'instance_name': instance.name,
        'num_cities': instance.num_cities,
        'optimal_value': instance.optimal_value,
        'num_runs': num_runs,
    }

    # Phase 1: Neighborhood comparison
    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 1: Comparing Neighborhood Structures")
        print("=" * 70)
    neighborhood_results = compare_neighborhoods(instance, num_runs=10, verbose=verbose)
    all_results['neighborhood_comparison'] = {
        ntype: stats.to_dict() for ntype, stats in neighborhood_results.items()
    }
    best_neighborhood = min(neighborhood_results.items(),
                            key=lambda x: x[1].mean_cost)[0]

    # Phase 2: Strategy comparison
    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 2: Comparing Improvement Strategies")
        print("=" * 70)
    strategy_results = compare_strategies(instance, num_runs=10, verbose=verbose)
    all_results['strategy_comparison'] = {
        s: stats.to_dict() for s, stats in strategy_results.items()
    }
    best_strategy = min(strategy_results.items(),
                        key=lambda x: x[1].mean_cost)[0]

    # Phase 3: Stability analysis with best params
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"PHASE 3: Stability Analysis (best={best_neighborhood}/{best_strategy})")
        print(f"{'=' * 70}")
    stability_stats = run_experiment(
        instance=instance,
        num_runs=num_runs,
        strategy=best_strategy,
        neighborhood=best_neighborhood,
        verbose=verbose,
    )
    all_results['stability_analysis'] = stability_stats.to_dict()

    results_file = output_path / f"{instance.name}_experiment_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    if verbose:
        print(f"\nResults saved to {results_file}")

    return all_results
