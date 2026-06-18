"""
Experiment Runner for Tabu Search on TSP
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from simulated_annealing.tsp_instance import TSPInstance
from .tabu_search import TabuSearch


@dataclass
class RunStats:
    """Statistics for a set of Tabu Search runs."""
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
    mean_tabu_overrides: float = 0.0

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
            'mean_tabu_overrides': self.mean_tabu_overrides,
            'num_runs': len(self.costs),
        }


def run_experiment(instance: TSPInstance,
                   num_runs: int = 20,
                   tabu_tenure: int = 7,
                   max_iterations: int = 500,
                   max_no_improve: int = 100,
                   region_iterations: int = 50,
                   neighborhood_iterations: int = 100,
                   use_aspiration: bool = True,
                   base_seed: int = 42,
                   verbose: bool = False) -> RunStats:
    """Run TS multiple times."""
    stats = RunStats()
    for run_idx in range(num_runs):
        seed = base_seed + run_idx * 100
        ts = TabuSearch(
            distance_matrix=instance.distance_matrix,
            tabu_tenure=tabu_tenure,
            max_iterations=max_iterations,
            max_no_improve=max_no_improve,
            region_iterations=region_iterations,
            neighborhood_iterations=neighborhood_iterations,
            use_aspiration=use_aspiration,
            seed=seed,
        )
        result = ts.run(verbose=False)
        stats.costs.append(result.best_cost)
        stats.times.append(result.elapsed_time)
        stats.iterations.append(result.iterations_run)
        stats.mean_tabu_overrides = (
            (stats.mean_tabu_overrides * (len(stats.costs) - 1)
             + result.tabu_overrides)
            / len(stats.costs)
        )
        if verbose and (run_idx + 1) % max(1, num_runs // 5) == 0:
            print(f"  TS Run {run_idx + 1}/{num_runs}: "
                  f"Best={result.best_cost:.2f}, Time={result.elapsed_time:.3f}s")
    stats.compute(instance.optimal_value)
    if verbose:
        print(f"\n  TS Results: Best={stats.best_cost:.2f}, "
              f"Mean={stats.mean_cost:.2f}±{stats.std_cost:.2f}, "
              f"Gap={stats.gap_from_optimal:.2f}%, "
              f"Time={stats.mean_time:.3f}±{stats.std_time:.3f}s")
    return stats


def compare_tabu_tenures(instance: TSPInstance, num_runs: int = 10,
                         tenures: Optional[List[int]] = None,
                         verbose: bool = True) -> Dict[int, RunStats]:
    """Compare different tabu tenure values."""
    if tenures is None:
        tenures = [3, 5, 7, 10, 15]

    results: Dict[int, RunStats] = {}
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Comparing Tabu Tenures on {instance.name}")
        print(f"{'=' * 70}")
        print(f"{'Tenure':<10} {'Best':<12} {'Mean±Std':<22} {'Gap%':<10} {'Time(s)':<10}")

    for t in tenures:
        if verbose:
            print(f"Testing tenure={t}...")
        stats = run_experiment(
            instance=instance, num_runs=num_runs, tabu_tenure=t, verbose=False,
        )
        results[t] = stats
        if verbose:
            print(f"  {t:<8} {stats.best_cost:<12.2f} "
                  f"{stats.mean_cost:<8.2f}±{stats.std_cost:<8.2f} "
                  f"{stats.gap_from_optimal:<10.2f} {stats.mean_time:<10.3f}")
    return results


def full_experiment(instance: TSPInstance, num_runs: int = 20,
                    output_dir: str = "results",
                    verbose: bool = True) -> Dict:
    """Comprehensive TS experiment: tenure comparison + stability."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results: Dict = {
        'instance_name': instance.name,
        'num_cities': instance.num_cities,
        'optimal_value': instance.optimal_value,
        'num_runs': num_runs,
    }

    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 1: Comparing Tabu Tenures")
        print("=" * 70)
    tenure_results = compare_tabu_tenures(instance, num_runs=10, verbose=verbose)
    all_results['tenure_comparison'] = {
        str(t): stats.to_dict() for t, stats in tenure_results.items()
    }
    best_tenure = min(tenure_results.items(),
                      key=lambda x: x[1].mean_cost)[0]

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"PHASE 2: Stability Analysis (tenure={best_tenure})")
        print(f"{'=' * 70}")
    stability_stats = run_experiment(
        instance=instance, num_runs=num_runs, tabu_tenure=best_tenure,
        verbose=verbose,
    )
    all_results['stability_analysis'] = stability_stats.to_dict()

    results_file = output_path / f"{instance.name}_experiment_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    if verbose:
        print(f"\nResults saved to {results_file}")
    return all_results
