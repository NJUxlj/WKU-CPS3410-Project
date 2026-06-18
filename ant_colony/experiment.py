"""
Experiment Runner for Ant Colony Optimization on TSP
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from simulated_annealing.tsp_instance import TSPInstance
from .aco import AntColony


@dataclass
class RunStats:
    """Statistics for a set of ACO runs."""
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
            'num_runs': len(self.costs),
        }


def run_experiment(instance: TSPInstance,
                   num_runs: int = 10,
                   num_ants: int = 30,
                   num_iterations: int = 500,
                   alpha: float = 1.0,
                   beta: float = 2.0,
                   rho: float = 0.1,
                   use_acs: bool = False,
                   use_best_only: bool = True,
                   base_seed: int = 42,
                   verbose: bool = False) -> RunStats:
    """Run ACO multiple times."""
    stats = RunStats()
    for run_idx in range(num_runs):
        seed = base_seed + run_idx * 100
        aco = AntColony(
            distance_matrix=instance.distance_matrix,
            num_ants=num_ants,
            num_iterations=num_iterations,
            alpha=alpha,
            beta=beta,
            rho=rho,
            use_acs=use_acs,
            use_best_only=use_best_only,
            seed=seed,
        )
        result = aco.run(verbose=False)
        stats.costs.append(result.best_cost)
        stats.times.append(result.elapsed_time)
        stats.iterations.append(result.iterations_run)
        if verbose and (run_idx + 1) % max(1, num_runs // 5) == 0:
            print(f"  ACO Run {run_idx + 1}/{num_runs}: "
                  f"Best={result.best_cost:.2f}, Time={result.elapsed_time:.3f}s")
    stats.compute(instance.optimal_value)
    if verbose:
        print(f"\n  ACO Results: Best={stats.best_cost:.2f}, "
              f"Mean={stats.mean_cost:.2f}±{stats.std_cost:.2f}, "
              f"Gap={stats.gap_from_optimal:.2f}%, "
              f"Time={stats.mean_time:.3f}±{stats.std_time:.3f}s")
    return stats


def compare_alpha_beta(instance: TSPInstance, num_runs: int = 5,
                       verbose: bool = True) -> Dict[str, RunStats]:
    """Compare different α/β parameter combinations."""
    configs = [
        ('alpha=1, beta=1', 1.0, 1.0),
        ('alpha=1, beta=2', 1.0, 2.0),
        ('alpha=1, beta=5', 1.0, 5.0),
        ('alpha=2, beta=1', 2.0, 1.0),
        ('alpha=2, beta=5', 2.0, 5.0),
    ]
    results: Dict[str, RunStats] = {}

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Comparing α/β Combinations on {instance.name}")
        print(f"{'=' * 70}")
        print(f"{'Config':<20} {'Best':<12} {'Mean±Std':<22} {'Gap%':<10} {'Time(s)':<10}")

    for name, a, b in configs:
        if verbose:
            print(f"Testing {name}...")
        stats = run_experiment(
            instance=instance, num_runs=num_runs,
            alpha=a, beta=b, verbose=False,
        )
        results[name] = stats
        if verbose:
            print(f"  {name:<18} {stats.best_cost:<12.2f} "
                  f"{stats.mean_cost:<8.2f}±{stats.std_cost:<8.2f} "
                  f"{stats.gap_from_optimal:<10.2f} {stats.mean_time:<10.3f}")
    return results


def full_experiment(instance: TSPInstance, num_runs: int = 10,
                    output_dir: str = "results",
                    verbose: bool = True) -> Dict:
    """Comprehensive ACO experiment: AS vs ACS + α/β comparison + stability."""
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
        print("PHASE 1: AS vs ACS Comparison")
        print("=" * 70)
    as_stats = run_experiment(instance=instance, num_runs=num_runs // 2,
                              use_acs=False, verbose=verbose)
    acs_stats = run_experiment(instance=instance, num_runs=num_runs // 2,
                               use_acs=True, verbose=verbose)
    all_results['as_vs_acs'] = {
        'AS': as_stats.to_dict(),
        'ACS': acs_stats.to_dict(),
    }

    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 2: α/β Parameter Comparison")
        print("=" * 70)
    ab_results = compare_alpha_beta(instance, num_runs=5, verbose=verbose)
    all_results['alpha_beta_comparison'] = {
        name: stats.to_dict() for name, stats in ab_results.items()
    }

    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 3: Stability Analysis (AS, α=1, β=2)")
        print("=" * 70)
    stability_stats = run_experiment(
        instance=instance, num_runs=num_runs, verbose=verbose,
    )
    all_results['stability_analysis'] = stability_stats.to_dict()

    results_file = output_path / f"{instance.name}_experiment_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    if verbose:
        print(f"\nResults saved to {results_file}")
    return all_results
