"""
Experiment Runner for VNS / RVNS on TSP
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from simulated_annealing.tsp_instance import TSPInstance
from .vns import VNS, RVNS


@dataclass
class RunStats:
    """Statistics for a set of VNS / RVNS runs."""
    costs: List[float] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    iterations: List[int] = field(default_factory=list)
    best_cost: float = float('inf')
    worst_cost: float = 0.0
    mean_cost: float = 0.0
    std_cost: float = 0.0
    mean_time: float = 0.0
    std_time: float = 0.0
    mean_unique_tours: float = 0.0
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
        self.mean_unique_tours = float(self.mean_unique_tours)
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
            'mean_unique_tours': self.mean_unique_tours,
            'gap_from_optimal_pct': self.gap_from_optimal,
            'optimal_hit_rate': self.optimal_hit_rate,
            'num_runs': len(self.costs),
        }


def run_vns_experiment(instance: TSPInstance, num_runs: int = 20,
                       k_max: int = 5, max_iterations: int = 100,
                       max_no_improve: int = 30,
                       local_search_iterations: int = 2000,
                       local_search_no_improve: int = 500,
                       base_seed: int = 42,
                       verbose: bool = False) -> RunStats:
    """Run VNS multiple times."""
    stats = RunStats()
    for run_idx in range(num_runs):
        seed = base_seed + run_idx * 100
        vns = VNS(
            distance_matrix=instance.distance_matrix,
            k_max=k_max,
            max_iterations=max_iterations,
            max_no_improve=max_no_improve,
            local_search_iterations=local_search_iterations,
            local_search_no_improve=local_search_no_improve,
            seed=seed,
        )
        result = vns.run(verbose=False)
        stats.costs.append(result.best_cost)
        stats.times.append(result.elapsed_time)
        stats.iterations.append(len(result.cost_history))
        stats.mean_unique_tours = (
            (stats.mean_unique_tours * (len(stats.costs) - 1)
             + result.unique_tours_explored)
            / len(stats.costs)
        )
        if verbose and (run_idx + 1) % max(1, num_runs // 5) == 0:
            print(f"  VNS Run {run_idx + 1}/{num_runs}: "
                  f"Best={result.best_cost:.2f}, Time={result.elapsed_time:.3f}s")
    stats.compute(instance.optimal_value)
    if verbose:
        print(f"\n  VNS Results: Best={stats.best_cost:.2f}, "
              f"Mean={stats.mean_cost:.2f}±{stats.std_cost:.2f}, "
              f"Gap={stats.gap_from_optimal:.2f}%, "
              f"Time={stats.mean_time:.3f}±{stats.std_time:.3f}s")
    return stats


def run_rvns_experiment(instance: TSPInstance, num_runs: int = 20,
                        k_max: int = 5, max_iterations: int = 500,
                        max_no_improve: int = 100,
                        base_seed: int = 42,
                        verbose: bool = False) -> RunStats:
    """Run RVNS multiple times."""
    stats = RunStats()
    for run_idx in range(num_runs):
        seed = base_seed + run_idx * 100
        rvns = RVNS(
            distance_matrix=instance.distance_matrix,
            k_max=k_max,
            max_iterations=max_iterations,
            max_no_improve=max_no_improve,
            seed=seed,
        )
        result = rvns.run(verbose=False)
        stats.costs.append(result.best_cost)
        stats.times.append(result.elapsed_time)
        stats.iterations.append(len(result.cost_history))
        stats.mean_unique_tours = (
            (stats.mean_unique_tours * (len(stats.costs) - 1)
             + result.unique_tours_explored)
            / len(stats.costs)
        )
        if verbose and (run_idx + 1) % max(1, num_runs // 5) == 0:
            print(f"  RVNS Run {run_idx + 1}/{num_runs}: "
                  f"Best={result.best_cost:.2f}, Time={result.elapsed_time:.3f}s")
    stats.compute(instance.optimal_value)
    if verbose:
        print(f"\n  RVNS Results: Best={stats.best_cost:.2f}, "
              f"Mean={stats.mean_cost:.2f}±{stats.std_cost:.2f}, "
              f"Gap={stats.gap_from_optimal:.2f}%, "
              f"Time={stats.mean_time:.3f}±{stats.std_time:.3f}s")
    return stats


def compare_vns_rvns(instance: TSPInstance, num_runs: int = 20,
                     verbose: bool = True) -> Dict[str, RunStats]:
    """Compare VNS vs RVNS on the same instance."""
    results: Dict[str, RunStats] = {}

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"VNS vs RVNS Comparison on {instance.name}")
        print(f"{'=' * 70}")

    if verbose:
        print("\n--- Basic VNS (with Local Search) ---")
    results['VNS'] = run_vns_experiment(
        instance, num_runs=num_runs, verbose=verbose,
    )

    if verbose:
        print("\n--- Reduced VNS (no Local Search) ---")
    results['RVNS'] = run_rvns_experiment(
        instance, num_runs=num_runs, verbose=verbose,
    )

    if verbose:
        print(f"\n{'Method':<10} {'Best':<12} {'Mean±Std':<22} {'Gap%':<10} "
              f"{'Time(s)':<10} {'Unique':<10}")
        for name, stats in results.items():
            print(f"  {name:<8} {stats.best_cost:<12.2f} "
                  f"{stats.mean_cost:<8.2f}±{stats.std_cost:<8.2f} "
                  f"{stats.gap_from_optimal:<10.2f} "
                  f"{stats.mean_time:<10.3f} {stats.mean_unique_tours:<10.0f}")
    return results


def full_experiment(instance: TSPInstance, num_runs: int = 20,
                    output_dir: str = "results",
                    verbose: bool = True) -> Dict:
    """Comprehensive VNS / RVNS experiment."""
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
        print("PHASE 1: VNS vs RVNS Head-to-Head")
        print("=" * 70)
    compare_results = compare_vns_rvns(instance, num_runs=num_runs, verbose=verbose)
    all_results['vns_vs_rvns'] = {
        name: stats.to_dict() for name, stats in compare_results.items()
    }

    results_file = output_path / f"{instance.name}_experiment_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    if verbose:
        print(f"\nResults saved to {results_file}")
    return all_results
