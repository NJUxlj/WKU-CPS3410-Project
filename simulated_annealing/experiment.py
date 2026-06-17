"""
Experiment Runner for Simulated Annealing on TSP

Runs SA multiple times with different parameter configurations and
collects statistics for quality, convergence time, and stability analysis.
"""

import numpy as np
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from .tsp_instance import TSPInstance
from .simulated_annealing import SimulatedAnnealing, SAResult, SAParams


@dataclass
class ExperimentConfig:
    """Configuration for a batch of experiments."""
    instance: TSPInstance
    num_runs: int = 30
    initial_temp: float = 10000.0
    cooling_rates: List[float] = field(default_factory=lambda: [0.99, 0.995, 0.999])
    max_iterations_per_temp: int = 100
    neighborhood_types: List[str] = field(default_factory=lambda: ['2-opt'])
    base_seed: int = 42
    output_dir: str = "results"


@dataclass
class RunStats:
    """Statistics for a set of SA runs."""
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
    gap_from_optimal: float = 0.0  # Percentage gap from known optimal
    optimal_hit_rate: float = 0.0  # Proportion of runs hitting optimal

    def compute(self, optimal_value: Optional[float] = None):
        """Compute summary statistics from collected data."""
        if not self.costs:
            return
        self.best_cost = min(self.costs)
        self.worst_cost = max(self.costs)
        self.mean_cost = np.mean(self.costs)
        self.std_cost = np.std(self.costs)
        self.mean_time = np.mean(self.times)
        self.std_time = np.std(self.times)
        self.mean_iterations = np.mean(self.iterations)

        if optimal_value and optimal_value > 0:
            self.gap_from_optimal = ((self.best_cost - optimal_value) /
                                    optimal_value * 100)
            self.optimal_hit_rate = (sum(1 for c in self.costs
                                        if abs(c - optimal_value) < 0.01) /
                                    len(self.costs))

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
            'num_runs': len(self.costs)
        }


def run_experiment(instance: TSPInstance, num_runs: int = 30,
                   initial_temp: float = 10000.0,
                   cooling_rate: float = 0.995,
                   max_iterations_per_temp: int = 100,
                   neighborhood_type: str = '2-opt',
                   base_seed: int = 42,
                   verbose: bool = False) -> RunStats:
    """
    Run SA multiple times on a given instance and collect statistics.

    Args:
        instance: TSP problem instance
        num_runs: Number of independent SA runs
        initial_temp: Starting temperature
        cooling_rate: Cooling factor
        max_iterations_per_temp: Iterations per temperature level
        neighborhood_type: Type of neighborhood move
        base_seed: Starting random seed (incremented for each run)
        verbose: Print progress

    Returns:
        RunStats with aggregated statistics
    """
    stats = RunStats()

    for run_idx in range(num_runs):
        seed = base_seed + run_idx * 100

        sa = SimulatedAnnealing(
            distance_matrix=instance.distance_matrix,
            initial_temp=initial_temp,
            cooling_rate=cooling_rate,
            min_temp=0.01,
            max_iterations_per_temp=max_iterations_per_temp,
            cooling_schedule='geometric',
            neighborhood_type=neighborhood_type,
            restart_stuck=True,
            stuck_threshold=50,
            seed=seed
        )

        result = sa.run(verbose=False)

        stats.costs.append(result.best_cost)
        stats.times.append(result.elapsed_time)
        stats.iterations.append(result.iteration_count)

        if verbose and (run_idx + 1) % 10 == 0:
            print(f"  Run {run_idx + 1}/{num_runs}: "
                  f"Best={result.best_cost:.2f}, Time={result.elapsed_time:.3f}s")

    stats.compute(instance.optimal_value)

    if verbose:
        print(f"\n  Results: Best={stats.best_cost:.2f}, "
              f"Mean={stats.mean_cost:.2f}±{stats.std_cost:.2f}, "
              f"Gap={stats.gap_from_optimal:.2f}%, "
              f"Time={stats.mean_time:.3f}±{stats.std_time:.3f}s")

    return stats


def compare_cooling_rates(instance: TSPInstance, num_runs: int = 10,
                          cooling_rates: List[float] = None,
                          verbose: bool = True) -> Dict[str, RunStats]:
    """
    Compare different cooling rates on the same instance.

    Args:
        instance: TSP problem instance
        num_runs: Number of runs per cooling rate
        cooling_rates: List of cooling rates to test
        verbose: Print comparison table

    Returns:
        Dictionary mapping cooling rate (str) to RunStats
    """
    if cooling_rates is None:
        cooling_rates = [0.9, 0.95, 0.99, 0.995, 0.999, 0.9995]

    results = {}

    if verbose:
        print(f"\n{'='*70}")
        print(f"Comparing Cooling Rates on {instance.name} ({instance.num_cities} cities)")
        print(f"{'='*70}")
        print(f"{'Rate':<10} {'Best':<12} {'Mean±Std':<22} {'Gap%':<10} {'Time(s)':<10}")

    for rate in cooling_rates:
        if verbose:
            print(f"Testing cooling_rate={rate}...")

        stats = run_experiment(
            instance=instance,
            num_runs=num_runs,
            cooling_rate=rate,
            verbose=False
        )
        results[str(rate)] = stats

        if verbose:
            print(f"  {rate:<8} {stats.best_cost:<12.2f} "
                  f"{stats.mean_cost:<8.2f}±{stats.std_cost:<8.2f} "
                  f"{stats.gap_from_optimal:<10.2f} {stats.mean_time:<10.3f}")

    return results


def compare_neighborhoods(instance: TSPInstance, num_runs: int = 10,
                          verbose: bool = True) -> Dict[str, RunStats]:
    """
    Compare different neighborhood structures.

    Args:
        instance: TSP problem instance
        num_runs: Number of runs per neighborhood type
        verbose: Print comparison table

    Returns:
        Dictionary mapping neighborhood type to RunStats
    """
    neighborhoods = ['2-opt', 'swap', 'insert']

    results = {}

    if verbose:
        print(f"\n{'='*70}")
        print(f"Comparing Neighborhood Structures on {instance.name}")
        print(f"{'='*70}")
        print(f"{'Type':<10} {'Best':<12} {'Mean±Std':<22} {'Gap%':<10} {'Time(s)':<10}")

    for ntype in neighborhoods:
        if verbose:
            print(f"Testing neighborhood={ntype}...")

        stats = run_experiment(
            instance=instance,
            num_runs=num_runs,
            neighborhood_type=ntype,
            verbose=False
        )
        results[ntype] = stats

        if verbose:
            print(f"  {ntype:<8} {stats.best_cost:<12.2f} "
                  f"{stats.mean_cost:<8.2f}±{stats.std_cost:<8.2f} "
                  f"{stats.gap_from_optimal:<10.2f} {stats.mean_time:<10.3f}")

    return results


def full_experiment(instance: TSPInstance, num_runs: int = 30,
                    output_dir: str = "results",
                    verbose: bool = True) -> Dict:
    """
    Run a comprehensive experiment:
    1. Compare cooling rates
    2. Compare neighborhood structures
    3. Run full stability analysis with best parameters

    Returns:
        Dictionary with all experiment results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = {
        'instance_name': instance.name,
        'num_cities': instance.num_cities,
        'optimal_value': instance.optimal_value,
        'num_runs': num_runs,
    }

    # 1. Compare cooling rates
    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 1: Comparing Cooling Rates")
        print("=" * 70)

    cooling_results = compare_cooling_rates(instance, num_runs=10, verbose=verbose)
    all_results['cooling_comparison'] = {
        rate: stats.to_dict() for rate, stats in cooling_results.items()
    }

    # Find best cooling rate
    best_rate = min(cooling_results.items(),
                    key=lambda x: x[1].mean_cost)[0]
    if verbose:
        print(f"\nBest cooling rate: {best_rate}")

    # 2. Compare neighborhood structures
    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 2: Comparing Neighborhood Structures")
        print("=" * 70)

    neighborhood_results = compare_neighborhoods(instance, num_runs=10, verbose=verbose)
    all_results['neighborhood_comparison'] = {
        ntype: stats.to_dict() for ntype, stats in neighborhood_results.items()
    }

    best_neighborhood = min(neighborhood_results.items(),
                            key=lambda x: x[1].mean_cost)[0]
    if verbose:
        print(f"\nBest neighborhood type: {best_neighborhood}")

    # 3. Full stability analysis
    if verbose:
        print(f"\n{'='*70}")
        print(f"PHASE 3: Stability Analysis (30 runs with best params)")
        print(f"{'='*70}")

    stability_stats = run_experiment(
        instance=instance,
        num_runs=num_runs,
        cooling_rate=float(best_rate),
        neighborhood_type=best_neighborhood,
        verbose=verbose
    )
    all_results['stability_analysis'] = stability_stats.to_dict()

    # Save results
    results_file = output_path / f"{instance.name}_experiment_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    if verbose:
        print(f"\nResults saved to {results_file}")

    return all_results
