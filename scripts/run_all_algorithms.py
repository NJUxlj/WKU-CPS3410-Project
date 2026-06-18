#!/usr/bin/env python3
"""
Run All Meta-heuristic Algorithms on TSP

Unified comparison of the 7 meta-heuristic algorithms on the TSP problem,
producing the 4 PPT-aligned evaluation dimensions:

    1. Quality         -> Best / Mean / Std / Gap% / CV%  (lower is better)
    2. Diversification -> # unique tours explored (search-space coverage)
    3. Intensification -> Iterations-to-target & cost-curve steepness
    4. Time Complexity -> Wall-clock seconds (single & per-iter)

For each TSP instance (10-city + 50-city), every algorithm is run `NUM_RUNS`
times with independent seeds; aggregate statistics are reported.

Usage:
    python scripts/run_all_algorithms.py                # full run
    python scripts/run_all_algorithms.py --quick       # smoke test (3 runs)
    python scripts/run_all_algorithms.py --skip-plots   # text output only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

# Matplotlib non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulated_annealing.tsp_instance import TSPInstance
from simulated_annealing.simulated_annealing import SimulatedAnnealing
from local_search import LocalSearch
from genetic_algorithm import GeneticAlgorithm
from vns import VNS, RVNS
from tabu_search import TabuSearch
from ant_colony import AntColony
from bee_algorithm import ArtificialBeeColony


# =========================================================
# Algorithm factories
# =========================================================
# Each factory returns a fresh, *unrun* algorithm instance bound to a given
# `distance_matrix` and `seed`. Default parameters are tuned for ~1 second
# per run on the 50-city instance; they match the settings described in
# the GA / TS / VNS / SA / ACO / ABC reports.

def make_sa(dm: np.ndarray, seed: int) -> SimulatedAnnealing:
    return SimulatedAnnealing(
        distance_matrix=dm,
        initial_temp=10000.0, cooling_rate=0.995, min_temp=0.01,
        max_iterations_per_temp=100, neighborhood_type="2-opt",
        restart_stuck=True, stuck_threshold=100, seed=seed,
    )


def make_ls(dm: np.ndarray, seed: int) -> LocalSearch:
    return LocalSearch(
        distance_matrix=dm,
        strategy="best", neighborhood="2-opt",
        max_iterations=5000, max_no_improve=500, seed=seed,
    )


# Number of random restarts for Local Search (matches LS report)
LS_NUM_RESTARTS = 20


def make_ga(dm: np.ndarray, seed: int) -> GeneticAlgorithm:
    return GeneticAlgorithm(
        distance_matrix=dm,
        population_size=100, num_generations=300, elite_count=20,
        crossover_rate=0.9, mutation_rate=0.2,
        selection_method="tournament", crossover_method="ox",
        mutation_method="swap", tournament_size=5,
        convergence_patience=100, seed=seed,
    )


def make_vns(dm: np.ndarray, seed: int) -> VNS:
    return VNS(
        distance_matrix=dm,
        k_max=5, max_iterations=100, max_no_improve=30,
        local_search_iterations=2000, seed=seed,
    )


def make_rvns(dm: np.ndarray, seed: int) -> RVNS:
    return RVNS(
        distance_matrix=dm,
        k_max=5, max_iterations=300, max_no_improve=100, seed=seed,
    )


def make_ts(dm: np.ndarray, seed: int) -> TabuSearch:
    return TabuSearch(
        distance_matrix=dm,
        tabu_tenure=15, max_iterations=2000, max_no_improve=200,
        region_iterations=50, seed=seed,
    )


def make_aco(dm: np.ndarray, seed: int) -> AntColony:
    return AntColony(
        distance_matrix=dm,
        num_ants=40, num_iterations=200,
        alpha=1.0, beta=2.0, rho=0.5, q0=0.9,
        use_acs=True, use_best_only=True,
        max_no_improve=100, seed=seed,
    )


def make_abc(dm: np.ndarray, seed: int) -> ArtificialBeeColony:
    return ArtificialBeeColony(
        distance_matrix=dm,
        colony_size=50, max_iterations=300, limit=100,
        neighborhood_type="2-opt", max_no_improve=100, seed=seed,
    )


ALGORITHMS: Dict[str, Callable[[np.ndarray, int], object]] = {
    "LS":  make_ls,
    "SA":  make_sa,
    "GA":  make_ga,
    "VNS": make_vns,
    "RVNS": make_rvns,
    "TS":  make_ts,
    "ACO": make_aco,
    "ABC": make_abc,
}

ALGORITHM_LABELS = {
    "LS":  "Local Search",
    "SA":  "Simulated Annealing",
    "GA":  "Genetic Algorithm",
    "VNS": "VNS",
    "RVNS": "Reduced VNS",
    "TS":  "Tabu Search",
    "ACO": "Ant Colony (ACS)",
    "ABC": "Artificial Bee Colony",
}

# Colors for plots (one per algorithm, used in all 4 dimensions)
ALGORITHM_COLORS = {
    "LS":   "#1f77b4",
    "SA":   "#ff7f0e",
    "GA":   "#2ca02c",
    "VNS":  "#d62728",
    "RVNS": "#9467bd",
    "TS":   "#8c564b",
    "ACO":  "#e377c2",
    "ABC":  "#17becf",
}


# =========================================================
# Per-algorithm normalization
# =========================================================
def get_cost_history(result) -> List[float]:
    """Return best-so-far cost history for plotting convergence.
    GA stores it as `best_cost_history`; others as `cost_history`.
    """
    if hasattr(result, "best_cost_history"):
        return list(result.best_cost_history)
    return list(result.cost_history)


def get_unique_tours(result) -> int:
    """Return # unique tours explored during the run.
    Falls back to 0 for algorithms that don't track it.
    """
    if hasattr(result, "unique_tours_explored"):
        return int(result.unique_tours_explored)
    return 0


# =========================================================
# Aggregate statistics
# =========================================================
@dataclass
class AlgoStats:
    name: str
    label: str
    runs: int
    best_cost: float          # best over all runs
    mean_cost: float          # mean of best-per-run
    std_cost: float           # std of best-per-run
    cv_cost: float            # std/mean (%)
    gap_pct: float            # (best_cost - optimal) / optimal * 100
    mean_time: float          # mean wall-clock seconds
    std_time: float
    mean_unique_tours: float
    iterations_to_95pct: float  # mean iterations to reach 0.95 * best
    # For convergence plots
    mean_convergence_curve: List[float] = field(default_factory=list)
    # Raw per-run data (for the JSON dump)
    raw_costs: List[float] = field(default_factory=list)
    raw_times: List[float] = field(default_factory=list)
    raw_iters: List[int] = field(default_factory=list)


def run_algorithm(
    name: str,
    factory: Callable,
    instance: TSPInstance,
    n_runs: int,
    base_seed: int = 42,
) -> AlgoStats:
    """Run `factory(distance_matrix, seed)` n_runs times and aggregate stats."""
    costs: List[float] = []
    times: List[float] = []
    iters: List[int] = []
    uniques: List[int] = []
    iters_to_target: List[int] = []

    # We accumulate convergence curves by re-sampling on a fixed x-grid
    # of 200 points (linear interpolation) so algorithms with different
    # iteration counts can be averaged and plotted on one axis.
    GRID = 200
    grid_curves: List[List[float]] = []

    for i in range(n_runs):
        seed = base_seed + i * 1000 + 7
        algo = factory(instance.distance_matrix, seed)
        t0 = time.perf_counter()
        # Local Search exposes restarts via run(num_restarts=...)
        if name == "LS":
            result = algo.run(num_restarts=LS_NUM_RESTARTS, verbose=False)
        else:
            result = algo.run(verbose=False)
        elapsed = time.perf_counter() - t0

        costs.append(float(result.best_cost))
        times.append(elapsed)
        uniques.append(get_unique_tours(result))
        iters.append(int(getattr(result, "iterations_run", 0)) or
                     int(getattr(result, "iteration_count", 0)) or
                     int(getattr(result, "generations_run", 0)) or 0)

        # Cost history for convergence
        ch = np.array(get_cost_history(result), dtype=float)
        if ch.size > 0:
            # Down-sample to GRID points
            if ch.size >= GRID:
                idx = np.linspace(0, ch.size - 1, GRID).astype(int)
                sampled = ch[idx]
            else:
                # Pad to GRID with last value
                sampled = np.concatenate([ch, np.full(GRID - ch.size, ch[-1])])
            grid_curves.append(sampled.tolist())

            # Iterations to reach 95% of best cost (intensification)
            target = result.best_cost * 1.05
            # find first idx where cost <= target
            below = np.where(ch <= target)[0]
            iters_to_target.append(int(below[0]) if below.size > 0 else ch.size)
        else:
            iters_to_target.append(0)

    best = float(min(costs))
    mean = float(np.mean(costs))
    std = float(np.std(costs))
    cv = std / mean * 100 if mean > 0 else 0.0
    gap = (best - instance.optimal_value) / instance.optimal_value * 100
    mean_curve = (np.mean(np.array(grid_curves), axis=0).tolist()
                  if grid_curves else [])

    return AlgoStats(
        name=name,
        label=ALGORITHM_LABELS[name],
        runs=n_runs,
        best_cost=best,
        mean_cost=mean,
        std_cost=std,
        cv_cost=cv,
        gap_pct=gap,
        mean_time=float(np.mean(times)),
        std_time=float(np.std(times)),
        mean_unique_tours=float(np.mean(uniques)),
        iterations_to_95pct=float(np.mean(iters_to_target)),
        mean_convergence_curve=mean_curve,
        raw_costs=costs,
        raw_times=times,
        raw_iters=iters,
    )


# =========================================================
# Reporting
# =========================================================
def print_comparison_table(
    instance: TSPInstance,
    all_stats: Dict[str, AlgoStats],
) -> None:
    print()
    print("=" * 110)
    print(f"INSTANCE: {instance.name}  (N={instance.num_cities}, "
          f"optimal = {instance.optimal_value:.2f})")
    print("=" * 110)
    print(f"{'Algorithm':<22}{'Best':>10}{'Mean':>10}{'Std':>8}{'CV%':>7}"
          f"{'Gap%':>8}{'Time(s)':>10}{'UniqueTours':>13}{'It→95%':>10}")
    print("-" * 110)
    # Order: LS, SA, GA, VNS, RVNS, TS, ACO, ABC
    order = ["LS", "SA", "GA", "VNS", "RVNS", "TS", "ACO", "ABC"]
    for name in order:
        s = all_stats[name]
        print(f"{s.label:<22}"
              f"{s.best_cost:>10.2f}{s.mean_cost:>10.2f}{s.std_cost:>8.2f}"
              f"{s.cv_cost:>7.2f}{s.gap_pct:>8.2f}"
              f"{s.mean_time:>10.3f}{s.mean_unique_tours:>13.0f}"
              f"{s.iterations_to_95pct:>10.0f}")
    print("=" * 110)


def print_ppt4d_summary(
    instance: TSPInstance,
    all_stats: Dict[str, AlgoStats],
) -> None:
    print()
    print(f"  [PPT 4-dimension summary on {instance.name}]")
    print(f"  1) Quality       (best  ): " + ", ".join(
        f"{n}={all_stats[n].best_cost:.0f}"
        for n in ["LS", "SA", "GA", "VNS", "RVNS", "TS", "ACO", "ABC"]))
    print(f"  2) Diversification (mean#): " + ", ".join(
        f"{n}={all_stats[n].mean_unique_tours:.0f}"
        for n in ["LS", "SA", "GA", "VNS", "RVNS", "TS", "ACO", "ABC"]))
    print(f"  3) Intensification  (iters): " + ", ".join(
        f"{n}={all_stats[n].iterations_to_95pct:.0f}"
        for n in ["LS", "SA", "GA", "VNS", "RVNS", "TS", "ACO", "ABC"]))
    print(f"  4) Time Complexity (sec):   " + ", ".join(
        f"{n}={all_stats[n].mean_time:.3f}"
        for n in ["LS", "SA", "GA", "VNS", "RVNS", "TS", "ACO", "ABC"]))


# =========================================================
# Plotting
# =========================================================
def plot_quality_bar(
    instance: TSPInstance,
    all_stats: Dict[str, AlgoStats],
    out_path: Path,
) -> None:
    """Bar chart of best-cost (gap%) per algorithm."""
    order = ["LS", "SA", "GA", "VNS", "RVNS", "TS", "ACO", "ABC"]
    gaps = [all_stats[n].gap_pct for n in order]
    colors = [ALGORITHM_COLORS[n] for n in order]
    labels = [all_stats[n].label for n in order]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(range(len(order)), gaps, color=colors, edgecolor="black")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Best Gap to Optimal (%)")
    ax.set_title(f"Quality — Best Cost Gap to Optimal "
                 f"({instance.name}, N={instance.num_cities})")
    for bar, gap in zip(bars, gaps):
        height = bar.get_height()
        offset = 0.3 if height >= 0 else -0.7
        ax.text(bar.get_x() + bar.get_width() / 2, height + offset,
                f"{gap:+.2f}%", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_convergence(
    instance: TSPInstance,
    all_stats: Dict[str, AlgoStats],
    out_path: Path,
) -> None:
    """Average best-so-far convergence curves for all algorithms."""
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, s in all_stats.items():
        if s.mean_convergence_curve:
            ax.plot(s.mean_convergence_curve,
                    label=s.label, color=ALGORITHM_COLORS[name],
                    linewidth=1.6)
    ax.set_xlabel("Iteration (normalized 0–200)")
    ax.set_ylabel("Best-so-far cost")
    ax.set_title(f"Convergence — {instance.name} "
                 f"(N={instance.num_cities}, optimal="
                 f"{instance.optimal_value:.0f})")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_diversification(
    instance: TSPInstance,
    all_stats: Dict[str, AlgoStats],
    out_path: Path,
) -> None:
    """Bar chart of mean # unique tours explored."""
    order = ["LS", "SA", "GA", "VNS", "RVNS", "TS", "ACO", "ABC"]
    vals = [all_stats[n].mean_unique_tours for n in order]
    colors = [ALGORITHM_COLORS[n] for n in order]
    labels = [all_stats[n].label for n in order]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(range(len(order)), vals, color=colors, edgecolor="black")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("# Unique tours explored (mean)")
    ax.set_title(f"Diversification — Search-space Coverage "
                 f"({instance.name}, N={instance.num_cities})")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v,
                f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_intensification(
    instance: TSPInstance,
    all_stats: Dict[str, AlgoStats],
    out_path: Path,
) -> None:
    """Bar chart of iterations-to-95%-of-best (intensification)."""
    order = ["LS", "SA", "GA", "VNS", "RVNS", "TS", "ACO", "ABC"]
    vals = [all_stats[n].iterations_to_95pct for n in order]
    colors = [ALGORITHM_COLORS[n] for n in order]
    labels = [all_stats[n].label for n in order]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(range(len(order)), vals, color=colors, edgecolor="black")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Iterations to reach 95% of best (mean)")
    ax.set_title(f"Intensification — Convergence Speed "
                 f"({instance.name}, N={instance.num_cities})")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v,
                f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_time_complexity(
    instance: TSPInstance,
    all_stats: Dict[str, AlgoStats],
    out_path: Path,
) -> None:
    """Bar chart of mean wall-clock time with error bars."""
    order = ["LS", "SA", "GA", "VNS", "RVNS", "TS", "ACO", "ABC"]
    means = [all_stats[n].mean_time for n in order]
    stds = [all_stats[n].std_time for n in order]
    colors = [ALGORITHM_COLORS[n] for n in order]
    labels = [all_stats[n].label for n in order]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(range(len(order)), means, yerr=stds,
                  color=colors, edgecolor="black", capsize=4)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Wall-clock time per run (s, mean ± std)")
    ax.set_title(f"Time Complexity — Single-Run Runtime "
                 f"({instance.name}, N={instance.num_cities})")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m,
                f"{m:.3f}s", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# =========================================================
# Main
# =========================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run all 7 meta-heuristics on TSP and produce the "
                    "PPT 4-dimension comparison.")
    parser.add_argument("--quick", action="store_true",
                        help="Run a quick smoke test (3 runs each)")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Skip matplotlib figure generation")
    parser.add_argument("--num-runs", type=int, default=None,
                        help="Override the default number of runs per algorithm")
    parser.add_argument("--out-dir", type=str,
                        default=str(ROOT / "scripts" / "outputs"),
                        help="Directory to write JSON/figures into")
    args = parser.parse_args()

    n_runs = args.num_runs or (3 if args.quick else 20)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Run All Meta-heuristic Algorithms on TSP")
    print(f"  n_runs per algorithm  : {n_runs}")
    print(f"  output directory      : {out_dir}")
    print(f"  plot generation       : {'off' if args.skip_plots else 'on'}")
    print("=" * 80)

    # --- Load TSP instances --------------------------------------------
    data_dir = ROOT / "simulated_annealing" / "data"
    instances = [
        TSPInstance.load(str(data_dir / "distances_between_cities_10.txt"),
                         name="cities_10", optimal_value=3473.0),
        TSPInstance.load(str(data_dir / "distances_between_cities_50.txt"),
                         name="cities_50", optimal_value=5644.0),
    ]

    # --- Run all algorithms on all instances --------------------------
    all_results: Dict[str, Dict[str, AlgoStats]] = {}
    t_start = time.perf_counter()
    for inst in instances:
        print(f"\n[+] Running on {inst.name} "
              f"(N={inst.num_cities}, optimal={inst.optimal_value:.0f}) ...")
        stats: Dict[str, AlgoStats] = {}
        for name, factory in ALGORITHMS.items():
            t0 = time.perf_counter()
            stats[name] = run_algorithm(name, factory, inst, n_runs=n_runs)
            dt = time.perf_counter() - t0
            print(f"    - {ALGORITHM_LABELS[name]:<22} "
                  f"best={stats[name].best_cost:>10.2f} "
                  f"gap={stats[name].gap_pct:+6.2f}%  "
                  f"cv={stats[name].cv_cost:5.2f}%  "
                  f"({dt:.1f}s total / {n_runs} runs)")
        all_results[inst.name] = stats
        print_comparison_table(inst, stats)
        print_ppt4d_summary(inst, stats)
    total_t = time.perf_counter() - t_start
    print(f"\n[Total wall-clock: {total_t:.1f}s]")

    # --- Persist JSON --------------------------------------------------
    json_path = out_dir / "run_all_algorithms_results.json"

    def _to_dict(s: AlgoStats) -> dict:
        d = asdict(s)
        return d

    payload = {
        "config": {
            "n_runs": n_runs,
            "wall_clock_seconds": total_t,
        },
        "instances": {
            inst_name: {
                s.name: _to_dict(s) for s in all_results[inst_name].values()
            }
            for inst_name in all_results
        },
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"\n[+] Wrote JSON results -> {json_path}")

    # --- Plots --------------------------------------------------------
    if not args.skip_plots:
        print("\n[+] Generating comparison plots ...")
        for inst in instances:
            prefix = f"{inst.name}_N{inst.num_cities}"
            plot_quality_bar(inst, all_results[inst.name],
                             out_dir / f"{prefix}_01_quality.png")
            plot_convergence(inst, all_results[inst.name],
                             out_dir / f"{prefix}_02_convergence.png")
            plot_diversification(inst, all_results[inst.name],
                                 out_dir / f"{prefix}_03_diversification.png")
            plot_intensification(inst, all_results[inst.name],
                                 out_dir / f"{prefix}_04_intensification.png")
            plot_time_complexity(inst, all_results[inst.name],
                                 out_dir / f"{prefix}_05_time.png")
            print(f"    - {inst.name}: 5 plots written")
        print(f"[+] All plots -> {out_dir}")

    print("\n[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
