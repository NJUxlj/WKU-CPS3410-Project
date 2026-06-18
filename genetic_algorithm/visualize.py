"""
Visualization Module for Genetic Algorithm TSP Experiments
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional

from simulated_annealing.tsp_instance import TSPInstance
from .genetic_algorithm import GAResult


plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12


def plot_convergence(result: GAResult, title: str = "GA Convergence Curve",
                     save_path: Optional[str] = None,
                     show_optimal: Optional[float] = None):
    """Plot best and mean cost per generation."""
    fig, ax = plt.subplots(figsize=(10, 6))
    generations = list(range(len(result.best_cost_history)))
    ax.plot(generations, result.best_cost_history, 'b-', linewidth=1.5,
            alpha=0.8, label='Best cost')
    ax.plot(generations, result.mean_cost_history, 'r-', linewidth=1.0,
            alpha=0.5, label='Mean population cost')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Tour Length (km)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if show_optimal is not None:
        ax.axhline(y=show_optimal, color='g', linestyle='--', linewidth=1.5,
                   label=f'Optimal ({show_optimal:.0f} km)')
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_diversity(result: GAResult, title: str = "Population Diversity",
                   save_path: Optional[str] = None):
    """Plot population diversity over generations (proxy for diversification)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    generations = list(range(len(result.diversity_history)))
    ax.plot(generations, result.diversity_history, 'g-', linewidth=1.5,
            alpha=0.8)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Diversity (unique individuals ratio)')
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_tour(instance: TSPInstance, tour: List[int],
              title: str = "TSP Tour Visualization",
              save_path: Optional[str] = None):
    """Visualize the TSP tour on a 2D map (requires coordinates)."""
    if instance.coordinates is None:
        print("  Warning: No coordinates available for tour visualization")
        return

    coords = instance.coordinates
    n = len(tour)
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, (x, y) in enumerate(coords):
        ax.scatter(x, y, c='red', s=100, zorder=5)
        ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold')

    for k in range(n):
        a, b = tour[k], tour[(k + 1) % n]
        x1, y1 = coords[a]
        x2, y2 = coords[b]
        ax.plot([x1, x2], [y1, y2], 'b-', linewidth=1.5, alpha=0.6)

    start_x, start_y = coords[tour[0]]
    ax.scatter(start_x, start_y, c='green', s=200, marker='*', zorder=6,
               label='Start')
    cost = instance.cost(tour)
    ax.set_xlabel('X Coordinate (km)')
    ax.set_ylabel('Y Coordinate (km)')
    ax.set_title(f"{title}\nTotal Distance: {cost:.0f} km")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_quality_comparison(all_costs: Dict[str, List[float]],
                            optimal_value: Optional[float] = None,
                            title: str = "Solution Quality Comparison",
                            save_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = list(all_costs.keys())
    data = [all_costs[label] for label in labels]
    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    showmeans=True, meanline=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightgreen')
        patch.set_alpha(0.7)
    if optimal_value is not None:
        ax.axhline(y=optimal_value, color='r', linestyle='--', linewidth=1.5,
                   label=f'Optimal ({optimal_value:.0f})')
        ax.legend()
    ax.set_ylabel('Tour Length (km)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_stability_analysis(costs_list: List[float],
                            title: str = "Stability Analysis",
                            save_path: Optional[str] = None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1 = axes[0]
    ax1.hist(costs_list, bins=min(20, len(costs_list) // 2),
             edgecolor='black', alpha=0.7, color='seagreen')
    ax1.set_xlabel('Tour Length (km)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Solution Costs')
    mean_cost = float(np.mean(costs_list))
    ax1.axvline(x=mean_cost, color='r', linestyle='--', linewidth=1.5,
                label=f'Mean ({mean_cost:.0f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    ax2 = axes[1]
    runs = list(range(1, len(costs_list) + 1))
    ax2.plot(runs, costs_list, 'go-', markersize=4, linewidth=0.8, alpha=0.7)
    ax2.axhline(y=mean_cost, color='r', linestyle='--', linewidth=1.0, alpha=0.5)
    ax2.set_xlabel('Run Number')
    ax2.set_ylabel('Tour Length (km)')
    ax2.set_title('Cost per Run')
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title)
    std_cost = float(np.std(costs_list))
    print(f"  Stability: Mean={mean_cost:.2f}, Std={std_cost:.2f}, "
          f"CV={std_cost / mean_cost * 100:.2f}%")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def generate_all_plots(instance: TSPInstance, result: GAResult,
                       output_dir: str = "results",
                       prefix: str = "ga"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_convergence(
        result,
        title=f"GA Convergence on {instance.name} ({instance.num_cities} cities)",
        save_path=str(output_path / f"{prefix}_convergence.png"),
        show_optimal=instance.optimal_value,
    )
    plot_diversity(
        result,
        title=f"GA Population Diversity on {instance.name}",
        save_path=str(output_path / f"{prefix}_diversity.png"),
    )
    if instance.coordinates is not None:
        plot_tour(
            instance, result.best_tour,
            title=f"Best Tour Found by GA - {instance.name}",
            save_path=str(output_path / f"{prefix}_tour.png"),
        )
