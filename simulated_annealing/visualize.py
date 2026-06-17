"""
Visualization Module for SA-TSP Experiments

Generates plots for:
- Convergence curves (cost vs. iteration)
- Temperature decay
- Acceptance rate over time
- Tour visualization (for small instances)
- Quality comparison box plots
- Stability analysis
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional
from .simulated_annealing import SAResult
from .tsp_instance import TSPInstance


# Set up matplotlib for English text rendering
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12


def plot_convergence(result: SAResult, title: str = "SA Convergence Curve",
                     save_path: Optional[str] = None,
                     show_optimal: Optional[float] = None):
    """
    Plot the convergence curve showing cost vs. iteration.

    Args:
        result: SAResult from a completed SA run
        title: Plot title
        save_path: Path to save the figure
        show_optimal: Optional horizontal line for known optimal value
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    iterations = list(range(len(result.cost_history)))
    ax.plot(iterations, result.cost_history, 'b-', linewidth=0.8, alpha=0.7,
            label='Current Cost')
    ax.set_xlabel('Iteration (× temperature step)')
    ax.set_ylabel('Tour Length (km)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if show_optimal is not None:
        ax.axhline(y=show_optimal, color='r', linestyle='--', linewidth=1.5,
                   label=f'Optimal ({show_optimal:.0f} km)')

    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.close()


def plot_temperature_decay(result: SAResult,
                           save_path: Optional[str] = None):
    """
    Plot the temperature decay over iterations.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    iterations = list(range(len(result.temp_history)))
    ax1.plot(iterations, result.temp_history, 'r-', linewidth=1.0,
             label='Temperature')
    ax1.set_xlabel('Iteration (× temperature step)')
    ax1.set_ylabel('Temperature', color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(range(len(result.cost_history)), result.cost_history,
             'b-', linewidth=0.8, alpha=0.5, label='Cost')
    ax2.set_ylabel('Tour Length (km)', color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    fig.suptitle('Temperature Decay & Cost Evolution')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.close()


def plot_acceptance_rate(acceptance_history: List[float],
                         save_path: Optional[str] = None):
    """
    Plot acceptance rate over temperature levels.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    levels = list(range(len(acceptance_history)))
    ax.bar(levels, acceptance_history, width=1.0, alpha=0.7, color='steelblue')
    ax.set_xlabel('Temperature Level')
    ax.set_ylabel('Acceptance Rate')
    ax.set_title('Acceptance Rate per Temperature Level')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.close()


def plot_tour(instance: TSPInstance, tour: List[int],
              title: str = "TSP Tour Visualization",
              save_path: Optional[str] = None):
    """
    Visualize the TSP tour on a 2D map (requires coordinates).

    Args:
        instance: TSPInstance with coordinates
        tour: List of city indices in visit order
        title: Plot title
        save_path: Path to save the figure
    """
    if instance.coordinates is None:
        print("  Warning: No coordinates available for tour visualization")
        return

    coords = instance.coordinates
    n = len(tour)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot cities
    for i, (x, y) in enumerate(coords):
        ax.scatter(x, y, c='red', s=100, zorder=5)
        ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold')

    # Plot tour edges
    for k in range(n):
        city_a = tour[k]
        city_b = tour[(k + 1) % n]
        x1, y1 = coords[city_a]
        x2, y2 = coords[city_b]
        ax.plot([x1, x2], [y1, y2], 'b-', linewidth=1.5, alpha=0.6)

    # Mark start city
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
    """
    Box plot comparing solution quality across different configurations.

    Args:
        all_costs: Dictionary mapping label → list of best costs from runs
        optimal_value: Optional optimal value to show as reference line
        title: Plot title
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = list(all_costs.keys())
    data = [all_costs[label] for label in labels]

    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    showmeans=True, meanline=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
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
    """
    Plot histogram of solution costs across multiple runs to show stability.

    Args:
        costs_list: List of best costs from multiple SA runs
        title: Plot title
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1 = axes[0]
    ax1.hist(costs_list, bins=min(20, len(costs_list) // 2),
             edgecolor='black', alpha=0.7, color='steelblue')
    ax1.set_xlabel('Tour Length (km)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Solution Costs')
    mean_cost = np.mean(costs_list)
    ax1.axvline(x=mean_cost, color='r', linestyle='--', linewidth=1.5,
                label=f'Mean ({mean_cost:.0f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Run-by-run plot
    ax2 = axes[1]
    runs = list(range(1, len(costs_list) + 1))
    ax2.plot(runs, costs_list, 'bo-', markersize=4, linewidth=0.8, alpha=0.7)
    ax2.axhline(y=mean_cost, color='r', linestyle='--', linewidth=1.0,
                alpha=0.5)
    ax2.set_xlabel('Run Number')
    ax2.set_ylabel('Tour Length (km)')
    ax2.set_title('Cost per Run')
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title)
    std_cost = np.std(costs_list)
    print(f"  Stability: Mean={mean_cost:.2f}, Std={std_cost:.2f}, "
          f"CV={std_cost/mean_cost*100:.2f}%")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.close()


def generate_all_plots(instance: TSPInstance, result: SAResult,
                       output_dir: str = "results",
                       prefix: str = "sa"):
    """
    Generate all standard plots for an SA run.

    Args:
        instance: TSP instance
        result: SA result
        output_dir: Directory to save plots
        prefix: Filename prefix
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Convergence curve
    plot_convergence(
        result,
        title=f"SA Convergence on {instance.name} ({instance.num_cities} cities)",
        save_path=str(output_path / f"{prefix}_convergence.png"),
        show_optimal=instance.optimal_value
    )

    # Temperature decay
    plot_temperature_decay(
        result,
        save_path=str(output_path / f"{prefix}_temperature_decay.png")
    )

    # Acceptance rate
    plot_acceptance_rate(
        result.acceptance_history,
        save_path=str(output_path / f"{prefix}_acceptance_rate.png")
    )

    # Tour visualization (only if coordinates available)
    if instance.coordinates is not None:
        plot_tour(
            instance, result.best_tour,
            title=f"Best Tour Found by SA - {instance.name}",
            save_path=str(output_path / f"{prefix}_tour.png")
        )
