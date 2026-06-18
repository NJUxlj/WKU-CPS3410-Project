"""
Local Search Algorithm for the Traveling Salesman Problem (TSP)

Implements hill climbing with multiple neighborhood structures:
- 1-opt (random swap):  swap two cities in the tour
- 2-opt:                reverse a segment of the tour
- or-opt:               move a single city to another position
- 3-opt:                reverse two segments simultaneously

Supports two improvement strategies:
- first-improvement: accept the first move that improves the cost
- best-improvement:  scan the entire neighborhood, take the best move

Multi-restart is essential because Local Search is very prone to getting
stuck in local optima (which is why we need the metaheuristics above it).
"""

import numpy as np
import random
import time
import math
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class LSParams:
    """Configuration parameters for Local Search."""
    strategy: str = 'best'        # 'best' or 'first' improvement
    neighborhood: str = '2-opt'   # '1-opt' / 'swap', '2-opt', 'or-opt', '3-opt'
    max_iterations: int = 10000   # Hard cap on total neighbor evaluations
    max_no_improve: int = 1000    # Stop if no improvement for this many iters
    num_restarts: int = 30        # Number of independent random restarts
    seed: Optional[int] = None


@dataclass
class LSResult:
    """Stores the results of a single Local Search run (all restarts aggregated)."""
    best_tour: List[int]
    best_cost: float
    cost_history: List[float]              # Best cost per restart
    final_costs_per_restart: List[float]   # Cost at the end of each restart
    iterations_per_restart: List[int]
    elapsed_time: float
    converged: bool
    # Tracking for diversification / intensification analysis
    unique_tours_explored: int = 0
    final_tours_per_restart: List[List[int]] = field(default_factory=list)


# =====================================================================
# Neighborhood move operators (pure functions working on a tour list)
# =====================================================================

def _swap_move(tour: List[int], i: int, j: int) -> List[int]:
    """Swap two positions in the tour (a.k.a. 1-opt)."""
    new_tour = tour.copy()
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour


def _two_opt_move(tour: List[int], i: int, j: int) -> List[int]:
    """Reverse the segment between i and j (inclusive)."""
    new_tour = tour.copy()
    if i > j:
        i, j = j, i
    new_tour[i:j + 1] = list(reversed(new_tour[i:j + 1]))
    return new_tour


def _or_opt_move(tour: List[int], i: int, j: int) -> List[int]:
    """Move a single city at position i to position j+1."""
    new_tour = tour.copy()
    city = new_tour.pop(i)
    target = j + 1
    if i < target:
        target -= 1
    new_tour.insert(target, city)
    return new_tour


def _three_opt_move(tour: List[int], i: int, j: int, k: int) -> List[int]:
    """
    A simplified 3-opt move: pick three break points and reverse one segment.
    True 3-opt enumerates 7 reconnections; here we sample a few patterns to
    keep the per-iteration cost low.
    """
    new_tour = tour.copy()
    a, b, c = sorted([i, j, k])
    pattern = random.randint(0, 3)
    if pattern == 0:
        new_tour[b:c + 1] = list(reversed(new_tour[b:c + 1]))
    elif pattern == 1:
        new_tour[a:b + 1] = list(reversed(new_tour[a:b + 1]))
    elif pattern == 2:
        new_tour[a:b + 1] = list(reversed(new_tour[a:b + 1]))
        new_tour[b:c + 1] = list(reversed(new_tour[b:c + 1]))
    return new_tour


# =====================================================================
# Local Search core class
# =====================================================================

class LocalSearch:
    """
    Hill climbing Local Search solver for the TSP.

    The algorithm performs the following loop:
        1. Start from an initial tour (random or supplied)
        2. Repeatedly examine the neighborhood
        3. Apply an improving move (first-improvement or best-improvement)
        4. Stop when no improving move exists or iteration cap is hit
        5. (Optional) Restart from a new random tour to escape local optima

    Because pure Local Search stops at the first local optimum, multi-restart
    is the canonical way to get a useful result.
    """

    def __init__(self,
                 distance_matrix: np.ndarray,
                 strategy: str = 'best',
                 neighborhood: str = '2-opt',
                 max_iterations: int = 10000,
                 max_no_improve: int = 1000,
                 seed: Optional[int] = None):
        """
        Args:
            distance_matrix: N×N symmetric distance matrix
            strategy: 'best' or 'first' improvement
            neighborhood: 'swap' / '1-opt', '2-opt', 'or-opt', '3-opt'
            max_iterations: Max neighbor evaluations per restart
            max_no_improve: Stop restart early after this many non-improving iters
            seed: Random seed
        """
        if strategy not in ('best', 'first'):
            raise ValueError(f"strategy must be 'best' or 'first', got {strategy!r}")
        if neighborhood not in NEIGHBORHOOD_FUNCTIONS:
            raise ValueError(
                f"neighborhood must be one of {list(NEIGHBORHOOD_FUNCTIONS)}, got {neighborhood!r}"
            )

        self.distance_matrix = distance_matrix
        self.num_cities = distance_matrix.shape[0]
        self.strategy = strategy
        self.neighborhood = neighborhood
        self.max_iterations = max_iterations
        self.max_no_improve = max_no_improve

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    # -----------------------------------------------------------------
    # Basic tour utilities
    # -----------------------------------------------------------------
    def _compute_cost(self, tour: List[int]) -> float:
        total = 0.0
        n = len(tour)
        for i in range(n):
            total += self.distance_matrix[tour[i]][tour[(i + 1) % n]]
        return total

    def _compute_delta(self, tour: List[int], i: int, j: int) -> float:
        """
        Compute cost delta for a 2-opt move WITHOUT copying the tour.
        Δ = (d[i-1][j] + d[i][j+1]) - (d[i-1][i] + d[j][j+1])
        Returns NaN if the move cannot be evaluated incrementally.
        """
        if self.neighborhood != '2-opt':
            return float('nan')
        n = len(tour)
        if i > j:
            i, j = j, i
        prev_i = tour[(i - 1) % n]
        city_i = tour[i]
        city_j = tour[j]
        next_j = tour[(j + 1) % n]
        if prev_i == city_j and city_i == next_j:
            return 0.0
        if prev_i == city_j or city_i == next_j:
            return float('nan')
        old_edge_cost = (self.distance_matrix[prev_i][city_i]
                         + self.distance_matrix[city_j][next_j])
        new_edge_cost = (self.distance_matrix[prev_i][city_j]
                         + self.distance_matrix[city_i][next_j])
        return new_edge_cost - old_edge_cost

    def _generate_initial_tour(self) -> List[int]:
        tour = list(range(self.num_cities))
        random.shuffle(tour)
        return tour

    def _random_indices(self) -> Tuple[int, int]:
        i = random.randrange(self.num_cities)
        j = random.randrange(self.num_cities)
        while j == i:
            j = random.randrange(self.num_cities)
        return i, j

    # -----------------------------------------------------------------
    # Move generation & application
    # -----------------------------------------------------------------
    def _apply_move(self, tour: List[int], i: int, j: int) -> List[int]:
        fn = NEIGHBORHOOD_FUNCTIONS[self.neighborhood]
        return fn(tour, i, j)

    # -----------------------------------------------------------------
    # One hill-climbing run (single restart)
    # -----------------------------------------------------------------
    def _climb_once(self, initial_tour: List[int]) -> Tuple[List[int], float, int]:
        """
        Run hill climbing from initial_tour until a local optimum is reached
        (or iteration / no-improve cap is hit).

        Returns: (best_tour, best_cost, iterations_used)
        """
        current_tour = initial_tour.copy()
        current_cost = self._compute_cost(current_tour)
        iterations = 0
        no_improve = 0

        while iterations < self.max_iterations and no_improve < self.max_no_improve:
            iterations += 1

            if self.neighborhood == '2-opt' and self.strategy == 'first':
                # Fast first-improvement: try random 2-opt moves
                i, j = self._random_indices()
                delta = self._compute_delta(current_tour, i, j)
                if not math.isnan(delta) and delta < 0:
                    new_tour = self._apply_move(current_tour, i, j)
                    new_cost = current_cost + delta
                    current_tour = new_tour
                    current_cost = new_cost
                    no_improve = 0
                else:
                    no_improve += 1
                continue

            # best-improvement: scan the entire neighborhood
            best_delta = 0.0
            best_move: Optional[Tuple[int, int]] = None
            best_new_tour: Optional[List[int]] = None

            if self.neighborhood in ('swap', '1-opt'):
                # O(n^2) move enumeration
                for i in range(self.num_cities):
                    for j in range(i + 1, self.num_cities):
                        new_tour = _swap_move(current_tour, i, j)
                        new_cost = self._compute_cost(new_tour)
                        delta = new_cost - current_cost
                        if delta < best_delta:
                            best_delta = delta
                            best_move = (i, j)
                            best_new_tour = new_tour

            elif self.neighborhood == '2-opt':
                # Use incremental delta for speed
                for i in range(self.num_cities):
                    for j in range(i + 1, self.num_cities):
                        delta = self._compute_delta(current_tour, i, j)
                        if not math.isnan(delta) and delta < best_delta:
                            best_delta = delta
                            best_move = (i, j)
                            best_new_tour = _two_opt_move(current_tour, i, j)

            elif self.neighborhood == 'or-opt':
                for i in range(self.num_cities):
                    for j in range(self.num_cities):
                        if i == j or (i + 1) % self.num_cities == j:
                            continue
                        new_tour = _or_opt_move(current_tour, i, j)
                        new_cost = self._compute_cost(new_tour)
                        delta = new_cost - current_cost
                        if delta < best_delta:
                            best_delta = delta
                            best_move = (i, j)
                            best_new_tour = new_tour

            elif self.neighborhood == '3-opt':
                # Sample a subset of 3-opt moves (full O(n^3) is too expensive)
                sample_size = min(200, self.num_cities * 2)
                for _ in range(sample_size):
                    a, b = self._random_indices()
                    c = random.randrange(self.num_cities)
                    while c == a or c == b:
                        c = random.randrange(self.num_cities)
                    new_tour = _three_opt_move(current_tour, a, b, c)
                    new_cost = self._compute_cost(new_tour)
                    delta = new_cost - current_cost
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (a, b)
                        best_new_tour = new_tour

            if best_new_tour is None:
                # Local optimum reached
                break

            current_tour = best_new_tour
            current_cost += best_delta
            no_improve = 0

        return current_tour, current_cost, iterations

    # -----------------------------------------------------------------
    # Multi-restart run
    # -----------------------------------------------------------------
    def run(self, num_restarts: int = 30, verbose: bool = False) -> LSResult:
        """
        Run multi-restart Local Search.

        Args:
            num_restarts: Number of independent random restarts
            verbose: Print progress

        Returns:
            LSResult with best tour across all restarts
        """
        start_time = time.time()
        cost_history: List[float] = []
        final_costs: List[float] = []
        final_tours: List[List[int]] = []
        iterations_per_restart: List[int] = []

        best_tour: Optional[List[int]] = None
        best_cost = float('inf')
        unique_tours_set: set = set()

        for r in range(num_restarts):
            init_tour = self._generate_initial_tour()
            tour, cost, iters = self._climb_once(init_tour)
            final_costs.append(cost)
            final_tours.append(tour)
            iterations_per_restart.append(iters)
            unique_tours_set.add(tuple(tour))

            if cost < best_cost:
                best_cost = cost
                best_tour = tour.copy()
                cost_history.append(cost)
            else:
                # Track best-so-far even if this restart didn't beat it
                cost_history.append(best_cost)

            if verbose and (r + 1) % max(1, num_restarts // 5) == 0:
                print(f"  Restart {r + 1}/{num_restarts}: "
                      f"Best={best_cost:.2f}, Last={cost:.2f}")

        elapsed = time.time() - start_time

        return LSResult(
            best_tour=best_tour if best_tour is not None else list(range(self.num_cities)),
            best_cost=best_cost,
            cost_history=cost_history,
            final_costs_per_restart=final_costs,
            iterations_per_restart=iterations_per_restart,
            elapsed_time=elapsed,
            converged=(best_cost > 0),
            unique_tours_explored=len(unique_tours_set),
            final_tours_per_restart=final_tours,
        )


# Lookup of neighborhood name -> move function
NEIGHBORHOOD_FUNCTIONS = {
    'swap': _swap_move,
    '1-opt': _swap_move,
    '2-opt': _two_opt_move,
    'or-opt': _or_opt_move,
    '3-opt': None,  # 3-opt uses special sampling, not a 2-arg move
}
