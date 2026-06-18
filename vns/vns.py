"""
Variable Neighborhood Search (VNS) and Reduced VNS (RVNS) for TSP.

Algorithm skeleton (Basic VNS):
    1. Choose neighborhood family N_k, k = 1..k_max
    2. Find initial solution x
    3. Repeat:
        a. k = 1
        b. While k <= k_max:
            - Shaking: generate x' at random from N_k(x)
            - Local search: x'' = LocalSearch(x')
            - Move or not:
                if f(x'') < f(x):  x = x'';  k = 1
                else:              k = k + 1

Reduced VNS (RVNS):
    - Same skeleton but skips the Local Search step
    - x' from N_k(x) directly replaces x if better
    - Much faster, more diverse exploration
"""

import numpy as np
import random
import time
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

from local_search.local_search import LocalSearch
from .neighborhoods import NEIGHBORHOOD_FAMILY, shake_swap


def _compute_cost(distance_matrix: np.ndarray, tour: List[int]) -> float:
    n = len(tour)
    total = 0.0
    for i in range(n):
        total += distance_matrix[tour[i]][tour[(i + 1) % n]]
    return total


@dataclass
class VNSParams:
    """Configuration parameters for VNS / RVNS."""
    k_max: int = 5                    # Number of neighborhood structures
    max_iterations: int = 100         # Outer loop iterations
    max_no_improve: int = 30          # Early stop after this many outer iters
    local_search_iterations: int = 2000  # LS iterations per descent
    local_search_no_improve: int = 500
    seed: Optional[int] = None


@dataclass
class VNSResult:
    """Stores the results of a single VNS / RVNS run."""
    best_tour: List[int]
    best_cost: float
    cost_history: List[float]               # Best-so-far per outer iteration
    iterations_per_neighborhood: List[int]  # How often each neighborhood improved
    total_ls_iterations: int
    elapsed_time: float
    converged: bool
    # Diversification analysis
    unique_tours_explored: int = 0


class VNS:
    """
    Basic Variable Neighborhood Search solver for the TSP.
    """

    def __init__(self,
                 distance_matrix: np.ndarray,
                 k_max: int = 5,
                 max_iterations: int = 100,
                 max_no_improve: int = 30,
                 local_search_iterations: int = 2000,
                 local_search_no_improve: int = 500,
                 seed: Optional[int] = None):
        self.distance_matrix = distance_matrix
        self.num_cities = distance_matrix.shape[0]
        self.k_max = min(k_max, len(NEIGHBORHOOD_FAMILY))
        self.max_iterations = max_iterations
        self.max_no_improve = max_no_improve
        self.local_search_iterations = local_search_iterations
        self.local_search_no_improve = local_search_no_improve
        self.rng = random.Random(seed)
        if seed is not None:
            np.random.seed(seed)

    def _initial_solution(self) -> List[int]:
        tour = list(range(self.num_cities))
        self.rng.shuffle(tour)
        return tour

    def _shake(self, tour: List[int], k: int) -> List[int]:
        """Generate a neighbor in N_k."""
        _, shake_fn = NEIGHBORHOOD_FAMILY[k - 1]
        return shake_fn(tour, self.rng)

    def _local_search(self, tour: List[int]) -> Tuple[List[int], float, int]:
        """Apply local search descent and return (tour, cost, iterations)."""
        ls = LocalSearch(
            distance_matrix=self.distance_matrix,
            strategy='best',
            neighborhood='2-opt',
            max_iterations=self.local_search_iterations,
            max_no_improve=self.local_search_no_improve,
            seed=self.rng.randint(0, 2**31 - 1),
        )
        # Use single restart from given tour
        # We simulate this by setting a small neighborhood sample.
        result = ls._climb_once(tour)
        return result

    def run(self, verbose: bool = False) -> VNSResult:
        start_time = time.time()
        current_tour = self._initial_solution()
        current_cost = _compute_cost(self.distance_matrix, current_tour)
        best_tour = current_tour.copy()
        best_cost = current_cost

        cost_history: List[float] = [best_cost]
        iter_per_nb = [0] * self.k_max
        total_ls_iterations = 0
        no_improve = 0
        unique_tours_set: set = set()
        unique_tours_set.add(tuple(current_tour))

        outer_iter = 0
        for outer in range(self.max_iterations):
            outer_iter = outer + 1
            improved_this_round = False

            for k in range(1, self.k_max + 1):
                # Shaking
                shaken_tour = self._shake(current_tour, k)
                unique_tours_set.add(tuple(shaken_tour))

                # Local search descent
                new_tour, new_cost, ls_iters = self._local_search(shaken_tour)
                total_ls_iterations += ls_iters
                unique_tours_set.add(tuple(new_tour))

                # Move / not
                if new_cost < current_cost - 1e-9:
                    current_tour = new_tour
                    current_cost = new_cost
                    improved_this_round = True
                    iter_per_nb[k - 1] += 1
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_tour = new_tour.copy()
                        no_improve = 0
                    break  # Restart neighborhood cycle from k=1
                # else: try next neighborhood (k+1)

            cost_history.append(best_cost)
            if improved_this_round:
                no_improve = 0
            else:
                no_improve += 1

            if verbose and (outer + 1) % max(1, self.max_iterations // 10) == 0:
                print(f"  Iter {outer + 1}/{self.max_iterations}: "
                      f"Best={best_cost:.2f}, Current={current_cost:.2f}")

            if no_improve >= self.max_no_improve:
                if verbose:
                    print(f"  Stopped at iter {outer + 1} (no improvement "
                          f"for {self.max_no_improve} outer iters)")
                break

        elapsed = time.time() - start_time
        return VNSResult(
            best_tour=best_tour,
            best_cost=best_cost,
            cost_history=cost_history,
            iterations_per_neighborhood=iter_per_nb,
            total_ls_iterations=total_ls_iterations,
            elapsed_time=elapsed,
            converged=(no_improve < self.max_no_improve),
            unique_tours_explored=len(unique_tours_set),
        )


class RVNS:
    """
    Reduced Variable Neighborhood Search — same skeleton as VNS but the
    Local Search step is replaced by direct acceptance of the shaken
    solution if it improves the current cost.

    This makes RVNS much faster than VNS, at the cost of solution quality.
    It also exhibits higher diversification (PPT Slide 5).
    """

    def __init__(self,
                 distance_matrix: np.ndarray,
                 k_max: int = 5,
                 max_iterations: int = 500,
                 max_no_improve: int = 100,
                 seed: Optional[int] = None):
        self.distance_matrix = distance_matrix
        self.num_cities = distance_matrix.shape[0]
        self.k_max = min(k_max, len(NEIGHBORHOOD_FAMILY))
        self.max_iterations = max_iterations
        self.max_no_improve = max_no_improve
        self.rng = random.Random(seed)
        if seed is not None:
            np.random.seed(seed)

    def _initial_solution(self) -> List[int]:
        tour = list(range(self.num_cities))
        self.rng.shuffle(tour)
        return tour

    def _shake(self, tour: List[int], k: int) -> List[int]:
        _, shake_fn = NEIGHBORHOOD_FAMILY[k - 1]
        return shake_fn(tour, self.rng)

    def run(self, verbose: bool = False) -> VNSResult:
        start_time = time.time()
        current_tour = self._initial_solution()
        current_cost = _compute_cost(self.distance_matrix, current_tour)
        best_tour = current_tour.copy()
        best_cost = current_cost

        cost_history: List[float] = [best_cost]
        iter_per_nb = [0] * self.k_max
        no_improve = 0
        unique_tours_set: set = set()
        unique_tours_set.add(tuple(current_tour))

        outer_iter = 0
        for outer in range(self.max_iterations):
            outer_iter = outer + 1
            improved_this_round = False

            for k in range(1, self.k_max + 1):
                shaken_tour = self._shake(current_tour, k)
                unique_tours_set.add(tuple(shaken_tour))
                shaken_cost = _compute_cost(self.distance_matrix, shaken_tour)

                if shaken_cost < current_cost - 1e-9:
                    current_tour = shaken_tour
                    current_cost = shaken_cost
                    improved_this_round = True
                    iter_per_nb[k - 1] += 1
                    if shaken_cost < best_cost:
                        best_cost = shaken_cost
                        best_tour = shaken_tour.copy()
                        no_improve = 0
                    break

            cost_history.append(best_cost)
            if improved_this_round:
                no_improve = 0
            else:
                no_improve += 1

            if verbose and (outer + 1) % max(1, self.max_iterations // 10) == 0:
                print(f"  Iter {outer + 1}/{self.max_iterations}: "
                      f"Best={best_cost:.2f}")

            if no_improve >= self.max_no_improve:
                if verbose:
                    print(f"  Stopped at iter {outer + 1} (no improvement "
                          f"for {self.max_no_improve} outer iters)")
                break

        elapsed = time.time() - start_time
        return VNSResult(
            best_tour=best_tour,
            best_cost=best_cost,
            cost_history=cost_history,
            iterations_per_neighborhood=iter_per_nb,
            total_ls_iterations=0,
            elapsed_time=elapsed,
            converged=(no_improve < self.max_no_improve),
            unique_tours_explored=len(unique_tours_set),
        )
