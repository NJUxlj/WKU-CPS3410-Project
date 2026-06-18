"""
Tabu Search (TS) for the Traveling Salesman Problem (TSP)

Implements the classic Tabu Search metaheuristic with:
- 1-opt (swap) neighborhood matching the TS report's design
- Aspiration criterion (override tabu status for new global best)
- Configurable tabu tenure and iteration cap
- Optional diversification via strategic oscillation
"""

import numpy as np
import random
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class TSParams:
    """Configuration parameters for Tabu Search."""
    tabu_tenure: int = 7               # How long a move stays tabu (1-10 per report)
    max_iterations: int = 500          # Total iterations
    max_no_improve: int = 100          # Early stop after N iters with no improvement
    region_iterations: int = 50        # Inner loops (matches TS report)
    neighborhood_iterations: int = 100 # Per-region neighbor evaluations (matches report)
    use_aspiration: bool = True        # Allow tabu move if it beats global best
    seed: Optional[int] = None


@dataclass
class TSResult:
    """Stores the results of a single Tabu Search run."""
    best_tour: List[int]
    best_cost: float
    cost_history: List[float]            # Current cost per iteration
    best_history: List[float]            # Best-so-far per iteration
    tabu_list_size_history: List[int]   # Tabu list active size over time
    elapsed_time: float
    iterations_run: int
    converged: bool
    # Tracking
    tabu_overrides: int = 0              # Times aspiration criterion kicked in
    unique_tours_explored: int = 0   # # of distinct tours visited


def _compute_cost(distance_matrix: np.ndarray, tour: List[int]) -> float:
    n = len(tour)
    total = 0.0
    for i in range(n):
        total += distance_matrix[tour[i]][tour[(i + 1) % n]]
    return total


class TabuSearch:
    """Tabu Search solver for the TSP."""

    def __init__(self,
                 distance_matrix: np.ndarray,
                 tabu_tenure: int = 7,
                 max_iterations: int = 500,
                 max_no_improve: int = 100,
                 region_iterations: int = 50,
                 neighborhood_iterations: int = 100,
                 use_aspiration: bool = True,
                 seed: Optional[int] = None):
        """
        Args:
            distance_matrix: N×N symmetric distance matrix
            tabu_tenure: how long a swap (i, j) stays tabu
            max_iterations: total outer iterations
            max_no_improve: early-stop threshold
            region_iterations: number of region restarts (matches report)
            neighborhood_iterations: per-region neighbor scan count (matches report)
            use_aspiration: enable aspiration criterion
            seed: random seed
        """
        self.distance_matrix = distance_matrix
        self.num_cities = distance_matrix.shape[0]
        self.tabu_tenure = tabu_tenure
        self.max_iterations = max_iterations
        self.max_no_improve = max_no_improve
        self.region_iterations = region_iterations
        self.neighborhood_iterations = neighborhood_iterations
        self.use_aspiration = use_aspiration
        self.rng = random.Random(seed)
        if seed is not None:
            np.random.seed(seed)

    def _initial_solution(self) -> List[int]:
        tour = list(range(self.num_cities))
        self.rng.shuffle(tour)
        return tour

    def _swap_cost_delta(self, tour: List[int], i: int, j: int) -> float:
        """
        Compute delta cost of swapping positions i and j in the tour.

        Non-adjacent case: 4 affected edges
            old: (prev_i, cur_i), (cur_i, nxt_i), (prev_j, cur_j), (cur_j, nxt_j)
            new: (prev_i, cur_j), (cur_j, nxt_i), (prev_j, cur_i), (cur_i, nxt_j)

        Adjacent case (j = i+1): 3 affected edges (two old edges merge into one
        shared edge, two new edges collapse because the swap endpoints are
        neighbors). We handle this explicitly to avoid double-counting.
        """
        n = len(tour)
        if i == j:
            return 0.0
        if i > j:
            i, j = j, i
        prev_i = tour[(i - 1) % n]
        cur_i = tour[i]
        nxt_i = tour[(i + 1) % n]
        prev_j = tour[(j - 1) % n]
        cur_j = tour[j]
        nxt_j = tour[(j + 1) % n]

        # Adjacent case (j = i+1): prev_j == cur_i and nxt_i == cur_j
        # Edges affected (3 each):
        #   old: (prev_i, cur_i), (cur_i, cur_j), (cur_j, nxt_j)
        #   new: (prev_i, cur_j), (cur_j, cur_i), (cur_i, nxt_j)
        if j == (i + 1) % n or (i == 0 and j == n - 1):
            # Generalised handling for wrap-around adjacency
            # j == i+1 in cyclic order
            # 3 old edges vs 3 new edges (no double-counting, no self-loops)
            # For the i==0, j==n-1 wrap-around case, prev_j == cur_i
            # holds (tour[n-2] -> tour[n-1] = cur_j; tour[i-1]=tour[n-1]=cur_j
            # and prev_j would be tour[j-1] = tour[n-2] which is NOT cur_i
            # for i=0,j=n-1). So we need the simpler (i+1)%n==j condition
            # is False for the wrap case -> it falls through to non-adjacent.
            # To keep it simple, treat only true j==i+1 as adjacent and
            # compute the wrap case with the non-adjacent formula (which is
            # correct because no edges overlap there).
            if j == i + 1:
                old_cost = (self.distance_matrix[prev_i][cur_i]
                            + self.distance_matrix[cur_i][cur_j]
                            + self.distance_matrix[cur_j][nxt_j])
                new_cost = (self.distance_matrix[prev_i][cur_j]
                            + self.distance_matrix[cur_j][cur_i]
                            + self.distance_matrix[cur_i][nxt_j])
                return new_cost - old_cost

        # Non-adjacent case: 4 distinct edges each
        old_cost = (self.distance_matrix[prev_i][cur_i]
                    + self.distance_matrix[cur_i][nxt_i]
                    + self.distance_matrix[prev_j][cur_j]
                    + self.distance_matrix[cur_j][nxt_j])
        new_cost = (self.distance_matrix[prev_i][cur_j]
                    + self.distance_matrix[cur_j][nxt_i]
                    + self.distance_matrix[prev_j][cur_i]
                    + self.distance_matrix[cur_i][nxt_j])
        return new_cost - old_cost

    def _apply_swap(self, tour: List[int], i: int, j: int) -> List[int]:
        new_tour = tour.copy()
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        return new_tour

    def _decrement_tabu_list(self, tabu_list: dict) -> int:
        """Decrement all tabu tenures by 1; remove expired entries."""
        expired = [k for k, v in tabu_list.items() if v <= 1]
        for k in expired:
            del tabu_list[k]
        for k in list(tabu_list.keys()):
            tabu_list[k] -= 1
        return len(tabu_list)

    def _run_region(self, start_tour: List[int], tabu_list: dict,
                    global_best_cost: float,
                    tracked_tours: set) -> Tuple[List[int], float, int, int]:
        """
        Run one TS region (inner loop) starting from start_tour.
        Returns (region_best_tour, region_best_cost, tabu_overrides_added,
                 region_iterations_used).
        """
        current_tour = start_tour.copy()
        current_cost = _compute_cost(self.distance_matrix, current_tour)
        region_best_tour = current_tour.copy()
        region_best_cost = current_cost
        region_overrides = 0
        tracked_tours.add(tuple(current_tour))

        for _ in range(self.neighborhood_iterations):
            best_delta = float('inf')
            best_move: Optional[Tuple[int, int]] = None
            best_new_tour: Optional[List[int]] = None
            best_was_tabu_override = False

            sample_size = min(self.num_cities * 2, 50)
            for _ in range(sample_size):
                i = self.rng.randrange(self.num_cities)
                j = self.rng.randrange(self.num_cities)
                while j == i:
                    j = self.rng.randrange(self.num_cities)
                move = (min(i, j), max(i, j))
                is_tabu = move in tabu_list and tabu_list[move] > 0
                if is_tabu and not self.use_aspiration:
                    continue
                delta = self._swap_cost_delta(current_tour, i, j)
                prospective = current_cost + delta
                if is_tabu and prospective >= global_best_cost:
                    continue
                if prospective < best_delta:
                    best_delta = prospective
                    best_move = move
                    best_new_tour = self._apply_swap(current_tour, i, j)
                    best_was_tabu_override = is_tabu

            if best_move is None or best_new_tour is None:
                break

            # Apply move and RECOMPUTE the cost from scratch to prevent
            # any delta accumulation drift (especially around wrap-around
            # edge cases).
            current_tour = best_new_tour
            current_cost = _compute_cost(self.distance_matrix, current_tour)
            tracked_tours.add(tuple(current_tour))
            i, j = best_move
            tabu_list[(i, j)] = self.tabu_tenure
            tabu_list[(j, i)] = self.tabu_tenure

            if best_was_tabu_override:
                region_overrides += 1

            if current_cost < region_best_cost:
                region_best_cost = current_cost
                region_best_tour = current_tour.copy()

        return region_best_tour, region_best_cost, region_overrides, self.neighborhood_iterations

    def run(self, verbose: bool = False) -> TSResult:
        start_time = time.time()
        current_tour = self._initial_solution()
        current_cost = _compute_cost(self.distance_matrix, current_tour)
        best_tour = current_tour.copy()
        best_cost = current_cost

        cost_history: List[float] = [current_cost]
        best_history: List[float] = [best_cost]
        tabu_size_history: List[int] = []
        tabu_list: dict = {}
        total_overrides = 0
        no_improve = 0
        total_iterations = 0
        tracked_tours: set = set()
        tracked_tours.add(tuple(current_tour))

        for outer in range(self.region_iterations):
            region_best_tour, region_best_cost, overrides, iters_used = (
                self._run_region(current_tour, tabu_list, best_cost, tracked_tours)
            )
            total_iterations += iters_used
            total_overrides += overrides

            # Decrement tabu tenures at end of each region
            self._decrement_tabu_list(tabu_list)
            tabu_size_history.append(len(tabu_list))

            current_tour = region_best_tour
            current_cost = region_best_cost
            cost_history.extend([current_cost] * iters_used)
            best_history.extend([best_cost] * iters_used)

            if current_cost < best_cost - 1e-9:
                best_cost = current_cost
                best_tour = current_tour.copy()
                no_improve = 0
            else:
                no_improve += 1

            if verbose and (outer + 1) % max(1, self.region_iterations // 10) == 0:
                print(f"  Region {outer + 1}/{self.region_iterations}: "
                      f"Best={best_cost:.2f}, Tabu size={len(tabu_list)}, "
                      f"Overrides={overrides}")

            if no_improve >= self.max_no_improve:
                if verbose:
                    print(f"  Stopped at region {outer + 1} "
                          f"(no improvement for {self.max_no_improve} regions)")
                break

        elapsed = time.time() - start_time
        return TSResult(
            best_tour=best_tour,
            best_cost=best_cost,
            cost_history=cost_history,
            best_history=best_history,
            tabu_list_size_history=tabu_size_history,
            elapsed_time=elapsed,
            iterations_run=total_iterations,
            converged=(no_improve < self.max_no_improve),
            tabu_overrides=total_overrides,
            unique_tours_explored=len(tracked_tours)
        )
