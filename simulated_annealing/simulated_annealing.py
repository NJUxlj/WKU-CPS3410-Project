"""
Simulated Annealing Algorithm for the Traveling Salesman Problem (TSP)

Implements the classic Simulated Annealing metaheuristic with:
- Multiple cooling schedules (geometric, linear, logarithmic)
- Multiple neighborhood structures (2-opt, 3-opt, swap, insert)
- Metropolis acceptance criterion
- Adaptive parameter tuning
"""

import numpy as np
import random
import time
import math
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class SAResult:
    """Stores the results of a single SA run."""
    best_tour: List[int]
    best_cost: float
    cost_history: List[float]
    temp_history: List[float]
    acceptance_history: List[float]   # Acceptance rate per temperature level
    iteration_count: int
    elapsed_time: float
    converged: bool


@dataclass
class SAParams:
    """Configuration parameters for the Simulated Annealing algorithm."""
    initial_temp: float = 10000.0
    cooling_rate: float = 0.995       # Geometric cooling: T *= cooling_rate
    min_temp: float = 0.01
    max_iterations_per_temp: int = 100  # Markov chain length at each temperature
    cooling_schedule: str = 'geometric'  # 'geometric', 'linear', 'logarithmic'
    neighborhood_type: str = '2-opt'     # '2-opt', 'swap', 'insert', '3-opt'
    restart_stuck: bool = True           # Restart if stuck for too long
    stuck_threshold: int = 50            # Iterations without improvement before restart


class SimulatedAnnealing:
    """
    Simulated Annealing solver for the TSP.

    The algorithm mimics the physical annealing process:
    1. Start at a high temperature T_0
    2. At each temperature, perform multiple iterations:
       a. Generate a neighbor solution
       b. If neighbor is better, accept it
       c. If neighbor is worse, accept with probability exp(-ΔE / T)
    3. Cool down: T = α * T (or other schedule)
    4. Repeat until T < T_min

    Key properties:
    - Escapes local optima by probabilistically accepting worse solutions
    - Acceptance probability decreases as temperature drops (intensification)
    - High temperature → high diversification; Low temperature → high intensification
    """

    def __init__(self,
                 distance_matrix: np.ndarray,
                 initial_temp: float = 10000.0,
                 cooling_rate: float = 0.995,
                 min_temp: float = 0.01,
                 max_iterations_per_temp: int = 100,
                 cooling_schedule: str = 'geometric',
                 neighborhood_type: str = '2-opt',
                 restart_stuck: bool = True,
                 stuck_threshold: int = 50,
                 seed: Optional[int] = None):
        """
        Initialize the SA solver.

        Args:
            distance_matrix: N×N symmetric distance matrix
            initial_temp: Starting temperature T_0
            cooling_rate: Geometric cooling factor α (0 < α < 1)
            min_temp: Stopping temperature
            max_iterations_per_temp: Number of iterations at each temperature level
            cooling_schedule: 'geometric', 'linear', or 'logarithmic'
            neighborhood_type: '2-opt', 'swap', 'insert', or '3-opt'
            restart_stuck: Whether to restart when stuck
            stuck_threshold: Consecutive non-improving iterations before restart
            seed: Random seed for reproducibility
        """
        self.distance_matrix = distance_matrix
        self.num_cities = distance_matrix.shape[0]
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.max_iterations_per_temp = max_iterations_per_temp
        self.cooling_schedule = cooling_schedule
        self.neighborhood_type = neighborhood_type
        self.restart_stuck = restart_stuck
        self.stuck_threshold = stuck_threshold

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def _compute_cost(self, tour: List[int]) -> float:
        """Compute the total cost of a tour."""
        total = 0.0
        n = len(tour)
        for i in range(n):
            total += self.distance_matrix[tour[i]][tour[(i + 1) % n]]
        return total

    def _compute_delta(self, tour: List[int], i: int, j: int,
                       move_type: str) -> float:
        """
        Compute the change in cost for a proposed move WITHOUT modifying the tour.
        This is an optimization to avoid recomputing the full tour cost.

        For 2-opt (reversing segment i..j):
            Old edges: (i-1, i) and (j, j+1)
            New edges: (i-1, j) and (i, j+1)
            Δ = (d[i-1][j] + d[i][j+1]) - (d[i-1][i] + d[j][j+1])
        """
        n = len(tour)
        if move_type == '2-opt':
            # Ensure i < j
            if i > j:
                i, j = j, i

            prev_i = tour[(i - 1) % n]
            city_i = tour[i]
            city_j = tour[j]
            next_j = tour[(j + 1) % n]

            # When the reversed segment is the entire tour (or nearly so),
            # the boundary edges overlap and the tour just reverses direction.
            # For symmetric TSP, this is a no-op (delta = 0).
            if prev_i == city_j and city_i == next_j:
                return 0
            if prev_i == city_j or city_i == next_j:
                # The formula is invalid when boundary edges share cities;
                # fall back to full cost computation
                return None

            old_edge_cost = (self.distance_matrix[prev_i][city_i] +
                           self.distance_matrix[city_j][next_j])
            new_edge_cost = (self.distance_matrix[prev_i][city_j] +
                           self.distance_matrix[city_i][next_j])
            return new_edge_cost - old_edge_cost

        elif move_type == 'swap':
            # Fall back to full cost computation for simplicity and correctness
            return None

        elif move_type == 'insert':
            # Fall back to full cost computation for simplicity and correctness
            return None

        elif move_type == '3-opt':
            # 3-opt: three-edge exchange (simplified: pick 3 break points)
            # For efficiency, we just use the direct computation approach
            return None  # Signal to compute full cost

        return 0

    def _apply_move(self, tour: List[int], i: int, j: int,
                    move_type: str) -> List[int]:
        """Apply a move to the tour and return the new tour."""
        new_tour = tour.copy()
        n = len(tour)

        if move_type == '2-opt':
            if i > j:
                i, j = j, i
            # Reverse the segment from i to j
            new_tour[i:j + 1] = list(reversed(new_tour[i:j + 1]))

        elif move_type == 'swap':
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]

        elif move_type == 'insert':
            city = new_tour.pop(i)
            if j > i:
                j -= 1  # Adjust index since we removed an element
            new_tour.insert(j + 1, city)

        elif move_type == '3-opt':
            # Simplified 3-opt: pick three indices and try all 7 possible reconnections
            # We randomly pick a reconnection pattern
            k = random.randrange(n)
            while k == i or k == j:
                k = random.randrange(n)
            indices = sorted([i, j, k])
            a, b, c = indices[0], indices[1], indices[2]
            # Apply a random 3-opt move
            pattern = random.randint(0, 3)
            if pattern == 0:
                # Reverse middle segment
                new_tour[b:c + 1] = reversed(new_tour[b:c + 1])
            elif pattern == 1:
                # Reverse first segment
                new_tour[a:b + 1] = reversed(new_tour[a:b + 1])
            elif pattern == 2:
                # Reverse last segment  
                new_tour[c:] = reversed(new_tour[c:])
                new_tour[:a + 1] = reversed(new_tour[:a + 1])
            # Pattern 3: swap two segments (simplified)

        return new_tour

    def _cooling_function(self, temp: float, iteration: int) -> float:
        """
        Compute the next temperature based on the cooling schedule.

        Args:
            temp: Current temperature
            iteration: Current iteration number (since start of SA)

        Returns:
            New temperature
        """
        if self.cooling_schedule == 'geometric':
            return temp * self.cooling_rate
        elif self.cooling_schedule == 'linear':
            return temp - self.cooling_rate  # cooling_rate acts as delta
        elif self.cooling_schedule == 'logarithmic':
            # T_k = T_0 / log(1 + k)
            return self.initial_temp / math.log(1 + iteration + 1)
        else:
            return temp * self.cooling_rate

    def _generate_initial_tour(self) -> List[int]:
        """Generate a random initial tour."""
        tour = list(range(self.num_cities))
        random.shuffle(tour)
        return tour

    def _generate_neighbor_indices(self) -> Tuple[int, int]:
        """Generate random indices for neighborhood move."""
        i = random.randrange(self.num_cities)
        j = random.randrange(self.num_cities)
        while j == i:
            j = random.randrange(self.num_cities)
        return i, j

    def run(self, verbose: bool = False) -> SAResult:
        """
        Execute the Simulated Annealing algorithm.

        Returns:
            SAResult containing the best tour, cost, and convergence history
        """
        start_time = time.time()

        # Initialize
        current_tour = self._generate_initial_tour()
        current_cost = self._compute_cost(current_tour)
        best_tour = current_tour.copy()
        best_cost = current_cost

        temperature = self.initial_temp
        iteration = 0
        stuck_count = 0

        # History tracking
        cost_history = []
        temp_history = []
        acceptance_history = []

        while temperature > self.min_temp:
            accepted = 0
            total_moves = 0

            for _ in range(self.max_iterations_per_temp):
                iteration += 1

                # Generate neighbor
                i, j = self._generate_neighbor_indices()

                # Compute delta cost efficiently
                delta = self._compute_delta(current_tour, i, j,
                                           self.neighborhood_type)

                if delta is None or self.neighborhood_type == '3-opt':
                    # Fall back to full cost computation
                    new_tour = self._apply_move(current_tour, i, j,
                                               self.neighborhood_type)
                    new_cost = self._compute_cost(new_tour)
                    delta = new_cost - current_cost
                else:
                    new_tour = None  # Lazy: only create if accepted

                total_moves += 1

                # Accept/reject decision (Metropolis criterion)
                if delta < 0:
                    # Better solution: always accept
                    if new_tour is None:
                        new_tour = self._apply_move(current_tour, i, j,
                                                    self.neighborhood_type)
                    current_tour = new_tour
                    current_cost += delta
                    accepted += 1

                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_tour = current_tour.copy()
                        stuck_count = 0
                    else:
                        stuck_count += 1
                else:
                    # Worse solution: accept with probability exp(-Δ/T)
                    probability = math.exp(-delta / temperature)
                    if random.random() < probability:
                        if new_tour is None:
                            new_tour = self._apply_move(current_tour, i, j,
                                                        self.neighborhood_type)
                        current_tour = new_tour
                        current_cost += delta
                        accepted += 1
                        stuck_count += 1
                    else:
                        stuck_count += 1

                # Record cost periodically
                if iteration % max(1, self.max_iterations_per_temp // 10) == 0:
                    cost_history.append(current_cost)
                    temp_history.append(temperature)

            # Record acceptance rate for this temperature level
            acc_rate = accepted / max(1, total_moves)
            acceptance_history.append(acc_rate)

            if verbose and iteration % (self.max_iterations_per_temp * 10) == 0:
                print(f"  T={temperature:.2f}, Best={best_cost:.2f}, "
                      f"Current={current_cost:.2f}, AccRate={acc_rate:.3f}")

            # Check if stuck and should restart
            if self.restart_stuck and stuck_count > self.stuck_threshold * self.max_iterations_per_temp:
                if verbose:
                    print(f"  Restarting at T={temperature:.2f} (stuck for {stuck_count} iters)")
                current_tour = self._generate_initial_tour()
                current_cost = self._compute_cost(current_tour)
                stuck_count = 0

            # Cool down
            temperature = self._cooling_function(temperature, iteration)

        elapsed = time.time() - start_time

        return SAResult(
            best_tour=best_tour,
            best_cost=best_cost,
            cost_history=cost_history,
            temp_history=temp_history,
            acceptance_history=acceptance_history,
            iteration_count=iteration,
            elapsed_time=elapsed,
            converged=(temperature <= self.min_temp)
        )

    def run_with_adaptive_params(self, verbose: bool = False) -> SAResult:
        """
        Run SA with adaptive parameter tuning.

        The cooling rate is adjusted based on acceptance rate:
        - If acceptance rate is too low (< 0.1), cool slower
        - If acceptance rate is too high (> 0.8), cool faster
        """
        start_time = time.time()
        current_tour = self._generate_initial_tour()
        current_cost = self._compute_cost(current_tour)
        best_tour = current_tour.copy()
        best_cost = current_cost

        temperature = self.initial_temp
        iteration = 0
        stuck_count = 0
        adaptive_rate = self.cooling_rate

        cost_history = []
        temp_history = []
        acceptance_history = []

        while temperature > self.min_temp:
            accepted = 0
            total_moves = 0

            for _ in range(self.max_iterations_per_temp):
                iteration += 1
                i, j = self._generate_neighbor_indices()
                delta = self._compute_delta(current_tour, i, j,
                                           self.neighborhood_type)

                if delta is None:
                    new_tour = self._apply_move(current_tour, i, j,
                                               self.neighborhood_type)
                    new_cost = self._compute_cost(new_tour)
                    delta = new_cost - current_cost
                    new_tour_cache = new_tour
                else:
                    new_tour_cache = None

                total_moves += 1

                if delta < 0:
                    if new_tour_cache is None:
                        new_tour_cache = self._apply_move(current_tour, i, j,
                                                          self.neighborhood_type)
                    current_tour = new_tour_cache
                    current_cost += delta
                    accepted += 1
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_tour = current_tour.copy()
                        stuck_count = 0
                    else:
                        stuck_count += 1
                else:
                    probability = math.exp(-delta / temperature)
                    if random.random() < probability:
                        if new_tour_cache is None:
                            new_tour_cache = self._apply_move(current_tour, i, j,
                                                              self.neighborhood_type)
                        current_tour = new_tour_cache
                        current_cost += delta
                        accepted += 1
                        stuck_count += 1
                    else:
                        stuck_count += 1

                if iteration % max(1, self.max_iterations_per_temp // 10) == 0:
                    cost_history.append(current_cost)
                    temp_history.append(temperature)

            acc_rate = accepted / max(1, total_moves)
            acceptance_history.append(acc_rate)

            # Adaptive cooling rate adjustment
            if acc_rate < 0.1:
                adaptive_rate = min(0.999, adaptive_rate + 0.002)
            elif acc_rate > 0.8:
                adaptive_rate = max(0.9, adaptive_rate - 0.002)

            if self.restart_stuck and stuck_count > self.stuck_threshold * self.max_iterations_per_temp:
                current_tour = self._generate_initial_tour()
                current_cost = self._compute_cost(current_tour)
                stuck_count = 0

            temperature *= adaptive_rate

        elapsed = time.time() - start_time

        return SAResult(
            best_tour=best_tour,
            best_cost=best_cost,
            cost_history=cost_history,
            temp_history=temp_history,
            acceptance_history=acceptance_history,
            iteration_count=iteration,
            elapsed_time=elapsed,
            converged=(temperature <= self.min_temp)
        )
