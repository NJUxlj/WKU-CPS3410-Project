"""
Ant Colony Optimization (ACO) for the Traveling Salesman Problem (TSP)

Implements the Ant System (AS) variant with optional Ant Colony System (ACS)
extensions:
- Pheromone matrix τ[i][j] tracking past tour quality on each edge
- Heuristic information η[i][j] = 1 / d[i][j] (closer = more attractive)
- Stochastic tour construction: probability ∝ τ^α * η^β
- Pheromone evaporation + deposit (ant-cycle update rule)
- Optional ACS local pheromone update + global best-only deposit
"""

import numpy as np
import random
import time
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class ACOParams:
    """Configuration parameters for Ant Colony Optimization."""
    num_ants: int = 30
    num_iterations: int = 500
    alpha: float = 1.0           # Pheromone exponent
    beta: float = 2.0            # Heuristic exponent
    rho: float = 0.1             # Evaporation rate (per iteration)
    q0: float = 0.9              # ACS pseudo-random proportional rule prob
    initial_pheromone: float = 1.0
    use_acs: bool = False        # If True, use Ant Colony System
    use_best_only: bool = True   # If True, only best-so-far deposits
    max_no_improve: int = 100    # Early stop if no improvement
    seed: Optional[int] = None


@dataclass
class ACOResult:
    """Stores results of a single ACO run."""
    best_tour: List[int]
    best_cost: float
    cost_history: List[float]            # Best-so-far per iteration
    iteration_best_history: List[float]  # Best cost found this iteration
    pheromone_final: np.ndarray         # Final pheromone matrix
    elapsed_time: float
    iterations_run: int
    converged: bool
    unique_tours_explored: int = 0   # # of distinct tours visited


class AntColony:
    """Ant Colony Optimization solver for the TSP."""

    def __init__(self,
                 distance_matrix: np.ndarray,
                 num_ants: int = 30,
                 num_iterations: int = 500,
                 alpha: float = 1.0,
                 beta: float = 2.0,
                 rho: float = 0.1,
                 q0: float = 0.9,
                 initial_pheromone: float = 1.0,
                 use_acs: bool = False,
                 use_best_only: bool = True,
                 max_no_improve: int = 100,
                 seed: Optional[int] = None):
        """
        Args:
            distance_matrix: N×N symmetric distance matrix
            num_ants: Number of ants per iteration
            num_iterations: Maximum iterations
            alpha: Pheromone importance factor
            beta: Heuristic (1/distance) importance factor
            rho: Pheromone evaporation rate (0, 1)
            q0: ACS pseudo-random proportional rule parameter (0 = pure random)
            initial_pheromone: Initial pheromone level
            use_acs: Enable Ant Colony System local pheromone update
            use_best_only: Only global-best deposits pheromone (vs all ants)
            max_no_improve: Early stop after N iters with no improvement
            seed: Random seed
        """
        self.distance_matrix = distance_matrix
        self.num_cities = distance_matrix.shape[0]
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0
        self.initial_pheromone = initial_pheromone
        self.use_acs = use_acs
        self.use_best_only = use_best_only
        self.max_no_improve = max_no_improve
        self.rng = random.Random(seed)
        if seed is not None:
            np.random.seed(seed)

    def _compute_cost(self, tour: List[int]) -> float:
        n = len(tour)
        total = 0.0
        for i in range(n):
            total += self.distance_matrix[tour[i]][tour[(i + 1) % n]]
        return total

    def _initialize_pheromone(self) -> np.ndarray:
        # Use nearest-neighbor heuristic to set a sensible initial value
        # Heuristic: τ_0 = num_ants / (nearest_neighbor_cost)
        # Simple fallback: τ_0 = initial_pheromone
        return np.full((self.num_cities, self.num_cities),
                       self.initial_pheromone, dtype=np.float64)

    def _heuristic_matrix(self) -> np.ndarray:
        """η[i][j] = 1 / d[i][j], with self-loops set to 0."""
        with np.errstate(divide='ignore'):
            eta = 1.0 / self.distance_matrix
        np.fill_diagonal(eta, 0.0)
        return eta

    def _construct_tour(self, pheromone: np.ndarray, eta: np.ndarray,
                        start_city: Optional[int] = None) -> List[int]:
        """
        Build one ant's tour using the probabilistic transition rule.
        Returns a permutation of [0, n_cities).
        """
        n = self.num_cities
        if start_city is None:
            start_city = self.rng.randrange(n)
        tour = [start_city]
        visited = {start_city}

        for _ in range(n - 1):
            current = tour[-1]
            # Compute unvisited probabilities
            unvisited = [j for j in range(n) if j not in visited]
            if not unvisited:
                break

            if self.use_acs and self.rng.random() < self.q0:
                # ACS pseudo-random proportional: pick the best neighbor
                best_j = max(
                    unvisited,
                    key=lambda j: (pheromone[current][j] ** self.alpha) *
                                  (eta[current][j] ** self.beta),
                )
                tour.append(best_j)
                visited.add(best_j)
            else:
                weights = np.array([
                    (pheromone[current][j] ** self.alpha) *
                    (eta[current][j] ** self.beta)
                    for j in unvisited
                ])
                total = weights.sum()
                if total <= 0:
                    # Degenerate — pick first unvisited
                    chosen = unvisited[0]
                else:
                    probs = weights / total
                    chosen = self.rng.choices(unvisited, weights=probs, k=1)[0]
                tour.append(chosen)
                visited.add(chosen)

        return tour

    def _evaporate(self, pheromone: np.ndarray) -> np.ndarray:
        pheromone *= (1.0 - self.rho)
        return pheromone

    def _deposit_pheromone(self, pheromone: np.ndarray, tours: List[List[int]],
                           costs: List[float]) -> np.ndarray:
        """Ant-cycle update: each ant deposits Q/cost on each visited edge."""
        n = self.num_cities
        Q = 1.0
        if self.use_best_only:
            best_idx = int(np.argmin(costs))
            tours_to_deposit = [tours[best_idx]]
            costs_to_deposit = [costs[best_idx]]
        else:
            tours_to_deposit = tours
            costs_to_deposit = costs

        for tour, cost in zip(tours_to_deposit, costs_to_deposit):
            if cost <= 0:
                continue
            delta = Q / cost
            for k in range(n):
                i = tour[k]
                j = tour[(k + 1) % n]
                pheromone[i][j] += delta
                pheromone[j][i] += delta  # Symmetric TSP
        return pheromone

    def _acs_local_update(self, pheromone: np.ndarray, tour: List[int]) -> None:
        """ACS local pheromone update — evaporate on visited edges."""
        n = len(tour)
        xi = 0.1  # ACS local evaporation factor
        for k in range(n):
            i = tour[k]
            j = tour[(k + 1) % n]
            pheromone[i][j] = (1.0 - xi) * pheromone[i][j] + xi * self.initial_pheromone
            pheromone[j][i] = pheromone[i][j]

    def run(self, verbose: bool = False) -> ACOResult:
        start_time = time.time()
        pheromone = self._initialize_pheromone()
        eta = self._heuristic_matrix()

        best_tour: Optional[List[int]] = None
        best_cost = float('inf')
        cost_history: List[float] = []
        iter_best_history: List[float] = []
        no_improve = 0
        iterations_run = 0
        tracked_tours: set = set()

        for it in range(self.num_iterations):
            iterations_run = it + 1
            iter_tours: List[List[int]] = []
            iter_costs: List[float] = []

            # Construct tours for all ants
            for ant in range(self.num_ants):
                tour = self._construct_tour(pheromone, eta)
                cost = self._compute_cost(tour)
                iter_tours.append(tour)
                iter_costs.append(cost)
                tracked_tours.add(tuple(tour))
                if cost < best_cost:
                    best_cost = cost
                    best_tour = tour.copy()
                    no_improve = 0
                # ACS local pheromone update (between ant solutions)
                if self.use_acs:
                    self._acs_local_update(pheromone, tour)

            # Global pheromone update (after all ants)
            if not self.use_acs:
                self._evaporate(pheromone)
            else:
                # ACS uses different evaporation factor for global update
                pheromone *= (1.0 - self.rho)

            self._deposit_pheromone(pheromone, iter_tours, iter_costs)

            cost_history.append(best_cost)
            iter_best_history.append(min(iter_costs))
            if best_cost > iter_best_history[-1]:
                no_improve += 1

            if verbose and (it + 1) % max(1, self.num_iterations // 10) == 0:
                print(f"  Iter {it + 1}/{self.num_iterations}: "
                      f"Best={best_cost:.2f}, IterBest={iter_best_history[-1]:.2f}")

            if no_improve >= self.max_no_improve:
                if verbose:
                    print(f"  Converged at iter {it + 1}")
                break

        elapsed = time.time() - start_time
        return ACOResult(
            best_tour=best_tour if best_tour is not None else list(range(self.num_cities)),
            best_cost=best_cost,
            cost_history=cost_history,
            iteration_best_history=iter_best_history,
            pheromone_final=pheromone,
            elapsed_time=elapsed,
            iterations_run=iterations_run,
            converged=(no_improve < self.max_no_improve),
            unique_tours_explored=len(tracked_tours)
        )
