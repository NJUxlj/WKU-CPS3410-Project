"""
Artificial Bee Colony (ABC) Algorithm for the Traveling Salesman Problem (TSP)

Implements the classic ABC algorithm:
- Food sources = candidate solutions (TSP tours)
- Employed bees: exploit each food source via neighborhood search
- Onlooker bees: probabilistically select good food sources (roulette)
- Scout bees: replace abandoned food sources after `limit` stagnant cycles

This is the population-based swarm variant of the Bee Algorithm described
in the project requirements. Each food source has:
    - position = permutation tour
    - fitness = 1 / cost (higher is better)
    - trial counter = how many iterations without improvement
"""

import numpy as np
import random
import time
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class ABCParams:
    """Configuration parameters for Artificial Bee Colony."""
    colony_size: int = 50              # Number of food sources (= employed bees)
    max_iterations: int = 500
    limit: int = 100                   # Abandonment threshold (trial count)
    neighborhood_type: str = '2-opt'   # 'swap', '2-opt', 'or-opt'
    max_no_improve: int = 100          # Early stop threshold
    seed: Optional[int] = None


@dataclass
class FoodSource:
    """A food source = a candidate TSP solution."""
    tour: List[int]
    cost: float
    fitness: float
    trial: int = 0  # Iterations without improvement


@dataclass
class ABCResult:
    """Stores results of a single ABC run."""
    best_tour: List[int]
    best_cost: float
    cost_history: List[float]              # Best-so-far per iteration
    mean_cost_history: List[float]          # Mean cost of all food sources
    abandoned_sources_history: List[int]    # # of scout-bee replacements per iter
    final_food_costs: List[float]
    elapsed_time: float
    iterations_run: int
    converged: bool
    unique_tours_explored: int = 0   # # of distinct tours visited


class ArtificialBeeColony:
    """Artificial Bee Colony solver for the TSP."""

    def __init__(self,
                 distance_matrix: np.ndarray,
                 colony_size: int = 50,
                 max_iterations: int = 500,
                 limit: int = 100,
                 neighborhood_type: str = '2-opt',
                 max_no_improve: int = 100,
                 seed: Optional[int] = None):
        """
        Args:
            distance_matrix: N×N symmetric distance matrix
            colony_size: Number of food sources (employed bee count)
            max_iterations: Maximum iterations
            limit: Trial limit before scout bee abandons a source
            neighborhood_type: Move operator for employed/onlooker bees
            max_no_improve: Early-stop threshold
            seed: Random seed
        """
        self.distance_matrix = distance_matrix
        self.num_cities = distance_matrix.shape[0]
        self.colony_size = colony_size
        self.max_iterations = max_iterations
        self.limit = limit
        self.neighborhood_type = neighborhood_type
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

    def _fitness(self, cost: float) -> float:
        """Standard ABC fitness: 1 / (1 + cost) so higher is always better."""
        return 1.0 / (1.0 + cost)

    def _random_tour(self) -> List[int]:
        tour = list(range(self.num_cities))
        self.rng.shuffle(tour)
        return tour

    def _initialize_sources(self) -> List[FoodSource]:
        sources: List[FoodSource] = []
        for _ in range(self.colony_size):
            tour = self._random_tour()
            cost = self._compute_cost(tour)
            sources.append(FoodSource(tour=tour, cost=cost,
                                      fitness=self._fitness(cost)))
        return sources

    # -----------------------------------------------------------------
    # Neighborhood operators (single source -> single neighbor)
    # -----------------------------------------------------------------
    def _mutate(self, tour: List[int]) -> List[int]:
        """Generate a neighbor via the chosen neighborhood operator."""
        n = self.num_cities
        if self.neighborhood_type == 'swap':
            i, j = self.rng.sample(range(n), 2)
            new_tour = tour.copy()
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
            return new_tour
        elif self.neighborhood_type == '2-opt':
            i, j = sorted(self.rng.sample(range(n), 2))
            new_tour = tour.copy()
            new_tour[i:j + 1] = list(reversed(new_tour[i:j + 1]))
            return new_tour
        elif self.neighborhood_type == 'or-opt':
            i, j = self.rng.sample(range(n), 2)
            new_tour = tour.copy()
            city = new_tour.pop(i)
            target = j + 1 if j > i else j
            new_tour.insert(target, city)
            return new_tour
        else:
            raise ValueError(f"Unknown neighborhood: {self.neighborhood_type}")

    # -----------------------------------------------------------------
    # Greedy selection: keep the better of two solutions
    # -----------------------------------------------------------------
    def _greedy_select(self, source: FoodSource, candidate_tour: List[int]) -> FoodSource:
        """If candidate is better, replace source and reset trial; else increment trial."""
        candidate_cost = self._compute_cost(candidate_tour)
        if candidate_cost < source.cost:
            return FoodSource(
                tour=candidate_tour,
                cost=candidate_cost,
                fitness=self._fitness(candidate_cost),
                trial=0,
            )
        return FoodSource(
            tour=source.tour,
            cost=source.cost,
            fitness=source.fitness,
            trial=source.trial + 1,
        )

    # -----------------------------------------------------------------
    # Employed bee phase
    # -----------------------------------------------------------------
    def _employed_bee_phase(self, sources: List[FoodSource]) -> List[FoodSource]:
        new_sources: List[FoodSource] = []
        for src in sources:
            candidate = self._mutate(src.tour)
            new_sources.append(self._greedy_select(src, candidate))
        return new_sources

    # -----------------------------------------------------------------
    # Onlooker bee phase: probabilistic selection proportional to fitness
    # -----------------------------------------------------------------
    def _onlooker_bee_phase(self, sources: List[FoodSource]) -> List[FoodSource]:
        total_fitness = sum(max(s.fitness, 0.0) for s in sources)
        if total_fitness <= 0:
            return self._employed_bee_phase(sources)

        new_sources: List[FoodSource] = []
        # Each onlooker picks a source with probability ∝ fitness
        picks: List[int] = []
        for _ in range(self.colony_size):
            r = self.rng.uniform(0, total_fitness)
            running = 0.0
            chosen_idx = len(sources) - 1
            for idx, s in enumerate(sources):
                running += max(s.fitness, 0.0)
                if running >= r:
                    chosen_idx = idx
                    break
            picks.append(chosen_idx)

        for idx in picks:
            src = sources[idx]
            candidate = self._mutate(src.tour)
            new_sources.append(self._greedy_select(src, candidate))
        return new_sources

    # -----------------------------------------------------------------
    # Scout bee phase: replace exhausted sources
    # -----------------------------------------------------------------
    def _scout_bee_phase(self, sources: List[FoodSource]) -> tuple:
        new_sources: List[FoodSource] = []
        abandoned = 0
        for src in sources:
            if src.trial >= self.limit:
                # Abandon and create new random source
                tour = self._random_tour()
                cost = self._compute_cost(tour)
                new_sources.append(FoodSource(
                    tour=tour, cost=cost,
                    fitness=self._fitness(cost), trial=0,
                ))
                abandoned += 1
            else:
                new_sources.append(src)
        return new_sources, abandoned

    # -----------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------
    def run(self, verbose: bool = False) -> ABCResult:
        start_time = time.time()
        sources = self._initialize_sources()

        best_tour = min(sources, key=lambda s: s.cost).tour.copy()
        best_cost = min(s.cost for s in sources)
        cost_history: List[float] = [best_cost]
        mean_cost_history: List[float] = [float(np.mean([s.cost for s in sources]))]
        abandoned_history: List[int] = [0]
        no_improve = 0
        tracked_tours: set = set()
        for s in sources:
            tracked_tours.add(tuple(s.tour))

        for it in range(self.max_iterations):
            # 1. Employed bees
            sources = self._employed_bee_phase(sources)
            # 2. Onlooker bees (probability-weighted selection)
            sources = self._onlooker_bee_phase(sources)
            # 3. Scout bees
            sources, abandoned = self._scout_bee_phase(sources)
            abandoned_history.append(abandoned)
            for s in sources:
                tracked_tours.add(tuple(s.tour))

            current_best = min(s.cost for s in sources)
            current_mean = float(np.mean([s.cost for s in sources]))
            cost_history.append(current_best)
            mean_cost_history.append(current_mean)

            if current_best < best_cost - 1e-9:
                best_cost = current_best
                best_tour = min(sources, key=lambda s: s.cost).tour.copy()
                no_improve = 0
            else:
                no_improve += 1

            if verbose and (it + 1) % max(1, self.max_iterations // 10) == 0:
                print(f"  Iter {it + 1}/{self.max_iterations}: "
                      f"Best={current_best:.2f}, Mean={current_mean:.2f}, "
                      f"Abandoned={abandoned}")

            if no_improve >= self.max_no_improve:
                if verbose:
                    print(f"  Converged at iter {it + 1}")
                break

        elapsed = time.time() - start_time
        return ABCResult(
            best_tour=best_tour,
            best_cost=best_cost,
            cost_history=cost_history,
            mean_cost_history=mean_cost_history,
            abandoned_sources_history=abandoned_history,
            final_food_costs=sorted(s.cost for s in sources),
            elapsed_time=elapsed,
            iterations_run=len(cost_history) - 1,
            converged=(no_improve < self.max_no_improve),
            unique_tours_explored=len(tracked_tours)
        )
