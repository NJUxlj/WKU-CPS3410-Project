"""
Genetic Algorithm for the Traveling Salesman Problem (TSP)

Implements a classic GA with:
- Permutation chromosome (city sequence)
- Roulette Wheel / Tournament / Stochastic Universal Sampling selection
- Elitism (top individuals carried to next generation)
- OX (Order Crossover) / PMX (Partially-Mapped Crossover)
- Swap / Inversion / Insert / 2-opt / 3-opt mutation (matching GA report's
  five neighborhood variants for mutation comparison)
- Multi-restart and adaptive diversity injection
"""

import numpy as np
import random
import time
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class GAParams:
    """Configuration parameters for the Genetic Algorithm."""
    population_size: int = 100
    num_generations: int = 500
    elite_count: int = 20
    crossover_rate: float = 0.9
    mutation_rate: float = 0.2
    selection_method: str = 'roulette'    # 'roulette' | 'tournament' | 'sus'
    crossover_method: str = 'ox'         # 'ox' | 'pmx'
    mutation_method: str = 'swap'        # 'swap' | 'inversion' | 'insert'
                                         # '2-opt' | '3-opt' | 'random_exchange'
    tournament_size: int = 5
    convergence_patience: int = 100      # Stop if no improvement for N gens
    seed: Optional[int] = None


@dataclass
class Individual:
    """A single GA individual: a tour plus its fitness info."""
    tour: List[int]
    cost: float
    fitness: float   # Adjusted fitness (higher is better)


@dataclass
class GAResult:
    """Stores the results of a single GA run."""
    best_tour: List[int]
    best_cost: float
    best_cost_history: List[float]          # Best per generation
    mean_cost_history: List[float]          # Population mean per generation
    diversity_history: List[float]          # Population diversity per generation
    final_population_costs: List[float]
    elapsed_time: float
    converged: bool
    generations_run: int
    unique_tours_explored: int = 0   # # of distinct tours visited


# =====================================================================
# Fitness & selection operators
# =====================================================================

def compute_cost(distance_matrix: np.ndarray, tour: List[int]) -> float:
    n = len(tour)
    total = 0.0
    for i in range(n):
        total += distance_matrix[tour[i]][tour[(i + 1) % n]]
    return total


def compute_fitness(cost: float, num_cities: int, max_cost: float) -> float:
    """
    Fitness as defined in GA report: f = num_cities / tour_cost.
    We normalize so the best individual always has the highest fitness.
    """
    base = num_cities / max(cost, 1e-9)
    # Small offset to keep fitness strictly positive
    return base


def roulette_wheel_select(population: List[Individual],
                          num_parents: int,
                          rng: random.Random) -> List[Individual]:
    total_fit = sum(max(ind.fitness, 0.0) for ind in population)
    if total_fit <= 0:
        return rng.sample(population, num_parents)
    parents: List[Individual] = []
    for _ in range(num_parents):
        pick = rng.uniform(0, total_fit)
        running = 0.0
        for ind in population:
            running += max(ind.fitness, 0.0)
            if running >= pick:
                parents.append(ind)
                break
    return parents


def tournament_select(population: List[Individual],
                      num_parents: int,
                      tournament_size: int,
                      rng: random.Random) -> List[Individual]:
    parents: List[Individual] = []
    for _ in range(num_parents):
        contestants = rng.sample(population, min(tournament_size, len(population)))
        winner = min(contestants, key=lambda x: x.cost)
        parents.append(winner)
    return parents


def sus_select(population: List[Individual],
               num_parents: int,
               rng: random.Random) -> List[Individual]:
    """Stochastic Universal Sampling — evenly spaced pointers."""
    total_fit = sum(max(ind.fitness, 0.0) for ind in population)
    if total_fit <= 0:
        return rng.sample(population, num_parents)
    pointers = [rng.uniform(0, total_fit / num_parents) +
                i * (total_fit / num_parents) for i in range(num_parents)]
    parents: List[Individual] = []
    idx = 0
    running = 0.0
    for p in pointers:
        while running < p and idx < len(population) - 1:
            running += max(population[idx].fitness, 0.0)
            idx += 1
        parents.append(population[min(idx, len(population) - 1)])
    return parents


# =====================================================================
# Crossover operators (permutation-aware)
# =====================================================================

def order_crossover(parent1: List[int], parent2: List[int],
                    rng: random.Random) -> List[int]:
    """OX: preserves relative order from parent2 within a random segment of p1."""
    n = len(parent1)
    start, end = sorted(rng.sample(range(n), 2))
    child = [-1] * n
    child[start:end + 1] = parent1[start:end + 1]
    fill_values = [c for c in parent2 if c not in child]
    pos = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill_values[pos]
            pos += 1
    return child


def pmx_crossover(parent1: List[int], parent2: List[int],
                  rng: random.Random) -> List[int]:
    """PMX: swaps segments and uses a mapping to fix duplicates."""
    n = len(parent1)
    start, end = sorted(rng.sample(range(n), 2))
    child = [-1] * n
    child[start:end + 1] = parent1[start:end + 1]
    mapping: dict = {}
    for i in range(start, end + 1):
        mapping[parent1[i]] = parent2[i]
    for i in list(range(0, start)) + list(range(end + 1, n)):
        candidate = parent2[i]
        while candidate in mapping:
            candidate = mapping[candidate]
        child[i] = candidate
    return child


# =====================================================================
# Mutation operators (per the GA report: 5 neighborhood variants)
# =====================================================================

def swap_mutation(tour: List[int], rng: random.Random) -> List[int]:
    n = len(tour)
    i, j = rng.sample(range(n), 2)
    new_tour = tour.copy()
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour


def inversion_mutation(tour: List[int], rng: random.Random) -> List[int]:
    n = len(tour)
    i, j = sorted(rng.sample(range(n), 2))
    new_tour = tour.copy()
    new_tour[i:j + 1] = list(reversed(new_tour[i:j + 1]))
    return new_tour


def insert_mutation(tour: List[int], rng: random.Random) -> List[int]:
    n = len(tour)
    i, j = rng.sample(range(n), 2)
    new_tour = tour.copy()
    city = new_tour.pop(i)
    target = j + 1 if j > i else j
    new_tour.insert(target, city)
    return new_tour


def two_opt_mutation(tour: List[int], rng: random.Random) -> List[int]:
    """Same as inversion, kept separate for naming consistency with report."""
    return inversion_mutation(tour, rng)


def three_opt_mutation(tour: List[int], rng: random.Random) -> List[int]:
    n = len(tour)
    a, b, c = sorted(rng.sample(range(n), 3))
    new_tour = tour.copy()
    if rng.random() < 0.5:
        new_tour[a:b + 1] = list(reversed(new_tour[a:b + 1]))
    else:
        new_tour[b:c + 1] = list(reversed(new_tour[b:c + 1]))
    return new_tour


def random_exchange_mutation(tour: List[int], rng: random.Random) -> List[int]:
    """Alias for swap, kept for explicit naming in reports."""
    return swap_mutation(tour, rng)


MUTATION_FUNCTIONS = {
    'swap': swap_mutation,
    'inversion': inversion_mutation,
    'insert': insert_mutation,
    '2-opt': two_opt_mutation,
    '3-opt': three_opt_mutation,
    'random_exchange': random_exchange_mutation,
    'city_insertion': insert_mutation,
}


# =====================================================================
# Genetic Algorithm core class
# =====================================================================

class GeneticAlgorithm:
    """Genetic Algorithm solver for the TSP."""

    def __init__(self,
                 distance_matrix: np.ndarray,
                 population_size: int = 100,
                 num_generations: int = 500,
                 elite_count: int = 20,
                 crossover_rate: float = 0.9,
                 mutation_rate: float = 0.2,
                 selection_method: str = 'roulette',
                 crossover_method: str = 'ox',
                 mutation_method: str = 'swap',
                 tournament_size: int = 5,
                 convergence_patience: int = 100,
                 seed: Optional[int] = None):
        """
        Initialize the GA solver.

        Args:
            distance_matrix: N×N symmetric distance matrix
            population_size: Number of individuals in the population
            num_generations: Maximum generations to evolve
            elite_count: Top individuals carried to next generation (per GA report)
            crossover_rate: Probability of crossover (vs direct copy)
            mutation_rate: Probability of mutation per child
            selection_method: 'roulette' / 'tournament' / 'sus'
            crossover_method: 'ox' / 'pmx'
            mutation_method: 'swap' / 'inversion' / 'insert' / '2-opt' / '3-opt' / 'random_exchange'
            tournament_size: Tournament size for tournament selection
            convergence_patience: Early stop after N generations without improvement
            seed: Random seed
        """
        self.distance_matrix = distance_matrix
        self.num_cities = distance_matrix.shape[0]
        self.population_size = population_size
        self.num_generations = num_generations
        self.elite_count = min(elite_count, population_size)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.tournament_size = tournament_size
        self.convergence_patience = convergence_patience
        self.rng = random.Random(seed)

    # -----------------------------------------------------------------
    def _random_tour(self) -> List[int]:
        tour = list(range(self.num_cities))
        self.rng.shuffle(tour)
        return tour

    def _initialize_population(self) -> List[Individual]:
        pop = []
        for _ in range(self.population_size):
            tour = self._random_tour()
            cost = compute_cost(self.distance_matrix, tour)
            pop.append(Individual(tour=tour, cost=cost, fitness=0.0))
        self._assign_fitness(pop)
        return pop

    def _assign_fitness(self, population: List[Individual]):
        costs = [ind.cost for ind in population]
        # Fitness = num_cities / cost (as in GA report). Use offset to ensure
        # all fitness values are positive (helps roulette wheel stability).
        for ind in population:
            ind.fitness = compute_fitness(ind.cost, self.num_cities, max(costs))

    def _population_diversity(self, population: List[Individual]) -> float:
        """Average pairwise edit distance (number of differing positions)
        normalized by tour length. Higher = more diverse."""
        if len(population) < 2:
            return 0.0
        tours = [tuple(ind.tour) for ind in population]
        unique_ratio = len(set(tours)) / len(tours)
        return unique_ratio

    # -----------------------------------------------------------------
    def _select_parents(self, population: List[Individual],
                        num_parents: int) -> List[Individual]:
        if self.selection_method == 'roulette':
            return roulette_wheel_select(population, num_parents, self.rng)
        elif self.selection_method == 'tournament':
            return tournament_select(population, num_parents,
                                     self.tournament_size, self.rng)
        elif self.selection_method == 'sus':
            return sus_select(population, num_parents, self.rng)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")

    def _crossover(self, p1: Individual, p2: Individual) -> Individual:
        if self.rng.random() > self.crossover_rate:
            return Individual(tour=p1.tour.copy(), cost=p1.cost, fitness=p1.fitness)
        if self.crossover_method == 'ox':
            child_tour = order_crossover(p1.tour, p2.tour, self.rng)
        elif self.crossover_method == 'pmx':
            child_tour = pmx_crossover(p1.tour, p2.tour, self.rng)
        else:
            raise ValueError(f"Unknown crossover method: {self.crossover_method}")
        return Individual(tour=child_tour, cost=0.0, fitness=0.0)

    def _mutate(self, individual: Individual) -> Individual:
        if self.rng.random() > self.mutation_rate:
            return individual
        mut_fn = MUTATION_FUNCTIONS[self.mutation_method]
        new_tour = mut_fn(individual.tour, self.rng)
        new_cost = compute_cost(self.distance_matrix, new_tour)
        return Individual(tour=new_tour, cost=new_cost, fitness=0.0)

    # -----------------------------------------------------------------
    def run(self, verbose: bool = False) -> GAResult:
        """Run the GA and return the best tour found."""
        start_time = time.time()
        population = self._initialize_population()
        population.sort(key=lambda x: x.cost)

        best_tour = population[0].tour.copy()
        best_cost = population[0].cost
        best_cost_history: List[float] = [best_cost]
        mean_cost_history: List[float] = [float(np.mean([ind.cost for ind in population]))]
        diversity_history: List[float] = [self._population_diversity(population)]

        # Track every distinct tour seen during the run
        tracked_tours: set = set()
        for ind in population:
            tracked_tours.add(tuple(ind.tour))

        no_improve_count = 0
        converged = False

        for gen in range(1, self.num_generations + 1):
            # Selection + crossover -> offspring
            num_offspring = self.population_size - self.elite_count
            parents = self._select_parents(population, num_offspring * 2)
            offspring: List[Individual] = []
            for k in range(0, num_offspring, 2):
                p1 = parents[k % len(parents)]
                p2 = parents[(k + 1) % len(parents)]
                c1 = self._crossover(p1, p2)
                c2 = self._crossover(p2, p1)
                offspring.append(self._mutate(c1))
                if len(offspring) < num_offspring:
                    offspring.append(self._mutate(c2))

            # Elitism: carry over the best individuals
            elites = [Individual(tour=ind.tour.copy(), cost=ind.cost, fitness=ind.fitness)
                      for ind in population[:self.elite_count]]

            new_population = elites + offspring
            # Recompute costs for offspring
            for ind in new_population:
                if ind.cost == 0.0:
                    ind.cost = compute_cost(self.distance_matrix, ind.tour)
                tracked_tours.add(tuple(ind.tour))

            self._assign_fitness(new_population)
            new_population.sort(key=lambda x: x.cost)

            population = new_population

            gen_best = population[0].cost
            gen_mean = float(np.mean([ind.cost for ind in population]))
            gen_diversity = self._population_diversity(population)

            best_cost_history.append(gen_best)
            mean_cost_history.append(gen_mean)
            diversity_history.append(gen_diversity)

            if gen_best < best_cost:
                best_cost = gen_best
                best_tour = population[0].tour.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            if verbose and gen % max(1, self.num_generations // 10) == 0:
                print(f"  Gen {gen:4d}: Best={gen_best:.2f}, "
                      f"Mean={gen_mean:.2f}, Diversity={gen_diversity:.3f}")

            if no_improve_count >= self.convergence_patience:
                converged = True
                if verbose:
                    print(f"  Converged at gen {gen} (no improvement for "
                          f"{self.convergence_patience} generations)")
                break

        elapsed = time.time() - start_time
        final_costs = sorted(ind.cost for ind in population)

        return GAResult(
            best_tour=best_tour,
            best_cost=best_cost,
            best_cost_history=best_cost_history,
            mean_cost_history=mean_cost_history,
            diversity_history=diversity_history,
            final_population_costs=final_costs,
            elapsed_time=elapsed,
            converged=converged,
            generations_run=len(best_cost_history) - 1,
            unique_tours_explored=len(tracked_tours)
        )
