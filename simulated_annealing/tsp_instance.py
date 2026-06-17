"""
TSP Instance Management Module
Handles loading, generating, and managing Traveling Salesman Problem instances.

Supports:
- Loading distance matrices from text files
- Generating random symmetric TSP instances from 2D coordinates
- Saving instances in a standard format
- Computing tour costs
"""

import numpy as np
import random
from pathlib import Path
from typing import Tuple, List, Optional


class TSPInstance:
    """Represents a TSP problem instance with distance matrix and metadata."""

    def __init__(self, name: str, distance_matrix: np.ndarray,
                 coordinates: Optional[np.ndarray] = None,
                 optimal_value: Optional[float] = None):
        """
        Initialize a TSP instance.

        Args:
            name: Instance name (e.g., 'cities_10', 'cities_50')
            distance_matrix: N×N symmetric matrix of pairwise distances
            coordinates: Optional N×2 array of city (x, y) coordinates
            optimal_value: Known optimal tour length (if available)
        """
        self.name = name
        self.distance_matrix = distance_matrix
        self.num_cities = distance_matrix.shape[0]
        self.coordinates = coordinates
        self.optimal_value = optimal_value

    def cost(self, tour: List[int]) -> float:
        """
        Compute the total cost (distance) of a TSP tour.

        Args:
            tour: List of city indices representing the visit order
                  (should NOT include return to start - it's added automatically)

        Returns:
            Total tour length
        """
        total = 0.0
        n = len(tour)
        for i in range(n):
            from_city = tour[i]
            to_city = tour[(i + 1) % n]
            total += self.distance_matrix[from_city][to_city]
        return total

    def cost_partial(self, tour: List[int]) -> float:
        """
        Compute cost of a partial tour (no return to start).
        """
        total = 0.0
        for i in range(len(tour) - 1):
            total += self.distance_matrix[tour[i]][tour[i + 1]]
        return total

    def save(self, filepath: str):
        """
        Save the distance matrix to a text file.
        Format: N on first line, then N×N matrix (space-separated integers).
        """
        with open(filepath, 'w') as f:
            f.write(f"{self.num_cities}\n")
            for i in range(self.num_cities):
                row = ' '.join(str(int(self.distance_matrix[i][j]))
                              for j in range(self.num_cities))
                f.write(row + '\n')

    @classmethod
    def load(cls, filepath: str, name: Optional[str] = None,
             optimal_value: Optional[float] = None) -> 'TSPInstance':
        """
        Load a TSP instance from a text file.

        Expected format:
            Line 1: N (number of cities)
            Lines 2 to N+1: N space-separated integers per line (distance matrix)

        Args:
            filepath: Path to the distance matrix file
            name: Optional instance name
            optimal_value: Optional known optimal tour length

        Returns:
            TSPInstance object
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()

        n = int(lines[0].strip())
        matrix = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            values = list(map(float, lines[i + 1].strip().split()))
            for j in range(n):
                matrix[i][j] = values[j]

        if name is None:
            name = Path(filepath).stem

        return cls(name=name, distance_matrix=matrix, optimal_value=optimal_value)

    @classmethod
    def generate_random(cls, num_cities: int, seed: int = 42,
                        max_coord: float = 500.0,
                        name: Optional[str] = None) -> 'TSPInstance':
        """
        Generate a random symmetric TSP instance from 2D Euclidean coordinates.

        Cities are randomly placed in a [0, max_coord] × [0, max_coord] square.
        Distances are Euclidean, rounded to integers (representing km).

        Args:
            num_cities: Number of cities
            seed: Random seed for reproducibility
            max_coord: Maximum coordinate value
            name: Optional instance name

        Returns:
            TSPInstance with random coordinates and distance matrix
        """
        rng = np.random.RandomState(seed)
        coordinates = rng.uniform(0, max_coord, size=(num_cities, 2))

        # Compute Euclidean distance matrix
        matrix = np.zeros((num_cities, num_cities), dtype=np.float64)
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    dx = coordinates[i][0] - coordinates[j][0]
                    dy = coordinates[i][1] - coordinates[j][1]
                    matrix[i][j] = np.sqrt(dx * dx + dy * dy)

        if name is None:
            name = f"random_{num_cities}"

        return cls(name=name, distance_matrix=matrix, coordinates=coordinates)

    @classmethod
    def generate_tuned(cls, num_cities: int, target_optimal: float,
                       seed: int = 42, tolerance: float = 0.02) -> 'TSPInstance':
        """
        Generate a random instance and scale it to have an optimal value
        approximately equal to target_optimal.

        Uses iterative scaling: generates random coordinates, finds approximate
        optimal via SA or brute force, then scales distances.

        Args:
            num_cities: Number of cities
            target_optimal: Desired optimal tour length
            seed: Random seed
            tolerance: Acceptable relative error from target

        Returns:
            TSPInstance with scaled distances
        """
        rng = np.random.RandomState(seed)

        # Generate random coordinates
        coordinates = rng.uniform(0, 500, size=(num_cities, 2))

        # Compute distance matrix
        matrix = np.zeros((num_cities, num_cities), dtype=np.float64)
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    dx = coordinates[i][0] - coordinates[j][0]
                    dy = coordinates[i][1] - coordinates[j][1]
                    matrix[i][j] = np.sqrt(dx * dx + dy * dy)

        # Find approximate optimal
        if num_cities <= 12:
            approx_optimal = find_optimal_brute_force(matrix)
        else:
            # Use SA with very slow cooling to estimate optimal
            from .simulated_annealing import SimulatedAnnealing
            sa = SimulatedAnnealing(
                distance_matrix=matrix,
                initial_temp=10000.0,
                cooling_rate=0.99999,
                min_temp=0.001,
                max_iterations_per_temp=500,
                neighborhood_type='2-opt',
                seed=seed + 1000
            )
            best_tour, best_cost, _ = sa.run()
            approx_optimal = best_cost
            # Try multiple times with different seeds
            for extra_seed in range(seed + 2000, seed + 2010):
                sa2 = SimulatedAnnealing(
                    distance_matrix=matrix,
                    initial_temp=10000.0,
                    cooling_rate=0.99999,
                    min_temp=0.001,
                    max_iterations_per_temp=500,
                    neighborhood_type='2-opt',
                    seed=extra_seed
                )
                _, cost, _ = sa2.run()
                if cost < approx_optimal:
                    approx_optimal = cost

        # Scale the distance matrix to achieve target optimal
        scale_factor = target_optimal / approx_optimal
        scaled_matrix = matrix * scale_factor

        name = f"tuned_{num_cities}"
        return cls(name=name, distance_matrix=scaled_matrix,
                   coordinates=coordinates, optimal_value=target_optimal)

    def __repr__(self) -> str:
        opt_str = f", optimal={self.optimal_value:.0f}" if self.optimal_value else ""
        return (f"TSPInstance(name='{self.name}', "
                f"cities={self.num_cities}{opt_str})")


def find_optimal_brute_force(distance_matrix: np.ndarray) -> float:
    """
    Find the exact optimal TSP tour length using brute force enumeration.

    Warning: O(N!) complexity. Only feasible for N ≤ 12.

    Args:
        distance_matrix: N×N distance matrix

    Returns:
        Optimal tour length
    """
    import itertools
    n = distance_matrix.shape[0]
    cities = list(range(1, n))  # Fix city 0 as start
    best_cost = float('inf')

    for perm in itertools.permutations(cities):
        tour = [0] + list(perm)
        cost = 0.0
        for i in range(n - 1):
            cost += distance_matrix[tour[i]][tour[i + 1]]
        cost += distance_matrix[tour[-1]][tour[0]]  # Return to start
        if cost < best_cost:
            best_cost = cost

    return best_cost
