"""
HAQOA - TSP Solver Module

Implements Traveling Salesman Problem (TSP) specific functionality for HAQOA.
This serves as the initial benchmark target per the research blueprint.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import random


@dataclass
class TSPInstance:
    """Represents a TSP problem instance."""
    
    n_cities: int
    coordinates: np.ndarray  # Shape: (n_cities, 2)
    distance_matrix: np.ndarray  # Shape: (n_cities, n_cities)
    name: str = "TSP"
    optimal_value: Optional[float] = None
    
    def __post_init__(self):
        if self.distance_matrix.shape != (self.n_cities, self.n_cities):
            raise ValueError("Distance matrix shape mismatch")
        if self.coordinates.shape != (self.n_cities, 2):
            raise ValueError("Coordinates shape mismatch")


@dataclass
class TSPSolution:
    """Represents a TSP solution (tour)."""
    
    tour: List[int]  # Ordered list of city indices
    total_distance: float = 0.0
    is_valid: bool = True
    
    def __post_init__(self):
        # Validate tour
        if len(set(self.tour)) != len(self.tour):
            self.is_valid = False
        if len(self.tour) == 0:
            self.is_valid = False


class TSPSolver:
    """
    TSP-specific solver using HAQOA framework.
    
    Provides:
    - Solution generation
    - Mutation operators
    - Recombination (crossover) operators
    - Quality evaluation
    """
    
    def __init__(self, tsp_instance: TSPInstance):
        """
        Initialize TSP solver.
        
        Args:
            tsp_instance: TSP problem instance
        """
        self.instance = tsp_instance
        self.n_cities = tsp_instance.n_cities
        self.dist_matrix = tsp_instance.distance_matrix
    
    def generate_random_tour(self) -> TSPSolution:
        """Generate a random valid tour."""
        tour = list(np.random.permutation(self.n_cities))
        return self.evaluate_solution(TSPSolution(tour=tour))
    
    def evaluate_solution(self, solution: TSPSolution) -> TSPSolution:
        """
        Evaluate a TSP solution by computing total distance.
        
        For maximization in HAQOA, we use negative distance or 1/distance.
        """
        if not solution.is_valid:
            solution.total_distance = float('inf')
            return solution
        
        tour = solution.tour
        total_dist = 0.0
        
        for i in range(len(tour)):
            from_city = tour[i]
            to_city = tour[(i + 1) % len(tour)]  # Wrap around
            total_dist += self.dist_matrix[from_city, to_city]
        
        solution.total_distance = total_dist
        return solution
    
    def compute_quality(self, solution: TSPSolution) -> float:
        """
        Compute quality score for HAQOA (higher is better).
        
        Uses inverse distance transformation.
        """
        if solution.total_distance == 0:
            return float('inf')
        
        # Quality = 1 / distance (higher quality for shorter tours)
        return 1.0 / solution.total_distance
    
    def mutate_2opt(self, solution: TSPSolution) -> TSPSolution:
        """
        Apply 2-opt mutation to a tour.
        
        Reverses a segment of the tour to potentially reduce distance.
        """
        tour = solution.tour.copy()
        n = len(tour)
        
        if n < 4:
            return solution
        
        # Select two cut points
        i, j = sorted(np.random.choice(n, size=2, replace=False))
        
        # Reverse segment between i and j
        tour[i:j+1] = tour[i:j+1][::-1]
        
        new_solution = TSPSolution(tour=tour)
        return self.evaluate_solution(new_solution)
    
    def mutate_swap(self, solution: TSPSolution) -> TSPSolution:
        """
        Apply swap mutation to a tour.
        
        Swaps two randomly selected cities.
        """
        tour = solution.tour.copy()
        n = len(tour)
        
        if n < 2:
            return solution
        
        # Select two positions to swap
        i, j = np.random.choice(n, size=2, replace=False)
        tour[i], tour[j] = tour[j], tour[i]
        
        new_solution = TSPSolution(tour=tour)
        return self.evaluate_solution(new_solution)
    
    def mutate_insertion(self, solution: TSPSolution) -> TSPSolution:
        """
        Apply insertion mutation to a tour.
        
        Removes a city and inserts it at a different position.
        """
        tour = solution.tour.copy()
        n = len(tour)
        
        if n < 3:
            return solution
        
        # Select city to move and destination
        city_idx = np.random.randint(n)
        dest_idx = np.random.randint(n - 1)
        
        city = tour.pop(city_idx)
        
        # Adjust destination index if needed
        if dest_idx >= city_idx:
            dest_idx = min(dest_idx, len(tour))
        else:
            dest_idx = min(dest_idx, len(tour))
        
        tour.insert(dest_idx, city)
        
        new_solution = TSPSolution(tour=tour)
        return self.evaluate_solution(new_solution)
    
    def mutate(self, solution: TSPSolution, method: str = "2opt") -> TSPSolution:
        """
        Apply mutation using specified method.
        
        Args:
            solution: Solution to mutate
            method: Mutation method ('2opt', 'swap', 'insertion', 'random')
        """
        if method == "2opt":
            return self.mutate_2opt(solution)
        elif method == "swap":
            return self.mutate_swap(solution)
        elif method == "insertion":
            return self.mutate_insertion(solution)
        elif method == "random":
            methods = ["2opt", "swap", "insertion"]
            method = np.random.choice(methods)
            return self.mutate(solution, method)
        else:
            return self.mutate_2opt(solution)
    
    def recombine_ox1(
        self,
        parent1: TSPSolution,
        parent2: TSPSolution
    ) -> Tuple[TSPSolution, TSPSolution]:
        """
        Order Crossover (OX1) operator.
        
        Preserves relative ordering of cities from parents.
        """
        tour1 = parent1.tour
        tour2 = parent2.tour
        n = len(tour1)
        
        # Select crossover segment
        start, end = sorted(np.random.choice(n, size=2, replace=False))
        
        # Create children
        child1 = [-1] * n
        child2 = [-1] * n
        
        # Copy segment from parents
        child1[start:end+1] = tour1[start:end+1]
        child2[start:end+1] = tour2[start:end+1]
        
        # Fill remaining positions
        def fill_child(child, parent_tour, start_idx, end_idx):
            pos = (end_idx + 1) % n
            idx = (end_idx + 1) % n
            for city in parent_tour:
                if city not in child:
                    # Find next available position
                    while child[pos] != -1:
                        pos = (pos + 1) % n
                    child[pos] = city
                    pos = (pos + 1) % n
        
        fill_child(child1, tour2, start, end)
        fill_child(child2, tour1, start, end)
        
        child1_sol = TSPSolution(tour=child1)
        child2_sol = TSPSolution(tour=child2)
        
        return (
            self.evaluate_solution(child1_sol),
            self.evaluate_solution(child2_sol)
        )
    
    def recombine_pmx(
        self,
        parent1: TSPSolution,
        parent2: TSPSolution
    ) -> Tuple[TSPSolution, TSPSolution]:
        """
        Partially Mapped Crossover (PMX) operator.
        """
        tour1 = parent1.tour
        tour2 = parent2.tour
        n = len(tour1)
        
        # Select crossover segment
        start, end = sorted(np.random.choice(n, size=2, replace=False))
        
        # Create mapping
        mapping = {}
        for i in range(start, end + 1):
            mapping[tour2[i]] = tour1[i]
        
        # Build child1
        child1 = []
        for city in tour2:
            if city in mapping:
                mapped = mapping[city]
                while mapped in mapping:
                    mapped = mapping[mapped]
                child1.append(mapped)
            else:
                child1.append(city)
        
        # Build child2 (reverse mapping)
        reverse_mapping = {v: k for k, v in mapping.items()}
        child2 = []
        for city in tour1:
            if city in reverse_mapping:
                mapped = reverse_mapping[city]
                while mapped in reverse_mapping:
                    mapped = reverse_mapping[mapped]
                child2.append(mapped)
            else:
                child2.append(city)
        
        child1_sol = TSPSolution(tour=child1)
        child2_sol = TSPSolution(tour=child2)
        
        return (
            self.evaluate_solution(child1_sol),
            self.evaluate_solution(child2_sol)
        )
    
    def recombine(
        self,
        parent1: TSPSolution,
        parent2: TSPSolution,
        method: str = "ox1"
    ) -> Tuple[TSPSolution, TSPSolution]:
        """
        Recombine two parent solutions.
        
        Args:
            parent1: First parent
            parent2: Second parent
            method: Crossover method ('ox1', 'pmx')
        """
        if method == "ox1":
            return self.recombine_ox1(parent1, parent2)
        elif method == "pmx":
            return self.recombine_pmx(parent1, parent2)
        else:
            return self.recombine_ox1(parent1, parent2)
    
    def greedy_initial_solution(self) -> TSPSolution:
        """Generate a greedy initial solution using nearest neighbor."""
        unvisited = set(range(self.n_cities))
        tour = []
        
        # Start from random city
        current = np.random.randint(self.n_cities)
        tour.append(current)
        unvisited.remove(current)
        
        # Greedily visit nearest unvisited city
        while unvisited:
            nearest = min(unvisited, key=lambda c: self.dist_matrix[current, c])
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        solution = TSPSolution(tour=tour)
        return self.evaluate_solution(solution)
    
    def get_neighbors(self, solution: TSPSolution) -> List[TSPSolution]:
        """Get all 2-opt neighbors of a solution."""
        tour = solution.tour
        n = len(tour)
        neighbors = []
        
        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue  # Skip full reversal
                
                new_tour = tour.copy()
                new_tour[i:j+1] = new_tour[i:j+1][::-1]
                
                neighbor = TSPSolution(tour=new_tour)
                neighbors.append(self.evaluate_solution(neighbor))
        
        return neighbors


def create_tsp_instance(
    n_cities: int,
    seed: Optional[int] = None,
    coordinates: Optional[np.ndarray] = None
) -> TSPInstance:
    """
    Create a TSP instance.
    
    Args:
        n_cities: Number of cities
        seed: Random seed for reproducibility
        coordinates: Optional pre-defined city coordinates
    
    Returns:
        TSPInstance object
    """
    if seed is not None:
        np.random.seed(seed)
    
    if coordinates is None:
        # Generate random coordinates in unit square
        coordinates = np.random.rand(n_cities, 2)
    
    # Compute Euclidean distance matrix
    diff = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=2))
    
    return TSPInstance(
        n_cities=n_cities,
        coordinates=coordinates,
        distance_matrix=distances,
        name=f"TSP-{n_cities}"
    )


def create_grid_tsp(n_per_side: int, spacing: float = 1.0) -> TSPInstance:
    """Create a TSP instance with cities on a regular grid."""
    n_cities = n_per_side * n_per_side
    coordinates = []
    
    for i in range(n_per_side):
        for j in range(n_per_side):
            coordinates.append([j * spacing, i * spacing])
    
    coordinates = np.array(coordinates)
    
    # Compute distance matrix
    diff = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=2))
    
    return TSPInstance(
        n_cities=n_cities,
        coordinates=coordinates,
        distance_matrix=distances,
        name=f"Grid-TSP-{n_per_side}x{n_per_side}"
    )
