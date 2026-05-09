"""
TSP Problem Definition — HAQOA
Provides TSPInstance, distance matrix, greedy / 2-opt heuristics,
and a BENCHMARK_SUITE of reproducible test instances.
"""

from __future__ import annotations

import random
import math
from typing import List, Optional, Tuple, Dict, Any, Callable

import numpy as np


class TSPInstance:
    """
    Euclidean TSP instance.

    Attributes:
        n           : number of cities
        coords      : (n, 2) array of (x, y) positions
        dist_matrix : (n, n) pairwise Euclidean distance matrix
        name        : human-readable label
    """

    def __init__(self, coords: np.ndarray, name: str = "tsp"):
        self.coords = np.array(coords, dtype=float)
        self.n = len(self.coords)
        self.name = name
        self.dist_matrix = self._build_distance_matrix()

    def _build_distance_matrix(self) -> np.ndarray:
        c = self.coords
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        return np.sqrt((diff ** 2).sum(axis=2))

    def route_distance(self, route: List[int]) -> float:
        total = 0.0
        n = len(route)
        dm = self.dist_matrix
        for i in range(n):
            total += dm[route[i], route[(i + 1) % n]]
        return total

    def random_route(self) -> List[int]:
        route = list(range(self.n))
        random.shuffle(route)
        return route

    def greedy_route(self, start: int = 0) -> List[int]:
        """Nearest-neighbour greedy tour construction."""
        dm = self.dist_matrix
        visited = [False] * self.n
        route = [start]
        visited[start] = True
        for _ in range(self.n - 1):
            current = route[-1]
            best_dist = float("inf")
            best_city = -1
            for j in range(self.n):
                if not visited[j] and dm[current, j] < best_dist:
                    best_dist = dm[current, j]
                    best_city = j
            route.append(best_city)
            visited[best_city] = True
        return route

    def two_opt_improve(self, route: List[int], max_iter: int = 500) -> List[int]:
        """Iterative 2-opt local search."""
        best = list(route)
        best_dist = self.route_distance(best)
        improved = True
        iteration = 0
        while improved and iteration < max_iter:
            improved = False
            iteration += 1
            for i in range(1, self.n - 1):
                for j in range(i + 1, self.n):
                    new_route = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                    new_dist = self.route_distance(new_route)
                    if new_dist < best_dist - 1e-10:
                        best = new_route
                        best_dist = new_dist
                        improved = True
                        break
                if improved:
                    break
        return best

    def __repr__(self) -> str:
        return f"TSPInstance(name={self.name!r}, n={self.n})"


def generate_random_tsp(n: int, seed: int = 0, name: str = "random") -> TSPInstance:
    """Uniformly random cities in [0, 100]²."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 100, size=(n, 2))
    return TSPInstance(coords, name=name)


def generate_clustered_tsp(
    n: int, n_clusters: int = 5, cluster_spread: float = 8.0,
    seed: int = 0, name: str = "clustered",
) -> TSPInstance:
    """Cities arranged in Gaussian clusters."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(10, 90, size=(n_clusters, 2))
    coords = []
    for i in range(n):
        c = centers[i % n_clusters]
        point = rng.normal(loc=c, scale=cluster_spread)
        point = np.clip(point, 0, 100)
        coords.append(point)
    rng.shuffle(coords)
    return TSPInstance(np.array(coords), name=name)


def generate_circle_tsp(n: int, name: str = "circle") -> TSPInstance:
    """Cities evenly spaced on a circle — optimal tour is known."""
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    coords = np.column_stack([50 + 45 * np.cos(angles), 50 + 45 * np.sin(angles)])
    return TSPInstance(coords, name=name)


def generate_grid_tsp(rows: int, cols: int, name: str = "grid") -> TSPInstance:
    """Regular grid layout."""
    coords = [[c * 10, r * 10] for r in range(rows) for c in range(cols)]
    return TSPInstance(np.array(coords, dtype=float), name=name)


BENCHMARK_SUITE: Dict[str, Tuple[Callable, Dict[str, Any]]] = {
    "tiny":      (generate_random_tsp,    {"n": 10,  "seed": 1}),
    "small":     (generate_random_tsp,    {"n": 20,  "seed": 42}),
    "medium":    (generate_clustered_tsp, {"n": 50,  "n_clusters": 6, "cluster_spread": 7.0, "seed": 7}),
    "large":     (generate_random_tsp,    {"n": 100, "seed": 99}),
    "circle_20": (generate_circle_tsp,    {"n": 20}),
    "grid_4x5":  (generate_grid_tsp,      {"rows": 4, "cols": 5}),
}
