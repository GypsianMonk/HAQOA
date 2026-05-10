"""
TSP Problem Definition — HAQOA-X  (Phase 2 + Phase 5)
Provides TSPInstance with fast local search for large instances,
all generator functions, and BENCHMARK_SUITE (tiny → n500).
"""

from __future__ import annotations

import random
import math
from typing import List, Tuple, Dict, Any, Callable

import numpy as np


# ─── Neighbour-list builder ───────────────────────────────────────────────────

def _build_neighbor_lists(dist_matrix: np.ndarray, k: int = 15) -> np.ndarray:
    """Pre-compute k nearest neighbours for each city. Returns (n, k) int32 array."""
    n = dist_matrix.shape[0]
    k = min(k, n - 1)
    return np.argsort(dist_matrix, axis=1)[:, 1:k + 1].astype(np.int32)


# ─── TSP Instance ─────────────────────────────────────────────────────────────

class TSPInstance:
    """
    Euclidean TSP instance.
    Phase 5 additions: two_opt_nn (O(k·n)), or_opt, local_search_large.
    """

    def __init__(self, coords: np.ndarray, name: str = "tsp"):
        self.coords      = np.array(coords, dtype=float)
        self.n           = len(self.coords)
        self.name        = name
        self.dist_matrix = self._build_distance_matrix()
        self._neighbors: np.ndarray = None   # built lazily

    def _build_distance_matrix(self) -> np.ndarray:
        c    = self.coords
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        return np.sqrt((diff ** 2).sum(axis=2))

    def _get_neighbors(self, k: int = 15) -> np.ndarray:
        """Lazy-build and cache k-NN lists."""
        if self._neighbors is None or self._neighbors.shape[1] < k:
            self._neighbors = _build_neighbor_lists(self.dist_matrix, k)
        return self._neighbors

    # ── Objective ─────────────────────────────────────────────────────────────

    def route_distance(self, route: List[int]) -> float:
        total = 0.0; n = len(route); dm = self.dist_matrix
        for i in range(n):
            total += dm[route[i], route[(i + 1) % n]]
        return total

    # ── Initial solutions ─────────────────────────────────────────────────────

    def random_route(self) -> List[int]:
        route = list(range(self.n)); random.shuffle(route); return route

    def greedy_route(self, start: int = 0) -> List[int]:
        """Nearest-neighbour greedy construction."""
        dm = self.dist_matrix; visited = [False] * self.n
        route = [start]; visited[start] = True
        for _ in range(self.n - 1):
            current = route[-1]; best_dist = float("inf"); best_city = -1
            for j in range(self.n):
                if not visited[j] and dm[current, j] < best_dist:
                    best_dist = dm[current, j]; best_city = j
            route.append(best_city); visited[best_city] = True
        return route

    # ── Standard 2-opt (small instances n ≤ 100) ──────────────────────────────

    def two_opt_improve(self, route: List[int], max_iter: int = 500) -> List[int]:
        """Full O(n²) 2-opt. Automatically delegates to two_opt_nn for n > 100."""
        if self.n > 100:
            return self.two_opt_nn(route, max_iter=max_iter)
        best = list(route); best_dist = self.route_distance(best)
        improved = True; iteration = 0
        while improved and iteration < max_iter:
            improved = False; iteration += 1
            for i in range(1, self.n - 1):
                for j in range(i + 1, self.n):
                    new_route = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                    new_dist  = self.route_distance(new_route)
                    if new_dist < best_dist - 1e-10:
                        best = new_route; best_dist = new_dist; improved = True; break
                if improved: break
        return best

    # ── Neighbour-list 2-opt (large instances n > 100) ────────────────────────

    def two_opt_nn(self, route: List[int], max_iter: int = 200, k: int = 15) -> List[int]:
        """
        Neighbour-list accelerated 2-opt.
        Only tests swaps involving a city's k nearest neighbours.
        O(k·n) per pass — ~10× faster than full 2-opt at n=500.
        """
        nbrs = self._get_neighbors(k)
        best = list(route); dm = self.dist_matrix; n = self.n
        best_dist = self.route_distance(best)

        pos = [0] * n
        for idx, city in enumerate(best):
            pos[city] = idx

        improved = True; iteration = 0
        while improved and iteration < max_iter:
            improved = False; iteration += 1
            for i in range(n):
                ci  = best[i]; ci1 = best[(i + 1) % n]
                d_ci = dm[ci, ci1]
                for nb in nbrs[ci]:
                    j = pos[nb]; cj = best[j]; cj1 = best[(j + 1) % n]
                    if abs(i - j) <= 1 or {i, j} == {0, n - 1}: continue
                    gain = d_ci + dm[cj, cj1] - dm[ci, cj] - dm[ci1, cj1]
                    if gain > 1e-10:
                        lo, hi = min(i + 1, j), max(i + 1, j)
                        best[lo:hi + 1] = best[lo:hi + 1][::-1]
                        for idx in range(lo, hi + 1): pos[best[idx]] = idx
                        best_dist -= gain; improved = True; break
                if improved: break
        return best

    # ── Or-opt (relocate segments) ────────────────────────────────────────────

    def or_opt(self, route: List[int], seg_len: int = 1, max_iter: int = 100) -> List[int]:
        """
        Or-opt: relocate a segment of length seg_len to a better position.
        O(n²) per pass but typically fast in practice.
        """
        best = list(route); best_dist = self.route_distance(best)
        dm = self.dist_matrix; n = self.n
        improved = True; iteration = 0
        while improved and iteration < max_iter:
            improved = False; iteration += 1
            for i in range(n):
                seg   = [best[(i + s) % n] for s in range(seg_len)]
                prev  = best[(i - 1) % n]; after = best[(i + seg_len) % n]
                remove_gain = (dm[prev, seg[0]] + dm[seg[-1], after] - dm[prev, after])
                for j in range(n):
                    if j in range(i - 1, i + seg_len + 1): continue
                    a, b = best[j], best[(j + 1) % n]
                    # skip if insertion point city is inside the removed segment
                    if a in seg or b in seg: continue
                    insert_cost = dm[a, seg[0]] + dm[seg[-1], b] - dm[a, b]
                    if remove_gain - insert_cost > 1e-10:
                        idxs = set((i + s) % n for s in range(seg_len))
                        rest = [best[k] for k in range(n) if k not in idxs]
                        if a not in rest: continue      # guard against wrap-around edge cases
                        ins  = rest.index(a) + 1
                        best = rest[:ins] + seg + rest[ins:]
                        best_dist -= (remove_gain - insert_cost)
                        improved = True; break
                if improved: break
        return best

    # ── Combined local search for large instances ─────────────────────────────

    def local_search_large(self, route: List[int]) -> List[int]:
        """2-opt-NN → or-opt(1) → or-opt(2). Used by Phase 5 runner."""
        r = self.two_opt_nn(route, max_iter=100, k=15)
        r = self.or_opt(r, seg_len=1, max_iter=50)
        r = self.or_opt(r, seg_len=2, max_iter=30)
        return r

    def __repr__(self) -> str:
        return f"TSPInstance(name={self.name!r}, n={self.n})"


# ─── Instance generators ──────────────────────────────────────────────────────

def generate_random_tsp(n: int, seed: int = 0, name: str = "random") -> TSPInstance:
    rng = np.random.default_rng(seed)
    return TSPInstance(rng.uniform(0, 100, size=(n, 2)), name=name)

def generate_clustered_tsp(n: int, n_clusters: int = 5, cluster_spread: float = 8.0,
                            seed: int = 0, name: str = "clustered") -> TSPInstance:
    rng = np.random.default_rng(seed)
    centers = rng.uniform(10, 90, size=(n_clusters, 2))
    coords  = []
    for i in range(n):
        c = centers[i % n_clusters]
        coords.append(np.clip(rng.normal(loc=c, scale=cluster_spread), 0, 100))
    rng.shuffle(coords)
    return TSPInstance(np.array(coords), name=name)

def generate_circle_tsp(n: int, name: str = "circle") -> TSPInstance:
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    coords = np.column_stack([50 + 45 * np.cos(angles), 50 + 45 * np.sin(angles)])
    return TSPInstance(coords, name=name)

def generate_grid_tsp(rows: int, cols: int, name: str = "grid") -> TSPInstance:
    coords = [[c * 10, r * 10] for r in range(rows) for c in range(cols)]
    return TSPInstance(np.array(coords, dtype=float), name=name)

def generate_mixed_tsp(n: int, n_clusters: int = 8, seed: int = 0,
                        name: str = "mixed") -> TSPInstance:
    """
    Phase 5: mixed landscape — clustered regions + random scatter.
    Harder than pure random or pure clustered.
    """
    rng = np.random.default_rng(seed)
    n_cluster = int(n * 0.65)
    n_random  = n - n_cluster
    centers   = rng.uniform(15, 85, size=(n_clusters, 2))
    clustered = []
    for i in range(n_cluster):
        c = centers[i % n_clusters]
        clustered.append(np.clip(rng.normal(loc=c, scale=6.0), 0, 100))
    scattered = rng.uniform(0, 100, size=(n_random, 2)).tolist()
    coords    = clustered + scattered
    rng.shuffle(coords)
    return TSPInstance(np.array(coords), name=name)


# ─── Benchmark Suite ──────────────────────────────────────────────────────────

BENCHMARK_SUITE: Dict[str, Tuple[Callable, Dict[str, Any]]] = {
    # Phase 2 instances
    "tiny":      (generate_random_tsp,    {"n": 10,  "seed": 1}),
    "small":     (generate_random_tsp,    {"n": 20,  "seed": 42}),
    "medium":    (generate_clustered_tsp, {"n": 50,  "n_clusters": 6, "cluster_spread": 7.0, "seed": 7}),
    "large":     (generate_random_tsp,    {"n": 100, "seed": 99}),
    "circle_20": (generate_circle_tsp,    {"n": 20}),
    "grid_4x5":  (generate_grid_tsp,      {"rows": 4, "cols": 5}),
    # Phase 5 instances
    "n200_random":   (generate_random_tsp,    {"n": 200, "seed": 7}),
    "n200_mixed":    (generate_mixed_tsp,     {"n": 200, "n_clusters": 10, "seed": 13}),
    "n300_clustered":(generate_clustered_tsp, {"n": 300, "n_clusters": 12, "cluster_spread": 6.0, "seed": 21}),
    "n500_random":   (generate_random_tsp,    {"n": 500, "seed": 3}),
    "n500_mixed":    (generate_mixed_tsp,     {"n": 500, "n_clusters": 15, "seed": 37}),
}
