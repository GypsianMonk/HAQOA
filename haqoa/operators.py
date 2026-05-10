"""
TSP Operators — HAQOA
Mutation, crossover, and objective function adapters for TSP.
"""

from __future__ import annotations

import random
from typing import List, Callable

from haqoa.problems.tsp import TSPInstance


# ─── Objective ────────────────────────────────────────────────────────────────

def make_tsp_objective(tsp: TSPInstance) -> Callable[[List[int]], float]:
    """Returns an objective function f(route) → tour_distance."""
    def objective(route: List[int]) -> float:
        return tsp.route_distance(route)
    return objective


# ─── Mutation Operators ───────────────────────────────────────────────────────

def swap_mutation(route: List[int]) -> List[int]:
    """Randomly swap two cities."""
    r = list(route)
    i, j = random.sample(range(len(r)), 2)
    r[i], r[j] = r[j], r[i]
    return r


def two_opt_mutation(route: List[int]) -> List[int]:
    """Reverse a random sub-segment (2-opt move)."""
    r = list(route)
    n = len(r)
    i = random.randint(0, n - 2)
    j = random.randint(i + 1, n - 1)
    r[i:j + 1] = r[i:j + 1][::-1]
    return r


def or_opt_mutation(route: List[int]) -> List[int]:
    """
    Or-opt: remove a random segment of length 1–3 and reinsert elsewhere.
    Stronger than single-swap for TSP.
    """
    r = list(route)
    n = len(r)
    seg_len = random.randint(1, min(3, n - 1))
    i = random.randint(0, n - seg_len - 1)
    segment = r[i:i + seg_len]
    remainder = r[:i] + r[i + seg_len:]
    insert_pos = random.randint(0, len(remainder))
    new_route = remainder[:insert_pos] + segment + remainder[insert_pos:]
    return new_route


def adaptive_mutation(route: List[int]) -> List[int]:
    """
    Adaptive mutation: randomly choose between swap, 2-opt, or or-opt.
    Mimics entropy-guided mutation.
    """
    op = random.choice([swap_mutation, two_opt_mutation, or_opt_mutation])
    return op(route)


# ─── Crossover Operators ──────────────────────────────────────────────────────

def order_crossover(parent_a: List[int], parent_b: List[int]) -> List[int]:
    """
    OX (Order Crossover): preserves relative ordering.
    Standard for permutation-based TSP.
    """
    n = len(parent_a)
    start, end = sorted(random.sample(range(n), 2))

    child = [-1] * n
    child[start:end + 1] = parent_a[start:end + 1]

    fill_values = [x for x in parent_b if x not in child[start:end + 1]]
    pos = [(end + 1 + i) % n for i in range(n - (end - start + 1))]
    for idx, val in zip(pos, fill_values):
        child[idx] = val

    return child


def pmx_crossover(parent_a: List[int], parent_b: List[int]) -> List[int]:
    """
    PMX (Partially Mapped Crossover) — fully corrected implementation.

    Algorithm:
      1. Copy segment [start, end] from parent_a → child.
      2. Build pos_in_a: gene_value → index, for positions in the segment.
      3. For every position outside the segment, take parent_b[i].
         If that value already appears in the copied segment, follow the
         chain:  val → parent_a position of val → parent_b at that position
         until a value not in the segment is reached.
         The chain always terminates because parent permutations are bijections.
    """
    n = len(parent_a)
    start, end = sorted(random.sample(range(n), 2))

    child = [-1] * n
    child[start:end + 1] = parent_a[start:end + 1]
    segment_set = set(parent_a[start:end + 1])

    # Map: gene value → its index within parent_a's segment
    pos_in_a = {parent_a[k]: k for k in range(start, end + 1)}

    for i in range(n):
        if start <= i <= end:
            continue                         # already filled from parent_a
        val = parent_b[i]
        while val in segment_set:            # follow chain until val is free
            val = parent_b[pos_in_a[val]]    # pos_in_a always has val (bijection)
        child[i] = val

    return child


def edge_assembly_crossover(parent_a: List[int], parent_b: List[int]) -> List[int]:
    """
    Simplified edge-assembly: inherits edges present in both parents first,
    then fills in using greedy nearest-unvisited.
    """
    n = len(parent_a)
    edges_a = set()
    for i in range(n):
        u, v = parent_a[i], parent_a[(i + 1) % n]
        edges_a.add((min(u, v), max(u, v)))
    edges_b = set()
    for i in range(n):
        u, v = parent_b[i], parent_b[(i + 1) % n]
        edges_b.add((min(u, v), max(u, v)))

    shared = edges_a & edges_b
    # Build adjacency from shared edges
    adj = {i: [] for i in range(n)}
    for u, v in shared:
        adj[u].append(v)
        adj[v].append(u)

    # Build route starting from parent_a[0]
    visited = [False] * n
    route = []
    current = parent_a[0]
    visited[current] = True
    route.append(current)

    for _ in range(n - 1):
        candidates = [nb for nb in adj[current] if not visited[nb]]
        if candidates:
            next_city = candidates[0]
        else:
            # Fall back: copy unvisited from parent_a ordering
            next_city = None
            for city in parent_a:
                if not visited[city]:
                    next_city = city
                    break
        visited[next_city] = True
        route.append(next_city)
        current = next_city

    return route


# ─── Operator bundles ─────────────────────────────────────────────────────────

def make_tsp_operators(strategy: str = "standard"):
    """
    Returns (mutation_fn, crossover_fn) pair for a given strategy.

    Strategies:
        'standard'  — swap mutation + OX crossover
        'aggressive' — adaptive mutation + PMX crossover
        'advanced'  — adaptive mutation + edge-assembly crossover
    """
    if strategy == "standard":
        return swap_mutation, order_crossover
    elif strategy == "aggressive":
        return adaptive_mutation, pmx_crossover
    elif strategy == "advanced":
        return adaptive_mutation, edge_assembly_crossover
    else:
        raise ValueError(f"Unknown strategy '{strategy}'")
