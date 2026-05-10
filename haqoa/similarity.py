"""
Similarity Density Field — HAQOA-X  (Part 6 of Master Spec)

Implements:
    D_i = (1/N) Σ_j δ(s_i, s_j)

Multiple similarity metrics for different problem types.
Approximate k-NN for large populations (n > 200).
"""

from __future__ import annotations

from typing import List, Callable, Optional
import numpy as np


# ─── TSP Similarity Metrics ───────────────────────────────────────────────────

def tsp_edge_jaccard(route_a: List[int], route_b: List[int]) -> float:
    """
    Jaccard similarity of undirected edge sets.
    sim = |edges_a ∩ edges_b| / |edges_a ∪ edges_b|
    Range: [0, 1] — 1 = identical routes, 0 = no shared edges.
    """
    n = len(route_a)
    def edges(r):
        return frozenset(
            (min(r[i], r[(i+1) % n]), max(r[i], r[(i+1) % n]))
            for i in range(n)
        )
    ea, eb = edges(route_a), edges(route_b)
    inter = len(ea & eb)
    union = len(ea | eb)
    return inter / union if union > 0 else 1.0


def tsp_positional_similarity(route_a: List[int], route_b: List[int]) -> float:
    """
    Positional overlap: fraction of cities in the same position.
    Fast O(n). Less informative than edge Jaccard but cheaper.
    """
    n = len(route_a)
    matches = sum(1 for i in range(n) if route_a[i] == route_b[i])
    return matches / n


def tsp_kendall_similarity(route_a: List[int], route_b: List[int]) -> float:
    """
    Kendall-tau based similarity via concordant city-pair ordering.
    Range [0,1]. O(n²) — use only for small n or sparse sampling.
    """
    n = len(route_a)
    pos_a = {city: i for i, city in enumerate(route_a)}
    pos_b = {city: i for i, city in enumerate(route_b)}
    concordant = 0
    total = 0
    cities = list(pos_a.keys())
    for i in range(min(n, 50)):   # cap at 50 for speed
        for j in range(i+1, min(n, 50)):
            ci, cj = cities[i], cities[j]
            sign_a = 1 if pos_a[ci] < pos_a[cj] else -1
            sign_b = 1 if pos_b[ci] < pos_b[cj] else -1
            if sign_a == sign_b:
                concordant += 1
            total += 1
    return concordant / total if total > 0 else 1.0


# ─── Generic Sequence Metrics ─────────────────────────────────────────────────

def hamming_similarity(seq_a: List, seq_b: List) -> float:
    """Normalised Hamming similarity (binary/integer sequences)."""
    n = len(seq_a)
    return sum(1 for a, b in zip(seq_a, seq_b) if a == b) / n


# ─── Density Matrix + Scores ─────────────────────────────────────────────────

def compute_density_matrix(
    solutions: List[List],
    similarity_fn: Callable,
    max_pairs: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """
    Compute pairwise similarity matrix.
    For large populations (> sqrt(max_pairs) ≈ 44), uses random sparse sampling
    to keep O(max_pairs) instead of O(n²).
    Returns (n, n) symmetric matrix in [0, 1].
    """
    n = len(solutions)
    D = np.zeros((n, n))

    if n * (n - 1) // 2 <= max_pairs:
        # Full pairwise
        for i in range(n):
            for j in range(i + 1, n):
                sim = similarity_fn(solutions[i], solutions[j])
                D[i, j] = D[j, i] = sim
        np.fill_diagonal(D, 1.0)
    else:
        # Sparse random sampling — approximate
        rng = np.random.default_rng(seed)
        counts = np.zeros(n, dtype=int)
        for _ in range(max_pairs):
            i, j = rng.choice(n, size=2, replace=False)
            if D[i, j] == 0:
                sim = similarity_fn(solutions[i], solutions[j])
                D[i, j] = D[j, i] = sim
                counts[i] += 1
                counts[j] += 1
        # Fill diagonal
        np.fill_diagonal(D, 1.0)

    return D


def compute_density_scores(
    solutions: List[List],
    similarity_fn: Callable,
    max_pairs: int = 2000,
) -> np.ndarray:
    """
    D_i = mean pairwise similarity of state i to all other states.
    High D_i → state is in a dense cluster → should be penalised.
    Returns normalised array in [0, 1].
    """
    n = len(solutions)
    if n == 1:
        return np.array([0.0])

    D = compute_density_matrix(solutions, similarity_fn, max_pairs=max_pairs)
    # Mean excluding self (diagonal = 1)
    scores = (D.sum(axis=1) - 1.0) / max(n - 1, 1)

    # Normalise to [0, 1]
    lo, hi = scores.min(), scores.max()
    if hi > lo:
        scores = (scores - lo) / (hi - lo)
    return scores
