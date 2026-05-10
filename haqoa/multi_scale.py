"""
Multi-Scale Search Geometry — HAQOA-X  (Part 9 of Master Spec)

Three-layer hierarchical search:
  Global   — broad exploration: diverse crossover across full population
  Regional — cluster refinement: intra-cluster crossover + medium mutation
  Local    — precision exploitation: fine-grained 2-opt / or-opt polish

The three layers feed into each other:
  Global influences regional priorities.
  Regional influences local refinement targets.
"""

from __future__ import annotations

import random
from typing import List, Callable, Optional, Any, Tuple
import numpy as np


# ─── Cluster Utilities ────────────────────────────────────────────────────────

def _quality_clusters(
    solutions: List[List],
    qualities: List[float],
    n_clusters: int = 3,
) -> List[List[int]]:
    """
    Partition population into clusters by quality rank.
    Returns list of index lists: [elite_cluster, mid_cluster, weak_cluster].
    Simple rank-based partitioning — no distance metric needed.
    """
    n = len(qualities)
    sorted_idx = np.argsort(qualities).tolist()   # ascending: best first
    chunk = max(1, n // n_clusters)
    clusters = []
    for k in range(n_clusters):
        start = k * chunk
        end = start + chunk if k < n_clusters - 1 else n
        clusters.append(sorted_idx[start:end])
    return clusters


# ─── Multi-Scale Evolution ────────────────────────────────────────────────────

class MultiScaleSearch:
    """
    Hierarchical 3-layer evolution engine.

    Usage:
        ms = MultiScaleSearch(mutation_fn, crossover_fn, local_search_fn)
        new_solutions = ms.evolve(solutions, qualities, n_offspring, entropy_ratio)
    """

    def __init__(
        self,
        mutation_fn: Callable[[List], List],
        crossover_fn: Callable[[List, List], List],
        local_search_fn: Optional[Callable[[List], List]] = None,
        global_mut_rate: float = 0.55,
        regional_mut_rate: float = 0.30,
        local_search_rate: float = 0.20,
        seed: int = 42,
    ):
        self.mutate       = mutation_fn
        self.crossover    = crossover_fn
        self.local_search = local_search_fn
        self.gm_rate      = global_mut_rate
        self.rm_rate      = regional_mut_rate
        self.ls_rate      = local_search_rate
        self._rng         = np.random.default_rng(seed)
        random.seed(seed)

    def evolve(
        self,
        solutions: List[List],
        qualities: List[float],
        n_offspring: int,
        entropy_ratio: float = 0.5,    # H(t)/H_max ∈ [0,1]
        apply_local: bool = False,
    ) -> List[Tuple[List[Any], str]]:
        """
        Generate n_offspring new solutions across three layers.

        Args:
            solutions:     Current population solutions.
            qualities:     Corresponding objective values.
            n_offspring:   How many new solutions to generate.
            entropy_ratio: Current entropy as fraction of max.
                           High entropy → more global; low → more local.
            apply_local:   Apply local search to local-layer offspring.

        Returns:
            List of (solution, layer_label) tuples.
        """
        n = len(solutions)
        if n == 0:
            return []

        # ── Adaptive layer split based on entropy ─────────────────────────────
        # High entropy (exploration phase) → heavy global
        # Low entropy (convergence phase)  → heavy local
        g_frac = 0.30 + 0.40 * entropy_ratio          # 30–70% global
        l_frac = 0.10 + 0.40 * (1.0 - entropy_ratio)  # 10–50% local
        r_frac = max(0.05, 1.0 - g_frac - l_frac)     # FIX BUG-5: now used

        n_global   = max(1, int(n_offspring * g_frac))
        n_local    = max(1, int(n_offspring * l_frac))
        n_regional = max(0, int(n_offspring * r_frac))   # FIX BUG-5: use r_frac
        # Correct rounding: distribute any remainder to global layer
        remainder = n_offspring - n_global - n_local - n_regional
        n_global  = max(1, n_global + remainder)

        clusters = _quality_clusters(solutions, qualities, n_clusters=3)
        elite_idx, mid_idx, _ = clusters[0], clusters[1], clusters[2]  # FIX weak_idx unused

        offspring = []

        # ── Global Layer: full-population diverse crossover ────────────────────
        for _ in range(n_global):
            i = int(self._rng.choice(n))
            j = int(self._rng.choice(n))
            while j == i:
                j = int(self._rng.choice(n))
            child = self.crossover(solutions[i], solutions[j])
            if random.random() < self.gm_rate:
                child = self.mutate(child)
            offspring.append((child, "global"))

        # ── Regional Layer: intra-cluster elite×mid crossover ─────────────────
        for _ in range(n_regional):
            if elite_idx and mid_idx:
                ia = int(self._rng.choice(elite_idx))
                ib = int(self._rng.choice(mid_idx))
                child = self.crossover(solutions[ia], solutions[ib])
            elif elite_idx:
                ia = int(self._rng.choice(elite_idx))
                child = self.mutate(solutions[ia])
            else:
                child = self.mutate(solutions[int(self._rng.choice(n))])
            if random.random() < self.rm_rate:
                child = self.mutate(child)
            offspring.append((child, "regional"))

        # ── Local Layer: fine mutation on elites ──────────────────────────────
        for _ in range(n_local):
            src_idx = int(self._rng.choice(elite_idx)) if elite_idx else 0
            child = self.mutate(solutions[src_idx])
            if random.random() < self.rm_rate:
                child = self.mutate(child)
            if apply_local and self.local_search is not None:
                if random.random() < self.ls_rate:
                    child = self.local_search(child)
            offspring.append((child, "local"))

        return offspring[:n_offspring]

    def local_polish(
        self,
        solutions: List[List],
        qualities: List[float],
        top_k: int = 5,
    ) -> List[Tuple[int, List]]:
        """
        Apply local search to the top-k solutions.
        Returns [(index, improved_solution), ...] only where quality improved.
        """
        if self.local_search is None:
            return []
        n = len(solutions)
        top_k = min(top_k, n)
        elite_idx = np.argsort(qualities)[:top_k]
        results = []
        for idx in elite_idx:
            improved = self.local_search(solutions[idx])
            results.append((int(idx), improved))
        return results
