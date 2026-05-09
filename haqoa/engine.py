"""
HAQOA Core Engine — AQSE-v1
Adaptive Quantum-Inspired State Evolution Engine

Mathematical model:
    |Ψ_t⟩  = Σ α_i |s_i⟩               (state superposition)
    P_i    = softmax(β · S_i)           (probability amplitudes)
    H_t    = -Σ P_i log P_i             (search entropy)
    β_t    = β_0 (1 + κ H_t)           (adaptive amplification)
    R(s_i) = w1·Q_i − w2·C_i − w3·E_i  (AI reward signal)
    C(s_i) = 1 if P_i > θ_t, else 0    (collapse gate)
"""

from __future__ import annotations

import time
import random
import math
import copy
import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class QuantumState:
    """
    A single candidate solution with its probability amplitude metadata.

    Attributes:
        solution:    The encoded solution (e.g. TSP route permutation).
        quality:     Raw objective value (lower = better for minimisation).
        score:       Normalised quality score ∈ [0, 1] (higher = better).
        amplitude:   Probability amplitude α_i.
        probability: Observation probability P_i.
        entropy_contrib: State's entropy contribution.
        age:         Generations since creation (for diversity tracking).
    """
    solution: List[Any]
    quality: float = float("inf")
    score: float = 0.0
    amplitude: float = 0.0
    probability: float = 0.0
    entropy_contrib: float = 0.0
    age: int = 0

    def copy(self) -> "QuantumState":
        s = QuantumState(
            solution=list(self.solution),
            quality=self.quality,
            score=self.score,
            amplitude=self.amplitude,
            probability=self.probability,
            entropy_contrib=self.entropy_contrib,
            age=self.age,
        )
        return s


@dataclass
class HAQOAConfig:
    """
    Full hyper-parameter configuration for HAQOA.
    All defaults follow the blueprint spec.
    """
    # Population
    population_size: int = 50          # |Ψ| — number of parallel states
    max_iterations: int = 500          # T_max — total evolution steps

    # Amplification (β dynamics)
    beta_base: float = 1.0             # β_0 — base softmax temperature
    beta_max: float = 10.0             # β ceiling
    entropy_sensitivity: float = 0.5  # κ — entropy→β coupling

    # Collapse gate
    collapse_threshold_base: float = 0.01   # θ_0 — base collapse threshold
    collapse_percentile: float = 20.0       # collapse bottom-X% by probability
    collapse_interval: int = 20             # apply collapse every N iterations

    # Evolution
    mutation_rate: float = 0.3         # per-state mutation probability
    crossover_rate: float = 0.6        # crossover probability
    elite_fraction: float = 0.1        # fraction of elites preserved unchanged

    # Entropy damping (prevents oscillation)
    entropy_damping: float = 0.1       # ε — smoothing coefficient
    diversity_floor: float = 0.3       # minimum target entropy ratio

    # AI guidance weights  R(s) = w1·Q − w2·C − w3·E
    w_quality: float = 0.70
    w_cost: float = 0.15
    w_entropy_penalty: float = 0.15

    # Stopping
    stagnation_limit: int = 80         # stop if no improvement for N iters
    target_quality: Optional[float] = None  # stop if quality reached

    # Reproducibility
    seed: Optional[int] = 42


@dataclass
class IterationRecord:
    """Snapshot of one generation for experiment logging."""
    iteration: int
    best_quality: float
    mean_quality: float
    entropy: float
    beta: float
    population_size: int
    elapsed_seconds: float
    collapsed: int = 0
    generated: int = 0


@dataclass
class HAQOAResult:
    """Full result record returned after optimisation."""
    best_solution: List[Any]
    best_quality: float
    iterations_run: int
    elapsed_seconds: float
    history: List[IterationRecord]
    config: HAQOAConfig
    termination_reason: str = "max_iterations"

    @property
    def convergence_curve(self) -> np.ndarray:
        return np.array([r.best_quality for r in self.history])

    @property
    def entropy_curve(self) -> np.ndarray:
        return np.array([r.entropy for r in self.history])


# ─── Core Engine ─────────────────────────────────────────────────────────────

class HAQOAEngine:
    """
    AQSE-v1: Adaptive Quantum-Inspired State Evolution Engine.

    Usage:
        engine = HAQOAEngine(config, objective_fn, mutation_fn, crossover_fn)
        result = engine.run(initial_solutions)
    """

    def __init__(
        self,
        config: HAQOAConfig,
        objective_fn: Callable[[List[Any]], float],
        mutation_fn: Callable[[List[Any]], List[Any]],
        crossover_fn: Callable[[List[Any], List[Any]], List[Any]],
        random_fn: Optional[Callable[[], List[Any]]] = None,
    ):
        """
        Args:
            config:        HAQOAConfig instance.
            objective_fn:  f(solution) → float  (minimisation).
            mutation_fn:   f(solution) → mutated_solution.
            crossover_fn:  f(sol_a, sol_b) → child_solution.
            random_fn:     f() → random_solution  (for state regeneration).
        """
        self.cfg = config
        self.obj = objective_fn
        self.mutate = mutation_fn
        self.crossover = crossover_fn
        self.random_sol = random_fn
        self._local_search = None  # optional; set via .set_local_search()

        self._rng = np.random.default_rng(config.seed)
        if config.seed is not None:
            random.seed(config.seed)

        # Runtime state
        self.population: List[QuantumState] = []
        self.best_state: Optional[QuantumState] = None
        self.history: List[IterationRecord] = []
        self._beta: float = config.beta_base
        self._entropy: float = 0.0
        self._prev_entropy: float = 0.0
        self._stagnation_counter: int = 0

    def set_local_search(self, fn):
        """Register an optional local-search polish function f(solution) → solution."""
        self._local_search = fn

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, initial_solutions: Optional[List[List[Any]]] = None) -> HAQOAResult:
        """
        Execute the full HAQOA optimisation loop.

        Args:
            initial_solutions: Seed population. If None and random_fn provided,
                                population is generated randomly.
        Returns:
            HAQOAResult with best solution, full history, and diagnostics.
        """
        t_start = time.perf_counter()
        self._initialise_population(initial_solutions)

        termination = "max_iterations"

        for iteration in range(self.cfg.max_iterations):
            # 1. Evaluate all states
            self._evaluate_population()

            # 2. Compute probabilities with adaptive amplification
            self._update_probabilities()

            # 3. Track entropy
            self._update_entropy()

            # 4. Record history
            elapsed = time.perf_counter() - t_start
            record = self._snapshot(iteration, elapsed)
            self.history.append(record)

            # 5. Log progress
            if iteration % 50 == 0:
                logger.info(
                    f"[{iteration:4d}] best={self.best_state.quality:.4f}  "
                    f"H={self._entropy:.4f}  β={self._beta:.3f}  "
                    f"pop={len(self.population)}"
                )

            # 6. Collapse weak states (periodic)
            n_collapsed = 0
            if (iteration + 1) % self.cfg.collapse_interval == 0:
                n_collapsed = self._collapse_weak_states()
                record.collapsed = n_collapsed

            # 7. Evolution — generate new states
            n_gen = self._evolve_population()
            record.generated = n_gen

            # 8. Adaptive β update
            self._adapt_beta()

            # 9. Local search polish on best state every 50 iters
            if iteration > 0 and iteration % 50 == 0 and hasattr(self, "_local_search"):
                polished = self._local_search(list(self.best_state.solution))
                polished_q = self.obj(polished)
                if polished_q < self.best_state.quality:
                    self.best_state.solution = polished
                    self.best_state.quality = polished_q
                    # Inject polished into population
                    self.population[0].solution = polished
                    self.population[0].quality = polished_q

            # 10. Stagnation check
            if self.best_state.quality == getattr(self, "_prev_best", float("inf")):
                self._stagnation_counter += 1
            else:
                self._stagnation_counter = 0
                self._prev_best = self.best_state.quality

            if self._stagnation_counter >= self.cfg.stagnation_limit:
                termination = "stagnation"
                logger.info(f"Stagnation stop at iteration {iteration}.")
                break

            if self.cfg.target_quality is not None:
                if self.best_state.quality <= self.cfg.target_quality:
                    termination = "target_reached"
                    break

        elapsed_total = time.perf_counter() - t_start
        return HAQOAResult(
            best_solution=list(self.best_state.solution),
            best_quality=self.best_state.quality,
            iterations_run=len(self.history),
            elapsed_seconds=elapsed_total,
            history=self.history,
            config=self.cfg,
            termination_reason=termination,
        )

    # ── Initialisation ────────────────────────────────────────────────────────

    def _initialise_population(self, seeds: Optional[List[List[Any]]]):
        """Build initial superposition from seeds + random fill."""
        self.population = []
        if seeds:
            for sol in seeds:
                self.population.append(QuantumState(solution=list(sol)))

        while len(self.population) < self.cfg.population_size:
            if self.random_sol:
                sol = self.random_sol()
            elif self.population:
                # Mutate a random existing state if no random_fn
                parent = random.choice(self.population).solution
                sol = self.mutate(parent)
            else:
                raise ValueError("Provide initial_solutions or a random_fn.")
            self.population.append(QuantumState(solution=sol))

        # Evaluate immediately so best_state is available
        self._evaluate_population()
        self._update_probabilities()

    # ── Evaluation & Scoring ──────────────────────────────────────────────────

    def _evaluate_population(self):
        """
        Evaluate objective for all states and update best_state.
        Uses rank-based scoring (fitness proportionate to rank, not raw value).
        This gives consistent selection pressure regardless of quality spread.
        Score_i = (N - rank_i) / N  ∈ (0, 1]  — higher rank = worse = lower score
        """
        qualities = []
        for state in self.population:
            q = self.obj(state.solution)
            state.quality = q
            state.age += 1
            qualities.append(q)

        q_arr = np.array(qualities)

        # Rank-based scoring: rank 0 = best quality → score = 1.0
        ranks = np.argsort(np.argsort(q_arr))   # 0 = best
        n = len(self.population)
        for state, rank in zip(self.population, ranks):
            # Non-linear rank scoring: emphasise top solutions
            state.score = ((n - rank) / n) ** 2   # squared → more contrast

        # Track global best
        best_now = min(self.population, key=lambda s: s.quality)
        if self.best_state is None or best_now.quality < self.best_state.quality:
            self.best_state = best_now.copy()

    # ── Probability Amplification ─────────────────────────────────────────────

    def _update_probabilities(self):
        """
        Compute softmax probabilities with adaptive β.
        P_i = exp(β · S_i) / Σ_j exp(β · S_j)
        """
        scores = np.array([s.score for s in self.population])

        # Numerically stable softmax
        shifted = self._beta * scores
        shifted -= shifted.max()
        exps = np.exp(shifted)
        probs = exps / exps.sum()

        for state, p in zip(self.population, probs):
            state.probability = float(p)
            state.amplitude = float(np.sqrt(p))  # |α_i|^2 = P_i

    # ── Entropy Monitoring ────────────────────────────────────────────────────

    def _update_entropy(self):
        """
        H_t = -Σ P_i log(P_i)
        With entropy damping to avoid oscillation:
          H_t ← (1-ε)·H_t + ε·H_{t-1}
        """
        probs = np.array([s.probability for s in self.population])
        # Clip to avoid log(0)
        probs = np.clip(probs, 1e-12, 1.0)
        raw_entropy = float(-np.sum(probs * np.log(probs)))

        # Damped entropy
        ε = self.cfg.entropy_damping
        self._entropy = (1 - ε) * raw_entropy + ε * self._prev_entropy

        for state, p in zip(self.population, probs):
            state.entropy_contrib = float(-p * np.log(p))

        self._prev_entropy = self._entropy

    # ── Adaptive Amplification ────────────────────────────────────────────────

    def _adapt_beta(self):
        """
        β_t = β_0 · (1 + κ · H_t)
        Clipped to [β_0, β_max].
        """
        self._beta = self.cfg.beta_base * (1.0 + self.cfg.entropy_sensitivity * self._entropy)
        self._beta = min(self._beta, self.cfg.beta_max)

    # ── Collapse Gate ─────────────────────────────────────────────────────────

    def _collapse_weak_states(self) -> int:
        """
        Remove states with P_i below the adaptive collapse threshold θ_t.
        θ_t is set as the p-th percentile of current probabilities.
        Always preserve at least 10% of population and all elites.
        Returns number of states collapsed.
        """
        if len(self.population) <= max(5, int(self.cfg.population_size * 0.1)):
            return 0

        probs = np.array([s.probability for s in self.population])
        theta = float(np.percentile(probs, self.cfg.collapse_percentile))

        # Dynamic floor: never remove more than 30% in one pass
        max_remove = int(0.30 * len(self.population))
        n_elite = max(1, int(self.cfg.elite_fraction * len(self.population)))

        # Sort by probability descending; always keep top-n_elite
        sorted_pop = sorted(self.population, key=lambda s: s.probability, reverse=True)
        survivors = sorted_pop[:n_elite]  # elites always survive

        n_collapsed = 0
        for state in sorted_pop[n_elite:]:
            if state.probability < theta and n_collapsed < max_remove:
                n_collapsed += 1
            else:
                survivors.append(state)

        self.population = survivors
        return n_collapsed

    # ── Evolution Engine ──────────────────────────────────────────────────────

    def _evolve_population(self) -> int:
        """
        Generate new states to refill population to target size.
        Strategy:
          1. Select parents by probability (weighted sampling).
          2. Apply crossover.
          3. Apply mutation.
          4. Optionally inject random states for diversity.
        Returns number of new states created.
        """
        target = self.cfg.population_size
        n_needed = target - len(self.population)
        if n_needed <= 0:
            return 0

        probs = np.array([s.probability for s in self.population])
        probs /= probs.sum()

        new_states = []
        diversity_inject = max(1, int(0.10 * n_needed))  # 10% random injections

        for i in range(n_needed):
            if i < diversity_inject and self.random_sol is not None:
                # Diversity injection: pure random state
                sol = self.random_sol()
            else:
                # Select two parents by probability
                idx_a, idx_b = self._rng.choice(
                    len(self.population), size=2, replace=False, p=probs
                )
                sol_a = self.population[idx_a].solution
                sol_b = self.population[idx_b].solution

                # Crossover
                if random.random() < self.cfg.crossover_rate:
                    child = self.crossover(sol_a, sol_b)
                else:
                    child = list(sol_a)

                # Mutation
                if random.random() < self.cfg.mutation_rate:
                    child = self.mutate(child)

                sol = child

            new_states.append(QuantumState(solution=sol))

        self.population.extend(new_states)
        return len(new_states)

    # ── Logging Helpers ───────────────────────────────────────────────────────

    def _snapshot(self, iteration: int, elapsed: float) -> IterationRecord:
        qualities = [s.quality for s in self.population]
        return IterationRecord(
            iteration=iteration,
            best_quality=self.best_state.quality,
            mean_quality=float(np.mean(qualities)),
            entropy=self._entropy,
            beta=self._beta,
            population_size=len(self.population),
            elapsed_seconds=elapsed,
        )
