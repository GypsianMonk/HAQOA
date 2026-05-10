"""
HAQOA-X Core Engine — AQSE-v2
Hyper-Adaptive Quantum-Inspired Optimization Architecture

Full implementation of the Master Specification:
  - 5-component energy function (cost + density + risk + volatility + noise)
  - Entropy-regulated Boltzmann amplification
  - Dynamic collapse threshold: θ(t) = θ₀ + α(1 − H(t)/H_max)
  - Entropy-triggered regeneration: G(t) = ρ · σ_H / (σ_H + ε)
  - AI reward model with learning potential
  - Turbulence monitoring and damping
  - Multi-scale 3-layer search (Global / Regional / Local)
  - Similarity density field with pairwise penalty
"""

from __future__ import annotations

import time
import random
import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Dict, Any

import numpy as np

from haqoa.similarity import compute_density_scores, tsp_edge_jaccard
from haqoa.multi_scale import MultiScaleSearch

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Data Structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class QuantumStateX:
    """
    HAQOA-X candidate state.
    Carries the full information structure from Part 2.3 of the spec.
    """
    solution:          List[Any]
    quality:           float = float("inf")

    # Energy components (normalised to [0,1])
    energy:            float = 0.0
    energy_cost:       float = 0.0     # C_i — objective cost component
    energy_density:    float = 0.0     # D_i — similarity density
    energy_risk:       float = 0.0     # R_i — instability risk
    energy_volatility: float = 0.0     # V_i — quality volatility
    energy_noise:      float = 0.0     # N_i — stochastic noise

    # Probability field
    probability:       float = 0.0
    amplitude:         float = 0.0
    entropy_contrib:   float = 0.0

    # AI reward
    reward:            float = 0.0
    learning_potential: float = 0.0

    # Metadata
    age:          int = 0
    layer:        str = "init"         # 'global' | 'regional' | 'local' | 'init'
    quality_history: List[float] = field(default_factory=list)

    def copy(self) -> "QuantumStateX":
        s = QuantumStateX(
            solution=list(self.solution),
            quality=self.quality,
            energy=self.energy,
            energy_cost=self.energy_cost,
            energy_density=self.energy_density,
            energy_risk=self.energy_risk,
            energy_volatility=self.energy_volatility,
            energy_noise=self.energy_noise,
            probability=self.probability,
            amplitude=self.amplitude,
            entropy_contrib=self.entropy_contrib,
            reward=self.reward,
            learning_potential=self.learning_potential,
            age=self.age,
            layer=self.layer,
            quality_history=list(self.quality_history[-10:]),  # keep last 10
        )
        return s


@dataclass
class HAQOAXConfig:
    """
    HAQOA-X hyper-parameter configuration.
    Every parameter maps directly to the Master Specification.
    """
    # ── Population ────────────────────────────────────────────────────────────
    population_size:  int   = 60
    max_iterations:   int   = 500

    # ── Amplification (β dynamics, Part 5) ───────────────────────────────────
    beta_base:             float = 0.5    # β₀
    beta_max:              float = 20.0   # β ceiling
    entropy_sensitivity:   float = 0.9    # κ — entropy→β coupling

    # ── Entropy (Part 4) ──────────────────────────────────────────────────────
    entropy_damping:       float = 0.05   # μ — H'(t) = (1−μ)H + μH(t−1)
    h_max_window:          int   = 20     # window for H_max rolling estimate

    # ── 5-Component Energy weights (Part 3.3) ─────────────────────────────────
    # E_i = w1·C_i + w2·D_i + w3·R_i + w4·V_i + w5·N_i
    w_cost:       float = 0.45   # w1 — objective cost
    w_density:    float = 0.20   # w2 — similarity density penalty
    w_risk:       float = 0.15   # w3 — instability risk
    w_volatility: float = 0.10   # w4 — quality volatility
    w_noise:      float = 0.10   # w5 — stochastic noise

    # ── Dynamic Collapse (Part 7) ─────────────────────────────────────────────
    # θ(t) = θ₀ + α·(1 − H(t)/H_max)
    collapse_threshold_base: float = 0.005  # θ₀
    collapse_alpha:          float = 0.30   # α
    collapse_interval:       int   = 25
    collapse_min_survivors:  float = 0.30   # never collapse below 30% of pop

    # ── Regeneration (Part 8) ─────────────────────────────────────────────────
    # G(t) = ρ · σ_H(t) / (σ_H(t) + ε)
    regeneration_rate:    float = 0.20    # ρ
    regeneration_epsilon: float = 1e-3    # ε

    # ── AI Reward (Part 10) ───────────────────────────────────────────────────
    # R_i = γ₁·Q_i − γ₂·C_i − γ₃·V_i + γ₄·L_i
    gamma_quality:    float = 0.55   # γ₁
    gamma_cost:       float = 0.15   # γ₂
    gamma_volatility: float = 0.10   # γ₃
    gamma_learning:   float = 0.20   # γ₄

    # ── Multi-Scale Search (Part 9) ───────────────────────────────────────────
    multi_scale:          bool  = True
    global_mut_rate:      float = 0.55
    regional_mut_rate:    float = 0.30
    local_search_rate:    float = 0.20
    local_search_interval: int  = 30    # apply local polish every N iters

    # ── Similarity density ────────────────────────────────────────────────────
    similarity_fn:    str   = "edge_jaccard"   # 'edge_jaccard' | 'positional'
    density_max_pairs: int  = 1500

    # ── Evolution ────────────────────────────────────────────────────────────
    mutation_rate:    float = 0.40
    crossover_rate:   float = 0.65
    elite_fraction:   float = 0.08
    diversity_inject: float = 0.08    # fraction of new states that are random

    # ── Turbulence (Part 11) ─────────────────────────────────────────────────
    turbulence_threshold: float = 0.40

    # ── Stopping ──────────────────────────────────────────────────────────────
    stagnation_limit: int            = 9999    # set to max_iterations to disable
    target_quality:   Optional[float] = None

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed: Optional[int] = 42


@dataclass
class IterationRecordX:
    """Full per-iteration diagnostic snapshot for HAQOA-X."""
    iteration:      int
    best_quality:   float
    mean_quality:   float
    entropy:        float
    beta:           float
    turbulence:     float
    population_size: int
    elapsed_seconds: float
    # Energy component averages
    mean_energy_cost:      float = 0.0
    mean_energy_density:   float = 0.0
    mean_energy_risk:      float = 0.0
    mean_energy_volatility: float = 0.0
    # Layer counts
    n_global:    int = 0
    n_regional:  int = 0
    n_local:     int = 0
    n_collapsed: int = 0
    n_generated: int = 0
    # H_max tracking
    h_max: float = 0.0


@dataclass
class HAQOAXResult:
    """Complete result from one HAQOA-X run."""
    best_solution:      List[Any]
    best_quality:       float
    iterations_run:     int
    elapsed_seconds:    float
    history:            List[IterationRecordX]
    config:             HAQOAXConfig
    termination_reason: str = "max_iterations"

    @property
    def convergence_curve(self) -> np.ndarray:
        return np.array([r.best_quality for r in self.history])

    @property
    def entropy_curve(self) -> np.ndarray:
        return np.array([r.entropy for r in self.history])

    @property
    def turbulence_curve(self) -> np.ndarray:
        return np.array([r.turbulence for r in self.history])

    @property
    def beta_curve(self) -> np.ndarray:
        return np.array([r.beta for r in self.history])

    @property
    def energy_breakdown(self) -> Dict[str, np.ndarray]:
        return {
            "cost":       np.array([r.mean_energy_cost      for r in self.history]),
            "density":    np.array([r.mean_energy_density    for r in self.history]),
            "risk":       np.array([r.mean_energy_risk       for r in self.history]),
            "volatility": np.array([r.mean_energy_volatility for r in self.history]),
        }


# ══════════════════════════════════════════════════════════════════════════════
# HAQOA-X Engine
# ══════════════════════════════════════════════════════════════════════════════

class HAQOAXEngine:
    """
    AQSE-v2: Hyper-Adaptive Quantum-Inspired State Evolution Engine.

    Implements the full HAQOA-X Master Specification.

    Usage:
        engine = HAQOAXEngine(config, objective_fn, mutation_fn,
                              crossover_fn, random_fn, similarity_fn)
        result = engine.run(initial_solutions)
    """

    def __init__(
        self,
        config: HAQOAXConfig,
        objective_fn:  Callable[[List[Any]], float],
        mutation_fn:   Callable[[List[Any]], List[Any]],
        crossover_fn:  Callable[[List[Any], List[Any]], List[Any]],
        random_fn:     Optional[Callable[[], List[Any]]] = None,
        similarity_fn: Optional[Callable] = None,
    ):
        self.cfg        = config
        self.obj        = objective_fn
        self.mutate     = mutation_fn
        self.crossover  = crossover_fn
        self.random_sol = random_fn
        self._sim_fn    = similarity_fn or tsp_edge_jaccard
        self._local_search: Optional[Callable] = None

        # Seeds
        self._rng = np.random.default_rng(config.seed)
        if config.seed is not None:
            random.seed(config.seed)

        # Runtime state
        self.population:  List[QuantumStateX] = []
        self.best_state:  Optional[QuantumStateX] = None
        self.history:     List[IterationRecordX] = []

        self._beta:        float = config.beta_base
        self._entropy:     float = 0.0
        self._prev_entropy: float = 0.0
        self._h_max:       float = 1e-6
        self._h_history:   List[float] = []
        self._turbulence:  float = 0.0
        self._stagnation:  int = 0
        self._prev_best:   float = float("inf")

        # Multi-scale search engine
        self._ms: Optional[MultiScaleSearch] = None

    def set_local_search(self, fn: Callable):
        """Register an optional local-search polish f(solution) → solution."""
        self._local_search = fn

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def run(
        self,
        initial_solutions: Optional[List[List[Any]]] = None,
    ) -> HAQOAXResult:
        """Execute the full HAQOA-X optimisation loop."""
        t0 = time.perf_counter()
        self._initialise(initial_solutions)

        termination = "max_iterations"

        for iteration in range(self.cfg.max_iterations):
            # ── 1. Evaluate population ────────────────────────────────────────
            self._evaluate_population()

            # ── 2. Compute similarity density field ───────────────────────────
            density_scores = self._compute_density_field(iteration)

            # ── 3. Build 5-component energy ───────────────────────────────────
            self._compute_energy(density_scores)

            # ── 4. Boltzmann probabilities ────────────────────────────────────
            self._compute_probabilities()

            # ── 5. Entropy + turbulence ───────────────────────────────────────
            self._update_entropy()
            self._update_turbulence()

            # ── 6. Adaptive β ─────────────────────────────────────────────────
            self._adapt_beta()

            # ── 7. AI reward + learning potential ─────────────────────────────
            self._compute_ai_reward()

            # ── 8. Record snapshot ────────────────────────────────────────────
            record = self._snapshot(iteration, time.perf_counter() - t0)
            self.history.append(record)

            if iteration % 50 == 0:
                logger.info(
                    f"[{iteration:4d}] best={self.best_state.quality:.4f}  "
                    f"H={self._entropy:.4f}  β={self._beta:.3f}  "
                    f"T={self._turbulence:.4f}  pop={len(self.population)}"
                )

            # ── 9. Dynamic collapse ───────────────────────────────────────────
            n_collapsed = 0
            if (iteration + 1) % self.cfg.collapse_interval == 0:
                n_collapsed = self._dynamic_collapse()
                record.n_collapsed = n_collapsed

            # ── 10. Entropy-triggered regeneration ────────────────────────────
            n_regen = self._regenerate()
            record.n_generated += n_regen

            # ── 11. Multi-scale or standard evolution ─────────────────────────
            n_new, layer_counts = self._evolve()
            record.n_generated += n_new
            record.n_global    = layer_counts.get("global", 0)
            record.n_regional  = layer_counts.get("regional", 0)
            record.n_local     = layer_counts.get("local", 0)

            # ── 12. Periodic local polish ─────────────────────────────────────
            if (self._local_search is not None and
                    iteration > 0 and
                    iteration % self.cfg.local_search_interval == 0):
                self._local_polish()

            # ── 13. Stagnation / target checks ────────────────────────────────
            if self.best_state.quality < self._prev_best:
                self._stagnation = 0
                self._prev_best  = self.best_state.quality
            else:
                self._stagnation += 1

            if self._stagnation >= self.cfg.stagnation_limit:
                termination = "stagnation"; break

            if (self.cfg.target_quality is not None and
                    self.best_state.quality <= self.cfg.target_quality):
                termination = "target_reached"; break

        return HAQOAXResult(
            best_solution=list(self.best_state.solution),
            best_quality=self.best_state.quality,
            iterations_run=len(self.history),
            elapsed_seconds=time.perf_counter() - t0,
            history=self.history,
            config=self.cfg,
            termination_reason=termination,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Initialisation
    # ──────────────────────────────────────────────────────────────────────────

    def _initialise(self, seeds: Optional[List[List[Any]]]):
        if self.cfg.multi_scale:
            self._ms = MultiScaleSearch(
                mutation_fn=self.mutate,
                crossover_fn=self.crossover,
                local_search_fn=self._local_search,
                global_mut_rate=self.cfg.global_mut_rate,
                regional_mut_rate=self.cfg.regional_mut_rate,
                local_search_rate=self.cfg.local_search_rate,
                seed=self.cfg.seed or 0,
            )

        self.population = []
        if seeds:
            for sol in seeds:
                self.population.append(QuantumStateX(solution=list(sol), layer="init"))

        while len(self.population) < self.cfg.population_size:
            sol = (self.random_sol() if self.random_sol
                   else self.mutate(random.choice(self.population).solution))
            self.population.append(QuantumStateX(solution=sol, layer="init"))

        self._evaluate_population()
        real_density = self._compute_density_field()   # FIX BUG-4: use real density
        self._compute_energy(real_density)
        self._compute_probabilities()
        self._update_entropy()

    # ──────────────────────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────────────────────

    def _evaluate_population(self):
        """
        Evaluate objective for all states that are marked dirty (newly created).
        Unchanged survivors (age > 0, not dirty) skip re-evaluation.
        FIX BUG-6: avoids redundant objective calls on elite states.
        """
        for state in self.population:
            # Only re-evaluate if state was just created (age==0) or explicitly dirty
            if state.age == 0 or getattr(state, "_dirty", True):
                q = self.obj(state.solution)
                state.quality_history.append(q)
                if len(state.quality_history) > 15:
                    state.quality_history.pop(0)
                state.quality = q
                state._dirty = False
            state.age += 1

        best_now = min(self.population, key=lambda s: s.quality)
        if self.best_state is None or best_now.quality < self.best_state.quality:
            self.best_state = best_now.copy()

    # ──────────────────────────────────────────────────────────────────────────
    # Similarity Density Field  (Part 6)
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_density_field(self, iteration: int = 0) -> np.ndarray:
        """
        D_i = (1/N) Σ_j δ(s_i, s_j)
        Returns normalised density scores in [0,1].
        FIX BUG-8: passes iteration as seed for varied sparse sampling.
        """
        solutions = [s.solution for s in self.population]
        scores = compute_density_scores(
            solutions, self._sim_fn,
            max_pairs=self.cfg.density_max_pairs,
            seed=iteration,
        )
        for state, d in zip(self.population, scores):
            state.energy_density = float(d)
        return scores

    # ──────────────────────────────────────────────────────────────────────────
    # 5-Component Energy  (Part 3.3)
    # E_i = w1·C_i + w2·D_i + w3·R_i + w4·V_i + w5·N_i
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_energy(self, density_scores: np.ndarray):
        qualities = np.array([s.quality for s in self.population])
        q_min, q_max = qualities.min(), qualities.max()
        q_range = max(q_max - q_min, 1e-10)

        for i, state in enumerate(self.population):
            q = state.quality

            # C_i: normalised objective cost
            C = (q - q_min) / q_range

            # D_i: similarity density (already computed)
            D = density_scores[i]

            # R_i: instability risk — age-weighted quality deviation from population mean
            # Distinct from C_i (which is absolute cost); R_i captures
            # relative instability: old states with poor quality are high-risk.
            mean_q = float(qualities.mean())
            age_factor = min(state.age / max(1, self.cfg.max_iterations * 0.1), 1.0)
            R = min(abs(q - mean_q) / (q_range + 1e-10) * (0.5 + 0.5 * age_factor), 1.0)

            # V_i: volatility — std of recent quality history
            if len(state.quality_history) > 2:
                V = float(np.std(state.quality_history) / (q_range + 1e-10))
                V = min(V, 1.0)
            else:
                V = C * 0.5

            # N_i: stochastic noise (exploration perturbation)
            N = float(self._rng.uniform(0, 0.05))

            E = (self.cfg.w_cost * C +
                 self.cfg.w_density * D +
                 self.cfg.w_risk * R +
                 self.cfg.w_volatility * V +
                 self.cfg.w_noise * N)

            state.energy_cost      = C
            state.energy_density   = D
            state.energy_risk      = R
            state.energy_volatility = V
            state.energy_noise     = N
            state.energy           = E

    # ──────────────────────────────────────────────────────────────────────────
    # Boltzmann Probabilities  (Part 3.1)
    # P_i = exp(−β·E_i) / Σ_j exp(−β·E_j)
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_probabilities(self):
        energies = np.array([s.energy for s in self.population])
        # Boltzmann distribution (minimisation: lower energy → higher probability)
        logits = -self._beta * energies
        logits -= logits.max()   # numerical stability
        exps   = np.exp(logits)
        probs  = exps / exps.sum()

        for state, p in zip(self.population, probs):
            state.probability   = float(p)
            state.amplitude     = float(np.sqrt(p))

    # ──────────────────────────────────────────────────────────────────────────
    # Entropy  (Part 4)
    # H(t) = −Σ P_i log P_i,  H'(t) = (1−μ)H + μH(t−1)
    # ──────────────────────────────────────────────────────────────────────────

    def _update_entropy(self):
        probs = np.array([s.probability for s in self.population])
        probs = np.clip(probs, 1e-12, 1.0)
        raw_H = float(-np.sum(probs * np.log(probs)))

        # Damped entropy — save previous BEFORE overwriting (needed for turbulence)
        mu = self.cfg.entropy_damping
        self._prev_entropy = self._entropy                   # FIX BUG-1
        self._entropy = (1 - mu) * raw_H + mu * self._prev_entropy

        # Update per-state contribution
        for state, p in zip(self.population, probs):
            state.entropy_contrib = float(-p * np.log(p))

        # Rolling H_max
        self._h_history.append(self._entropy)
        if len(self._h_history) > self.cfg.h_max_window:
            self._h_history.pop(0)
        self._h_max = max(max(self._h_history), 1e-6)

    # ──────────────────────────────────────────────────────────────────────────
    # Turbulence  (Part 11)
    # T(t) = |H(t) − H(t−1)|
    # ──────────────────────────────────────────────────────────────────────────

    def _update_turbulence(self):
        self._turbulence = abs(self._entropy - self._prev_entropy)

    # ──────────────────────────────────────────────────────────────────────────
    # Adaptive Amplification  (Part 5)
    # β(t) = β₀ · (1 + κ · H(t)/H_max)
    # ──────────────────────────────────────────────────────────────────────────

    def _adapt_beta(self):
        h_ratio = self._entropy / self._h_max
        self._beta = self.cfg.beta_base * (1.0 + self.cfg.entropy_sensitivity * h_ratio)
        self._beta = min(self._beta, self.cfg.beta_max)

        # Turbulence mitigation: reduce β when chaotic
        if self._turbulence > self.cfg.turbulence_threshold:
            self._beta *= 0.85

    # ──────────────────────────────────────────────────────────────────────────
    # AI Reward Model  (Part 10)
    # R_i = γ₁·Q_i − γ₂·C_i − γ₃·V_i + γ₄·L_i
    # L_i = ΔQ_i / (Δt + ε)  (learning potential)
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_ai_reward(self):
        qualities = np.array([s.quality for s in self.population])
        q_min, q_max = qualities.min(), qualities.max()
        q_range = max(q_max - q_min, 1e-10)

        for state in self.population:
            # Q_i: normalised quality score (higher = better)
            Q = 1.0 - (state.quality - q_min) / q_range

            # L_i: learning potential — improvement rate over history
            if len(state.quality_history) >= 3:
                recent = state.quality_history[-3:]
                delta_q = max(0, recent[0] - recent[-1])   # improvement
                L = delta_q / (q_range + self.cfg.regeneration_epsilon)
            else:
                L = 0.0

            state.learning_potential = L
            state.reward = (
                self.cfg.gamma_quality  * Q
                - self.cfg.gamma_cost       * state.energy_cost
                - self.cfg.gamma_volatility * state.energy_volatility
                + self.cfg.gamma_learning   * L
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Dynamic Collapse  (Part 7)
    # θ(t) = θ₀ + α · (1 − H(t)/H_max)
    # ──────────────────────────────────────────────────────────────────────────

    def _dynamic_collapse(self) -> int:
        min_survivors = max(5, int(self.cfg.collapse_min_survivors * self.cfg.population_size))
        if len(self.population) <= min_survivors:
            return 0

        h_ratio = min(self._entropy / self._h_max, 1.0)
        theta = (self.cfg.collapse_threshold_base +
                 self.cfg.collapse_alpha * (1.0 - h_ratio))

        # Sort by energy ascending (lower energy = higher quality = keep)
        sorted_pop = sorted(self.population, key=lambda s: s.energy)
        n_elite    = max(1, int(self.cfg.elite_fraction * len(self.population)))

        survivors  = sorted_pop[:n_elite]   # elites always survive
        n_collapsed = 0
        max_remove  = len(self.population) - min_survivors

        # Use reward to decide marginal survivors: high-reward states get reprieve
        for state in sorted_pop[n_elite:]:
            should_collapse = (
                state.probability < theta
                and state.reward < 0.0         # AI says not worth keeping
                and n_collapsed < max_remove
            )
            if should_collapse:
                n_collapsed += 1
            else:
                survivors.append(state)

        self.population = survivors
        return n_collapsed

    # ──────────────────────────────────────────────────────────────────────────
    # Entropy-Triggered Regeneration  (Part 8)
    # G(t) = ρ · σ_H(t) / (σ_H(t) + ε)
    # ──────────────────────────────────────────────────────────────────────────

    def _regenerate(self) -> int:
        if len(self._h_history) < 3:
            return 0

        sigma_h = float(np.std(self._h_history[-10:]))
        G = (self.cfg.regeneration_rate * sigma_h /
             (sigma_h + self.cfg.regeneration_epsilon))

        n_new = int(G * self.cfg.population_size)
        if n_new <= 0 or self.random_sol is None:
            return 0

        for _ in range(n_new):
            sol = self.random_sol()
            ns = QuantumStateX(solution=sol, layer="regen")
            ns._dirty = True
            self.population.append(ns)

        # FIX BUG-7: cap population to prevent unbounded growth
        max_pop = int(self.cfg.population_size * 1.5)
        if len(self.population) > max_pop:
            # Keep best max_pop states by quality
            self.population.sort(key=lambda s: s.quality)
            self.population = self.population[:max_pop]

        return n_new

    # ──────────────────────────────────────────────────────────────────────────
    # Evolution  (Multi-scale or standard)  (Part 9)
    # ──────────────────────────────────────────────────────────────────────────

    def _evolve(self) -> Tuple[int, Dict[str, int]]:
        """Fill population to target size. Returns (n_new, layer_counts)."""
        target   = self.cfg.population_size
        n_needed = target - len(self.population)
        if n_needed <= 0:
            return 0, {}

        solutions = [s.solution for s in self.population]
        qualities = [s.quality  for s in self.population]
        probs     = np.array([s.probability for s in self.population])
        probs     /= probs.sum()

        layer_counts: Dict[str, int] = {}
        new_states: List[QuantumStateX] = []

        if self.cfg.multi_scale and self._ms is not None:
            h_ratio = self._entropy / max(self._h_max, 1e-6)
            apply_ls = (len(self.history) % self.cfg.local_search_interval == 0
                        and self._local_search is not None)
            offspring = self._ms.evolve(
                solutions, qualities,
                n_offspring=n_needed,
                entropy_ratio=min(h_ratio, 1.0),
                apply_local=apply_ls,
            )
            for sol, layer in offspring:
                new_states.append(QuantumStateX(solution=sol, layer=layer))
                layer_counts[layer] = layer_counts.get(layer, 0) + 1
        else:
            # Standard evolution fallback
            for i in range(n_needed):
                if (i < int(self.cfg.diversity_inject * n_needed)
                        and self.random_sol is not None):
                    sol = self.random_sol()
                    layer = "random"
                else:
                    ia = int(self._rng.choice(len(solutions), p=probs))
                    ib = int(self._rng.choice(len(solutions), p=probs))
                    sa, sb = solutions[ia], solutions[ib]
                    if random.random() < self.cfg.crossover_rate:
                        sol = self.crossover(sa, sb)
                    else:
                        sol = list(sa)
                    if random.random() < self.cfg.mutation_rate:
                        sol = self.mutate(sol)
                    layer = "standard"
                new_states.append(QuantumStateX(solution=sol, layer=layer))
                layer_counts[layer] = layer_counts.get(layer, 0) + 1

        for s in new_states:
            s._dirty = True
        self.population.extend(new_states)
        return len(new_states), layer_counts

    # ──────────────────────────────────────────────────────────────────────────
    # Local Polish
    # ──────────────────────────────────────────────────────────────────────────

    def _local_polish(self):
        """Apply local search to top-k states; update best if improved."""
        if self.cfg.multi_scale and self._ms is not None:
            solutions = [s.solution for s in self.population]
            qualities = [s.quality  for s in self.population]
            improved  = self._ms.local_polish(solutions, qualities, top_k=5)
            for idx, new_sol in improved:
                new_q = self.obj(new_sol)
                if new_q < self.population[idx].quality:
                    self.population[idx].solution = new_sol
                    self.population[idx].quality  = new_q
                    if new_q < self.best_state.quality:
                        self.best_state = self.population[idx].copy()
        elif self._local_search is not None:
            # Fallback: polish best state only
            polished = self._local_search(list(self.best_state.solution))
            pq       = self.obj(polished)
            if pq < self.best_state.quality:
                self.best_state.solution = polished
                self.best_state.quality  = pq
                self.population[0].solution = polished
                self.population[0].quality  = pq

    # ──────────────────────────────────────────────────────────────────────────
    # Snapshot
    # ──────────────────────────────────────────────────────────────────────────

    def _snapshot(self, iteration: int, elapsed: float) -> IterationRecordX:
        qualities = [s.quality for s in self.population]
        return IterationRecordX(
            iteration=iteration,
            best_quality=self.best_state.quality,
            mean_quality=float(np.mean(qualities)),
            entropy=self._entropy,
            beta=self._beta,
            turbulence=self._turbulence,
            population_size=len(self.population),
            elapsed_seconds=elapsed,
            mean_energy_cost=float(np.mean([s.energy_cost for s in self.population])),
            mean_energy_density=float(np.mean([s.energy_density for s in self.population])),
            mean_energy_risk=float(np.mean([s.energy_risk for s in self.population])),
            mean_energy_volatility=float(np.mean([s.energy_volatility for s in self.population])),
            h_max=self._h_max,
        )
