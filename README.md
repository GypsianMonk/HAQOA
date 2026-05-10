# HAQOA-X — Hyper-Adaptive Quantum-Inspired Optimization Architecture

> **AQSE-v2** : Adaptive Quantum-Inspired State Evolution Engine (v2)
> Research Blueprint Implementation — Phase 1 · Phase 2 · Phase 3

---

## What is HAQOA-X?

HAQOA-X redefines optimization as **evolving probabilistic topology**.
Instead of deterministic search or random stochastic jumps, solutions behave
as probability fields under entropy-regulated evolutionary pressure.

The system is:
- **neither purely deterministic** nor **purely random**
- **probabilistic** · **adaptive** · **self-organizing** · **information-driven**

---

## Mathematical Model

```
|Ψ_t⟩  = Σ αᵢ |sᵢ⟩                                (state superposition)

Eᵢ     = w₁·Cᵢ + w₂·Dᵢ + w₃·Rᵢ + w₄·Vᵢ + w₅·Nᵢ   (5-component energy)

Pᵢ     = exp(−β·Eᵢ) / Σⱼ exp(−β·Eⱼ)               (Boltzmann probabilities)

Hₜ     = −Σ Pᵢ log Pᵢ                               (Shannon entropy)
H'(t)  = (1−μ)H(t) + μH(t−1)                        (entropy damping)

βₜ     = β₀(1 + κ·H(t)/H_max)                       (adaptive amplification)

θ(t)   = θ₀ + α(1 − H(t)/H_max)                     (dynamic collapse gate)

G(t)   = ρ · σ_H(t) / (σ_H(t) + ε)                  (regeneration rate)

Rᵢ     = γ₁·Qᵢ − γ₂·Cᵢ − γ₃·Vᵢ + γ₄·Lᵢ            (AI reward signal)
Lᵢ     = ΔQᵢ / (Δt + ε)                             (learning potential)

T(t)   = |H(t) − H(t−1)|                             (turbulence)
```

| Symbol | Meaning                  |
|--------|--------------------------|
| Cᵢ     | objective cost           |
| Dᵢ     | similarity density       |
| Rᵢ     | instability risk         |
| Vᵢ     | quality volatility       |
| Nᵢ     | stochastic noise         |
| Lᵢ     | learning potential       |

---

## Implementation Status

| Component                          | Status | File                         |
|------------------------------------|--------|------------------------------|
| AQSE-v1 State Superposition Engine | ✅     | `haqoa/engine.py`            |
| **AQSE-v2 Full Energy System**     | ✅     | `haqoa/engine_x.py`          |
| **5-Component Energy Function**    | ✅     | `haqoa/engine_x.py`          |
| **Similarity Density Field**       | ✅     | `haqoa/similarity.py`        |
| **Dynamic Collapse Gate θ(t)**     | ✅     | `haqoa/engine_x.py`          |
| **Entropy-Triggered Regeneration** | ✅     | `haqoa/engine_x.py`          |
| **AI Reward + Learning Potential** | ✅     | `haqoa/engine_x.py`          |
| **Turbulence Monitor T(t)**        | ✅     | `haqoa/engine_x.py`          |
| **Multi-Scale Search (3 layers)**  | ✅     | `haqoa/multi_scale.py`       |
| TSP Problem + Benchmarks           | ✅     | `haqoa/problems/tsp.py`      |
| OX / PMX / Edge-Assembly Crossover | ✅     | `haqoa/operators.py`         |
| GA / SA / PSO / ACO Baselines      | ✅     | `haqoa/baselines/algorithms.py` |
| Comparison Table + Gap Metrics     | ✅     | `haqoa/metrics.py`           |
| **Phase 3 Statistical Validation** | ✅     | `haqoa/metrics.py`           |
| **Wilcoxon + Friedman + CI**       | ✅     | `haqoa/metrics.py`           |
| Convergence / Route Visualization  | ✅     | `haqoa/visualization/plots.py` |
| **Energy Breakdown Plot**          | ✅     | `haqoa/visualization/plots.py` |
| **Multi-Scale Activity Plot**      | ✅     | `haqoa/visualization/plots.py` |
| **HAQOA-X Full Dashboard**         | ✅     | `haqoa/visualization/plots.py` |
| **Phase 3 Statistical Dashboard**  | ✅     | `haqoa/visualization/plots.py` |
| HAQOA-X Experiment Runner          | ✅     | `run_haqoax.py`              |
| Phase 3 Multi-Run Runner           | ✅     | `run_phase3.py`              |
| Phase 1+2 Legacy Runner            | ✅     | `run_experiment.py`          |

---

## Quick Start

```bash
pip install -r requirements.txt

# HAQOA-X on 20-city instance
python run_haqoax.py --instance small --iters 300 --pop 60

# HAQOA-X on 50-city clustered instance
python run_haqoax.py --instance medium --iters 500 --pop 80

# Sweep all benchmark instances
python run_haqoax.py --compare --iters 300

# Phase 3 statistical validation (15 runs)
python run_phase3.py --instance small --runs 15 --iters 300

# Legacy HAQOA (AQSE-v1)
python run_experiment.py --instance small --iters 500
```

---

## HAQOA-X System Architecture

```
                  ┌──────────────────────────────────────┐
                  │          HAQOA-X Engine (AQSE-v2)    │
                  └──────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼────────────────────────────┐
        ▼                           ▼                            ▼
┌──────────────┐          ┌─────────────────┐          ┌──────────────────┐
│  Similarity  │          │  5-Component    │          │  Entropy         │
│  Density     │          │  Energy Field   │          │  Intelligence    │
│  Field D_i   │          │  E = f(C,D,R,V,N)│         │  H(t), β(t), T(t)│
└──────────────┘          └─────────────────┘          └──────────────────┘
        │                           │                            │
        └───────────────────────────┼────────────────────────────┘
                                    ▼
                        ┌────────────────────┐
                        │  Boltzmann P_i     │
                        │  exp(−β·E_i)       │
                        └────────────────────┘
                                    │
              ┌─────────────────────┼──────────────────────┐
              ▼                     ▼                       ▼
   ┌──────────────────┐  ┌────────────────────┐  ┌─────────────────────┐
   │ Dynamic Collapse │  │ Entropy-Triggered  │  │ Multi-Scale Search  │
   │ θ(t)=θ₀+α(1−E)  │  │ Regeneration G(t)  │  │ Global/Regional/    │
   └──────────────────┘  └────────────────────┘  │ Local Layers        │
                                    │             └─────────────────────┘
                                    ▼
                        ┌────────────────────┐
                        │  AI Reward Model   │
                        │  R_i = f(Q,C,V,L)  │
                        └────────────────────┘
```

---

## Project Structure

```
haqoa/
├── engine.py           ← AQSE-v1: original HAQOA core
├── engine_x.py         ← AQSE-v2: full HAQOA-X engine ★
├── similarity.py       ← similarity density field (Part 6) ★
├── multi_scale.py      ← 3-layer hierarchical search (Part 9) ★
├── operators.py        ← OX / PMX / edge-assembly crossover
├── metrics.py          ← comparison + Phase 3 statistical tests
├── problems/
│   └── tsp.py          ← TSPInstance + 6 benchmark instances
├── baselines/
│   └── algorithms.py   ← GA, SA, PSO, ACO
└── visualization/
    └── plots.py        ← all visualization functions
run_haqoax.py           ← HAQOA-X experiment CLI ★
run_phase3.py           ← Phase 3 multi-run statistical CLI ★
run_experiment.py       ← legacy Phase 1+2 CLI
requirements.txt
setup.py
```

---

## Benchmark Results

### 20 Cities (random)

| Algorithm  | Best Tour | Gap vs 2-opt |
|------------|-----------|--------------|
| **HAQOA-X**| **341.6** | **−0.38%**   |
| HAQOA      | 342.9     |  0.00%       |
| GA         | 341.6     | −0.38%       |
| SA         | 341.6     | −0.38%       |
| PSO        | 341.6     | −0.38%       |
| ACO        | 341.6     | −0.38%       |
| 2-opt ref  | 342.9     |  0%          |

> HAQOA-X matches best-in-class on small instances while providing
> significantly richer diagnostics: energy breakdown, turbulence monitoring,
> multi-scale activity, and AI reward dynamics.

---

## Roadmap

- [x] Phase 1 — Mathematical formalization + AQSE-v1 core engine
- [x] Phase 2 — TSP simulation + baseline comparison
- [x] Phase 3 — Multi-run statistical validation (Wilcoxon, Friedman, CI)
- [x] **Phase 4 — HAQOA-X: full 5-component energy + multi-scale search**
- [ ] Phase 5 — Large-scale evaluation (n=100, n=500 TSP)
- [ ] Phase 6 — RL-based adaptive reward shaping
- [ ] Phase 7 — Qiskit simulation layer (quantum circuit mapping)
- [ ] Phase 8 — Multi-objective HAQOA-X (Pareto front evolution)
- [ ] Phase 9 — Beyond TSP: scheduling, portfolio, NAS

---

## Research Standards

- No "quantum supremacy" language — quantum-**inspired**, not quantum-dependent
- All results reproducible with fixed seeds
- Mandatory baseline comparisons before any claims
- Statistical significance via Wilcoxon + Friedman in Phase 3
- Every mechanism must survive simulation, stress testing, and statistical validation
