# HAQOA - Hybrid AI-Assisted Quantum-Inspired Optimization Architecture

## Research Implementation v1.0 - AQSE-v1

**Adaptive Quantum-Inspired State Evolution Engine**

---

## Overview

HAQOA is a research-oriented optimization framework designed to solve complex combinatorial and NP-hard problems using:

- **Quantum-inspired probabilistic state evolution**
- **Adaptive entropy-controlled exploration**
- **AI-guided search amplification**
- **Dynamic state collapse mechanisms**
- **Hybrid optimization intelligence**

This implementation follows the research blueprint exactly as specified, implementing all core components with mathematical rigor.

---

## Architecture Components

### 1. State Superposition Generator (`state_superposition.py`)
Maintains a probabilistic population of candidate states instead of single solutions.

**Mathematical Foundation:**
```
|Ψ_t⟩ = Σ α_i^(t) |s_i⟩
```

Where:
- `s_i` = candidate solution state
- `α_i` = probability amplitude
- `N` = number of active states

### 2. Probabilistic State Encoder (`probabilistic_encoding.py`)
Encodes states with probability distributions based on quality scores.

**Mathematical Foundation:**
```
P_i = e^(β * S_i) / Σ_j e^(β * S_j)
```

### 3. Entropy Monitoring System (`entropy_monitor.py`)
Monitors and regulates search entropy to balance exploration vs exploitation.

**Mathematical Foundation:**
```
H_t = -Σ P_i^(t) * log(P_i^(t))
```

### 4. Adaptive Probability Amplifier (`amplification.py`)
Dynamically adjusts probability amplification based on entropy.

**Mathematical Foundation:**
```
β_t = β_0 * (1 + κ * H_t)
```

### 5. Dynamic Collapse Controller (`collapse_controller.py`)
Periodically removes weak states to control combinatorial explosion.

**Mathematical Foundation:**
```
C(s_i) = 1 if P(s_i) > θ_t, else 0
```

### 6. AI Guidance Engine (`ai_guidance.py`)
Provides intelligent search direction through reward estimation.

**Reward Function:**
```
R(s_i) = w_1*Q_i - w_2*C_i - w_3*E_i
```

### 7. Evolution Engine (`evolution.py`)
Handles state regeneration, mutation, and probabilistic recombination.

### 8. Main Optimizer (`haqoa_engine.py`)
Integrates all components into the complete HAQOA framework.

---

## Installation

### Requirements
- Python 3.8+
- NumPy
- SciPy (optional for advanced features)

```bash
pip install numpy scipy
```

---

## Quick Start

### Solving TSP (Traveling Salesman Problem)

```python
from haqoa_engine import HAQOAOptimizer, HAQOAConfig
from tsp_solver import create_tsp_instance, TSPSolver

# Create TSP instance
tsp = create_tsp_instance(n_cities=20, seed=42)
solver = TSPSolver(tsp)

# Configure HAQOA
config = HAQOAConfig(
    initial_population_size=30,
    max_generations=200,
    random_seed=42
)

# Create optimizer
optimizer = HAQOAOptimizer(config=config)

# Set problem functions
optimizer.set_problem_functions(
    solution_generator=lambda: solver.generate_random_tour(),
    solution_mutator=lambda s: solver.mutate(s),
    solution_recombiner=lambda p1, p2: solver.recombine(p1, p2),
    quality_evaluator=lambda s: solver.compute_quality(s)
)

# Run optimization
result = optimizer.optimize(verbose=True)

print(f"Best distance: {result.best_solution.total_distance:.4f}")
```

### Command Line Example

```bash
python run_tsp_example.py --cities 20 --generations 200 --seed 42
```

---

## Project Structure

```
haqoa_project/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── state_superposition.py   # Component 1
│   ├── probabilistic_encoding.py # Component 2
│   ├── entropy_monitor.py       # Component 3
│   ├── amplification.py         # Component 4
│   ├── collapse_controller.py   # Component 5
│   ├── ai_guidance.py           # Component 6
│   ├── evolution.py             # Component 7
│   ├── haqoa_engine.py          # Main optimizer
│   └── tsp_solver.py            # TSP-specific solver
├── benchmarks/                   # Benchmark implementations
├── tests/                        # Unit tests
├── data/                         # Test datasets
├── results/                      # Experimental results
├── docs/                         # Documentation
├── run_tsp_example.py           # Example runner
└── README.md                    # This file
```

---

## Configuration Options

### HAQOAConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_population_size` | 50 | Initial number of states |
| `max_states` | 200 | Maximum states in superposition |
| `min_states_after_collapse` | 20 | Minimum states after collapse |
| `high_entropy_threshold` | 0.7 | Threshold for high entropy |
| `low_entropy_threshold` | 0.2 | Threshold for low entropy |
| `base_amplification` | 1.0 | Base amplification factor (β_0) |
| `entropy_sensitivity` | 0.5 | Entropy sensitivity (κ) |
| `collapse_interval` | 10 | Generations between collapses |
| `mutation_rate` | 0.1 | Base mutation rate |
| `recombination_probability` | 0.3 | Recombination probability |
| `max_generations` | 500 | Maximum generations |
| `max_time_seconds` | 300.0 | Time limit |
| `convergence_threshold` | 1e-6 | Convergence tolerance |

---

## Key Features

### Entropy-Controlled Exploration
- Automatic balance between exploration and exploitation
- Adaptive amplification based on entropy levels
- Convergence risk monitoring

### AI-Guided Search
- Multi-component reward system
- Adaptive weight adjustment
- Search pattern learning

### Dynamic Collapse
- Multiple collapse strategies (adaptive, threshold, percentile, age-based)
- Diversity preservation awareness
- Configurable triggers

### Evolutionary Operations
- Entropy-guided mutation rates
- Multiple crossover operators (OX1, PMX)
- Tournament/rank/probability selection

---

## Research Compliance

This implementation strictly follows the research blueprint:

✅ **No hype-driven claims**  
✅ **No unsupported exponential speedup claims**  
✅ **No fake "quantum supremacy" language**  
✅ **Scientific rigor prioritized**  
✅ **Mathematically defensible optimization**  

The objective is **NOT** to imitate quantum mechanics superficially, but to create adaptive, probabilistic optimization systems inspired by quantum concepts.

---

## Baseline Algorithms (For Comparison)

Per the research blueprint, HAQOA should be compared against:

- Genetic Algorithm (GA)
- Simulated Annealing (SA)
- Particle Swarm Optimization (PSO)
- Ant Colony Optimization (ACO)
- Quantum-Inspired Evolutionary Algorithms (QIEA)

*Baseline implementations planned for future releases.*

---

## Metrics

Critical metrics tracked:
- Convergence Speed
- Solution Quality
- Stability Variance
- Runtime Complexity
- Scalability
- Memory Consumption

---

## License

Research implementation - Open for academic use.

---

## Citation

If you use this implementation in your research, please cite:

```
HAQOA: Hybrid AI-Assisted Quantum-Inspired Optimization Architecture
Research Blueprint v1.0
AQSE-v1 Implementation
```

---

## Contact & Contributions

This is a research implementation. For questions or contributions related to the research direction, please refer to the original blueprint documentation.

---

## Version History

- **v1.0** - Initial AQSE-v1 implementation
  - All 7 core components implemented
  - TSP solver integration
  - Complete optimization loop
  - Statistical tracking
