# HAQOA Research Execution Report (Phase 2/3 Snapshot)

Date: 2026-05-09

## Objective

Execute an initial multi-seed benchmark study for HAQOA on TSP instances and compare against GA, SA, PSO, and ACO under a shared iteration/population budget.

## Protocol

- Instances: `small` (20 cities), `medium` (50 cities)
- Seeds: `11, 22, 33`
- Budget per run: `200` iterations, population `50`
- Reference: 2-opt route quality computed inside the experiment runner
- Command pattern used:
  - `run_experiment(instance_name=<instance>, max_iterations=200, population_size=50, seed=<seed>)`

## Raw Best-Quality Results

### Small (n=20)

| Seed | HAQOA | GA | SA | PSO | ACO |
|---:|---:|---:|---:|---:|---:|
| 11 | 341.578 | 428.252 | 341.578 | 342.868 | 341.578 |
| 22 | 342.868 | 341.578 | 341.578 | 341.578 | 341.578 |
| 33 | 342.868 | 366.957 | 341.578 | 341.578 | 341.578 |

### Medium (n=50)

| Seed | HAQOA | GA | SA | PSO | ACO |
|---:|---:|---:|---:|---:|---:|
| 11 | 365.284 | 662.358 | 360.677 | 736.463 | 372.935 |
| 22 | 365.284 | 569.675 | 363.563 | 689.131 | 368.840 |
| 33 | 365.284 | 512.811 | 410.858 | 835.548 | 364.414 |

## Aggregate Summary (lower is better)

### Small (n=20)

- Mean best quality:
  - HAQOA: **342.438**
  - GA: 378.929
  - SA: **341.578**
  - PSO: 342.008
  - ACO: **341.578**
- Observation: HAQOA is competitive, but SA/ACO are slightly stronger on this tiny instance under this budget.

### Medium (n=50)

- Mean best quality:
  - HAQOA: **365.284**
  - GA: 581.614
  - SA: 378.366
  - PSO: 753.714
  - ACO: **368.730**
- Observation: HAQOA strongly outperforms GA/PSO and is close to ACO/SA; SA and ACO can still win some seeds.

## Key Findings

1. HAQOA showed very strong robustness on `medium`: identical best-quality across all three seeds in this run.
2. GA and PSO were consistently weakest under this setup.
3. SA and ACO remain strong baselines and must be treated as primary comparators in future ablations.
4. Current evidence supports a **promising heuristic**, not a claim of universal dominance.

## Next Research Actions

1. Increase statistical power: run >= 30 seeds per instance.
2. Add significance testing (Wilcoxon signed-rank) between HAQOA and each baseline.
3. Add ablations:
   - disable entropy adaptation,
   - disable collapse,
   - disable local-search enhancement.
4. Expand benchmark set to `large` and structured synthetic variants.

