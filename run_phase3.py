"""
HAQOA Phase 3 — Multi-Run Statistical Validation
Runs each algorithm N times with different seeds, then performs:
  - Bootstrap confidence intervals
  - Wilcoxon signed-rank tests (pairwise vs HAQOA)
  - Friedman test (global)
  - Average rank / critical difference
  - Full visualization dashboard

Usage:
    python run_phase3.py                          # default: small, 15 runs
    python run_phase3.py --instance medium --runs 30 --iters 300
    python run_phase3.py --instance small  --runs 5  --iters 200  --pop 40
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from haqoa.engine import HAQOAEngine, HAQOAConfig
from haqoa.operators import make_tsp_objective, make_tsp_operators, adaptive_mutation
from haqoa.problems.tsp import BENCHMARK_SUITE, TSPInstance
from haqoa.baselines.algorithms import (
    GeneticAlgorithm, SimulatedAnnealing,
    DiscreteParticleSwarm, AntColonyOptimisation,
)
from haqoa.metrics import (
    phase3_report,
    critical_difference_ranks,
)
from haqoa.visualization.plots import (
    plot_boxplots,
    plot_violin,
    plot_convergence_bands,
    plot_critical_difference,
    plot_significance_heatmap,
    plot_phase3_dashboard,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Single HAQOA run ─────────────────────────────────────────────────────────

def _run_haqoa(tsp: TSPInstance, max_iterations: int,
               population_size: int, seed: int) -> dict:
    cfg = HAQOAConfig(
        population_size=population_size,
        max_iterations=max_iterations,
        beta_base=0.5, beta_max=15.0,
        entropy_sensitivity=0.8,
        collapse_threshold_base=0.005,
        collapse_percentile=15.0,
        collapse_interval=25,
        mutation_rate=0.45,
        crossover_rate=0.60,
        elite_fraction=0.08,
        entropy_damping=0.05,
        w_quality=0.70, w_cost=0.15, w_entropy_penalty=0.15,
        stagnation_limit=max_iterations,
        seed=seed,
    )
    obj_fn        = make_tsp_objective(tsp)
    mut_fn, cx_fn = make_tsp_operators("advanced")
    engine        = HAQOAEngine(cfg, obj_fn, mut_fn, cx_fn, random_fn=tsp.random_route)
    engine.set_local_search(lambda r: tsp.two_opt_improve(r, max_iter=100))

    import random as rnd; rnd.seed(seed); np.random.seed(seed)
    greedy  = tsp.greedy_route()
    seeds   = [greedy] + [adaptive_mutation(greedy) for _ in range(population_size//5)]
    while len(seeds) < population_size:
        seeds.append(tsp.random_route())

    result = engine.run(initial_solutions=seeds[:population_size])
    return {
        "best_quality":    result.best_quality,
        "best_solution":   result.best_solution,
        "elapsed_seconds": result.elapsed_seconds,
        "history":         result.convergence_curve.tolist(),
    }


# ─── Single baseline run ──────────────────────────────────────────────────────

def _run_baseline(name: str, tsp: TSPInstance,
                  max_iterations: int, population_size: int, seed: int) -> dict:
    if name == "GA":
        algo = GeneticAlgorithm(tsp, population_size=population_size,
                                max_generations=max_iterations, seed=seed)
    elif name == "SA":
        algo = SimulatedAnnealing(tsp, max_iterations=max_iterations*100, seed=seed)
    elif name == "PSO":
        algo = DiscreteParticleSwarm(tsp, swarm_size=population_size,
                                     max_iterations=max_iterations, seed=seed)
    elif name == "ACO":
        algo = AntColonyOptimisation(tsp, n_ants=population_size//2,
                                     max_iterations=max_iterations, seed=seed)
    else:
        raise ValueError(f"Unknown baseline: {name}")
    return algo.run()


# ─── Main Phase-3 runner ──────────────────────────────────────────────────────

def run_phase3(
    instance_name: str = "small",
    n_runs: int = 15,
    max_iterations: int = 300,
    population_size: int = 50,
    base_seed: int = 0,
    out_dir: str = "results/phase3",
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    t_total = time.perf_counter()

    logger.info(f"═══ HAQOA Phase 3  |  instance={instance_name}  n_runs={n_runs} ═══")

    # ── Load TSP instance ─────────────────────────────────────────────────────
    gen_fn, kwargs = BENCHMARK_SUITE[instance_name]
    tsp = gen_fn(**kwargs, name=instance_name)
    ref_q = tsp.route_distance(tsp.two_opt_improve(tsp.greedy_route()))
    logger.info(f"Instance: {tsp}   2-opt ref: {ref_q:.3f}")

    algos    = ["HAQOA", "GA", "SA", "PSO", "ACO"]
    results  = {a: [] for a in algos}
    seeds    = [base_seed + i * 7 for i in range(n_runs)]

    # ── Multi-run loop ────────────────────────────────────────────────────────
    for run_idx, seed in enumerate(seeds):
        logger.info(f"  Run {run_idx+1:>2}/{n_runs}  seed={seed}")
        results["HAQOA"].append(
            _run_haqoa(tsp, max_iterations, population_size, seed)
        )
        for algo in ["GA", "SA", "PSO", "ACO"]:
            results[algo].append(
                _run_baseline(algo, tsp, max_iterations, population_size, seed)
            )

    elapsed = time.perf_counter() - t_total
    logger.info(f"All runs complete in {elapsed:.1f}s")

    # ── Statistical Report ────────────────────────────────────────────────────
    print("\n")
    report = phase3_report(results, reference="HAQOA", baseline_quality=ref_q)
    print(report)

    # Save report to text file
    report_path = f"{out_dir}/phase3_report_{instance_name}.txt"
    with open(report_path, "w") as f:
        f.write(report + "\n")
    logger.info(f"Report saved: {report_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    logger.info("Generating Phase 3 plots...")

    ranks = critical_difference_ranks(results)

    p = f"{out_dir}/boxplots_{instance_name}.png"
    plot_boxplots(results, title=f"Quality Distribution — {tsp.name}  ({n_runs} runs)",
                  save_path=p, baseline_quality=ref_q)
    logger.info(f"  {p}")

    p = f"{out_dir}/violin_{instance_name}.png"
    plot_violin(results, title=f"Quality Violin — {tsp.name}  ({n_runs} runs)",
                save_path=p, baseline_quality=ref_q)
    logger.info(f"  {p}")

    p = f"{out_dir}/convergence_bands_{instance_name}.png"
    plot_convergence_bands(results,
                           title=f"Convergence Mean ± Std — {tsp.name}  ({n_runs} runs)",
                           save_path=p)
    logger.info(f"  {p}")

    p = f"{out_dir}/cd_diagram_{instance_name}.png"
    plot_critical_difference(ranks, title=f"Average Rank — {tsp.name}", save_path=p)
    logger.info(f"  {p}")

    p = f"{out_dir}/significance_heatmap_{instance_name}.png"
    plot_significance_heatmap(results,
                              title=f"Pairwise Wilcoxon p-values — {tsp.name}",
                              save_path=p)
    logger.info(f"  {p}")

    p = f"{out_dir}/dashboard_{instance_name}.png"
    plot_phase3_dashboard(results, ranks, tsp_name=tsp.name,
                          baseline_quality=ref_q, save_path=p)
    logger.info(f"  {p}")

    logger.info(f"All Phase 3 outputs saved to ./{out_dir}/")
    return results, ranks


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HAQOA Phase 3 — Statistical Validation")
    parser.add_argument("--instance", default="small",
                        choices=list(BENCHMARK_SUITE.keys()))
    parser.add_argument("--runs",   type=int, default=15, help="Number of independent runs")
    parser.add_argument("--iters",  type=int, default=300, help="Max iterations per run")
    parser.add_argument("--pop",    type=int, default=50,  help="Population size")
    parser.add_argument("--seed",   type=int, default=0,   help="Base seed")
    parser.add_argument("--outdir", type=str, default="results/phase3")
    args = parser.parse_args()

    run_phase3(
        instance_name=args.instance,
        n_runs=args.runs,
        max_iterations=args.iters,
        population_size=args.pop,
        base_seed=args.seed,
        out_dir=args.outdir,
    )
