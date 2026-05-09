"""
HAQOA Experiment Runner
Runs HAQOA + all baselines on TSP benchmarks and produces comparison report + plots.

Usage:
    python run_experiment.py
    python run_experiment.py --instance small --iters 300 --pop 40
    python run_experiment.py --instance medium --iters 500 --pop 50
"""

from __future__ import annotations

import argparse
import logging
import sys
import os
from pathlib import Path

import numpy as np

# ── Ensure package is importable from this directory ─────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from haqoa.engine import HAQOAEngine, HAQOAConfig
from haqoa.operators import make_tsp_objective, make_tsp_operators, adaptive_mutation
from haqoa.problems.tsp import BENCHMARK_SUITE, TSPInstance
from haqoa.baselines.algorithms import run_all_baselines
from haqoa.metrics import comparison_table, convergence_speed, compute_gap
from haqoa.visualization.plots import (
    plot_convergence,
    plot_entropy_dynamics,
    plot_routes,
    plot_quality_bars,
    plot_population_heatmap,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── TSP Adapter ─────────────────────────────────────────────────────────────

def build_haqoa_for_tsp(
    tsp: TSPInstance,
    cfg: HAQOAConfig,
    operator_strategy: str = "advanced",
) -> HAQOAEngine:
    """Wire a TSP instance into the HAQOA engine."""
    obj_fn           = make_tsp_objective(tsp)
    mut_fn, cx_fn    = make_tsp_operators(operator_strategy)
    random_fn        = tsp.random_route

    return HAQOAEngine(
        config=cfg,
        objective_fn=obj_fn,
        mutation_fn=mut_fn,
        crossover_fn=cx_fn,
        random_fn=random_fn,
    )


# ─── Seed population builder ─────────────────────────────────────────────────

def build_seed_population(tsp: TSPInstance, n: int, seed: int = 42) -> list:
    """
    Mix of:
      - 1 greedy route (nearest-neighbour)
      - rest random (70%) + or-opt mutations of greedy (30%)
    Not including 2-opt improved — keep more diversity.
    """
    import random as rnd
    rnd.seed(seed)
    seeds = []

    greedy = tsp.greedy_route()
    seeds.append(greedy)

    n_mutated = max(1, n // 5)
    for _ in range(n_mutated):
        mut = adaptive_mutation(greedy)
        seeds.append(mut)

    while len(seeds) < n:
        seeds.append(tsp.random_route())

    return seeds[:n]


# ─── Main Experiment ──────────────────────────────────────────────────────────

def run_experiment(
    instance_name: str = "small",
    max_iterations: int = 300,
    population_size: int = 50,
    seed: int = 42,
    out_dir: str = "results",
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"═══ HAQOA Experiment: instance={instance_name} ═══")

    # ── Load TSP instance ─────────────────────────────────────────────────────
    gen_fn, kwargs = BENCHMARK_SUITE[instance_name]
    tsp = gen_fn(**kwargs, name=instance_name)
    logger.info(f"Instance: {tsp}  (n={tsp.n} cities)")

    # Greedy baseline quality
    greedy_route   = tsp.greedy_route()
    greedy_quality = tsp.route_distance(greedy_route)
    improved_route = tsp.two_opt_improve(greedy_route)
    improved_q     = tsp.route_distance(improved_route)
    logger.info(f"Greedy NN quality   : {greedy_quality:.3f}")
    logger.info(f"2-opt improved      : {improved_q:.3f}")

    # ── HAQOA config ──────────────────────────────────────────────────────────
    cfg = HAQOAConfig(
        population_size=population_size,
        max_iterations=max_iterations,
        # Keep β low initially → broad exploration; rises as entropy falls
        beta_base=0.5,
        beta_max=15.0,
        entropy_sensitivity=0.8,     # stronger entropy→β coupling
        collapse_threshold_base=0.005,
        collapse_percentile=15.0,    # less aggressive collapse = more diversity
        collapse_interval=25,
        mutation_rate=0.45,          # higher mutation for escape from local optima
        crossover_rate=0.60,
        elite_fraction=0.08,
        entropy_damping=0.05,        # faster entropy response
        w_quality=0.70,
        w_cost=0.15,
        w_entropy_penalty=0.15,
        stagnation_limit=max_iterations,  # disable stagnation stop — run full budget
        seed=seed,
    )

    # ── Run HAQOA ─────────────────────────────────────────────────────────────
    logger.info("Running HAQOA...")
    engine = build_haqoa_for_tsp(tsp, cfg, operator_strategy="advanced")
    engine.set_local_search(lambda r: tsp.two_opt_improve(r, max_iter=100))
    seeds  = build_seed_population(tsp, n=cfg.population_size, seed=seed)
    result = engine.run(initial_solutions=seeds)

    logger.info(
        f"HAQOA finished: best={result.best_quality:.3f}  "
        f"iters={result.iterations_run}  "
        f"time={result.elapsed_seconds:.2f}s  "
        f"reason={result.termination_reason}"
    )

    # ── Run Baselines ─────────────────────────────────────────────────────────
    logger.info("Running baselines...")
    baseline_results = run_all_baselines(
        tsp,
        max_iterations=max_iterations,
        population_size=population_size,
        seed=seed,
    )

    # ── Aggregate results ─────────────────────────────────────────────────────
    all_results = {
        "HAQOA": {
            "best_quality":    result.best_quality,
            "best_solution":   result.best_solution,
            "elapsed_seconds": result.elapsed_seconds,
            "history":         result.convergence_curve.tolist(),
        },
        **baseline_results,
    }

    # ── Print comparison ──────────────────────────────────────────────────────
    print("\n" + "═" * 62)
    print(f"  HAQOA vs Baselines — {tsp.name}  (n={tsp.n})")
    print("═" * 62)
    table = comparison_table(all_results, baseline_quality=improved_q)
    print(table)

    # Gap summary
    haqoa_gap  = compute_gap(result.best_quality, improved_q)
    ga_gap     = compute_gap(baseline_results["GA"]["best_quality"], improved_q)
    sa_gap     = compute_gap(baseline_results["SA"]["best_quality"], improved_q)
    print(f"  HAQOA gap vs 2-opt : {haqoa_gap:+.2f}%")
    print(f"  GA    gap vs 2-opt : {ga_gap:+.2f}%")
    print(f"  SA    gap vs 2-opt : {sa_gap:+.2f}%")

    # ── Save plots ────────────────────────────────────────────────────────────
    logger.info("Saving plots...")

    p1 = f"{out_dir}/convergence_{instance_name}.png"
    plot_convergence(all_results, title=f"Convergence — {tsp.name}", save_path=p1)
    logger.info(f"  Saved: {p1}")

    p2 = f"{out_dir}/entropy_dynamics_{instance_name}.png"
    plot_entropy_dynamics(result, save_path=p2)
    logger.info(f"  Saved: {p2}")

    p3 = f"{out_dir}/routes_{instance_name}.png"
    routes_to_plot = {
        "HAQOA": result.best_solution,
        "GA":    baseline_results["GA"]["best_solution"],
        "SA":    baseline_results["SA"]["best_solution"],
        "ACO":   baseline_results["ACO"]["best_solution"],
    }
    plot_routes(tsp, routes_to_plot, save_path=p3)
    logger.info(f"  Saved: {p3}")

    p4 = f"{out_dir}/quality_bars_{instance_name}.png"
    plot_quality_bars(all_results, title=f"Solution Quality — {tsp.name}", save_path=p4)
    logger.info(f"  Saved: {p4}")

    p5 = f"{out_dir}/population_dynamics_{instance_name}.png"
    plot_population_heatmap(result, save_path=p5)
    logger.info(f"  Saved: {p5}")

    logger.info(f"All outputs in ./{out_dir}/")
    return all_results, result


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HAQOA Experiment Runner")
    parser.add_argument(
        "--instance", default="small",
        choices=list(BENCHMARK_SUITE.keys()),
        help="TSP instance from benchmark suite",
    )
    parser.add_argument("--iters",  type=int, default=300, help="Max iterations")
    parser.add_argument("--pop",    type=int, default=50,  help="Population size")
    parser.add_argument("--seed",   type=int, default=42,  help="Random seed")
    parser.add_argument("--outdir", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    run_experiment(
        instance_name=args.instance,
        max_iterations=args.iters,
        population_size=args.pop,
        seed=args.seed,
        out_dir=args.outdir,
    )
