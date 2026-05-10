"""
HAQOA-X Experiment Runner  — AQSE-v2
Runs HAQOA-X + all baselines on TSP benchmarks.
Produces full diagnostic report + 8 plot types.

Usage:
    python run_haqoax.py                               # small, 300 iters
    python run_haqoax.py --instance medium --iters 500
    python run_haqoax.py --instance large  --iters 500 --pop 80
    python run_haqoax.py --compare                     # run all instances
"""

from __future__ import annotations

import argparse, logging, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from haqoa.engine_x import HAQOAXEngine, HAQOAXConfig
from haqoa.operators import make_tsp_operators, adaptive_mutation
from haqoa.operators import make_tsp_objective
from haqoa.problems.tsp import BENCHMARK_SUITE, TSPInstance
from haqoa.baselines.algorithms import run_all_baselines
from haqoa.similarity import tsp_edge_jaccard, tsp_positional_similarity
from haqoa.metrics import comparison_table, compute_gap
from haqoa.visualization.plots import (
    plot_convergence,
    plot_quality_bars,
    plot_energy_breakdown,
    plot_multi_scale_activity,
    plot_haqoax_dashboard,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Build seed population ────────────────────────────────────────────────────

def build_seeds(tsp: TSPInstance, n: int, seed: int = 42) -> list:
    import random; random.seed(seed); np.random.seed(seed)
    greedy = tsp.greedy_route()
    seeds  = [greedy]
    n_mut  = max(1, n // 6)
    for _ in range(n_mut):
        seeds.append(adaptive_mutation(greedy))
    while len(seeds) < n:
        seeds.append(tsp.random_route())
    return seeds[:n]


# ─── Single HAQOA-X run ───────────────────────────────────────────────────────

def run_haqoax(
    tsp: TSPInstance,
    max_iterations: int = 400,
    population_size: int = 60,
    seed: int = 42,
):
    cfg = HAQOAXConfig(
        population_size=population_size,
        max_iterations=max_iterations,
        # Amplification
        beta_base=0.4,
        beta_max=20.0,
        entropy_sensitivity=0.9,
        # Entropy
        entropy_damping=0.05,
        # Energy weights
        w_cost=0.45,
        w_density=0.20,
        w_risk=0.15,
        w_volatility=0.10,
        w_noise=0.10,
        # Collapse
        collapse_threshold_base=0.004,
        collapse_alpha=0.30,
        collapse_interval=25,
        collapse_min_survivors=0.30,
        # Regeneration
        regeneration_rate=0.20,
        # AI Reward
        gamma_quality=0.55,
        gamma_cost=0.15,
        gamma_volatility=0.10,
        gamma_learning=0.20,
        # Multi-scale
        multi_scale=True,
        global_mut_rate=0.55,
        regional_mut_rate=0.30,
        local_search_rate=0.20,
        local_search_interval=30,
        # Evolution
        mutation_rate=0.40,
        crossover_rate=0.65,
        elite_fraction=0.08,
        # Turbulence
        turbulence_threshold=0.40,
        # Stop
        stagnation_limit=max_iterations,
        seed=seed,
    )

    obj_fn        = make_tsp_objective(tsp)
    mut_fn, cx_fn = make_tsp_operators("advanced")

    # Use fast positional sim for large instances
    sim_fn = tsp_positional_similarity if tsp.n > 80 else tsp_edge_jaccard

    engine = HAQOAXEngine(
        config=cfg,
        objective_fn=obj_fn,
        mutation_fn=mut_fn,
        crossover_fn=cx_fn,
        random_fn=tsp.random_route,
        similarity_fn=sim_fn,
    )
    engine.set_local_search(lambda r: tsp.two_opt_improve(r, max_iter=80))

    seeds  = build_seeds(tsp, n=population_size, seed=seed)
    result = engine.run(initial_solutions=seeds)
    return result


# ─── Main experiment ──────────────────────────────────────────────────────────

def run_experiment(
    instance_name: str = "small",
    max_iterations: int = 300,
    population_size: int = 60,
    seed: int = 42,
    out_dir: str = "results/haqoax",
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"═══ HAQOA-X  |  instance={instance_name}  iters={max_iterations} ═══")

    gen_fn, kwargs = BENCHMARK_SUITE[instance_name]
    tsp = gen_fn(**kwargs, name=instance_name)
    ref_q = tsp.route_distance(tsp.two_opt_improve(tsp.greedy_route()))
    logger.info(f"Instance: {tsp}   2-opt ref: {ref_q:.3f}")

    # ── HAQOA-X ───────────────────────────────────────────────────────────────
    logger.info("Running HAQOA-X (AQSE-v2)...")
    result = run_haqoax(tsp, max_iterations, population_size, seed)
    logger.info(
        f"HAQOA-X: best={result.best_quality:.3f}  "
        f"iters={result.iterations_run}  "
        f"time={result.elapsed_seconds:.2f}s  "
        f"reason={result.termination_reason}"
    )

    # ── Baselines ─────────────────────────────────────────────────────────────
    logger.info("Running baselines...")
    baselines = run_all_baselines(tsp, max_iterations=max_iterations,
                                  population_size=population_size, seed=seed)

    all_results = {
        "HAQOA-X": {
            "best_quality":    result.best_quality,
            "best_solution":   result.best_solution,
            "elapsed_seconds": result.elapsed_seconds,
            "history":         result.convergence_curve.tolist(),
        },
        **baselines,
    }

    # ── Comparison table ──────────────────────────────────────────────────────
    print("\n" + "═"*70)
    print(f"  HAQOA-X vs Baselines — {tsp.name}  (n={tsp.n})")
    print("═"*70)
    print(comparison_table(all_results, baseline_quality=ref_q))
    for name, res in all_results.items():
        gap = compute_gap(res["best_quality"], ref_q)
        print(f"  {name:<12}  gap vs 2-opt: {gap:+.2f}%")

    # ── HAQOA-X specific plots ────────────────────────────────────────────────
    logger.info("Generating plots...")

    comparison_hist = {k: v for k, v in all_results.items() if k != "HAQOA-X"}

    p = f"{out_dir}/dashboard_{instance_name}.png"
    plot_haqoax_dashboard(result, comparison=comparison_hist, save_path=p)
    logger.info(f"  {p}")

    p = f"{out_dir}/energy_breakdown_{instance_name}.png"
    plot_energy_breakdown(result, save_path=p)
    logger.info(f"  {p}")

    p = f"{out_dir}/multi_scale_{instance_name}.png"
    plot_multi_scale_activity(result, save_path=p)
    logger.info(f"  {p}")

    p = f"{out_dir}/convergence_{instance_name}.png"
    plot_convergence(all_results, title=f"Convergence — {tsp.name}", save_path=p)
    logger.info(f"  {p}")

    p = f"{out_dir}/quality_bars_{instance_name}.png"
    plot_quality_bars(all_results, title=f"Quality — {tsp.name}", save_path=p)
    logger.info(f"  {p}")

    logger.info(f"All outputs → ./{out_dir}/")
    return result, all_results


# ─── All-instance comparison sweep ───────────────────────────────────────────

def run_all_instances(iters=300, pop=60, seed=42, out_dir="results/haqoax"):
    """Run HAQOA-X on all benchmark instances and produce summary."""
    instances = ["tiny", "small", "medium", "large", "circle_20"]
    summary = {}
    for inst in instances:
        logger.info(f"\n{'─'*50}")
        result, all_res = run_experiment(inst, iters, pop, seed, out_dir)
        summary[inst] = {
            "haqoax": result.best_quality,
            "ga":     all_res["GA"]["best_quality"],
            "sa":     all_res["SA"]["best_quality"],
        }
    print("\n\n" + "═"*60)
    print("  HAQOA-X Sweep Summary")
    print("═"*60)
    print(f"{'Instance':<14} {'HAQOA-X':>10} {'GA':>10} {'SA':>10} {'HAQOA-X wins?':>14}")
    print("─"*60)
    for inst, s in summary.items():
        wins = "✓" if s["haqoax"] <= min(s["ga"], s["sa"]) else "✗"
        print(f"{inst:<14} {s['haqoax']:>10.2f} {s['ga']:>10.2f} {s['sa']:>10.2f} {wins:>14}")
    print("═"*60)


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HAQOA-X Experiment Runner")
    parser.add_argument("--instance", default="small",
                        choices=list(BENCHMARK_SUITE.keys()))
    parser.add_argument("--iters",   type=int, default=300)
    parser.add_argument("--pop",     type=int, default=60)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--outdir",  type=str, default="results/haqoax")
    parser.add_argument("--compare", action="store_true",
                        help="Run all instances in sequence")
    args = parser.parse_args()

    if args.compare:
        run_all_instances(args.iters, args.pop, args.seed, args.outdir)
    else:
        run_experiment(args.instance, args.iters, args.pop, args.seed, args.outdir)
