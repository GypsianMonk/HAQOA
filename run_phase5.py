"""
HAQOA-X Phase 5 — Large-Scale Evaluation & Scalability Analysis
Tests HAQOA-X against all baselines on n=100, 200, 300, 500 instances.
Produces:
  - Scalability plot (n vs best quality, n vs runtime)
  - Gap vs 2-opt-NN reference across scales
  - Algorithm ranking by instance size

Usage:
    python run_phase5.py                         # quick demo (n=100,200)
    python run_phase5.py --full                  # all sizes including n=500
    python run_phase5.py --instance n300_clustered
"""

from __future__ import annotations

import argparse, logging, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from haqoa.engine_x      import HAQOAXEngine, HAQOAXConfig
from haqoa.operators     import make_tsp_objective, make_tsp_operators, adaptive_mutation
from haqoa.problems.tsp  import BENCHMARK_SUITE, TSPInstance
from haqoa.baselines.algorithms import (
    GeneticAlgorithm, SimulatedAnnealing,
    DiscreteParticleSwarm, AntColonyOptimisation,
)
from haqoa.metrics       import comparison_table, compute_gap
from haqoa.similarity    import tsp_positional_similarity, tsp_edge_jaccard
from haqoa.visualization.plots import plot_convergence, plot_quality_bars

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ─── Config per scale ─────────────────────────────────────────────────────────

def _cfg_for_n(n: int, pop: int, iters: int, seed: int) -> HAQOAXConfig:
    """Tune HAQOA-X parameters for instance size."""
    return HAQOAXConfig(
        population_size  = pop,
        max_iterations   = iters,
        beta_base        = 0.4,
        beta_max         = 20.0,
        entropy_sensitivity = 0.9,
        w_cost           = 0.45,
        w_density        = 0.15 if n >= 200 else 0.20,  # cheaper density at large n
        w_risk           = 0.18,
        w_volatility     = 0.12,
        w_noise          = 0.10,
        collapse_alpha   = 0.28,
        collapse_interval = 30,
        regeneration_rate = 0.18,
        gamma_quality    = 0.55,
        gamma_learning   = 0.22,
        multi_scale      = True,
        global_mut_rate  = 0.60,
        regional_mut_rate= 0.30,
        local_search_interval = 40,
        density_max_pairs= 800 if n >= 300 else 1500,   # sparse sampling for large n
        stagnation_limit = iters,
        seed             = seed,
    )


def run_haqoax_large(tsp: TSPInstance, pop: int, iters: int, seed: int):
    cfg = _cfg_for_n(tsp.n, pop, iters, seed)
    sim_fn = tsp_positional_similarity if tsp.n > 100 else tsp_edge_jaccard
    ls_fn  = tsp.local_search_large if tsp.n > 100 else \
              (lambda r: tsp.two_opt_improve(r, max_iter=80))

    engine = HAQOAXEngine(
        config       = cfg,
        objective_fn = make_tsp_objective(tsp),
        mutation_fn  = make_tsp_operators("advanced")[0],
        crossover_fn = make_tsp_operators("advanced")[1],
        random_fn    = tsp.random_route,
        similarity_fn= sim_fn,
    )
    engine.set_local_search(ls_fn)

    greedy = tsp.greedy_route()
    seeds  = [greedy] + [adaptive_mutation(greedy) for _ in range(min(9, pop//6))]
    while len(seeds) < pop:
        seeds.append(tsp.random_route())

    return engine.run(initial_solutions=seeds[:pop])


def run_baselines_large(tsp: TSPInstance, pop: int, iters: int, seed: int) -> dict:
    """Run all 4 baselines, scaled for large n."""
    results = {}
    algos   = {
        "GA":  GeneticAlgorithm(tsp, population_size=pop,
                                max_generations=iters, seed=seed),
        "SA":  SimulatedAnnealing(tsp, max_iterations=iters * 100, seed=seed),
        "PSO": DiscreteParticleSwarm(tsp, swarm_size=pop,
                                     max_iterations=iters, seed=seed),
        "ACO": AntColonyOptimisation(tsp, n_ants=max(10, pop // 3),
                                     max_iterations=iters, seed=seed),
    }
    for name, algo in algos.items():
        logger.info(f"    {name}...")
        results[name] = algo.run()
    return results


# ─── Scalability plot ─────────────────────────────────────────────────────────

def plot_scalability(scale_results: list, out_dir: str):
    """
    scale_results: list of dicts with keys:
      n, instance, haqoax_gap, ga_gap, sa_gap, pso_gap, aco_gap,
      haqoax_time, ga_time, sa_time
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns   = [r["n"]          for r in scale_results]
    hx_g = [r["haqoax_gap"] for r in scale_results]
    ga_g = [r["ga_gap"]     for r in scale_results]
    sa_g = [r["sa_gap"]     for r in scale_results]
    pso_g= [r["pso_gap"]    for r in scale_results]
    aco_g= [r["aco_gap"]    for r in scale_results]

    hx_t = [r["haqoax_time"] for r in scale_results]
    ga_t = [r["ga_time"]     for r in scale_results]
    sa_t = [r["sa_time"]     for r in scale_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor="#F8FAFC")

    colors = {"HAQOA-X":"#2563EB","GA":"#16A34A","SA":"#DC2626","PSO":"#9333EA","ACO":"#D97706"}

    ax1.set_facecolor("#F1F5F9")
    ax1.plot(ns, hx_g,  "o-", color=colors["HAQOA-X"], lw=2.5, ms=7, label="HAQOA-X", zorder=5)
    ax1.plot(ns, ga_g,  "s--",color=colors["GA"],      lw=1.4, ms=5, label="GA")
    ax1.plot(ns, sa_g,  "^-.",color=colors["SA"],      lw=1.4, ms=5, label="SA")
    ax1.plot(ns, pso_g, "D:",  color=colors["PSO"],     lw=1.4, ms=5, label="PSO")
    ax1.plot(ns, aco_g, "P--", color=colors["ACO"],     lw=1.4, ms=5, label="ACO")
    ax1.axhline(0, color="#64748B", lw=1, ls="--", alpha=0.5, label="2-opt-NN ref")
    ax1.set_xlabel("Problem size (n cities)", fontsize=11)
    ax1.set_ylabel("Gap vs 2-opt-NN reference (%)", fontsize=11)
    ax1.set_title("Solution Quality vs Scale", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.35); ax1.spines[["top","right"]].set_visible(False)

    ax2.set_facecolor("#F1F5F9")
    ax2.plot(ns, hx_t, "o-", color=colors["HAQOA-X"], lw=2.5, ms=7, label="HAQOA-X")
    ax2.plot(ns, ga_t, "s--",color=colors["GA"],      lw=1.4, ms=5, label="GA")
    ax2.plot(ns, sa_t, "^-.",color=colors["SA"],      lw=1.4, ms=5, label="SA")
    ax2.set_xlabel("Problem size (n cities)", fontsize=11)
    ax2.set_ylabel("Runtime (seconds)", fontsize=11)
    ax2.set_title("Runtime vs Scale", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.35); ax2.spines[["top","right"]].set_visible(False)

    fig.suptitle("HAQOA-X Phase 5 — Scalability Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    p = f"{out_dir}/scalability.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    logger.info(f"  Saved: {p}")


# ─── Main ─────────────────────────────────────────────────────────────────────

SCALE_PLAN = {
    # instance_key: (pop, iters)   — tuned for speed vs quality tradeoff
    "large":          (60,  300),
    "n200_random":    (70,  350),
    "n200_mixed":     (70,  350),
    "n300_clustered": (80,  400),
    "n500_random":    (90,  450),
    "n500_mixed":     (90,  450),
}

QUICK_PLAN = {
    "large":       (50, 200),
    "n200_random": (60, 200),
    "n200_mixed":  (60, 200),
}


def run_phase5(full: bool = False, single: str = None,
               seed: int = 42, out_dir: str = "results/phase5"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    plan = SCALE_PLAN if full else QUICK_PLAN
    if single:
        plan = {single: SCALE_PLAN.get(single, (60, 300))}

    logger.info(f"═══ HAQOA-X Phase 5 — Large-Scale Evaluation  ({'full' if full else 'quick'}) ═══")

    scale_results = []
    summary_lines = []

    for inst_name, (pop, iters) in plan.items():
        gen_fn, kwargs = BENCHMARK_SUITE[inst_name]
        tsp = gen_fn(**kwargs, name=inst_name)

        # Reference quality
        t0    = time.perf_counter()
        greedy = tsp.greedy_route()
        ref_r  = tsp.local_search_large(greedy) if tsp.n > 100 else \
                 tsp.two_opt_improve(greedy)
        ref_q  = tsp.route_distance(ref_r)
        ref_t  = time.perf_counter() - t0

        logger.info(f"\n{'─'*60}")
        logger.info(f"Instance: {tsp}   2-opt-NN ref: {ref_q:.2f}  ({ref_t:.2f}s)")

        # HAQOA-X
        logger.info("  HAQOA-X...")
        hx_res = run_haqoax_large(tsp, pop, iters, seed)
        hx_gap = compute_gap(hx_res.best_quality, ref_q)

        # Baselines
        base = run_baselines_large(tsp, pop, iters, seed)

        all_res = {
            "HAQOA-X": {"best_quality": hx_res.best_quality,
                         "elapsed_seconds": hx_res.elapsed_seconds,
                         "history": hx_res.convergence_curve.tolist()},
            **base,
        }

        # Table
        print(f"\n{'═'*65}")
        print(f"  {inst_name}  (n={tsp.n})")
        print("═"*65)
        print(comparison_table(all_res, baseline_quality=ref_q))
        for name, res in all_res.items():
            print(f"  {name:<12}  gap: {compute_gap(res['best_quality'], ref_q):+.2f}%  "
                  f"  time: {res['elapsed_seconds']:.1f}s")

        # Save plots
        plot_convergence(all_res,
                         title=f"Convergence — {inst_name}  (n={tsp.n})",
                         save_path=f"{out_dir}/convergence_{inst_name}.png")
        plot_quality_bars(all_res,
                          title=f"Quality — {inst_name}  (n={tsp.n})",
                          save_path=f"{out_dir}/quality_{inst_name}.png")

        scale_results.append({
            "n":           tsp.n,
            "instance":    inst_name,
            "haqoax_gap":  hx_gap,
            "ga_gap":      compute_gap(base["GA"]["best_quality"],  ref_q),
            "sa_gap":      compute_gap(base["SA"]["best_quality"],  ref_q),
            "pso_gap":     compute_gap(base["PSO"]["best_quality"], ref_q),
            "aco_gap":     compute_gap(base["ACO"]["best_quality"], ref_q),
            "haqoax_time": hx_res.elapsed_seconds,
            "ga_time":     base["GA"]["elapsed_seconds"],
            "sa_time":     base["SA"]["elapsed_seconds"],
        })

        wins = hx_res.best_quality <= min(r["best_quality"] for r in base.values()) + 0.01
        summary_lines.append(
            f"  {inst_name:<20} n={tsp.n:<5} gap={hx_gap:+.2f}%  "
            f"wins={'YES' if wins else 'no '}"
        )

    # Scalability plot (only if multiple sizes)
    if len(scale_results) > 1:
        plot_scalability(scale_results, out_dir)

    # Summary
    wins_total = sum(1 for r in scale_results if r["haqoax_gap"] <= min(r["ga_gap"],r["sa_gap"],r["pso_gap"],r["aco_gap"]) + 0.5)
    print(f"\n{'═'*65}")
    print("  PHASE 5 SUMMARY")
    print("═"*65)
    for line in summary_lines: print(line)
    print(f"\n  HAQOA-X wins or ties: {wins_total}/{len(scale_results)}")
    print(f"  All outputs → ./{out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HAQOA-X Phase 5 — Large-Scale Evaluation")
    parser.add_argument("--full",     action="store_true", help="Run all sizes including n=500")
    parser.add_argument("--instance", type=str, default=None,
                        choices=list(SCALE_PLAN.keys()))
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--outdir",   type=str, default="results/phase5")
    args = parser.parse_args()
    run_phase5(full=args.full, single=args.instance, seed=args.seed, out_dir=args.outdir)
