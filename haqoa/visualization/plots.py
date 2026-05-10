"""
Visualization — HAQOA  (Phase 2 + Phase 3)

Phase 2: plot_convergence, plot_entropy_dynamics, plot_routes,
         plot_quality_bars, plot_population_heatmap
Phase 3: plot_boxplots, plot_violin, plot_convergence_bands,
         plot_critical_difference, plot_significance_heatmap,
         plot_phase3_dashboard
"""

from __future__ import annotations

import math
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap


# ─── Style ────────────────────────────────────────────────────────────────────

ALGO_COLORS = {
    "HAQOA": "#2563EB",
    "GA":    "#16A34A",
    "SA":    "#DC2626",
    "PSO":   "#9333EA",
    "ACO":   "#D97706",
}
ALGO_STYLES = {
    "HAQOA": dict(linewidth=2.5, linestyle="-",  zorder=5),
    "GA":    dict(linewidth=1.4, linestyle="--", zorder=4),
    "SA":    dict(linewidth=1.4, linestyle="-.", zorder=4),
    "PSO":   dict(linewidth=1.4, linestyle=":",  zorder=4),
    "ACO":   dict(linewidth=1.4, linestyle="--", zorder=4),
}

def _c(name): return ALGO_COLORS.get(name, "#64748B")
def _s(name): return ALGO_STYLES.get(name, {"linewidth": 1.2, "linestyle": "--"})

def _save(fig, path, dpi=150):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def _ax_style(ax):
    ax.set_facecolor("#F1F5F9")
    ax.grid(True, alpha=0.35, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 Plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_convergence(results, title="Convergence Comparison",
                     save_path=None, figsize=(10, 5)):
    fig, ax = plt.subplots(figsize=figsize, facecolor="#F8FAFC")
    _ax_style(ax)
    for name, res in results.items():
        hist = res.get("history", [])
        if hist:
            ax.plot(range(len(hist)), hist, color=_c(name),
                    label=f"{name}  ({res['best_quality']:.1f})", **_s(name))
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Tour Distance", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.legend(framealpha=0.9, fontsize=9)
    fig.tight_layout(); _save(fig, save_path); return fig


def plot_entropy_dynamics(result, save_path=None, figsize=(10, 6)):
    history    = result.history
    iterations = [r.iteration    for r in history]
    best_q     = [r.best_quality for r in history]
    entropy    = [r.entropy      for r in history]
    beta       = [r.beta         for r in history]

    fig = plt.figure(figsize=figsize, facecolor="#F8FAFC", layout="constrained")
    gs  = GridSpec(2, 1, figure=fig, hspace=0.38)

    ax1 = fig.add_subplot(gs[0]); _ax_style(ax1)
    ax1.plot(iterations, best_q, color="#2563EB", linewidth=2, label="Best Quality")
    ax1.set_ylabel("Tour Distance", fontsize=10)
    ax1.set_title("HAQOA — Convergence & Entropy Dynamics", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)

    ax2 = fig.add_subplot(gs[1]); _ax_style(ax2)
    l1, = ax2.plot(iterations, entropy, color="#10B981", linewidth=1.8, label="Entropy H(t)")
    ax2.set_ylabel("Entropy H(t)", fontsize=10, color="#10B981")
    ax2.tick_params(axis="y", labelcolor="#10B981")
    ax3 = ax2.twinx()
    l2, = ax3.plot(iterations, beta, color="#F59E0B", linewidth=1.8, linestyle="--", label="beta(t)")
    ax3.set_ylabel("beta(t)", fontsize=10, color="#F59E0B")
    ax3.tick_params(axis="y", labelcolor="#F59E0B")
    ax2.set_xlabel("Iteration", fontsize=10)
    ax2.legend(handles=[l1, l2], fontsize=9, loc="upper right")
    fig.tight_layout(); _save(fig, save_path); return fig


def plot_routes(tsp, routes, save_path=None, figsize=(14, 10)):
    n_algos = len(routes)
    ncols   = min(n_algos, 2)
    nrows   = math.ceil(n_algos / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             facecolor="#F8FAFC", squeeze=False)
    coords = tsp.coords
    for ax_idx, (name, route) in enumerate(routes.items()):
        r, c = divmod(ax_idx, ncols)
        ax   = axes[r][c]; ax.set_facecolor("#F1F5F9")
        for i in range(len(route)):
            a, b = route[i], route[(i + 1) % len(route)]
            ax.plot([coords[a,0], coords[b,0]], [coords[a,1], coords[b,1]],
                    color=_c(name), linewidth=1.0, alpha=0.7)
        ax.scatter(coords[:,0], coords[:,1], s=40, color=_c(name),
                   zorder=5, edgecolors="white", linewidths=0.5)
        ax.scatter(coords[route[0],0], coords[route[0],1], s=90,
                   color="#1E293B", zorder=6, marker="*")
        dist = tsp.route_distance(route)
        ax.set_title(f"{name}  —  {dist:.1f}", fontsize=10, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[["top","right","left","bottom"]].set_visible(False)
    for ax_idx in range(n_algos, nrows * ncols):
        r, c = divmod(ax_idx, ncols); axes[r][c].set_visible(False)
    fig.suptitle(f"Route Comparison — {tsp.name}  (n={tsp.n})",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout(); _save(fig, save_path); return fig


def plot_quality_bars(results, title="Solution Quality Comparison",
                      save_path=None, figsize=(9, 5)):
    names     = list(results.keys())
    qualities = [results[n]["best_quality"] for n in names]
    fig, ax   = plt.subplots(figsize=figsize, facecolor="#F8FAFC"); _ax_style(ax)
    bars = ax.barh(names, qualities, color=[_c(n) for n in names],
                   edgecolor="white", linewidth=0.8, height=0.55)
    for bar, q in zip(bars, qualities):
        ax.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
                f"{q:.1f}", va="center", ha="left", fontsize=9)
    if "HAQOA" in results:
        ax.axvline(results["HAQOA"]["best_quality"], color="#2563EB",
                   linewidth=1.5, linestyle="--", alpha=0.6)
    ax.set_xlabel("Tour Distance (lower is better)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.invert_yaxis()
    fig.tight_layout(); _save(fig, save_path); return fig


def plot_population_heatmap(result, save_path=None, figsize=(11, 4), max_points=200):
    history = result.history
    iters = np.array([r.iteration    for r in history])
    bq    = np.array([r.best_quality for r in history])
    entr  = np.array([r.entropy      for r in history])
    beta  = np.array([r.beta         for r in history])
    if len(iters) > max_points:
        idx = np.linspace(0, len(iters)-1, max_points, dtype=int)
        iters, bq, entr, beta = iters[idx], bq[idx], entr[idx], beta[idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor="#F8FAFC")
    sc1 = ax1.scatter(iters, bq, c=entr, cmap="viridis", s=18, alpha=0.8)
    fig.colorbar(sc1, ax=ax1, label="Entropy H(t)")
    ax1.set_xlabel("Iteration"); ax1.set_ylabel("Best Quality")
    ax1.set_title("Quality vs Entropy", fontsize=11, fontweight="bold")
    _ax_style(ax1)
    ax2.fill_between(iters, beta, alpha=0.3, color="#F59E0B")
    ax2.plot(iters, beta, color="#F59E0B", linewidth=1.5)
    ax2.set_xlabel("Iteration"); ax2.set_ylabel("beta(t)")
    ax2.set_title("Adaptive Amplification beta(t)", fontsize=11, fontweight="bold")
    _ax_style(ax2)
    fig.suptitle("HAQOA Population Dynamics", fontsize=12, fontweight="bold")
    fig.tight_layout(); _save(fig, save_path); return fig


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3 Plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_boxplots(
    multi_results: Dict[str, List[Dict[str, Any]]],
    title: str = "Solution Quality Distribution",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
    baseline_quality: Optional[float] = None,
) -> plt.Figure:
    """
    Box plots of best_quality distributions across N runs per algorithm.
    Each box shows median, IQR, whiskers and fliers.
    """
    names  = list(multi_results.keys())
    data   = [[r["best_quality"] for r in multi_results[n]] for n in names]
    colors = [_c(n) for n in names]

    fig, ax = plt.subplots(figsize=figsize, facecolor="#F8FAFC"); _ax_style(ax)

    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="white", linewidth=2.5),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker="o", markersize=4, alpha=0.5))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    for whisker, cap, color in zip(
            [w for pair in zip(bp["whiskers"][::2], bp["whiskers"][1::2]) for w in pair],
            [c for pair in zip(bp["caps"][::2], bp["caps"][1::2]) for c in pair],
            [c for c in colors for _ in range(2)]):
        whisker.set_color(color); cap.set_color(color)

    if baseline_quality:
        ax.axhline(baseline_quality, color="#64748B", linewidth=1.5,
                   linestyle="--", alpha=0.7, label=f"2-opt ref ({baseline_quality:.1f})")
        ax.legend(fontsize=9)

    ax.set_xticks(range(1, len(names)+1)); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Tour Distance", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    # Annotate median values
    for i, d in enumerate(data):
        med = float(np.median(d))
        ax.text(i+1, med, f" {med:.1f}", va="center", ha="left",
                fontsize=8, color="white", fontweight="bold")

    fig.tight_layout(); _save(fig, save_path); return fig


def plot_violin(
    multi_results: Dict[str, List[Dict[str, Any]]],
    title: str = "Quality Distribution — Violin",
    save_path: Optional[str] = None,
    figsize: tuple = (11, 6),
    baseline_quality: Optional[float] = None,
) -> plt.Figure:
    """
    Violin plots — show full probability density of results across runs.
    Inner box shows IQR; white dot = median.
    """
    names  = list(multi_results.keys())
    data   = [[r["best_quality"] for r in multi_results[n]] for n in names]
    colors = [_c(n) for n in names]

    fig, ax = plt.subplots(figsize=figsize, facecolor="#F8FAFC"); _ax_style(ax)

    parts = ax.violinplot(data, positions=range(1, len(names)+1),
                          showmeans=False, showmedians=False, showextrema=False)

    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color); pc.set_alpha(0.65); pc.set_edgecolor(color)

    # Overlay box stats manually
    for i, d in enumerate(data):
        arr = np.array(d)
        q1, med, q3 = np.percentile(arr, [25, 50, 75])
        ax.vlines(i+1, q1, q3, color="white", linewidth=5, zorder=5)
        ax.scatter([i+1], [med], color="white", s=30, zorder=6)
        ax.scatter([i+1], [arr.mean()], color=colors[i], s=20,
                   marker="D", zorder=7, edgecolors="white", linewidths=0.5)

    if baseline_quality:
        ax.axhline(baseline_quality, color="#64748B", linewidth=1.5,
                   linestyle="--", alpha=0.8, label=f"2-opt ref ({baseline_quality:.1f})")
        ax.legend(fontsize=9)

    ax.set_xticks(range(1, len(names)+1)); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Tour Distance", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    legend_items = [
        mpatches.Patch(color="white", label="Median (white bar)"),
        plt.Line2D([0],[0], marker="D", color="gray", label="Mean (diamond)", linestyle="None"),
    ]
    ax.legend(handles=legend_items, fontsize=8, loc="upper right")
    fig.tight_layout(); _save(fig, save_path); return fig


def plot_convergence_bands(
    multi_results: Dict[str, List[Dict[str, Any]]],
    title: str = "Convergence — Mean ± Std Across Runs",
    save_path: Optional[str] = None,
    figsize: tuple = (11, 5),
) -> plt.Figure:
    """
    Mean convergence curve ± 1 std shaded band for each algorithm.
    All histories are truncated to the shortest one.
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor="#F8FAFC"); _ax_style(ax)

    for name, runs in multi_results.items():
        histories = [r.get("history", []) for r in runs if r.get("history")]
        if not histories:
            continue
        min_len = min(len(h) for h in histories)
        matrix  = np.array([h[:min_len] for h in histories])   # (n_runs, T)
        mean    = matrix.mean(axis=0)
        std     = matrix.std(axis=0)
        xs      = np.arange(min_len)

        ax.plot(xs, mean, color=_c(name), label=f"{name}  (μ={mean[-1]:.1f})", **_s(name))
        ax.fill_between(xs, mean - std, mean + std, color=_c(name), alpha=0.15)

    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Tour Distance", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.legend(framealpha=0.9, fontsize=9)
    fig.tight_layout(); _save(fig, save_path); return fig


def plot_critical_difference(
    ranks: Dict[str, float],
    significance_pairs: Optional[List[Tuple[str, str]]] = None,
    title: str = "Critical Difference Diagram",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 4),
) -> plt.Figure:
    """
    Horizontal CD-style diagram showing average rank of each algorithm.
    significance_pairs: list of (algo_a, algo_b) that are NOT significantly different.
    """
    sorted_names = sorted(ranks, key=ranks.get)
    rank_vals    = [ranks[n] for n in sorted_names]
    colors       = [_c(n) for n in sorted_names]

    fig, ax = plt.subplots(figsize=figsize, facecolor="#F8FAFC")
    ax.set_facecolor("#F1F5F9")
    ax.set_xlim(0.5, len(ranks) + 0.5)
    ax.set_ylim(-1.2, 1.5)
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=16)

    # Rank axis line
    ax.axhline(0, color="#94A3B8", linewidth=2.0, xmin=0.05, xmax=0.95)
    ax.text(0.02, 0, "Best →", transform=ax.transAxes,
            fontsize=9, color="#64748B", va="center")

    for i, (name, rv, col) in enumerate(zip(sorted_names, rank_vals, colors)):
        x = (rv - min(rank_vals)) / max(max(rank_vals) - min(rank_vals), 1) * \
            (len(ranks) - 1) + 1
        y_tick = 0.22 if i % 2 == 0 else -0.22
        ax.plot(x, 0, "o", color=col, markersize=14, zorder=5)
        ax.plot([x, x], [0, y_tick], color=col, linewidth=1.2, alpha=0.6)
        ax.text(x, y_tick + (0.12 if i % 2 == 0 else -0.18),
                f"{name}\n({rv:.2f})",
                ha="center", va="bottom" if i % 2 == 0 else "top",
                fontsize=9, fontweight="bold" if name == "HAQOA" else "normal",
                color=col)

    # Draw cliques (non-significantly different groups) as horizontal bars
    if significance_pairs:
        for idx, (a, b) in enumerate(significance_pairs):
            if a in ranks and b in ranks:
                xa = (ranks[a] - min(rank_vals)) / max(max(rank_vals) - min(rank_vals), 1) \
                     * (len(ranks)-1) + 1
                xb = (ranks[b] - min(rank_vals)) / max(max(rank_vals) - min(rank_vals), 1) \
                     * (len(ranks)-1) + 1
                y  = -0.8 - idx * 0.2
                ax.hlines(y, min(xa, xb), max(xa, xb), colors="#64748B",
                          linewidth=3, alpha=0.5)

    fig.tight_layout(); _save(fig, save_path); return fig


def plot_significance_heatmap(
    multi_results: Dict[str, List[Dict[str, Any]]],
    title: str = "Pairwise Wilcoxon p-values",
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """
    Heatmap of pairwise Wilcoxon p-values.
    Green = significant (p < 0.05), red = not significant.
    """
    from haqoa.metrics import wilcoxon_test   # local import to avoid circular

    names = list(multi_results.keys())
    n     = len(names)
    pmat  = np.ones((n, n))

    for i, na in enumerate(names):
        for j, nb in enumerate(names):
            if i == j:
                continue
            qa = [r["best_quality"] for r in multi_results[na]]
            qb = [r["best_quality"] for r in multi_results[nb]]
            res = wilcoxon_test(qa, qb, alternative="two-sided")
            if "p_value" in res:
                pmat[i, j] = res["p_value"]

    cmap = LinearSegmentedColormap.from_list(
        "sig", ["#16A34A", "#FEF08A", "#DC2626"], N=256
    )
    fig, ax = plt.subplots(figsize=figsize, facecolor="#F8FAFC")
    im = ax.imshow(pmat, cmap=cmap, vmin=0, vmax=0.20, aspect="auto")
    fig.colorbar(im, ax=ax, label="p-value  (green < 0.05 = significant)")

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(names, fontsize=10)

    for i in range(n):
        for j in range(n):
            txt = "—" if i == j else f"{pmat[i,j]:.3f}"
            color = "white" if pmat[i, j] < 0.05 else "#1E293B"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=color)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout(); _save(fig, save_path); return fig


def plot_phase3_dashboard(
    multi_results: Dict[str, List[Dict[str, Any]]],
    ranks: Dict[str, float],
    tsp_name: str = "TSP",
    baseline_quality: Optional[float] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (16, 10),
) -> plt.Figure:
    """
    4-panel Phase-3 summary dashboard:
    [box plots | convergence bands]
    [sig heatmap | CD diagram       ]
    """
    from haqoa.metrics import wilcoxon_test

    names = list(multi_results.keys())
    fig   = plt.figure(figsize=figsize, facecolor="#F8FAFC")
    gs    = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)
    fig.suptitle(f"HAQOA Phase 3 — Statistical Validation Dashboard  [{tsp_name}]",
                 fontsize=14, fontweight="bold")

    # ── Panel 1: Box plots ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0]); _ax_style(ax1)
    data   = [[r["best_quality"] for r in multi_results[n]] for n in names]
    bp = ax1.boxplot(data, patch_artist=True, notch=False,
                     medianprops=dict(color="white", linewidth=2.5),
                     whiskerprops=dict(linewidth=1.2),
                     capprops=dict(linewidth=1.5),
                     flierprops=dict(marker="o", markersize=4, alpha=0.5))
    for patch, name in zip(bp["boxes"], names):
        patch.set_facecolor(_c(name)); patch.set_alpha(0.8)
    if baseline_quality:
        ax1.axhline(baseline_quality, color="#64748B", linestyle="--",
                    linewidth=1.2, alpha=0.7)
    ax1.set_xticks(range(1, len(names)+1)); ax1.set_xticklabels(names, fontsize=9)
    ax1.set_ylabel("Tour Distance"); ax1.set_title("Quality Distribution", fontweight="bold")

    # ── Panel 2: Convergence bands ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1]); _ax_style(ax2)
    for name, runs in multi_results.items():
        hists = [r.get("history", []) for r in runs if r.get("history")]
        if not hists: continue
        min_len = min(len(h) for h in hists)
        mat  = np.array([h[:min_len] for h in hists])
        mean = mat.mean(axis=0); std = mat.std(axis=0)
        xs   = np.arange(min_len)
        ax2.plot(xs, mean, color=_c(name), label=name, **_s(name))
        ax2.fill_between(xs, mean-std, mean+std, color=_c(name), alpha=0.12)
    ax2.set_xlabel("Iteration"); ax2.set_ylabel("Tour Distance")
    ax2.set_title("Convergence ± Std", fontweight="bold")
    ax2.legend(fontsize=8, framealpha=0.9)

    # ── Panel 3: p-value heatmap ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    n   = len(names)
    pmat = np.ones((n, n))
    for i, na in enumerate(names):
        for j, nb in enumerate(names):
            if i == j: continue
            qa = [r["best_quality"] for r in multi_results[na]]
            qb = [r["best_quality"] for r in multi_results[nb]]
            res = wilcoxon_test(qa, qb)
            if "p_value" in res: pmat[i, j] = res["p_value"]
    cmap = LinearSegmentedColormap.from_list("sig", ["#16A34A","#FEF08A","#DC2626"], N=256)
    im   = ax3.imshow(pmat, cmap=cmap, vmin=0, vmax=0.20, aspect="auto")
    fig.colorbar(im, ax=ax3, label="p-value", fraction=0.046, pad=0.04)
    ax3.set_xticks(range(n)); ax3.set_yticks(range(n))
    ax3.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax3.set_yticklabels(names, fontsize=8)
    for i in range(n):
        for j in range(n):
            txt   = "—" if i==j else f"{pmat[i,j]:.2f}"
            color = "white" if pmat[i,j]<0.05 else "#1E293B"
            ax3.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)
    ax3.set_title("Pairwise Wilcoxon p-values", fontweight="bold")

    # ── Panel 4: Average ranks ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1]); _ax_style(ax4)
    sorted_names = sorted(ranks, key=ranks.get)
    rank_vals    = [ranks[n] for n in sorted_names]
    bars = ax4.barh(sorted_names, rank_vals,
                    color=[_c(n) for n in sorted_names],
                    edgecolor="white", height=0.55)
    for bar, rv in zip(bars, rank_vals):
        ax4.text(bar.get_width()+0.02, bar.get_y()+bar.get_height()/2,
                 f"{rv:.2f}", va="center", ha="left", fontsize=9)
    ax4.axvline(1, color="#64748B", linewidth=1, linestyle="--", alpha=0.5)
    ax4.set_xlabel("Average Rank (1 = best)")
    ax4.set_title("Average Rank Across Runs", fontweight="bold")
    ax4.invert_yaxis()

    fig.tight_layout(); _save(fig, save_path); return fig


# ══════════════════════════════════════════════════════════════════════════════
# HAQOA-X Specific Plots  (Phase 4 visualizations)
# ══════════════════════════════════════════════════════════════════════════════

def plot_energy_breakdown(
    result,                   # HAQOAXResult
    save_path=None,
    figsize=(12, 5),
):
    """
    Stacked area chart of the 5 energy components over iterations.
    Shows how cost, density, risk, and volatility evolve.
    """
    bd   = result.energy_breakdown
    iters = np.arange(len(result.history))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor="#F8FAFC")

    # Left: stacked area
    _ax_style(ax1)
    colors = ["#2563EB", "#D97706", "#DC2626", "#9333EA"]
    labels = ["Cost", "Density", "Risk", "Volatility"]
    data   = [bd["cost"], bd["density"], bd["risk"], bd["volatility"]]
    ax1.stackplot(iters, data, labels=labels, colors=colors, alpha=0.75)
    ax1.set_xlabel("Iteration"); ax1.set_ylabel("Mean Energy Component")
    ax1.set_title("Energy Component Evolution", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper right")

    # Right: convergence with turbulence overlay
    _ax_style(ax2)
    bq   = result.convergence_curve
    turb = result.turbulence_curve
    ax2.plot(iters, bq, color="#2563EB", linewidth=2, label="Best Quality")
    ax2.set_ylabel("Tour Distance", color="#2563EB")
    ax2.tick_params(axis="y", labelcolor="#2563EB")
    ax2t = ax2.twinx()
    ax2t.fill_between(iters, turb, alpha=0.25, color="#F59E0B")
    ax2t.plot(iters, turb, color="#F59E0B", linewidth=1.2, label="Turbulence T(t)")
    ax2t.set_ylabel("Turbulence T(t)", color="#F59E0B")
    ax2t.tick_params(axis="y", labelcolor="#F59E0B")
    ax2.set_xlabel("Iteration")
    ax2.set_title("Convergence vs Turbulence", fontsize=11, fontweight="bold")
    lines1, lbs1 = ax2.get_legend_handles_labels()
    lines2, lbs2 = ax2t.get_legend_handles_labels()
    ax2.legend(lines1+lines2, lbs1+lbs2, fontsize=9)

    fig.suptitle("HAQOA-X Energy & Turbulence Dynamics", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_multi_scale_activity(
    result,                   # HAQOAXResult
    save_path=None,
    figsize=(11, 4),
):
    """
    Bar chart of per-layer offspring counts over iterations.
    Shows how the search shifts between Global / Regional / Local.
    """
    iters    = np.arange(len(result.history))
    n_global = np.array([r.n_global   for r in result.history])
    n_reg    = np.array([r.n_regional for r in result.history])
    n_local  = np.array([r.n_local    for r in result.history])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor="#F8FAFC")

    # Stacked bar (sampled every 10 iters for readability)
    step = max(1, len(iters) // 50)
    xs   = iters[::step]
    _ax_style(ax1)
    ax1.bar(xs, n_global[::step],  color="#2563EB", label="Global",   alpha=0.8)
    ax1.bar(xs, n_reg[::step],     color="#D97706", bottom=n_global[::step], label="Regional", alpha=0.8)
    ax1.bar(xs, n_local[::step],
            bottom=(n_global+n_reg)[::step],
            color="#16A34A", label="Local", alpha=0.8)
    ax1.set_xlabel("Iteration"); ax1.set_ylabel("Offspring Count")
    ax1.set_title("Multi-Scale Layer Activity", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)

    # Entropy ratio over time
    _ax_style(ax2)
    h_arr  = result.entropy_curve
    hmax   = np.array([r.h_max for r in result.history])
    ratio  = np.where(hmax > 0, h_arr / np.maximum(hmax, 1e-9), 0)
    ax2.fill_between(iters, ratio, alpha=0.3, color="#2563EB")
    ax2.plot(iters, ratio, color="#2563EB", linewidth=1.5, label="H(t)/H_max")
    ax2.axhline(0.5, color="#64748B", linewidth=1, linestyle="--", alpha=0.5)
    ax2.set_xlabel("Iteration"); ax2.set_ylabel("Entropy Ratio")
    ax2.set_title("Entropy Ratio (drives layer split)", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=9)

    fig.suptitle("HAQOA-X Multi-Scale Search Dynamics", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_haqoax_dashboard(
    result,                   # HAQOAXResult
    comparison: Optional[Dict] = None,  # {algo_name: single_run_result}
    save_path=None,
    figsize=(16, 12),
):
    """
    6-panel HAQOA-X full dashboard:
    [convergence | entropy+β ]
    [energy comp | turbulence]
    [multi-scale | quality   ]
    """
    hist  = result.history
    iters = np.arange(len(hist))
    bq    = result.convergence_curve
    entr  = result.entropy_curve
    beta  = result.beta_curve
    turb  = result.turbulence_curve
    bd    = result.energy_breakdown

    fig = plt.figure(figsize=figsize, facecolor="#F8FAFC", layout="constrained")
    gs  = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.32)
    fig.suptitle("HAQOA-X — Full System Dashboard", fontsize=14, fontweight="bold")

    # ── Panel 1: Convergence ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0]); _ax_style(ax1)
    ax1.plot(iters, bq, color="#2563EB", linewidth=2.5, label="HAQOA-X", zorder=5)
    if comparison:
        for name, res in comparison.items():
            h = res.get("history", [])
            if h:
                ax1.plot(range(len(h)), h, color=_c(name),
                         linewidth=1.2, linestyle="--", label=name, alpha=0.8)
    ax1.set_xlabel("Iteration"); ax1.set_ylabel("Tour Distance")
    ax1.set_title("Convergence", fontweight="bold")
    ax1.legend(fontsize=8)

    # ── Panel 2: Entropy + β ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1]); _ax_style(ax2)
    l1, = ax2.plot(iters, entr, color="#10B981", linewidth=2, label="H(t)")
    ax2.set_ylabel("Entropy H(t)", color="#10B981")
    ax2.tick_params(axis="y", labelcolor="#10B981")
    ax2b = ax2.twinx()
    l2, = ax2b.plot(iters, beta, color="#F59E0B", linewidth=2,
                    linestyle="--", label="β(t)")
    ax2b.set_ylabel("β(t)", color="#F59E0B")
    ax2b.tick_params(axis="y", labelcolor="#F59E0B")
    ax2.set_xlabel("Iteration"); ax2.legend(handles=[l1,l2], fontsize=8)
    ax2.set_title("Entropy & Amplification", fontweight="bold")

    # ── Panel 3: Energy breakdown ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0]); _ax_style(ax3)
    colors = ["#2563EB","#D97706","#DC2626","#9333EA"]
    labels = ["Cost","Density","Risk","Volatility"]
    data   = [bd["cost"], bd["density"], bd["risk"], bd["volatility"]]
    ax3.stackplot(iters, data, labels=labels, colors=colors, alpha=0.75)
    ax3.set_xlabel("Iteration"); ax3.set_ylabel("Mean Energy")
    ax3.set_title("Energy Decomposition", fontweight="bold")
    ax3.legend(fontsize=8, loc="upper right")

    # ── Panel 4: Turbulence ───────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1]); _ax_style(ax4)
    ax4.fill_between(iters, turb, alpha=0.3, color="#F59E0B")
    ax4.plot(iters, turb, color="#F59E0B", linewidth=1.5)
    thr = result.config.turbulence_threshold
    ax4.axhline(thr, color="#DC2626", linewidth=1.5, linestyle="--",
                alpha=0.7, label=f"Threshold ({thr})")
    ax4.set_xlabel("Iteration"); ax4.set_ylabel("Turbulence T(t)")
    ax4.set_title("Optimization Turbulence", fontweight="bold")
    ax4.legend(fontsize=8)

    # ── Panel 5: Multi-scale activity ─────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0]); _ax_style(ax5)
    n_g = np.array([r.n_global   for r in hist])
    n_r = np.array([r.n_regional for r in hist])
    n_l = np.array([r.n_local    for r in hist])
    step = max(1, len(iters)//40)
    xs   = iters[::step]
    ax5.bar(xs, n_g[::step], color="#2563EB", label="Global",   alpha=0.8, width=step*0.9)
    ax5.bar(xs, n_r[::step], bottom=n_g[::step], color="#D97706",
            label="Regional", alpha=0.8, width=step*0.9)
    ax5.bar(xs, n_l[::step], bottom=(n_g+n_r)[::step],
            color="#16A34A", label="Local", alpha=0.8, width=step*0.9)
    ax5.set_xlabel("Iteration"); ax5.set_ylabel("Offspring")
    ax5.set_title("Multi-Scale Layer Activity", fontweight="bold")
    ax5.legend(fontsize=8)

    # ── Panel 6: Collapse + regen events ──────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1]); _ax_style(ax6)
    n_col = np.array([r.n_collapsed for r in hist])
    n_gen = np.array([r.n_generated for r in hist])
    ax6.bar(iters, n_col, color="#DC2626", label="Collapsed", alpha=0.7, width=1.0)
    ax6.bar(iters, n_gen, color="#16A34A", label="Generated", alpha=0.5, width=1.0)
    pop_size = np.array([r.population_size for r in hist])
    ax6b = ax6.twinx()
    ax6b.plot(iters, pop_size, color="#64748B", linewidth=1.5,
              linestyle=":", label="Pop size")
    ax6b.set_ylabel("Population Size", color="#64748B")
    ax6b.tick_params(axis="y", labelcolor="#64748B")
    ax6.set_xlabel("Iteration"); ax6.set_ylabel("State Count")
    ax6.set_title("Collapse & Regeneration Dynamics", fontweight="bold")
    lines1, lbs1 = ax6.get_legend_handles_labels()
    lines2, lbs2 = ax6b.get_legend_handles_labels()
    ax6.legend(lines1+lines2, lbs1+lbs2, fontsize=8)

    fig.tight_layout()
    _save(fig, save_path)
    return fig
