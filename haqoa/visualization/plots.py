"""
Visualization — HAQOA
Convergence curves, entropy dynamics, route maps, quality bars,
and population heatmaps. All functions save to disk and optionally display.
"""

from __future__ import annotations

import math
from typing import Dict, Any, Optional, List
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


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

def _color(name):
    return ALGO_COLORS.get(name, "#64748B")

def _style(name):
    return ALGO_STYLES.get(name, {"linewidth": 1.2, "linestyle": "--"})

def _save(fig, path, dpi=150):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(results, title="Convergence Comparison", save_path=None, figsize=(10, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F1F5F9")
    for name, res in results.items():
        hist = res.get("history", [])
        if not hist:
            continue
        ax.plot(range(len(hist)), hist, color=_color(name),
                label=f"{name}  ({res['best_quality']:.1f})", **_style(name))
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Tour Distance", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.legend(framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.35, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_entropy_dynamics(result, save_path=None, figsize=(10, 6)):
    history = result.history
    iterations = [r.iteration    for r in history]
    best_q     = [r.best_quality for r in history]
    entropy    = [r.entropy      for r in history]
    beta       = [r.beta         for r in history]

    fig = plt.figure(figsize=figsize, facecolor="#F8FAFC")
    gs  = GridSpec(2, 1, figure=fig, hspace=0.38)

    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#F1F5F9")
    ax1.plot(iterations, best_q, color="#2563EB", linewidth=2, label="Best Quality")
    ax1.set_ylabel("Tour Distance", fontsize=10)
    ax1.set_title("HAQOA — Convergence & Entropy Dynamics", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, linewidth=0.6)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(fontsize=9)

    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#F1F5F9")
    l1, = ax2.plot(iterations, entropy, color="#10B981", linewidth=1.8, label="Entropy H(t)")
    ax2.set_ylabel("Entropy H(t)", fontsize=10, color="#10B981")
    ax2.tick_params(axis="y", labelcolor="#10B981")

    ax3 = ax2.twinx()
    l2, = ax3.plot(iterations, beta, color="#F59E0B", linewidth=1.8, linestyle="--", label="beta(t)")
    ax3.set_ylabel("beta(t)", fontsize=10, color="#F59E0B")
    ax3.tick_params(axis="y", labelcolor="#F59E0B")
    ax2.set_xlabel("Iteration", fontsize=10)
    ax2.grid(True, alpha=0.3, linewidth=0.6)
    ax2.spines["top"].set_visible(False)
    ax2.legend(handles=[l1, l2], fontsize=9, loc="upper right")

    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_routes(tsp, routes, save_path=None, figsize=(14, 10)):
    n_algos = len(routes)
    ncols   = min(n_algos, 2)
    nrows   = math.ceil(n_algos / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor="#F8FAFC", squeeze=False)
    coords = tsp.coords

    for ax_idx, (name, route) in enumerate(routes.items()):
        r, c = divmod(ax_idx, ncols)
        ax   = axes[r][c]
        ax.set_facecolor("#F1F5F9")
        for i in range(len(route)):
            a, b = route[i], route[(i + 1) % len(route)]
            ax.plot([coords[a, 0], coords[b, 0]], [coords[a, 1], coords[b, 1]],
                    color=_color(name), linewidth=1.0, alpha=0.7)
        ax.scatter(coords[:, 0], coords[:, 1], s=40, color=_color(name),
                   zorder=5, edgecolors="white", linewidths=0.5)
        ax.scatter(coords[route[0], 0], coords[route[0], 1], s=80,
                   color="#1E293B", zorder=6, marker="*")
        dist = tsp.route_distance(route)
        ax.set_title(f"{name}  —  {dist:.1f}", fontsize=10, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

    for ax_idx in range(n_algos, nrows * ncols):
        r, c = divmod(ax_idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(f"Route Comparison — {tsp.name}  (n={tsp.n})",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_quality_bars(results, title="Solution Quality Comparison", save_path=None, figsize=(9, 5)):
    names     = list(results.keys())
    qualities = [results[n]["best_quality"] for n in names]
    colors    = [_color(n) for n in names]

    fig, ax = plt.subplots(figsize=figsize, facecolor="#F8FAFC")
    ax.set_facecolor("#F1F5F9")
    bars = ax.barh(names, qualities, color=colors, edgecolor="white",
                   linewidth=0.8, height=0.55)
    for bar, q in zip(bars, qualities):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{q:.1f}", va="center", ha="left", fontsize=9)
    if "HAQOA" in results:
        ax.axvline(results["HAQOA"]["best_quality"], color="#2563EB",
                   linewidth=1.5, linestyle="--", alpha=0.6)
    ax.set_xlabel("Tour Distance (lower is better)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.35, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_population_heatmap(result, save_path=None, figsize=(11, 4), max_points=200):
    history = result.history
    iters = np.array([r.iteration    for r in history])
    bq    = np.array([r.best_quality for r in history])
    entr  = np.array([r.entropy      for r in history])
    beta  = np.array([r.beta         for r in history])

    if len(iters) > max_points:
        idx  = np.linspace(0, len(iters) - 1, max_points, dtype=int)
        iters = iters[idx]; bq = bq[idx]; entr = entr[idx]; beta = beta[idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor="#F8FAFC")
    sc1 = ax1.scatter(iters, bq, c=entr, cmap="viridis", s=18, alpha=0.8)
    fig.colorbar(sc1, ax=ax1, label="Entropy H(t)")
    ax1.set_xlabel("Iteration"); ax1.set_ylabel("Best Quality")
    ax1.set_title("Quality vs Entropy", fontsize=11, fontweight="bold")
    ax1.set_facecolor("#F1F5F9"); ax1.grid(True, alpha=0.3)
    ax1.spines[["top","right"]].set_visible(False)

    ax2.fill_between(iters, beta, alpha=0.3, color="#F59E0B")
    ax2.plot(iters, beta, color="#F59E0B", linewidth=1.5)
    ax2.set_xlabel("Iteration"); ax2.set_ylabel("beta(t)")
    ax2.set_title("Adaptive Amplification beta(t)", fontsize=11, fontweight="bold")
    ax2.set_facecolor("#F1F5F9"); ax2.grid(True, alpha=0.3)
    ax2.spines[["top","right"]].set_visible(False)

    fig.suptitle("HAQOA Population Dynamics", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_path)
    return fig
