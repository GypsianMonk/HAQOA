"""
Metrics & Statistical Evaluation — HAQOA
Provides comparison tables, convergence speed, gap computations,
and placeholders for Phase-3 statistical tests (Wilcoxon, CI).
"""

from __future__ import annotations

import math
from typing import Dict, Any, List, Optional

import numpy as np


# ─── Gap Computation ──────────────────────────────────────────────────────────

def compute_gap(quality: float, reference: float) -> float:
    """
    Percentage gap relative to a reference solution.
    gap = (quality - reference) / reference * 100
    Negative means better than reference.
    """
    if reference == 0:
        return 0.0
    return (quality - reference) / reference * 100.0


# ─── Convergence Speed ────────────────────────────────────────────────────────

def convergence_speed(
    history: List[float],
    threshold_fraction: float = 0.99,
) -> int:
    """
    Return the iteration at which the algorithm reaches
    `threshold_fraction` of its final best quality.
    E.g. threshold_fraction=0.99 → iteration to within 1% of final.
    Returns -1 if never reached.
    """
    if not history:
        return -1
    final = history[-1]
    target = final / threshold_fraction if threshold_fraction != 0 else final
    for i, q in enumerate(history):
        if q <= target:
            return i
    return -1


def area_under_curve(history: List[float]) -> float:
    """Normalised area under convergence curve (lower = faster convergence)."""
    if not history:
        return 0.0
    arr = np.array(history, dtype=float)
    # Normalise to [0,1]
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return 0.0
    normalised = (arr - lo) / (hi - lo)
    return float(np.trapz(normalised) / len(normalised))


# ─── Comparison Table ─────────────────────────────────────────────────────────

def comparison_table(
    results: Dict[str, Dict[str, Any]],
    baseline_quality: Optional[float] = None,
    width: int = 62,
) -> str:
    """
    Render a formatted ASCII comparison table.

    Args:
        results          : dict mapping algo_name → result_dict
                           result_dict must have keys:
                           'best_quality', 'elapsed_seconds', 'history'
        baseline_quality : optional reference quality for gap computation
        width            : total table width in characters

    Returns:
        Multi-line string.
    """
    header = f"{'Algorithm':<12} {'Best':>10} {'Gap%':>8} {'Conv@99%':>10} {'Time(s)':>8}"
    sep = "─" * width

    rows = [sep, header, sep]

    for name, res in results.items():
        q = res["best_quality"]
        t = res.get("elapsed_seconds", 0.0)
        hist = res.get("history", [])

        gap_str = "  —"
        if baseline_quality is not None:
            gap = compute_gap(q, baseline_quality)
            gap_str = f"{gap:+.2f}%"

        conv = convergence_speed(hist)
        conv_str = str(conv) if conv >= 0 else "—"

        rows.append(
            f"{name:<12} {q:>10.3f} {gap_str:>8} {conv_str:>10} {t:>8.2f}s"
        )

    rows.append(sep)
    return "\n".join(rows)


# ─── Statistical Tests (Phase 3 stubs) ───────────────────────────────────────

def wilcoxon_test(
    samples_a: List[float],
    samples_b: List[float],
) -> Dict[str, Any]:
    """
    Wilcoxon signed-rank test between two sets of results.
    Requires scipy. Returns p-value and effect size.
    Phase 3 placeholder — raises ImportError gracefully if scipy absent.
    """
    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(samples_a, samples_b, alternative="two-sided")
        n = len(samples_a)
        effect_r = stat / math.sqrt(n * (n + 1) * (2 * n + 1) / 6)
        return {"statistic": stat, "p_value": p, "effect_r": effect_r}
    except ImportError:
        return {"error": "scipy not installed — run: pip install scipy"}


def bootstrap_ci(
    samples: List[float],
    n_bootstrap: int = 2000,
    ci: float = 0.95,
) -> Dict[str, float]:
    """
    Bootstrap confidence interval for the mean of `samples`.
    Returns {'mean': ..., 'lower': ..., 'upper': ...}.
    """
    arr = np.array(samples)
    means = [
        np.mean(np.random.choice(arr, size=len(arr), replace=True))
        for _ in range(n_bootstrap)
    ]
    alpha = 1.0 - ci
    lower = float(np.percentile(means, 100 * alpha / 2))
    upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return {"mean": float(arr.mean()), "lower": lower, "upper": upper}


def summarise_multi_run(
    run_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Summarise statistics across multiple independent runs of one algorithm.
    Input: list of result dicts (each with 'best_quality').
    Returns mean, std, best, worst, and 95% CI.
    """
    qualities = [r["best_quality"] for r in run_results]
    arr = np.array(qualities)
    ci = bootstrap_ci(qualities)
    return {
        "mean":   float(arr.mean()),
        "std":    float(arr.std()),
        "best":   float(arr.min()),
        "worst":  float(arr.max()),
        "ci_95":  ci,
        "n_runs": len(run_results),
    }
