"""
Metrics & Statistical Evaluation — HAQOA  (Phase 2 + Phase 3)

Phase 2 : comparison_table, compute_gap, convergence_speed
Phase 3 : wilcoxon_test, bootstrap_ci, summarise_multi_run,
          pairwise_significance_table, effect_size_cohens_d,
          friedman_test, critical_difference_ranks
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, Any, List, Optional, Tuple

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Basic Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_gap(quality: float, reference: float) -> float:
    """% gap vs reference. Negative = better than reference."""
    if reference == 0:
        return 0.0
    return (quality - reference) / reference * 100.0


def convergence_speed(history: List[float], threshold_fraction: float = 0.99) -> int:
    """Iteration at which algorithm reaches threshold_fraction of final quality."""
    if not history:
        return -1
    target = history[-1] / threshold_fraction if threshold_fraction != 0 else history[-1]
    for i, q in enumerate(history):
        if q <= target:
            return i
    return -1


def area_under_curve(history: List[float]) -> float:
    """Normalised AUC of convergence curve (lower = faster)."""
    if not history:
        return 0.0
    arr = np.array(history, dtype=float)
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return 0.0
    return float(np.trapz((arr - lo) / (hi - lo)) / len(arr))


def comparison_table(
    results: Dict[str, Dict[str, Any]],
    baseline_quality: Optional[float] = None,
    width: int = 70,
) -> str:
    header = f"{'Algorithm':<12} {'Best':>10} {'Gap%':>8} {'Conv@99%':>10} {'Time(s)':>8}"
    sep = "─" * width
    rows = [sep, header, sep]
    for name, res in results.items():
        q   = res["best_quality"]
        t   = res.get("elapsed_seconds", 0.0)
        hist = res.get("history", [])
        gap_str  = f"{compute_gap(q, baseline_quality):+.2f}%" if baseline_quality else "  —"
        conv     = convergence_speed(hist)
        conv_str = str(conv) if conv >= 0 else "—"
        rows.append(f"{name:<12} {q:>10.3f} {gap_str:>8} {conv_str:>10} {t:>8.2f}s")
    rows.append(sep)
    return "\n".join(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3 — Statistical Validation
# ══════════════════════════════════════════════════════════════════════════════

# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_ci(
    samples: List[float],
    n_bootstrap: int = 5000,
    ci: float = 0.95,
    statistic: str = "mean",   # 'mean' | 'median' | 'min'
) -> Dict[str, float]:
    """
    Bootstrap confidence interval.
    Returns: {mean, median, std, best, lower, upper, ci_level}
    """
    arr = np.array(samples, dtype=float)
    stat_fn = {"mean": np.mean, "median": np.median, "min": np.min}[statistic]

    boot_stats = [
        stat_fn(np.random.choice(arr, size=len(arr), replace=True))
        for _ in range(n_bootstrap)
    ]
    alpha = 1.0 - ci
    return {
        "mean":     float(arr.mean()),
        "median":   float(np.median(arr)),
        "std":      float(arr.std()),
        "best":     float(arr.min()),
        "worst":    float(arr.max()),
        "lower":    float(np.percentile(boot_stats, 100 * alpha / 2)),
        "upper":    float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
        "ci_level": ci,
        "n":        len(samples),
    }


# ── Effect Sizes ──────────────────────────────────────────────────────────────

def effect_size_cohens_d(a: List[float], b: List[float]) -> float:
    """
    Cohen's d: standardised mean difference.
    |d| < 0.2 trivial, 0.2–0.5 small, 0.5–0.8 medium, > 0.8 large.
    """
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    pooled_std = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if pooled_std == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def effect_size_r(wilcoxon_stat: float, n: int) -> float:
    """
    Rank-biserial correlation r from Wilcoxon statistic.
    |r| < 0.1 trivial, 0.1–0.3 small, 0.3–0.5 medium, > 0.5 large.
    """
    denom = n * (n + 1) * (2 * n + 1) / 6
    return float(wilcoxon_stat / denom) if denom > 0 else 0.0


def _interpret_d(d: float) -> str:
    ad = abs(d)
    if ad < 0.2: return "trivial"
    if ad < 0.5: return "small"
    if ad < 0.8: return "medium"
    return "large"


# ── Wilcoxon Signed-Rank Test ─────────────────────────────────────────────────

def wilcoxon_test(
    samples_a: List[float],
    samples_b: List[float],
    alternative: str = "two-sided",   # 'two-sided' | 'less' | 'greater'
) -> Dict[str, Any]:
    """
    Wilcoxon signed-rank test (paired, non-parametric).

    Null hypothesis H0: distributions of a and b are the same.
    alternative='less'  → H1: median(a) < median(b)  (a is better for minimisation)

    Returns:
        statistic, p_value, effect_r, cohens_d, interpretation, significant
    """
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return {"error": "scipy not installed — run: pip install scipy"}

    a = np.array(samples_a, dtype=float)
    b = np.array(samples_b, dtype=float)

    if len(a) != len(b):
        raise ValueError("Paired test requires equal-length samples.")
    if np.all(a == b):
        return {
            "statistic": 0.0, "p_value": 1.0,
            "effect_r": 0.0, "cohens_d": 0.0,
            "interpretation": "identical samples", "significant": False,
        }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p = wilcoxon(a, b, alternative=alternative, zero_method="wilcox")

    r   = effect_size_r(float(stat), len(a))
    d   = effect_size_cohens_d(list(a), list(b))
    sig = bool(p < 0.05)

    direction = "a < b" if a.mean() < b.mean() else "a > b"
    return {
        "statistic":      float(stat),
        "p_value":        float(p),
        "effect_r":       r,
        "cohens_d":       d,
        "effect_label":   _interpret_d(d),
        "direction":      direction,
        "significant":    sig,
        "alpha":          0.05,
    }


# ── Multi-run Summary ─────────────────────────────────────────────────────────

def summarise_multi_run(
    run_results: List[Dict[str, Any]],
    n_bootstrap: int = 5000,
) -> Dict[str, Any]:
    """
    Full statistics across N independent runs of one algorithm.
    Input : list of result dicts, each with key 'best_quality'.
    Returns: mean, median, std, best, worst, 95% CI, AUC mean.
    """
    qualities = [r["best_quality"] for r in run_results]
    ci        = bootstrap_ci(qualities, n_bootstrap=n_bootstrap)
    aucs      = [area_under_curve(r.get("history", [])) for r in run_results]
    return {**ci, "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs))}


# ── Pairwise Significance Table ───────────────────────────────────────────────

def pairwise_significance_table(
    multi_results: Dict[str, List[Dict[str, Any]]],
    reference: str = "HAQOA",
    width: int = 78,
) -> str:
    """
    Run Wilcoxon test between `reference` algorithm and every other.
    multi_results: algo_name → list of per-run result dicts.

    Returns formatted ASCII table.
    """
    if reference not in multi_results:
        return f"Reference algorithm '{reference}' not found."

    ref_q = [r["best_quality"] for r in multi_results[reference]]

    header = (f"{'Comparison':<22} {'p-value':>9} {'Sig?':>6} "
              f"{'Cohen d':>9} {'Effect':>8} {'Δ mean%':>9}")
    sep = "─" * width
    rows = [sep, f"  Wilcoxon Signed-Rank Tests  (reference = {reference})", sep,
            header, sep]

    for name, run_list in multi_results.items():
        if name == reference:
            continue
        cmp_q = [r["best_quality"] for r in run_list]
        res   = wilcoxon_test(ref_q, cmp_q, alternative="two-sided")
        if "error" in res:
            rows.append(f"{reference} vs {name:<16}  ERROR: {res['error']}")
            continue

        p    = res["p_value"]
        sig  = "✓" if res["significant"] else "✗"
        d    = res["cohens_d"]
        eff  = res["effect_label"]
        dm   = compute_gap(float(np.mean(ref_q)), float(np.mean(cmp_q)))
        rows.append(
            f"{reference} vs {name:<16} {p:>9.4f} {sig:>6} "
            f"{d:>9.3f} {eff:>8} {dm:>+9.2f}%"
        )

    rows.append(sep)
    rows.append("  ✓ = p < 0.05 (statistically significant)")
    rows.append(sep)
    return "\n".join(rows)


# ── Friedman Test ─────────────────────────────────────────────────────────────

def friedman_test(
    multi_results: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Friedman test: non-parametric k-sample test across all algorithms.
    Tests H0: all algorithms have the same median quality.
    """
    try:
        from scipy.stats import friedmanchisquare
    except ImportError:
        return {"error": "scipy not installed"}

    names  = list(multi_results.keys())
    groups = [
        [r["best_quality"] for r in multi_results[n]]
        for n in names
    ]
    # All groups must have same length
    min_len = min(len(g) for g in groups)
    groups  = [g[:min_len] for g in groups]

    stat, p = friedmanchisquare(*groups)
    return {
        "statistic":   float(stat),
        "p_value":     float(p),
        "significant": bool(p < 0.05),
        "algorithms":  names,
        "n_runs":      min_len,
    }


# ── Average Rank (for CD diagrams) ────────────────────────────────────────────

def critical_difference_ranks(
    multi_results: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, float]:
    """
    Compute average rank of each algorithm across runs.
    Rank 1 = best (lowest quality) per run.
    Used for critical difference diagrams (Phase 3 visualization).
    """
    names  = list(multi_results.keys())
    n_runs = min(len(v) for v in multi_results.values())
    rank_sums = {n: 0.0 for n in names}

    for i in range(n_runs):
        qualities = {n: multi_results[n][i]["best_quality"] for n in names}
        sorted_names = sorted(qualities, key=qualities.get)
        for rank, name in enumerate(sorted_names, start=1):
            rank_sums[name] += rank

    return {n: rank_sums[n] / n_runs for n in names}


# ── Full Phase-3 Report ───────────────────────────────────────────────────────

def phase3_report(
    multi_results: Dict[str, List[Dict[str, Any]]],
    reference: str = "HAQOA",
    baseline_quality: Optional[float] = None,
    width: int = 78,
) -> str:
    """
    Produce the complete Phase-3 statistical report as a formatted string.
    """
    sep  = "═" * width
    sep2 = "─" * width
    lines = [sep, "  HAQOA — Phase 3 Statistical Validation Report", sep]

    # ── Per-algorithm summary ────────────────────────────────────────────────
    lines += ["", "  [1] Multi-Run Summary Statistics", sep2]
    hdr = (f"{'Algorithm':<12} {'Mean':>9} {'Std':>7} {'Best':>9} "
           f"{'Worst':>9} {'CI-low':>9} {'CI-hi':>9} {'n':>4}")
    lines += [hdr, sep2]

    for name, runs in multi_results.items():
        s = summarise_multi_run(runs)
        gap = f" ({compute_gap(s['mean'], baseline_quality):+.1f}%)" if baseline_quality else ""
        lines.append(
            f"{name:<12} {s['mean']:>9.2f} {s['std']:>7.2f} {s['best']:>9.2f} "
            f"{s['worst']:>9.2f} {s['lower']:>9.2f} {s['upper']:>9.2f} {s['n']:>4}"
            + gap
        )
    lines += [sep2]

    # ── Friedman test ────────────────────────────────────────────────────────
    lines += ["", "  [2] Friedman Test (global — H0: all algorithms equal)", sep2]
    fr = friedman_test(multi_results)
    if "error" in fr:
        lines.append(f"  {fr['error']}")
    else:
        sig_str = "REJECT H0 (algorithms differ significantly)" if fr["significant"] \
                  else "FAIL TO REJECT H0"
        lines += [
            f"  χ² = {fr['statistic']:.4f}   p = {fr['p_value']:.4f}   → {sig_str}",
            f"  Algorithms tested: {', '.join(fr['algorithms'])}   n_runs = {fr['n_runs']}",
        ]
    lines += [sep2]

    # ── Pairwise Wilcoxon ────────────────────────────────────────────────────
    lines += ["", "  [3] Pairwise Wilcoxon Tests"]
    lines.append(pairwise_significance_table(multi_results, reference=reference, width=width))

    # ── Average ranks ────────────────────────────────────────────────────────
    lines += ["", "  [4] Average Rank Across Runs (1 = best)", sep2]
    ranks = critical_difference_ranks(multi_results)
    for name, rank in sorted(ranks.items(), key=lambda x: x[1]):
        bar = "█" * int(rank * 4)
        lines.append(f"  {name:<12} {rank:5.2f}  {bar}")
    lines += [sep2, sep]

    return "\n".join(lines)
