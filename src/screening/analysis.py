"""Model-correlation, error-subgroup, and bootstrap analyses (spec sections 22, 23, 26).

These tie the pipeline results back to the project's statistical hypothesis (spec section
24): the best pipeline depends on how models' *errors* relate, not just their individual
accuracy.  Everything operates on arrays, so it works identically for simulated and real
models.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .metrics import _spearman, top_k_recall


# ---------------------------------------------------------------------------
# Model error correlation (spec section 22)
# ---------------------------------------------------------------------------
def _regression_residuals(scores: np.ndarray, y: np.ndarray, degree: int = 1) -> np.ndarray:
    """Residuals of ``scores`` after regressing out a polynomial trend in ``y``.

    Removing ``E[score | y]`` by (polynomial) regression strips the *shared* affinity
    signal completely -- unlike coarse binning, it also removes the within-stratum
    affinity variation that otherwise leaks a spurious positive correlation between the
    residuals of two accurate models.  What remains is each model's idiosyncratic error
    conditional on true affinity, which is what spec section 22 says to correlate.
    """
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.vander(y, N=degree + 1, increasing=True)  # [1, y, y^2, ...]
    beta, *_ = np.linalg.lstsq(X, scores, rcond=None)
    return scores - X @ beta


def _conditional_residuals(scores: np.ndarray, y: np.ndarray, n_bins: int) -> np.ndarray:
    """Residuals of ``scores`` after removing the mean score within each ``y`` bin.

    Used for the *within-bin* view, where each stratum is already narrow in ``y``.
    """
    bins = pd.qcut(y, q=min(n_bins, len(np.unique(y))), labels=False, duplicates="drop")
    resid = np.array(scores, dtype=float).copy()
    for b in np.unique(bins):
        m = bins == b
        resid[m] = resid[m] - resid[m].mean()
    return resid


def error_correlations(
    scores: Dict[str, np.ndarray],
    y: np.ndarray,
    n_bins: int = 10,
    degree: int = 1,
) -> pd.DataFrame:
    """Pairwise model-error relationships (spec section 22).

    For each model pair returns:
      * ``rank_corr``      -- Spearman of the raw scores (high even when both are simply
                              correct, so *not* the right independence measure alone);
      * ``residual_corr``  -- Pearson of affinity-conditional residuals (the honest
                              "do they make the *same mistakes*?" measure), with the
                              affinity trend removed by degree-``degree`` regression.

    High ``rank_corr`` with low ``residual_corr`` means the models agree on the easy
    ranking but err independently -- exactly the regime where combining them helps.
    ``n_bins`` is retained for API symmetry with the within-bin analysis.
    """
    y = np.asarray(y, dtype=float)
    resid = {name: _regression_residuals(s, y, degree) for name, s in scores.items()}
    rows = []
    for a, b in combinations(scores.keys(), 2):
        rows.append({
            "model_a": a,
            "model_b": b,
            "rank_corr": _spearman(scores[a], scores[b]),
            "residual_corr": float(np.corrcoef(resid[a], resid[b])[0, 1]),
        })
    return pd.DataFrame(rows)


def within_bin_error_correlation(
    score_a: np.ndarray,
    score_b: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Residual error correlation of two models *within* affinity bins (spec section 22)."""
    y = np.asarray(y, dtype=float)
    bins = pd.qcut(y, q=min(n_bins, len(np.unique(y))), labels=False, duplicates="drop")
    ra = _conditional_residuals(score_a, y, n_bins)
    rb = _conditional_residuals(score_b, y, n_bins)
    rows = []
    for b in np.unique(bins):
        m = bins == b
        if m.sum() < 3:
            continue
        rows.append({
            "bin": int(b),
            "n": int(m.sum()),
            "affinity_low": float(y[m].min()),
            "affinity_high": float(y[m].max()),
            "residual_corr": float(np.corrcoef(ra[m], rb[m])[0, 1]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Error analysis by subgroup (spec section 23)
# ---------------------------------------------------------------------------
def subgroup_error(
    scores: np.ndarray,
    y: np.ndarray,
    is_hit: np.ndarray,
    subgroups: Dict[str, np.ndarray],
    top_k_fraction: float = 0.05,
) -> pd.DataFrame:
    """Per-subgroup model error/recovery (spec section 23).

    ``subgroups`` maps a subgroup name (e.g. ``"mw_bin"``, ``"scaffold"``, ``"target"``)
    to a per-molecule label array.  For each subgroup value we report MAE, RMSE, Spearman
    vs. ``y``, and top-k recall -- revealing whether a model fails on particular
    chemistry/targets, and (across models) whether they fail on the *same* subgroups.
    """
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=float)
    is_hit = np.asarray(is_hit, dtype=bool)
    rows = []
    for gname, labels in subgroups.items():
        labels = np.asarray(labels)
        for val in pd.unique(labels):
            m = labels == val
            n = int(m.sum())
            if n < 3:
                continue
            err = scores[m] - y[m]
            rows.append({
                "subgroup": gname,
                "value": val,
                "n": n,
                "n_hits": int(is_hit[m].sum()),
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "spearman": _spearman(scores[m], y[m]),
                f"recall_top_{top_k_fraction:g}": (
                    top_k_recall(scores[m], is_hit[m], top_k_fraction)
                    if is_hit[m].any() else float("nan")
                ),
            })
    return pd.DataFrame(rows)


def bin_labels(values: np.ndarray, n_bins: int = 5, prefix: str = "bin") -> np.ndarray:
    """Quantile-bin a continuous property into string labels (for subgroup analysis)."""
    b = pd.qcut(np.asarray(values, dtype=float), q=n_bins, labels=False,
                duplicates="drop")
    return np.array([f"{prefix}_{x}" for x in b])


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals at the correct unit (spec section 26)
# ---------------------------------------------------------------------------
def grouped_bootstrap_ci(
    group_labels: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Bootstrap a statistic by resampling **whole groups** with replacement.

    The independent unit is the group (target or scaffold), *not* the individual
    measurement (spec section 26): molecules sharing a scaffold or target are correlated,
    so resampling rows would understate uncertainty.  ``statistic`` receives the array of
    row indices belonging to the resampled groups and returns a scalar.
    """
    group_labels = np.asarray(group_labels)
    groups = pd.unique(group_labels)
    group_to_rows = {g: np.where(group_labels == g)[0] for g in groups}
    rng = np.random.default_rng(seed)

    point = statistic(np.arange(len(group_labels)))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        idx = np.concatenate([group_to_rows[g] for g in sampled])
        boots[i] = statistic(idx)

    alpha = (1.0 - ci) / 2.0
    return {
        "point": float(point),
        "mean": float(np.nanmean(boots)),
        "ci_low": float(np.nanquantile(boots, alpha)),
        "ci_high": float(np.nanquantile(boots, 1.0 - alpha)),
        "n_groups": int(len(groups)),
        "n_boot": n_boot,
    }
