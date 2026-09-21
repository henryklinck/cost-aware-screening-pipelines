"""Evaluation metrics (spec sections 6, 20, 23).

Two families:

* **Pipeline metrics** -- given the final survivors of a pipeline, how well did we
  recover hits and at what cost?  The primary research metric is *recall at a fixed
  computational cost* (spec section 20); the rest are secondary.
* **Model quality metrics** -- given a single model's continuous scores and the ground
  truth, how good is that model on its own (rank correlation, top-k recovery, and the
  classification metrics of spec section 23)?  These describe the *ingredients* whose
  interaction the project studies.

Everything is computed from arrays; nothing here fabricates numbers.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .pipelines import PipelineResult
from .simulation import MoleculeLibrary


def evaluate_pipeline(library: MoleculeLibrary, result: PipelineResult) -> Dict[str, float]:
    """Compute recovery/cost metrics for one pipeline result.

    Returns a flat dict suitable for a results table row.
    """
    is_hit = library.is_hit
    total_hits = int(is_hit.sum())
    n_total = library.n

    survivors = result.survivors
    n_selected = int(len(survivors))
    hits_recovered = int(is_hit[survivors].sum())

    recall = hits_recovered / total_hits if total_hits > 0 else float("nan")
    precision = hits_recovered / n_selected if n_selected > 0 else 0.0
    prevalence = total_hits / n_total if n_total > 0 else float("nan")
    enrichment = (precision / prevalence) if prevalence > 0 else float("nan")

    cost_per_hit = (
        result.total_cost / hits_recovered if hits_recovered > 0 else float("inf")
    )
    # Number needed to test: candidates one must assay to find a single hit.
    nnt = n_selected / hits_recovered if hits_recovered > 0 else float("inf")

    return {
        "pipeline": result.pipeline_label(),
        "stages": "->".join(result.pipeline),
        "thresholds": ",".join(f"{t:.3f}" for t in result.thresholds),
        "n_stages": len(result.pipeline),
        "total_cost": result.total_cost,
        "initial_molecules": n_total,
        "final_molecules": n_selected,
        "total_hits": total_hits,
        "hits_recovered": hits_recovered,
        "recall": recall,
        "precision": precision,
        "prevalence": prevalence,
        "enrichment": enrichment,
        "cost_per_hit": cost_per_hit,
        "number_needed_to_test": nnt,
    }


def top_k_recall(
    scores: np.ndarray,
    is_hit: np.ndarray,
    k_fraction: float,
) -> float:
    """Fraction of all hits captured in the top ``k_fraction`` of ``scores``.

    e.g. ``k_fraction=0.01`` -> recall within the top 1% highest-scoring molecules.
    """
    if not 0.0 < k_fraction <= 1.0:
        raise ValueError("k_fraction must be in (0, 1]")
    total_hits = int(is_hit.sum())
    if total_hits == 0:
        return float("nan")
    n = len(scores)
    k = max(1, int(n * k_fraction))
    top = np.argsort(-scores, kind="stable")[:k]
    return int(is_hit[top].sum()) / total_hits


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation without a SciPy dependency for this hot path."""
    if len(x) < 2:
        return float("nan")
    xr = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)
    yr = np.argsort(np.argsort(y, kind="stable"), kind="stable").astype(float)
    xr -= xr.mean()
    yr -= yr.mean()
    denom = np.sqrt((xr**2).sum() * (yr**2).sum())
    return float((xr * yr).sum() / denom) if denom > 0 else float("nan")


def _auroc(scores: np.ndarray, is_hit: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic."""
    pos = scores[is_hit]
    neg = scores[~is_hit]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Average-rank (mid-rank) assignment so AUROC is unbiased under ties.
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    start = cum - counts
    avg_rank_per_group = (start + cum + 1) / 2.0
    ranks = avg_rank_per_group[inv]
    n_pos = len(pos)
    n_neg = len(neg)
    sum_ranks_pos = ranks[is_hit].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def model_quality_metrics(
    scores: np.ndarray,
    library: MoleculeLibrary,
    reference: str = "true",
    top_k_fractions=(0.01, 0.05, 0.10),
) -> Dict[str, float]:
    """Single-model quality summary (spec section 23).

    Parameters
    ----------
    scores:
        The model's continuous scores over the full library.
    reference:
        ``"true"`` compares against latent affinity, ``"obs"`` against the measured
        affinity.  The two differ by assay noise, which is exactly the true-vs-measured
        distinction the project keeps explicit.
    """
    y = library.y_true if reference == "true" else library.y_obs
    is_hit = library.is_hit

    err = scores - y
    out: Dict[str, float] = {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "pearson": float(np.corrcoef(scores, y)[0, 1]) if len(scores) > 1 else float("nan"),
        "spearman": _spearman(scores, y),
        "auroc": _auroc(scores, is_hit),
    }
    for kf in top_k_fractions:
        out[f"recall_top_{kf:g}"] = top_k_recall(scores, is_hit, kf)
    return out
