"""Pipeline enumeration, threshold optimization, and Pareto analysis (spec 7, 19, 20).

With only a handful of models we can afford to *exhaustively* enumerate every ordered
combination and, for each, grid-search retention thresholds (spec section 7: "Do not
arbitrarily limit the ordering").  The primary decision problem (spec section 20) is:

    maximize   recall
    subject to computational cost <= budget B

which we answer for several budgets by scanning the full evaluated table.
"""

from __future__ import annotations

from itertools import permutations, product
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .metrics import evaluate_pipeline
from .models.base import ScreeningModel
from .pipelines import run_pipeline
from .simulation import MoleculeLibrary

# Retention rates from spec section 19.
DEFAULT_RETENTION_GRID = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50)


def enumerate_pipelines(
    model_names: Sequence[str],
    min_stages: int = 1,
    max_stages: int = 3,
) -> List[Tuple[str, ...]]:
    """All ordered pipelines using each model at most once.

    Ordering matters (an expensive model behaves differently first vs. last), so we
    enumerate *permutations*, not just subsets.  A model never repeats within a pipeline.
    """
    names = list(model_names)
    max_stages = min(max_stages, len(names))
    out: List[Tuple[str, ...]] = []
    for length in range(min_stages, max_stages + 1):
        out.extend(permutations(names, length))
    return out


def threshold_grid(
    n_stages: int,
    retention_values: Sequence[float] = DEFAULT_RETENTION_GRID,
) -> List[Tuple[float, ...]]:
    """Cartesian product of retention fractions across ``n_stages`` stages."""
    return list(product(retention_values, repeat=n_stages))


def evaluate_configuration(
    library: MoleculeLibrary,
    models: Dict[str, ScreeningModel],
    pipeline: Sequence[str],
    thresholds: Sequence[float],
) -> Dict[str, float]:
    """Run one (pipeline, thresholds) configuration and return its metrics row."""
    result = run_pipeline(library, models, pipeline, thresholds)
    return evaluate_pipeline(library, result)


def search_all(
    library: MoleculeLibrary,
    models: Dict[str, ScreeningModel],
    min_stages: int = 1,
    max_stages: int = 3,
    retention_values: Sequence[float] = DEFAULT_RETENTION_GRID,
    model_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Exhaustively evaluate every pipeline x threshold configuration.

    Returns a tidy :class:`pandas.DataFrame`, one row per configuration, with all
    metrics from :func:`~screening.metrics.evaluate_pipeline`.
    """
    if model_names is None:
        model_names = list(models.keys())

    rows: List[Dict[str, float]] = []
    for pipeline in enumerate_pipelines(model_names, min_stages, max_stages):
        for thresholds in threshold_grid(len(pipeline), retention_values):
            rows.append(evaluate_configuration(library, models, pipeline, thresholds))
    return pd.DataFrame(rows)


def constrain_to_test_budget(
    df: pd.DataFrame,
    test_budget: int,
    final_col: str = "final_molecules",
) -> pd.DataFrame:
    """Keep only configurations whose final shortlist fits an experimental testing budget.

    This is the constraint that makes "recall vs computational cost" a *meaningful*
    trade-off.  Without it, recall is trivially maximised by cheaply keeping a huge
    fraction of the library (e.g. the cheapest model at 50% retention recovers ~all
    hits) -- which is useless because you could never assay that many molecules.

    In real screening the final candidates are what you pay to test experimentally, so
    their count is capped by a testing budget ``K``.  Within ``final <= K``, achieving
    high recall in a small shortlist *requires* good (and usually more expensive)
    screening, so compute genuinely buys recall.
    """
    return df[df[final_col] <= test_budget].copy()


def pareto_mask(
    df: pd.DataFrame,
    cost_col: str = "total_cost",
    recall_col: str = "recall",
) -> np.ndarray:
    """Boolean mask selecting the cost/recall Pareto frontier (min cost, max recall).

    A configuration is Pareto-optimal if no other configuration is *at least as cheap*
    and *at least as good* on recall while being strictly better on one of them.
    """
    cost = df[cost_col].to_numpy(dtype=float)
    recall = df[recall_col].to_numpy(dtype=float)
    n = len(df)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        # Does any other point dominate i?
        dominated = (
            (cost <= cost[i]) & (recall >= recall[i]) &
            ((cost < cost[i]) | (recall > recall[i]))
        )
        if dominated.any():
            keep[i] = False
    return keep


def pareto_frontier(
    df: pd.DataFrame,
    cost_col: str = "total_cost",
    recall_col: str = "recall",
) -> pd.DataFrame:
    """Return the Pareto-optimal rows, sorted by cost then recall."""
    frontier = df[pareto_mask(df, cost_col, recall_col)].copy()
    return frontier.sort_values([cost_col, recall_col]).reset_index(drop=True)


def best_under_budget(
    df: pd.DataFrame,
    budget: float,
    cost_col: str = "total_cost",
    recall_col: str = "recall",
) -> Optional[pd.Series]:
    """The maximum-recall configuration whose cost does not exceed ``budget``.

    Ties on recall are broken by lower cost.  Returns ``None`` if nothing fits.
    """
    feasible = df[df[cost_col] <= budget]
    if feasible.empty:
        return None
    order = feasible.sort_values([recall_col, cost_col], ascending=[False, True])
    return order.iloc[0]


def recall_at_budgets(
    df: pd.DataFrame,
    budgets: Iterable[float],
    cost_col: str = "total_cost",
    recall_col: str = "recall",
) -> pd.DataFrame:
    """Best achievable recall at each budget -- the core cost/recall trade-off table."""
    rows = []
    for b in budgets:
        best = best_under_budget(df, b, cost_col, recall_col)
        if best is None:
            rows.append({"budget": b, "recall": float("nan"), "pipeline": None,
                         "total_cost": float("nan")})
        else:
            rows.append({
                "budget": b,
                "recall": float(best[recall_col]),
                "pipeline": best["pipeline"],
                "total_cost": float(best[cost_col]),
            })
    return pd.DataFrame(rows)
