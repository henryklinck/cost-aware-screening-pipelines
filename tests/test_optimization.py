"""Tests for enumeration, search, and Pareto analysis (spec section 30)."""

import numpy as np
import pandas as pd
import pytest

from screening.models import build_simulated_models, default_model_specs
from screening.optimization import (
    best_under_budget,
    constrain_to_test_budget,
    enumerate_pipelines,
    pareto_frontier,
    pareto_mask,
    recall_at_budgets,
    search_all,
    threshold_grid,
)
from screening.simulation import make_library


def test_enumerate_counts_and_no_repeats():
    names = ["A", "B", "C"]
    p1 = enumerate_pipelines(names, 1, 1)
    p2 = enumerate_pipelines(names, 2, 2)
    p3 = enumerate_pipelines(names, 3, 3)
    assert len(p1) == 3       # 3 singles
    assert len(p2) == 6       # ordered pairs P(3,2)
    assert len(p3) == 6       # ordered triples P(3,3)
    # No model repeats within a pipeline.
    for combo in p1 + p2 + p3:
        assert len(set(combo)) == len(combo)


def test_threshold_grid_size():
    grid = threshold_grid(2, [0.1, 0.5])
    assert len(grid) == 4
    assert (0.1, 0.5) in grid


def test_pareto_mask_simple():
    df = pd.DataFrame({
        "total_cost": [1, 2, 3, 2],
        "recall": [0.5, 0.5, 0.9, 0.8],
    })
    mask = pareto_mask(df)
    # (cost1,rec.5) dominates (cost2,rec.5). (cost2,rec.8) and (cost3,rec.9) survive.
    assert mask.tolist() == [True, False, True, True]


def test_pareto_frontier_is_monotone():
    df = pd.DataFrame({
        "total_cost": [1, 2, 3, 4, 5],
        "recall": [0.2, 0.5, 0.4, 0.7, 0.6],
    })
    pf = pareto_frontier(df)
    # Frontier sorted by cost must have non-decreasing recall.
    assert list(pf["recall"]) == sorted(pf["recall"])


def test_best_under_budget():
    df = pd.DataFrame({
        "total_cost": [10, 20, 30],
        "recall": [0.3, 0.6, 0.9],
    })
    assert best_under_budget(df, 25)["recall"] == 0.6
    assert best_under_budget(df, 5) is None


def test_search_all_shapes_and_constraint():
    lib = make_library(n_molecules=8000, prevalence=0.02, seed=0)
    models = build_simulated_models(lib, default_model_specs(), correlation=0.0, seed=1)
    df = search_all(lib, models, min_stages=1, max_stages=2,
                    retention_values=[0.1, 0.5])
    # 20 singles/pairs * grid; every row has required columns.
    assert {"pipeline", "total_cost", "recall", "final_molecules"} <= set(df.columns)
    assert (df["recall"] <= 1.0 + 1e-9).all()
    K = int(0.05 * lib.n)
    c = constrain_to_test_budget(df, K)
    assert (c["final_molecules"] <= K).all()


def test_multistage_beats_single_at_matched_cost():
    """The central hypothesis on synthetic data: a cascade recovers more hits per
    unit compute than the best single model can at the same budget."""
    lib = make_library(n_molecules=60000, prevalence=0.01, sigma_assay=0.3, seed=42)
    models = build_simulated_models(lib, default_model_specs(), correlation=0.3, seed=7)
    df = search_all(lib, models, min_stages=1, max_stages=3)
    K = int(0.01 * lib.n)
    c = constrain_to_test_budget(df, K)

    # Best single model's recall at any cost.
    singles = c[c["n_stages"] == 1]
    best_single_recall = singles["recall"].max()
    best_single_cost = singles.loc[singles["recall"].idxmax(), "total_cost"]

    # A multi-stage pipeline should match that recall for strictly less compute.
    multi = c[c["n_stages"] >= 2]
    match = multi[multi["recall"] >= best_single_recall]
    assert not match.empty
    assert match["total_cost"].min() < best_single_cost


def test_recall_at_budgets_infeasible_returns_nan():
    df = pd.DataFrame({"total_cost": [100.0], "recall": [0.5],
                       "pipeline": ["A(0.10)"]})
    out = recall_at_budgets(df, [10.0, 1000.0])
    assert np.isnan(out.iloc[0]["recall"])          # budget too small
    assert out.iloc[1]["recall"] == 0.5             # budget fits
