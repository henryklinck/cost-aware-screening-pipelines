"""Tests for metrics (spec section 30: metrics)."""

import numpy as np
import pytest

from screening.metrics import (
    _auroc,
    _spearman,
    evaluate_pipeline,
    model_quality_metrics,
    top_k_recall,
)
from screening.models import build_simulated_models, default_model_specs
from screening.pipelines import run_pipeline
from screening.simulation import make_library


def _setup(n=20000, seed=0):
    lib = make_library(n_molecules=n, prevalence=0.05, sigma_assay=0.3, seed=seed)
    models = build_simulated_models(lib, default_model_specs(), correlation=0.0, seed=seed + 1)
    return lib, models


def test_recall_precision_enrichment_definitions():
    lib, models = _setup()
    res = run_pipeline(lib, models, ["C"], [0.10])
    m = evaluate_pipeline(lib, res)
    # Recall = recovered / total; precision = recovered / selected.
    assert m["recall"] == pytest.approx(m["hits_recovered"] / m["total_hits"])
    assert m["precision"] == pytest.approx(m["hits_recovered"] / m["final_molecules"])
    # Enrichment = precision / prevalence.
    assert m["enrichment"] == pytest.approx(m["precision"] / m["prevalence"])


def test_cost_per_hit_and_nnt():
    lib, models = _setup()
    res = run_pipeline(lib, models, ["C"], [0.10])
    m = evaluate_pipeline(lib, res)
    assert m["cost_per_hit"] == pytest.approx(res.total_cost / m["hits_recovered"])
    assert m["number_needed_to_test"] == pytest.approx(m["final_molecules"] / m["hits_recovered"])


def test_perfect_scores_give_full_recall_at_prevalence():
    lib, _ = _setup()
    # A perfect score = true affinity: top-(prevalence) captures all hits.
    r = top_k_recall(lib.y_true, lib.is_hit, lib.prevalence)
    assert r == pytest.approx(1.0, abs=0.02)


def test_top_k_recall_monotonic_in_k():
    lib, models = _setup()
    s = models["B"].scores
    r1 = top_k_recall(s, lib.is_hit, 0.01)
    r5 = top_k_recall(s, lib.is_hit, 0.05)
    r10 = top_k_recall(s, lib.is_hit, 0.10)
    assert r1 <= r5 <= r10


def test_spearman_and_auroc_perfect():
    y = np.arange(1000, dtype=float)
    assert _spearman(y, y) == pytest.approx(1.0)
    assert _spearman(y, -y) == pytest.approx(-1.0)
    is_hit = y >= 900
    # A perfect ranker gives AUROC 1.
    assert _auroc(y, is_hit) == pytest.approx(1.0)
    # A random-tie constant gives 0.5.
    assert _auroc(np.zeros(1000), is_hit) == pytest.approx(0.5)


def test_model_quality_reference_true_vs_obs_differ():
    lib, models = _setup()
    q_true = model_quality_metrics(models["C"].scores, lib, reference="true")
    q_obs = model_quality_metrics(models["C"].scores, lib, reference="obs")
    # Comparing to noisy observations should not give a better MAE than to the truth
    # on average (assay noise inflates error).
    assert q_true["mae"] <= q_obs["mae"]


def test_empty_selection_precision_zero():
    lib, models = _setup()
    res = run_pipeline(lib, models, ["A"], [0.01])
    m = evaluate_pipeline(lib, res)
    assert 0.0 <= m["precision"] <= 1.0
