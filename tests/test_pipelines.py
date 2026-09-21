"""Tests for pipeline execution (spec section 30: pipelines)."""

import numpy as np
import pytest

from screening.models import build_simulated_models, default_model_specs
from screening.pipelines import run_pipeline, select_top_fraction
from screening.simulation import make_library


def _setup(n=20000, corr=0.0, seed=0):
    lib = make_library(n_molecules=n, prevalence=0.02, sigma_assay=0.3, seed=seed)
    models = build_simulated_models(lib, default_model_specs(), correlation=corr, seed=seed + 1)
    return lib, models


def test_thresholding_retention_counts():
    lib, models = _setup()
    res = run_pipeline(lib, models, ["A", "C"], [0.20, 0.10])
    # After A: 20% of n. After C: 10% of that.
    n_after_a = int(lib.n * 0.20)
    assert res.stages[0].molecules_after == n_after_a
    assert res.stages[1].molecules_after == int(n_after_a * 0.10)
    assert res.final_molecules == res.stages[-1].molecules_after


def test_survivors_are_strict_subset_no_double_entry():
    lib, models = _setup()
    res = run_pipeline(lib, models, ["A", "C", "E"], [0.3, 0.2, 0.5])
    survivors = res.survivors
    # No molecule appears twice; all are valid library indices.
    assert len(np.unique(survivors)) == len(survivors)
    assert survivors.min() >= 0 and survivors.max() < lib.n
    # Each stage's survivors are a subset of the previous stage's inputs.
    prev = set(lib.indices.tolist())
    running = lib.indices
    for name, frac in zip(res.pipeline, res.thresholds):
        running = select_top_fraction(models[name], running, frac,
                                      is_full_library=(running is lib.indices),
                                      library_size=lib.n)
        assert set(running.tolist()).issubset(prev)
        prev = set(running.tolist())


def test_cost_is_sum_over_stages_on_entering_counts():
    lib, models = _setup()
    res = run_pipeline(lib, models, ["A", "E"], [0.10, 0.50])
    cost_a = models["A"].estimate_cost(lib.n)
    n_after_a = int(lib.n * 0.10)
    cost_e = models["E"].estimate_cost(n_after_a)
    assert res.total_cost == pytest.approx(cost_a + cost_e)


def test_ordering_changes_cost_dramatically():
    """Cheap-first vs expensive-first: same models, very different compute."""
    lib, models = _setup()
    cheap_first = run_pipeline(lib, models, ["A", "E"], [0.10, 0.10])
    exp_first = run_pipeline(lib, models, ["E", "A"], [0.10, 0.10])
    # Expensive model E runs on the whole library when placed first -> far costlier.
    assert exp_first.total_cost > cheap_first.total_cost


def test_top_fraction_uses_global_ranking_on_full_library():
    lib, models = _setup()
    m = models["B"]
    sel = select_top_fraction(m, lib.indices, 0.05, is_full_library=True,
                              library_size=lib.n)
    # Must equal the true top-5% by score.
    k = int(lib.n * 0.05)
    expected = np.argsort(-m.scores, kind="stable")[:k]
    assert np.array_equal(np.sort(sel), np.sort(expected))


def test_at_least_one_molecule_retained():
    lib, models = _setup(n=100)
    res = run_pipeline(lib, models, ["A"], [0.0001])  # rounds toward 0
    assert res.final_molecules >= 1


def test_invalid_pipeline_arguments():
    lib, models = _setup(n=1000)
    with pytest.raises(ValueError):
        run_pipeline(lib, models, ["A", "C"], [0.1])  # length mismatch
    with pytest.raises(ValueError):
        run_pipeline(lib, models, [], [])  # empty
    with pytest.raises(ValueError):
        run_pipeline(lib, models, ["Z"], [0.1])  # unknown model
    with pytest.raises(ValueError):
        run_pipeline(lib, models, ["A"], [1.5])  # fraction > 1
