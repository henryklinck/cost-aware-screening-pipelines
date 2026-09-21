"""Tests for simulated models (spec section 30: models)."""

import numpy as np
import pytest

from screening.models import (
    ModelSpec,
    SimulatedModel,
    build_simulated_models,
    default_model_specs,
)
from screening.models.base import ScreeningModel
from screening.simulation import make_library


def _lib(n=20000, seed=0):
    return make_library(n_molecules=n, prevalence=0.02, sigma_assay=0.3, seed=seed)


def test_predict_returns_one_score_per_molecule():
    lib = _lib()
    models = build_simulated_models(lib, default_model_specs(), correlation=0.0, seed=1)
    idx = np.arange(100)
    scores = models["A"].predict(idx)
    assert scores.shape == (100,)


def test_predict_is_deterministic_and_consistent_across_subsets():
    lib = _lib()
    models = build_simulated_models(lib, default_model_specs(), correlation=0.0, seed=1)
    m = models["C"]
    # Same molecule scored via different subsets gets the same score every time.
    full = m.predict(lib.indices)
    subset_idx = np.array([5, 100, 7777])
    assert np.allclose(m.predict(subset_idx), full[subset_idx])
    assert np.allclose(m.predict(subset_idx), m.predict(subset_idx))


def test_all_models_implement_interface():
    lib = _lib()
    models = build_simulated_models(lib, default_model_specs(), correlation=0.0, seed=1)
    for m in models.values():
        assert isinstance(m, ScreeningModel)


def test_cost_scales_with_molecule_count():
    lib = _lib()
    models = build_simulated_models(lib, default_model_specs(), correlation=0.0, seed=1)
    m = models["E"]
    assert m.estimate_cost(0) == 0.0
    assert m.estimate_cost(1000) == 1000 * m.spec.cost_per_molecule
    assert m.estimate_cost(2000) == 2 * m.estimate_cost(1000)
    with pytest.raises(ValueError):
        m.estimate_cost(-1)


def test_accuracy_increases_from_A_to_E():
    """Lower-noise (more expensive) models must rank hits better."""
    from screening.metrics import model_quality_metrics

    lib = _lib(n=40000)
    models = build_simulated_models(lib, default_model_specs(), correlation=0.0, seed=2)
    spearmans = [model_quality_metrics(models[n].scores, lib)["spearman"]
                 for n in ["A", "B", "C", "D", "E"]]
    assert spearmans == sorted(spearmans)  # monotonically non-decreasing


def test_error_correlation_is_reproduced():
    """The requested cross-model error correlation should hold empirically."""
    lib = _lib(n=100000)
    rho = 0.6
    models = build_simulated_models(lib, default_model_specs(), correlation=rho, seed=3)
    # Model errors relative to the (scaled) truth.
    err_a = models["A"].scores - models["A"].spec.gain * lib.y_true
    err_b = models["B"].scores - models["B"].spec.gain * lib.y_true
    empirical = np.corrcoef(err_a, err_b)[0, 1]
    assert abs(empirical - rho) < 0.03


def test_independent_errors_have_low_correlation():
    lib = _lib(n=100000)
    models = build_simulated_models(lib, default_model_specs(), correlation=0.0, seed=4)
    err_a = models["A"].scores - lib.y_true
    err_c = models["C"].scores - lib.y_true
    assert abs(np.corrcoef(err_a, err_c)[0, 1]) < 0.03


def test_duplicate_model_names_rejected():
    lib = _lib(n=1000)
    specs = [ModelSpec("X", 0.5, 1.0), ModelSpec("X", 0.4, 2.0)]
    with pytest.raises(ValueError):
        build_simulated_models(lib, specs, correlation=0.0, seed=0)


def test_invalid_equicorrelation_rejected():
    lib = _lib(n=1000)
    specs = default_model_specs()
    with pytest.raises(ValueError):
        build_simulated_models(lib, specs, correlation=1.5, seed=0)
