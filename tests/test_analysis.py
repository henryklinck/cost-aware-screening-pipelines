"""Tests for correlation/subgroup/bootstrap analysis (spec sections 22, 23, 26)."""

import numpy as np
import pytest

from screening import analysis
from screening.metrics import top_k_recall
from screening.models import build_simulated_models, default_model_specs
from screening.simulation import make_library


def test_residual_correlation_recovers_injected_error_correlation():
    """The whole point of spec section 22: residual (not raw) correlation reflects the
    true error dependence we set."""
    lib = make_library(n_molecules=60000, prevalence=0.02, seed=1)
    rho = 0.6
    models = build_simulated_models(lib, default_model_specs(), correlation=rho, seed=2)
    ec = analysis.error_correlations({k: m.scores for k, m in models.items()},
                                     lib.y_true, n_bins=12)
    # Residual correlations should sit near the injected rho (well below raw rank corr).
    resid = ec["residual_corr"].to_numpy()
    assert np.all(resid < ec["rank_corr"].to_numpy() + 1e-9)
    assert abs(resid.mean() - rho) < 0.12


def test_independent_models_have_low_residual_correlation():
    lib = make_library(n_molecules=60000, prevalence=0.02, seed=1)
    models = build_simulated_models(lib, default_model_specs(), correlation=0.0, seed=2)
    ec = analysis.error_correlations({k: m.scores for k, m in models.items()},
                                     lib.y_true, n_bins=12)
    assert ec["residual_corr"].abs().mean() < 0.1


def test_within_bin_correlation_columns():
    lib = make_library(n_molecules=20000, prevalence=0.05, seed=0)
    models = build_simulated_models(lib, default_model_specs(), correlation=0.4, seed=1)
    wb = analysis.within_bin_error_correlation(models["A"].scores, models["B"].scores,
                                               lib.y_true, n_bins=8)
    assert {"bin", "n", "residual_corr"} <= set(wb.columns)
    assert len(wb) >= 1


def test_subgroup_error_reports_per_group():
    lib = make_library(n_molecules=20000, prevalence=0.05, seed=0)
    models = build_simulated_models(lib, default_model_specs(), correlation=0.0, seed=1)
    subgroups = {"affinity_bin": analysis.bin_labels(lib.y_true, 4, "aff")}
    se = analysis.subgroup_error(models["C"].scores, lib.y_true, lib.is_hit, subgroups)
    assert {"subgroup", "value", "n", "mae", "rmse", "spearman"} <= set(se.columns)
    assert (se["n"] >= 3).all()


def test_grouped_bootstrap_ci_brackets_point():
    lib = make_library(n_molecules=30000, prevalence=0.02, seed=3)
    models = build_simulated_models(lib, default_model_specs(), correlation=0.0, seed=4)
    groups = analysis.bin_labels(lib.y_true, 20, "g")

    def stat(idx):
        return top_k_recall(models["D"].scores[idx], lib.is_hit[idx], 0.02)

    ci = analysis.grouped_bootstrap_ci(groups, stat, n_boot=200, seed=0)
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]
    assert 0.0 <= ci["ci_low"] <= ci["ci_high"] <= 1.0
    assert ci["n_groups"] == 20
