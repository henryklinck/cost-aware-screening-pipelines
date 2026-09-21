"""Tests for the synthetic world (spec section 30: data)."""

import numpy as np
import pytest

from screening.simulation import (
    MoleculeLibrary,
    add_assay_noise,
    format_molecule_id,
    make_library,
    threshold_for_prevalence,
)


def test_molecule_id_format():
    assert format_molecule_id(1) == "molecule_000001"
    assert format_molecule_id(123456) == "molecule_123456"


def test_true_affinity_is_continuous_not_binary():
    lib = make_library(n_molecules=5000, prevalence=0.01, seed=0)
    # The latent truth must be continuous: many distinct values, not just {0,1}.
    assert len(np.unique(lib.y_true)) > 1000
    assert lib.y_true.dtype.kind == "f"


def test_assay_noise_separates_true_and_observed():
    y_true = np.zeros(10000)
    y_obs = add_assay_noise(y_true, sigma_assay=0.5, seed=1)
    # Observed differs from true, with roughly the requested spread.
    assert not np.allclose(y_true, y_obs)
    assert abs(np.std(y_obs) - 0.5) < 0.05
    # Zero assay noise reproduces the truth exactly.
    assert np.allclose(add_assay_noise(y_true, 0.0, seed=1), y_true)


def test_prevalence_target_is_met():
    lib = make_library(n_molecules=100000, prevalence=0.02, seed=2)
    assert abs(lib.prevalence - 0.02) < 0.002


def test_threshold_for_prevalence_quantile():
    y = np.linspace(0, 1, 10001)
    thr = threshold_for_prevalence(y, 0.1)
    assert abs((y >= thr).mean() - 0.1) < 0.01


def test_reproducibility_same_seed():
    a = make_library(n_molecules=1000, prevalence=0.05, seed=7)
    b = make_library(n_molecules=1000, prevalence=0.05, seed=7)
    assert np.allclose(a.y_true, b.y_true)
    assert np.allclose(a.y_obs, b.y_obs)


def test_different_seed_differs():
    a = make_library(n_molecules=1000, prevalence=0.05, seed=7)
    b = make_library(n_molecules=1000, prevalence=0.05, seed=8)
    assert not np.allclose(a.y_true, b.y_true)


def test_hits_on_obs_uses_observed_affinity():
    lib = make_library(n_molecules=5000, prevalence=0.05, hits_on="obs", seed=3)
    assert lib.hits_on == "obs"
    assert np.array_equal(lib.is_hit, lib.y_obs >= lib.hit_threshold)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        make_library(n_molecules=100)  # neither prevalence nor threshold
    with pytest.raises(ValueError):
        make_library(n_molecules=100, prevalence=0.1, hit_threshold=1.0)  # both
    with pytest.raises(ValueError):
        add_assay_noise(np.zeros(10), sigma_assay=-1.0)
