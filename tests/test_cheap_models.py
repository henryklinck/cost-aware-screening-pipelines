"""Tests for the runnable real cheap models (spec section 14)."""

import numpy as np
import pytest

pytest.importorskip("rdkit")

from screening.data import CompoundLibrary, load_bindingdb
from screening.models import FingerprintSimilarity, RDKitPropertyFilter
from screening.models.base import ScreeningModel
from screening.pipelines import run_pipeline


def test_property_filter_scores_and_interface():
    smiles = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "invalid_smiles"]
    m = RDKitPropertyFilter(smiles)
    assert isinstance(m, ScreeningModel)
    scores = m.predict(np.arange(len(smiles)))
    assert scores.shape == (4,)
    # Invalid molecule ranks last (-inf).
    assert np.isneginf(scores[-1])
    # QED scores for valid molecules are in [0, 1].
    valid = scores[:3]
    assert np.all((valid >= 0) & (valid <= 1))


def test_fingerprint_similarity_to_self_is_high():
    smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
    # Reference = ethanol; ethanol query should score 1.0.
    m = FingerprintSimilarity(smiles, reference_actives=["CCO"])
    scores = m.predict(np.arange(3))
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] < 1.0
    assert m.n_references == 1


def test_fingerprint_no_references_is_uninformative():
    smiles = ["CCO", "c1ccccc1"]
    m = FingerprintSimilarity(smiles, reference_actives=[])
    scores = m.predict(np.arange(2))
    assert np.allclose(scores, 0.0)  # no -inf, all equal


def test_measured_cost_is_nonneg_and_scales():
    smiles = ["CCO"] * 20
    m = RDKitPropertyFilter(smiles)
    assert m.estimate_cost(0) == 0.0
    assert m.estimate_cost(10) == pytest.approx(10 * m._cost_per_molecule)
    assert m.estimate_cost(20) >= m.estimate_cost(10)


def test_cheap_models_run_in_pipeline(bindingdb_sample_path):
    ds = load_bindingdb(bindingdb_sample_path, "Kd")
    lib = CompoundLibrary(ds.df, endpoint="pKd", hit_threshold=7.0)
    rd = RDKitPropertyFilter(lib.smiles)
    fp = FingerprintSimilarity(lib.smiles, reference_actives=["CCO", "c1ccccc1"])
    res = run_pipeline(lib, {"rdkit_filter": rd, "fingerprint": fp},
                       ["rdkit_filter", "fingerprint"], [0.5, 0.2])
    # The unchanged Phase-1 pipeline executes real models end to end.
    assert res.final_molecules >= 1
    assert res.total_cost >= 0
    assert len(np.unique(res.survivors)) == res.final_molecules
