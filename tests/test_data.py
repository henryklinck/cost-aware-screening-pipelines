"""Tests for BindingDB preprocessing (spec sections 9, 10, 30: data)."""

import numpy as np
import pytest

pytest.importorskip("rdkit")

from screening import data
from screening.data import CompoundLibrary


def test_parse_measurement_handles_qualifiers():
    assert data.parse_measurement("100") == (100.0, False)
    assert data.parse_measurement(">10000") == (10000.0, True)
    assert data.parse_measurement("<1") == (1.0, True)
    assert data.parse_measurement("") == (None, False)
    assert data.parse_measurement("nan") == (None, False)


def test_pkd_conversion():
    # Kd = 1 nM -> pKd = 9; Kd = 100 nM -> pKd = 7.
    assert data.nm_to_p_endpoint(1.0) == pytest.approx(9.0)
    assert data.nm_to_p_endpoint(100.0) == pytest.approx(7.0)


def test_load_pipeline_shapes_and_provenance(bindingdb_sample_path):
    ds = data.load_bindingdb(bindingdb_sample_path, measurement_type="Kd")
    s = ds.summary()
    assert ds.endpoint == "pKd"
    assert s["n_raw"] >= s["n_after_type_filter"] >= s["n_after_valid_smiles"] >= s["n_after_dedup"]
    # Required tidy columns preserved (spec section 9).
    for col in ["smiles_std", "target_id", "measurement_type", "pKd", "value_nm",
                "year", "scaffold", "n_measurements"]:
        assert col in ds.df.columns
    assert (ds.df["measurement_type"] == "Kd").all()


def test_invalid_smiles_dropped(bindingdb_sample_path):
    ds = data.load_bindingdb(bindingdb_sample_path, measurement_type="Kd")
    # The deliberately-invalid fixture row must not survive.
    assert ds.df["smiles_std"].map(lambda s: isinstance(s, str) and len(s) > 0).all()


def test_deduplication_collapses_replicates(bindingdb_sample_path):
    ds = data.load_bindingdb(bindingdb_sample_path, measurement_type="Kd")
    # One row per (compound, target).
    assert not ds.df.duplicated(subset=["smiles_std", "target_id"]).any()
    assert (ds.df["n_measurements"] >= 1).all()


def test_censored_dropped_by_default(bindingdb_sample_path):
    kept = data.load_bindingdb(bindingdb_sample_path, "Kd", drop_censored=True)
    with_censored = data.load_bindingdb(bindingdb_sample_path, "Kd", drop_censored=False)
    assert with_censored.n_after_valid_smiles >= kept.n_after_valid_smiles


def test_measurement_types_not_mixed(bindingdb_sample_path):
    # Loading Kd must not pull in Ki rows -- endpoints are kept separate (spec section 10).
    kd = data.load_bindingdb(bindingdb_sample_path, "Kd")
    ki = data.load_bindingdb(bindingdb_sample_path, "Ki")
    assert (kd.df["measurement_type"] == "Kd").all()
    assert (ki.df["measurement_type"] == "Ki").all()


def test_compound_library_hit_labels(bindingdb_sample_path):
    ds = data.load_bindingdb(bindingdb_sample_path, "Kd")
    lib = CompoundLibrary(ds.df, endpoint="pKd", hit_threshold=7.0)
    assert lib.n == len(ds.df)
    assert np.array_equal(lib.is_hit, ds.df["pKd"].to_numpy() >= 7.0)
    # subset preserves the endpoint/threshold.
    sub = lib.subset(np.arange(min(10, lib.n)))
    assert sub.endpoint == "pKd" and sub.hit_threshold == 7.0


def test_unknown_measurement_type_rejected(bindingdb_sample_path):
    with pytest.raises(ValueError):
        data.load_bindingdb(bindingdb_sample_path, "Kx")
