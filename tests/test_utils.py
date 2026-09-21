"""Tests for reproducibility helpers (spec sections 27, 29)."""

import json

from screening.utils import ExperimentRecord, Timer, freeze_selection


def test_experiment_record_roundtrip(tmp_path):
    rec = ExperimentRecord(experiment_id="t1", config={"a": 1}, seed=7,
                           results={"recall": 0.5})
    p = rec.to_json(tmp_path / "rec.json")
    loaded = json.loads(p.read_text())
    assert loaded["experiment_id"] == "t1"
    assert loaded["seed"] == 7
    assert loaded["results"]["recall"] == 0.5
    assert "hardware" in loaded and "created_utc" in loaded


def test_timer_measures_positive():
    with Timer() as t:
        sum(range(10000))
    assert t.seconds is not None and t.seconds >= 0


def test_freeze_selection_is_deterministic_hash(tmp_path):
    cands = ["CCO", "c1ccccc1", "CC(=O)O"]
    r1 = freeze_selection(cands, ["m"], [0.1], {"x": 1}, tmp_path / "a.json")
    r2 = freeze_selection(cands, ["m"], [0.1], {"x": 1}, tmp_path / "b.json")
    # Same candidates -> same hash; changing the list changes the hash.
    assert r1["candidates_sha256"] == r2["candidates_sha256"]
    r3 = freeze_selection(cands + ["CCN"], ["m"], [0.1], {"x": 1}, tmp_path / "c.json")
    assert r3["candidates_sha256"] != r1["candidates_sha256"]
    assert r1["n_candidates"] == 3
