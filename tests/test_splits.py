"""Tests for leakage-aware splits (spec sections 11, 30: leakage)."""

import numpy as np
import pandas as pd
import pytest

from screening import splits


def _toy():
    # 6 compounds, 3 scaffolds (each scaffold shared by 2 compounds), 2 targets, 2 years.
    return pd.DataFrame({
        "smiles_std": ["A", "A", "B", "C", "D", "E"],
        "scaffold":   ["s1", "s1", "s1", "s2", "s2", "s3"],
        "target_id":  ["T1", "T1", "T2", "T1", "T2", "T2"],
        "year":       [2010, 2011, 2012, 2020, 2021, 2022],
        "pKd":        [8.0, 5.0, 7.5, 6.0, 9.0, 4.0],
    })


def test_compound_split_no_compound_overlap():
    df = _toy()
    s = splits.compound_split(df, test_fraction=0.4, seed=0)
    splits.verify_no_leakage(df, s, "smiles_std")  # must not raise
    assert s.n_train + s.n_test == len(df)


def test_scaffold_split_no_scaffold_overlap():
    df = _toy()
    s = splits.scaffold_split(df, test_fraction=0.4, seed=1)
    splits.verify_no_leakage(df, s, "scaffold")


def test_temporal_split_orders_by_year():
    df = _toy()
    s = splits.temporal_split(df, cutoff_year=2015)
    assert set(df.iloc[s.train_idx]["year"]) <= {2010, 2011, 2012}
    assert set(df.iloc[s.test_idx]["year"]) >= {2020, 2021, 2022}


def test_temporal_split_requires_years():
    df = _toy().assign(year=np.nan)
    with pytest.raises(ValueError):
        splits.temporal_split(df)


def test_target_split_holds_out_whole_targets():
    df = _toy()
    s = splits.target_split(df, test_targets=["T2"])
    assert set(df.iloc[s.test_idx]["target_id"]) == {"T2"}
    assert "T2" not in set(df.iloc[s.train_idx]["target_id"])


def test_verify_no_leakage_detects_overlap():
    df = _toy()
    # A deliberately leaky split: same compound "A" on both sides.
    bad = splits.Split(train_idx=np.array([0]), test_idx=np.array([1]),
                       kind="bad", key="smiles_std", meta={})
    with pytest.raises(AssertionError):
        splits.verify_no_leakage(df, bad, "smiles_std")


def test_make_splits_skips_infeasible():
    df = _toy().assign(year=np.nan)  # temporal infeasible
    out = splits.make_splits(df, seed=0)
    assert "temporal" not in out
    assert {"compound", "scaffold", "target"} <= set(out)
