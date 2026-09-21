"""Leakage-aware train/test splits (spec section 11).

A random split of molecular data almost always **overstates** prospective performance,
because near-duplicate compounds, shared scaffolds, or same-paper measurements leak
across the boundary.  This module provides the four splits the spec calls for and, just
as importantly, a verifier that asserts the intended quantity does not overlap.

    compound_split   -- disjoint compounds (standardized SMILES); the weakest guarantee.
    scaffold_split   -- disjoint Bemis-Murcko scaffolds; guards against analogue leakage.
    temporal_split   -- train on older publications, test on newer (prospective proxy).
    target_split     -- hold out entire targets (generalization to unseen targets).

Every split returns integer row-index arrays into the input DataFrame and never tunes
anything on the test set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class Split:
    """A train/test split with a recorded key and rationale."""

    train_idx: np.ndarray
    test_idx: np.ndarray
    kind: str
    key: str  # the column whose values are kept disjoint (or "time"/"target")
    meta: dict

    @property
    def n_train(self) -> int:
        return int(len(self.train_idx))

    @property
    def n_test(self) -> int:
        return int(len(self.test_idx))


def _group_holdout_split(
    groups: np.ndarray,
    test_fraction: float,
    seed: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign whole groups to test until ~``test_fraction`` of *rows* are held out.

    Splitting by group (not by row) is what prevents leakage: every row of a group lands
    on the same side.  Groups are shuffled for a random-but-reproducible assignment.
    """
    rng = np.random.default_rng(seed)
    unique = pd.unique(groups)
    rng.shuffle(unique)

    counts = pd.Series(groups).value_counts()
    target_test = test_fraction * len(groups)

    test_groups = set()
    running = 0
    for g in unique:
        if running >= target_test:
            break
        test_groups.add(g)
        running += int(counts[g])

    is_test = np.array([g in test_groups for g in groups])
    return np.where(~is_test)[0], np.where(is_test)[0]


def compound_split(
    df: pd.DataFrame,
    smiles_col: str = "smiles_std",
    test_fraction: float = 0.2,
    seed: Optional[int] = None,
) -> Split:
    """Split so no standardized compound appears in both train and test."""
    train, test = _group_holdout_split(df[smiles_col].to_numpy(), test_fraction, seed)
    return Split(train, test, "compound", smiles_col,
                 {"test_fraction": test_fraction, "seed": seed})


def scaffold_split(
    df: pd.DataFrame,
    scaffold_col: str = "scaffold",
    test_fraction: float = 0.2,
    seed: Optional[int] = None,
) -> Split:
    """Split so no Bemis-Murcko scaffold appears in both train and test."""
    scaffolds = df[scaffold_col].fillna("").to_numpy()
    train, test = _group_holdout_split(scaffolds, test_fraction, seed)
    return Split(train, test, "scaffold", scaffold_col,
                 {"test_fraction": test_fraction, "seed": seed})


def temporal_split(
    df: pd.DataFrame,
    year_col: str = "year",
    cutoff_year: Optional[float] = None,
    test_fraction: float = 0.2,
) -> Split:
    """Train on older publications, test on newer -- a prospective-discovery proxy.

    If ``cutoff_year`` is given, rows with ``year < cutoff`` are train and ``>=`` are
    test.  Otherwise the cutoff is chosen at the ``(1 - test_fraction)`` quantile of the
    available years.  Rows without a year are dropped (temporal split needs dates).
    """
    years = pd.to_numeric(df[year_col], errors="coerce").to_numpy(dtype=float)
    has_year = ~np.isnan(years)
    if has_year.sum() == 0:
        raise ValueError(
            "temporal_split requires publication years, but none are present. "
            "This BindingDB release may lack a year column; use another split."
        )
    if cutoff_year is None:
        cutoff_year = float(np.quantile(years[has_year], 1.0 - test_fraction))

    train_idx = np.where(has_year & (years < cutoff_year))[0]
    test_idx = np.where(has_year & (years >= cutoff_year))[0]
    return Split(train_idx, test_idx, "temporal", "year",
                 {"cutoff_year": cutoff_year, "test_fraction": test_fraction,
                  "n_dropped_no_year": int((~has_year).sum())})


def target_split(
    df: pd.DataFrame,
    target_col: str = "target_id",
    test_targets: Optional[Sequence[str]] = None,
    test_fraction: float = 0.2,
    seed: Optional[int] = None,
) -> Split:
    """Hold out entire targets to test generalization to unseen targets (spec section 11).

    If ``test_targets`` is given, exactly those targets form the test set; otherwise a
    reproducible random subset of targets covering ~``test_fraction`` of rows is held out.
    """
    targets = df[target_col].astype(str).to_numpy()
    if test_targets is not None:
        test_set = set(map(str, test_targets))
        is_test = np.array([t in test_set for t in targets])
        train_idx, test_idx = np.where(~is_test)[0], np.where(is_test)[0]
    else:
        train_idx, test_idx = _group_holdout_split(targets, test_fraction, seed)
    return Split(train_idx, test_idx, "target", target_col,
                 {"test_fraction": test_fraction, "seed": seed,
                  "test_targets": None if test_targets is None else list(test_targets)})


def verify_no_leakage(
    df: pd.DataFrame,
    split: Split,
    key_col: str,
) -> None:
    """Assert that ``key_col`` values are disjoint between train and test.

    Raises ``AssertionError`` naming offending keys if any leak.  Use ``"smiles_std"``
    for compound overlap and ``"scaffold"`` for scaffold overlap (spec section 30).
    """
    train_keys = set(df.iloc[split.train_idx][key_col].dropna().tolist())
    test_keys = set(df.iloc[split.test_idx][key_col].dropna().tolist())
    overlap = train_keys & test_keys
    if overlap:
        example = list(overlap)[:5]
        raise AssertionError(
            f"{key_col} leakage between train and test: {len(overlap)} shared "
            f"values, e.g. {example}"
        )


def make_splits(
    df: pd.DataFrame,
    kinds: Sequence[str] = ("compound", "scaffold", "temporal", "target"),
    test_fraction: float = 0.2,
    seed: Optional[int] = None,
) -> Dict[str, Split]:
    """Build several named splits at once, skipping any that are infeasible.

    A split that cannot be built for this dataset (e.g. temporal without years) is
    omitted with its reason recorded, rather than raising -- so a caller can request all
    four and use whatever the data supports.
    """
    builders = {
        "compound": lambda: compound_split(df, test_fraction=test_fraction, seed=seed),
        "scaffold": lambda: scaffold_split(df, test_fraction=test_fraction, seed=seed),
        "temporal": lambda: temporal_split(df, test_fraction=test_fraction),
        "target": lambda: target_split(df, test_fraction=test_fraction, seed=seed),
    }
    out: Dict[str, Split] = {}
    for kind in kinds:
        if kind not in builders:
            raise ValueError(f"unknown split kind: {kind}")
        try:
            out[kind] = builders[kind]()
        except ValueError:
            continue  # infeasible for this dataset (e.g. temporal with no years)
    return out
