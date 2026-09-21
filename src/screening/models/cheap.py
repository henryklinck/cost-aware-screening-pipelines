"""Runnable low-cost baseline models for the real-data layer (spec section 14).

These are the "spend almost nothing before invoking expensive models" stages:

    RDKitPropertyFilter  -- a physicochemical drug-likeness prior (QED / Lipinski).
                            NOT a binding predictor; it establishes what a near-free
                            first stage buys you.
    FingerprintSimilarity -- Morgan-fingerprint similarity to *known actives*, a genuine
                            cheap signal.  References must come from the TRAINING hits
                            only, never the test set (leakage).

Both conform to :class:`~screening.models.base.ScreeningModel`, so they drop into the
same pipeline/optimization code as the simulated models.  Cost is **measured**, not
invented (spec section 17): each model times its own scoring and reports seconds per
molecule as its per-molecule cost.
"""

from __future__ import annotations

import time
from typing import Callable, List, Optional, Sequence

import numpy as np

from .. import chem
from .base import ScreeningModel


class PrecomputedRealModel(ScreeningModel):
    """Base for real models that score every library molecule once and cache the result.

    Scoring is timed so that ``estimate_cost`` reflects *measured* wall-clock cost
    (seconds), consistent regardless of how many molecules a pipeline stage scores.
    Invalid molecules (unparseable SMILES) receive ``-inf`` so they rank last.
    """

    def __init__(self, name: str, smiles: Sequence[str],
                 score_fn: Callable[[List[str]], np.ndarray]):
        self.name = name
        smiles = list(smiles)
        t0 = time.perf_counter()
        scores = np.asarray(score_fn(smiles), dtype=float)
        elapsed = time.perf_counter() - t0

        self._scores = scores
        self._global_ranking = np.argsort(-self._scores, kind="stable")
        self._n = len(smiles)
        # Measured per-molecule cost in seconds (>=0). Guard tiny/zero timings.
        self._cost_per_molecule = max(elapsed / self._n, 0.0) if self._n else 0.0
        self.measured_seconds = elapsed

    @property
    def scores(self) -> np.ndarray:
        return self._scores

    @property
    def global_ranking(self) -> np.ndarray:
        return self._global_ranking

    def predict(self, molecule_ids: np.ndarray) -> np.ndarray:
        return self._scores[np.asarray(molecule_ids)]

    def estimate_cost(self, n_molecules: int) -> float:
        if n_molecules < 0:
            raise ValueError("n_molecules must be non-negative")
        return n_molecules * self._cost_per_molecule


class RDKitPropertyFilter(PrecomputedRealModel):
    """Cheap physicochemical drug-likeness prior (spec section 14).

    Ranks by RDKit QED (Quantitative Estimate of Drug-likeness, 0-1, continuous) when
    available, falling back to ``-Lipinski_violations``.  This is deliberately *not* a
    binding model -- its enrichment for true binders should be modest -- so the pipeline
    experiment can show what a near-free filtering stage does and does not buy.
    """

    def __init__(self, smiles: Sequence[str], name: str = "rdkit_filter",
                 use_qed: bool = True):
        chem.require_rdkit()
        from rdkit.Chem import QED  # local import; rdkit is optional

        def score_fn(smis: List[str]) -> np.ndarray:
            out = np.full(len(smis), -np.inf)
            for i, s in enumerate(smis):
                mol = chem.mol_from_smiles(s)
                if mol is None:
                    continue
                if use_qed:
                    try:
                        out[i] = float(QED.qed(mol))
                        continue
                    except Exception:
                        pass
                d = chem.descriptors(s)
                out[i] = -float(d["lipinski_violations"]) if d else -np.inf
            return out

        super().__init__(name, smiles, score_fn)


class FingerprintSimilarity(PrecomputedRealModel):
    """Max Morgan-fingerprint Tanimoto similarity to a set of known actives (spec 14).

    ``reference_actives`` are SMILES of known binders -- in a real experiment these must
    be drawn from the **training** hits only, so no test information leaks in.  A molecule
    similar to a known active is a cheap but genuine hit signal.
    """

    def __init__(self, smiles: Sequence[str], reference_actives: Sequence[str],
                 name: str = "fingerprint", radius: int = 2, n_bits: int = 2048):
        chem.require_rdkit()
        ref_fps = [fp for fp in
                   (chem.morgan_fingerprint(s, radius, n_bits) for s in reference_actives)
                   if fp is not None]

        def score_fn(smis: List[str]) -> np.ndarray:
            out = np.full(len(smis), -np.inf)
            if not ref_fps:
                # No valid references -> uninformative (all equal); avoid -inf everywhere.
                return np.zeros(len(smis))
            for i, s in enumerate(smis):
                fp = chem.morgan_fingerprint(s, radius, n_bits)
                if fp is None:
                    continue
                out[i] = chem.max_similarity_to_references(fp, ref_fps)
            return out

        self.n_references = len(ref_fps)
        super().__init__(name, smiles, score_fn)
