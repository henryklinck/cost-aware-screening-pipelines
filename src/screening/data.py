"""Real experimental data loading -- **Phase 2, not implemented in Phase 1**.

This module is a deliberate placeholder.  The spec (sections 2, 9-11, 32) is emphatic
that Phase 1 must be completed and validated on the *synthetic* world before any real
data is touched, and that real datasets must never be assumed leakage-free or fabricated.

Phase 2 will implement, against the downloadable BindingDB TSV (spec section 9):
  * SMILES standardization and validity filtering,
  * compound deduplication and unit standardization,
  * conversion to a single consistent endpoint (e.g. pKd, spec section 10) -- WITHOUT
    blindly merging Kd/Ki/IC50,
  * preservation of assay info, publication dates, target IDs, and measurement type,
  * leakage-aware splits: compound-overlap, Bemis-Murcko scaffold, temporal, and
    target-holdout (spec section 11).

Until then, the pipeline/optimization/metrics code is exercised entirely by the
synthetic :class:`~screening.simulation.MoleculeLibrary`.
"""

from __future__ import annotations

_PHASE2_MESSAGE = (
    "Real-data loading is Phase 2 and is intentionally not implemented yet. "
    "Complete and validate the synthetic simulation (Phase 1) first; see the module "
    "docstring and README for the planned BindingDB pipeline."
)


def load_bindingdb(*args, **kwargs):  # noqa: D401 - stub
    """Not implemented in Phase 1."""
    raise NotImplementedError(_PHASE2_MESSAGE)


def make_splits(*args, **kwargs):  # noqa: D401 - stub
    """Not implemented in Phase 1 (compound/scaffold/temporal/target splits)."""
    raise NotImplementedError(_PHASE2_MESSAGE)
