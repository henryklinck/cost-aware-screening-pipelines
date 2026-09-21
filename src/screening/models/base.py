"""Standard screening-model interface (spec section 16).

Every model -- simulated or real -- exposes the *same* two operations:

    predict(molecule_ids)      -> one continuous score per molecule (higher = more
                                  promising).  The pipeline ranks and thresholds on
                                  this score and never inspects how it was produced.
    estimate_cost(n_molecules) -> the computational cost of scoring ``n_molecules``.

Keeping this contract narrow is what lets the Phase-1 ``SimulatedModel`` be swapped
for Phase-2 adapters (``GninaModel``, ``Boltz2Model``, RDKit filters) *without touching
the pipeline or optimization code*.

Important cross-model caveat (spec sections 12, 13, 31): scores from different models
are NOT assumed to be on a common calibrated scale.  The pipeline only ever compares a
model's score against *its own* other scores (via ranking), never model A's score
against model B's.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ScreeningModel(ABC):
    """Abstract base class for anything that can score molecules.

    Subclasses must set ``self.name`` (a short unique identifier) and implement
    :meth:`predict` and :meth:`estimate_cost`.
    """

    name: str

    @abstractmethod
    def predict(self, molecule_ids: np.ndarray) -> np.ndarray:
        """Return one score per molecule, aligned with ``molecule_ids``.

        Parameters
        ----------
        molecule_ids:
            1-D array identifying the molecules to score.  In Phase 1 these are
            integer indices into a :class:`~screening.simulation.MoleculeLibrary`.
            Real adapters would instead receive whatever identifier they need
            (e.g. SMILES) and map it to features internally.

        Returns
        -------
        np.ndarray
            Float scores, same length and order as ``molecule_ids``.  Higher means
            "more likely to be a high-affinity hit".
        """

    @abstractmethod
    def estimate_cost(self, n_molecules: int) -> float:
        """Return the estimated computational cost of scoring ``n_molecules``.

        Cost is expressed in abstract *compute units* in Phase 1 (see
        :class:`~screening.models.simulated.SimulatedModel`).  Phase-2 adapters will
        return measured wall-clock / GPU-hours converted to a documented unit
        (spec section 17).  The unit only needs to be *consistent across models* for
        the cost/recall trade-off to be meaningful.
        """

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r})"
