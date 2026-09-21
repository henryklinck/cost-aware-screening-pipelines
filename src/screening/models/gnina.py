"""GNINA adapter -- **Phase 2 stub** (spec section 12).

GNINA (https://github.com/gnina/gnina) is an open-source molecular docking framework
with CNN-based scoring.  The Phase-2 adapter will conform to
:class:`~screening.models.base.ScreeningModel` by shelling out to an installed GNINA
binary, recording (per spec section 12): docking time, molecule count, CNN score,
docking score, hardware, configuration, exhaustiveness and runtime -- and documenting
the exact version/commit used.

It is a stub here on purpose: Phase 1 must not invent model performance or costs
(spec section 31).  Because it implements the same interface as ``SimulatedModel``, it
will drop into the existing pipeline/optimization code unchanged (spec section 32).
"""

from __future__ import annotations

import numpy as np

from .base import ScreeningModel

_MSG = "GninaModel is a Phase-2 adapter and is not implemented in Phase 1."


class GninaModel(ScreeningModel):
    name = "gnina"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_MSG)

    def predict(self, molecule_ids: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError(_MSG)

    def estimate_cost(self, n_molecules: int) -> float:  # pragma: no cover
        raise NotImplementedError(_MSG)
