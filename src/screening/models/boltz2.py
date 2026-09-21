"""Boltz-2 adapter -- **Phase 2 stub** (spec section 13).

Boltz-2 (https://github.com/jwohlwend/boltz, MIT-licensed model and weights) outputs
``affinity_probability_binary`` (suited to binder detection) and ``affinity_pred_value``
(affinity estimation).  The Phase-2 adapter will conform to
:class:`~screening.models.base.ScreeningModel`, defaulting to
``affinity_probability_binary`` as the ranking score for hit discovery and separately
supporting ``affinity_pred_value`` for affinity-ranking experiments -- never mixing the
two without documenting why (spec section 13).  It will record runtime, GPU, batch size,
MSA configuration, model version, inference settings and the output score used.

Stubbed here so Phase 1 fabricates no results (spec section 31); it will slot into the
unchanged pipeline/optimization code once implemented (spec section 32).
"""

from __future__ import annotations

import numpy as np

from .base import ScreeningModel

_MSG = "Boltz2Model is a Phase-2 adapter and is not implemented in Phase 1."


class Boltz2Model(ScreeningModel):
    name = "boltz2"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_MSG)

    def predict(self, molecule_ids: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError(_MSG)

    def estimate_cost(self, n_molecules: int) -> float:  # pragma: no cover
        raise NotImplementedError(_MSG)
