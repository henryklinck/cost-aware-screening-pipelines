"""Simulated screening models with correlated errors (spec section 5).

A simulated model predicts a continuous score for each molecule:

    y_hat_{i,m} = gain_m * y_true_i + bias_m + eps_{i,m}

where the *model error* ``eps_{i,m} = noise_std_m * Z_{i,m}`` and the standardized
errors ``Z_{i,·}`` across models are drawn jointly from a multivariate normal with a
specified correlation matrix ``R``.  So:

    Corr(eps_A, eps_B) = R[A, B]      (independent of the per-model noise scale)

Controllable knobs per model (all central to the research question):

    noise_std   -> accuracy   (smaller = more accurate; usually costs more)
    bias        -> systematic offset
    gain        -> how strongly the score tracks true affinity (signal strength)
    cost_*      -> computational cost

Why precompute?  Each model's scores for the *entire* library are generated once at
construction and cached.  This guarantees (a) determinism -- a molecule gets the same
score every time it is scored, at any pipeline stage -- and (b) a coherent joint error
structure across models.  ``predict`` is then a pure lookup, which also makes exhaustive
pipeline/threshold search fast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from ..simulation import MoleculeLibrary
from .base import ScreeningModel


@dataclass
class ModelSpec:
    """Tunable characteristics of a simulated model."""

    name: str
    noise_std: float          # model error scale (accuracy: smaller = better)
    cost_per_molecule: float  # compute units per molecule scored
    bias: float = 0.0         # systematic additive offset
    gain: float = 1.0         # multiplier on true affinity (signal strength)
    fixed_overhead: float = 0.0  # one-off compute units regardless of count


class SimulatedModel(ScreeningModel):
    """A model whose scores are a noisy function of hidden true affinity.

    Instances are normally created by :func:`build_simulated_models` so that their
    errors share a joint correlation structure.  The constructor takes the *already
    generated* score vector to keep the correlated-error sampling in one place.
    """

    def __init__(self, spec: ModelSpec, scores: np.ndarray):
        self.name = spec.name
        self.spec = spec
        self._scores = np.asarray(scores, dtype=float)
        # Cache the global descending ranking once.  Selecting the top-k of the full
        # library is then O(k); selecting the top-k of a subset stays cheap because
        # subsets shrink rapidly down the pipeline.
        self._global_ranking = np.argsort(-self._scores, kind="stable")

    @property
    def scores(self) -> np.ndarray:
        """The (immutable) precomputed score for every molecule in the library."""
        return self._scores

    @property
    def global_ranking(self) -> np.ndarray:
        """Molecule indices sorted by descending score over the whole library."""
        return self._global_ranking

    def predict(self, molecule_ids: np.ndarray) -> np.ndarray:
        molecule_ids = np.asarray(molecule_ids)
        return self._scores[molecule_ids]

    def estimate_cost(self, n_molecules: int) -> float:
        if n_molecules < 0:
            raise ValueError("n_molecules must be non-negative")
        if n_molecules == 0:
            return 0.0
        return self.spec.fixed_overhead + n_molecules * self.spec.cost_per_molecule


def _correlation_matrix(
    correlation: Union[float, np.ndarray, None],
    n_models: int,
) -> np.ndarray:
    """Build an ``n_models x n_models`` correlation matrix.

    ``correlation`` may be:
      * ``None`` or ``0`` -> identity (independent model errors),
      * a scalar in [0, 1) -> equicorrelation ``(1-rho) I + rho J``,
      * a full matrix -> used as-is (validated to be square and symmetric).
    """
    if correlation is None:
        return np.eye(n_models)
    if np.isscalar(correlation):
        rho = float(correlation)
        # An equicorrelation matrix is positive-definite only for
        # rho in (-1/(m-1), 1); outside that range the covariance is invalid.
        lower = -1.0 / (n_models - 1) if n_models > 1 else -1.0
        if not (lower < rho < 1.0):
            raise ValueError(
                f"equicorrelation rho={rho} is not positive-definite for "
                f"{n_models} models (need {lower:.3f} < rho < 1)"
            )
        return (1.0 - rho) * np.eye(n_models) + rho * np.ones((n_models, n_models))
    R = np.asarray(correlation, dtype=float)
    if R.shape != (n_models, n_models):
        raise ValueError("correlation matrix shape does not match number of models")
    if not np.allclose(R, R.T):
        raise ValueError("correlation matrix must be symmetric")
    return R


def build_simulated_models(
    library: MoleculeLibrary,
    specs: Sequence[ModelSpec],
    correlation: Union[float, np.ndarray, None] = 0.0,
    seed: Optional[int] = None,
) -> Dict[str, SimulatedModel]:
    """Create simulated models whose errors share a joint correlation structure.

    Parameters
    ----------
    library:
        The world providing ``y_true``.
    specs:
        One :class:`ModelSpec` per model.  The order defines the rows/cols of a full
        correlation matrix if one is supplied.
    correlation:
        Error correlation across models (see :func:`_correlation_matrix`).  This is the
        knob for the "are the models independent?" question (spec sections 5, 22, 31).
    seed:
        Seed for the joint error draw.

    Returns
    -------
    dict
        Mapping ``name -> SimulatedModel``.
    """
    specs = list(specs)
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError("model names must be unique")

    n = library.n
    m = len(specs)
    R = _correlation_matrix(correlation, m)

    rng = np.random.default_rng(seed)
    # Joint standardized errors Z: shape (n, m), rows ~ MVN(0, R).
    L = np.linalg.cholesky(R)
    Z = rng.standard_normal(size=(n, m)) @ L.T  # each column ~ N(0,1), cross-corr = R

    y_true = library.y_true
    models: Dict[str, SimulatedModel] = {}
    for j, spec in enumerate(specs):
        eps = spec.bias + spec.noise_std * Z[:, j]
        scores = spec.gain * y_true + eps
        models[spec.name] = SimulatedModel(spec, scores)
    return models


def default_model_specs() -> List[ModelSpec]:
    """The five reference models from spec section 5 (A cheap/rough -> E costly/excellent).

    ``noise_std`` is expressed in the same units as true affinity (``sigma=1`` by
    default), so e.g. ``noise_std=1.2`` means the model error is larger than the natural
    spread of affinities (a weak model), while ``0.25`` is a strong model.  Costs span
    more than two orders of magnitude, matching a cheap heuristic vs. an expensive
    physics/ML model.
    """
    return [
        ModelSpec(name="A", noise_std=1.20, cost_per_molecule=1.0),     # cheap, rough
        ModelSpec(name="B", noise_std=0.85, cost_per_molecule=3.0),     # cheap-medium
        ModelSpec(name="C", noise_std=0.55, cost_per_molecule=10.0),    # medium, good
        ModelSpec(name="D", noise_std=0.35, cost_per_molecule=50.0),    # expensive
        ModelSpec(name="E", noise_std=0.22, cost_per_molecule=250.0),   # very expensive
    ]
