"""Phase 1 synthetic world (spec sections 4-5).

This module builds the *ground truth* the rest of Phase 1 is evaluated against.

Design decisions that mirror the spec:

* Molecules are opaque **IDs** (``molecule_000001`` ...), not chemical structures.
  The latent biology is a single continuous number per molecule.
* We keep three distinct quantities separate throughout the project (spec section 4.2):

      y_true   the true (latent) affinity              -- never observed in reality
      y_obs    the experimentally measured affinity    -- y_true + assay noise
      y_hat    a model's predicted score               -- see models.simulated

  A "hit" is defined by thresholding an affinity (spec section 10).  Because the
  underlying truth is continuous, we can define hits on ``y_true`` (the estimand we
  ultimately care about) while models are trained/scored against noisy observations.

Nothing here is binary at its core: prevalence is an *emergent* consequence of where
the hit threshold sits on the continuous affinity distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def format_molecule_id(index: int, width: int = 6) -> str:
    """Return the canonical string ID for a molecule, e.g. ``molecule_000001``."""
    return f"molecule_{index:0{width}d}"


@dataclass
class MoleculeLibrary:
    """A simulated compound library with hidden ground-truth affinities.

    Attributes
    ----------
    y_true:
        Latent true affinity per molecule (higher = stronger binder).  This is the
        quantity a perfect model would recover; it is *never* directly observable.
    y_obs:
        Experimentally measured affinity = ``y_true`` + assay noise.  This is the best
        real-world proxy for the truth and is what a real assay would report.
    hit_threshold:
        Affinity value at or above which a molecule counts as a high-affinity hit.
    hits_on:
        Which affinity the hit label is derived from: ``"true"`` or ``"obs"``.
        Defaults to ``"true"`` -- we score pipelines on recovery of the underlying
        biological truth, not of noisy measurements.
    seed:
        The seed used to generate this library (recorded for provenance).
    """

    y_true: np.ndarray
    y_obs: np.ndarray
    hit_threshold: float
    hits_on: str = "true"
    seed: Optional[int] = None
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        """Number of molecules in the library."""
        return int(self.y_true.shape[0])

    @property
    def indices(self) -> np.ndarray:
        """Integer index of every molecule (the Phase-1 molecule identifier)."""
        return np.arange(self.n)

    @property
    def hit_affinity(self) -> np.ndarray:
        """The affinity array the hit label is defined on."""
        return self.y_true if self.hits_on == "true" else self.y_obs

    @property
    def is_hit(self) -> np.ndarray:
        """Boolean ground-truth hit label per molecule."""
        return self.hit_affinity >= self.hit_threshold

    @property
    def n_hits(self) -> int:
        """Total number of true hits in the library."""
        return int(self.is_hit.sum())

    @property
    def prevalence(self) -> float:
        """Emergent fraction of the library that are hits."""
        return self.n_hits / self.n if self.n else 0.0

    def molecule_ids(self) -> np.ndarray:
        """String IDs (``molecule_000001`` ...) for every molecule."""
        return np.array([format_molecule_id(i) for i in range(self.n)])


def generate_true_affinity(
    n_molecules: int,
    mu: float = 0.0,
    sigma: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Draw the latent true affinity for every molecule: ``y_i ~ N(mu, sigma)``.

    A Normal latent variable (spec section 4.1) gives a smooth affinity landscape where
    strong binders are rare in the right tail -- prevalence then falls naturally out of
    the hit threshold rather than being imposed as a binary coin flip.
    """
    if n_molecules <= 0:
        raise ValueError("n_molecules must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    rng = np.random.default_rng(seed)
    return rng.normal(loc=mu, scale=sigma, size=n_molecules)


def add_assay_noise(
    y_true: np.ndarray,
    sigma_assay: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Return observed affinity ``y_obs = y_true + eps``, ``eps ~ N(0, sigma_assay)``.

    This is the experimental measurement error (spec section 4.2).  Keeping it separate
    from *model* error is deliberate: a real assay is itself noisy, and a model can be
    "wrong" either because it mispredicts or because the label it is compared to is a
    noisy measurement.  The distinction is preserved throughout the project.
    """
    if sigma_assay < 0:
        raise ValueError("sigma_assay must be non-negative")
    rng = np.random.default_rng(seed)
    return y_true + rng.normal(loc=0.0, scale=sigma_assay, size=y_true.shape[0])


def threshold_for_prevalence(y: np.ndarray, prevalence: float) -> float:
    """Return the affinity cutoff whose upper tail contains ``prevalence`` of ``y``.

    Lets an experiment *target* a prevalence (e.g. 1% hits) while keeping the truth
    continuous: we simply place the hit threshold at the appropriate quantile.
    """
    if not 0.0 < prevalence < 1.0:
        raise ValueError("prevalence must be in (0, 1)")
    return float(np.quantile(y, 1.0 - prevalence))


def make_library(
    n_molecules: int,
    mu: float = 0.0,
    sigma: float = 1.0,
    sigma_assay: float = 0.3,
    prevalence: Optional[float] = None,
    hit_threshold: Optional[float] = None,
    hits_on: str = "true",
    seed: Optional[int] = None,
) -> MoleculeLibrary:
    """Build a complete :class:`MoleculeLibrary`.

    Exactly one of ``prevalence`` or ``hit_threshold`` should be given to define hits.
    If ``prevalence`` is supplied, the hit threshold is placed at the matching quantile
    of the affinity the label is defined on (``hits_on``).

    Parameters
    ----------
    sigma_assay:
        Standard deviation of experimental measurement noise added to ``y_true``.
    hits_on:
        ``"true"`` (default) defines hits on the latent affinity; ``"obs"`` defines them
        on the measured affinity.
    """
    if hits_on not in ("true", "obs"):
        raise ValueError("hits_on must be 'true' or 'obs'")
    if (prevalence is None) == (hit_threshold is None):
        raise ValueError("provide exactly one of prevalence or hit_threshold")

    # Derive independent sub-seeds so affinity and assay noise are reproducible but
    # not accidentally correlated.
    ss = np.random.SeedSequence(seed)
    affinity_seed, assay_seed = ss.spawn(2)

    y_true = generate_true_affinity(
        n_molecules, mu=mu, sigma=sigma, seed=affinity_seed
    )
    y_obs = add_assay_noise(y_true, sigma_assay=sigma_assay, seed=assay_seed)

    label_affinity = y_true if hits_on == "true" else y_obs
    if hit_threshold is None:
        hit_threshold = threshold_for_prevalence(label_affinity, prevalence)

    return MoleculeLibrary(
        y_true=y_true,
        y_obs=y_obs,
        hit_threshold=float(hit_threshold),
        hits_on=hits_on,
        seed=seed,
        meta={
            "mu": mu,
            "sigma": sigma,
            "sigma_assay": sigma_assay,
            "requested_prevalence": prevalence,
            "n_molecules": n_molecules,
        },
    )
