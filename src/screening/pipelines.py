"""Multi-stage screening pipeline execution (spec section 6).

A *pipeline* is an ordered list of model names with a *retention fraction* per stage:

    pipeline   = ["A", "C", "E"]
    thresholds = [0.20, 0.10, 0.10]

    1_000_000 --A--> keep top 20% --> 200_000 --C--> keep top 10% --> 20_000
              --E--> keep top 10% --> 2_000

Each stage scores only the survivors of the previous stage, keeps the top fraction by
that stage's own score, and passes them on.  Cost accrues per stage on the *number of
molecules entering that stage* (the expensive model only pays for what it actually
scores -- the whole point of cascading).

Invariants enforced here (spec section 30):
  * survivors of a stage are a strict subset of its inputs (no molecule is ever
    re-introduced, and none is scored twice within a stage);
  * at least one molecule is retained per stage (a fraction that rounds to zero keeps 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from .models.base import ScreeningModel
from .simulation import MoleculeLibrary


def select_top_fraction(
    model: ScreeningModel,
    candidate_indices: np.ndarray,
    fraction: float,
    is_full_library: bool,
    library_size: int,
) -> np.ndarray:
    """Return the indices of the top ``fraction`` of ``candidate_indices`` by score.

    When the candidates are the entire library and ``model`` exposes a cached global
    ranking (as :class:`~screening.models.simulated.SimulatedModel` does), we reuse it
    instead of re-sorting -- the single biggest speed win for exhaustive search.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("retention fraction must be in (0, 1]")

    n_before = len(candidate_indices)
    n_keep = max(1, int(n_before * fraction))

    global_ranking = getattr(model, "global_ranking", None)
    if is_full_library and global_ranking is not None and len(global_ranking) == library_size:
        return global_ranking[:n_keep]

    scores = model.predict(candidate_indices)
    # Descending order; stable keeps ties reproducible.
    order = np.argsort(-scores, kind="stable")[:n_keep]
    return candidate_indices[order]


@dataclass
class StageResult:
    model: str
    fraction: float
    molecules_before: int
    molecules_after: int
    stage_cost: float
    hits_before: int
    hits_after: int


@dataclass
class PipelineResult:
    """Outcome of running one pipeline configuration.

    ``survivors`` holds the final candidate molecule indices; downstream metrics
    (:mod:`screening.metrics`) turn this plus the library into recall/precision/etc.
    """

    pipeline: List[str]
    thresholds: List[float]
    survivors: np.ndarray
    total_cost: float
    stages: List[StageResult] = field(default_factory=list)
    initial_molecules: int = 0

    @property
    def final_molecules(self) -> int:
        return int(len(self.survivors))

    def pipeline_label(self) -> str:
        """Human-readable label, e.g. ``A(0.20)->C(0.10)->E(0.10)``."""
        parts = [f"{m}({t:.2f})" for m, t in zip(self.pipeline, self.thresholds)]
        return "->".join(parts)


def run_pipeline(
    library: MoleculeLibrary,
    models: Dict[str, ScreeningModel],
    pipeline: Sequence[str],
    thresholds: Sequence[float],
) -> PipelineResult:
    """Execute a screening pipeline and return its :class:`PipelineResult`.

    Parameters
    ----------
    library:
        The molecule library (provides hit labels for per-stage bookkeeping).
    models:
        Mapping of model name -> model.  Every name in ``pipeline`` must be present.
    pipeline:
        Ordered model names to apply.
    thresholds:
        Retention fraction per stage; must match ``pipeline`` in length.
    """
    pipeline = list(pipeline)
    thresholds = list(thresholds)
    if len(pipeline) != len(thresholds):
        raise ValueError("pipeline and thresholds must have equal length")
    if len(pipeline) == 0:
        raise ValueError("pipeline must have at least one stage")
    missing = [name for name in pipeline if name not in models]
    if missing:
        raise ValueError(f"unknown model(s) in pipeline: {missing}")

    is_hit = library.is_hit
    current = library.indices  # start with the whole library
    total_cost = 0.0
    stages: List[StageResult] = []

    for stage_idx, (name, fraction) in enumerate(zip(pipeline, thresholds)):
        model = models[name]
        n_before = len(current)
        hits_before = int(is_hit[current].sum())

        total_cost += model.estimate_cost(n_before)

        current = select_top_fraction(
            model,
            current,
            fraction,
            is_full_library=(stage_idx == 0),
            library_size=library.n,
        )

        stages.append(
            StageResult(
                model=name,
                fraction=fraction,
                molecules_before=n_before,
                molecules_after=len(current),
                stage_cost=model.estimate_cost(n_before),
                hits_before=hits_before,
                hits_after=int(is_hit[current].sum()),
            )
        )

    return PipelineResult(
        pipeline=pipeline,
        thresholds=thresholds,
        survivors=current,
        total_cost=total_cost,
        stages=stages,
        initial_molecules=library.n,
    )
