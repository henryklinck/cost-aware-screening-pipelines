"""Cost-aware molecular screening pipelines.

Phase 1 (implemented): a synthetic simulation for developing the methodology --
continuous latent affinity, assay noise, simulated models with controllable
accuracy/cost/error-correlation, multi-stage pipeline execution, exhaustive
pipeline/threshold search, cost/recall metrics and Pareto analysis.

Phase 2/3 (see docstrings in ``data``, ``models.gnina``, ``models.boltz2``): real
models and experimental data; not implemented yet, by design.
"""

from . import metrics, optimization, pipelines, simulation, utils
from .models import (
    ModelSpec,
    ScreeningModel,
    SimulatedModel,
    build_simulated_models,
    default_model_specs,
)
from .simulation import MoleculeLibrary, make_library

__version__ = "0.1.0"

__all__ = [
    "simulation",
    "pipelines",
    "metrics",
    "optimization",
    "utils",
    "MoleculeLibrary",
    "make_library",
    "ScreeningModel",
    "SimulatedModel",
    "ModelSpec",
    "build_simulated_models",
    "default_model_specs",
]
