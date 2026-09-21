"""Screening models: a common interface plus simulated (Phase 1) and real (Phase 2) impls.

The real-model classes (``RDKitPropertyFilter``, ``FingerprintSimilarity``,
``GninaModel``, ``Boltz2Model``) are imported lazily via :func:`__getattr__` so that
Phase-1 usage never requires RDKit or external tools to be installed.
"""

from .base import ScreeningModel
from .simulated import (
    ModelSpec,
    SimulatedModel,
    build_simulated_models,
    default_model_specs,
)

__all__ = [
    "ScreeningModel",
    "SimulatedModel",
    "ModelSpec",
    "build_simulated_models",
    "default_model_specs",
    # Phase 2 (lazy):
    "RDKitPropertyFilter",
    "FingerprintSimilarity",
    "GninaModel",
    "Boltz2Model",
]

_LAZY = {
    "RDKitPropertyFilter": ("cheap", "RDKitPropertyFilter"),
    "FingerprintSimilarity": ("cheap", "FingerprintSimilarity"),
    "GninaModel": ("gnina", "GninaModel"),
    "Boltz2Model": ("boltz2", "Boltz2Model"),
}


def __getattr__(name):  # PEP 562 lazy attribute access
    if name in _LAZY:
        import importlib

        module, attr = _LAZY[name]
        mod = importlib.import_module(f".{module}", __name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
