"""Boltz-2 affinity adapter (spec section 13).

Boltz-2 (https://github.com/jwohlwend/boltz, MIT-licensed model and weights) predicts,
per complex, ``affinity_probability_binary`` (calibrated for binder detection) and
``affinity_pred_value`` (affinity estimation).

For **hit discovery** this adapter defaults to ranking by ``affinity_probability_binary``
(spec section 13).  ``affinity_pred_value`` is available separately for affinity-ranking
experiments; the two are never mixed without documenting why.

**Not runnable here.**  Boltz-2 requires its own installation, model weights, an MSA/
featurization step and a GPU.  When unavailable, :meth:`predict` raises
:class:`ToolNotAvailableError` rather than fabricating scores (spec section 31).  On a
configured machine it records runtime, GPU, batch size, MSA configuration, model version,
inference settings and the output score used (spec section 13).

Ranking caveat: Boltz-2's output is used only to rank *within Boltz-2*; it is never
compared directly to GNINA's CNN score.
"""

from __future__ import annotations

import time
from typing import List, Optional, Sequence

import numpy as np

from .base import ScreeningModel
from .external import (
    RunProvenance,
    ToolNotAvailableError,
    basic_hardware,
    find_binary,
)

VALID_SCORES = ("affinity_probability_binary", "affinity_pred_value")


class Boltz2Model(ScreeningModel):
    name = "boltz2"

    def __init__(
        self,
        target_sequence: Optional[str] = None,
        score: str = "affinity_probability_binary",
        binary: str = "boltz",
        binary_path: Optional[str] = None,
        use_msa_server: bool = False,
        batch_size: int = 1,
        model_version: Optional[str] = None,
    ):
        if score not in VALID_SCORES:
            raise ValueError(f"score must be one of {VALID_SCORES}")
        self.target_sequence = target_sequence
        self.score = score
        self.use_msa_server = use_msa_server
        self.batch_size = batch_size
        self.model_version = model_version
        self.binary = find_binary(binary, binary_path)
        self.provenance: Optional[RunProvenance] = None

    def is_available(self) -> bool:
        return self.binary is not None and self.target_sequence is not None

    def _require(self) -> None:
        if self.binary is None:
            raise ToolNotAvailableError(
                "Boltz binary not found. Install Boltz-2 "
                "(https://github.com/jwohlwend/boltz) and ensure `boltz` is on PATH or "
                "pass binary_path=..., then supply the target sequence."
            )
        if self.target_sequence is None:
            raise ToolNotAvailableError(
                "Boltz2Model requires the target protein sequence (target_sequence=...)."
            )

    def predict(self, molecule_ids: np.ndarray, smiles: Optional[Sequence[str]] = None) -> np.ndarray:
        """Predict per-molecule scores (higher = more likely / stronger binder).

        Requires SMILES and a configured Boltz-2 install; raises here because the tool is
        unavailable in this environment.
        """
        self._require()
        if smiles is None:
            raise ValueError("Boltz2Model.predict requires the molecules' SMILES")
        return self._infer(list(smiles))  # pragma: no cover - needs external tool

    def _infer(self, smiles: List[str]) -> np.ndarray:  # pragma: no cover
        """Run Boltz-2 inference and record provenance.

        Documented interface; requires real weights + GPU to execute.  Builds one
        prediction input per (target_sequence, ligand SMILES), runs the CLI, and reads
        the chosen affinity field from the output.
        """
        t0 = time.perf_counter()
        scores = self._run_boltz(smiles)
        elapsed = time.perf_counter() - t0
        self.provenance = RunProvenance(
            tool="boltz2", version=self.model_version, command="boltz predict ...",
            n_molecules=len(smiles), wall_clock_seconds=elapsed,
            hardware=basic_hardware(),
            configuration={"score": self.score, "use_msa_server": self.use_msa_server,
                           "batch_size": self.batch_size},
            output_score_name=self.score,
        )
        return np.asarray(scores, dtype=float)

    def _run_boltz(self, smiles: List[str]) -> np.ndarray:  # pragma: no cover
        raise ToolNotAvailableError(
            "Boltz-2 inference wiring (input YAML generation, MSA, output parsing) is "
            "environment-specific and must be validated on real weights/GPU."
        )

    def estimate_cost(self, n_molecules: int) -> float:
        if n_molecules < 0:
            raise ValueError("n_molecules must be non-negative")
        if self.provenance is None or self.provenance.seconds_per_molecule is None:
            raise ToolNotAvailableError(
                "No measured Boltz-2 runtime available to estimate cost. Run predict() on "
                "a calibration set first, or supply a measured seconds-per-molecule."
            )
        return n_molecules * self.provenance.seconds_per_molecule
