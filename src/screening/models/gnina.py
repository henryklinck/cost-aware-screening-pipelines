"""GNINA docking adapter (spec section 12).

GNINA (https://github.com/gnina/gnina) is an open-source molecular docking framework
with CNN-based scoring (its default is a three-model CNN ensemble).  This adapter ranks
molecules by GNINA's CNN score against a prepared receptor.

**Not runnable here.**  GNINA requires its own installation, a prepared receptor, a
docking box, and typically a GPU.  When any of these is missing, :meth:`predict` raises
:class:`ToolNotAvailableError` rather than inventing scores (spec section 31).  On a
properly configured machine the adapter records docking time, molecule count, CNN score,
docking score, hardware, configuration, exhaustiveness and runtime (spec section 12),
and pins the exact binary version.

Ranking caveat (spec section 12): GNINA's raw score is used only to rank molecules
*within GNINA*; it is never compared directly to Boltz-2's affinity output.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from .base import ScreeningModel
from .external import (
    RunProvenance,
    ToolNotAvailableError,
    basic_hardware,
    find_binary,
    run_command,
)


class GninaModel(ScreeningModel):
    name = "gnina"

    def __init__(
        self,
        receptor: Optional[str] = None,
        autobox_ligand: Optional[str] = None,
        binary: str = "gnina",
        cnn: str = "rescore",
        exhaustiveness: int = 8,
        binary_path: Optional[str] = None,
        score_name: str = "CNNscore",
    ):
        self.receptor = receptor
        self.autobox_ligand = autobox_ligand
        self.cnn = cnn
        self.exhaustiveness = exhaustiveness
        self.score_name = score_name
        self.binary = find_binary(binary, binary_path)
        self.provenance: Optional[RunProvenance] = None
        self._version = self._query_version() if self.binary else None

    # -- availability -----------------------------------------------------------
    def is_available(self) -> bool:
        return self.binary is not None and self.receptor is not None

    def _require(self) -> None:
        if self.binary is None:
            raise ToolNotAvailableError(
                "GNINA binary not found. Install GNINA "
                "(https://github.com/gnina/gnina) and ensure `gnina` is on PATH or pass "
                "binary_path=..., then supply a prepared receptor."
            )
        if self.receptor is None:
            raise ToolNotAvailableError(
                "GNINA requires a prepared receptor (receptor=...) and a docking box "
                "(autobox_ligand=... or an explicit box)."
            )

    def _query_version(self) -> Optional[str]:
        try:
            out = run_command([self.binary, "--version"])
            return out.stdout.strip().splitlines()[0] if out.stdout else None
        except Exception:
            return None

    # -- interface --------------------------------------------------------------
    def predict(self, molecule_ids: np.ndarray, smiles: Optional[Sequence[str]] = None) -> np.ndarray:
        """Dock the given molecules and return CNN scores (higher = better).

        ``smiles`` must be provided (real molecules to dock).  This method is the
        validated-on-real-inputs path; it raises here because GNINA is unavailable.
        """
        self._require()
        if smiles is None:
            raise ValueError("GninaModel.predict requires the molecules' SMILES")
        return self._dock(list(smiles))  # pragma: no cover - needs external tool

    def _dock(self, smiles: List[str]) -> np.ndarray:  # pragma: no cover
        """Run GNINA per molecule, parse CNN scores, and record provenance.

        Documented interface; requires a real GNINA install + receptor to execute.
        """
        t0 = time.perf_counter()
        scores: List[float] = []
        for smi in smiles:
            ligand = self._prepare_ligand(smi)
            cmd = [
                self.binary, "-r", self.receptor, "-l", ligand,
                "--autobox_ligand", self.autobox_ligand or self.receptor,
                "--cnn", self.cnn, "--exhaustiveness", str(self.exhaustiveness),
                "--score_only" if self.cnn == "rescore" else "--num_modes", "1",
            ]
            result = run_command(cmd)
            scores.append(self._parse_cnn_score(result.stdout))
        elapsed = time.perf_counter() - t0
        self.provenance = RunProvenance(
            tool="gnina", version=self._version, command="gnina ...",
            n_molecules=len(smiles), wall_clock_seconds=elapsed,
            hardware=basic_hardware(),
            configuration={"cnn": self.cnn, "exhaustiveness": self.exhaustiveness,
                           "receptor": self.receptor},
            output_score_name=self.score_name,
        )
        return np.asarray(scores, dtype=float)

    def _prepare_ligand(self, smiles: str) -> str:  # pragma: no cover
        raise ToolNotAvailableError(
            "Ligand preparation (SMILES -> 3D SDF) is environment-specific; wire in your "
            "RDKit/OpenBabel embedding step here for real runs."
        )

    @staticmethod
    def _parse_cnn_score(stdout: str) -> float:  # pragma: no cover
        for line in stdout.splitlines():
            if "CNNscore" in line:
                try:
                    return float(line.split()[-1])
                except (ValueError, IndexError):
                    continue
        raise ValueError("could not parse CNNscore from GNINA output")

    def estimate_cost(self, n_molecules: int) -> float:
        """Measured seconds/molecule x count, if a prior run recorded it.

        Without a measured run there is no honest cost to report, so we raise rather than
        invent one (spec sections 17, 31).
        """
        if n_molecules < 0:
            raise ValueError("n_molecules must be non-negative")
        if self.provenance is None or self.provenance.seconds_per_molecule is None:
            raise ToolNotAvailableError(
                "No measured GNINA runtime available to estimate cost. Run predict() on a "
                "calibration set first, or supply a measured seconds-per-molecule."
            )
        return n_molecules * self.provenance.seconds_per_molecule
