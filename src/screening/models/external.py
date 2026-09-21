"""Shared scaffolding for external-tool model adapters (GNINA, Boltz-2).

These models wrap heavyweight third-party programs that require their own installation,
input structures (receptors, MSAs) and usually a GPU.  They are **not runnable in this
environment**, and this project must never fabricate their scores or costs (spec section
31).  So the adapters here:

  * conform to :class:`~screening.models.base.ScreeningModel`;
  * locate the external binary and validate inputs up front;
  * raise :class:`ToolNotAvailableError` with actionable guidance when the tool or its
    inputs are missing (the situation in CI / this repo);
  * record the full provenance the spec asks for (runtime, hardware, configuration,
    version) whenever they *do* run on a properly configured machine.

The invocation/parsing paths are written against each tool's documented interface but
are marked as requiring validation on real inputs -- they have not been exercised here.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


class ToolNotAvailableError(RuntimeError):
    """Raised when an external tool (or its required inputs) is not available."""


@dataclass
class RunProvenance:
    """Provenance recorded for an external-tool run (spec sections 12, 13, 17, 29)."""

    tool: str
    version: Optional[str] = None
    command: Optional[str] = None
    n_molecules: int = 0
    wall_clock_seconds: Optional[float] = None
    hardware: Dict[str, object] = field(default_factory=dict)
    configuration: Dict[str, object] = field(default_factory=dict)
    output_score_name: Optional[str] = None

    @property
    def seconds_per_molecule(self) -> Optional[float]:
        if self.wall_clock_seconds is None or self.n_molecules == 0:
            return None
        return self.wall_clock_seconds / self.n_molecules


def find_binary(name: str, explicit_path: Optional[str] = None) -> Optional[str]:
    """Return a usable path to ``name`` (or ``explicit_path``), else ``None``."""
    if explicit_path:
        return explicit_path if shutil.which(explicit_path) or _is_file(explicit_path) \
            else None
    return shutil.which(name)


def _is_file(path: str) -> bool:
    import os

    return os.path.isfile(path)


def basic_hardware() -> Dict[str, object]:
    import os

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        # GPU details are filled in by the concrete adapter when it can query them.
    }


def run_command(cmd: List[str], timeout: Optional[float] = None) -> subprocess.CompletedProcess:
    """Run ``cmd`` capturing output; raises on non-zero exit. Times the call upstream."""
    return subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout)
