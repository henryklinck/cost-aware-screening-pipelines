"""Reproducibility and provenance helpers (spec section 29).

Every experiment should be reconstructable from what it records: the git commit, the
config, the seed, the hardware, and the runtime.  This module keeps that bookkeeping in
one place so experiment scripts stay focused on the science.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def git_commit(short: bool = False) -> Optional[str]:
    """Return the current git commit hash, or ``None`` if unavailable."""
    cmd = ["git", "rev-parse", "--short" if short else "HEAD"]
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def git_is_dirty() -> Optional[bool]:
    """True if the working tree has uncommitted changes; ``None`` if git unavailable."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        )
        return bool(out.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def hardware_info() -> Dict[str, Any]:
    """A small, portable snapshot of the machine (spec section 17/29).

    Deliberately minimal and stdlib-only.  Phase-2 GPU adapters will extend this with
    GPU type / memory measured at runtime.
    """
    import os

    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }


@dataclass
class ExperimentRecord:
    """The provenance envelope saved alongside every experiment's results.

    Mirrors the example manifest in spec section 29.  ``results`` and ``config`` are
    free-form dicts so this works for both Phase-1 simulations and Phase-2 real runs.
    """

    experiment_id: str
    config: Dict[str, Any]
    seed: Optional[int] = None
    git_commit: Optional[str] = field(default_factory=git_commit)
    git_dirty: Optional[bool] = field(default_factory=git_is_dirty)
    hardware: Dict[str, Any] = field(default_factory=hardware_info)
    created_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    runtime_seconds: Optional[float] = None
    results: Dict[str, Any] = field(default_factory=dict)

    def to_json(self, path: Path) -> Path:
        """Write the record to ``path`` as pretty JSON and return the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str))
        return path


class Timer:
    """Context manager measuring wall-clock seconds.

    >>> with Timer() as t:
    ...     do_work()
    >>> t.seconds
    """

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        self.seconds: Optional[float] = None
        return self

    def __exit__(self, *exc) -> None:
        self.seconds = time.perf_counter() - self._start
