"""Shared helpers for Python parity scripts (safe sys.path for in-repo venv)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Must run before importing evoc/numba.
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

def _evoc_root() -> Path:
    for key in ("EVOC_ROOT", "EVOC_PARITY_ROOT"):
        value = os.environ.get(key)
        if value:
            return Path(value)
    raise RuntimeError(
        "Set EVOC_ROOT (or EVOC_PARITY_ROOT) to your Python EVoC checkout "
        "(https://github.com/TutteInstitute/evoc)."
    )


EVOC_ROOT = _evoc_root()
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def scrub_project_root_from_syspath() -> None:
    """Remove only the evoc-rs repo root from sys.path, not .venv inside it."""
    proj = PROJECT_ROOT.resolve()
    sys.path[:] = [p for p in sys.path if Path(p).resolve() != proj]


def prepend_evoc_package() -> None:
    scrub_project_root_from_syspath()
    evoc_s = str(EVOC_ROOT.resolve())
    if evoc_s not in sys.path:
        sys.path.insert(0, evoc_s)
