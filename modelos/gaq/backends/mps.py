"""Matrix-product state backend interface stubs."""
from __future__ import annotations

import numpy as np

__all__ = ["to_mps", "apply_gate_mps", "truncate"]


def to_mps(state: np.ndarray) -> object:
    """Convert a dense statevector into an internal MPS representation."""
    raise NotImplementedError


def apply_gate_mps(mps_state: object, gate: np.ndarray, sites: tuple[int, ...]) -> object:
    """Apply ``gate`` to ``mps_state`` on the provided ``sites``."""
    raise NotImplementedError


def truncate(mps_state: object, max_bond: int, tol: float = 1e-9) -> object:
    """Return a truncated copy of ``mps_state`` respecting ``max_bond``."""
    raise NotImplementedError
