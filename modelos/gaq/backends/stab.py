"""Stabilizer + T-gate backend interface stubs."""
from __future__ import annotations

import numpy as np

__all__ = ["to_stabilizer_state", "apply_clifford", "inject_t"]


def to_stabilizer_state(state: np.ndarray) -> object:
    """Convert a statevector into a stabilizer tableau or simulator object."""
    raise NotImplementedError


def apply_clifford(stab_state: object, clifford: str) -> object:
    """Apply a Clifford gate/sequence labelled by ``clifford``."""
    raise NotImplementedError


def inject_t(stab_state: object, location: int) -> object:
    """Inject a T gate at ``location`` returning a non-Clifford resource state."""
    raise NotImplementedError
