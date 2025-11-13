"""Coin operators expressed via GA rotors."""
from __future__ import annotations

import numpy as np

from gaq.core.ga import rotor_to_unitary

__all__ = ["coin_rotor"]


def coin_rotor(theta: float, axis: tuple[float, float, float] = (0.0, 1.0, 0.0)) -> np.ndarray:
    """Return the single-qubit coin obtained from a GA rotor."""
    direction = np.asarray(axis, dtype=float)
    return rotor_to_unitary(direction, theta)
