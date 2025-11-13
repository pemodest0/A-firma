"""Positive-operator valued measures and instruments."""
from __future__ import annotations

import numpy as np

from .ga import IDENTITY, PAULI_X, PAULI_Y, PAULI_Z

__all__ = ["effect", "instrument_post"]


def effect(alpha: float, direction: np.ndarray) -> np.ndarray:
    """Return a single-qubit POVM effect with Bloch weight ``alpha`` and axis."""
    alpha = float(alpha)
    if not 0.0 <= alpha <= 2.0:
        raise ValueError("alpha must lie in [0, 2].")
    vec = np.asarray(direction, dtype=float)
    if vec.shape != (3,):
        raise ValueError("direction must be a 3-vector.")
    norm = float(np.linalg.norm(vec))
    bound = min(alpha, 2.0 - alpha) + 1e-12
    if norm > bound:
        raise ValueError("Direction norm incompatible with a valid effect.")
    bloch = vec[0] * PAULI_X + vec[1] * PAULI_Y + vec[2] * PAULI_Z
    return 0.5 * (alpha * IDENTITY + bloch)


def instrument_post(
    kraus_ops: list[list[np.ndarray]],
    rho: np.ndarray,
) -> tuple[list[float], list[np.ndarray | None]]:
    """Evaluate a measurement instrument returning probabilities and posterior states."""
    if not kraus_ops:
        raise ValueError("At least one measurement outcome is required.")
    state = np.asarray(rho, dtype=complex)
    if state.shape != (2, 2):
        raise ValueError("instrument_post expects a 2x2 state.")
    state = 0.5 * (state + state.conj().T)
    trace = np.trace(state)
    if abs(trace) < 1e-12:
        raise ValueError("State must have non-zero trace.")
    state /= trace
    probs: list[float] = []
    posts: list[np.ndarray | None] = []
    for ops in kraus_ops:
        if not ops:
            probs.append(0.0)
            posts.append(None)
            continue
        accum = np.zeros_like(state)
        for op in ops:
            mat = np.asarray(op, dtype=complex)
            if mat.shape != (2, 2):
                raise ValueError("Kraus operators must be 2x2.")
            accum += mat @ state @ mat.conj().T
        prob = float(np.real(np.trace(accum)))
        prob = float(np.clip(prob, 0.0, 1.0))
        probs.append(prob)
        if prob <= 1e-12:
            posts.append(None)
        else:
            new_state = accum / prob
            new_state = 0.5 * (new_state + new_state.conj().T)
            posts.append(new_state)
    return probs, posts
