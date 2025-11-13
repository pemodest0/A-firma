"""Kraus operators, Choi matrices, and standard single-qubit channels."""
from __future__ import annotations

import numpy as np

from .ga import IDENTITY, PAULI_X, PAULI_Y, PAULI_Z

__all__ = [
    "apply_kraus",
    "is_tp",
    "choi_matrix",
    "dephasing",
    "depolarizing",
    "amplitude_damping",
]


def _as_matrix(mat: np.ndarray) -> np.ndarray:
    arr = np.asarray(mat, dtype=complex)
    if arr.shape != (2, 2):
        raise ValueError("Only single-qubit operators are supported.")
    return arr


def _as_density(rho: np.ndarray) -> np.ndarray:
    mat = _as_matrix(rho)
    mat = 0.5 * (mat + mat.conj().T)
    trace = np.trace(mat)
    if abs(trace) < 1e-12:
        raise ValueError("Density matrix must have non-zero trace.")
    return mat / trace


def _validate_channel(kraus_ops: list[np.ndarray]) -> list[np.ndarray]:
    if not kraus_ops:
        raise ValueError("At least one Kraus operator is required.")
    return [_as_matrix(k) for k in kraus_ops]


def apply_kraus(rho: np.ndarray, kraus_ops: list[np.ndarray]) -> np.ndarray:
    """Apply a Kraus map defined by ``kraus_ops`` to ``rho``."""
    state = _as_density(rho)
    ops = _validate_channel(kraus_ops)
    result = np.zeros_like(state)
    for op in ops:
        result += op @ state @ op.conj().T
    return result


def is_tp(kraus_ops: list[np.ndarray], tol: float = 1e-10) -> bool:
    """Return ``True`` if Kraus operators define (approximately) a TP map."""
    ops = _validate_channel(kraus_ops)
    metric = np.zeros_like(IDENTITY)
    for op in ops:
        metric += op.conj().T @ op
    return np.linalg.norm(metric - IDENTITY, ord=np.inf) <= tol


def choi_matrix(kraus_ops: list[np.ndarray]) -> np.ndarray:
    """Return the Choi matrix of the CP map defined by ``kraus_ops``."""
    ops = _validate_channel(kraus_ops)
    dim = ops[0].shape[0]
    choi = np.zeros((dim * dim, dim * dim), dtype=complex)
    for op in ops:
        vec = op.reshape(dim * dim, order="F")
        choi += np.outer(vec, vec.conj())
    return choi


def dephasing(p: float) -> list[np.ndarray]:
    """Return Kraus operators for a single-qubit dephasing channel."""
    if not 0 <= p <= 1:
        raise ValueError("p must lie in [0, 1].")
    k0 = np.sqrt(1 - p) * IDENTITY
    k1 = np.sqrt(p) * PAULI_Z
    return [k0, k1]


def depolarizing(p: float) -> list[np.ndarray]:
    """Return Kraus operators for the single-qubit depolarizing channel."""
    if not 0 <= p <= 1:
        raise ValueError("p must lie in [0, 1].")
    k0 = np.sqrt(1 - p) * IDENTITY
    scale = np.sqrt(p / 3) if p else 0.0
    kraus = [k0]
    if p:
        kraus.extend(scale * m for m in (PAULI_X, PAULI_Y, PAULI_Z))
    return kraus


def amplitude_damping(gamma: float) -> list[np.ndarray]:
    """Return Kraus operators for amplitude damping with parameter ``gamma``."""
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must lie in [0, 1].")
    k0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1 - gamma)]], dtype=complex)
    k1 = np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=complex)
    return [k0, k1]
