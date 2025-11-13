"""Pauli transfer matrix utilities."""
from __future__ import annotations

import numpy as np

from gaq.core.ga import IDENTITY, PAULI_X, PAULI_Y, PAULI_Z

__all__ = ["pauli_basis", "kraus_to_ptm", "compose", "apply_ptm"]


def pauli_basis() -> list[np.ndarray]:
    """Return ordered single-qubit Pauli basis matrices."""
    return [IDENTITY, PAULI_X, PAULI_Y, PAULI_Z]


def _validate_kraus(kraus_ops: list[np.ndarray]) -> list[np.ndarray]:
    if not kraus_ops:
        raise ValueError("At least one Kraus operator is required.")
    validated = []
    for op in kraus_ops:
        arr = np.asarray(op, dtype=complex)
        if arr.shape != (2, 2):
            raise ValueError("Kraus operators must be 2x2.")
        validated.append(arr)
    return validated


def _apply_to_operator(kraus_ops: list[np.ndarray], operator: np.ndarray) -> np.ndarray:
    result = np.zeros((2, 2), dtype=complex)
    for op in kraus_ops:
        result += op @ operator @ op.conj().T
    return result


def kraus_to_ptm(kraus_ops: list[np.ndarray]) -> np.ndarray:
    """Return the Pauli transfer matrix for the channel defined by ``kraus_ops``."""
    ops = _validate_kraus(kraus_ops)
    basis = pauli_basis()
    ptm_matrix = np.zeros((4, 4), dtype=float)
    for j, bj in enumerate(basis):
        image = _apply_to_operator(ops, bj)
        for i, bi in enumerate(basis):
            coeff = 0.5 * np.real(np.trace(bi.conj().T @ image))
            ptm_matrix[i, j] = coeff
    ptm_matrix[0] = np.array([1.0, 0.0, 0.0, 0.0])
    return ptm_matrix


def compose(ptm_a: np.ndarray, ptm_b: np.ndarray) -> np.ndarray:
    """Return the PTM representing applying ``ptm_b`` followed by ``ptm_a``."""
    a = np.asarray(ptm_a, dtype=float)
    b = np.asarray(ptm_b, dtype=float)
    if a.shape != (4, 4) or b.shape != (4, 4):
        raise ValueError("compose expects 4x4 matrices.")
    return a @ b


def apply_ptm(ptm: np.ndarray, bloch: np.ndarray) -> np.ndarray:
    """Apply PTM to a Bloch vector (including affine component)."""
    matrix = np.asarray(ptm, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError("PTM must be 4x4.")
    vec = np.asarray(bloch, dtype=float)
    if vec.shape != (3,):
        raise ValueError("Bloch vector must have length 3.")
    augmented = np.concatenate(([1.0], vec))
    result = matrix @ augmented
    return result[1:]
