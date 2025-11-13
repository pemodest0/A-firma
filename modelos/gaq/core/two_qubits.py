"""Two-qubit helper states and CHSH functional."""
from __future__ import annotations

import numpy as np

from .ga import PAULI_X, PAULI_Y, PAULI_Z, rho_from_bloch

__all__ = ["singlet", "chsh_value", "projector_axis"]

TOL = 1e-12


def _normalize(axis: np.ndarray) -> np.ndarray:
    vec = np.asarray(axis, dtype=float)
    if vec.shape != (3,):
        raise ValueError("Measurement axes must be 3-vectors.")
    norm = np.linalg.norm(vec)
    if norm < TOL:
        raise ValueError("Axis must be non-zero.")
    return vec / norm


def _as_two_qubit_state(rho: np.ndarray) -> np.ndarray:
    mat = np.asarray(rho, dtype=complex)
    if mat.shape != (4, 4):
        raise ValueError("Input must be a 4x4 density matrix.")
    mat = 0.5 * (mat + mat.conj().T)
    trace = np.trace(mat)
    if abs(trace) < TOL:
        raise ValueError("State must have non-zero trace.")
    return mat / trace


def singlet() -> np.ndarray:
    """Return the singlet Bell state density matrix."""
    psi = np.array([0.0, 1 / np.sqrt(2.0), -1 / np.sqrt(2.0), 0.0], dtype=complex)
    return np.outer(psi, psi.conj())


def projector_axis(axis: np.ndarray) -> np.ndarray:
    """Projector for measurement outcome along ``axis``."""
    return rho_from_bloch(_normalize(axis))


def chsh_value(
    rho: np.ndarray,
    a1: np.ndarray,
    a2: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
) -> float:
    """Return the CHSH correlator value for settings ``a1,a2,b1,b2``."""
    state = _as_two_qubit_state(rho)
    axes = [_normalize(vec) for vec in (a1, a2, b1, b2)]
    a1_vec, a2_vec, b1_vec, b2_vec = axes
    paulis = (PAULI_X, PAULI_Y, PAULI_Z)
    correl = np.zeros((3, 3), dtype=float)
    for i, sigma_a in enumerate(paulis):
        for j, sigma_b in enumerate(paulis):
            op = np.kron(sigma_a, sigma_b)
            correl[i, j] = float(np.real(np.trace(state @ op)))

    def expect(a_vec: np.ndarray, b_vec: np.ndarray) -> float:
        return float(a_vec @ correl @ b_vec)

    return expect(a1_vec, b1_vec) + expect(a1_vec, b2_vec) + expect(a2_vec, b1_vec) - expect(a2_vec, b2_vec)
