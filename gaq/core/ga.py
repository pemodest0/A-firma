"""Single-qubit geometric algebra primitives (Born rule, Lüders, rotors)."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

__all__ = [
    "rho_from_bloch",
    "bloch_from_rho",
    "projector_n",
    "born_prob",
    "lueders",
    "rotor_to_unitary",
    "expm",
]

PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
PAULI_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
PAULIS: tuple[np.ndarray, ...] = (PAULI_X, PAULI_Y, PAULI_Z)
IDENTITY = np.eye(2, dtype=complex)
TOL = 1e-12


def _as_vector(vec: Iterable[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    if arr.shape != (3,):
        raise ValueError("Bloch vectors must have dimension 3.")
    return arr


def _normalize(vec: Iterable[float]) -> np.ndarray:
    arr = _as_vector(vec)
    norm = np.linalg.norm(arr)
    if norm < TOL:
        raise ValueError("Direction vector must be non-zero.")
    return arr / norm


def _as_density(rho: np.ndarray) -> np.ndarray:
    mat = np.asarray(rho, dtype=complex)
    if mat.shape != (2, 2):
        raise ValueError("Density matrices must be 2x2.")
    mat = 0.5 * (mat + mat.conj().T)
    trace = np.trace(mat)
    if abs(trace) < TOL:
        raise ValueError("Density matrix must have non-zero trace.")
    mat /= trace
    return mat


def rho_from_bloch(m: np.ndarray) -> np.ndarray:
    """Return the density matrix that corresponds to Bloch vector ``m``."""
    bloch = _as_vector(m)
    if np.linalg.norm(bloch) > 1 + 1e-9:
        raise ValueError("Bloch vector norm must be ≤ 1.")
    state = IDENTITY.copy()
    for coeff, pauli in zip(bloch, PAULIS, strict=True):
        state += coeff * pauli
    return 0.5 * state


def bloch_from_rho(rho: np.ndarray) -> np.ndarray:
    """Return the Bloch vector for a single-qubit state ``rho``."""
    mat = _as_density(rho)
    comps = [float(np.real(np.trace(mat @ pauli))) for pauli in PAULIS]
    return np.array(comps, dtype=float)


def projector_n(n: np.ndarray) -> np.ndarray:
    """Return the rank-1 projector aligned with direction ``n``."""
    direction = _normalize(n)
    return rho_from_bloch(direction)


def born_prob(rho: np.ndarray, n: np.ndarray) -> float:
    """Return the Born probability for measuring ``rho`` along ``n``."""
    state = _as_density(rho)
    proj = projector_n(n)
    prob = float(np.real(np.trace(proj @ state)))
    prob = float(np.clip(prob, 0.0, 1.0))
    return prob


def lueders(rho: np.ndarray, n: np.ndarray) -> tuple[float, np.ndarray]:
    """Return probability and post-measurement state for projective axis ``n``."""
    state = _as_density(rho)
    proj = projector_n(n)
    post = proj @ state @ proj
    prob = float(np.real(np.trace(post)))
    prob = float(np.clip(prob, 0.0, 1.0))
    if prob <= TOL:
        return 0.0, proj.copy()
    new_state = post / prob
    new_state = 0.5 * (new_state + new_state.conj().T)
    return prob, new_state


def rotor_to_unitary(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return SU(2) unitary for rotation around ``axis`` by ``angle`` radians."""
    if abs(angle) < TOL:
        return IDENTITY.copy()
    direction = _normalize(axis)
    generator = direction[0] * PAULI_X + direction[1] * PAULI_Y + direction[2] * PAULI_Z
    return expm(-0.5j * angle * generator)


def expm(a: np.ndarray) -> np.ndarray:
    """Simple scaling-and-squaring matrix exponential for 2x2 matrices."""
    mat = np.asarray(a, dtype=complex)
    if mat.shape != (2, 2):
        raise ValueError("Only 2x2 matrices are supported.")
    norm = float(np.linalg.norm(mat, ord=np.inf))
    if norm == 0.0:
        return IDENTITY.copy()
    if norm <= 0.5:
        scale = 0
    else:
        scale = max(0, int(math.ceil(math.log2(norm))) + 1)
    scaled = mat / (2**scale) if scale else mat.copy()
    result = IDENTITY.copy()
    term = IDENTITY.copy()
    for k in range(1, 33):
        term = (term @ scaled) / k
        result = result + term
        if np.linalg.norm(term, ord=np.inf) < 1e-16:
            break
    for _ in range(scale):
        result = result @ result
    return result
