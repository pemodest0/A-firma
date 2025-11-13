from __future__ import annotations

import math
from typing import Iterable

import numpy as np

DEFAULT_COIN_STATE = np.full(4, 0.5, dtype=np.complex128)


def hadamard_coin_2d() -> np.ndarray:
    """Return the tensor product Hadamard⊗Hadamard coin."""
    h1d = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / math.sqrt(2.0)
    return np.kron(h1d, h1d)


def grover_coin_2d() -> np.ndarray:
    """Return the 4x4 Grover diffusion coin."""
    dim = 4
    j = np.ones((dim, dim), dtype=np.complex128)
    return (2.0 / dim) * j - np.eye(dim, dtype=np.complex128)


def _normalize_coin_matrix(coin_matrix: np.ndarray) -> np.ndarray:
    coin_matrix = np.asarray(coin_matrix, dtype=np.complex128)
    if coin_matrix.shape != (4, 4):
        raise ValueError("coin_matrix must be a 4x4 array.")
    return coin_matrix


def quantum_walk_2d_probabilities(
    n_steps: int,
    coin_matrix: np.ndarray | None = None,
    coin_state: Iterable[complex] | None = None,
) -> np.ndarray:
    """Return the final probability distribution of a 2D quantum walk."""
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative.")

    if coin_matrix is None:
        coin_matrix = grover_coin_2d()
    else:
        coin_matrix = _normalize_coin_matrix(coin_matrix)

    if coin_state is None:
        coin_state = DEFAULT_COIN_STATE
    else:
        coin_state = np.asarray(coin_state, dtype=np.complex128)
        if coin_state.shape != (4,):
            raise ValueError("coin_state must have length 4.")

    size = 2 * n_steps + 1
    origin = n_steps

    psi = np.zeros((4, size, size), dtype=np.complex128)
    psi[:, origin, origin] = coin_state

    for _ in range(n_steps):
        psi_coin = coin_matrix @ psi.reshape(4, -1)
        psi_coin = psi_coin.reshape(4, size, size)

        shifted = np.zeros_like(psi)
        shifted[0, 1:, :] = psi_coin[0, :-1, :]  # +x
        shifted[1, :-1, :] = psi_coin[1, 1:, :]  # -x
        shifted[2, :, 1:] = psi_coin[2, :, :-1]  # +y
        shifted[3, :, :-1] = psi_coin[3, :, 1:]  # -y
        psi = shifted

    probabilities = np.sum(np.abs(psi) ** 2, axis=0)
    total = probabilities.sum()
    if not np.isclose(total, 1.0):
        probabilities = probabilities / total
    return probabilities
