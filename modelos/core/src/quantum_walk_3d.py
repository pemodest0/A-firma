import math
from typing import Iterable

import numpy as np

DEFAULT_COIN_STATE = np.full(6, 1.0 / math.sqrt(6.0), dtype=np.complex128)


def grover_coin_3d() -> np.ndarray:
    """Return the 6x6 Grover diffusion coin for a 3D walk."""
    dim = 6
    j = np.ones((dim, dim), dtype=np.complex128)
    return (2.0 / dim) * j - np.eye(dim, dtype=np.complex128)


def _normalize_coin_matrix(coin_matrix: np.ndarray) -> np.ndarray:
    coin_matrix = np.asarray(coin_matrix, dtype=np.complex128)
    if coin_matrix.shape != (6, 6):
        raise ValueError("coin_matrix must be a 6x6 array.")
    return coin_matrix


def quantum_walk_3d_probabilities(
    n_steps: int,
    coin_matrix: np.ndarray | None = None,
    coin_state: Iterable[complex] | None = None,
) -> np.ndarray:
    """Return the probability distribution of a 3D discrete-time quantum walk."""
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative.")

    if coin_matrix is None:
        coin_matrix = grover_coin_3d()
    else:
        coin_matrix = _normalize_coin_matrix(coin_matrix)

    if coin_state is None:
        coin_state = DEFAULT_COIN_STATE
    else:
        coin_state = np.asarray(coin_state, dtype=np.complex128)
        if coin_state.shape != (6,):
            raise ValueError("coin_state must have length 6.")

    size = 2 * n_steps + 1
    origin = n_steps

    psi = np.zeros((6, size, size, size), dtype=np.complex128)
    psi[:, origin, origin, origin] = coin_state

    for _ in range(n_steps):
        psi_coin = coin_matrix @ psi.reshape(6, -1)
        psi_coin = psi_coin.reshape(6, size, size, size)

        shifted = np.zeros_like(psi)
        # +x (east)
        shifted[0, 1:, :, :] = psi_coin[0, :-1, :, :]
        # -x (west)
        shifted[1, :-1, :, :] = psi_coin[1, 1:, :, :]
        # +y (north)
        shifted[2, :, 1:, :] = psi_coin[2, :, :-1, :]
        # -y (south)
        shifted[3, :, :-1, :] = psi_coin[3, :, 1:, :]
        # +z (up)
        shifted[4, :, :, 1:] = psi_coin[4, :, :, :-1]
        # -z (down)
        shifted[5, :, :, :-1] = psi_coin[5, :, :, 1:]

        psi = shifted

    probabilities = np.sum(np.abs(psi) ** 2, axis=0)
    total = probabilities.sum()
    if not np.isclose(total, 1.0):
        probabilities = probabilities / total
    return probabilities
