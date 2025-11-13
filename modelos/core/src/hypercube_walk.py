import math
from typing import Callable, Iterable

import numpy as np


def classical_hypercube_time_series(dimensions: int, n_steps: int) -> np.ndarray:
    """Return the probability time series for a classical walk on the d-hypercube."""
    if dimensions <= 0:
        raise ValueError("dimensions must be positive.")
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative.")

    num_vertices = 1 << dimensions
    indices = np.arange(num_vertices, dtype=np.int64)

    series = np.zeros((n_steps + 1, num_vertices), dtype=float)
    current = np.zeros(num_vertices, dtype=float)
    current[0] = 1.0
    series[0] = current

    for step in range(1, n_steps + 1):
        new_prob = np.zeros_like(current)
        for axis in range(dimensions):
            neighbors = indices ^ (1 << axis)
            new_prob += current[neighbors]
        current = new_prob / dimensions
        series[step] = current

    return series


def classical_hypercube_walk(dimensions: int, n_steps: int) -> np.ndarray:
    """Return the final probability distribution of the classical walk."""
    return classical_hypercube_time_series(dimensions, n_steps)[-1]


def _hadamard_matrix(order: int) -> np.ndarray:
    if order == 1:
        return np.array([[1.0]], dtype=np.complex128)
    if order & (order - 1) != 0:
        raise ValueError("Hadamard matrix requires order to be a power of two.")
    H = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
    current = 2
    while current < order:
        H = np.block([[H, H], [H, -H]])
        current *= 2
    return H


def hadamard_coin(dimensions: int) -> np.ndarray:
    """Return a normalized Hadamard coin matrix for degree d."""
    H = _hadamard_matrix(dimensions)
    return H / math.sqrt(dimensions)


def grover_coin(dimensions: int) -> np.ndarray:
    """Return the Grover diffusion coin for degree d."""
    if dimensions <= 0:
        raise ValueError("dimensions must be positive.")
    J = np.ones((dimensions, dimensions), dtype=np.complex128)
    identity = np.eye(dimensions, dtype=np.complex128)
    return (2.0 / dimensions) * J - identity


def quantum_hypercube_time_series(
    dimensions: int,
    n_steps: int,
    coin_matrix: np.ndarray | None = None,
    coin_state: Iterable[complex] | None = None,
) -> np.ndarray:
    """Return the probability time series for the quantum walk on the hypercube."""
    if dimensions <= 0:
        raise ValueError("dimensions must be positive.")
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative.")

    if coin_matrix is None:
        coin_matrix = hadamard_coin(dimensions)
    else:
        coin_matrix = np.asarray(coin_matrix, dtype=np.complex128)
        if coin_matrix.shape != (dimensions, dimensions):
            raise ValueError("coin_matrix must be of shape (dimensions, dimensions).")

    if coin_state is None:
        coin_state = np.full(dimensions, 1.0 / math.sqrt(dimensions), dtype=np.complex128)
    else:
        coin_state = np.asarray(coin_state, dtype=np.complex128)
        if coin_state.shape != (dimensions,):
            raise ValueError("coin_state must have length equal to dimensions.")

    num_vertices = 1 << dimensions
    indices = np.arange(num_vertices, dtype=np.int64)

    series = np.zeros((n_steps + 1, num_vertices), dtype=float)
    psi = np.zeros((dimensions, num_vertices), dtype=np.complex128)
    psi[:, 0] = coin_state
    series[0] = np.sum(np.abs(psi) ** 2, axis=0)

    for step in range(1, n_steps + 1):
        psi_coin = coin_matrix @ psi
        shifted = np.zeros_like(psi)
        for axis in range(dimensions):
            neighbors = indices ^ (1 << axis)
            shifted[axis, neighbors] = psi_coin[axis]
        psi = shifted

        probabilities = np.sum(np.abs(psi) ** 2, axis=0)
        total = probabilities.sum()
        if not np.isclose(total, 1.0):
            probabilities /= total
        series[step] = probabilities

    return series


def quantum_hypercube_walk(
    dimensions: int,
    n_steps: int,
    coin_matrix: np.ndarray | None = None,
    coin_state: Iterable[complex] | None = None,
) -> np.ndarray:
    """Return the final probability distribution for a quantum walk on a hypercube."""
    return quantum_hypercube_time_series(dimensions, n_steps, coin_matrix, coin_state)[-1]


def quantum_hypercube_time_series_adaptive(
    dimensions: int,
    n_steps: int,
    coin_callback: Callable[[int, np.ndarray], np.ndarray],
    coin_state: Iterable[complex] | None = None,
) -> np.ndarray:
    """Quantum walk where the coin may change every step based on the current distribution."""
    if dimensions <= 0:
        raise ValueError("dimensions must be positive.")
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative.")
    if not callable(coin_callback):
        raise TypeError("coin_callback must be callable.")

    if coin_state is None:
        coin_state = np.full(dimensions, 1.0 / math.sqrt(dimensions), dtype=np.complex128)
    else:
        coin_state = np.asarray(coin_state, dtype=np.complex128)
        if coin_state.shape != (dimensions,):
            raise ValueError("coin_state must have length equal to dimensions.")

    num_vertices = 1 << dimensions
    indices = np.arange(num_vertices, dtype=np.int64)

    series = np.zeros((n_steps + 1, num_vertices), dtype=float)
    psi = np.zeros((dimensions, num_vertices), dtype=np.complex128)
    psi[:, 0] = coin_state
    series[0] = np.sum(np.abs(psi) ** 2, axis=0)

    for step in range(1, n_steps + 1):
        distribution = series[step - 1]
        coin_matrix = coin_callback(step - 1, distribution)
        matrix = np.asarray(coin_matrix, dtype=np.complex128)
        if matrix.shape != (dimensions, dimensions):
            raise ValueError("coin_callback must return a matrix of shape (dimensions, dimensions).")

        psi_coin = matrix @ psi
        shifted = np.zeros_like(psi)
        for axis in range(dimensions):
            neighbors = indices ^ (1 << axis)
            shifted[axis, neighbors] = psi_coin[axis]
        psi = shifted

        probabilities = np.sum(np.abs(psi) ** 2, axis=0)
        total = probabilities.sum()
        if not np.isclose(total, 1.0):
            probabilities /= total
        series[step] = probabilities

    return series


def hamming_weights(dimensions: int) -> np.ndarray:
    num_vertices = 1 << dimensions
    indices = np.arange(num_vertices, dtype=np.int64)
    weights = np.zeros(num_vertices, dtype=np.int64)
    for bit in range(dimensions):
        weights += (indices >> bit) & 1
    return weights


def aggregate_by_weight(probabilities: np.ndarray, weights: np.ndarray) -> np.ndarray:
    max_weight = weights.max()
    agg = np.zeros(max_weight + 1, dtype=float)
    for w in range(max_weight + 1):
        agg[w] = probabilities[weights == w].sum()
    return agg
