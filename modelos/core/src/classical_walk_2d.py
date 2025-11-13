from __future__ import annotations

import numpy as np


def classical_random_walk_2d(n_steps: int) -> np.ndarray:
    """Return the probability distribution of a 2D classical random walk after n_steps."""
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative.")

    size = 2 * n_steps + 1
    probs = np.zeros((size, size), dtype=float)
    origin = n_steps
    probs[origin, origin] = 1.0

    for _ in range(n_steps):
        new_probs = np.zeros_like(probs)
        new_probs[1:, :] += 0.25 * probs[:-1, :]  # shift +x
        new_probs[:-1, :] += 0.25 * probs[1:, :]  # shift -x
        new_probs[:, 1:] += 0.25 * probs[:, :-1]  # shift +y
        new_probs[:, :-1] += 0.25 * probs[:, 1:]  # shift -y
        probs = new_probs

    return probs

