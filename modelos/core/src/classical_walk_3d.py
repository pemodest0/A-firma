import numpy as np


def classical_random_walk_3d(n_steps: int) -> np.ndarray:
    """Return the probability distribution of a 3D classical random walk after n_steps."""
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative.")

    size = 2 * n_steps + 1
    probs = np.zeros((size, size, size), dtype=float)
    origin = n_steps
    probs[origin, origin, origin] = 1.0

    for _ in range(n_steps):
        new_probs = np.zeros_like(probs)
        # +x / -x directions
        new_probs[1:, :, :] += (1 / 6) * probs[:-1, :, :]
        new_probs[:-1, :, :] += (1 / 6) * probs[1:, :, :]
        # +y / -y directions
        new_probs[:, 1:, :] += (1 / 6) * probs[:, :-1, :]
        new_probs[:, :-1, :] += (1 / 6) * probs[:, 1:, :]
        # +z / -z directions
        new_probs[:, :, 1:] += (1 / 6) * probs[:, :, :-1]
        new_probs[:, :, :-1] += (1 / 6) * probs[:, :, 1:]
        probs = new_probs

    return probs
