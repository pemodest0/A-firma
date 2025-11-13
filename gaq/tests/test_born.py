from __future__ import annotations

import numpy as np

from gaq.core.ga import born_prob, rho_from_bloch


def _random_direction(rng: np.random.Generator) -> np.ndarray:
    vec = rng.normal(size=3)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return _random_direction(rng)
    return vec / norm


def _random_bloch(rng: np.random.Generator) -> np.ndarray:
    direction = _random_direction(rng)
    radius = rng.random()
    return direction * radius


def test_born_matches_bloch_formula() -> None:
    rng = np.random.default_rng(314159)
    for _ in range(1000):
        m = _random_bloch(rng)
        n = _random_direction(rng)
        rho = rho_from_bloch(m)
        prob = born_prob(rho, n)
        ref = 0.5 * (1.0 + float(np.dot(m, n)))
        assert abs(prob - ref) < 1e-12
        assert -1e-12 <= prob <= 1 + 1e-12
