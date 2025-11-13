from __future__ import annotations

import numpy as np

from gaq.core.ga import bloch_from_rho, lueders, rho_from_bloch


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


def test_lueders_aligns_bloch_vector_with_measurement_axis() -> None:
    rng = np.random.default_rng(2718)
    for _ in range(128):
        rho = rho_from_bloch(_random_bloch(rng))
        axis = _random_direction(rng)
        p_plus, rho_plus = lueders(rho, axis)
        p_minus, rho_minus = lueders(rho, -axis)
        assert abs(p_plus + p_minus - 1.0) < 1e-12

        if p_plus > 1e-14:
            bloch_plus = bloch_from_rho(rho_plus)
            assert np.linalg.norm(bloch_plus - axis) < 1e-12
        if p_minus > 1e-14:
            bloch_minus = bloch_from_rho(rho_minus)
            assert np.linalg.norm(bloch_minus + axis) < 1e-12
