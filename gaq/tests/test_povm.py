from __future__ import annotations

import numpy as np

from gaq.core.ga import bloch_from_rho, projector_n, rho_from_bloch
from gaq.core.povm import effect, instrument_post


TRINE_AXES = [
    np.array([1.0, 0.0, 0.0]),
    np.array([-0.5, np.sqrt(3) / 2, 0.0]),
    np.array([-0.5, -np.sqrt(3) / 2, 0.0]),
]


def test_trine_povm_effects_form_identity_partition() -> None:
    factors = 2.0 / 3.0
    effects = [effect(factors, factors * axis) for axis in TRINE_AXES]
    total = sum(effects)
    assert np.linalg.norm(total - np.eye(2)) < 1e-12
    for eff in effects:
        eigvals = np.linalg.eigvalsh(eff)
        assert eigvals.min() >= -1e-12
        assert eigvals.max() <= 1 + 1e-12


def test_instrument_post_returns_probabilities_and_states() -> None:
    # Computational basis projectors as Kraus ops
    k0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    k1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
    rho = rho_from_bloch(np.array([0.2, -0.4, 0.6]))
    probs, posts = instrument_post([[k0], [k1]], rho)
    assert abs(sum(probs) - 1.0) < 1e-12
    for prob in probs:
        assert 0.0 - 1e-12 <= prob <= 1.0 + 1e-12
    if probs[0] > 1e-12:
        assert np.linalg.norm(posts[0] - projector_n([0.0, 0.0, 1.0])) < 1e-12
    if probs[1] > 1e-12:
        assert np.linalg.norm(posts[1] - projector_n([0.0, 0.0, -1.0])) < 1e-12


def test_instrument_handles_zero_probability_outcomes() -> None:
    plus_state = projector_n([1.0, 0.0, 0.0])
    kraus_plus = [np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)]
    kraus_minus = [np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=complex)]
    probs, posts = instrument_post([kraus_plus, kraus_minus], plus_state)
    assert abs(sum(probs) - 1.0) < 1e-12
    assert probs[1] < 1e-12
    assert posts[1] is None
    bloch_post = bloch_from_rho(posts[0])
    assert np.linalg.norm(bloch_post - np.array([1.0, 0.0, 0.0])) < 1e-12
