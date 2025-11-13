from __future__ import annotations

import numpy as np

from gaq.core.ga import rotor_to_unitary

PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
PAULI_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
IDENTITY = np.eye(2, dtype=complex)


def _random_direction(rng: np.random.Generator) -> np.ndarray:
    vec = rng.normal(size=3)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return _random_direction(rng)
    return vec / norm


def _reference_unitary(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    generator = axis[0] * PAULI_X + axis[1] * PAULI_Y + axis[2] * PAULI_Z
    return (
        np.cos(angle / 2.0) * IDENTITY - 1j * np.sin(angle / 2.0) * generator
    )


def test_rotor_matches_closed_form_unitary() -> None:
    rng = np.random.default_rng(707)
    for _ in range(256):
        axis = _random_direction(rng)
        angle = rng.uniform(-4.0 * np.pi, 4.0 * np.pi)
        rotor = rotor_to_unitary(axis, angle)
        reference = _reference_unitary(axis, angle)
        diff = np.linalg.norm(rotor - reference)
        assert diff < 1e-12
        unitary_check = rotor.conj().T @ rotor
        assert np.linalg.norm(unitary_check - IDENTITY) < 1e-12
