from __future__ import annotations

import numpy as np

from gaq.core.channels import (
    amplitude_damping,
    apply_kraus,
    choi_matrix,
    dephasing,
    depolarizing,
    is_tp,
)
from gaq.core.ga import rho_from_bloch


def _random_density(rng: np.random.Generator) -> np.ndarray:
    vec = rng.normal(size=3)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return rho_from_bloch(np.array([0.0, 0.0, 1.0]))
    bloch = vec / norm * rng.random()
    return rho_from_bloch(bloch)


def test_apply_kraus_preserves_trace_and_positivity() -> None:
    rng = np.random.default_rng(1234)
    rho = _random_density(rng)
    channel = depolarizing(0.2)
    rho_out = apply_kraus(rho, channel)
    assert abs(np.trace(rho_out) - 1.0) < 1e-12
    eigvals = np.linalg.eigvalsh(rho_out)
    assert eigvals.min() >= -1e-12


def test_dephasing_suppresses_off_diagonal() -> None:
    p = 0.35
    rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
    rho_out = apply_kraus(rho, dephasing(p))
    expected_off = (1 - 2 * p) * 0.5
    assert abs(rho_out[0, 1] - expected_off) < 1e-12
    assert abs(rho_out[1, 0] - expected_off) < 1e-12


def test_depolarizing_moves_towards_identity() -> None:
    rng = np.random.default_rng(2)
    rho = _random_density(rng)
    p = 0.4
    rho_out = apply_kraus(rho, depolarizing(p))
    paulis = [
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
        np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
        np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
    ]
    sigma_sum = sum(p @ rho @ p.conj().T for p in paulis)
    expected = (1 - p) * rho + (p / 3) * sigma_sum
    assert np.linalg.norm(rho_out - expected) < 1e-12


def test_amplitude_damping_populates_ground_state() -> None:
    gamma = 0.6
    rho_excited = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
    rho_out = apply_kraus(rho_excited, amplitude_damping(gamma))
    assert abs(rho_out[0, 0] - gamma) < 1e-12
    assert abs(rho_out[1, 1] - (1 - gamma)) < 1e-12


def test_choi_is_psd_and_channels_trace_preserving() -> None:
    channels = [dephasing(0.2), depolarizing(0.1), amplitude_damping(0.05)]
    for ops in channels:
        assert is_tp(ops)
        choi = choi_matrix(ops)
        choi = 0.5 * (choi + choi.conj().T)
        eigvals = np.linalg.eigvalsh(choi)
        assert eigvals.min() >= -1e-12
