from __future__ import annotations

import numpy as np
import pytest

from gaq.backends import mps, ptm, stab
from gaq.core import channels
from gaq.core.ga import bloch_from_rho, rho_from_bloch


def _random_bloch(rng: np.random.Generator) -> np.ndarray:
    vec = rng.normal(size=3)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0]) * 0.0
    return vec / norm * rng.random()


def test_ptm_matches_sequential_channels() -> None:
    rng = np.random.default_rng(404)
    bloch = _random_bloch(rng)
    rho = rho_from_bloch(bloch)
    chan_a = channels.amplitude_damping(0.15)
    chan_b = channels.dephasing(0.25)
    rho_seq = channels.apply_kraus(channels.apply_kraus(rho, chan_b), chan_a)

    ptm_a = ptm.kraus_to_ptm(chan_a)
    ptm_b = ptm.kraus_to_ptm(chan_b)
    combined = ptm.compose(ptm_a, ptm_b)
    new_bloch = ptm.apply_ptm(combined, bloch)
    rho_ptm = rho_from_bloch(new_bloch)
    assert np.linalg.norm(rho_seq - rho_ptm) < 1e-12


def test_ptm_trace_preserving_component() -> None:
    channel = channels.depolarizing(0.33)
    ptm_matrix = ptm.kraus_to_ptm(channel)
    assert np.allclose(ptm_matrix[0], np.array([1.0, 0.0, 0.0, 0.0]))
    rng = np.random.default_rng(1)
    bloch = _random_bloch(rng)
    new_bloch = ptm.apply_ptm(ptm_matrix, bloch)
    rho_out = channels.apply_kraus(rho_from_bloch(bloch), channel)
    assert np.linalg.norm(new_bloch - bloch_from_rho(rho_out)) < 1e-12


def test_backend_stubs_exist() -> None:
    dummy_state = np.array([1.0, 0.0], dtype=complex)
    with pytest.raises(NotImplementedError):
        mps.to_mps(dummy_state)
    with pytest.raises(NotImplementedError):
        stab.to_stabilizer_state(dummy_state)
    with pytest.raises(NotImplementedError):
        mps.apply_gate_mps(object(), np.eye(2), (0,))
    with pytest.raises(NotImplementedError):
        stab.apply_clifford(object(), "H")
    with pytest.raises(NotImplementedError):
        mps.truncate(object(), 4)
    with pytest.raises(NotImplementedError):
        stab.inject_t(object(), 0)
