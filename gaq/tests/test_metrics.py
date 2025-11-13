from __future__ import annotations

import numpy as np

from gaq.analysis.metrics import autocorrelation, hamming_diffusion, spectral_gap_unbiased


def test_spectral_gap_matches_closed_form() -> None:
    evals, gap = spectral_gap_unbiased(2)
    assert abs(gap - 1.0) < 1e-9
    _, gap3 = spectral_gap_unbiased(3)
    assert abs(gap3 - (2 / 3)) < 1e-9


def test_autocorrelation_normalizes_and_decays() -> None:
    series = np.sin(np.linspace(0, 2 * np.pi, 50))
    ac = autocorrelation(series, max_lag=5)
    assert ac[0] == 1.0
    assert ac[1] < ac[0]


def test_hamming_diffusion_simple_case() -> None:
    history = [
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.5, 0.5, 0.0, 0.0]),
        np.array([0.25, 0.25, 0.25, 0.25]),
    ]
    diff = hamming_diffusion(history)
    assert diff[0] == 0.0
    assert diff[-1] > diff[0]
