from __future__ import annotations

import numpy as np

from engine.structural.rmt import mp_bounds, rmt_report, significant_eigs


def test_mp_bounds_basic_values() -> None:
    lmin, lmax = mp_bounds(T=100, N=25, sigma=1.0)
    assert np.isclose(lmin, 0.25, atol=1e-9)
    assert np.isclose(lmax, 2.25, atol=1e-9)


def test_significant_eigs_mask_and_indices() -> None:
    eigs = np.array([3.0, 2.0, 1.0, 0.5], dtype=float)
    mask, idx = significant_eigs(eigs, lambda_max=1.5)
    np.testing.assert_array_equal(mask, np.array([True, True, False, False]))
    np.testing.assert_array_equal(idx, np.array([0, 1]))


def test_rmt_report_has_expected_fields() -> None:
    eigs = np.array([4.0, 2.0, 1.0, 0.8], dtype=float)
    rep = rmt_report(eigs=eigs, T=120, N=4, sigma=1.0)
    assert rep["lambda1"] == 4.0
    assert rep["count_sig"] >= 1
    assert 0.0 <= rep["frac_sig"] <= 1.0
    assert rep["q"] == 30.0
