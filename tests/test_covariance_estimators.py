from __future__ import annotations

import numpy as np

from engine.structural.covariance_estimators import cov_to_corr, ensure_psd, estimate_corr, estimate_cov


def _is_psd(mat: np.ndarray, tol: float = 1e-8) -> bool:
    vals = np.linalg.eigvalsh(np.asarray(mat, dtype=float))
    return bool(np.min(vals) >= -float(tol))


def test_estimate_cov_supported_methods_are_psd() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(180, 24))
    for method in ["sample", "ewma", "ledoit_wolf", "oas"]:
        cov = estimate_cov(x, method=method, ewma_lambda=0.94)
        assert cov.shape == (24, 24)
        assert np.isfinite(cov).all()
        assert _is_psd(cov)


def test_estimate_corr_has_unit_diagonal() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(220, 16))
    corr = estimate_corr(x, method="ledoit_wolf")
    assert corr.shape == (16, 16)
    assert np.isfinite(corr).all()
    np.testing.assert_allclose(np.diag(corr), np.ones(16), atol=1e-8)
    assert _is_psd(corr)


def test_sample_correlation_is_scale_invariant() -> None:
    rng = np.random.default_rng(23)
    x = rng.normal(size=(140, 12))
    scale = np.linspace(0.5, 3.5, 12)
    corr_a = estimate_corr(x, method="sample")
    corr_b = estimate_corr(x * scale[None, :], method="sample")
    np.testing.assert_allclose(corr_a, corr_b, atol=1e-8)


def test_ledoitwolf_conditioning_is_not_worse_than_sample_in_rank_stress() -> None:
    rng = np.random.default_rng(101)
    x = rng.normal(size=(40, 80))
    cov_sample = estimate_cov(x, method="sample")
    cov_lw = estimate_cov(x, method="ledoit_wolf")
    c_sample = np.linalg.cond(cov_sample)
    c_lw = np.linalg.cond(cov_lw)
    assert np.isfinite(c_sample)
    assert np.isfinite(c_lw)
    assert c_lw <= c_sample


def test_cov_to_corr_and_ensure_psd_roundtrip() -> None:
    rng = np.random.default_rng(5)
    a = rng.normal(size=(10, 10))
    cov = a @ a.T
    cov[0, 1] *= 1.4
    cov[1, 0] = cov[0, 1]
    corr = cov_to_corr(cov)
    corr2 = ensure_psd(corr)
    corr3 = cov_to_corr(corr2)
    np.testing.assert_allclose(np.diag(corr3), np.ones(10), atol=1e-8)
    assert _is_psd(corr3)
