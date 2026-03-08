from __future__ import annotations

import numpy as np

from engine.structural.rmt_clean import clean_correlation_mp_clip, mp_bounds, mp_bounds_from_q


def _random_corr(t: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(t, n))
    c = np.corrcoef(x, rowvar=False)
    c = 0.5 * (c + c.T)
    np.fill_diagonal(c, 1.0)
    return c


def test_mp_bounds_consistency() -> None:
    lmin_a, lmax_a = mp_bounds(T=200, N=50, sigma2=1.0)
    lmin_b, lmax_b = mp_bounds_from_q(q=50 / 200, sigma2=1.0)
    assert np.isclose(lmin_a, lmin_b, atol=1e-12)
    assert np.isclose(lmax_a, lmax_b, atol=1e-12)


def test_rmt_clean_output_contract() -> None:
    corr = _random_corr(t=160, n=40, seed=3)
    clean, info = clean_correlation_mp_clip(corr, T=160, mode="clip", keep_top_k=0)
    assert clean.shape == corr.shape
    assert np.isfinite(clean).all()
    np.testing.assert_allclose(np.diag(clean), np.ones(clean.shape[0]), atol=1e-8)
    vals = np.linalg.eigvalsh(clean)
    assert np.min(vals) >= -1e-8
    assert info["N"] == 40
    assert info["T"] == 160
    assert info["n_noise_eigs"] >= 0


def test_rmt_clean_threshold_mode_keeps_psd() -> None:
    corr = _random_corr(t=120, n=30, seed=13)
    clean, _ = clean_correlation_mp_clip(corr, T=120, mode="threshold", keep_top_k=3)
    vals = np.linalg.eigvalsh(clean)
    assert np.min(vals) >= -1e-8
    assert np.max(np.abs(np.diag(clean) - 1.0)) <= 1e-8


def test_rmt_clean_has_small_effect_when_t_much_larger_than_n() -> None:
    corr = _random_corr(t=3000, n=15, seed=19)
    clean, _ = clean_correlation_mp_clip(corr, T=3000, mode="clip", keep_top_k=0)
    diff = np.linalg.norm(clean - corr, ord="fro")
    assert diff < 2.0
