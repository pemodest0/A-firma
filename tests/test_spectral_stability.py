from __future__ import annotations

import numpy as np
import pandas as pd

from engine.structural.stability_metrics import (
    ModeStabilityThresholds,
    apply_mode_stability_gate,
    dominant_mode_series,
    summarize_mode_stability,
)
from scripts.lab.run_corr_macro_offline import _estimate_corr_matrix


def test_estimate_corr_matrix_profiles_are_finite() -> None:
    rng = np.random.default_rng(17)
    x = rng.normal(size=(180, 20))
    profiles = [
        ("sample", False),
        ("ewma", False),
        ("ledoit_wolf", False),
        ("ledoit_wolf", True),
    ]
    for method, clean in profiles:
        corr, info = _estimate_corr_matrix(
            x,
            cov_estimator=method,
            cov_ewma_lambda=0.94,
            rmt_cleaning=clean,
            rmt_cleaning_mode="clip",
            rmt_keep_top_k=0,
        )
        assert corr is not None, info
        assert corr.shape == (20, 20)
        assert np.isfinite(corr).all()
        np.testing.assert_allclose(np.diag(corr), np.ones(20), atol=1e-8)


def test_mode_stability_gate_smoke_on_stable_factor_series() -> None:
    rng = np.random.default_rng(31)
    idx = pd.date_range("2010-01-31", periods=80, freq="ME")
    n_assets = 25
    factor = rng.normal(scale=0.7, size=(80, 1))
    noise = rng.normal(scale=0.3, size=(80, n_assets))
    x = factor @ np.linspace(0.8, 1.2, n_assets)[None, :] + noise
    monthly = pd.DataFrame(x, index=idx, columns=[f"a{i:02d}" for i in range(n_assets)])
    stability_df, _ = dominant_mode_series(monthly, window_months=12, min_assets=20, min_obs_asset=8)
    summary = summarize_mode_stability(stability_df)
    gate = apply_mode_stability_gate(summary, ModeStabilityThresholds())
    assert summary["status"] == "ok"
    assert gate["passed"] is True


def test_mode_stability_gate_detects_instability_when_overlap_is_low() -> None:
    summary = {
        "status": "ok",
        "median_overlap": 0.10,
        "p10_overlap": 0.05,
        "share_overlap_lt_05": 0.90,
        "max_drift": 0.99,
    }
    gate = apply_mode_stability_gate(summary, ModeStabilityThresholds())
    assert gate["passed"] is False
    assert len(gate["reasons"]) >= 1
