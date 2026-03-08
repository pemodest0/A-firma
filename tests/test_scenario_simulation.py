from __future__ import annotations

import numpy as np
import pandas as pd

from engine.portfolio.scenario_simulation import (
    covariance_cholesky,
    estimate_regime_moments,
    estimate_transition_matrix,
    rolling_regime_conditioned_summary,
    simulate_correlated_paths,
    simulate_regime_conditioned_paths,
    summarize_portfolio_distribution,
)


def test_covariance_cholesky_and_correlated_paths_preserve_shape() -> None:
    cov = np.array([[0.04, 0.018], [0.018, 0.09]], dtype=float)
    chol = covariance_cholesky(cov)
    assert chol.shape == cov.shape
    sims = simulate_correlated_paths(
        mean=np.array([0.001, 0.002], dtype=float),
        cov=cov,
        horizon=20,
        n_paths=2000,
        random_state=11,
    )
    assert sims.shape == (2000, 20, 2)
    flat = sims.reshape(-1, 2)
    sample_corr = np.corrcoef(flat.T)[0, 1]
    target_corr = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    assert abs(sample_corr - target_corr) < 0.08


def test_regime_moments_and_transition_matrix_are_estimated() -> None:
    idx = pd.date_range("2024-01-01", periods=80, freq="B")
    returns = pd.DataFrame(
        {
            "a": np.r_[np.full(40, 0.01), np.full(40, -0.01)],
            "b": np.r_[np.full(40, 0.005), np.full(40, -0.005)],
        },
        index=idx,
    )
    regime = pd.Series(["stable"] * 40 + ["stress"] * 40, index=idx)
    moments = estimate_regime_moments(returns, regime, min_obs=10)
    assert set(moments) == {"stable", "stress"}
    assert moments["stable"].mean[0] > 0
    assert moments["stress"].mean[0] < 0
    states, transition = estimate_transition_matrix(regime, state_order=["stable", "stress"])
    assert states == ["stable", "stress"]
    assert transition.shape == (2, 2)
    assert np.allclose(transition.sum(axis=1), 1.0)


def test_regime_conditioned_simulation_and_summary_work() -> None:
    idx = pd.date_range("2024-01-01", periods=120, freq="B")
    rets = pd.DataFrame(
        {
            "crypto": np.r_[np.full(60, 0.01), np.full(60, -0.02)],
            "equity": np.r_[np.full(60, 0.004), np.full(60, -0.006)],
        },
        index=idx,
    )
    regime = pd.Series(["stable"] * 60 + ["stress"] * 60, index=idx)
    moments = estimate_regime_moments(rets, regime, min_obs=10)
    states, transition = estimate_transition_matrix(regime, state_order=["stable", "stress"])
    sim, path = simulate_regime_conditioned_paths(
        regime_moments=moments,
        transition_matrix=transition,
        states=states,
        start_state="stable",
        horizon=15,
        n_paths=500,
        random_state=3,
    )
    assert sim.shape == (500, 15, 2)
    assert path.shape == (500, 15)
    summary = summarize_portfolio_distribution(sim, np.array([0.6, 0.4], dtype=float))
    assert "terminal_p05" in summary
    weights = pd.DataFrame({"crypto": 0.6, "equity": 0.4}, index=idx)
    rolling = rolling_regime_conditioned_summary(
        rets,
        regime,
        weights,
        lookback=60,
        horizon=10,
        n_paths=200,
        step=10,
        random_state=5,
    )
    assert not rolling.empty
    assert {"terminal_p05", "terminal_p50", "max_drawdown_p95"}.issubset(set(rolling.columns))
