from __future__ import annotations

import numpy as np
import pandas as pd

from engine.portfolio.hmm_challenger import build_hmm_feature_frame, fit_hmm_challenger


def test_hmm_challenger_outputs_probabilities_and_labels() -> None:
    idx = pd.date_range("2023-01-01", periods=240, freq="B")
    primary = pd.Series(
        np.r_[np.random.default_rng(7).normal(0.001, 0.01, 120), np.random.default_rng(8).normal(-0.001, 0.02, 120)],
        index=idx,
    )
    secondary = pd.Series(
        np.r_[np.random.default_rng(9).normal(0.0005, 0.008, 120), np.random.default_rng(10).normal(-0.0002, 0.012, 120)],
        index=idx,
    )
    feat = build_hmm_feature_frame(primary_ret=primary, secondary_ret=secondary, volatility_window=21)
    result = fit_hmm_challenger(feat, n_states=3, train_end=pd.Timestamp("2023-10-31"), random_state=13)
    assert not result.states.empty
    assert set(result.regime_label.unique()).issubset({"risk_on", "neutral", "risk_off"})
    assert np.allclose(result.state_probabilities.sum(axis=1).to_numpy(dtype=float), 1.0)
    assert ((result.risk_on_probability >= 0.0) & (result.risk_on_probability <= 1.0)).all()
