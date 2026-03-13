from __future__ import annotations

import numpy as np
import pandas as pd

from engine.portfolio import MetaModeSelectorConfig, run_causal_meta_mode_selector


def _monthly_index() -> pd.DatetimeIndex:
    return pd.date_range("2020-01-31", periods=30, freq="ME")


def test_meta_mode_selector_waits_for_min_training_window() -> None:
    idx = _monthly_index()
    feature_frame = pd.DataFrame(
        {
            "state": np.linspace(0.2, 0.8, len(idx)),
            "stress": np.linspace(0.7, 0.1, len(idx)),
        },
        index=idx,
    )
    candidate_returns = pd.DataFrame(
        {
            "attack": np.where(feature_frame["state"] > 0.5, 0.05, -0.01),
            "protect": np.where(feature_frame["state"] > 0.5, 0.01, 0.02),
        },
        index=idx,
    )
    benchmark = pd.Series(0.01, index=idx, dtype=float)
    out = run_causal_meta_mode_selector(
        feature_frame=feature_frame,
        candidate_returns=candidate_returns,
        benchmark_returns=benchmark,
        config=MetaModeSelectorConfig(training_months=12, min_training_months=12, neighbor_months=6, min_neighbors=4),
    )
    assert out["selected_mode"].iloc[:12].isna().all()
    assert out["selected_mode"].iloc[12:].notna().any()


def test_meta_mode_selector_is_causal_for_past_choices() -> None:
    idx = _monthly_index()
    feature_frame = pd.DataFrame(
        {
            "state": np.r_[np.repeat(0.2, 15), np.repeat(0.8, 15)],
            "stress": np.r_[np.repeat(0.7, 15), np.repeat(0.2, 15)],
        },
        index=idx,
    )
    candidate_returns = pd.DataFrame(
        {
            "attack": np.r_[np.repeat(-0.01, 15), np.repeat(0.06, 15)],
            "protect": np.r_[np.repeat(0.02, 15), np.repeat(0.01, 15)],
        },
        index=idx,
    )
    benchmark = pd.Series(0.01, index=idx, dtype=float)
    cfg = MetaModeSelectorConfig(training_months=18, min_training_months=12, neighbor_months=8, min_neighbors=4)

    base = run_causal_meta_mode_selector(
        feature_frame=feature_frame,
        candidate_returns=candidate_returns,
        benchmark_returns=benchmark,
        config=cfg,
    )

    changed = candidate_returns.copy()
    changed.loc[idx[-4]:, "attack"] = -0.20
    changed.loc[idx[-4]:, "protect"] = 0.20
    altered = run_causal_meta_mode_selector(
        feature_frame=feature_frame,
        candidate_returns=changed,
        benchmark_returns=benchmark,
        config=cfg,
    )

    pd.testing.assert_series_equal(
        base["selected_mode"].iloc[:-4],
        altered["selected_mode"].iloc[:-4],
        check_names=False,
    )


def test_meta_mode_selector_prefers_attack_in_attack_like_months() -> None:
    idx = _monthly_index()
    feature_frame = pd.DataFrame(
        {
            "state": np.r_[np.repeat(0.2, 12), np.repeat(0.85, 18)],
            "stress": np.r_[np.repeat(0.7, 12), np.repeat(0.15, 18)],
            "breadth": np.r_[np.repeat(0.3, 12), np.repeat(0.8, 18)],
        },
        index=idx,
    )
    candidate_returns = pd.DataFrame(
        {
            "attack": np.r_[np.repeat(-0.01, 12), np.repeat(0.05, 18)],
            "protect": np.r_[np.repeat(0.02, 12), np.repeat(0.01, 18)],
            "balanced": np.r_[np.repeat(0.005, 12), np.repeat(0.02, 18)],
        },
        index=idx,
    )
    benchmark = pd.Series(np.r_[np.repeat(0.005, 12), np.repeat(0.015, 18)], index=idx, dtype=float)
    out = run_causal_meta_mode_selector(
        feature_frame=feature_frame,
        candidate_returns=candidate_returns,
        benchmark_returns=benchmark,
        config=MetaModeSelectorConfig(training_months=18, min_training_months=12, neighbor_months=10, min_neighbors=5),
    )
    recent = out.dropna().iloc[-6:]
    assert not recent.empty
    assert (recent["selected_mode"] == "attack").mean() >= 0.5
