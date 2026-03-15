from __future__ import annotations

import pandas as pd

from scripts.bench.validation.run_profit_one_year_payoff_audit import (
    _forward_path_frame,
    _one_year_payoff_row,
)


def test_forward_path_frame_tracks_terminal_max_and_min() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    ret = pd.Series([0.0, 1.0, 1.0, 1.0], index=idx, dtype=float)
    frame = _forward_path_frame(ret, horizon_days=3, monthly_start=False)
    assert frame.shape[0] == 1
    row = frame.iloc[0]
    assert row["terminal_multiple"] == 8.0
    assert row["max_multiple"] == 8.0
    assert row["min_multiple"] == 2.0


def test_one_year_payoff_row_detects_fast_multi_bagger() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    ret = pd.Series([0.0, 1.0, 1.0, 1.0], index=idx, dtype=float)
    row = _one_year_payoff_row(
        scenario="test",
        candidate_id="fast",
        net_returns=ret,
        horizon_days=3,
        monthly_start=False,
    )
    assert row["starts_considered"] == 1
    assert row["hit_rate_5x_252d"] == 1.0
    assert row["hit_rate_6x_252d"] == 1.0
    assert row["touch_loss_50_252d"] == 0.0
    assert row["end_below_50_252d"] == 0.0
    assert row["median_return_252d"] == 7.0


def test_one_year_payoff_row_detects_large_loss() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    ret = pd.Series([0.0, -0.6, 0.0], index=idx, dtype=float)
    row = _one_year_payoff_row(
        scenario="test",
        candidate_id="loss",
        net_returns=ret,
        horizon_days=1,
        monthly_start=False,
    )
    assert row["starts_considered"] == 2
    assert row["touch_loss_50_252d"] == 0.5
    assert row["end_below_50_252d"] == 0.5
    assert row["touch_loss_90_252d"] == 0.0
    assert row["end_below_90_252d"] == 0.0
