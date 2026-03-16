from __future__ import annotations

import pandas as pd

from scripts.bench.validation.run_profit_structural_extraction_audit import (
    _apply_cooldown,
    _binary_tail_event,
    _build_operational_risk,
    _change_points,
    _future_window_metrics,
    _roc_auc,
    _smooth_binary,
    _turn_hit_rate,
)


def test_future_window_metrics_returns_forward_columns() -> None:
    idx = pd.date_range("2025-01-01", periods=12, freq="D")
    returns = pd.Series([0.01, -0.02, 0.01, 0.0, 0.03, -0.01, 0.02, 0.0, -0.01, 0.01, 0.0, 0.02], index=idx)
    out = _future_window_metrics(returns, horizon=5)
    assert {"future_return", "future_max_drawdown", "future_vol"} == set(out.columns)
    assert out.notna().sum().sum() > 0


def test_binary_tail_event_marks_lower_tail() -> None:
    series = pd.Series([0.0, -0.1, -0.2, 0.1, -0.3], dtype=float)
    event = _binary_tail_event(series, tail="lower", quantile=0.2)
    assert int(event.sum()) >= 1
    assert int(event.iloc[-1]) == 1


def test_roc_auc_is_above_random_when_scores_are_ordered() -> None:
    y = pd.Series([0, 0, 1, 1], dtype=int)
    s = pd.Series([0.1, 0.2, 0.8, 0.9], dtype=float)
    auc = _roc_auc(y, s)
    assert auc is not None
    assert auc > 0.9


def test_operational_risk_builds_lagged_binary_series() -> None:
    idx = pd.date_range("2025-01-01", periods=6, freq="D")
    signal = pd.Series([0.1, 0.8, 0.75, 0.2, 0.9, 0.1], index=idx, dtype=float)
    risk = _build_operational_risk(signal, threshold=0.7, min_run=1, cooldown=0)
    assert int(risk.iloc[0]) == 0
    assert int(risk.iloc[2]) == 1
    assert int(risk.iloc[4]) == 0


def test_turn_hit_rate_and_smoothing_are_well_formed() -> None:
    idx = pd.date_range("2025-01-01", periods=8, freq="D")
    proxy = pd.Series([0, 0, 1, 1, 0, 0, 1, 1], index=idx, dtype=int)
    engine = pd.Series([0, 1, 1, 0, 0, 1, 1, 0], index=idx, dtype=int)
    smoothed = _smooth_binary(engine, min_run=2)
    cooled = _apply_cooldown(smoothed, cooldown=1)
    stats = _turn_hit_rate(_change_points(proxy), _change_points(cooled), window_days=1)
    assert 0.0 <= float(stats["hit_rate"]) <= 1.0
    assert 0.0 <= float(stats["false_alarm_rate"]) <= 1.0
