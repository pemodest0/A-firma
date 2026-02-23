from __future__ import annotations

import numpy as np
import pandas as pd

from engine.structural.ground_truth import (
    build_event_label,
    build_regime_future_event_label,
    classification_report_binary,
    forward_max_drawdown_from_equity,
    threshold_from_train,
)


def test_forward_max_drawdown_from_equity_basic() -> None:
    eq = pd.Series([1.0, 0.95, 0.9, 0.92, 1.05], dtype=float)
    dd = forward_max_drawdown_from_equity(eq, horizon_days=2)
    assert np.isfinite(dd.iloc[0])
    assert dd.iloc[0] <= -0.0999


def test_build_event_label_marks_drawdown_events() -> None:
    eq = pd.Series([1.0, 0.98, 0.93, 0.96, 1.02], dtype=float)
    y = build_event_label(equity=eq, horizon_days=3, dd_threshold=0.05)
    assert int(y.iloc[0]) == 1


def test_classification_report_binary_values() -> None:
    y_true = [1, 0, 1, 0, 1, 0]
    y_pred = [1, 0, 0, 0, 1, 1]
    rep = classification_report_binary(y_true, y_pred)
    assert rep["n"] == 6.0
    assert rep["tp"] == 2.0
    assert rep["fp"] == 1.0
    assert 0.0 <= rep["precision"] <= 1.0


def test_threshold_from_train_uses_train_slice() -> None:
    score = pd.Series([0.1, 0.2, 0.3, 0.8, 0.9], dtype=float)
    train_mask = pd.Series([True, True, True, False, False])
    thr = threshold_from_train(score, train_mask=train_mask, q=0.8)
    assert 0.1 <= thr <= 0.3


def test_build_regime_future_event_label() -> None:
    regime = pd.Series(["stable", "stable", "transition", "stable", "stress"], dtype=object)
    y = build_regime_future_event_label(regime, horizon_days=2, target_regimes={"stress", "transition"})
    assert int(y.iloc[0]) == 1
    assert int(y.iloc[1]) == 1
