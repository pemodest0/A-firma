from __future__ import annotations

import pandas as pd

from scripts.structural.build_historical_structure_assessment import _alert_frame, _build_yearly_performance, _verdict_tag


def test_verdict_tag_rules() -> None:
    assert _verdict_tag(lift=1.3, recall=0.2, months=12) == "forte"
    assert _verdict_tag(lift=1.05, recall=0.1, months=12) == "moderado"
    assert _verdict_tag(lift=0.9, recall=0.2, months=12) == "fraco"
    assert _verdict_tag(lift=1.4, recall=0.3, months=3) == "insuficiente"


def test_build_yearly_performance_picks_best_model() -> None:
    monthly = pd.DataFrame(
        {
            "month": ["2020-01", "2020-02", "2020-01", "2020-02"],
            "month_start": ["2020-01-01", "2020-02-01", "2020-01-01", "2020-02-01"],
            "target": ["drawdown_label", "drawdown_label", "drawdown_label", "drawdown_label"],
            "mode": ["expanding", "expanding", "expanding", "expanding"],
            "model": ["linear", "linear", "regime_only", "regime_only"],
            "f1": [0.30, 0.32, 0.10, 0.10],
            "precision": [0.4, 0.41, 0.2, 0.2],
            "recall": [0.2, 0.21, 0.05, 0.05],
            "lift_precision_vs_random": [1.3, 1.35, 0.9, 0.9],
            "event_rate": [0.2, 0.2, 0.2, 0.2],
            "alert_rate": [0.1, 0.1, 0.1, 0.1],
        }
    )
    yearly, best = _build_yearly_performance(monthly)
    assert not yearly.empty
    assert int(best.shape[0]) == 1
    assert str(best.iloc[0]["model"]) == "linear"


def test_alert_frame_builds_alert_column() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=60, freq="D"),
            "score": [0.1] * 59 + [5.0],
        }
    )
    out = _alert_frame(df, lookback_days=20, alert_budget=0.10, dedupe_days=0)
    assert "alert" in out.columns
    assert bool(out["alert"].iloc[-1]) is True


def test_alert_frame_dedupe_reduces_alert_density() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=120, freq="D"),
            "score": [1.0] * 120,
        }
    )
    out = _alert_frame(df, lookback_days=20, alert_budget=0.50, dedupe_days=20)
    raw = out["raw_alert"].fillna(False).astype(bool).sum()
    ded = out["alert"].fillna(False).astype(bool).sum()
    assert ded <= raw
