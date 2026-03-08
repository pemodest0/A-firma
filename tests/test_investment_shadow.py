from __future__ import annotations

import pandas as pd

from scripts.ops.run_investment_shadow import build_portfolio_history, next_available_date, summarize_portfolio_history


def test_next_available_date_uses_strictly_later_bar() -> None:
    dates = pd.to_datetime(["2026-03-02", "2026-03-03", "2026-03-04"])
    assert str(next_available_date(dates, "2026-03-02").date()) == "2026-03-03"
    assert next_available_date(dates, "2026-03-04") is None


def test_build_portfolio_history_constant_exposure() -> None:
    idx = pd.to_datetime(["2026-03-02", "2026-03-03", "2026-03-04"])
    risk = pd.Series([0.01, 0.0, 0.02], index=idx, dtype=float)
    defensive = pd.Series([0.0, 0.0, 0.0], index=idx, dtype=float)
    signals = pd.DataFrame(
        [
            {
                "generated_at_utc": "2026-03-01T21:00:00Z",
                "signal_date": "2026-03-01",
                "effective_date": "2026-03-02",
                "target_exposure": 0.5,
                "regime": "stable",
                "lab_run_dir": "x",
                "gate_blocked": False,
            }
        ]
    )

    hist = build_portfolio_history(
        risk_returns=risk,
        defensive_returns=defensive,
        signals=signals,
        initial_capital=1000.0,
        cost_bps=0.0,
        max_daily_turnover=1.0,
        initial_exposure=0.7,
    )

    assert hist.shape[0] == 3
    assert hist["executed_exposure"].round(6).tolist() == [0.5, 0.5, 0.5]
    assert round(float(hist.iloc[0]["capital"]), 6) == 1005.0
    assert round(float(hist.iloc[-1]["capital"]), 6) == 1015.05


def test_build_portfolio_history_caps_turnover() -> None:
    idx = pd.to_datetime(["2026-03-02", "2026-03-03", "2026-03-04"])
    risk = pd.Series([0.0, 0.0, 0.0], index=idx, dtype=float)
    defensive = pd.Series([0.0, 0.0, 0.0], index=idx, dtype=float)
    signals = pd.DataFrame(
        [
            {
                "generated_at_utc": "2026-03-01T21:00:00Z",
                "signal_date": "2026-03-01",
                "effective_date": "2026-03-02",
                "target_exposure": 0.2,
                "regime": "stress",
                "lab_run_dir": "x",
                "gate_blocked": False,
            },
            {
                "generated_at_utc": "2026-03-02T21:00:00Z",
                "signal_date": "2026-03-02",
                "effective_date": "2026-03-03",
                "target_exposure": 1.0,
                "regime": "dispersion",
                "lab_run_dir": "x",
                "gate_blocked": False,
            },
        ]
    )

    hist = build_portfolio_history(
        risk_returns=risk,
        defensive_returns=defensive,
        signals=signals,
        initial_capital=1000.0,
        cost_bps=0.0,
        max_daily_turnover=0.2,
        initial_exposure=0.0,
    )

    assert hist["executed_exposure"].round(6).tolist() == [0.2, 0.4, 0.6]
    assert hist["turnover"].round(6).tolist() == [0.0, 0.2, 0.2]


def test_summarize_portfolio_history_reports_latest_fields() -> None:
    history = pd.DataFrame(
        [
            {
                "date": "2026-03-02",
                "signal_date": "2026-03-01",
                "effective_date": "2026-03-02",
                "regime": "stable",
                "target_exposure": 0.7,
                "executed_exposure": 0.7,
                "turnover": 0.0,
                "cost": 0.0,
                "risk_return": 0.01,
                "defensive_return": 0.0,
                "portfolio_return": 0.007,
                "benchmark_return": 0.01,
                "capital": 1007.0,
                "capital_peak": 1007.0,
                "drawdown": 0.0,
                "benchmark_capital": 1010.0,
            },
            {
                "date": "2026-03-03",
                "signal_date": "2026-03-02",
                "effective_date": "2026-03-03",
                "regime": "dispersion",
                "target_exposure": 0.9,
                "executed_exposure": 0.8,
                "turnover": 0.1,
                "cost": 0.0,
                "risk_return": 0.0,
                "defensive_return": 0.0,
                "portfolio_return": 0.0,
                "benchmark_return": 0.0,
                "capital": 1007.0,
                "capital_peak": 1007.0,
                "drawdown": 0.0,
                "benchmark_capital": 1010.0,
            },
        ]
    )

    summary = summarize_portfolio_history(history)
    assert summary["status"] == "ok"
    assert summary["latest_regime"] == "dispersion"
    assert summary["latest_target_exposure"] == 0.9
    assert summary["latest_executed_exposure"] == 0.8
    assert summary["n_days"] == 2
