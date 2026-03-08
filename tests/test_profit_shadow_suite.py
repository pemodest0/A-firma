from __future__ import annotations

import pandas as pd
import pytest

from scripts.ops.run_profit_shadow_suite import _build_ensemble_monthly_eval, build_daily_replay, summarize_daily_replay


def test_build_daily_replay_uses_monthly_weights_for_each_day() -> None:
    dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-02-03"])
    returns_wide = pd.DataFrame(
        {
            "AAA": [0.01, -0.02, 0.03],
            "BBB": [0.00, 0.01, 0.02],
            "SPY": [0.005, -0.005, 0.01],
        },
        index=dates,
    )
    monthly = pd.DataFrame(
        [
            {
                "ym": "2026-01",
                "executed_weights_json": "{\"AAA\": 0.6, \"BBB\": 0.2}",
                "cash_weight": 0.2,
                "hedge_weight": 0.0,
                "executed_assets": "AAA,BBB",
                "risk_bucket": "stable",
            },
            {
                "ym": "2026-02",
                "executed_weights_json": "{\"BBB\": 1.0}",
                "cash_weight": 0.0,
                "hedge_weight": 0.0,
                "executed_assets": "BBB",
                "risk_bucket": "dispersion",
            },
        ]
    )

    hist = build_daily_replay(
        monthly_eval=monthly,
        returns_wide=returns_wide,
        benchmark_symbol="SPY",
        initial_capital=1000.0,
    )

    assert hist.shape[0] == 3
    assert hist["ym"].tolist() == ["2026-01", "2026-01", "2026-02"]
    assert round(float(hist.iloc[0]["portfolio_return"]), 6) == 0.006
    assert round(float(hist.iloc[1]["portfolio_return"]), 6) == -0.01
    assert round(float(hist.iloc[2]["portfolio_return"]), 6) == 0.02
    assert hist.iloc[2]["risk_bucket"] == "dispersion"


def test_build_daily_replay_uses_explicit_benchmark_series_when_symbol_is_missing() -> None:
    dates = pd.to_datetime(["2026-01-02", "2026-01-05"])
    returns_wide = pd.DataFrame({"AAA": [0.01, -0.02]}, index=dates)
    monthly = pd.DataFrame(
        [
            {
                "ym": "2026-01",
                "executed_weights_json": "{\"AAA\": 1.0}",
                "cash_weight": 0.0,
                "hedge_weight": 0.0,
                "executed_assets": "AAA",
                "risk_bucket": "stable",
            }
        ]
    )
    benchmark = pd.Series([0.003, -0.004], index=dates, dtype=float)

    hist = build_daily_replay(
        monthly_eval=monthly,
        returns_wide=returns_wide,
        benchmark_symbol="SPY",
        benchmark_returns=benchmark,
        initial_capital=1000.0,
    )

    assert hist["benchmark_return"].round(6).tolist() == [0.003, -0.004]


def test_build_daily_replay_raises_when_benchmark_is_missing() -> None:
    dates = pd.to_datetime(["2026-01-02"])
    returns_wide = pd.DataFrame({"AAA": [0.01]}, index=dates)
    monthly = pd.DataFrame(
        [
            {
                "ym": "2026-01",
                "executed_weights_json": "{\"AAA\": 1.0}",
                "cash_weight": 0.0,
                "hedge_weight": 0.0,
                "executed_assets": "AAA",
                "risk_bucket": "stable",
            }
        ]
    )

    with pytest.raises(ValueError, match="benchmark symbol SPY"):
        build_daily_replay(
            monthly_eval=monthly,
            returns_wide=returns_wide,
            benchmark_symbol="SPY",
            initial_capital=1000.0,
        )


def test_summarize_daily_replay_reports_latest_exposures() -> None:
    history = pd.DataFrame(
        [
            {
                "date": "2026-01-02",
                "ym": "2026-01",
                "risk_bucket": "stable",
                "selected_assets": "AAA,BBB",
                "n_assets": 2,
                "cash_weight": 0.2,
                "hedge_weight": 0.0,
                "gross_exposure": 0.8,
                "net_exposure": 0.8,
                "portfolio_return": 0.01,
                "benchmark_return": 0.005,
                "capital": 1010.0,
                "benchmark_capital": 1005.0,
                "capital_peak": 1010.0,
                "drawdown": 0.0,
            },
            {
                "date": "2026-01-05",
                "ym": "2026-01",
                "risk_bucket": "stable",
                "selected_assets": "AAA,BBB",
                "n_assets": 2,
                "cash_weight": 0.2,
                "hedge_weight": -0.3,
                "gross_exposure": 1.1,
                "net_exposure": 0.5,
                "portfolio_return": -0.02,
                "benchmark_return": -0.01,
                "capital": 989.8,
                "benchmark_capital": 994.95,
                "capital_peak": 1010.0,
                "drawdown": -0.02,
            },
        ]
    )

    summary = summarize_daily_replay(history)
    assert summary["status"] == "ok"
    assert summary["latest_risk_bucket"] == "stable"
    assert summary["latest_n_assets"] == 2
    assert summary["latest_gross_exposure"] == 1.1
    assert summary["latest_net_exposure"] == 0.5


def test_build_ensemble_monthly_eval_averages_members_and_votes_weights() -> None:
    monthly_a = pd.DataFrame(
        [
            {
                "ym": "2026-01",
                "ret": 0.02,
                "eqw_ret": 0.01,
                "mkt_ret": 0.015,
                "motor_ret": 0.012,
                "risk_bucket": "stable",
                "cash_weight": 0.2,
                "hedge_weight": 0.0,
                "turnover": 0.1,
                "executed_weights_json": "{\"AAA\": 0.6, \"BBB\": 0.2}",
            }
        ]
    )
    monthly_b = pd.DataFrame(
        [
            {
                "ym": "2026-01",
                "ret": 0.00,
                "eqw_ret": 0.01,
                "mkt_ret": 0.005,
                "motor_ret": 0.006,
                "risk_bucket": "dispersion",
                "cash_weight": 0.1,
                "hedge_weight": 0.0,
                "turnover": 0.2,
                "executed_weights_json": "{\"AAA\": 0.4, \"CCC\": 0.5}",
            }
        ]
    )

    out = _build_ensemble_monthly_eval([monthly_a, monthly_b], vote_threshold=0.5)

    assert out.shape[0] == 1
    assert out.iloc[0]["ym"] == "2026-01"
    assert round(float(out.iloc[0]["ret"]), 6) == 0.01
    assert out.iloc[0]["risk_bucket"] in {"stable", "dispersion"}
    weights = out.iloc[0]["executed_weights_json"]
    assert "AAA" in weights
