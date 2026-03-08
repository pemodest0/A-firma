from __future__ import annotations

import pandas as pd
import pytest

from scripts.bench.validation.run_profit_attack_validation_suite import build_daily_replay_with_rebalance


def test_build_daily_replay_with_rebalance_uses_explicit_benchmark_series() -> None:
    dates = pd.to_datetime(["2026-01-02", "2026-01-05"])
    returns_wide = pd.DataFrame({"AAA": [0.01, -0.02]}, index=dates)
    monthly = pd.DataFrame(
        [
            {
                "ym": "2026-01",
                "executed_weights_json": "{\"AAA\": 1.0}",
                "cash_weight": 0.0,
                "hedge_weight": 0.0,
            }
        ]
    )
    benchmark = pd.Series([0.004, -0.003], index=dates, dtype=float)

    history = build_daily_replay_with_rebalance(
        monthly_eval=monthly,
        returns_wide=returns_wide,
        benchmark_symbol="SPY",
        benchmark_returns=benchmark,
        initial_capital=1000.0,
        cost_bps=0.0,
        rebalance_frequency="monthly",
    )

    assert history["benchmark_return"].round(6).tolist() == [0.004, -0.003]


def test_build_daily_replay_with_rebalance_raises_when_benchmark_is_missing() -> None:
    dates = pd.to_datetime(["2026-01-02"])
    returns_wide = pd.DataFrame({"AAA": [0.01]}, index=dates)
    monthly = pd.DataFrame(
        [
            {
                "ym": "2026-01",
                "executed_weights_json": "{\"AAA\": 1.0}",
                "cash_weight": 0.0,
                "hedge_weight": 0.0,
            }
        ]
    )

    with pytest.raises(ValueError, match="benchmark symbol SPY"):
        build_daily_replay_with_rebalance(
            monthly_eval=monthly,
            returns_wide=returns_wide,
            benchmark_symbol="SPY",
            initial_capital=1000.0,
            cost_bps=0.0,
            rebalance_frequency="monthly",
        )
