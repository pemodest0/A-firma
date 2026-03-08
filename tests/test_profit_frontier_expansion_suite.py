from __future__ import annotations

import pandas as pd

from execution.net_assumptions import NetAssumptionProfile
from scripts.bench.validation.run_profit_frontier_expansion_suite import (
    StrategyResult,
    _build_meta_switch,
    _select_crypto_tiers,
)


def _profile(profile_id: str) -> NetAssumptionProfile:
    return NetAssumptionProfile(
        profile_id=profile_id,
        label=profile_id,
        jurisdiction="test",
        transaction_cost_bps_assumed=0.0,
        fx_spread_bps_assumed=0.0,
        capital_gains_tax_rate=0.0,
        tax_timing="monthly_positive_proxy",
        dividend_withholding_mode="not_applicable",
        monthly_sales_exemption_modeled=False,
        notes=(),
    )


def test_select_crypto_tiers_builds_major_and_mid_buckets() -> None:
    asset_table = pd.DataFrame(
        {
            "ticker": [f"C{i}" for i in range(10)],
            "liquidity_proxy": list(range(10, 0, -1)),
        }
    )
    viability = pd.DataFrame({"ticker": [f"C{i}" for i in range(10)], "days_available": list(range(100, 110))})

    out = _select_crypto_tiers(asset_table, viability)

    assert len(out["crypto_all"]) == 10
    assert len(out["crypto_major8"]) == 8
    assert len(out["crypto_midcap"]) == 2


def test_build_meta_switch_returns_finite_metrics() -> None:
    idx = pd.date_range("2024-01-01", periods=260, freq="B")
    crypto_gross = pd.Series([0.004] * len(idx), index=idx, dtype=float)
    equity_gross = pd.Series([0.001] * len(idx), index=idx, dtype=float)
    zero_turnover = pd.Series([0.0] * len(idx), index=idx, dtype=float)
    zero_bench = pd.Series([0.0] * len(idx), index=idx, dtype=float)
    btc_prices = pd.Series(range(100, 360), index=idx, dtype=float)
    spy_prices = pd.Series(range(100, 360), index=idx, dtype=float)

    crypto = StrategyResult(
        suite="crypto_rule",
        candidate_id="crypto",
        family="crypto",
        benchmark_ticker="BTC-USD",
        gross_ret=crypto_gross,
        turnover=zero_turnover,
        net_ret=crypto_gross,
        benchmark_net_ret=zero_bench,
        net_ann_return=0.0,
        net_total_return=0.0,
        net_sharpe=0.0,
        net_max_drawdown=0.0,
        edge_vs_benchmark=0.0,
        avg_turnover_daily=0.0,
        hit_rate_10x_5y=0.0,
        years_to_10x_full=0.0,
        notes="",
    )
    equity = StrategyResult(
        suite="equities_causal",
        candidate_id="equity",
        family="equity",
        benchmark_ticker="SPY",
        gross_ret=equity_gross,
        turnover=zero_turnover,
        net_ret=equity_gross,
        benchmark_net_ret=zero_bench,
        net_ann_return=0.0,
        net_total_return=0.0,
        net_sharpe=0.0,
        net_max_drawdown=0.0,
        edge_vs_benchmark=0.0,
        avg_turnover_daily=0.0,
        hit_rate_10x_5y=0.0,
        years_to_10x_full=0.0,
        notes="",
    )

    out = _build_meta_switch(
        candidate_id="meta",
        crypto=crypto,
        equities=equity,
        btc_prices=btc_prices,
        spy_prices=spy_prices,
        crypto_profile=_profile("crypto"),
        equity_profile=_profile("equity"),
    )

    assert out.net_ann_return > 0.0
    assert out.net_total_return > 0.0
    assert out.benchmark_ticker == "BTC_SPY_50_50"
