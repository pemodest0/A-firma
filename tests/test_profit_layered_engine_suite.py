from __future__ import annotations

import pandas as pd

from execution.net_assumptions import NetAssumptionProfile
from scripts.bench.validation.run_profit_layered_engine_suite import (
    StrategyBundle,
    _build_breadth_signal,
    _continuous_crypto_weight,
    _fragility_penalty_from_attribution,
    _profile_scaled,
    _robust_objective_row,
    _walkforward_score,
)
from scripts.bench.validation.run_profit_frontier_expansion_suite import StrategyResult


def _profile(profile_id: str) -> NetAssumptionProfile:
    return NetAssumptionProfile(
        profile_id=profile_id,
        label=profile_id,
        jurisdiction="test",
        transaction_cost_bps_assumed=5.0,
        fx_spread_bps_assumed=5.0,
        capital_gains_tax_rate=0.15,
        tax_timing="monthly_positive_proxy",
        dividend_withholding_mode="not_applicable",
        monthly_sales_exemption_modeled=False,
        notes=(),
    )


def test_continuous_crypto_weight_respects_bounds_and_switches_off() -> None:
    assert _continuous_crypto_weight(
        btc_ok=False,
        spy_ok=False,
        crypto_fast=0.2,
        crypto_slow=0.1,
        equity_fast=0.05,
        crypto_vol=0.3,
        crypto_vol_cap=0.5,
        max_crypto_weight=0.85,
    ) == 0.0

    weight = _continuous_crypto_weight(
        btc_ok=True,
        spy_ok=True,
        crypto_fast=0.3,
        crypto_slow=0.2,
        equity_fast=0.05,
        crypto_vol=0.2,
        crypto_vol_cap=0.4,
        max_crypto_weight=0.85,
    )

    assert 0.0 < weight <= 0.85


def test_walkforward_score_is_finite_for_positive_strategy() -> None:
    idx = pd.date_range("2019-01-01", periods=400, freq="B")
    net = pd.Series([0.002] * len(idx), index=idx, dtype=float)
    bench = pd.Series([0.0005] * len(idx), index=idx, dtype=float)
    zero_turnover = pd.Series([0.0] * len(idx), index=idx, dtype=float)
    result = StrategyResult(
        suite="meta_switch_v2",
        candidate_id="meta_v2_test",
        family="meta",
        benchmark_ticker="BTC_SPY_50_50",
        gross_ret=net,
        turnover=zero_turnover,
        net_ret=net,
        benchmark_net_ret=bench,
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
    profile = _profile("base")
    bundle = StrategyBundle(
        result=result,
        benchmark_gross_ret=bench,
        profile=profile,
        benchmark_profile=_profile_scaled(profile, profile_id="bench", label="bench"),
    )

    score = _walkforward_score(bundle, "2019-01-01", "2020-06-01")

    assert score > 0.0


def test_build_breadth_signal_stays_in_unit_interval() -> None:
    idx = pd.date_range("2024-01-01", periods=260, freq="B")
    prices = pd.DataFrame(
        {
            "A": range(100, 360),
            "B": range(80, 340),
            "C": range(120, 380),
        },
        index=idx,
        dtype=float,
    )
    returns = prices.pct_change().fillna(0.0)

    breadth = _build_breadth_signal(
        returns=returns,
        prices=prices,
        tickers=["A", "B", "C"],
        lookback_days=63,
        ma_days=100,
    )

    assert ((breadth >= 0.0) & (breadth <= 1.0)).all()
    assert breadth.iloc[-1] > 0.5


def test_robust_objective_penalizes_fragility() -> None:
    idx = pd.date_range("2019-01-01", periods=260, freq="B")
    net = pd.Series([0.002] * len(idx), index=idx, dtype=float)
    bench = pd.Series([0.0005] * len(idx), index=idx, dtype=float)
    zero_turnover = pd.Series([0.0] * len(idx), index=idx, dtype=float)
    result = StrategyResult(
        suite="meta_switch_v3",
        candidate_id="meta_v3_test",
        family="meta",
        benchmark_ticker="BTC_SPY_50_50",
        gross_ret=net,
        turnover=zero_turnover,
        net_ret=net,
        benchmark_net_ret=bench,
        net_ann_return=0.2,
        net_total_return=0.4,
        net_sharpe=0.8,
        net_max_drawdown=-0.2,
        edge_vs_benchmark=0.3,
        avg_turnover_daily=0.0,
        hit_rate_10x_5y=0.0,
        years_to_10x_full=0.0,
        notes="",
    )
    bundle = StrategyBundle(
        result=result,
        benchmark_gross_ret=bench,
        profile=_profile("meta"),
        benchmark_profile=_profile("bench"),
    )
    stress_df = pd.DataFrame(
        [
            {"candidate_id": "meta_v3_test", "stress_label": "hard_cost", "net_ann_return": 0.15, "edge_vs_benchmark_net_total_return": 0.2},
            {"candidate_id": "meta_v3_test", "stress_label": "delay_d1", "net_ann_return": 0.18, "edge_vs_benchmark_net_total_return": 0.25},
        ]
    )
    wf_df = pd.DataFrame(
        [
            {"candidate_id": "meta_v3_test", "block": "test_2022", "edge_vs_benchmark_net_total_return": 0.1},
            {"candidate_id": "meta_v3_test", "block": "test_2023_2024", "edge_vs_benchmark_net_total_return": 0.05},
            {"candidate_id": "meta_v3_test", "block": "test_2025_now", "edge_vs_benchmark_net_total_return": -0.02},
        ]
    )

    low_frag = _robust_objective_row(bundle=bundle, stress_df=stress_df, wf_df=wf_df, fragility_penalty=0.1)
    high_frag = _robust_objective_row(bundle=bundle, stress_df=stress_df, wf_df=wf_df, fragility_penalty=1.0)

    assert low_frag["robust_score"] > high_frag["robust_score"]


def test_fragility_penalty_grows_when_core_assets_are_removed() -> None:
    attribution_df = pd.DataFrame(
        [
            {"candidate_id": "attr__base", "net_ann_return": 0.30, "edge_vs_benchmark_net_total_return": 0.80},
            {"candidate_id": "attr__no_btc", "net_ann_return": 0.10, "edge_vs_benchmark_net_total_return": 0.20},
            {"candidate_id": "attr__no_sol", "net_ann_return": 0.05, "edge_vs_benchmark_net_total_return": 0.10},
        ]
    )

    penalty = _fragility_penalty_from_attribution(attribution_df)

    assert penalty > 0.0
