from __future__ import annotations

import pandas as pd

from scripts.bench.validation.run_profit_execution_phase_suite import (
    build_rebalance_mask,
    calendar_year_rows,
    simulate_allocation_execution,
)
from scripts.bench.validation.run_profit_alpha_hardening_suite import AllocationBundle
from scripts.bench.validation.run_profit_frontier_expansion_suite import StrategyResult
from scripts.bench.validation.run_profit_layered_engine_suite import StrategyBundle
from execution.net_assumptions import NetAssumptionProfile


def _dummy_bundle() -> AllocationBundle:
    idx = pd.date_range("2024-01-01", periods=6, freq="B")
    result = StrategyResult(
        suite="test",
        candidate_id="dummy",
        family="test",
        benchmark_ticker="TEST",
        gross_ret=pd.Series([0.0] * len(idx), index=idx, dtype=float),
        turnover=pd.Series([0.0] * len(idx), index=idx, dtype=float),
        net_ret=pd.Series([0.0] * len(idx), index=idx, dtype=float),
        benchmark_net_ret=pd.Series([0.0] * len(idx), index=idx, dtype=float),
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
    profile = NetAssumptionProfile(
        profile_id="dummy",
        label="Dummy",
        jurisdiction="test",
        transaction_cost_bps_assumed=10.0,
        fx_spread_bps_assumed=0.0,
        capital_gains_tax_rate=0.0,
        tax_timing="monthly_positive_proxy",
        dividend_withholding_mode="not_modeled",
    )
    bundle = StrategyBundle(
        result=result,
        benchmark_gross_ret=pd.Series([0.0] * len(idx), index=idx, dtype=float),
        profile=profile,
        benchmark_profile=profile,
    )
    weights = pd.DataFrame(
        {
            "crypto": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "equity": [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            "cash": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        index=idx,
    )
    source = pd.Series(["crypto", "crypto", "equity", "equity", "equity", "equity"], index=idx)
    return AllocationBundle(bundle=bundle, weights=weights, source=source)


def test_build_rebalance_mask_monthly_marks_first_day() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-02-01", "2024-02-02"])
    mask = build_rebalance_mask(idx, "monthly")
    assert mask.tolist() == [True, False, True, False]


def test_simulate_allocation_execution_generates_turnover_and_year_rows() -> None:
    allocation = _dummy_bundle()
    idx = allocation.weights.index
    sleeve_returns = pd.DataFrame(
        {
            "crypto": [0.01, 0.02, 0.00, 0.00, 0.00, 0.00],
            "equity": [0.00, 0.00, 0.01, -0.01, 0.02, 0.01],
        },
        index=idx,
    )
    benchmark = pd.Series([0.0, 0.0, 0.005, 0.0, 0.0, 0.0], index=idx, dtype=float)
    history = simulate_allocation_execution(
        allocation=allocation,
        sleeve_returns=sleeve_returns,
        benchmark_returns=benchmark,
        rebalance_frequency="daily",
        delay_days=0,
        extra_cost_bps=0.0,
        extra_spread_bps=0.0,
        extra_slippage_bps=0.0,
        initial_capital=10000.0,
    )
    assert not history.empty
    assert int(history["operation_day"].sum()) >= 2
    rows = calendar_year_rows(history, candidate_id="dummy", candidate_label="Dummy", scenario="base_daily")
    assert rows
    assert rows[0]["year"] == 2024
