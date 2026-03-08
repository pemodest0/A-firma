from __future__ import annotations

import numpy as np
import pandas as pd

from execution.net_assumptions import NetAssumptionProfile
from scripts.bench.validation.run_profit_equity_improvement_suite import _equity_trailing_switch_bundle, _regime_scale_bundle
from scripts.bench.validation.run_profit_frontier_expansion_suite import StrategyResult
from scripts.bench.validation.run_profit_layered_engine_suite import StrategyBundle


def _bundle(candidate_id: str, values: list[float]) -> StrategyBundle:
    idx = pd.date_range("2024-01-01", periods=len(values), freq="D")
    s = pd.Series(values, index=idx, dtype=float)
    profile = NetAssumptionProfile(
        profile_id="t",
        label="t",
        jurisdiction="test",
        transaction_cost_bps_assumed=0.0,
        fx_spread_bps_assumed=0.0,
        capital_gains_tax_rate=0.0,
        tax_timing="none",
        dividend_withholding_mode="not_applicable",
        monthly_sales_exemption_modeled=False,
        notes=(),
    )
    result = StrategyResult(
        suite="equities",
        candidate_id=candidate_id,
        family="equities",
        benchmark_ticker="SPY",
        gross_ret=s,
        turnover=pd.Series(np.zeros(len(s), dtype=float), index=idx),
        net_ret=s,
        benchmark_net_ret=pd.Series(np.zeros(len(s), dtype=float), index=idx),
        net_ann_return=0.1,
        net_total_return=0.1,
        net_sharpe=1.0,
        net_max_drawdown=-0.1,
        edge_vs_benchmark=0.0,
        avg_turnover_daily=0.0,
        hit_rate_10x_5y=float("nan"),
        years_to_10x_full=float("nan"),
        notes="",
    )
    return StrategyBundle(result=result, benchmark_gross_ret=pd.Series(np.zeros(len(s), dtype=float), index=idx), profile=profile, benchmark_profile=profile)


def test_regime_scale_bundle_reduces_stress_days() -> None:
    bundle = _bundle("base", [0.1, 0.1, 0.1])
    regimes = pd.Series(["stress", "stable", "stable"], index=bundle.result.gross_ret.index, dtype=object)
    out = _regime_scale_bundle(
        candidate_id="scaled",
        bundle=bundle,
        regime_series=regimes,
        mapping={"stress": 0.0, "stable": 1.0},
        notes="",
    )
    assert float(out.result.gross_ret.iloc[0]) == 0.0
    assert float(out.result.gross_ret.iloc[1]) == 0.1


def test_equity_trailing_switch_uses_robust_in_stress() -> None:
    aggressive = _bundle("agg", [0.05, 0.05, 0.05, 0.05])
    robust = _bundle("rob", [0.01, 0.01, 0.01, 0.01])
    idx = aggressive.result.gross_ret.index
    regimes = pd.Series(["stress", "stable", "stable", "stable"], index=idx, dtype=object)
    spy_prices = pd.Series([100.0, 101.0, 102.0, 103.0], index=idx, dtype=float)
    out = _equity_trailing_switch_bundle(
        candidate_id="switch",
        aggressive_bundle=aggressive,
        robust_bundle=robust,
        regime_series=regimes,
        spy_prices=spy_prices,
        mode="regime_switch",
    )
    assert float(out.result.gross_ret.iloc[0]) == 0.0
