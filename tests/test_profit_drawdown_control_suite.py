from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.bench.validation.run_profit_drawdown_control_suite import (
    _apply_scale_overlay,
    _build_conviction_scale,
    _build_crypto_guard_scale,
    _build_drawdown_guard_scale,
    _build_meta_early_exit_bundle,
    _build_regime_aware_crypto_guard_scale,
    _build_regime_scaled_series,
    _build_source_specific_scale,
    _build_vol_target_scale,
)
from scripts.bench.validation.run_profit_frontier_expansion_suite import StrategyResult
from scripts.bench.validation.run_profit_layered_engine_suite import StrategyBundle
from execution.net_assumptions import NetAssumptionProfile


def _bundle(ret: list[float]) -> StrategyBundle:
    idx = pd.date_range("2024-01-01", periods=len(ret), freq="D")
    series = pd.Series(ret, index=idx, dtype=float)
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
        suite="meta_switch",
        candidate_id="base",
        family="meta_switch",
        benchmark_ticker="B",
        gross_ret=series,
        turnover=pd.Series(np.zeros(len(series), dtype=float), index=idx),
        net_ret=series,
        benchmark_net_ret=pd.Series(np.zeros(len(series), dtype=float), index=idx),
        net_ann_return=0.1,
        net_total_return=0.1,
        net_sharpe=1.0,
        net_max_drawdown=-0.1,
        edge_vs_benchmark=0.05,
        avg_turnover_daily=0.0,
        hit_rate_10x_5y=float("nan"),
        years_to_10x_full=float("nan"),
        notes="",
    )
    return StrategyBundle(result=result, benchmark_gross_ret=pd.Series(np.zeros(len(series), dtype=float), index=idx), profile=profile, benchmark_profile=profile)


def test_build_vol_target_scale_clips_to_range() -> None:
    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    ret = pd.Series([0.02 if i % 2 == 0 else -0.015 for i in range(120)], index=idx, dtype=float)
    scale = _build_vol_target_scale(ret, target_ann_vol=0.2, window=21, min_scale=0.25, max_scale=1.0)
    assert ((scale >= 0.25) & (scale <= 1.0)).all()


def test_build_drawdown_guard_scale_activates_after_threshold() -> None:
    idx = pd.date_range("2024-01-01", periods=8, freq="D")
    ret = pd.Series([0.0, -0.10, -0.08, 0.0, 0.0, 0.05, 0.05, 0.05], index=idx, dtype=float)
    scale = _build_drawdown_guard_scale(ret, trigger_dd=0.12, release_dd=0.05, reduced_scale=0.0, cooldown_days=2)
    assert float(scale.iloc[3]) == 0.0
    assert float(scale.iloc[4]) == 0.0


def test_build_conviction_scale_returns_zero_for_cash_and_bounded_for_active() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    context = pd.DataFrame(
        {
            "source": ["cash", "crypto", "equity"],
            "btc_ok": [False, True, False],
            "spy_ok": [False, True, True],
            "crypto_trail63": [0.0, 0.20, 0.05],
            "equity_trail63": [0.0, 0.05, 0.10],
            "crypto_breadth": [0.0, 0.8, 0.4],
        },
        index=idx,
    )
    scale = _build_conviction_scale(context, min_active=0.25, max_active=1.0)
    assert float(scale.iloc[0]) == 0.0
    assert 0.25 <= float(scale.iloc[1]) <= 1.0
    assert 0.25 <= float(scale.iloc[2]) <= 1.0


def test_apply_scale_overlay_preserves_index_and_reduces_returns() -> None:
    bundle = _bundle([0.10, -0.05, 0.02])
    scale = pd.Series([1.0, 0.5, 0.0], index=bundle.result.gross_ret.index, dtype=float)
    out = _apply_scale_overlay(
        candidate_id="scaled",
        base_bundle=bundle,
        scale=scale,
        suite="drawdown_control",
        family="scale",
        notes="test",
    )
    assert list(out.result.gross_ret.index) == list(bundle.result.gross_ret.index)
    assert float(out.result.gross_ret.iloc[1]) == -0.025
    assert float(out.result.gross_ret.iloc[2]) == 0.0


def test_build_source_specific_scale_maps_by_source() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    context = pd.DataFrame({"source": ["cash", "equity", "crypto"]}, index=idx)
    crypto_scale = pd.Series([0.3, 0.4, 0.5], index=idx, dtype=float)
    out = _build_source_specific_scale(context, crypto_scale=crypto_scale, equity_scale=1.0, cash_scale=0.0)
    assert float(out.iloc[0]) == 0.0
    assert float(out.iloc[1]) == 1.0
    assert float(out.iloc[2]) == 0.5


def test_build_crypto_guard_scale_only_reduces_crypto_leg() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    context = pd.DataFrame(
        {
            "source": ["equity", "crypto", "crypto"],
            "btc_ok": [False, True, True],
            "crypto_breadth": [0.0, 0.40, 0.85],
            "crypto_trail63": [0.0, 0.06, 0.20],
            "equity_trail63": [0.0, 0.04, 0.05],
        },
        index=idx,
    )
    out = _build_crypto_guard_scale(context, low_breadth=0.45, high_breadth=0.75, min_crypto_scale=0.2, edge_floor=0.0, edge_full=0.15)
    assert float(out.iloc[0]) == 1.0
    assert 0.2 <= float(out.iloc[1]) < 1.0
    assert 0.9 <= float(out.iloc[2]) <= 1.0


def test_build_meta_early_exit_bundle_routes_crypto_to_equity_or_cash() -> None:
    bundle = _bundle([0.01, 0.01, 0.01, 0.01])
    idx = bundle.result.gross_ret.index
    context = pd.DataFrame(
        {
            "source": ["crypto", "crypto", "equity", "cash"],
            "btc_ok": [True, True, True, False],
            "spy_ok": [True, True, True, False],
            "crypto_ret": [0.04, -0.10, 0.02, 0.01],
            "equity_ret": [0.01, 0.02, 0.03, 0.00],
            "crypto_trail21": [0.10, -0.05, 0.01, -0.01],
            "equity_trail21": [0.02, 0.01, 0.02, 0.00],
            "crypto_trail63": [0.12, -0.03, 0.04, -0.01],
            "equity_trail63": [0.03, 0.02, 0.03, 0.00],
            "crypto_drawdown_prev": [0.0, -0.20, -0.05, -0.10],
            "crypto_breadth": [0.80, 0.35, 0.60, 0.20],
        },
        index=idx,
    )
    out = _build_meta_early_exit_bundle(
        candidate_id="exit",
        context=context,
        base_bundle=bundle,
        exit_breadth=0.45,
        reentry_breadth=0.60,
        exit_edge21=0.0,
        reentry_edge21=0.03,
        exit_drawdown=0.12,
        exit_mode="equity",
        cooldown_days=1,
    )
    assert float(out.result.gross_ret.iloc[0]) == 0.04
    assert float(out.result.gross_ret.iloc[1]) == 0.02


def test_build_regime_scaled_series_maps_regimes() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    regimes = pd.Series(["stress", "stable"], index=[idx[0], idx[1]], dtype=object)
    out = _build_regime_scaled_series(idx, regimes, {"stress": 0.5, "stable": 1.0}, default=0.8)
    assert list(out.astype(float)) == [0.5, 1.0, 1.0]


def test_build_regime_aware_crypto_guard_scale_changes_by_regime() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    context = pd.DataFrame(
        {
            "source": ["crypto", "crypto"],
            "btc_ok": [True, True],
            "crypto_breadth": [0.58, 0.58],
            "crypto_trail63": [0.12, 0.12],
            "equity_trail63": [0.03, 0.03],
        },
        index=idx,
    )
    regimes = pd.Series(["stress", "stable"], index=idx, dtype=object)
    out = _build_regime_aware_crypto_guard_scale(
        context,
        regimes,
        params_by_regime={
            "stress": {"low_breadth": 0.60, "high_breadth": 0.80, "min_crypto_scale": 0.10, "edge_floor": 0.02, "edge_full": 0.20},
            "stable": {"low_breadth": 0.45, "high_breadth": 0.72, "min_crypto_scale": 0.30, "edge_floor": 0.00, "edge_full": 0.18},
        },
    )
    assert float(out.iloc[0]) < float(out.iloc[1])
