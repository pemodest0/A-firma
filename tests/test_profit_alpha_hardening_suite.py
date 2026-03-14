from __future__ import annotations

import pandas as pd

from execution.net_assumptions import NetAssumptionProfile
from scripts.bench.validation.run_profit_alpha_hardening_suite import (
    _build_alpha_meta_allocation_bundle,
    _build_promoted_attack_confidence_score,
)
from scripts.bench.validation.run_profit_frontier_expansion_suite import StrategyResult
from scripts.bench.validation.run_profit_layered_engine_suite import StrategyBundle


def _profile() -> NetAssumptionProfile:
    return NetAssumptionProfile(
        profile_id="test",
        label="test",
        jurisdiction="test",
        transaction_cost_bps_assumed=0.0,
        fx_spread_bps_assumed=0.0,
        capital_gains_tax_rate=0.0,
        tax_timing="monthly_positive_proxy",
        dividend_withholding_mode="not_applicable",
    )


def _bundle(candidate_id: str, returns: pd.Series) -> StrategyBundle:
    profile = _profile()
    zeros = pd.Series(0.0, index=returns.index, dtype=float)
    result = StrategyResult(
        suite="test",
        candidate_id=candidate_id,
        family="test",
        benchmark_ticker="TEST",
        gross_ret=returns.astype(float),
        turnover=zeros.copy(),
        net_ret=returns.astype(float),
        benchmark_net_ret=zeros.copy(),
        net_ann_return=0.0,
        net_total_return=0.0,
        net_sharpe=0.0,
        net_max_drawdown=0.0,
        edge_vs_benchmark=0.0,
        avg_turnover_daily=0.0,
        hit_rate_10x_5y=0.0,
        years_to_10x_full=float("nan"),
        notes="test",
    )
    return StrategyBundle(
        result=result,
        benchmark_gross_ret=zeros,
        profile=profile,
        benchmark_profile=profile,
    )


def test_attack_confidence_score_extends_past_regime_tail() -> None:
    idx = pd.date_range("2025-07-01", periods=260, freq="D")
    attack_returns = pd.DataFrame(
        {
            "crypto": [0.01 if i % 7 else -0.005 for i in range(len(idx))],
            "equity": [0.004 if i % 5 else -0.002 for i in range(len(idx))],
        },
        index=idx,
        dtype=float,
    )
    context = {
        "btc_prices": pd.Series([100 + i for i in range(len(idx))], index=idx, dtype=float),
        "spy_prices": pd.Series([200 + i for i in range(len(idx))], index=idx, dtype=float),
        "regime_series": pd.Series(
            ["stable", "transition", "stable", "stable"],
            index=idx[:4],
            dtype=object,
        ),
    }

    score = _build_promoted_attack_confidence_score(context, attack_returns)

    assert list(score.index) == list(idx)
    assert pd.notna(score.iloc[-1])


def test_attack_confidence_score_defaults_when_regime_is_empty() -> None:
    idx = pd.date_range("2025-07-01", periods=260, freq="D")
    attack_returns = pd.DataFrame(
        {
            "crypto": [0.008 if i % 6 else -0.004 for i in range(len(idx))],
            "equity": [0.003 if i % 4 else -0.001 for i in range(len(idx))],
        },
        index=idx,
        dtype=float,
    )
    context = {
        "btc_prices": pd.Series([100 + i for i in range(len(idx))], index=idx, dtype=float),
        "spy_prices": pd.Series([200 + i for i in range(len(idx))], index=idx, dtype=float),
        "regime_series": pd.Series(dtype=object),
    }

    score = _build_promoted_attack_confidence_score(context, attack_returns)

    assert list(score.index) == list(idx)
    assert pd.notna(score.iloc[-1])


def test_attack_confidence_score_does_not_backfill_future_regime() -> None:
    idx = pd.date_range("2025-07-01", periods=260, freq="D")
    attack_returns = pd.DataFrame(
        {
            "crypto": [0.006 if i % 9 else -0.003 for i in range(len(idx))],
            "equity": [0.002 if i % 5 else -0.001 for i in range(len(idx))],
        },
        index=idx,
        dtype=float,
    )
    base_context = {
        "btc_prices": pd.Series([100 + i for i in range(len(idx))], index=idx, dtype=float),
        "spy_prices": pd.Series([200 + i for i in range(len(idx))], index=idx, dtype=float),
    }
    score_empty = _build_promoted_attack_confidence_score(
        {**base_context, "regime_series": pd.Series(dtype=object)},
        attack_returns,
    )
    future_stress = pd.Series(["stress"] * 30, index=idx[-30:], dtype=object)
    score_future = _build_promoted_attack_confidence_score(
        {**base_context, "regime_series": future_stress},
        attack_returns,
    )

    early_date = idx[150]
    assert float(score_empty.loc[early_date]) == float(score_future.loc[early_date])


def test_attack_confidence_score_ignores_same_day_return_shock() -> None:
    idx = pd.date_range("2025-01-01", periods=260, freq="D")
    base_returns = pd.DataFrame(
        {
            "crypto": [0.01 if i % 9 else -0.006 for i in range(len(idx))],
            "equity": [0.006 if i % 7 else -0.004 for i in range(len(idx))],
        },
        index=idx,
        dtype=float,
    )
    shocked_returns = base_returns.copy()
    shock_day = idx[200]
    shocked_returns.loc[shock_day, "crypto"] = 0.30
    context = {
        "btc_prices": pd.Series([100 + i for i in range(len(idx))], index=idx, dtype=float),
        "spy_prices": pd.Series([200 + i for i in range(len(idx))], index=idx, dtype=float),
        "regime_series": pd.Series(["stable"] * len(idx), index=idx, dtype=object),
    }

    score_base = _build_promoted_attack_confidence_score(context, base_returns)
    score_shock = _build_promoted_attack_confidence_score(context, shocked_returns)

    assert float(score_base.loc[shock_day]) == float(score_shock.loc[shock_day])
    assert float((score_base - score_shock).abs().loc[idx[201:220]].max()) > 0.0


def test_alpha_meta_bundle_enters_after_not_on_same_day_shock() -> None:
    idx = pd.date_range("2025-01-01", periods=260, freq="D")
    crypto = pd.Series(0.0, index=idx, dtype=float)
    equity = pd.Series(0.0, index=idx, dtype=float)
    shock_day = idx[150]
    crypto.loc[shock_day] = 0.25
    btc = pd.Series([100 + i for i in range(len(idx))], index=idx, dtype=float)
    spy = pd.Series([200 + i for i in range(len(idx))], index=idx, dtype=float)

    bundle = _build_alpha_meta_allocation_bundle(
        candidate_id="test_alpha_meta",
        crypto_bundle=_bundle("crypto", crypto),
        equity_bundle=_bundle("equity", equity),
        btc_prices=btc,
        spy_prices=spy,
        profile=_profile(),
        entry_lookback=14,
        exit_lookback=21,
        entry_margin=0.01,
        exit_margin=0.01,
        risk_off_mode="cash",
        min_crypto_hold_days=0,
    )

    assert float(bundle.weights.loc[shock_day, "crypto"]) == 0.0
    assert float(bundle.weights.loc[idx[151], "crypto"]) == 1.0
