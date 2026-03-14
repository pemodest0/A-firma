#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.structural.run_manifest import write_run_manifest  # noqa: E402
from execution.cost_model import summarize_return_series  # noqa: E402
from execution.net_assumptions import NetAssumptionProfile, load_net_assumption_profiles  # noqa: E402
from scripts.ops.official_structural_regime import load_official_structural_regime_series  # noqa: E402
from scripts.bench.validation.run_profit_frontier_expansion_suite import (  # noqa: E402
    EQUITY_EXCLUDED,
    StrategyResult,
    _build_equity_group_map,
    _build_meta_switch,
    _ensure_benchmark_columns,
    _evaluate_net,
    _load_asset_table,
    _load_daily_universe,
    _oos_block_rows,
    _precompute_scores_skip,
    _result_row,
    _rolling_ten_x_stats,
    _run_id,
    _safe_float,
    _select_crypto_tiers,
    _simulate_asset_rule,
    _simulate_equity_group_sleeve,
    _write_json,
)


@dataclass(frozen=True)
class StrategyBundle:
    result: StrategyResult
    benchmark_gross_ret: pd.Series
    profile: NetAssumptionProfile
    benchmark_profile: NetAssumptionProfile


def _load_structural_regime_series_local(root: Path) -> pd.Series:
    regime_series, _meta = load_official_structural_regime_series(root, official_window=120)
    return regime_series


def _regime_forward_fill_local(index: pd.Index, regime_series: pd.Series) -> pd.Series:
    idx = pd.to_datetime(index, errors="coerce")
    if regime_series.empty:
        return pd.Series(["stable"] * len(idx), index=index, dtype=object)
    reg = regime_series.copy()
    reg.index = pd.to_datetime(reg.index, errors="coerce")
    reg = reg[~reg.index.isna()]
    out = reg.reindex(idx.union(reg.index)).sort_index().ffill().reindex(idx)
    return out.fillna("stable").astype(str)


def _simulate_equity_trail_switch_bundle(
    *,
    candidate_id: str,
    aggressive_bundle: StrategyBundle,
    robust_bundle: StrategyBundle,
    regime_series: pd.Series,
    spy_prices: pd.Series,
) -> StrategyBundle:
    idx = aggressive_bundle.result.gross_ret.index.intersection(robust_bundle.result.gross_ret.index).intersection(spy_prices.index)
    agg_ret = pd.to_numeric(aggressive_bundle.result.gross_ret.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    rob_ret = pd.to_numeric(robust_bundle.result.gross_ret.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    benchmark = aggressive_bundle.benchmark_gross_ret.reindex(idx).fillna(0.0).astype(float)
    spy = pd.to_numeric(spy_prices.reindex(idx), errors="coerce").astype(float)
    market_ok = (spy.shift(1) > spy.shift(1).rolling(200, min_periods=100).mean()).fillna(False)
    agg_signal_ret = agg_ret.shift(1).fillna(0.0)
    rob_signal_ret = rob_ret.shift(1).fillna(0.0)
    agg_trail = (1.0 + agg_signal_ret).rolling(63, min_periods=21).apply(np.prod, raw=True) - 1.0
    rob_trail = (1.0 + rob_signal_ret).rolling(63, min_periods=21).apply(np.prod, raw=True) - 1.0
    reg = _regime_forward_fill_local(idx, regime_series)

    gross = pd.Series(np.zeros(len(idx), dtype=float), index=idx, dtype=float)
    turnover = pd.Series(np.zeros(len(idx), dtype=float), index=idx, dtype=float)
    prev_weights: dict[str, float] = {"cash": 1.0}
    for dt in idx:
        regime = str(reg.loc[dt]).lower()
        agg = _safe_float(agg_trail.loc[dt], 0.0)
        rob = _safe_float(rob_trail.loc[dt], 0.0)
        m_ok = bool(market_ok.loc[dt])
        if regime == "stress":
            weights = {"cash": 1.0}
        elif not m_ok:
            weights = {"robust": 0.75, "cash": 0.25}
        elif agg > rob + 0.01:
            weights = {"aggressive": 1.0}
        elif rob > agg - 0.01:
            weights = {"robust": 1.0}
        else:
            weights = {"aggressive": 0.50, "robust": 0.50}
        keys = sorted(set(prev_weights) | set(weights))
        turnover.loc[dt] = 0.5 * float(sum(abs(float(prev_weights.get(k, 0.0)) - float(weights.get(k, 0.0))) for k in keys))
        prev_weights = dict(weights)
        gross.loc[dt] = float(weights.get("aggressive", 0.0)) * float(agg_ret.loc[dt]) + float(weights.get("robust", 0.0)) * float(rob_ret.loc[dt])

    perf = _evaluate_net(
        gross_ret=gross,
        turnover=turnover,
        profile=aggressive_bundle.profile,
        benchmark_ret=benchmark,
        benchmark_profile=aggressive_bundle.benchmark_profile,
    )
    hit5 = _rolling_ten_x_stats(perf["net_ret"], horizon_days=1260)
    wealth = (1.0 + perf["net_ret"]).cumprod()
    hit_full = wealth[wealth >= 10.0]
    years_to_10x = float((hit_full.index[0] - wealth.index[0]).days / 365.25) if not hit_full.empty else float("nan")
    result = StrategyResult(
        suite="equities_meta",
        candidate_id=candidate_id,
        family="equities_meta",
        benchmark_ticker=aggressive_bundle.result.benchmark_ticker,
        gross_ret=gross,
        turnover=turnover,
        net_ret=perf["net_ret"],
        benchmark_net_ret=perf["benchmark_net_ret"],
        net_ann_return=_safe_float(perf.get("net_ann_return")),
        net_total_return=_safe_float(perf.get("net_total_return")),
        net_sharpe=_safe_float(perf.get("net_sharpe")),
        net_max_drawdown=_safe_float(perf.get("net_max_drawdown")),
        edge_vs_benchmark=_safe_float(perf.get("edge_vs_benchmark")),
        avg_turnover_daily=_safe_float(perf.get("avg_turnover_daily")),
        hit_rate_10x_5y=_safe_float(hit5.get("hit_rate")),
        years_to_10x_full=years_to_10x,
        notes=f"mode=trail_switch;agg={aggressive_bundle.result.candidate_id};rob={robust_bundle.result.candidate_id}",
    )
    return StrategyBundle(
        result=result,
        benchmark_gross_ret=benchmark,
        profile=aggressive_bundle.profile,
        benchmark_profile=aggressive_bundle.benchmark_profile,
    )


def _profile_scaled(
    profile: NetAssumptionProfile,
    *,
    profile_id: str,
    label: str,
    transaction_cost_bps: float | None = None,
    fx_spread_bps: float | None = None,
    capital_gains_tax_rate: float | None = None,
    tax_timing: str | None = None,
) -> NetAssumptionProfile:
    return NetAssumptionProfile(
        profile_id=profile_id,
        label=label,
        jurisdiction=profile.jurisdiction,
        transaction_cost_bps_assumed=float(transaction_cost_bps if transaction_cost_bps is not None else profile.transaction_cost_bps_assumed),
        fx_spread_bps_assumed=float(fx_spread_bps if fx_spread_bps is not None else profile.fx_spread_bps_assumed),
        capital_gains_tax_rate=float(capital_gains_tax_rate if capital_gains_tax_rate is not None else profile.capital_gains_tax_rate),
        tax_timing=str(tax_timing if tax_timing is not None else profile.tax_timing),
        dividend_withholding_mode=profile.dividend_withholding_mode,
        monthly_sales_exemption_modeled=profile.monthly_sales_exemption_modeled,
        monthly_sales_exemption_brl=profile.monthly_sales_exemption_brl,
        capital_gains_brackets=profile.capital_gains_brackets,
        loss_compensation_enabled=profile.loss_compensation_enabled,
        withholding_bps_on_sales=profile.withholding_bps_on_sales,
        withholding_compensates_tax=profile.withholding_compensates_tax,
        assumed_portfolio_base_brl=profile.assumed_portfolio_base_brl,
        sell_turnover_fraction_proxy=profile.sell_turnover_fraction_proxy,
        cash_yield_enabled=profile.cash_yield_enabled,
        cash_rate_source_path=profile.cash_rate_source_path,
        cash_rate_annual_fallback=profile.cash_rate_annual_fallback,
        notes=tuple(profile.notes) + ("stress_profile",),
    )


def _rolling_total(ret: pd.Series, window: int, *, min_periods: int) -> pd.Series:
    return (1.0 + pd.to_numeric(ret, errors="coerce").fillna(0.0).astype(float)).rolling(window, min_periods=min_periods).apply(np.prod, raw=True) - 1.0


def _rolling_pos_rate(ret: pd.Series, window: int, *, min_periods: int) -> pd.Series:
    x = (pd.to_numeric(ret, errors="coerce").fillna(0.0).astype(float) > 0.0).astype(float)
    return x.rolling(window, min_periods=min_periods).mean()


def _rolling_corr(a: pd.Series, b: pd.Series, window: int, *, min_periods: int) -> pd.Series:
    ax = pd.to_numeric(a, errors="coerce").fillna(0.0).astype(float)
    bx = pd.to_numeric(b, errors="coerce").fillna(0.0).astype(float)
    return ax.rolling(window, min_periods=min_periods).corr(bx).replace([np.inf, -np.inf], np.nan)


def _stress_bundle(
    bundle: StrategyBundle,
    *,
    delay_days: int,
    profile: NetAssumptionProfile,
    benchmark_profile: NetAssumptionProfile | None = None,
    label: str,
) -> dict[str, Any]:
    gross = pd.to_numeric(bundle.result.gross_ret, errors="coerce").fillna(0.0).astype(float)
    turnover = pd.to_numeric(bundle.result.turnover, errors="coerce").fillna(0.0).astype(float)
    if int(delay_days) > 0:
        gross = gross.shift(int(delay_days)).fillna(0.0).astype(float)
        turnover = turnover.shift(int(delay_days)).fillna(0.0).astype(float)
    perf = _evaluate_net(
        gross_ret=gross,
        turnover=turnover,
        profile=profile,
        benchmark_ret=bundle.benchmark_gross_ret.reindex(gross.index).fillna(0.0).astype(float),
        benchmark_profile=benchmark_profile or bundle.benchmark_profile,
    )
    return {
        "candidate_id": bundle.result.candidate_id,
        "suite": bundle.result.suite,
        "stress_label": label,
        "delay_days": int(delay_days),
        "net_ann_return": _safe_float(perf.get("net_ann_return")),
        "net_total_return": _safe_float(perf.get("net_total_return")),
        "net_sharpe": _safe_float(perf.get("net_sharpe")),
        "net_max_drawdown": _safe_float(perf.get("net_max_drawdown")),
        "edge_vs_benchmark_net_total_return": _safe_float(perf.get("edge_vs_benchmark")),
        "avg_turnover_daily": _safe_float(perf.get("avg_turnover_daily")),
    }


def _continuous_crypto_weight(
    *,
    btc_ok: bool,
    spy_ok: bool,
    crypto_fast: float,
    crypto_slow: float,
    equity_fast: float,
    crypto_vol: float,
    crypto_vol_cap: float,
    max_crypto_weight: float,
) -> float:
    if not btc_ok and not spy_ok:
        return 0.0
    score = 0.0
    if btc_ok:
        score += 0.35
    if np.isfinite(crypto_fast) and np.isfinite(equity_fast) and crypto_fast > equity_fast:
        score += 0.25
    if np.isfinite(crypto_slow) and crypto_slow > 0.0:
        score += 0.20
    if np.isfinite(crypto_vol_cap) and np.isfinite(crypto_vol) and crypto_vol <= crypto_vol_cap:
        score += 0.15
    if spy_ok and np.isfinite(equity_fast) and equity_fast > 0.0:
        score -= 0.10
    return float(np.clip(score, 0.0, float(max_crypto_weight)))


def _build_breadth_signal(
    *,
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    tickers: list[str],
    lookback_days: int,
    ma_days: int,
) -> pd.Series:
    available = [ticker for ticker in tickers if ticker in returns.columns and ticker in prices.columns]
    if not available:
        return pd.Series(dtype=float)
    ret_block = returns[available].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    px_block = prices[available].apply(pd.to_numeric, errors="coerce").astype(float)
    mom = px_block.shift(1) / px_block.shift(1 + int(lookback_days)) - 1.0
    ma = px_block.shift(1).rolling(int(ma_days), min_periods=max(30, int(ma_days) // 2)).mean()
    breadth = 0.55 * (mom > 0.0).mean(axis=1, skipna=True) + 0.45 * (px_block.shift(1) > ma).mean(axis=1, skipna=True)
    return pd.to_numeric(breadth, errors="coerce").fillna(0.0).clip(0.0, 1.0).astype(float)


def _apply_breadth_overlay_to_bundle(
    *,
    candidate_id: str,
    bundle: StrategyBundle,
    breadth_signal: pd.Series,
    low_threshold: float,
    high_threshold: float,
    mode: str,
) -> StrategyBundle:
    idx = bundle.result.gross_ret.index.intersection(breadth_signal.index)
    gross = pd.to_numeric(bundle.result.gross_ret.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    turnover = pd.to_numeric(bundle.result.turnover.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    breadth = pd.to_numeric(breadth_signal.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    lo = float(low_threshold)
    hi = float(max(high_threshold, low_threshold + 1e-6))
    if mode == "gate":
        scale = (breadth >= hi).astype(float)
    else:
        scale = ((breadth - lo) / (hi - lo)).clip(0.0, 1.0)
    scale = scale.astype(float)
    gross = gross * scale
    scale_turn = scale.diff().abs().fillna(scale.abs()).astype(float) * 0.5
    turnover = turnover * scale + scale_turn
    perf = _evaluate_net(
        gross_ret=gross,
        turnover=turnover,
        profile=bundle.profile,
        benchmark_ret=bundle.benchmark_gross_ret.reindex(idx).fillna(0.0).astype(float),
        benchmark_profile=bundle.benchmark_profile,
    )
    hit5 = _rolling_ten_x_stats(perf["net_ret"], horizon_days=1260)
    wealth = (1.0 + perf["net_ret"]).cumprod()
    hit_full = wealth[wealth >= 10.0]
    years_to_10x = float((hit_full.index[0] - wealth.index[0]).days / 365.25) if not hit_full.empty else float("nan")
    result = StrategyResult(
        suite="crypto_breadth",
        candidate_id=candidate_id,
        family="crypto_breadth",
        benchmark_ticker=bundle.result.benchmark_ticker,
        gross_ret=gross,
        turnover=turnover,
        net_ret=perf["net_ret"],
        benchmark_net_ret=perf["benchmark_net_ret"],
        net_ann_return=_safe_float(perf.get("net_ann_return")),
        net_total_return=_safe_float(perf.get("net_total_return")),
        net_sharpe=_safe_float(perf.get("net_sharpe")),
        net_max_drawdown=_safe_float(perf.get("net_max_drawdown")),
        edge_vs_benchmark=_safe_float(perf.get("edge_vs_benchmark")),
        avg_turnover_daily=_safe_float(perf.get("avg_turnover_daily")),
        hit_rate_10x_5y=_safe_float(hit5.get("hit_rate")),
        years_to_10x_full=years_to_10x,
        notes=f"breadth_mode={mode};low={lo:.2f};high={hi:.2f};base={bundle.result.candidate_id}",
    )
    return StrategyBundle(
        result=result,
        benchmark_gross_ret=bundle.benchmark_gross_ret.reindex(idx).fillna(0.0).astype(float),
        profile=bundle.profile,
        benchmark_profile=bundle.benchmark_profile,
    )


def _capped_group_weights(weights: pd.Series, *, max_weight: float) -> pd.Series:
    base = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0).astype(float)
    if base.empty or float(base.sum()) <= 0.0:
        return pd.Series(dtype=float)
    capped = (base / float(base.sum())).clip(upper=float(max_weight))
    if float(capped.sum()) <= 0.0:
        return pd.Series(dtype=float)
    return (capped / float(capped.sum())).astype(float)


def _build_meta_switch_v2(
    *,
    candidate_id: str,
    crypto_bundle: StrategyBundle,
    equity_bundle: StrategyBundle,
    btc_prices: pd.Series,
    spy_prices: pd.Series,
    mode: str,
    fast_window: int,
    slow_window: int,
    vol_window: int,
    vol_quantile: float,
    max_crypto_weight: float,
) -> StrategyBundle:
    idx = crypto_bundle.result.gross_ret.index.intersection(equity_bundle.result.gross_ret.index)
    crypto_ret = crypto_bundle.result.gross_ret.reindex(idx).fillna(0.0).astype(float)
    equity_ret = equity_bundle.result.gross_ret.reindex(idx).fillna(0.0).astype(float)
    btc_close = pd.to_numeric(btc_prices.reindex(idx), errors="coerce").astype(float)
    spy_close = pd.to_numeric(spy_prices.reindex(idx), errors="coerce").astype(float)
    btc_ok = (btc_close.shift(1) > btc_close.shift(1).rolling(200, min_periods=100).mean()).fillna(False)
    spy_ok = (spy_close.shift(1) > spy_close.shift(1).rolling(200, min_periods=100).mean()).fillna(False)
    crypto_fast = _rolling_total(crypto_ret, int(fast_window), min_periods=max(10, fast_window // 2))
    crypto_slow = _rolling_total(crypto_ret, int(slow_window), min_periods=max(20, slow_window // 2))
    equity_fast = _rolling_total(equity_ret, int(fast_window), min_periods=max(10, fast_window // 2))
    crypto_vol = crypto_ret.rolling(int(vol_window), min_periods=max(20, vol_window // 2)).std(ddof=0) * np.sqrt(252.0)
    crypto_vol_cap = crypto_vol.rolling(252, min_periods=126).quantile(float(vol_quantile))
    crypto_vol_cap = crypto_vol_cap.fillna(crypto_vol.expanding(min_periods=20).median())

    gross = pd.Series(np.zeros(len(idx), dtype=float), index=idx, dtype=float)
    turnover = pd.Series(np.zeros(len(idx), dtype=float), index=idx, dtype=float)
    prev_weights = {"cash": 1.0}

    for dt in idx:
        c_ok = bool(btc_ok.loc[dt])
        e_ok = bool(spy_ok.loc[dt])
        c_fast = _safe_float(crypto_fast.loc[dt], 0.0)
        c_slow = _safe_float(crypto_slow.loc[dt], 0.0)
        e_fast = _safe_float(equity_fast.loc[dt], 0.0)
        c_vol = _safe_float(crypto_vol.loc[dt], float("inf"))
        c_cap = _safe_float(crypto_vol_cap.loc[dt], float("inf"))

        if mode == "discrete":
            if not c_ok and not e_ok:
                weights = {"cash": 1.0}
                gross.loc[dt] = 0.0
            elif c_ok and c_fast > e_fast and c_slow > 0.0 and c_vol <= c_cap:
                weights = {"crypto": 1.0}
                gross.loc[dt] = float(crypto_ret.loc[dt])
            elif e_ok:
                weights = {"equity": 1.0}
                gross.loc[dt] = float(equity_ret.loc[dt])
            elif c_ok and c_fast > 0.0:
                weights = {"crypto": 0.5, "cash": 0.5}
                gross.loc[dt] = 0.5 * float(crypto_ret.loc[dt])
            else:
                weights = {"cash": 1.0}
                gross.loc[dt] = 0.0
        else:
            cw = _continuous_crypto_weight(
                btc_ok=c_ok,
                spy_ok=e_ok,
                crypto_fast=c_fast,
                crypto_slow=c_slow,
                equity_fast=e_fast,
                crypto_vol=c_vol,
                crypto_vol_cap=c_cap,
                max_crypto_weight=float(max_crypto_weight),
            )
            if not c_ok and not e_ok:
                ew = 0.0
            elif e_ok:
                ew = max(0.0, 1.0 - cw)
            else:
                ew = 0.0
            cash = max(0.0, 1.0 - cw - ew)
            weights = {}
            if cw > 0.0:
                weights["crypto"] = cw
            if ew > 0.0:
                weights["equity"] = ew
            if cash > 0.0:
                weights["cash"] = cash
            gross.loc[dt] = cw * float(crypto_ret.loc[dt]) + ew * float(equity_ret.loc[dt])

        keys = sorted(set(prev_weights) | set(weights))
        turnover.loc[dt] = 0.5 * float(sum(abs(float(prev_weights.get(k, 0.0)) - float(weights.get(k, 0.0))) for k in keys))
        prev_weights = dict(weights)

    btc_bench = pd.to_numeric(btc_close.pct_change(), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    spy_bench = pd.to_numeric(spy_close.pct_change(), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    benchmark_gross = 0.5 * btc_bench + 0.5 * spy_bench
    blended_profile = NetAssumptionProfile(
        profile_id=f"meta_blend_{mode}",
        label=f"Meta blend {mode}",
        jurisdiction="blended",
        transaction_cost_bps_assumed=0.5 * crypto_bundle.profile.transaction_cost_bps_assumed + 0.5 * equity_bundle.profile.transaction_cost_bps_assumed,
        fx_spread_bps_assumed=0.5 * crypto_bundle.profile.fx_spread_bps_assumed + 0.5 * equity_bundle.profile.fx_spread_bps_assumed,
        capital_gains_tax_rate=0.5 * crypto_bundle.profile.capital_gains_tax_rate + 0.5 * equity_bundle.profile.capital_gains_tax_rate,
        tax_timing="monthly_positive_proxy",
        dividend_withholding_mode="not_applicable",
        monthly_sales_exemption_modeled=False,
        notes=(f"mode={mode}", f"fast={fast_window}", f"slow={slow_window}", f"vol_q={vol_quantile}"),
    )
    perf = _evaluate_net(
        gross_ret=gross,
        turnover=turnover,
        profile=blended_profile,
        benchmark_ret=benchmark_gross,
        benchmark_profile=blended_profile,
    )
    hit5 = _rolling_ten_x_stats(perf["net_ret"], horizon_days=1260)
    wealth = (1.0 + perf["net_ret"]).cumprod()
    hit_full = wealth[wealth >= 10.0]
    years_to_10x = float((hit_full.index[0] - wealth.index[0]).days / 365.25) if not hit_full.empty else float("nan")
    result = StrategyResult(
        suite="meta_switch_v2",
        candidate_id=candidate_id,
        family=f"meta_{mode}",
        benchmark_ticker="BTC_SPY_50_50",
        gross_ret=gross,
        turnover=turnover,
        net_ret=perf["net_ret"],
        benchmark_net_ret=perf["benchmark_net_ret"],
        net_ann_return=_safe_float(perf["net_ann_return"]),
        net_total_return=_safe_float(perf["net_total_return"]),
        net_sharpe=_safe_float(perf["net_sharpe"]),
        net_max_drawdown=_safe_float(perf["net_max_drawdown"]),
        edge_vs_benchmark=_safe_float(perf["edge_vs_benchmark"]),
        avg_turnover_daily=_safe_float(perf["avg_turnover_daily"]),
        hit_rate_10x_5y=_safe_float(hit5.get("hit_rate")),
        years_to_10x_full=years_to_10x,
        notes=f"mode={mode};fast={fast_window};slow={slow_window};vol_q={vol_quantile};max_crypto={max_crypto_weight}",
    )
    return StrategyBundle(result=result, benchmark_gross_ret=benchmark_gross, profile=blended_profile, benchmark_profile=blended_profile)


def _meta_blended_profile(crypto_profile: NetAssumptionProfile, equity_profile: NetAssumptionProfile, *, profile_id: str, label: str) -> NetAssumptionProfile:
    return NetAssumptionProfile(
        profile_id=profile_id,
        label=label,
        jurisdiction="blended",
        transaction_cost_bps_assumed=0.5 * crypto_profile.transaction_cost_bps_assumed + 0.5 * equity_profile.transaction_cost_bps_assumed,
        fx_spread_bps_assumed=0.5 * crypto_profile.fx_spread_bps_assumed + 0.5 * equity_profile.fx_spread_bps_assumed,
        capital_gains_tax_rate=0.5 * crypto_profile.capital_gains_tax_rate + 0.5 * equity_profile.capital_gains_tax_rate,
        tax_timing="monthly_realistic_proxy",
        dividend_withholding_mode="not_applicable",
        monthly_sales_exemption_modeled=bool(
            crypto_profile.monthly_sales_exemption_modeled or equity_profile.monthly_sales_exemption_modeled
        ),
        monthly_sales_exemption_brl=0.5 * crypto_profile.monthly_sales_exemption_brl + 0.5 * equity_profile.monthly_sales_exemption_brl,
        capital_gains_brackets=tuple(crypto_profile.capital_gains_brackets or equity_profile.capital_gains_brackets),
        loss_compensation_enabled=bool(crypto_profile.loss_compensation_enabled or equity_profile.loss_compensation_enabled),
        withholding_bps_on_sales=0.5 * crypto_profile.withholding_bps_on_sales + 0.5 * equity_profile.withholding_bps_on_sales,
        withholding_compensates_tax=bool(
            crypto_profile.withholding_compensates_tax or equity_profile.withholding_compensates_tax
        ),
        assumed_portfolio_base_brl=0.5 * crypto_profile.assumed_portfolio_base_brl + 0.5 * equity_profile.assumed_portfolio_base_brl,
        sell_turnover_fraction_proxy=0.5 * crypto_profile.sell_turnover_fraction_proxy + 0.5 * equity_profile.sell_turnover_fraction_proxy,
        cash_yield_enabled=bool(crypto_profile.cash_yield_enabled or equity_profile.cash_yield_enabled),
        cash_rate_source_path=str(crypto_profile.cash_rate_source_path or equity_profile.cash_rate_source_path),
        cash_rate_annual_fallback=0.5 * crypto_profile.cash_rate_annual_fallback + 0.5 * equity_profile.cash_rate_annual_fallback,
        notes=("meta_blended", "monthly_realistic_proxy"),
    )


def _build_meta_switch_v3(
    *,
    candidate_id: str,
    crypto_bundle: StrategyBundle,
    equity_bundle: StrategyBundle,
    btc_prices: pd.Series,
    spy_prices: pd.Series,
    breadth_signal: pd.Series,
    fast_window: int,
    slow_window: int,
    entry_breadth: float,
    exit_breadth: float,
    max_crypto_weight: float,
    cash_floor: float,
) -> StrategyBundle:
    idx = crypto_bundle.result.gross_ret.index.intersection(equity_bundle.result.gross_ret.index).intersection(breadth_signal.index)
    crypto_ret = crypto_bundle.result.gross_ret.reindex(idx).fillna(0.0).astype(float)
    equity_ret = equity_bundle.result.gross_ret.reindex(idx).fillna(0.0).astype(float)
    breadth = pd.to_numeric(breadth_signal.reindex(idx), errors="coerce").fillna(0.0).clip(0.0, 1.0).astype(float)
    btc_close = pd.to_numeric(btc_prices.reindex(idx), errors="coerce").astype(float)
    spy_close = pd.to_numeric(spy_prices.reindex(idx), errors="coerce").astype(float)
    btc_ok = (btc_close.shift(1) > btc_close.shift(1).rolling(200, min_periods=100).mean()).fillna(False)
    spy_ok = (spy_close.shift(1) > spy_close.shift(1).rolling(200, min_periods=100).mean()).fillna(False)
    crypto_signal_ret = crypto_ret.shift(1).fillna(0.0)
    equity_signal_ret = equity_ret.shift(1).fillna(0.0)
    crypto_fast = _rolling_total(crypto_signal_ret, int(fast_window), min_periods=max(10, fast_window // 2))
    crypto_slow = _rolling_total(crypto_signal_ret, int(slow_window), min_periods=max(20, slow_window // 2))
    equity_fast = _rolling_total(equity_signal_ret, int(fast_window), min_periods=max(10, fast_window // 2))
    equity_slow = _rolling_total(equity_signal_ret, int(slow_window), min_periods=max(20, slow_window // 2))

    gross = pd.Series(np.zeros(len(idx), dtype=float), index=idx, dtype=float)
    turnover = pd.Series(np.zeros(len(idx), dtype=float), index=idx, dtype=float)
    prev_weights = {"cash": 1.0}
    avg_crypto_weight = 0.0

    for dt in idx:
        c_ok = bool(btc_ok.loc[dt])
        e_ok = bool(spy_ok.loc[dt])
        breadth_now = _safe_float(breadth.loc[dt], 0.0)
        c_fast = _safe_float(crypto_fast.loc[dt], 0.0)
        c_slow = _safe_float(crypto_slow.loc[dt], 0.0)
        e_fast = _safe_float(equity_fast.loc[dt], 0.0)
        e_slow = _safe_float(equity_slow.loc[dt], 0.0)
        prev_cw = float(prev_weights.get("crypto", 0.0))

        severe_off = (not c_ok and not e_ok) or breadth_now < float(exit_breadth) * 0.75
        if severe_off:
            cw = 0.0
            ew = 0.0
        else:
            entry_ok = c_ok and breadth_now >= float(entry_breadth) and c_fast > e_fast and c_slow > 0.0
            stay_ok = c_ok and breadth_now >= float(exit_breadth) and c_fast > -0.04
            if entry_ok:
                signal = 0.45 + 0.35 * np.clip((breadth_now - float(entry_breadth)) / max(1e-6, 1.0 - float(entry_breadth)), 0.0, 1.0)
                signal += 0.20 if c_slow > e_slow else 0.0
                cw = float(np.clip(signal, 0.0, float(max_crypto_weight)))
            elif prev_cw > 0.0 and stay_ok:
                cw = float(np.clip(max(prev_cw * 0.75, 0.25 * float(max_crypto_weight)), 0.0, float(max_crypto_weight)))
            else:
                cw = 0.0

            if e_ok:
                eq_budget = 1.0 - float(cash_floor)
                eq_signal = 0.35 + 0.35 * np.clip(max(e_fast, 0.0), 0.0, 0.20) / 0.20
                if breadth_now >= float(entry_breadth) and c_ok and c_fast > e_fast:
                    eq_signal *= 0.6
                ew = max(0.0, min(eq_budget - cw, eq_signal))
            else:
                ew = 0.0

        cash = max(float(cash_floor), 1.0 - cw - ew) if (cw > 0.0 or ew > 0.0) else 1.0
        scale = 1.0 / max(cw + ew + cash, 1e-9)
        weights = {}
        if cw > 0.0:
            weights["crypto"] = cw * scale
        if ew > 0.0:
            weights["equity"] = ew * scale
        if cash > 0.0:
            weights["cash"] = cash * scale
        keys = sorted(set(prev_weights) | set(weights))
        turnover.loc[dt] = 0.5 * float(sum(abs(float(prev_weights.get(k, 0.0)) - float(weights.get(k, 0.0))) for k in keys))
        prev_weights = dict(weights)
        avg_crypto_weight += float(weights.get("crypto", 0.0))
        gross.loc[dt] = float(weights.get("crypto", 0.0)) * float(crypto_ret.loc[dt]) + float(weights.get("equity", 0.0)) * float(equity_ret.loc[dt])

    btc_bench = pd.to_numeric(btc_close.pct_change(), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    spy_bench = pd.to_numeric(spy_close.pct_change(), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    benchmark_gross = 0.5 * btc_bench + 0.5 * spy_bench
    blended_profile = _meta_blended_profile(crypto_bundle.profile, equity_bundle.profile, profile_id=f"{candidate_id}_profile", label=f"{candidate_id} profile")
    perf = _evaluate_net(
        gross_ret=gross,
        turnover=turnover,
        profile=blended_profile,
        benchmark_ret=benchmark_gross,
        benchmark_profile=blended_profile,
    )
    hit5 = _rolling_ten_x_stats(perf["net_ret"], horizon_days=1260)
    wealth = (1.0 + perf["net_ret"]).cumprod()
    hit_full = wealth[wealth >= 10.0]
    years_to_10x = float((hit_full.index[0] - wealth.index[0]).days / 365.25) if not hit_full.empty else float("nan")
    avg_crypto_weight = avg_crypto_weight / float(max(len(idx), 1))
    result = StrategyResult(
        suite="meta_switch_v3",
        candidate_id=candidate_id,
        family="meta_v3",
        benchmark_ticker="BTC_SPY_50_50",
        gross_ret=gross,
        turnover=turnover,
        net_ret=perf["net_ret"],
        benchmark_net_ret=perf["benchmark_net_ret"],
        net_ann_return=_safe_float(perf["net_ann_return"]),
        net_total_return=_safe_float(perf["net_total_return"]),
        net_sharpe=_safe_float(perf["net_sharpe"]),
        net_max_drawdown=_safe_float(perf["net_max_drawdown"]),
        edge_vs_benchmark=_safe_float(perf["edge_vs_benchmark"]),
        avg_turnover_daily=_safe_float(perf["avg_turnover_daily"]),
        hit_rate_10x_5y=_safe_float(hit5.get("hit_rate")),
        years_to_10x_full=years_to_10x,
        notes=(
            f"fast={fast_window};slow={slow_window};entry_breadth={entry_breadth:.2f};"
            f"exit_breadth={exit_breadth:.2f};max_crypto={max_crypto_weight:.2f};cash_floor={cash_floor:.2f};"
            f"avg_crypto_weight={avg_crypto_weight:.3f}"
        ),
    )
    return StrategyBundle(result=result, benchmark_gross_ret=benchmark_gross, profile=blended_profile, benchmark_profile=blended_profile)


def _equity_v2_group_scores(
    *,
    fast_row: pd.Series,
    slow_row: pd.Series,
    persist_row: pd.Series,
    corr_row: pd.Series,
) -> pd.Series:
    df = pd.concat(
        [
            pd.to_numeric(fast_row, errors="coerce").rename("fast"),
            pd.to_numeric(slow_row, errors="coerce").rename("slow"),
            pd.to_numeric(persist_row, errors="coerce").rename("persist"),
            pd.to_numeric(corr_row, errors="coerce").rename("corr_spy"),
        ],
        axis=1,
    )
    df = df.dropna(subset=["fast", "slow"]).copy()
    if df.empty:
        return pd.Series(dtype=float)
    corr_pen = pd.to_numeric(df["corr_spy"], errors="coerce").fillna(df["corr_spy"].median())
    score = 0.45 * df["slow"] + 0.25 * df["fast"] + 0.20 * (df["persist"] - 0.5) - 0.10 * corr_pen
    return score.sort_values(ascending=False)


def _simulate_equity_group_sleeve_v2(
    *,
    candidate_id: str,
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    asset_table: pd.DataFrame,
    equity_groups: dict[str, list[str]],
    benchmark_ticker: str,
    group_lookback_fast: int,
    group_lookback_slow: int,
    group_top_k: int,
    assets_per_group: int,
    asset_lookback: int,
    asset_ma_days: int,
    market_ma_days: int,
    profile: NetAssumptionProfile,
    benchmark_profile: NetAssumptionProfile,
) -> StrategyBundle | None:
    all_tickers = sorted({ticker for tickers in equity_groups.values() for ticker in tickers if ticker in returns.columns})
    if not all_tickers:
        return None
    score_map, asset_ma_filters, benchmark_filters = _precompute_scores_skip(
        returns[all_tickers + ([benchmark_ticker] if benchmark_ticker in returns.columns and benchmark_ticker not in all_tickers else [])],
        prices[all_tickers + ([benchmark_ticker] if benchmark_ticker in prices.columns and benchmark_ticker not in all_tickers else [])],
        lookbacks=[int(asset_lookback), int(group_lookback_fast)],
        asset_ma_days_list=[0, int(asset_ma_days), int(market_ma_days)],
        benchmark_ticker=benchmark_ticker,
        skip_recent_days=0,
    )
    asset_scores = score_map[(int(asset_lookback), "mom_vol_adj")]
    asset_fast_scores = score_map[(int(group_lookback_fast), "mom_total")]
    group_returns = {
        group: returns[tickers].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True).fillna(0.0).astype(float)
        for group, tickers in equity_groups.items()
    }
    group_ret_df = pd.concat(group_returns, axis=1).sort_index().fillna(0.0)
    spy_ret = pd.to_numeric(returns.get(benchmark_ticker), errors="coerce").fillna(0.0).astype(float)
    market_ok = benchmark_filters[int(market_ma_days)].reindex(returns.index).fillna(False)
    group_fast_minp = max(10, int(group_lookback_fast) // 2)
    group_slow_minp = max(20, int(group_lookback_slow) // 2)
    group_fast_totals = (1.0 + group_ret_df).rolling(int(group_lookback_fast), min_periods=group_fast_minp).apply(np.prod, raw=True) - 1.0
    group_slow_totals = (1.0 + group_ret_df).rolling(int(group_lookback_slow), min_periods=group_slow_minp).apply(np.prod, raw=True) - 1.0
    group_persist = (group_ret_df > 0.0).astype(float).rolling(int(group_lookback_fast), min_periods=group_fast_minp).mean()
    group_corr = group_ret_df.rolling(int(group_lookback_slow), min_periods=group_slow_minp).corr(spy_ret).replace([np.inf, -np.inf], np.nan)
    asset_persist = (
        returns[all_tickers].apply(pd.to_numeric, errors="coerce").fillna(0.0).gt(0.0).astype(float).rolling(int(group_lookback_fast), min_periods=group_fast_minp).mean()
    )
    asset_ma_mask = asset_ma_filters[int(asset_ma_days)].reindex(index=returns.index, columns=asset_scores.columns).fillna(False)
    liquidity_map = (
        asset_table.drop_duplicates(subset=["ticker"], keep="first")
        .set_index("ticker")
        .get("liquidity_proxy", pd.Series(dtype=float))
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .astype(float)
    )

    warmup = max(int(group_lookback_slow), int(asset_lookback), int(asset_ma_days), int(market_ma_days)) + 2
    rebalance_positions = list(range(int(max(1, warmup)), returns.shape[0], 21))
    if not rebalance_positions:
        return None

    daily_ret = np.zeros(returns.shape[0], dtype=float)
    daily_turnover = np.zeros(returns.shape[0], dtype=float)
    prev_weights: dict[str, float] = {"CASH": 1.0}
    for pos_idx, pos in enumerate(rebalance_positions):
        next_pos = rebalance_positions[pos_idx + 1] if pos_idx + 1 < len(rebalance_positions) else returns.shape[0]
        if int(market_ma_days) > 0 and not bool(market_ok.iloc[pos]):
            weights = {"CASH": 1.0}
            daily_turnover[pos] += 0.5 * float(sum(abs(float(prev_weights.get(k, 0.0)) - float(weights.get(k, 0.0))) for k in sorted(set(prev_weights) | set(weights))))
            prev_weights = weights
            continue
        group_scores = _equity_v2_group_scores(
            fast_row=group_fast_totals.iloc[pos],
            slow_row=group_slow_totals.iloc[pos],
            persist_row=group_persist.iloc[pos],
            corr_row=group_corr.iloc[pos],
        )
        chosen_groups = group_scores.head(int(max(1, group_top_k))).index.astype(str).tolist()
        chosen_tickers: list[str] = []
        for group in chosen_groups:
            eligible = [ticker for ticker in equity_groups.get(group, []) if ticker in asset_scores.columns]
            if not eligible:
                continue
            base = pd.to_numeric(asset_scores.loc[asset_scores.index[pos], eligible], errors="coerce").dropna().astype(float)
            fast = pd.to_numeric(asset_fast_scores.loc[asset_fast_scores.index[pos], eligible], errors="coerce").dropna().astype(float)
            persist = pd.to_numeric(asset_persist.iloc[pos].reindex(eligible), errors="coerce").astype(float)
            merged = pd.concat([base.rename("base"), fast.rename("fast"), persist.rename("persist")], axis=1)
            merged = merged.dropna(subset=["base", "fast"]).copy()
            if merged.empty:
                continue
            if asset_ma_days > 0:
                ma_ok = asset_ma_mask.iloc[pos]
                merged = merged[ma_ok.reindex(merged.index).fillna(False)]
            if merged.empty:
                continue
            liq = liquidity_map.reindex(merged.index).fillna(float(liquidity_map.median()) if not liquidity_map.empty else 0.0)
            liq_rank = liq.rank(method="average", pct=True).fillna(0.5)
            merged["liq"] = liq_rank
            merged["score"] = 0.55 * merged["base"] + 0.20 * merged["fast"] + 0.15 * (merged["persist"].fillna(0.5) - 0.5) + 0.10 * merged["liq"]
            chosen_tickers.extend(merged.sort_values("score", ascending=False).head(int(max(1, assets_per_group))).index.astype(str).tolist())
        chosen_tickers = sorted(set(chosen_tickers))
        if not chosen_tickers:
            weights = {"CASH": 1.0}
            daily_turnover[pos] += 0.5 * float(sum(abs(float(prev_weights.get(k, 0.0)) - float(weights.get(k, 0.0))) for k in sorted(set(prev_weights) | set(weights))))
            prev_weights = weights
            continue
        group_weight_map = group_scores.head(int(max(1, group_top_k))).clip(lower=0.0)
        if group_weight_map.sum() <= 0:
            group_weight_map = pd.Series(np.ones(len(chosen_groups), dtype=float), index=chosen_groups)
        group_weight_map = group_weight_map / float(group_weight_map.sum())
        weights: dict[str, float] = {}
        for group in chosen_groups:
            local = [ticker for ticker in chosen_tickers if ticker in equity_groups.get(group, [])]
            if not local:
                continue
            per = float(group_weight_map.get(group, 0.0)) / float(len(local))
            for ticker in local:
                weights[ticker] = weights.get(ticker, 0.0) + per
        total = float(sum(weights.values()))
        if total <= 0.0:
            weights = {"CASH": 1.0}
        else:
            weights = {k: float(v / total) for k, v in weights.items()}
        daily_turnover[pos] += 0.5 * float(sum(abs(float(prev_weights.get(k, 0.0)) - float(weights.get(k, 0.0))) for k in sorted(set(prev_weights) | set(weights))))
        prev_weights = dict(weights)
        if "CASH" in weights:
            continue
        block = returns.loc[returns.index[pos:next_pos], list(weights.keys())].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        wvec = np.array([weights[t] for t in block.columns.astype(str)], dtype=float)
        daily_ret[pos:next_pos] = block.to_numpy(dtype=float) @ wvec

    gross = pd.Series(daily_ret, index=returns.index, dtype=float)
    turnover = pd.Series(daily_turnover, index=returns.index, dtype=float)
    benchmark_gross = pd.to_numeric(returns.get(benchmark_ticker), errors="coerce").reindex(returns.index).fillna(0.0).astype(float)
    perf = _evaluate_net(
        gross_ret=gross,
        turnover=turnover,
        profile=profile,
        benchmark_ret=benchmark_gross,
        benchmark_profile=benchmark_profile,
    )
    result = StrategyResult(
        suite="equities_causal_v2",
        candidate_id=candidate_id,
        family="equities_causal_v2",
        benchmark_ticker=benchmark_ticker,
        gross_ret=gross,
        turnover=turnover,
        net_ret=perf["net_ret"],
        benchmark_net_ret=perf["benchmark_net_ret"],
        net_ann_return=_safe_float(perf["net_ann_return"]),
        net_total_return=_safe_float(perf["net_total_return"]),
        net_sharpe=_safe_float(perf["net_sharpe"]),
        net_max_drawdown=_safe_float(perf["net_max_drawdown"]),
        edge_vs_benchmark=_safe_float(perf["edge_vs_benchmark"]),
        avg_turnover_daily=_safe_float(perf["avg_turnover_daily"]),
        hit_rate_10x_5y=float("nan"),
        years_to_10x_full=float("nan"),
        notes=f"group_fast={group_lookback_fast};group_slow={group_lookback_slow};group_top_k={group_top_k};assets_per_group={assets_per_group}",
    )
    return StrategyBundle(result=result, benchmark_gross_ret=benchmark_gross, profile=profile, benchmark_profile=benchmark_profile)


def _simulate_equity_group_sleeve_v3(
    *,
    candidate_id: str,
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    asset_table: pd.DataFrame,
    equity_groups: dict[str, list[str]],
    benchmark_ticker: str,
    group_lookback_fast: int,
    group_lookback_slow: int,
    group_top_k: int,
    assets_per_group: int,
    asset_lookback: int,
    asset_ma_days: int,
    market_ma_days: int,
    min_group_breadth: float,
    max_group_weight: float,
    profile: NetAssumptionProfile,
    benchmark_profile: NetAssumptionProfile,
) -> StrategyBundle | None:
    base = _simulate_equity_group_sleeve_v2(
        candidate_id=f"{candidate_id}__base",
        returns=returns,
        prices=prices,
        asset_table=asset_table,
        equity_groups=equity_groups,
        benchmark_ticker=benchmark_ticker,
        group_lookback_fast=group_lookback_fast,
        group_lookback_slow=group_lookback_slow,
        group_top_k=group_top_k,
        assets_per_group=assets_per_group,
        asset_lookback=asset_lookback,
        asset_ma_days=asset_ma_days,
        market_ma_days=market_ma_days,
        profile=profile,
        benchmark_profile=benchmark_profile,
    )
    if base is None:
        return None

    group_returns = {
        group: returns[tickers].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True).fillna(0.0).astype(float)
        for group, tickers in equity_groups.items()
    }
    group_ret_df = pd.concat(group_returns, axis=1).sort_index().fillna(0.0)
    breadth_fast = (1.0 + group_ret_df).rolling(int(group_lookback_fast), min_periods=max(10, int(group_lookback_fast) // 2)).apply(np.prod, raw=True) - 1.0
    breadth_ratio = (breadth_fast > 0.0).mean(axis=1, skipna=True).fillna(0.0).astype(float)

    idx = base.result.gross_ret.index
    gross = pd.to_numeric(base.result.gross_ret.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    turnover = pd.to_numeric(base.result.turnover.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    base_breadth = breadth_ratio.reindex(idx).fillna(0.0).astype(float)
    scale = ((base_breadth - float(min_group_breadth)) / max(1e-6, 1.0 - float(min_group_breadth))).clip(0.0, 1.0)
    concentration_penalty = ((1.0 - base_breadth) / max(1e-6, 1.0 - float(max_group_weight))).clip(0.0, 1.0)
    adjusted_scale = (scale * (1.0 - 0.35 * concentration_penalty)).clip(0.0, 1.0)
    gross = gross * adjusted_scale
    turnover = turnover * adjusted_scale + adjusted_scale.diff().abs().fillna(adjusted_scale.abs()) * 0.25

    benchmark_gross = pd.to_numeric(base.benchmark_gross_ret.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    perf = _evaluate_net(
        gross_ret=gross,
        turnover=turnover,
        profile=profile,
        benchmark_ret=benchmark_gross,
        benchmark_profile=benchmark_profile,
    )
    result = StrategyResult(
        suite="equities_causal_v3",
        candidate_id=candidate_id,
        family="equities_causal_v3",
        benchmark_ticker=benchmark_ticker,
        gross_ret=gross,
        turnover=turnover,
        net_ret=perf["net_ret"],
        benchmark_net_ret=perf["benchmark_net_ret"],
        net_ann_return=_safe_float(perf["net_ann_return"]),
        net_total_return=_safe_float(perf["net_total_return"]),
        net_sharpe=_safe_float(perf["net_sharpe"]),
        net_max_drawdown=_safe_float(perf["net_max_drawdown"]),
        edge_vs_benchmark=_safe_float(perf["edge_vs_benchmark"]),
        avg_turnover_daily=_safe_float(perf["avg_turnover_daily"]),
        hit_rate_10x_5y=float("nan"),
        years_to_10x_full=float("nan"),
        notes=(
            f"group_fast={group_lookback_fast};group_slow={group_lookback_slow};group_top_k={group_top_k};"
            f"assets_per_group={assets_per_group};min_breadth={min_group_breadth:.2f};max_group_weight={max_group_weight:.2f}"
        ),
    )
    return StrategyBundle(result=result, benchmark_gross_ret=benchmark_gross, profile=profile, benchmark_profile=benchmark_profile)


def _fragility_penalty_from_attribution(attribution_df: pd.DataFrame) -> float:
    if attribution_df.empty or "candidate_id" not in attribution_df.columns:
        return 0.0
    base = attribution_df[attribution_df["candidate_id"].astype(str) == "attr__base"]
    if base.empty:
        return 0.0
    base_ann = _safe_float(base.iloc[0].get("net_ann_return"), 0.0)
    base_edge = _safe_float(base.iloc[0].get("edge_vs_benchmark_net_total_return"), 0.0)
    comp = attribution_df[attribution_df["candidate_id"].astype(str) != "attr__base"].copy()
    if comp.empty:
        return 0.0
    ann_loss = (-pd.to_numeric(comp["net_ann_return"], errors="coerce").fillna(base_ann) + float(base_ann)).clip(lower=0.0)
    edge_loss = (-pd.to_numeric(comp["edge_vs_benchmark_net_total_return"], errors="coerce").fillna(base_edge) + float(base_edge)).clip(lower=0.0)
    ann_norm = ann_loss / max(abs(float(base_ann)), 0.05)
    edge_norm = edge_loss / max(abs(float(base_edge)), 0.25)
    return float(np.clip(0.5 * ann_norm.mean() + 0.5 * edge_norm.mean(), 0.0, 2.0))


def _robust_objective_row(
    *,
    bundle: StrategyBundle,
    stress_df: pd.DataFrame,
    wf_df: pd.DataFrame,
    fragility_penalty: float,
) -> dict[str, Any]:
    stress_sub = stress_df[stress_df["candidate_id"].astype(str) == str(bundle.result.candidate_id)].copy()
    wf_sub = wf_df[wf_df["candidate_id"].astype(str) == str(bundle.result.candidate_id)].copy()
    test_sub = wf_sub[wf_sub["block"].astype(str).str.startswith("test_")].copy()

    hard_cost_ann = _safe_float(
        stress_sub.loc[stress_sub["stress_label"].astype(str) == "hard_cost", "net_ann_return"].iloc[0] if not stress_sub.empty and (stress_sub["stress_label"].astype(str) == "hard_cost").any() else float("nan"),
        float("nan"),
    )
    delay_ann = _safe_float(
        stress_sub.loc[stress_sub["stress_label"].astype(str) == "delay_d1", "net_ann_return"].iloc[0] if not stress_sub.empty and (stress_sub["stress_label"].astype(str) == "delay_d1").any() else float("nan"),
        float("nan"),
    )
    hard_cost_edge = _safe_float(
        stress_sub.loc[stress_sub["stress_label"].astype(str) == "hard_cost", "edge_vs_benchmark_net_total_return"].iloc[0] if not stress_sub.empty and (stress_sub["stress_label"].astype(str) == "hard_cost").any() else float("nan"),
        float("nan"),
    )

    mean_test_edge = float(pd.to_numeric(test_sub.get("edge_vs_benchmark_net_total_return"), errors="coerce").dropna().mean()) if not test_sub.empty else float("nan")
    worst_test_edge = float(pd.to_numeric(test_sub.get("edge_vs_benchmark_net_total_return"), errors="coerce").dropna().min()) if not test_sub.empty else float("nan")
    positive_test_share = float((pd.to_numeric(test_sub.get("edge_vs_benchmark_net_total_return"), errors="coerce") > 0.0).mean()) if not test_sub.empty else 0.0
    base_ann = float(bundle.result.net_ann_return)
    ann_retention = float(hard_cost_ann / max(base_ann, 1e-6)) if np.isfinite(hard_cost_ann) and base_ann > 0.0 else 0.0
    delay_retention = float(delay_ann / max(base_ann, 1e-6)) if np.isfinite(delay_ann) and base_ann > 0.0 else 0.0
    drawdown_penalty = max(0.0, abs(float(bundle.result.net_max_drawdown)) - 0.60)
    worst_edge_penalty = max(0.0, -_safe_float(worst_test_edge, 0.0))

    robust_score = (
        0.35 * float(bundle.result.net_ann_return)
        + 0.20 * float(bundle.result.net_sharpe)
        + 0.15 * _safe_float(mean_test_edge, 0.0)
        + 0.10 * positive_test_share
        + 0.10 * max(0.0, ann_retention)
        + 0.05 * max(0.0, delay_retention)
        + 0.05 * max(0.0, _safe_float(hard_cost_edge, 0.0))
        - 0.15 * float(fragility_penalty)
        - 0.10 * drawdown_penalty
        - 0.10 * worst_edge_penalty
    )
    return {
        "candidate_id": bundle.result.candidate_id,
        "suite": bundle.result.suite,
        "robust_score": robust_score,
        "net_ann_return": bundle.result.net_ann_return,
        "net_sharpe": bundle.result.net_sharpe,
        "net_max_drawdown": bundle.result.net_max_drawdown,
        "edge_vs_benchmark_net_total_return": bundle.result.edge_vs_benchmark,
        "mean_test_edge": mean_test_edge,
        "worst_test_edge": worst_test_edge,
        "positive_test_share": positive_test_share,
        "hard_cost_ann_return": hard_cost_ann,
        "hard_cost_edge": hard_cost_edge,
        "hard_cost_retention": ann_retention,
        "delay_d1_retention": delay_retention,
        "fragility_penalty": fragility_penalty,
    }


def _walkforward_score(bundle: StrategyBundle, start: str, end: str) -> float:
    idx = bundle.result.net_ret.index
    mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
    ret = bundle.result.net_ret.loc[mask]
    bench = bundle.result.benchmark_net_ret.loc[ret.index]
    if ret.empty:
        return float("-inf")
    perf = summarize_return_series(ret, periods_per_year=252)
    bench_perf = summarize_return_series(bench, periods_per_year=252)
    ann = _safe_float(perf.get("annualized_return"), -1.0)
    sharpe = _safe_float(perf.get("sharpe"), -2.0)
    mdd = _safe_float(perf.get("max_drawdown"), -1.0)
    edge = _safe_float(perf.get("total_return"), -1.0) - _safe_float(bench_perf.get("total_return"), -1.0)
    return float(0.55 * edge + 0.25 * ann + 0.15 * sharpe + 0.05 * (1.0 + mdd))


def _walkforward_rows(bundle: StrategyBundle, blocks: list[tuple[str, str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, start, end in blocks:
        idx = bundle.result.net_ret.index
        mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
        ret = bundle.result.net_ret.loc[mask]
        bench = bundle.result.benchmark_net_ret.loc[ret.index]
        if ret.empty:
            continue
        perf = summarize_return_series(ret, periods_per_year=252)
        bench_perf = summarize_return_series(bench, periods_per_year=252)
        rows.append(
            {
                "candidate_id": bundle.result.candidate_id,
                "block": label,
                "start": start,
                "end": end,
                "net_ann_return": _safe_float(perf.get("annualized_return")),
                "net_total_return": _safe_float(perf.get("total_return")),
                "net_sharpe": _safe_float(perf.get("sharpe")),
                "net_max_drawdown": _safe_float(perf.get("max_drawdown")),
                "benchmark_total_return": _safe_float(bench_perf.get("total_return")),
                "edge_vs_benchmark_net_total_return": _safe_float(perf.get("total_return")) - _safe_float(bench_perf.get("total_return")),
            }
        )
    return rows


def _research_rows(
    results: list[StrategyResult],
    *,
    outdir: Path,
    summary_path: Path,
    status_map: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        status = (status_map or {}).get(
            result.candidate_id,
            "keep" if result.edge_vs_benchmark > 0.0 and result.net_ann_return > 0.10 else ("watch" if result.edge_vs_benchmark > 0.0 else "kill"),
        )
        rows.append(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "candidate_id": result.candidate_id,
                "label": result.candidate_id,
                "methodology": f"layered_{result.suite}",
                "status": status,
                "gross_ann_return": result.net_ann_return,
                "net_ann_return": result.net_ann_return,
                "gross_total_return": result.net_total_return,
                "net_total_return": result.net_total_return,
                "sharpe": result.net_sharpe,
                "max_drawdown": result.net_max_drawdown,
                "benchmark_ticker": result.benchmark_ticker,
                "edge_vs_benchmark_net_total_return": result.edge_vs_benchmark,
                "avg_foreign_share": 1.0,
                "groups": result.family,
                "artifacts": {
                    "suite_dir": str(outdir),
                    "summary_json": str(summary_path),
                },
                "notes": result.notes,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Suite layered do Eigen Engine: regime + ranking + execucao + meta-switch.")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--net-assumptions", default="config/profit_net_assumptions.json")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--outdir-root", default="results/validation/profit_layered_engine_suite")
    args = ap.parse_args()

    outdir = (ROOT / args.outdir_root / _run_id()).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    prices_dir = (ROOT / args.prices_dir).resolve()

    profiles = load_net_assumption_profiles((ROOT / args.net_assumptions).resolve())
    foreign_profile = profiles["profiles"]["foreign_financial_brazil_resident"]
    crypto_profile = profiles["profiles"]["crypto_global_brazil_resident_conservative"]
    crypto_hard_profile = _profile_scaled(
        crypto_profile,
        profile_id="crypto_hard",
        label="Crypto hard frictions",
        transaction_cost_bps=45.0,
        fx_spread_bps=45.0,
        capital_gains_tax_rate=0.15,
        tax_timing="monthly_positive_proxy",
    )
    foreign_hard_profile = _profile_scaled(
        foreign_profile,
        profile_id="foreign_hard",
        label="Foreign hard frictions",
        transaction_cost_bps=20.0,
        fx_spread_bps=45.0,
        capital_gains_tax_rate=0.15,
        tax_timing="annual_positive_proxy",
    )

    crypto_assets = _load_asset_table((ROOT / args.crypto_asset_groups).resolve(), (ROOT / args.crypto_asset_metadata).resolve())
    crypto_returns, crypto_prices, crypto_viability = _load_daily_universe(
        prices_dir=prices_dir,
        asset_table=crypto_assets,
        min_history_days=600,
        max_abs_daily_return=1.5,
    )
    crypto_returns, crypto_prices = _ensure_benchmark_columns(crypto_returns, crypto_prices, prices_dir, [str(args.benchmark_crypto), "ETH-USD"])
    crypto_tiers = _select_crypto_tiers(crypto_assets, crypto_viability)
    crypto_breadth_all = _build_breadth_signal(
        returns=crypto_returns,
        prices=crypto_prices,
        tickers=crypto_tiers["crypto_all"],
        lookback_days=63,
        ma_days=200,
    )
    crypto_breadth_major = _build_breadth_signal(
        returns=crypto_returns,
        prices=crypto_prices,
        tickers=crypto_tiers["crypto_major8"],
        lookback_days=63,
        ma_days=200,
    )

    equity_assets = _load_asset_table((ROOT / args.equity_asset_groups).resolve(), (ROOT / args.equity_asset_metadata).resolve())
    equity_assets = equity_assets[~equity_assets["asset_group"].astype(str).isin(EQUITY_EXCLUDED)].copy()
    equity_returns, equity_prices, _ = _load_daily_universe(
        prices_dir=prices_dir,
        asset_table=equity_assets,
        min_history_days=1200,
        max_abs_daily_return=0.8,
    )
    equity_returns, equity_prices = _ensure_benchmark_columns(equity_returns, equity_prices, prices_dir, [str(args.benchmark_equity)])
    equity_group_map = _build_equity_group_map(equity_assets, equity_returns)

    crypto_candidates: list[StrategyBundle] = []
    crypto_specs = [
        ("crypto_all__momvol21_base", crypto_tiers["crypto_all"], dict(score_mode="mom_vol_adj", lookback_days=21, rebalance_days=7, top_k=3, asset_ma_days=0, market_ma_days=200, relative_to_benchmark=False, skip_recent_days=0, trailing_stop_dd=None, hard_stop_loss=None)),
        ("crypto_all__momvol21_hard15", crypto_tiers["crypto_all"], dict(score_mode="mom_vol_adj", lookback_days=21, rebalance_days=7, top_k=3, asset_ma_days=0, market_ma_days=200, relative_to_benchmark=False, skip_recent_days=0, trailing_stop_dd=None, hard_stop_loss=0.15)),
        ("crypto_all__slowrel252", crypto_tiers["crypto_all"], dict(score_mode="mom_total", lookback_days=252, rebalance_days=7, top_k=2, asset_ma_days=200, market_ma_days=200, relative_to_benchmark=True, skip_recent_days=0, trailing_stop_dd=None, hard_stop_loss=None)),
        ("crypto_major8__momvol21", crypto_tiers["crypto_major8"], dict(score_mode="mom_vol_adj", lookback_days=21, rebalance_days=7, top_k=3, asset_ma_days=0, market_ma_days=200, relative_to_benchmark=False, skip_recent_days=0, trailing_stop_dd=None, hard_stop_loss=None)),
    ]
    for candidate_id, tickers, kwargs in crypto_specs:
        result = _simulate_asset_rule(
            candidate_id=candidate_id,
            family="crypto",
            allowed_tickers=tickers,
            returns=crypto_returns,
            prices=crypto_prices,
            asset_table=crypto_assets,
            benchmark_ticker=str(args.benchmark_crypto),
            fallback_ticker=str(args.benchmark_crypto),
            profile=crypto_profile,
            benchmark_profile=crypto_profile,
            stop_to_cash=True,
            **kwargs,
        )
        if result is not None:
            crypto_candidates.append(
                StrategyBundle(
                    result=result,
                    benchmark_gross_ret=pd.to_numeric(crypto_returns[str(args.benchmark_crypto)], errors="coerce").fillna(0.0).astype(float),
                    profile=crypto_profile,
                    benchmark_profile=crypto_profile,
                )
            )
    if not crypto_candidates:
        raise SystemExit("no crypto candidates")
    base_crypto_candidates = list(crypto_candidates)
    breadth_overlays = [
        ("crypto_breadth_all__55_70", base_crypto_candidates[0], crypto_breadth_all, 0.55, 0.70, "scale"),
        ("crypto_breadth_all__60_75", base_crypto_candidates[0], crypto_breadth_all, 0.60, 0.75, "scale"),
        ("crypto_breadth_major__55_70", base_crypto_candidates[0], crypto_breadth_major, 0.55, 0.70, "scale"),
    ]
    for candidate_id, base_bundle, breadth_signal, low, high, mode in breadth_overlays:
        crypto_candidates.append(
            _apply_breadth_overlay_to_bundle(
                candidate_id=candidate_id,
                bundle=base_bundle,
                breadth_signal=breadth_signal,
                low_threshold=low,
                high_threshold=high,
                mode=mode,
            )
        )
    crypto_candidate_df = pd.DataFrame([_result_row(b.result) for b in crypto_candidates]).sort_values(
        ["edge_vs_benchmark_net_total_return", "net_ann_return", "net_sharpe"], ascending=[False, False, False]
    )
    best_crypto_bundle = next(b for b in crypto_candidates if b.result.candidate_id == str(crypto_candidate_df.iloc[0]["candidate_id"]))

    attribution_specs = [
        ("attr__base", [ticker for ticker in crypto_tiers["crypto_all"]]),
        ("attr__no_btc", [ticker for ticker in crypto_tiers["crypto_all"] if ticker != "BTC-USD"]),
        ("attr__no_sol", [ticker for ticker in crypto_tiers["crypto_all"] if ticker != "SOL-USD"]),
        ("attr__no_bnb", [ticker for ticker in crypto_tiers["crypto_all"] if ticker != "BNB-USD"]),
        ("attr__no_btc_sol", [ticker for ticker in crypto_tiers["crypto_all"] if ticker not in {"BTC-USD", "SOL-USD"}]),
        ("attr__majors_only", [ticker for ticker in crypto_tiers["crypto_major8"]]),
        ("attr__midcaps_only", [ticker for ticker in crypto_tiers["crypto_midcap"]]),
    ]
    attribution_rows: list[dict[str, Any]] = []
    for tag, tickers in attribution_specs:
        result = _simulate_asset_rule(
            candidate_id=tag,
            family="crypto_attr",
            allowed_tickers=tickers,
            returns=crypto_returns,
            prices=crypto_prices,
            asset_table=crypto_assets,
            benchmark_ticker=str(args.benchmark_crypto),
            fallback_ticker=str(args.benchmark_crypto),
            score_mode="mom_vol_adj",
            lookback_days=21,
            rebalance_days=7,
            top_k=3,
            asset_ma_days=0,
            market_ma_days=200,
            relative_to_benchmark=False,
            skip_recent_days=0,
            trailing_stop_dd=None,
            hard_stop_loss=0.15,
            stop_to_cash=True,
            profile=crypto_profile,
            benchmark_profile=crypto_profile,
        )
        if result is None:
            continue
        attribution_rows.append(
            {
                "candidate_id": tag,
                "net_ann_return": result.net_ann_return,
                "net_total_return": result.net_total_return,
                "net_sharpe": result.net_sharpe,
                "net_max_drawdown": result.net_max_drawdown,
                "edge_vs_benchmark_net_total_return": result.edge_vs_benchmark,
            }
        )
    attribution_df = pd.DataFrame(attribution_rows)
    if not attribution_df.empty:
        base_row = attribution_df[attribution_df["candidate_id"] == "attr__base"].iloc[0]
        attribution_df["delta_vs_base_net_ann_return"] = pd.to_numeric(attribution_df["net_ann_return"], errors="coerce") - float(base_row["net_ann_return"])
        attribution_df["delta_vs_base_edge"] = pd.to_numeric(attribution_df["edge_vs_benchmark_net_total_return"], errors="coerce") - float(base_row["edge_vs_benchmark_net_total_return"])
    crypto_fragility_penalty = _fragility_penalty_from_attribution(attribution_df)

    equity_candidates: list[StrategyBundle] = []
    eq_v1 = _simulate_equity_group_sleeve(
        candidate_id="equities_v1__lb126__g2__a2",
        returns=equity_returns,
        prices=equity_prices,
        equity_groups=equity_group_map,
        benchmark_ticker=str(args.benchmark_equity),
        lookback_days=126,
        rebalance_days=21,
        group_top_k=2,
        assets_per_group=2,
        asset_ma_days=200,
        market_ma_days=200,
        score_mode="mom_vol_adj",
        profile=foreign_profile,
        benchmark_profile=foreign_profile,
    )
    if eq_v1 is not None:
        equity_candidates.append(
            StrategyBundle(
                result=eq_v1,
                benchmark_gross_ret=pd.to_numeric(equity_returns[str(args.benchmark_equity)], errors="coerce").fillna(0.0).astype(float),
                profile=foreign_profile,
                benchmark_profile=foreign_profile,
            )
        )
    v2_specs = [
        ("equities_v2__slow126__g2__a2", 63, 126, 2, 2, 126, 200, 200),
        ("equities_v2__slow126__g2__a3", 63, 126, 2, 3, 126, 200, 200),
        ("equities_v2__slow126__g3__a2", 63, 126, 3, 2, 126, 200, 200),
        ("equities_v2__slow126__g3__a3", 63, 126, 3, 3, 126, 200, 200),
        ("equities_v2__slow189__g2__a2", 63, 189, 2, 2, 126, 200, 200),
        ("equities_v2__slow189__g2__a3", 63, 189, 2, 3, 126, 200, 200),
        ("equities_v2__slow189__g3__a2", 63, 189, 3, 2, 126, 200, 200),
        ("equities_v2__slow189__g3__a3", 63, 189, 3, 3, 126, 200, 200),
        ("equities_v2__slow189__g3__a1", 63, 189, 3, 1, 126, 200, 200),
        ("equities_v2__slow189__g4__a1", 63, 189, 4, 1, 126, 200, 200),
        ("equities_v2__slow252__g3__a1", 63, 252, 3, 1, 126, 200, 200),
        ("equities_v2__slow252__g3__a2_m150", 63, 252, 3, 2, 126, 150, 150),
    ]
    for cid, gf, gs, gk, apg, alb, ama, mma in v2_specs:
        bundle = _simulate_equity_group_sleeve_v2(
            candidate_id=cid,
            returns=equity_returns,
            prices=equity_prices,
            asset_table=equity_assets,
            equity_groups=equity_group_map,
            benchmark_ticker=str(args.benchmark_equity),
            group_lookback_fast=int(gf),
            group_lookback_slow=int(gs),
            group_top_k=int(gk),
            assets_per_group=int(apg),
            asset_lookback=int(alb),
            asset_ma_days=int(ama),
            market_ma_days=int(mma),
            profile=foreign_profile,
            benchmark_profile=foreign_profile,
        )
        if bundle is not None:
            equity_candidates.append(bundle)
    v3_specs = [
        ("equities_v3__slow189__g2__a2__br30__cap45", 63, 189, 2, 2, 126, 200, 200, 0.30, 0.45),
        ("equities_v3__slow189__g2__a2__br35__cap40", 63, 189, 2, 2, 126, 200, 200, 0.35, 0.40),
        ("equities_v3__slow189__g3__a2__br30__cap45", 63, 189, 3, 2, 126, 200, 200, 0.30, 0.45),
        ("equities_v3__slow189__g3__a2__br35__cap40", 63, 189, 3, 2, 126, 200, 200, 0.35, 0.40),
        ("equities_v3__slow252__g3__a2__br30__cap45", 63, 252, 3, 2, 126, 200, 200, 0.30, 0.45),
        ("equities_v3__slow252__g2__a2__br35__cap40", 63, 252, 2, 2, 126, 200, 200, 0.35, 0.40),
    ]
    for cid, gf, gs, gk, apg, alb, ama, mma, br, cap in v3_specs:
        bundle = _simulate_equity_group_sleeve_v3(
            candidate_id=cid,
            returns=equity_returns,
            prices=equity_prices,
            asset_table=equity_assets,
            equity_groups=equity_group_map,
            benchmark_ticker=str(args.benchmark_equity),
            group_lookback_fast=int(gf),
            group_lookback_slow=int(gs),
            group_top_k=int(gk),
            assets_per_group=int(apg),
            asset_lookback=int(alb),
            asset_ma_days=int(ama),
            market_ma_days=int(mma),
            min_group_breadth=float(br),
            max_group_weight=float(cap),
            profile=foreign_profile,
            benchmark_profile=foreign_profile,
        )
        if bundle is not None:
            equity_candidates.append(bundle)
    if not equity_candidates:
        raise SystemExit("no equity candidates")
    regime_series = _load_structural_regime_series_local(ROOT)
    equity_base_df = pd.DataFrame([_result_row(b.result) for b in equity_candidates]).sort_values(
        ["net_ann_return", "net_sharpe"], ascending=[False, False]
    )
    robust_base_df = equity_base_df.copy()
    robust_base_df["robust_score"] = (
        0.45 * pd.to_numeric(robust_base_df["net_ann_return"], errors="coerce").fillna(0.0)
        + 0.35 * pd.to_numeric(robust_base_df["net_sharpe"], errors="coerce").fillna(0.0)
        + 0.20 * (1.0 + pd.to_numeric(robust_base_df["net_max_drawdown"], errors="coerce").fillna(-1.0))
    )
    equity_map = {b.result.candidate_id: b for b in equity_candidates}
    ann_pool = [equity_map[str(cid)] for cid in equity_base_df.head(4)["candidate_id"].astype(str).tolist()]
    robust_pool = [equity_map[str(cid)] for cid in robust_base_df.sort_values(["robust_score", "net_sharpe"], ascending=[False, False]).head(4)["candidate_id"].astype(str).tolist()]
    for agg_rank, agg_bundle in enumerate(ann_pool, start=1):
        for rob_rank, rob_bundle in enumerate(robust_pool, start=1):
            if agg_bundle.result.candidate_id == rob_bundle.result.candidate_id:
                continue
            equity_candidates.append(
                _simulate_equity_trail_switch_bundle(
                    candidate_id=f"equities_meta__trail_switch__a{agg_rank}__r{rob_rank}",
                    aggressive_bundle=agg_bundle,
                    robust_bundle=rob_bundle,
                    regime_series=regime_series,
                    spy_prices=pd.to_numeric(equity_prices[str(args.benchmark_equity)], errors="coerce"),
                )
            )
    equity_candidate_df = pd.DataFrame([_result_row(b.result) for b in equity_candidates]).sort_values(
        ["edge_vs_benchmark_net_total_return", "net_ann_return", "net_sharpe"], ascending=[False, False, False]
    )
    best_equity_bundle = next(b for b in equity_candidates if b.result.candidate_id == str(equity_candidate_df.iloc[0]["candidate_id"]))

    btc_prices = pd.to_numeric(crypto_prices[str(args.benchmark_crypto)], errors="coerce")
    spy_prices = pd.to_numeric(equity_prices[str(args.benchmark_equity)], errors="coerce")
    meta_candidates: list[StrategyBundle] = []
    meta_v1 = _build_meta_switch(
        candidate_id="meta_v1__btc63_vs_equity",
        crypto=best_crypto_bundle.result,
        equities=best_equity_bundle.result,
        btc_prices=btc_prices,
        spy_prices=spy_prices,
        crypto_profile=crypto_profile,
        equity_profile=foreign_profile,
    )
    meta_candidates.append(
        StrategyBundle(
            result=meta_v1,
            benchmark_gross_ret=(0.5 * pd.to_numeric(btc_prices.pct_change(), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0) + 0.5 * pd.to_numeric(spy_prices.pct_change(), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)).astype(float),
            profile=_profile_scaled(
                crypto_profile,
                profile_id="meta_v1_profile",
                label="Meta v1 blended",
                transaction_cost_bps=0.5 * crypto_profile.transaction_cost_bps_assumed + 0.5 * foreign_profile.transaction_cost_bps_assumed,
                fx_spread_bps=0.5 * crypto_profile.fx_spread_bps_assumed + 0.5 * foreign_profile.fx_spread_bps_assumed,
                capital_gains_tax_rate=0.5 * crypto_profile.capital_gains_tax_rate + 0.5 * foreign_profile.capital_gains_tax_rate,
                tax_timing="monthly_positive_proxy",
            ),
            benchmark_profile=_profile_scaled(
                crypto_profile,
                profile_id="meta_v1_profile",
                label="Meta v1 blended",
                transaction_cost_bps=0.5 * crypto_profile.transaction_cost_bps_assumed + 0.5 * foreign_profile.transaction_cost_bps_assumed,
                fx_spread_bps=0.5 * crypto_profile.fx_spread_bps_assumed + 0.5 * foreign_profile.fx_spread_bps_assumed,
                capital_gains_tax_rate=0.5 * crypto_profile.capital_gains_tax_rate + 0.5 * foreign_profile.capital_gains_tax_rate,
                tax_timing="monthly_positive_proxy",
            ),
        )
    )
    meta_candidates.append(
        _build_meta_switch_v2(
            candidate_id="meta_v2_disc__63_126",
            crypto_bundle=best_crypto_bundle,
            equity_bundle=best_equity_bundle,
            btc_prices=btc_prices,
            spy_prices=spy_prices,
            mode="discrete",
            fast_window=63,
            slow_window=126,
            vol_window=63,
            vol_quantile=0.80,
            max_crypto_weight=0.9,
        )
    )
    meta_candidates.append(
        _build_meta_switch_v2(
            candidate_id="meta_v2_cont__63_126_q80",
            crypto_bundle=best_crypto_bundle,
            equity_bundle=best_equity_bundle,
            btc_prices=btc_prices,
            spy_prices=spy_prices,
            mode="continuous",
            fast_window=63,
            slow_window=126,
            vol_window=63,
            vol_quantile=0.80,
            max_crypto_weight=0.85,
        )
    )
    meta_candidates.append(
        _build_meta_switch_v2(
            candidate_id="meta_v2_cont__126_189_q75",
            crypto_bundle=best_crypto_bundle,
            equity_bundle=best_equity_bundle,
            btc_prices=btc_prices,
            spy_prices=spy_prices,
            mode="continuous",
            fast_window=126,
            slow_window=189,
            vol_window=63,
            vol_quantile=0.75,
            max_crypto_weight=0.75,
        )
    )
    meta_candidates.append(
        _build_meta_switch_v3(
            candidate_id="meta_v3_asym__63_126__br55_45",
            crypto_bundle=best_crypto_bundle,
            equity_bundle=best_equity_bundle,
            btc_prices=btc_prices,
            spy_prices=spy_prices,
            breadth_signal=crypto_breadth_all,
            fast_window=63,
            slow_window=126,
            entry_breadth=0.55,
            exit_breadth=0.45,
            max_crypto_weight=0.80,
            cash_floor=0.10,
        )
    )
    meta_candidates.append(
        _build_meta_switch_v3(
            candidate_id="meta_v3_asym__63_126__br60_50",
            crypto_bundle=best_crypto_bundle,
            equity_bundle=best_equity_bundle,
            btc_prices=btc_prices,
            spy_prices=spy_prices,
            breadth_signal=crypto_breadth_all,
            fast_window=63,
            slow_window=126,
            entry_breadth=0.60,
            exit_breadth=0.50,
            max_crypto_weight=0.75,
            cash_floor=0.12,
        )
    )
    meta_candidates.append(
        _build_meta_switch_v3(
            candidate_id="meta_v3_major__63_126__br55_45",
            crypto_bundle=best_crypto_bundle,
            equity_bundle=best_equity_bundle,
            btc_prices=btc_prices,
            spy_prices=spy_prices,
            breadth_signal=crypto_breadth_major,
            fast_window=63,
            slow_window=126,
            entry_breadth=0.55,
            exit_breadth=0.45,
            max_crypto_weight=0.70,
            cash_floor=0.15,
        )
    )
    meta_df = pd.DataFrame([_result_row(b.result) for b in meta_candidates]).sort_values(
        ["edge_vs_benchmark_net_total_return", "net_ann_return", "net_sharpe"], ascending=[False, False, False]
    )

    stress_rows: list[dict[str, Any]] = []
    for bundle in [best_crypto_bundle, best_equity_bundle, *meta_candidates]:
        if bundle.result.suite.startswith("equities"):
            hard_profile = foreign_hard_profile
            hard_bench_profile = foreign_hard_profile
        elif bundle.result.suite.startswith("meta"):
            hard_profile = _profile_scaled(
                bundle.profile,
                profile_id=f"{bundle.profile.profile_id}_hard",
                label=f"{bundle.profile.label} hard",
                transaction_cost_bps=bundle.profile.transaction_cost_bps_assumed + 20.0,
                fx_spread_bps=bundle.profile.fx_spread_bps_assumed + 15.0,
                capital_gains_tax_rate=bundle.profile.capital_gains_tax_rate,
                tax_timing=bundle.profile.tax_timing,
            )
            hard_bench_profile = hard_profile
        else:
            hard_profile = crypto_hard_profile
            hard_bench_profile = crypto_hard_profile
        stress_rows.append(_stress_bundle(bundle, delay_days=0, profile=bundle.profile, benchmark_profile=bundle.benchmark_profile, label="base"))
        stress_rows.append(_stress_bundle(bundle, delay_days=1, profile=bundle.profile, benchmark_profile=bundle.benchmark_profile, label="delay_d1"))
        stress_rows.append(_stress_bundle(bundle, delay_days=2, profile=bundle.profile, benchmark_profile=bundle.benchmark_profile, label="delay_d2"))
        stress_rows.append(_stress_bundle(bundle, delay_days=0, profile=hard_profile, benchmark_profile=hard_bench_profile, label="hard_cost"))
        stress_rows.append(_stress_bundle(bundle, delay_days=1, profile=hard_profile, benchmark_profile=hard_bench_profile, label="hard_cost_delay_d1"))
    stress_df = pd.DataFrame(stress_rows)

    train_start, train_end = "2016-02-18", "2021-12-31"
    wf_blocks = [
        ("train_2016_2021", train_start, train_end),
        ("test_2022", "2022-01-01", "2022-12-31"),
        ("test_2023_2024", "2023-01-01", "2024-12-31"),
        ("test_2025_now", "2025-01-01", str(pd.Timestamp.now("UTC").date())),
    ]
    wf_scores = pd.DataFrame(
        [
            {"candidate_id": b.result.candidate_id, "train_score": _walkforward_score(b, train_start, train_end)}
            for b in meta_candidates
        ]
    ).sort_values("train_score", ascending=False)
    frozen_bundle = next(b for b in meta_candidates if b.result.candidate_id == str(wf_scores.iloc[0]["candidate_id"]))
    wf_rows: list[dict[str, Any]] = []
    for bundle in meta_candidates:
        wf_rows.extend(_walkforward_rows(bundle, wf_blocks))
    wf_df = pd.DataFrame(wf_rows)
    meta_rows: list[dict[str, Any]] = []
    for bundle in meta_candidates:
        crypto_corr = pd.to_numeric(bundle.result.gross_ret, errors="coerce").corr(
            pd.to_numeric(best_crypto_bundle.result.gross_ret.reindex(bundle.result.gross_ret.index), errors="coerce")
        )
        meta_fragility = float(max(0.0, min(1.0, abs(_safe_float(crypto_corr, 0.0))))) * float(crypto_fragility_penalty)
        meta_rows.append(
            _robust_objective_row(
                bundle=bundle,
                stress_df=stress_df,
                wf_df=wf_df,
                fragility_penalty=meta_fragility,
            )
        )
    tournament_df = pd.DataFrame(meta_rows).sort_values(["robust_score", "net_ann_return", "net_sharpe"], ascending=[False, False, False]).reset_index(drop=True)
    tournament_bundle = next(b for b in meta_candidates if b.result.candidate_id == str(tournament_df.iloc[0]["candidate_id"]))

    crypto_candidate_df.to_csv(outdir / "crypto_candidate_compare.csv", index=False)
    attribution_df.to_csv(outdir / "crypto_attribution.csv", index=False)
    equity_candidate_df.to_csv(outdir / "equity_candidate_compare.csv", index=False)
    meta_df.to_csv(outdir / "meta_candidate_compare.csv", index=False)
    stress_df.to_csv(outdir / "stress_compare.csv", index=False)
    wf_scores.to_csv(outdir / "walkforward_train_scores.csv", index=False)
    wf_df.to_csv(outdir / "walkforward_blocks.csv", index=False)
    tournament_df.to_csv(outdir / "tournament_compare.csv", index=False)

    previous_frontier_summary_path = ROOT / "results/validation/profit_frontier_expansion_suite/20260307T015636Z/summary.json"
    previous_frontier = json.loads(previous_frontier_summary_path.read_text(encoding="utf-8")) if previous_frontier_summary_path.exists() else {}
    prev_meta = previous_frontier.get("best_meta_switch", {}) if isinstance(previous_frontier, dict) else {}
    previous_layered_summaries = sorted((ROOT / "results/validation/profit_layered_engine_suite").glob("*/summary.json"))
    previous_layered = json.loads(previous_layered_summaries[-1].read_text(encoding="utf-8")) if previous_layered_summaries else {}
    prev_promoted = {}
    if isinstance(previous_layered, dict):
        prev_promoted = (
            previous_layered.get("promoted_candidate")
            or previous_layered.get("tournament_winner")
            or previous_layered.get("frozen_walkforward_winner")
            or {}
        )
    prev_promoted_id = str(prev_promoted.get("candidate_id", "")).strip()
    prev_robust_score = _safe_float(prev_promoted.get("robust_score"), float("-inf"))
    winner_row = tournament_df.iloc[0].to_dict()
    mean_test_edge = _safe_float(winner_row.get("mean_test_edge"), 0.0)
    positive_test_share = _safe_float(winner_row.get("positive_test_share"), 0.0)
    hard_cost_edge = _safe_float(winner_row.get("hard_cost_edge"), 0.0)
    if not prev_promoted_id:
        promotion_action = "promote_first"
        promoted = True
    elif prev_promoted_id == str(tournament_bundle.result.candidate_id):
        promotion_action = "keep_current"
        promoted = True
    elif float(winner_row.get("robust_score", float("-inf"))) > float(prev_robust_score) + 0.02 and mean_test_edge > 0.0 and positive_test_share >= 2.0 / 3.0 and hard_cost_edge > 0.0:
        promotion_action = "promote_new"
        promoted = True
    else:
        promotion_action = "hold_previous"
        promoted = False
    promoted_candidate_payload = _result_row(tournament_bundle.result)
    promoted_candidate_payload["robust_score"] = _safe_float(winner_row.get("robust_score"))
    promoted_candidate_payload["promotion_action"] = promotion_action
    promoted_candidate_payload["promoted"] = promoted
    promoted_candidate_payload["mean_test_edge"] = mean_test_edge
    promoted_candidate_payload["positive_test_share"] = positive_test_share
    promoted_candidate_payload["hard_cost_edge"] = hard_cost_edge
    status_map = {
        best_crypto_bundle.result.candidate_id: ("watch" if best_crypto_bundle.result.edge_vs_benchmark > 0.0 else "kill"),
        best_equity_bundle.result.candidate_id: ("watch" if best_equity_bundle.result.edge_vs_benchmark > 0.0 else "kill"),
        tournament_bundle.result.candidate_id: ("keep" if promoted else "watch"),
    }
    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "outdir": str(outdir),
        "best_crypto": _result_row(best_crypto_bundle.result),
        "best_equity": _result_row(best_equity_bundle.result),
        "best_meta_candidate": _result_row(meta_candidates[0].result if meta_df.empty else next(b.result for b in meta_candidates if b.result.candidate_id == str(meta_df.iloc[0]['candidate_id']))),
        "frozen_walkforward_winner": _result_row(frozen_bundle.result),
        "tournament_winner": {**_result_row(tournament_bundle.result), "robust_score": _safe_float(winner_row.get("robust_score"))},
        "promotion_decision": {
            "action": promotion_action,
            "promoted": promoted,
            "previous_candidate_id": prev_promoted_id,
            "new_candidate_id": tournament_bundle.result.candidate_id,
            "previous_robust_score": _safe_float(prev_robust_score),
            "new_robust_score": _safe_float(winner_row.get("robust_score")),
            "mean_test_edge": mean_test_edge,
            "positive_test_share": positive_test_share,
            "hard_cost_edge": hard_cost_edge,
        },
        "promoted_candidate": promoted_candidate_payload,
        "improvement_vs_frontier_meta": {
            "previous_candidate": str(prev_meta.get("candidate_id", "")),
            "previous_net_ann_return": _safe_float(prev_meta.get("net_ann_return")),
            "previous_net_sharpe": _safe_float(prev_meta.get("net_sharpe")),
            "previous_net_max_drawdown": _safe_float(prev_meta.get("net_max_drawdown")),
            "new_candidate": tournament_bundle.result.candidate_id,
            "new_net_ann_return": tournament_bundle.result.net_ann_return,
            "new_net_sharpe": tournament_bundle.result.net_sharpe,
            "new_net_max_drawdown": tournament_bundle.result.net_max_drawdown,
            "new_robust_score": _safe_float(winner_row.get("robust_score")),
            "delta_net_ann_return": tournament_bundle.result.net_ann_return - _safe_float(prev_meta.get("net_ann_return"), 0.0),
            "delta_net_sharpe": tournament_bundle.result.net_sharpe - _safe_float(prev_meta.get("net_sharpe"), 0.0),
            "delta_net_max_drawdown": tournament_bundle.result.net_max_drawdown - _safe_float(prev_meta.get("net_max_drawdown"), 0.0),
        },
        "insights": [
            f"Melhor cripto do stack: {best_crypto_bundle.result.candidate_id} com net_ann={best_crypto_bundle.result.net_ann_return:.4f} e edge_vs_benchmark={best_crypto_bundle.result.edge_vs_benchmark:.4f}.",
            f"Melhor equities base: {best_equity_bundle.result.candidate_id} com net_ann={best_equity_bundle.result.net_ann_return:.4f} e edge_vs_benchmark={best_equity_bundle.result.edge_vs_benchmark:.4f}.",
            f"Vencedor do torneio walk-forward: {tournament_bundle.result.candidate_id} com robust_score={_safe_float(winner_row.get('robust_score'), float('nan')):.4f}.",
            f"Decisao de promocao: {promotion_action}.",
        ],
        "artifacts": {
            "crypto_candidate_compare_csv": str(outdir / "crypto_candidate_compare.csv"),
            "crypto_attribution_csv": str(outdir / "crypto_attribution.csv"),
            "equity_candidate_compare_csv": str(outdir / "equity_candidate_compare.csv"),
            "meta_candidate_compare_csv": str(outdir / "meta_candidate_compare.csv"),
            "stress_compare_csv": str(outdir / "stress_compare.csv"),
            "walkforward_train_scores_csv": str(outdir / "walkforward_train_scores.csv"),
            "walkforward_blocks_csv": str(outdir / "walkforward_blocks.csv"),
            "tournament_compare_csv": str(outdir / "tournament_compare.csv"),
        },
    }
    summary_path = outdir / "summary.json"
    _write_json(summary_path, summary)

    research_rows = _research_rows(
        [best_crypto_bundle.result, best_equity_bundle.result, tournament_bundle.result],
        outdir=outdir,
        summary_path=summary_path,
        status_map=status_map,
    )
    (outdir / "profit_research_rows.json").write_text(json.dumps(research_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    write_run_manifest(
        outdir,
        script="scripts/bench/validation/run_profit_layered_engine_suite.py",
        params={
            "crypto_asset_groups": str(args.crypto_asset_groups),
            "equity_asset_groups": str(args.equity_asset_groups),
            "benchmark_crypto": str(args.benchmark_crypto),
            "benchmark_equity": str(args.benchmark_equity),
        },
        paths={
            "summary_json": str(summary_path),
            "crypto_candidate_compare_csv": str(outdir / "crypto_candidate_compare.csv"),
            "crypto_attribution_csv": str(outdir / "crypto_attribution.csv"),
            "equity_candidate_compare_csv": str(outdir / "equity_candidate_compare.csv"),
            "meta_candidate_compare_csv": str(outdir / "meta_candidate_compare.csv"),
            "stress_compare_csv": str(outdir / "stress_compare.csv"),
            "walkforward_train_scores_csv": str(outdir / "walkforward_train_scores.csv"),
            "walkforward_blocks_csv": str(outdir / "walkforward_blocks.csv"),
            "tournament_compare_csv": str(outdir / "tournament_compare.csv"),
            "profit_research_rows_json": str(outdir / "profit_research_rows.json"),
        },
        extra={
            "notes": [
                "Layered suite cobre: meta-switch v3, breadth cripto, equities base v3, objetivo com fragilidade e torneio walk-forward.",
            ]
        },
    )
    print(json.dumps({"status": "ok", "outdir": str(outdir), "walkforward_winner": frozen_bundle.result.candidate_id}, ensure_ascii=False))


if __name__ == "__main__":
    main()
