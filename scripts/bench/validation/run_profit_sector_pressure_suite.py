#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.portfolio import (  # noqa: E402
    estimate_regime_moments,
    estimate_transition_matrix,
    simulate_regime_conditioned_paths,
    summarize_portfolio_distribution,
)
from engine.structural.covariance_estimators import estimate_corr  # noqa: E402
from engine.structural.run_manifest import write_run_manifest  # noqa: E402
from execution.net_assumptions import load_net_assumption_profiles  # noqa: E402
from scripts.bench.validation.run_profit_frontier_expansion_suite import (  # noqa: E402
    StrategyResult,
    _build_equity_group_map,
    _ensure_benchmark_columns,
    _evaluate_net,
    _load_asset_table,
    _load_daily_universe,
    _precompute_scores_skip,
    _select_crypto_tiers,
    _simulate_asset_rule,
    _write_json,
)
from scripts.bench.validation.run_profit_layered_engine_suite import (  # noqa: E402
    StrategyBundle,
    _equity_v2_group_scores,
    _load_structural_regime_series_local,
    _simulate_equity_group_sleeve_v2,
    _simulate_equity_group_sleeve_v3,
    _simulate_equity_trail_switch_bundle,
)
from scripts.bench.validation.run_profit_regime_simulation_suite import (  # noqa: E402
    _apply_mc_guard,
    _blended_profile,
    _build_meta_v1_allocation,
)


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def _rank01(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    valid = x.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return x.rank(pct=True, method="average").astype(float)


def _orient_eigenvector(v: np.ndarray) -> np.ndarray:
    out = np.asarray(v, dtype=float).copy()
    if np.nansum(out) < 0:
        out *= -1.0
    return out


def _regime_forward_fill(idx: pd.Index, regime_series: pd.Series) -> pd.Series:
    reg = pd.Series(regime_series.copy())
    reg.index = pd.to_datetime(reg.index, errors="coerce")
    reg = reg[reg.index.notna()].sort_index()
    out = reg.reindex(pd.to_datetime(idx, errors="coerce")).ffill().bfill()
    return out.astype(str)


def _group_pressure_snapshot(
    *,
    group_ret_df: pd.DataFrame,
    regime_series: pd.Series,
    pos: int,
    lookback: int,
    horizon: int,
    n_paths: int,
) -> tuple[pd.Series, pd.DataFrame]:
    hist = group_ret_df.iloc[max(0, pos - int(lookback)) : pos].copy()
    hist = hist.dropna(how="all")
    if hist.shape[0] < max(40, int(lookback) // 2) or hist.shape[1] < 2:
        empty = pd.Series(np.zeros(group_ret_df.shape[1], dtype=float), index=group_ret_df.columns, dtype=float)
        detail = pd.DataFrame({"group": group_ret_df.columns.astype(str), "pressure_score": 0.0, "ruin_prob_m10": 0.0, "avg_abs_corr": 0.0, "v1_abs": 0.0})
        return empty, detail

    corr = pd.DataFrame(
        estimate_corr(hist.fillna(0.0).to_numpy(dtype=float), method="ledoit_wolf"),
        index=hist.columns,
        columns=hist.columns,
    )
    eigvals, eigvecs = np.linalg.eigh(corr.to_numpy(dtype=float))
    order = np.argsort(eigvals)[::-1]
    v1 = _orient_eigenvector(eigvecs[:, order][:, 0])
    avg_abs_corr = corr.abs().where(~np.eye(len(corr), dtype=bool)).mean(axis=1).fillna(0.0)

    hist_regime = _regime_forward_fill(hist.index, regime_series)
    moments = estimate_regime_moments(hist.fillna(0.0), hist_regime, min_obs=20)
    states, transition = estimate_transition_matrix(hist_regime)
    start_state = str(hist_regime.iloc[-1]).lower()
    sim_paths, _ = simulate_regime_conditioned_paths(
        regime_moments=moments,
        transition_matrix=transition,
        states=states,
        start_state=start_state,
        horizon=int(horizon),
        n_paths=int(n_paths),
        random_state=23 + int(pos),
    )
    ruin_rows: list[dict[str, Any]] = []
    for idx, group in enumerate(hist.columns.astype(str)):
        weights = np.zeros(hist.shape[1], dtype=float)
        weights[idx] = 1.0
        dist = summarize_portfolio_distribution(sim_paths, weights=weights)
        ruin_rows.append(
            {
                "group": group,
                "ruin_prob_m10": _safe_float(dist.get("ruin_prob_m10"), 0.0),
                "sim_p05_21d": _safe_float(dist.get("terminal_p05"), 0.0),
                "expected_shortfall_21d": _safe_float(dist.get("expected_shortfall_p05"), 0.0),
            }
        )
    ruin_df = pd.DataFrame(ruin_rows).set_index("group")
    detail = pd.DataFrame(
        {
            "group": hist.columns.astype(str),
            "v1_abs": np.abs(v1),
            "avg_abs_corr": avg_abs_corr.reindex(hist.columns).to_numpy(dtype=float),
        }
    ).set_index("group")
    detail = detail.join(ruin_df, how="left").fillna(0.0)
    detail["pressure_score"] = (
        0.50 * _rank01(detail["v1_abs"]).fillna(0.0)
        + 0.25 * _rank01(detail["avg_abs_corr"]).fillna(0.0)
        + 0.25 * _rank01(detail["ruin_prob_m10"]).fillna(0.0)
    )
    return detail["pressure_score"].astype(float), detail.reset_index()


def _asset_systemic_snapshot(asset_hist: pd.DataFrame) -> pd.Series:
    hist = asset_hist.dropna(how="all")
    if hist.shape[0] < 40 or hist.shape[1] < 2:
        return pd.Series(np.zeros(asset_hist.shape[1], dtype=float), index=asset_hist.columns, dtype=float)
    corr = pd.DataFrame(
        estimate_corr(hist.fillna(0.0).to_numpy(dtype=float), method="ledoit_wolf"),
        index=hist.columns,
        columns=hist.columns,
    )
    eigvals, eigvecs = np.linalg.eigh(corr.to_numpy(dtype=float))
    order = np.argsort(eigvals)[::-1]
    v1 = np.abs(_orient_eigenvector(eigvecs[:, order][:, 0]))
    avg_abs_corr = corr.abs().where(~np.eye(len(corr), dtype=bool)).mean(axis=1).fillna(0.0)
    detail = pd.DataFrame(
        {
            "v1_abs": pd.Series(v1, index=corr.index, dtype=float),
            "avg_abs_corr": avg_abs_corr.astype(float),
        }
    )
    detail["systemic_role_score"] = (
        0.60 * _rank01(detail["v1_abs"]).fillna(0.0) + 0.40 * _rank01(detail["avg_abs_corr"]).fillna(0.0)
    ).astype(float)
    return detail["systemic_role_score"].astype(float)


def _simulate_equity_group_sleeve_v4_sector_pressure(
    *,
    candidate_id: str,
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    asset_table: pd.DataFrame,
    equity_groups: dict[str, list[str]],
    benchmark_ticker: str,
    regime_series: pd.Series,
    group_lookback_fast: int,
    group_lookback_slow: int,
    group_top_k: int,
    assets_per_group: int,
    asset_lookback: int,
    asset_ma_days: int,
    market_ma_days: int,
    pressure_lookback: int,
    pressure_horizon: int,
    pressure_penalty: float,
    profile,
    benchmark_profile,
) -> tuple[StrategyBundle | None, pd.DataFrame]:
    all_tickers = sorted({ticker for tickers in equity_groups.values() for ticker in tickers if ticker in returns.columns})
    if not all_tickers:
        return None, pd.DataFrame()

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
    group_ret_df = pd.concat(group_returns, axis=1, sort=False).sort_index().fillna(0.0)
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

    warmup = max(int(group_lookback_slow), int(asset_lookback), int(asset_ma_days), int(market_ma_days), int(pressure_lookback)) + 2
    rebalance_positions = list(range(int(max(1, warmup)), returns.shape[0], 21))
    if not rebalance_positions:
        return None, pd.DataFrame()

    daily_ret = np.zeros(returns.shape[0], dtype=float)
    daily_turnover = np.zeros(returns.shape[0], dtype=float)
    prev_weights: dict[str, float] = {"CASH": 1.0}
    pressure_log: list[dict[str, Any]] = []

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
        pressure_score, pressure_detail = _group_pressure_snapshot(
            group_ret_df=group_ret_df,
            regime_series=regime_series,
            pos=pos,
            lookback=int(pressure_lookback),
            horizon=int(pressure_horizon),
            n_paths=300,
        )
        adj_group_scores = group_scores * (1.0 - float(pressure_penalty) * pressure_score.reindex(group_scores.index).fillna(0.0).clip(0.0, 1.0))
        chosen_groups = adj_group_scores.sort_values(ascending=False).head(int(max(1, group_top_k))).index.astype(str).tolist()
        top_pressure = pressure_detail.sort_values("pressure_score", ascending=False).head(1)
        pressure_log.append(
            {
                "date": str(returns.index[pos].date()),
                "top_pressure_group": str(top_pressure["group"].iloc[0]) if not top_pressure.empty else "n/d",
                "top_pressure_score": _safe_float(top_pressure["pressure_score"].iloc[0], 0.0) if not top_pressure.empty else 0.0,
                "pressure_mean": _safe_float(pressure_score.mean(), 0.0),
                "pressure_penalty": float(pressure_penalty),
            }
        )

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
        group_weight_map = adj_group_scores.reindex(chosen_groups).clip(lower=0.0)
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
        suite="equities_sector_pressure",
        candidate_id=candidate_id,
        family="equities_sector_pressure",
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
        notes=f"sector_pressure_penalty={pressure_penalty:.2f};lookback={pressure_lookback};horizon={pressure_horizon}",
    )
    return StrategyBundle(result=result, benchmark_gross_ret=benchmark_gross, profile=profile, benchmark_profile=benchmark_profile), pd.DataFrame(pressure_log)


def _simulate_equity_group_sleeve_v5_hybrid_rank(
    *,
    candidate_id: str,
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    asset_table: pd.DataFrame,
    equity_groups: dict[str, list[str]],
    benchmark_ticker: str,
    regime_series: pd.Series,
    group_lookback_fast: int,
    group_lookback_slow: int,
    group_top_k: int,
    assets_per_group: int,
    asset_lookback: int,
    asset_ma_days: int,
    market_ma_days: int,
    pressure_lookback: int,
    pressure_horizon: int,
    pressure_penalty: float,
    systemic_penalty: float,
    profile,
    benchmark_profile,
) -> tuple[StrategyBundle | None, pd.DataFrame]:
    all_tickers = sorted({ticker for tickers in equity_groups.values() for ticker in tickers if ticker in returns.columns})
    if not all_tickers:
        return None, pd.DataFrame()

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
    group_ret_df = pd.concat(group_returns, axis=1, sort=False).sort_index().fillna(0.0)
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

    warmup = max(int(group_lookback_slow), int(asset_lookback), int(asset_ma_days), int(market_ma_days), int(pressure_lookback)) + 2
    rebalance_positions = list(range(int(max(1, warmup)), returns.shape[0], 21))
    if not rebalance_positions:
        return None, pd.DataFrame()

    daily_ret = np.zeros(returns.shape[0], dtype=float)
    daily_turnover = np.zeros(returns.shape[0], dtype=float)
    prev_weights: dict[str, float] = {"CASH": 1.0}
    pressure_log: list[dict[str, Any]] = []

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
        pressure_score, pressure_detail = _group_pressure_snapshot(
            group_ret_df=group_ret_df,
            regime_series=regime_series,
            pos=pos,
            lookback=int(pressure_lookback),
            horizon=int(pressure_horizon),
            n_paths=300,
        )
        adj_group_scores = group_scores * (1.0 - float(pressure_penalty) * pressure_score.reindex(group_scores.index).fillna(0.0).clip(0.0, 1.0))
        chosen_groups = adj_group_scores.sort_values(ascending=False).head(int(max(1, group_top_k))).index.astype(str).tolist()
        top_pressure = pressure_detail.sort_values("pressure_score", ascending=False).head(1)
        pressure_log.append(
            {
                "date": str(returns.index[pos].date()),
                "top_pressure_group": str(top_pressure["group"].iloc[0]) if not top_pressure.empty else "n/d",
                "top_pressure_score": _safe_float(top_pressure["pressure_score"].iloc[0], 0.0) if not top_pressure.empty else 0.0,
                "pressure_mean": _safe_float(pressure_score.mean(), 0.0),
                "pressure_penalty": float(pressure_penalty),
                "systemic_penalty": float(systemic_penalty),
            }
        )

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
            hist_start = max(0, pos - int(pressure_lookback))
            asset_systemic = _asset_systemic_snapshot(
                returns.loc[returns.index[hist_start:pos], merged.index].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            ).reindex(merged.index).fillna(0.0)
            merged["liq"] = liq_rank
            merged["systemic"] = asset_systemic
            merged["score"] = (
                0.52 * merged["base"]
                + 0.18 * merged["fast"]
                + 0.10 * (merged["persist"].fillna(0.5) - 0.5)
                + 0.10 * merged["liq"]
                + 0.10 * (1.0 - merged["systemic"])
                - float(systemic_penalty) * merged["systemic"]
            )
            chosen_tickers.extend(merged.sort_values("score", ascending=False).head(int(max(1, assets_per_group))).index.astype(str).tolist())
        chosen_tickers = sorted(set(chosen_tickers))
        if not chosen_tickers:
            weights = {"CASH": 1.0}
            daily_turnover[pos] += 0.5 * float(sum(abs(float(prev_weights.get(k, 0.0)) - float(weights.get(k, 0.0))) for k in sorted(set(prev_weights) | set(weights))))
            prev_weights = weights
            continue
        group_weight_map = adj_group_scores.reindex(chosen_groups).clip(lower=0.0)
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
        suite="equities_hybrid_rank",
        candidate_id=candidate_id,
        family="equities_hybrid_rank",
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
        notes=f"hybrid_rank;pressure_penalty={pressure_penalty:.2f};systemic_penalty={systemic_penalty:.2f};lookback={pressure_lookback};horizon={pressure_horizon}",
    )
    return StrategyBundle(result=result, benchmark_gross_ret=benchmark_gross, profile=profile, benchmark_profile=benchmark_profile), pd.DataFrame(pressure_log)


def _result_row(result: StrategyResult) -> dict[str, Any]:
    return {
        "candidate_id": result.candidate_id,
        "suite": result.suite,
        "family": result.family,
        "benchmark_ticker": result.benchmark_ticker,
        "net_ann_return": result.net_ann_return,
        "net_total_return": result.net_total_return,
        "net_sharpe": result.net_sharpe,
        "net_max_drawdown": result.net_max_drawdown,
        "edge_vs_benchmark_net_total_return": result.edge_vs_benchmark,
        "avg_turnover_daily": result.avg_turnover_daily,
        "notes": result.notes,
    }


def _research_row(result: StrategyResult, *, outdir: Path, status: str, methodology: str, label: str) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": result.candidate_id,
        "label": label,
        "methodology": methodology,
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
        "groups": "meta_switch,sector_pressure",
        "artifacts": {
            "suite_dir": str(outdir),
            "summary_json": str(outdir / "summary.json"),
        },
        "notes": result.notes,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Test sector pressure overlays from correlation/eigen insights on the best profit meta.")
    ap.add_argument("--outdir", default="")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--top-k-crypto", type=int, default=3)
    args = ap.parse_args()

    run_id = _run_id()
    outdir = Path(args.outdir) if args.outdir else ROOT / "results" / "validation" / "profit_sector_pressure_suite" / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    prices_dir = (ROOT / args.prices_dir).resolve()

    profiles = load_net_assumption_profiles(ROOT / "config" / "profit_net_assumptions.json")
    foreign_profile = profiles["profiles"]["foreign_financial_brazil_resident"]
    crypto_profile = profiles["profiles"]["crypto_global_brazil_resident_conservative"]
    blended_profile = _blended_profile(
        crypto_profile,
        foreign_profile,
        profile_id="sector_pressure_blended",
        label="Sector pressure blended",
    )

    crypto_assets = _load_asset_table((ROOT / args.crypto_asset_groups).resolve(), (ROOT / args.crypto_asset_metadata).resolve())
    crypto_returns, crypto_prices, viability = _load_daily_universe(
        prices_dir=prices_dir,
        asset_table=crypto_assets,
        min_history_days=600,
        max_abs_daily_return=1.5,
    )
    crypto_returns, crypto_prices = _ensure_benchmark_columns(
        crypto_returns,
        crypto_prices,
        prices_dir,
        [str(args.benchmark_crypto), "ETH-USD"],
    )
    crypto_tiers = _select_crypto_tiers(crypto_assets, viability)

    equity_assets = _load_asset_table((ROOT / args.equity_asset_groups).resolve(), (ROOT / args.equity_asset_metadata).resolve())
    equity_returns, equity_prices, _ = _load_daily_universe(
        prices_dir=prices_dir,
        asset_table=equity_assets,
        min_history_days=1200,
        max_abs_daily_return=0.8,
    )
    equity_returns, equity_prices = _ensure_benchmark_columns(
        equity_returns,
        equity_prices,
        prices_dir,
        [str(args.benchmark_equity)],
    )
    returns = pd.concat([equity_returns, crypto_returns], axis=1)
    prices = pd.concat([equity_prices, crypto_prices], axis=1)
    returns = returns.loc[:, ~returns.columns.duplicated()].sort_index()
    prices = prices.loc[:, ~prices.columns.duplicated()].sort_index()
    equity_group_map = _build_equity_group_map(equity_assets, equity_returns)

    crypto_major_result = _simulate_asset_rule(
        candidate_id="crypto_major8__momvol21",
        family="crypto",
        allowed_tickers=crypto_tiers["crypto_major8"],
        returns=crypto_returns[crypto_tiers["crypto_major8"] + [str(args.benchmark_crypto)]].copy() if str(args.benchmark_crypto) not in crypto_tiers["crypto_major8"] else crypto_returns[crypto_tiers["crypto_major8"]].copy(),
        prices=crypto_prices[crypto_tiers["crypto_major8"] + [str(args.benchmark_crypto)]].copy() if str(args.benchmark_crypto) not in crypto_tiers["crypto_major8"] else crypto_prices[crypto_tiers["crypto_major8"]].copy(),
        asset_table=crypto_assets,
        benchmark_ticker=str(args.benchmark_crypto),
        fallback_ticker=str(args.benchmark_crypto),
        score_mode="mom_vol_adj",
        lookback_days=21,
        rebalance_days=7,
        top_k=int(args.top_k_crypto),
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
    if crypto_major_result is None:
        raise SystemExit("failed to build crypto major8 baseline")
    crypto_major_bundle = StrategyBundle(
        result=crypto_major_result,
        benchmark_gross_ret=crypto_major_result.gross_ret * 0.0,
        profile=crypto_profile,
        benchmark_profile=crypto_profile,
    )
    crypto_major_bundle = replace(
        crypto_major_bundle,
        benchmark_gross_ret=pd.to_numeric(crypto_returns[str(args.benchmark_crypto)], errors="coerce").reindex(crypto_major_result.gross_ret.index).fillna(0.0).astype(float),
    )

    eq_a2 = _simulate_equity_group_sleeve_v2(
        candidate_id="equities_v2__slow189__g4__a1",
        returns=equity_returns,
        prices=equity_prices,
        asset_table=equity_assets,
        equity_groups=equity_group_map,
        benchmark_ticker=str(args.benchmark_equity),
        group_lookback_fast=63,
        group_lookback_slow=189,
        group_top_k=4,
        assets_per_group=1,
        asset_lookback=126,
        asset_ma_days=200,
        market_ma_days=200,
        profile=foreign_profile,
        benchmark_profile=foreign_profile,
    )
    eq_r1 = _simulate_equity_group_sleeve_v3(
        candidate_id="equities_v3__slow189__g3__a2__br35__cap40",
        returns=equity_returns,
        prices=equity_prices,
        asset_table=equity_assets,
        equity_groups=equity_group_map,
        benchmark_ticker=str(args.benchmark_equity),
        group_lookback_fast=63,
        group_lookback_slow=189,
        group_top_k=3,
        assets_per_group=2,
        asset_lookback=126,
        asset_ma_days=200,
        market_ma_days=200,
        min_group_breadth=0.35,
        max_group_weight=0.40,
        profile=foreign_profile,
        benchmark_profile=foreign_profile,
    )
    if eq_a2 is None or eq_r1 is None:
        raise SystemExit("failed to rebuild equity baseline sleeves")

    regime_series = _load_structural_regime_series_local(ROOT)
    equity_meta_a2r1 = _simulate_equity_trail_switch_bundle(
        candidate_id="equities_meta__trail_switch__a2__r1",
        aggressive_bundle=eq_a2,
        robust_bundle=eq_r1,
        regime_series=regime_series,
        spy_prices=pd.to_numeric(prices[str(args.benchmark_equity)], errors="coerce"),
    )

    pressure_logs: list[pd.DataFrame] = []
    pressure_bundles: list[StrategyBundle] = []
    for penalty in [0.25, 0.40]:
        eq_sp, pressure_log = _simulate_equity_group_sleeve_v4_sector_pressure(
            candidate_id=f"equities_v4__sector_pressure_p{int(round(penalty * 100)):02d}",
            returns=returns,
            prices=prices,
            asset_table=equity_assets,
            equity_groups=equity_group_map,
            benchmark_ticker=str(args.benchmark_equity),
            regime_series=regime_series,
            group_lookback_fast=63,
            group_lookback_slow=189,
            group_top_k=4,
            assets_per_group=1,
            asset_lookback=126,
            asset_ma_days=200,
            market_ma_days=200,
            pressure_lookback=120,
            pressure_horizon=21,
            pressure_penalty=float(penalty),
            profile=foreign_profile,
            benchmark_profile=foreign_profile,
        )
        if eq_sp is not None:
            pressure_logs.append(pressure_log.assign(candidate_id=eq_sp.result.candidate_id))
            pressure_bundles.append(
                _simulate_equity_trail_switch_bundle(
                    candidate_id=f"equities_meta__trail_switch__sector_p{int(round(penalty * 100)):02d}",
                    aggressive_bundle=eq_sp,
                    robust_bundle=eq_r1,
                    regime_series=regime_series,
                    spy_prices=pd.to_numeric(prices[str(args.benchmark_equity)], errors="coerce"),
                )
            )

    hybrid_logs: list[pd.DataFrame] = []
    hybrid_bundles: list[StrategyBundle] = []
    for pressure_penalty, systemic_penalty in [(0.15, 0.15), (0.25, 0.20)]:
        eq_hybrid, hybrid_log = _simulate_equity_group_sleeve_v5_hybrid_rank(
            candidate_id=f"equities_v5__hybrid_p{int(round(pressure_penalty * 100)):02d}_s{int(round(systemic_penalty * 100)):02d}",
            returns=equity_returns,
            prices=equity_prices,
            asset_table=equity_assets,
            equity_groups=equity_group_map,
            benchmark_ticker=str(args.benchmark_equity),
            regime_series=regime_series,
            group_lookback_fast=63,
            group_lookback_slow=189,
            group_top_k=4,
            assets_per_group=1,
            asset_lookback=126,
            asset_ma_days=200,
            market_ma_days=200,
            pressure_lookback=120,
            pressure_horizon=21,
            pressure_penalty=float(pressure_penalty),
            systemic_penalty=float(systemic_penalty),
            profile=foreign_profile,
            benchmark_profile=foreign_profile,
        )
        if eq_hybrid is not None:
            hybrid_logs.append(hybrid_log.assign(candidate_id=eq_hybrid.result.candidate_id))
            hybrid_bundles.append(
                _simulate_equity_trail_switch_bundle(
                    candidate_id=f"equities_meta__trail_switch__hybrid_p{int(round(pressure_penalty * 100)):02d}_s{int(round(systemic_penalty * 100)):02d}",
                    aggressive_bundle=eq_hybrid,
                    robust_bundle=eq_r1,
                    regime_series=regime_series,
                    spy_prices=pd.to_numeric(prices[str(args.benchmark_equity)], errors="coerce"),
                )
            )

    btc_prices = pd.to_numeric(prices[str(args.benchmark_crypto)], errors="coerce")
    spy_prices = pd.to_numeric(prices[str(args.benchmark_equity)], errors="coerce")
    baseline_raw = _build_meta_v1_allocation(
        crypto_bundle=crypto_major_bundle,
        equity_bundle=equity_meta_a2r1,
        btc_prices=btc_prices,
        spy_prices=spy_prices,
        profile=blended_profile,
    )
    baseline = replace(
        baseline_raw,
        bundle=replace(
            baseline_raw.bundle,
            result=replace(
                baseline_raw.bundle.result,
                candidate_id="meta_major8_eq_a2r1",
                notes="best_profit_baseline; crypto_set=major8; equity_set=a2r1",
            ),
        ),
    )

    candidates: list[StrategyResult] = [baseline.bundle.result]
    research_rows: list[dict[str, Any]] = [
        _research_row(baseline.bundle.result, outdir=outdir, status="keep", methodology="sector_pressure_meta_baseline", label="Meta lucro maximo atual"),
    ]

    mc_guard, _ = _apply_mc_guard(
        candidate_id="meta_major8_eq_a2r1_mc_guard",
        base=baseline,
        returns=pd.concat(
            {
                "crypto": pd.to_numeric(crypto_major_bundle.result.gross_ret, errors="coerce"),
                "equity": pd.to_numeric(equity_meta_a2r1.result.gross_ret, errors="coerce"),
            },
            axis=1,
            sort=False,
        ).dropna(how="all"),
        regime=regime_series,
        profile=blended_profile,
        lookback=252,
        horizon=21,
        n_paths=400,
        step=42,
    )
    candidates.append(mc_guard.bundle.result)
    research_rows.append(_research_row(mc_guard.bundle.result, outdir=outdir, status="watch", methodology="sector_pressure_meta_mc_guard", label="Meta lucro maximo com guard monte carlo"))

    for bundle in pressure_bundles:
        meta_raw = _build_meta_v1_allocation(
            crypto_bundle=crypto_major_bundle,
            equity_bundle=bundle,
            btc_prices=btc_prices,
            spy_prices=spy_prices,
            profile=blended_profile,
        )
        meta_candidate = replace(
            meta_raw,
            bundle=replace(
                meta_raw.bundle,
                result=replace(
                    meta_raw.bundle.result,
                    candidate_id=f"meta_major8_{bundle.result.candidate_id}",
                    notes=f"major8 + {bundle.result.candidate_id}",
                ),
            ),
        )
        candidates.append(meta_candidate.bundle.result)
        status = "watch"
        if meta_candidate.bundle.result.net_ann_return > baseline.bundle.result.net_ann_return and meta_candidate.bundle.result.net_sharpe >= baseline.bundle.result.net_sharpe:
            status = "keep"
        research_rows.append(
            _research_row(
                meta_candidate.bundle.result,
                outdir=outdir,
                status=status,
                methodology="sector_pressure_meta_switch",
                label=f"Meta com penalidade setorial {bundle.result.candidate_id}",
            )
        )

    for bundle in hybrid_bundles:
        meta_raw = _build_meta_v1_allocation(
            crypto_bundle=crypto_major_bundle,
            equity_bundle=bundle,
            btc_prices=btc_prices,
            spy_prices=spy_prices,
            profile=blended_profile,
        )
        meta_candidate = replace(
            meta_raw,
            bundle=replace(
                meta_raw.bundle,
                result=replace(
                    meta_raw.bundle.result,
                    candidate_id=f"meta_major8_{bundle.result.candidate_id}",
                    notes=f"major8 + {bundle.result.candidate_id}",
                ),
            ),
        )
        candidates.append(meta_candidate.bundle.result)
        status = "watch"
        if meta_candidate.bundle.result.net_ann_return > baseline.bundle.result.net_ann_return and meta_candidate.bundle.result.net_sharpe >= baseline.bundle.result.net_sharpe:
            status = "keep"
        research_rows.append(
            _research_row(
                meta_candidate.bundle.result,
                outdir=outdir,
                status=status,
                methodology="hybrid_rank_meta_switch",
                label=f"Meta com ranking híbrido {bundle.result.candidate_id}",
            )
        )

    compare = pd.DataFrame([_result_row(r) for r in candidates]).sort_values(["net_ann_return", "net_sharpe"], ascending=[False, False]).reset_index(drop=True)
    compare.to_csv(outdir / "candidate_compare.csv", index=False)
    log_frames = [df for df in pressure_logs + hybrid_logs if not df.empty]
    if log_frames:
        pd.concat(log_frames, axis=0, sort=False).to_csv(outdir / "pressure_log.csv", index=False)

    best = compare.iloc[0].to_dict() if not compare.empty else {}
    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "outdir": str(outdir),
        "baseline_candidate": baseline.bundle.result.candidate_id,
        "best_candidate": best,
        "insights": [
            "Overlay setorial usa matriz de correlacao + autovetor dominante + Monte Carlo por grupo.",
            "O foco foi testar se a melhor estrategia de lucro melhora quando penalizamos grupos estruturalmente mais congestionados.",
            "As penalidades foram aplicadas so na perna de equities, preservando o meta-switch cripto/acoes.",
        ],
        "artifacts": {
            "candidate_compare_csv": str(outdir / "candidate_compare.csv"),
            "pressure_log_csv": str(outdir / "pressure_log.csv") if pressure_logs else "",
            "profit_research_rows_json": str(outdir / "profit_research_rows.json"),
        },
    }
    (outdir / "profit_research_rows.json").write_text(json.dumps(research_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_json(outdir / "summary.json", summary)
    write_run_manifest(
        outdir=outdir,
        script=str(Path(__file__).resolve()),
        params={
            "benchmark_equity": str(args.benchmark_equity),
            "benchmark_crypto": str(args.benchmark_crypto),
            "top_k_crypto": int(args.top_k_crypto),
        },
        paths={
            "summary_json": str(outdir / "summary.json"),
            "candidate_compare_csv": str(outdir / "candidate_compare.csv"),
            "profit_research_rows_json": str(outdir / "profit_research_rows.json"),
        },
        extra={"insights": summary["insights"]},
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
