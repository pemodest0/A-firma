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
from execution.net_assumptions import (  # noqa: E402
    NetAssumptionProfile,
    apply_net_assumptions,
    load_net_assumption_profiles,
)
from scripts.bench.validation.run_profit_10x_rule_search import (  # noqa: E402
    _ensure_benchmark_columns,
    _load_asset_table,
    _load_daily_universe,
    _rolling_ten_x_stats,
    _safe_float,
    _top_k_indices,
)


CRYPTO_EXCLUDED = set()
EQUITY_EXCLUDED = {"crypto", "fx", "vol_regime", "bonds_credit", "bonds_rates", "commodities", "miscellaneous"}


@dataclass(frozen=True)
class StrategyResult:
    suite: str
    candidate_id: str
    family: str
    benchmark_ticker: str
    gross_ret: pd.Series
    turnover: pd.Series
    net_ret: pd.Series
    benchmark_net_ret: pd.Series
    net_ann_return: float
    net_total_return: float
    net_sharpe: float
    net_max_drawdown: float
    edge_vs_benchmark: float
    avg_turnover_daily: float
    hit_rate_10x_5y: float
    years_to_10x_full: float
    notes: str


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _precompute_scores_skip(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    lookbacks: list[int],
    asset_ma_days_list: list[int],
    benchmark_ticker: str,
    skip_recent_days: int,
) -> tuple[dict[tuple[int, str], pd.DataFrame], dict[int, pd.DataFrame], dict[int, pd.Series]]:
    score_map: dict[tuple[int, str], pd.DataFrame] = {}
    skip = int(max(0, skip_recent_days))
    for lookback in lookbacks:
        lb = int(max(2, lookback))
        end_price = prices.shift(1 + skip)
        start_price = prices.shift(1 + skip + lb)
        total = end_price / start_price - 1.0
        score_map[(lb, "mom_total")] = total.replace([np.inf, -np.inf], np.nan)

        vol = returns.shift(1 + skip).rolling(lb, min_periods=max(20, lb // 2)).std(ddof=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            vol_adj = total / vol.replace(0.0, np.nan)
        score_map[(lb, "mom_vol_adj")] = vol_adj.replace([np.inf, -np.inf], np.nan)

    asset_ma_filters: dict[int, pd.DataFrame] = {0: pd.DataFrame(True, index=prices.index, columns=prices.columns)}
    for days in asset_ma_days_list:
        dd = int(days)
        if dd <= 0:
            continue
        close = prices.shift(1)
        ma = close.rolling(dd, min_periods=max(20, dd // 2)).mean()
        asset_ma_filters[dd] = (close > ma).fillna(False)

    benchmark_filters: dict[int, pd.Series] = {0: pd.Series(True, index=prices.index, dtype=bool)}
    if benchmark_ticker in prices.columns:
        bench = pd.to_numeric(prices[benchmark_ticker], errors="coerce").astype(float)
        close = bench.shift(1)
        for days in asset_ma_days_list:
            dd = int(days)
            if dd <= 0:
                continue
            ma = close.rolling(dd, min_periods=max(20, dd // 2)).mean()
            benchmark_filters[dd] = (close > ma).fillna(False).astype(bool)
    return score_map, asset_ma_filters, benchmark_filters


def _evaluate_net(
    *,
    gross_ret: pd.Series,
    turnover: pd.Series,
    profile: NetAssumptionProfile,
    benchmark_ret: pd.Series,
    benchmark_profile: NetAssumptionProfile,
    cash_weight: pd.Series | None = None,
    initial_capital_brl: float | None = None,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    gross_ret = pd.to_numeric(gross_ret, errors="coerce").fillna(0.0).astype(float)
    turnover = pd.to_numeric(turnover, errors="coerce").reindex(gross_ret.index).fillna(0.0).astype(float)
    benchmark_ret = pd.to_numeric(benchmark_ret, errors="coerce").reindex(gross_ret.index).fillna(0.0).astype(float)

    net_frame = apply_net_assumptions(
        gross_ret,
        turnover,
        profile=profile,
        periods_index=gross_ret.index,
        cash_weight=cash_weight,
        initial_capital_brl=initial_capital_brl,
    )
    benchmark_net = apply_net_assumptions(
        benchmark_ret,
        pd.Series(np.zeros(len(benchmark_ret), dtype=float), index=benchmark_ret.index, dtype=float),
        profile=benchmark_profile,
        periods_index=benchmark_ret.index,
    )
    summary = summarize_return_series(net_frame["net_ret"], periods_per_year=periods_per_year)
    bench = summarize_return_series(benchmark_net["net_ret"], periods_per_year=periods_per_year)
    return {
        "net_ret": pd.to_numeric(net_frame["net_ret"], errors="coerce").fillna(0.0).astype(float),
        "benchmark_net_ret": pd.to_numeric(benchmark_net["net_ret"], errors="coerce").fillna(0.0).astype(float),
        "net_ann_return": _safe_float(summary.get("annualized_return")),
        "net_total_return": _safe_float(summary.get("total_return")),
        "net_sharpe": _safe_float(summary.get("sharpe")),
        "net_max_drawdown": _safe_float(summary.get("max_drawdown")),
        "edge_vs_benchmark": _safe_float(summary.get("total_return")) - _safe_float(bench.get("total_return")),
        "avg_turnover_daily": float(turnover.mean()) if not turnover.empty else float("nan"),
        "avg_cash_ret": float(pd.to_numeric(net_frame.get("cash_ret"), errors="coerce").fillna(0.0).mean()) if "cash_ret" in net_frame else 0.0,
        "avg_withholding_ret": float(pd.to_numeric(net_frame.get("withholding_ret"), errors="coerce").fillna(0.0).mean()) if "withholding_ret" in net_frame else 0.0,
    }


def _series_total_return(ret: pd.Series) -> float:
    x = pd.to_numeric(ret, errors="coerce").dropna().astype(float)
    if x.empty:
        return float("nan")
    return float(np.prod(1.0 + x.to_numpy(dtype=float)) - 1.0)


def _slice_stats(net_ret: pd.Series) -> dict[str, float]:
    stats = summarize_return_series(net_ret, periods_per_year=252)
    return {
        "ann_return": _safe_float(stats.get("annualized_return")),
        "total_return": _safe_float(stats.get("total_return")),
        "sharpe": _safe_float(stats.get("sharpe")),
        "max_drawdown": _safe_float(stats.get("max_drawdown")),
    }


def _equal_weight_series(returns: pd.DataFrame, tickers: list[str]) -> pd.Series:
    available = [ticker for ticker in tickers if ticker in returns.columns]
    if not available:
        return pd.Series(dtype=float)
    block = returns[available].apply(pd.to_numeric, errors="coerce")
    return block.mean(axis=1, skipna=True).fillna(0.0).astype(float)


def _select_crypto_tiers(asset_table: pd.DataFrame, viability: pd.DataFrame) -> dict[str, list[str]]:
    meta = asset_table.drop_duplicates(subset=["ticker"], keep="first").copy()
    joined = meta.merge(viability[["ticker", "days_available"]], on="ticker", how="left")
    joined["days_available"] = pd.to_numeric(joined["days_available"], errors="coerce").fillna(0.0)
    joined["liquidity_proxy"] = pd.to_numeric(joined["liquidity_proxy"], errors="coerce").fillna(0.0)
    joined = joined.sort_values(["days_available", "liquidity_proxy", "ticker"], ascending=[False, False, True]).reset_index(drop=True)
    all_assets = joined["ticker"].astype(str).tolist()
    majors = joined.head(8)["ticker"].astype(str).tolist()
    mids = [ticker for ticker in all_assets if ticker not in set(majors)]
    return {
        "crypto_all": all_assets,
        "crypto_major8": majors,
        "crypto_midcap": mids,
    }


def _build_equity_group_map(asset_table: pd.DataFrame, returns: pd.DataFrame) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for group, sub in asset_table.groupby("asset_group", sort=True):
        tickers = [ticker for ticker in sub["ticker"].astype(str).tolist() if ticker in returns.columns]
        if len(tickers) >= 6:
            groups[str(group)] = tickers
    return groups


def _simulate_asset_rule(
    *,
    candidate_id: str,
    family: str,
    allowed_tickers: list[str],
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    asset_table: pd.DataFrame,
    benchmark_ticker: str,
    fallback_ticker: str,
    score_mode: str,
    lookback_days: int,
    rebalance_days: int,
    top_k: int,
    asset_ma_days: int,
    market_ma_days: int,
    relative_to_benchmark: bool,
    skip_recent_days: int,
    trailing_stop_dd: float | None,
    hard_stop_loss: float | None,
    stop_to_cash: bool,
    profile: NetAssumptionProfile,
    benchmark_profile: NetAssumptionProfile,
) -> StrategyResult | None:
    all_tickers = list(returns.columns.astype(str))
    ticker_to_col = {ticker: idx for idx, ticker in enumerate(all_tickers)}
    allowed_idx = np.array([ticker_to_col[t] for t in allowed_tickers if t in ticker_to_col], dtype=int)
    if allowed_idx.size == 0:
        return None

    score_map, asset_ma_filters, benchmark_filters = _precompute_scores_skip(
        returns,
        prices,
        lookbacks=[int(lookback_days)],
        asset_ma_days_list=[0, int(asset_ma_days), int(market_ma_days)],
        benchmark_ticker=benchmark_ticker,
        skip_recent_days=int(skip_recent_days),
    )
    score_df = score_map[(int(lookback_days), str(score_mode))]
    score_arr = score_df.reindex(index=returns.index, columns=all_tickers).to_numpy(dtype=float)
    asset_ma_arr = asset_ma_filters[int(asset_ma_days)].reindex(index=returns.index, columns=all_tickers).fillna(False).to_numpy(dtype=bool)
    benchmark_ok = benchmark_filters[int(market_ma_days)].reindex(returns.index).fillna(False).to_numpy(dtype=bool)
    ret_arr = returns.reindex(columns=all_tickers).to_numpy(dtype=float)
    rel_scores = None
    if bool(relative_to_benchmark) and fallback_ticker in score_df.columns:
        rel_scores = pd.to_numeric(score_df[fallback_ticker], errors="coerce").to_numpy(dtype=float)

    warmup = max(int(lookback_days), int(asset_ma_days), int(market_ma_days), int(skip_recent_days)) + 2
    rebalance_positions = list(range(int(max(1, warmup)), ret_arr.shape[0], int(max(1, rebalance_days))))
    if not rebalance_positions:
        return None

    daily_ret = np.zeros(ret_arr.shape[0], dtype=float)
    daily_turnover = np.zeros(ret_arr.shape[0], dtype=float)
    prev_weights: dict[str, float] = {"CASH": 1.0}
    stop_events = 0

    for pos_idx, pos in enumerate(rebalance_positions):
        next_pos = rebalance_positions[pos_idx + 1] if pos_idx + 1 < len(rebalance_positions) else ret_arr.shape[0]
        score_row = score_arr[pos]
        valid = np.zeros(score_row.shape[0], dtype=bool)
        valid[allowed_idx] = True
        valid &= np.isfinite(score_row)
        valid &= asset_ma_arr[pos]
        valid &= score_row > 0.0
        if pos > 0:
            valid &= np.isfinite(ret_arr[pos - 1])
        if rel_scores is not None and pos < rel_scores.shape[0] and np.isfinite(rel_scores[pos]):
            valid &= score_row > float(rel_scores[pos])
        if int(market_ma_days) > 0 and not bool(benchmark_ok[pos]):
            valid[:] = False

        selected_idx = _top_k_indices(score_row, valid, int(top_k))
        if not selected_idx:
            target_weights: dict[str, float] = {"CASH": 1.0}
            daily_turnover[pos] += float(abs(_safe_float(prev_weights.get("CASH", 0.0), 0.0) - 1.0) + sum(abs(v) for k, v in prev_weights.items() if k != "CASH")) / 2.0
            prev_weights = {"CASH": 1.0}
            continue

        weight = 1.0 / float(len(selected_idx))
        tickers = [all_tickers[idx] for idx in selected_idx]
        target_weights = {ticker: weight for ticker in tickers}
        daily_turnover[pos] += 0.5 * float(sum(abs(float(prev_weights.get(k, 0.0)) - float(target_weights.get(k, 0.0))) for k in sorted(set(prev_weights) | set(target_weights))))
        prev_weights = dict(target_weights)

        cum = 1.0
        peak = 1.0
        active = True
        for day in range(pos, next_pos):
            if not active:
                daily_ret[day] = 0.0
                continue
            block = np.nan_to_num(ret_arr[day, selected_idx], nan=0.0)
            day_ret = float(np.mean(block))
            daily_ret[day] = day_ret
            cum *= 1.0 + day_ret
            peak = max(peak, cum)
            dd = 1.0 - (cum / peak if peak > 0.0 else 1.0)
            loss_from_entry = 1.0 - cum
            trailing_hit = trailing_stop_dd is not None and np.isfinite(float(trailing_stop_dd)) and dd >= float(trailing_stop_dd)
            hard_hit = hard_stop_loss is not None and np.isfinite(float(hard_stop_loss)) and loss_from_entry >= float(hard_stop_loss)
            if trailing_hit or hard_hit:
                stop_events += 1
                active = False
                exit_day = day + 1
                if exit_day < next_pos:
                    daily_turnover[exit_day] += 1.0
                prev_weights = {"CASH": 1.0} if bool(stop_to_cash) else dict(target_weights)

    gross = pd.Series(daily_ret, index=returns.index, dtype=float)
    turnover = pd.Series(daily_turnover, index=returns.index, dtype=float)
    benchmark_ret = pd.to_numeric(returns.get(benchmark_ticker), errors="coerce").reindex(returns.index).fillna(0.0).astype(float)
    perf = _evaluate_net(
        gross_ret=gross,
        turnover=turnover,
        profile=profile,
        benchmark_ret=benchmark_ret,
        benchmark_profile=benchmark_profile,
    )
    hit5 = _rolling_ten_x_stats(perf["net_ret"], horizon_days=1260)
    wealth = (1.0 + perf["net_ret"]).cumprod()
    hit_full = wealth[wealth >= 10.0]
    years_to_10x = float((hit_full.index[0] - wealth.index[0]).days / 365.25) if not hit_full.empty else float("nan")
    notes = f"skip_recent={skip_recent_days};trail={trailing_stop_dd};hard={hard_stop_loss};stops={stop_events}"
    return StrategyResult(
        suite="crypto_rule" if family.startswith("crypto") else "asset_rule",
        candidate_id=candidate_id,
        family=family,
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
        hit_rate_10x_5y=_safe_float(hit5.get("hit_rate")),
        years_to_10x_full=years_to_10x,
        notes=notes,
    )


def _simulate_equity_group_sleeve(
    *,
    candidate_id: str,
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    equity_groups: dict[str, list[str]],
    benchmark_ticker: str,
    lookback_days: int,
    rebalance_days: int,
    group_top_k: int,
    assets_per_group: int,
    asset_ma_days: int,
    market_ma_days: int,
    score_mode: str,
    profile: NetAssumptionProfile,
    benchmark_profile: NetAssumptionProfile,
) -> StrategyResult | None:
    all_equity_tickers = sorted({ticker for tickers in equity_groups.values() for ticker in tickers if ticker in returns.columns})
    if not all_equity_tickers:
        return None

    score_map, asset_ma_filters, benchmark_filters = _precompute_scores_skip(
        returns[all_equity_tickers + ([benchmark_ticker] if benchmark_ticker in returns.columns and benchmark_ticker not in all_equity_tickers else [])],
        prices[all_equity_tickers + ([benchmark_ticker] if benchmark_ticker in prices.columns and benchmark_ticker not in all_equity_tickers else [])],
        lookbacks=[int(lookback_days)],
        asset_ma_days_list=[0, int(asset_ma_days), int(market_ma_days)],
        benchmark_ticker=benchmark_ticker,
        skip_recent_days=0,
    )
    asset_scores = score_map[(int(lookback_days), str(score_mode))]
    group_returns = {group: _equal_weight_series(returns, tickers) for group, tickers in equity_groups.items()}
    group_prices = {group: (1.0 + pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)).cumprod() for group, series in group_returns.items()}
    group_ret_df = pd.concat(group_returns, axis=1).sort_index()
    group_price_df = pd.concat(group_prices, axis=1).sort_index()
    group_score_map, _, _ = _precompute_scores_skip(
        group_ret_df,
        group_price_df,
        lookbacks=[int(lookback_days)],
        asset_ma_days_list=[0],
        benchmark_ticker=benchmark_ticker if benchmark_ticker in group_price_df.columns else "",
        skip_recent_days=0,
    )
    group_scores = group_score_map[(int(lookback_days), str(score_mode))]
    market_ok = benchmark_filters[int(market_ma_days)].reindex(returns.index).fillna(False)

    warmup = max(int(lookback_days), int(asset_ma_days), int(market_ma_days)) + 2
    rebalance_positions = list(range(int(max(1, warmup)), returns.shape[0], int(max(1, rebalance_days))))
    if not rebalance_positions:
        return None

    daily_ret = np.zeros(returns.shape[0], dtype=float)
    daily_turnover = np.zeros(returns.shape[0], dtype=float)
    prev_weights: dict[str, float] = {"CASH": 1.0}
    all_tickers = list(returns.columns.astype(str))

    for pos_idx, pos in enumerate(rebalance_positions):
        next_pos = rebalance_positions[pos_idx + 1] if pos_idx + 1 < len(rebalance_positions) else returns.shape[0]
        if int(market_ma_days) > 0 and not bool(market_ok.iloc[pos]):
            weights = {"CASH": 1.0}
            daily_turnover[pos] += 0.5 * float(sum(abs(float(prev_weights.get(k, 0.0)) - float(weights.get(k, 0.0))) for k in sorted(set(prev_weights) | set(weights))))
            prev_weights = weights
            continue
        gscore = pd.to_numeric(group_scores.iloc[pos], errors="coerce").dropna().astype(float)
        gscore = gscore[gscore > 0.0]
        if gscore.empty:
            weights = {"CASH": 1.0}
            daily_turnover[pos] += 0.5 * float(sum(abs(float(prev_weights.get(k, 0.0)) - float(weights.get(k, 0.0))) for k in sorted(set(prev_weights) | set(weights))))
            prev_weights = weights
            continue
        chosen_groups = gscore.sort_values(ascending=False).head(int(max(1, group_top_k))).index.astype(str).tolist()
        chosen_tickers: list[str] = []
        for group in chosen_groups:
            eligible = [ticker for ticker in equity_groups.get(group, []) if ticker in asset_scores.columns]
            if not eligible:
                continue
            ascore = pd.to_numeric(asset_scores.loc[asset_scores.index[pos], eligible], errors="coerce").dropna().astype(float)
            if asset_ma_days > 0:
                ma_ok = asset_ma_filters[int(asset_ma_days)].reindex(index=returns.index, columns=asset_scores.columns).iloc[pos]
                ma_ok = ma_ok.reindex(eligible).fillna(False)
                ascore = ascore[ma_ok.reindex(ascore.index).fillna(False)]
            ascore = ascore[ascore > 0.0]
            if ascore.empty:
                continue
            chosen_tickers.extend(ascore.sort_values(ascending=False).head(int(max(1, assets_per_group))).index.astype(str).tolist())
        chosen_tickers = sorted(set(chosen_tickers))
        if not chosen_tickers:
            weights = {"CASH": 1.0}
            daily_turnover[pos] += 0.5 * float(sum(abs(float(prev_weights.get(k, 0.0)) - float(weights.get(k, 0.0))) for k in sorted(set(prev_weights) | set(weights))))
            prev_weights = weights
            continue
        w = 1.0 / float(len(chosen_tickers))
        weights = {ticker: w for ticker in chosen_tickers}
        daily_turnover[pos] += 0.5 * float(sum(abs(float(prev_weights.get(k, 0.0)) - float(weights.get(k, 0.0))) for k in sorted(set(prev_weights) | set(weights))))
        prev_weights = dict(weights)
        block = returns.loc[returns.index[pos:next_pos], chosen_tickers].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        daily_ret[pos:next_pos] = block.mean(axis=1).to_numpy(dtype=float)

    gross = pd.Series(daily_ret, index=returns.index, dtype=float)
    turnover = pd.Series(daily_turnover, index=returns.index, dtype=float)
    benchmark_ret = pd.to_numeric(returns.get(benchmark_ticker), errors="coerce").reindex(returns.index).fillna(0.0).astype(float)
    perf = _evaluate_net(
        gross_ret=gross,
        turnover=turnover,
        profile=profile,
        benchmark_ret=benchmark_ret,
        benchmark_profile=benchmark_profile,
    )
    return StrategyResult(
        suite="equities_causal",
        candidate_id=candidate_id,
        family="equities_causal",
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
        notes=f"group_top_k={group_top_k};assets_per_group={assets_per_group};lookback={lookback_days}",
    )


def _build_meta_switch(
    *,
    candidate_id: str,
    crypto: StrategyResult,
    equities: StrategyResult,
    btc_prices: pd.Series,
    spy_prices: pd.Series,
    crypto_profile: NetAssumptionProfile,
    equity_profile: NetAssumptionProfile,
) -> StrategyResult:
    idx = crypto.gross_ret.index.intersection(equities.gross_ret.index)
    crypto_ret = crypto.gross_ret.reindex(idx).fillna(0.0).astype(float)
    eq_ret = equities.gross_ret.reindex(idx).fillna(0.0).astype(float)
    btc_close = pd.to_numeric(btc_prices.reindex(idx), errors="coerce").astype(float)
    spy_close = pd.to_numeric(spy_prices.reindex(idx), errors="coerce").astype(float)
    btc_ok = (btc_close.shift(1) > btc_close.shift(1).rolling(200, min_periods=100).mean()).fillna(False)
    spy_ok = (spy_close.shift(1) > spy_close.shift(1).rolling(200, min_periods=100).mean()).fillna(False)

    crypto_trail = (1.0 + crypto_ret).rolling(63, min_periods=21).apply(np.prod, raw=True) - 1.0
    eq_trail = (1.0 + eq_ret).rolling(63, min_periods=21).apply(np.prod, raw=True) - 1.0

    gross = pd.Series(np.zeros(len(idx), dtype=float), index=idx, dtype=float)
    turnover = pd.Series(np.zeros(len(idx), dtype=float), index=idx, dtype=float)
    prev = "cash"
    for dt in idx:
        prefer_crypto = bool(btc_ok.loc[dt]) and (_safe_float(crypto_trail.loc[dt], -1.0) > _safe_float(eq_trail.loc[dt], -1.0))
        if not bool(spy_ok.loc[dt]) and not bool(btc_ok.loc[dt]):
            source = "cash"
        elif prefer_crypto:
            source = "crypto"
        else:
            source = "equity"
        if source != prev:
            turnover.loc[dt] = 1.0 if prev != "cash" else 0.5
        prev = source
        if source == "crypto":
            gross.loc[dt] = float(crypto_ret.loc[dt])
        elif source == "equity":
            gross.loc[dt] = float(eq_ret.loc[dt])
        else:
            gross.loc[dt] = 0.0

    btc_bench = pd.to_numeric(btc_close.pct_change(), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    spy_bench = pd.to_numeric(spy_close.pct_change(), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    blended_benchmark = 0.5 * btc_bench + 0.5 * spy_bench
    blended_profile = NetAssumptionProfile(
        profile_id="meta_switch_blended",
        label="Meta switch blended",
        jurisdiction="blended",
        transaction_cost_bps_assumed=0.5 * crypto_profile.transaction_cost_bps_assumed + 0.5 * equity_profile.transaction_cost_bps_assumed,
        fx_spread_bps_assumed=0.5 * crypto_profile.fx_spread_bps_assumed + 0.5 * equity_profile.fx_spread_bps_assumed,
        capital_gains_tax_rate=0.5 * crypto_profile.capital_gains_tax_rate + 0.5 * equity_profile.capital_gains_tax_rate,
        tax_timing="monthly_positive_proxy",
        dividend_withholding_mode="not_applicable",
        monthly_sales_exemption_modeled=False,
        notes=("meta switch blended profile",),
    )
    perf = _evaluate_net(
        gross_ret=gross,
        turnover=turnover,
        profile=blended_profile,
        benchmark_ret=blended_benchmark,
        benchmark_profile=blended_profile,
    )
    hit5 = _rolling_ten_x_stats(perf["net_ret"], horizon_days=1260)
    wealth = (1.0 + perf["net_ret"]).cumprod()
    hit_full = wealth[wealth >= 10.0]
    years_to_10x = float((hit_full.index[0] - wealth.index[0]).days / 365.25) if not hit_full.empty else float("nan")
    return StrategyResult(
        suite="meta_switch",
        candidate_id=candidate_id,
        family="meta_switch",
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
        notes="crypto if BTC risk-on and trailing 63d beats equities; else equities; cash if both BTC and SPY below MM200.",
    )


def _result_row(result: StrategyResult) -> dict[str, Any]:
    return {
        "suite": result.suite,
        "candidate_id": result.candidate_id,
        "family": result.family,
        "benchmark_ticker": result.benchmark_ticker,
        "net_ann_return": result.net_ann_return,
        "net_total_return": result.net_total_return,
        "net_sharpe": result.net_sharpe,
        "net_max_drawdown": result.net_max_drawdown,
        "edge_vs_benchmark_net_total_return": result.edge_vs_benchmark,
        "avg_turnover_daily": result.avg_turnover_daily,
        "hit_rate_10x_5y": result.hit_rate_10x_5y,
        "years_to_10x_full": result.years_to_10x_full,
        "notes": result.notes,
    }


def _oos_block_rows(result: StrategyResult, blocks: list[tuple[str, str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, start, end in blocks:
        sub = result.net_ret.loc[(result.net_ret.index >= pd.Timestamp(start)) & (result.net_ret.index <= pd.Timestamp(end))]
        bench = result.benchmark_net_ret.loc[sub.index]
        if sub.empty:
            continue
        rows.append(
            {
                "candidate_id": result.candidate_id,
                "suite": result.suite,
                "block": label,
                "start": start,
                "end": end,
                **_slice_stats(sub),
                "benchmark_total_return": _slice_stats(bench).get("total_return"),
                "edge_total_return": _safe_float(_slice_stats(sub).get("total_return")) - _safe_float(_slice_stats(bench).get("total_return")),
            }
        )
    return rows


def _research_rows(results: list[StrategyResult], *, outdir: Path, summary_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        status = "keep" if result.edge_vs_benchmark > 0.0 and result.net_ann_return > 0.15 else ("watch" if result.edge_vs_benchmark > 0.0 else "kill")
        rows.append(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "candidate_id": result.candidate_id,
                "label": result.candidate_id,
                "methodology": f"frontier_{result.suite}",
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
    ap = argparse.ArgumentParser(description="Bateria de expansao de fronteira: crypto stops/custos, equities causais e meta-switch.")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--net-assumptions", default="config/profit_net_assumptions.json")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--outdir-root", default="results/validation/profit_frontier_expansion_suite")
    args = ap.parse_args()

    prices_dir = (ROOT / args.prices_dir).resolve()
    outdir = (ROOT / args.outdir_root / _run_id()).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    profiles = load_net_assumption_profiles((ROOT / args.net_assumptions).resolve())
    foreign_profile = profiles["profiles"]["foreign_financial_brazil_resident"]
    crypto_profile = profiles["profiles"]["crypto_global_brazil_resident_conservative"]

    crypto_assets = _load_asset_table((ROOT / args.crypto_asset_groups).resolve(), (ROOT / args.crypto_asset_metadata).resolve())
    crypto_returns, crypto_prices, crypto_viability = _load_daily_universe(
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
    crypto_tiers = _select_crypto_tiers(crypto_assets, crypto_viability)

    equity_assets = _load_asset_table((ROOT / args.equity_asset_groups).resolve(), (ROOT / args.equity_asset_metadata).resolve())
    equity_assets = equity_assets[~equity_assets["asset_group"].astype(str).isin(EQUITY_EXCLUDED)].copy()
    equity_returns, equity_prices, equity_viability = _load_daily_universe(
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
    equity_group_map = _build_equity_group_map(equity_assets, equity_returns)

    crypto_results: list[StrategyResult] = []
    crypto_specs = [
        {"base_id": "momvol", "lookback": 21, "rebalance": 7, "top_k": 3, "score_mode": "mom_vol_adj", "asset_ma": 0, "market_ma": 200, "relative": False},
        {"base_id": "fastrel", "lookback": 21, "rebalance": 7, "top_k": 2, "score_mode": "mom_total", "asset_ma": 0, "market_ma": 0, "relative": True},
        {"base_id": "slowrel", "lookback": 252, "rebalance": 7, "top_k": 2, "score_mode": "mom_total", "asset_ma": 200, "market_ma": 200, "relative": True},
    ]
    overlay_specs = [
        {"suffix": "base", "skip_recent": 0, "trail": None, "hard": None, "rebalance": None},
        {"suffix": "trail25", "skip_recent": 0, "trail": 0.25, "hard": None, "rebalance": None},
        {"suffix": "trail35", "skip_recent": 0, "trail": 0.35, "hard": None, "rebalance": None},
        {"suffix": "hard15", "skip_recent": 0, "trail": None, "hard": 0.15, "rebalance": None},
        {"suffix": "trail35_skip21", "skip_recent": 21, "trail": 0.35, "hard": None, "rebalance": None},
        {"suffix": "rb21", "skip_recent": 0, "trail": 0.25, "hard": None, "rebalance": 21},
    ]
    for tier_name, tickers in crypto_tiers.items():
        for base in crypto_specs:
            for overlay in overlay_specs:
                rb = int(overlay["rebalance"] or base["rebalance"])
                candidate_id = (
                    f"{tier_name}__{base['base_id']}"
                    f"__lb{int(base['lookback']):03d}"
                    f"__rb{rb:02d}"
                    f"__k{int(base['top_k'])}"
                    f"__{overlay['suffix']}"
                )
                result = _simulate_asset_rule(
                    candidate_id=candidate_id,
                    family=tier_name,
                    allowed_tickers=tickers,
                    returns=crypto_returns,
                    prices=crypto_prices,
                    asset_table=crypto_assets,
                    benchmark_ticker=str(args.benchmark_crypto),
                    fallback_ticker=str(args.benchmark_crypto),
                    score_mode=str(base["score_mode"]),
                    lookback_days=int(base["lookback"]),
                    rebalance_days=rb,
                    top_k=int(base["top_k"]),
                    asset_ma_days=int(base["asset_ma"]),
                    market_ma_days=int(base["market_ma"]),
                    relative_to_benchmark=bool(base["relative"]),
                    skip_recent_days=int(overlay["skip_recent"]),
                    trailing_stop_dd=overlay["trail"],
                    hard_stop_loss=overlay["hard"],
                    stop_to_cash=True,
                    profile=crypto_profile,
                    benchmark_profile=crypto_profile,
                )
                if result is not None:
                    crypto_results.append(result)

    equities_results: list[StrategyResult] = []
    for lookback in [63, 126]:
        for group_top_k in [2, 3]:
            for assets_per_group in [1, 2]:
                candidate_id = f"equities_causal__lb{lookback:03d}__g{group_top_k}__a{assets_per_group}"
                result = _simulate_equity_group_sleeve(
                    candidate_id=candidate_id,
                    returns=equity_returns,
                    prices=equity_prices,
                    equity_groups=equity_group_map,
                    benchmark_ticker=str(args.benchmark_equity),
                    lookback_days=int(lookback),
                    rebalance_days=21,
                    group_top_k=int(group_top_k),
                    assets_per_group=int(assets_per_group),
                    asset_ma_days=200,
                    market_ma_days=200,
                    score_mode="mom_vol_adj",
                    profile=foreign_profile,
                    benchmark_profile=foreign_profile,
                )
                if result is not None:
                    equities_results.append(result)

    if not crypto_results:
        raise SystemExit("no crypto results produced")
    if not equities_results:
        raise SystemExit("no equities results produced")

    crypto_df = pd.DataFrame([_result_row(x) for x in crypto_results]).sort_values(
        ["edge_vs_benchmark_net_total_return", "net_ann_return", "net_sharpe"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    equities_df = pd.DataFrame([_result_row(x) for x in equities_results]).sort_values(
        ["edge_vs_benchmark_net_total_return", "net_ann_return", "net_sharpe"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    best_crypto = next(x for x in crypto_results if x.candidate_id == str(crypto_df.iloc[0]["candidate_id"]))
    best_equities = next(x for x in equities_results if x.candidate_id == str(equities_df.iloc[0]["candidate_id"]))

    btc_prices = pd.to_numeric(crypto_prices[str(args.benchmark_crypto)], errors="coerce")
    spy_prices = pd.to_numeric(equity_prices[str(args.benchmark_equity)], errors="coerce")
    meta_result = _build_meta_switch(
        candidate_id="meta_switch__btc63_vs_equity63",
        crypto=best_crypto,
        equities=best_equities,
        btc_prices=btc_prices,
        spy_prices=spy_prices,
        crypto_profile=crypto_profile,
        equity_profile=foreign_profile,
    )
    meta_df = pd.DataFrame([_result_row(meta_result)])

    crypto_benchmarks = [
        ("BTC-USD", pd.to_numeric(crypto_returns[str(args.benchmark_crypto)], errors="coerce").fillna(0.0).astype(float), crypto_profile),
        ("BTC_ETH_50_50", 0.5 * pd.to_numeric(crypto_returns[str(args.benchmark_crypto)], errors="coerce").fillna(0.0).astype(float) + 0.5 * pd.to_numeric(crypto_returns.get("ETH-USD"), errors="coerce").reindex(crypto_returns.index).fillna(0.0).astype(float), crypto_profile),
        ("EW_CRYPTO_ALL", _equal_weight_series(crypto_returns, crypto_tiers["crypto_all"]), crypto_profile),
        ("EW_CRYPTO_MAJOR8", _equal_weight_series(crypto_returns, crypto_tiers["crypto_major8"]), crypto_profile),
    ]
    equity_benchmarks = [
        ("SPY", pd.to_numeric(equity_returns[str(args.benchmark_equity)], errors="coerce").fillna(0.0).astype(float), foreign_profile),
        ("EW_EQUITIES_ALL", _equal_weight_series(equity_returns, sorted({t for v in equity_group_map.values() for t in v})), foreign_profile),
    ]
    benchmark_rows: list[dict[str, Any]] = []
    for name, series, profile in crypto_benchmarks:
        stats = _evaluate_net(
            gross_ret=series,
            turnover=pd.Series(np.zeros(len(series), dtype=float), index=series.index, dtype=float),
            profile=profile,
            benchmark_ret=pd.to_numeric(crypto_returns[str(args.benchmark_crypto)], errors="coerce").fillna(0.0).astype(float),
            benchmark_profile=crypto_profile,
        )
        benchmark_rows.append({"suite": "crypto_benchmark", "candidate_id": name, **{k: v for k, v in stats.items() if k not in {"net_ret", "benchmark_net_ret"}}})
    for name, series, profile in equity_benchmarks:
        stats = _evaluate_net(
            gross_ret=series,
            turnover=pd.Series(np.zeros(len(series), dtype=float), index=series.index, dtype=float),
            profile=profile,
            benchmark_ret=pd.to_numeric(equity_returns[str(args.benchmark_equity)], errors="coerce").fillna(0.0).astype(float),
            benchmark_profile=foreign_profile,
        )
        benchmark_rows.append({"suite": "equity_benchmark", "candidate_id": name, **{k: v for k, v in stats.items() if k not in {"net_ret", "benchmark_net_ret"}}})
    benchmark_rows.append(_result_row(meta_result))
    benchmark_rows.append(_result_row(best_crypto))
    benchmark_rows.append(_result_row(best_equities))
    benchmark_compare = pd.DataFrame(benchmark_rows)

    blocks = [
        ("bull_2020_2021", "2020-01-01", "2021-12-31"),
        ("bear_2022", "2022-01-01", "2022-12-31"),
        ("recent_2023_2024", "2023-01-01", "2024-12-31"),
        ("recent_2025_now", "2025-01-01", str(pd.Timestamp.now("UTC").date())),
    ]
    oos_rows = _oos_block_rows(best_crypto, blocks) + _oos_block_rows(best_equities, blocks) + _oos_block_rows(meta_result, blocks)
    oos_df = pd.DataFrame(oos_rows)

    crypto_df.to_csv(outdir / "crypto_results.csv", index=False)
    equities_df.to_csv(outdir / "equities_results.csv", index=False)
    meta_df.to_csv(outdir / "meta_results.csv", index=False)
    benchmark_compare.to_csv(outdir / "benchmark_compare.csv", index=False)
    oos_df.to_csv(outdir / "oos_blocks.csv", index=False)

    previous_crypto_summary = ROOT / "results/validation/profit_10x_rule_search_crypto_plus/20260307T013640Z/summary.json"
    previous_crypto = json.loads(previous_crypto_summary.read_text(encoding="utf-8"))["top_candidates"]["best_goal_score"] if previous_crypto_summary.exists() else {}
    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "outdir": str(outdir),
        "crypto_universe": {
            "tiers": {k: len(v) for k, v in crypto_tiers.items()},
            "assets_loaded": int(crypto_returns.shape[1]),
        },
        "equity_universe": {
            "groups": int(len(equity_group_map)),
            "assets_loaded": int(equity_returns.shape[1]),
        },
        "best_crypto": _result_row(best_crypto),
        "best_equities": _result_row(best_equities),
        "best_meta_switch": _result_row(meta_result),
        "improvement_vs_previous_crypto_best": {
            "previous_candidate": previous_crypto.get("candidate_id", ""),
            "previous_net_ann_return": _safe_float(previous_crypto.get("net_ann_return")),
            "previous_net_max_drawdown": _safe_float(previous_crypto.get("net_max_drawdown")),
            "new_candidate": best_crypto.candidate_id,
            "new_net_ann_return": best_crypto.net_ann_return,
            "new_net_max_drawdown": best_crypto.net_max_drawdown,
            "delta_net_ann_return": best_crypto.net_ann_return - _safe_float(previous_crypto.get("net_ann_return"), 0.0),
            "delta_max_drawdown": best_crypto.net_max_drawdown - _safe_float(previous_crypto.get("net_max_drawdown"), 0.0),
        },
        "insights": [
            f"Melhor crypto apos stops/custos/tier: {best_crypto.candidate_id} com net_ann_return={best_crypto.net_ann_return:.4f}, edge_vs_benchmark={best_crypto.edge_vs_benchmark:.4f}, mdd={best_crypto.net_max_drawdown:.4f}.",
            f"Melhor sleeve causal de equities: {best_equities.candidate_id} com net_ann_return={best_equities.net_ann_return:.4f}, edge_vs_benchmark={best_equities.edge_vs_benchmark:.4f}.",
            f"Meta-switch: {meta_result.candidate_id} com net_ann_return={meta_result.net_ann_return:.4f}, edge_vs_benchmark={meta_result.edge_vs_benchmark:.4f}.",
        ],
        "artifacts": {
            "crypto_results_csv": str(outdir / "crypto_results.csv"),
            "equities_results_csv": str(outdir / "equities_results.csv"),
            "meta_results_csv": str(outdir / "meta_results.csv"),
            "benchmark_compare_csv": str(outdir / "benchmark_compare.csv"),
            "oos_blocks_csv": str(outdir / "oos_blocks.csv"),
        },
    }
    summary_path = outdir / "summary.json"
    _write_json(summary_path, summary)

    research_rows = _research_rows([best_crypto, best_equities, meta_result], outdir=outdir, summary_path=summary_path)
    (outdir / "profit_research_rows.json").write_text(json.dumps(research_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    write_run_manifest(
        outdir,
        script="scripts/bench/validation/run_profit_frontier_expansion_suite.py",
        params={
            "crypto_asset_groups": str(args.crypto_asset_groups),
            "crypto_asset_metadata": str(args.crypto_asset_metadata),
            "equity_asset_groups": str(args.equity_asset_groups),
            "equity_asset_metadata": str(args.equity_asset_metadata),
            "prices_dir": str(args.prices_dir),
            "net_assumptions": str(args.net_assumptions),
        },
        paths={
            "crypto_asset_groups": str((ROOT / args.crypto_asset_groups).resolve()),
            "equity_asset_groups": str((ROOT / args.equity_asset_groups).resolve()),
            "prices_dir": str(prices_dir),
            "summary_json": str(summary_path),
            "crypto_results_csv": str(outdir / "crypto_results.csv"),
            "equities_results_csv": str(outdir / "equities_results.csv"),
            "meta_results_csv": str(outdir / "meta_results.csv"),
            "benchmark_compare_csv": str(outdir / "benchmark_compare.csv"),
            "oos_blocks_csv": str(outdir / "oos_blocks.csv"),
            "profit_research_rows_json": str(outdir / "profit_research_rows.json"),
        },
        extra={
            "notes": [
                "Suite integra 7 frentes: crypto stops, custos cripto, sleeve causal equities, meta-switch, tiers crypto, OOS por blocos e benchmarks duros.",
            ]
        },
    )
    print(json.dumps({"status": "ok", "outdir": str(outdir), "best_crypto": best_crypto.candidate_id, "best_meta": meta_result.candidate_id}, ensure_ascii=False))


if __name__ == "__main__":
    main()
