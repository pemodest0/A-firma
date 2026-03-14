#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.portfolio.exogenous_features import adjust_confidence_with_feature, build_exogenous_feature_panel  # noqa: E402
from engine.structural.run_manifest import write_run_manifest  # noqa: E402
from execution.net_assumptions import load_net_assumption_profiles  # noqa: E402
from scripts.ops.data_review_policy import DEFERRED_REVIEW_TICKERS  # noqa: E402
from scripts.bench.validation.run_profit_alpha_war_suite import _blended_profile  # noqa: E402
from scripts.bench.validation.run_profit_frontier_expansion_suite import (  # noqa: E402
    EQUITY_EXCLUDED,
    StrategyResult,
    _evaluate_net,
    _ensure_benchmark_columns,
    _load_asset_table,
    _load_daily_universe,
    _safe_float,
    _select_crypto_tiers,
    _simulate_asset_rule,
    _write_json,
)
from scripts.bench.validation.run_profit_investment_yearbook import (  # noqa: E402
    _calendar_rows,
    _result_row,
)
from scripts.bench.validation.run_profit_layered_engine_suite import (  # noqa: E402
    StrategyBundle,
    _build_equity_group_map,
    _load_structural_regime_series_local,
    _simulate_equity_group_sleeve_v2,
    _simulate_equity_group_sleeve_v3,
    _simulate_equity_trail_switch_bundle,
    _stress_bundle,
)
from scripts.bench.validation.run_profit_regime_simulation_suite import (  # noqa: E402
    _apply_mc_guard,
    _build_meta_v1_allocation,
)
from scripts.bench.validation.run_profit_sector_pressure_suite import (  # noqa: E402
    _research_row,
    _simulate_equity_group_sleeve_v4_sector_pressure,
)


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _candidate_bundle(
    *,
    candidate_id: str,
    bundle: StrategyBundle,
) -> StrategyBundle:
    return replace(
        bundle,
        result=replace(
            bundle.result,
            candidate_id=str(candidate_id),
        ),
    )


def _tail_return(series: pd.Series, lookback: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    min_periods = max(10, int(lookback) // 3)
    return (1.0 + values).rolling(int(lookback), min_periods=min_periods).apply(np.prod, raw=True) - 1.0


def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    if values.empty:
        return values
    min_periods = max(10, int(window) // 3)

    def _pct(arr: np.ndarray) -> float:
        arr = arr[np.isfinite(arr)]
        if arr.size <= 1:
            return float("nan")
        last = float(arr[-1])
        return float(np.mean(arr <= last))

    return values.rolling(int(window), min_periods=min_periods).apply(_pct, raw=True)


def _build_promoted_attack_confidence_score(context: dict[str, Any], attack_returns: pd.DataFrame) -> pd.Series:
    idx = (
        attack_returns.index.intersection(context["btc_prices"].index)
        .intersection(context["spy_prices"].index)
    )
    crypto_ret = pd.to_numeric(attack_returns["crypto"].reindex(idx), errors="coerce").fillna(0.0).astype(float)
    equity_ret = pd.to_numeric(attack_returns["equity"].reindex(idx), errors="coerce").fillna(0.0).astype(float)
    crypto_fast = _tail_return(crypto_ret, 21)
    equity_fast = _tail_return(equity_ret, 21)
    crypto_slow = _tail_return(crypto_ret, 63)
    btc = pd.to_numeric(context["btc_prices"].reindex(idx), errors="coerce").astype(float)
    spy = pd.to_numeric(context["spy_prices"].reindex(idx), errors="coerce").astype(float)
    btc_ok = (btc.shift(1) > btc.shift(1).rolling(200, min_periods=100).mean()).fillna(False).astype(float)
    spy_ok = (spy.shift(1) > spy.shift(1).rolling(200, min_periods=100).mean()).fillna(False).astype(float)
    regime = (
        pd.Series(context["regime_series"], index=context["regime_series"].index)
        .reindex(idx)
        .ffill()
        .bfill()
        .fillna("stable")
        .astype(str)
        .str.lower()
    )
    structural_clean = regime.isin(["stable", "dispersion"]).astype(float)
    spread_fast = ((crypto_fast - equity_fast + 0.08) / 0.16).clip(0.0, 1.0)
    spread_slow = ((crypto_slow + 0.10) / 0.20).clip(0.0, 1.0)
    raw_score = 0.12 + 0.26 * structural_clean + 0.24 * btc_ok + 0.10 * spy_ok + 0.18 * spread_fast + 0.10 * spread_slow
    return _rolling_percentile(raw_score.clip(0.0, 1.0), 126).fillna(raw_score).clip(0.0, 1.0).astype(float)


def _confidence_weight_from_score(score: pd.Series) -> pd.Series:
    weight = pd.Series(0.15, index=score.index, dtype=float)
    clean = pd.to_numeric(score, errors="coerce").fillna(0.0).astype(float)
    weight.loc[clean >= 0.48] = 0.75
    weight.loc[clean >= 0.63] = 1.00
    return weight.clip(0.0, 1.0)


def _blend_allocation_bundles(
    *,
    candidate_id: str,
    notes: str,
    attack_alloc: "AllocationBundle",
    protect_alloc: "AllocationBundle",
    attack_weight: pd.Series,
) -> "AllocationBundle":
    idx = (
        attack_alloc.bundle.result.gross_ret.index.intersection(protect_alloc.bundle.result.gross_ret.index)
        .intersection(attack_alloc.weights.index)
        .intersection(protect_alloc.weights.index)
        .intersection(attack_weight.index)
    )
    aw = pd.to_numeric(attack_weight.reindex(idx), errors="coerce").fillna(0.0).clip(0.0, 1.0).astype(float)
    pw = 1.0 - aw
    attack_gross = pd.to_numeric(attack_alloc.bundle.result.gross_ret.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    protect_gross = pd.to_numeric(protect_alloc.bundle.result.gross_ret.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    gross = aw * attack_gross + pw * protect_gross

    attack_w = attack_alloc.weights.reindex(idx).fillna(0.0).astype(float)
    protect_w = protect_alloc.weights.reindex(idx).fillna(0.0).astype(float)
    weights = attack_w.mul(aw, axis=0).add(protect_w.mul(pw, axis=0), fill_value=0.0)
    weights["cash"] = 1.0 - weights[["crypto", "equity"]].sum(axis=1)
    weights["cash"] = weights["cash"].clip(lower=0.0, upper=1.0)
    turnover = (
        weights[["crypto", "equity", "cash"]]
        .diff()
        .abs()
        .sum(axis=1)
        .fillna(weights[["crypto", "equity", "cash"]].abs().sum(axis=1))
        / 2.0
    )
    benchmark = pd.to_numeric(attack_alloc.bundle.benchmark_gross_ret.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    perf = _evaluate_net(
        gross_ret=gross,
        turnover=turnover,
        profile=attack_alloc.bundle.profile,
        benchmark_ret=benchmark,
        benchmark_profile=attack_alloc.bundle.benchmark_profile,
    )
    result = StrategyResult(
        suite="alpha_hardening",
        candidate_id=str(candidate_id),
        family="promoted_attack",
        benchmark_ticker=attack_alloc.bundle.result.benchmark_ticker,
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
        notes=str(notes),
    )
    bundle = StrategyBundle(
        result=result,
        benchmark_gross_ret=benchmark.reindex(result.gross_ret.index).fillna(0.0).astype(float),
        profile=attack_alloc.bundle.profile,
        benchmark_profile=attack_alloc.bundle.benchmark_profile,
    )
    source = pd.Series(np.where(aw >= 0.5, "attack", "protect"), index=idx, dtype=object)
    return AllocationBundle(bundle=bundle, weights=weights, source=source)


def _build_alpha_meta_allocation_bundle(
    *,
    candidate_id: str,
    crypto_bundle: StrategyBundle,
    equity_bundle: StrategyBundle,
    btc_prices: pd.Series,
    spy_prices: pd.Series,
    profile,
    entry_lookback: int,
    exit_lookback: int,
    entry_margin: float,
    exit_margin: float,
    risk_off_mode: str,
    min_crypto_hold_days: int,
) -> AllocationBundle:
    idx = (
        crypto_bundle.result.gross_ret.index.intersection(equity_bundle.result.gross_ret.index)
        .intersection(btc_prices.index)
        .intersection(spy_prices.index)
    )
    crypto_ret = pd.to_numeric(crypto_bundle.result.gross_ret.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    equity_ret = pd.to_numeric(equity_bundle.result.gross_ret.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    btc = pd.to_numeric(btc_prices.reindex(idx), errors="coerce").astype(float)
    spy = pd.to_numeric(spy_prices.reindex(idx), errors="coerce").astype(float)
    btc_ok = (btc.shift(1) > btc.shift(1).rolling(200, min_periods=100).mean()).fillna(False)
    spy_ok = (spy.shift(1) > spy.shift(1).rolling(200, min_periods=100).mean()).fillna(False)
    crypto_entry = _tail_return(crypto_ret, int(entry_lookback))
    equity_entry = _tail_return(equity_ret, int(entry_lookback))
    crypto_exit = _tail_return(crypto_ret, int(exit_lookback))
    equity_exit = _tail_return(equity_ret, int(exit_lookback))

    weights = pd.DataFrame(0.0, index=idx, columns=["crypto", "equity", "cash"], dtype=float)
    source = pd.Series(index=idx, dtype=object)
    state = "cash"
    hold_days = 0
    for dt in idx:
        btc_good = bool(btc_ok.loc[dt])
        spy_good = bool(spy_ok.loc[dt])
        both_bad = not btc_good and not spy_good
        ce = float(crypto_entry.loc[dt]) if pd.notna(crypto_entry.loc[dt]) else -1.0
        ee = float(equity_entry.loc[dt]) if pd.notna(equity_entry.loc[dt]) else -1.0
        cx = float(crypto_exit.loc[dt]) if pd.notna(crypto_exit.loc[dt]) else -1.0
        ex = float(equity_exit.loc[dt]) if pd.notna(equity_exit.loc[dt]) else -1.0
        choose = state
        if both_bad:
            if risk_off_mode == "equity25":
                weights.loc[dt, "equity"] = 0.25
                weights.loc[dt, "cash"] = 0.75
                choose = "equity25"
            elif risk_off_mode == "equity50":
                weights.loc[dt, "equity"] = 0.50
                weights.loc[dt, "cash"] = 0.50
                choose = "equity50"
            else:
                weights.loc[dt, "cash"] = 1.0
                choose = "cash"
            state = choose
            hold_days = 0
            source.loc[dt] = choose
            continue
        enter_crypto = btc_good and (ce > ee + float(entry_margin))
        exit_crypto = spy_good and (ex >= cx + float(exit_margin))
        if state == "crypto":
            hold_days += 1
            if hold_days < int(min_crypto_hold_days) and btc_good:
                choose = "crypto"
            elif not btc_good and spy_good:
                choose = "equity"
                hold_days = 0
            elif exit_crypto:
                choose = "equity"
                hold_days = 0
            else:
                choose = "crypto"
        else:
            if enter_crypto:
                choose = "crypto"
                hold_days = 1
            elif spy_good:
                choose = "equity"
                hold_days = 0
            elif btc_good:
                choose = "crypto"
                hold_days = 1
            else:
                choose = "cash"
                hold_days = 0
        if choose == "crypto":
            weights.loc[dt, "crypto"] = 1.0
        elif choose == "equity":
            weights.loc[dt, "equity"] = 1.0
        elif choose == "equity25":
            weights.loc[dt, "equity"] = 0.25
            weights.loc[dt, "cash"] = 0.75
        elif choose == "equity50":
            weights.loc[dt, "equity"] = 0.50
            weights.loc[dt, "cash"] = 0.50
        else:
            weights.loc[dt, "cash"] = 1.0
        state = choose
        source.loc[dt] = choose

    bench = 0.5 * btc.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float) + 0.5 * spy.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    from scripts.bench.validation.run_profit_regime_simulation_suite import _evaluate_allocation_candidate  # noqa: E402

    bundle = _evaluate_allocation_candidate(
        candidate_id=candidate_id,
        family="meta_switch_alpha_search",
        weights=weights,
        crypto_ret=crypto_ret,
        equity_ret=equity_ret,
        benchmark_ret=bench,
        profile=profile,
        benchmark_profile=profile,
        notes=(
            f"entry_lb={entry_lookback};exit_lb={exit_lookback};entry_margin={entry_margin:.2f};"
            f"exit_margin={exit_margin:.2f};risk_off={risk_off_mode};hold={min_crypto_hold_days}"
        ),
    )
    return AllocationBundle(
        bundle=bundle,
        weights=weights.reindex(bundle.result.gross_ret.index).fillna(0.0),
        source=source.reindex(bundle.result.gross_ret.index).fillna("cash"),
    )


def _build_candidates(
    *,
    prices_dir: Path,
    crypto_groups: Path,
    crypto_meta: Path,
    equity_groups: Path,
    equity_meta: Path,
    benchmark_crypto: str,
    benchmark_equity: str,
):
    profiles = load_net_assumption_profiles(ROOT / "config" / "profit_net_assumptions.json")
    foreign_profile = profiles["profiles"]["foreign_financial_brazil_resident"]
    crypto_profile = profiles["profiles"]["crypto_global_brazil_resident_conservative"]
    blended_profile = _blended_profile(
        crypto_profile,
        foreign_profile,
        profile_id="alpha_hardening_blended",
        label="Alpha hardening blended",
    )

    crypto_assets = _load_asset_table(crypto_groups, crypto_meta)
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
        [str(benchmark_crypto), "ETH-USD"],
    )
    crypto_tiers = _select_crypto_tiers(crypto_assets, crypto_viability)

    equity_assets = _load_asset_table(equity_groups, equity_meta)
    equity_assets = equity_assets[~equity_assets["asset_group"].astype(str).isin(EQUITY_EXCLUDED)].copy()
    ticker_col = "ticker" if "ticker" in equity_assets.columns else "asset_id"
    equity_assets = equity_assets[~equity_assets[ticker_col].astype(str).isin(DEFERRED_REVIEW_TICKERS)].copy()
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
        [str(benchmark_equity)],
    )
    equity_group_map = _build_equity_group_map(equity_assets, equity_returns)
    regime_series = _load_structural_regime_series_local(ROOT)

    eq_a2 = _simulate_equity_group_sleeve_v2(
        candidate_id="equities_v2__slow189__g4__a1",
        returns=equity_returns,
        prices=equity_prices,
        asset_table=equity_assets,
        equity_groups=equity_group_map,
        benchmark_ticker=str(benchmark_equity),
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
        benchmark_ticker=str(benchmark_equity),
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
    eq_sp, _ = _simulate_equity_group_sleeve_v4_sector_pressure(
        candidate_id="equities_v4__sector_pressure_p25",
        returns=equity_returns,
        prices=equity_prices,
        asset_table=equity_assets,
        equity_groups=equity_group_map,
        benchmark_ticker=str(benchmark_equity),
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
        pressure_penalty=0.25,
        profile=foreign_profile,
        benchmark_profile=foreign_profile,
    )
    if eq_a2 is None or eq_r1 is None or eq_sp is None:
        raise SystemExit("falha ao reconstruir sleeves de equities")

    equity_base = _simulate_equity_trail_switch_bundle(
        candidate_id="equities_meta__trail_switch__a2__r1",
        aggressive_bundle=eq_a2,
        robust_bundle=eq_r1,
        regime_series=regime_series,
        spy_prices=pd.to_numeric(equity_prices[str(benchmark_equity)], errors="coerce"),
    )
    equity_attack = _simulate_equity_trail_switch_bundle(
        candidate_id="equities_meta__trail_switch__sector_p25",
        aggressive_bundle=eq_sp,
        robust_bundle=eq_r1,
        regime_series=regime_series,
        spy_prices=pd.to_numeric(equity_prices[str(benchmark_equity)], errors="coerce"),
    )

    baseline_crypto_result = _simulate_asset_rule(
        candidate_id="crypto_major8__mom_vol_adj_lb021_rb07_k3",
        family="crypto_major8_search",
        allowed_tickers=crypto_tiers["crypto_major8"],
        returns=crypto_returns,
        prices=crypto_prices,
        asset_table=crypto_assets,
        benchmark_ticker=str(benchmark_crypto),
        fallback_ticker=str(benchmark_crypto),
        score_mode="mom_vol_adj",
        lookback_days=21,
        rebalance_days=7,
        top_k=3,
        asset_ma_days=0,
        market_ma_days=200,
        relative_to_benchmark=False,
        skip_recent_days=0,
        trailing_stop_dd=None,
        hard_stop_loss=None,
        stop_to_cash=True,
        profile=crypto_profile,
        benchmark_profile=crypto_profile,
    )
    attack_crypto_result = _simulate_asset_rule(
        candidate_id="crypto_major8__mom_total_lb021_rb07_k1",
        family="crypto_major8_search",
        allowed_tickers=crypto_tiers["crypto_major8"],
        returns=crypto_returns,
        prices=crypto_prices,
        asset_table=crypto_assets,
        benchmark_ticker=str(benchmark_crypto),
        fallback_ticker=str(benchmark_crypto),
        score_mode="mom_total",
        lookback_days=21,
        rebalance_days=7,
        top_k=1,
        asset_ma_days=0,
        market_ma_days=200,
        relative_to_benchmark=False,
        skip_recent_days=0,
        trailing_stop_dd=None,
        hard_stop_loss=None,
        stop_to_cash=True,
        profile=crypto_profile,
        benchmark_profile=crypto_profile,
    )
    if baseline_crypto_result is None or attack_crypto_result is None:
        raise SystemExit("falha ao reconstruir sleeves de cripto")

    baseline_crypto_bundle = StrategyBundle(
        result=baseline_crypto_result,
        benchmark_gross_ret=pd.to_numeric(crypto_returns[str(benchmark_crypto)], errors="coerce").reindex(baseline_crypto_result.gross_ret.index).fillna(0.0).astype(float),
        profile=crypto_profile,
        benchmark_profile=crypto_profile,
    )
    attack_crypto_bundle = StrategyBundle(
        result=attack_crypto_result,
        benchmark_gross_ret=pd.to_numeric(crypto_returns[str(benchmark_crypto)], errors="coerce").reindex(attack_crypto_result.gross_ret.index).fillna(0.0).astype(float),
        profile=crypto_profile,
        benchmark_profile=crypto_profile,
    )

    btc_prices = pd.to_numeric(crypto_prices[str(benchmark_crypto)], errors="coerce")
    spy_prices = pd.to_numeric(equity_prices[str(benchmark_equity)], errors="coerce")

    baseline = _build_meta_v1_allocation(
        crypto_bundle=baseline_crypto_bundle,
        equity_bundle=equity_base,
        btc_prices=btc_prices,
        spy_prices=spy_prices,
        profile=blended_profile,
    )
    baseline = replace(
        baseline,
        bundle=_candidate_bundle(candidate_id="meta_major8_eq_a2r1", bundle=baseline.bundle),
    )

    base_returns = pd.concat(
        {
            "crypto": pd.to_numeric(baseline_crypto_bundle.result.gross_ret, errors="coerce"),
            "equity": pd.to_numeric(equity_base.result.gross_ret, errors="coerce"),
        },
        axis=1,
        sort=False,
    ).dropna(how="all")
    baseline_guard, base_mc_summary = _apply_mc_guard(
        candidate_id="meta_major8_eq_a2r1_mc_guard",
        base=baseline,
        returns=base_returns,
        regime=regime_series,
        profile=blended_profile,
        lookback=252,
        horizon=21,
        n_paths=400,
        step=42,
    )

    raw_attack = _build_alpha_meta_allocation_bundle(
        candidate_id="alpha_attack_major8_equity25_raw",
        crypto_bundle=attack_crypto_bundle,
        equity_bundle=equity_attack,
        btc_prices=btc_prices,
        spy_prices=spy_prices,
        profile=blended_profile,
        entry_lookback=14,
        exit_lookback=63,
        entry_margin=0.02,
        exit_margin=0.05,
        risk_off_mode="equity25",
        min_crypto_hold_days=0,
    )
    attack_score = _build_promoted_attack_confidence_score(
        {
            "btc_prices": btc_prices,
            "spy_prices": spy_prices,
            "regime_series": regime_series,
        },
        pd.concat(
            {
                "crypto": pd.to_numeric(attack_crypto_bundle.result.gross_ret, errors="coerce"),
                "equity": pd.to_numeric(equity_attack.result.gross_ret, errors="coerce"),
            },
            axis=1,
            sort=False,
        ).dropna(how="all"),
    )
    attack_legacy = _blend_allocation_bundles(
        candidate_id="alpha_attack_major8_equity25",
        notes=(
            "modo ataque promovido com entrada cripto mais rapida e sizing por confianca relativa "
            "ao historico recente"
        ),
        attack_alloc=raw_attack,
        protect_alloc=baseline_guard,
        attack_weight=_confidence_weight_from_score(attack_score),
    )

    exogenous_panel = build_exogenous_feature_panel(
        prices_dir=prices_dir,
        crypto_returns=crypto_returns,
        crypto_prices=crypto_prices,
        benchmark_crypto=str(benchmark_crypto),
        macro_index=attack_score.index,
    )
    attack_score_exogenous = adjust_confidence_with_feature(
        base_score=attack_score,
        feature=exogenous_panel.panel.get("liquidation"),
        mode="penalty",
        weight=0.14,
    )
    attack = _blend_allocation_bundles(
        candidate_id="alpha_attack_major8_equity25",
        notes=(
            "modo ataque promovido com entrada cripto mais rapida, sizing por confianca relativa "
            "ao historico recente e overlay de liquidacao cripto"
        ),
        attack_alloc=raw_attack,
        protect_alloc=baseline_guard,
        attack_weight=_confidence_weight_from_score(attack_score_exogenous),
    )

    attack_returns = pd.concat(
        {
            "crypto": pd.to_numeric(attack_crypto_bundle.result.gross_ret, errors="coerce"),
            "equity": pd.to_numeric(equity_attack.result.gross_ret, errors="coerce"),
        },
        axis=1,
        sort=False,
    ).dropna(how="all")
    attack_guard, attack_mc_summary = _apply_mc_guard(
        candidate_id="alpha_attack_major8_equity25_mc_guard",
        base=attack,
        returns=attack_returns,
        regime=regime_series,
        profile=blended_profile,
        lookback=252,
        horizon=21,
        n_paths=400,
        step=42,
    )
    return {
        "baseline": baseline.bundle,
        "attack": attack.bundle,
        "baseline_guard": baseline_guard.bundle,
        "attack_guard": attack_guard.bundle,
        "allocations": {
            "baseline": baseline,
            "attack": attack,
            "attack_legacy": attack_legacy,
            "baseline_guard": baseline_guard,
            "attack_guard": attack_guard,
        },
        "sleeve_returns": {
            "baseline": base_returns,
            "baseline_guard": base_returns,
            "attack": attack_returns,
            "attack_legacy": attack_returns,
            "attack_guard": attack_returns,
        },
        "mc_summaries": {
            "baseline": base_mc_summary,
            "attack": attack_mc_summary,
        },
        "context": {
            "profiles": {
                "crypto": crypto_profile,
                "foreign": foreign_profile,
                "blended": blended_profile,
            },
            "crypto_assets": crypto_assets,
            "crypto_returns": crypto_returns,
            "crypto_prices": crypto_prices,
            "crypto_tiers": crypto_tiers,
            "equity_assets": equity_assets,
            "equity_returns": equity_returns,
            "equity_prices": equity_prices,
            "equity_base": equity_base,
            "equity_attack": equity_attack,
            "btc_prices": btc_prices,
            "spy_prices": spy_prices,
            "benchmark_crypto": str(benchmark_crypto),
            "benchmark_equity": str(benchmark_equity),
            "regime_series": regime_series,
            "attack_score_legacy": attack_score,
            "attack_score_exogenous": attack_score_exogenous,
            "exogenous_panel": exogenous_panel.panel,
        },
        "research_rows": [
            _research_row(baseline.bundle.result, outdir=ROOT / "results" / "validation", status="keep", methodology="alpha_hardening_baseline", label="Base atual de lucro"),
            _research_row(attack.bundle.result, outdir=ROOT / "results" / "validation", status="watch", methodology="alpha_hardening_attack", label="Modo ataque focado em lucro"),
            _research_row(attack_legacy.bundle.result, outdir=ROOT / "results" / "validation", status="watch", methodology="alpha_hardening_attack_legacy", label="Modo ataque anterior"),
            _research_row(baseline_guard.bundle.result, outdir=ROOT / "results" / "validation", status="watch", methodology="alpha_hardening_balanced", label="Base com guard monte carlo"),
            _research_row(attack_guard.bundle.result, outdir=ROOT / "results" / "validation", status="watch", methodology="alpha_hardening_attack_guard", label="Modo ataque com guard monte carlo"),
        ],
    }


class AllocationBundle:
    def __init__(self, *, bundle: StrategyBundle, weights: pd.DataFrame, source: pd.Series) -> None:
        self.bundle = bundle
        self.weights = weights
        self.source = source


def main() -> None:
    ap = argparse.ArgumentParser(description="Teste duro do modo de lucro maximo contra atual e guard.")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--capital-brl", type=float, default=10000.0)
    ap.add_argument("--outdir-root", default="results/validation/profit_alpha_hardening_suite")
    args = ap.parse_args()

    outdir = (ROOT / args.outdir_root / _run_id()).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    built = _build_candidates(
        prices_dir=(ROOT / args.prices_dir).resolve(),
        crypto_groups=(ROOT / args.crypto_asset_groups).resolve(),
        crypto_meta=(ROOT / args.crypto_asset_metadata).resolve(),
        equity_groups=(ROOT / args.equity_asset_groups).resolve(),
        equity_meta=(ROOT / args.equity_asset_metadata).resolve(),
        benchmark_crypto=str(args.benchmark_crypto),
        benchmark_equity=str(args.benchmark_equity),
    )

    bundles = [
        built["baseline"],
        built["attack"],
        built["baseline_guard"],
        built["attack_guard"],
    ]

    profiles = load_net_assumption_profiles(ROOT / "config" / "profit_net_assumptions.json")
    foreign_profile = profiles["profiles"]["foreign_financial_brazil_resident"]
    crypto_profile = profiles["profiles"]["crypto_global_brazil_resident_conservative"]
    blended_profile = _blended_profile(
        crypto_profile,
        foreign_profile,
        profile_id="alpha_hardening_blended",
        label="Alpha hardening blended",
    )
    hard_profile = replace(
        blended_profile,
        profile_id="alpha_hardening_hard",
        label="Alpha hardening hard",
        transaction_cost_bps_assumed=blended_profile.transaction_cost_bps_assumed + 20.0,
    )
    brutal_profile = replace(
        blended_profile,
        profile_id="alpha_hardening_brutal",
        label="Alpha hardening brutal",
        transaction_cost_bps_assumed=blended_profile.transaction_cost_bps_assumed + 40.0,
    )

    stress_rows: list[dict[str, Any]] = []
    calendar_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for bundle in bundles:
        summary_rows.append(_result_row(bundle.result))
        for profile, label, delay in [
            (bundle.profile, "base", 0),
            (bundle.profile, "delay_d1", 1),
            (bundle.profile, "delay_d2", 2),
            (hard_profile, "hard_cost", 0),
            (hard_profile, "hard_cost_delay_d1", 1),
            (brutal_profile, "brutal_cost", 0),
            (brutal_profile, "brutal_cost_delay_d1", 1),
        ]:
            stress_rows.append(
                _stress_bundle(
                    bundle,
                    delay_days=delay,
                    profile=profile,
                    benchmark_profile=profile,
                    label=label,
                )
            )
        calendar_rows.extend(_calendar_rows(result=bundle.result, capital_brl=float(args.capital_brl)))

    candidate_df = pd.DataFrame(summary_rows).sort_values(
        ["net_total_return", "net_ann_return", "net_sharpe"],
        ascending=[False, False, False],
    )
    stress_df = pd.DataFrame(stress_rows).sort_values(["stress_label", "net_total_return"], ascending=[True, False])
    calendar_df = pd.DataFrame(calendar_rows).sort_values(["year", "profit_brl"], ascending=[True, False])

    candidate_df.to_csv(outdir / "candidate_compare.csv", index=False)
    stress_df.to_csv(outdir / "stress_compare.csv", index=False)
    calendar_df.to_csv(outdir / "calendar_year_compare.csv", index=False)
    built["mc_summaries"]["baseline"].to_csv(outdir / "baseline_mc_guard_summary.csv", index=True)
    built["mc_summaries"]["attack"].to_csv(outdir / "attack_mc_guard_summary.csv", index=True)
    (outdir / "profit_research_rows.json").write_text(
        json.dumps(built["research_rows"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    best_base = candidate_df.iloc[0].to_dict() if not candidate_df.empty else {}
    stress_winners = (
        stress_df.sort_values(["stress_label", "net_total_return"], ascending=[True, False])
        .groupby("stress_label", as_index=False)
        .head(1)
        .reset_index(drop=True)
        .to_dict(orient="records")
    )
    yearly_winners = (
        calendar_df.sort_values(["year", "profit_brl"], ascending=[True, False])
        .groupby("year", as_index=False)
        .head(1)
        .reset_index(drop=True)
        .to_dict(orient="records")
    )
    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "outdir": str(outdir),
        "base_ranking": candidate_df.to_dict(orient="records"),
        "best_base_candidate": best_base,
        "stress_winners": stress_winners,
        "yearly_winners": yearly_winners,
        "insights": [
            "A bateria confronta o novo modo de lucro com o atual e duas versões protegidas.",
            "Os testes duros aplicam atraso de execução e custo mais alto para verificar se o ganho final sobrevive.",
            "A leitura ano a ano mostra se o ganho veio de poucos anos extraordinários ou de melhora mais espalhada.",
        ],
        "artifacts": {
            "candidate_compare_csv": str(outdir / "candidate_compare.csv"),
            "stress_compare_csv": str(outdir / "stress_compare.csv"),
            "calendar_year_compare_csv": str(outdir / "calendar_year_compare.csv"),
            "profit_research_rows_json": str(outdir / "profit_research_rows.json"),
        },
    }
    _write_json(outdir / "summary.json", summary)
    write_run_manifest(
        outdir,
        script="scripts/bench/validation/run_profit_alpha_hardening_suite.py",
        params={
            "benchmark_crypto": args.benchmark_crypto,
            "benchmark_equity": args.benchmark_equity,
            "capital_brl": args.capital_brl,
            "stress_labels": stress_df["stress_label"].drop_duplicates().tolist(),
        },
        paths=summary["artifacts"],
        extra={"summary_json": str(outdir / "summary.json")},
    )


if __name__ == "__main__":
    main()
