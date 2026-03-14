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

from engine.portfolio import (  # noqa: E402
    build_hmm_feature_frame,
    estimate_regime_moments,
    estimate_transition_matrix,
    fit_hmm_challenger,
    hrp_weights,
    rolling_regime_conditioned_summary,
    simulate_regime_conditioned_paths,
    summarize_portfolio_distribution,
)
from engine.structural.ground_truth import (  # noqa: E402
    build_event_label,
    build_regime_future_event_label,
    classification_report_binary,
)
from engine.structural.run_manifest import write_run_manifest  # noqa: E402
from execution.cost_model import summarize_return_series  # noqa: E402
from execution.net_assumptions import NetAssumptionProfile, load_net_assumption_profiles  # noqa: E402
from scripts.bench.validation.run_profit_frontier_expansion_suite import (  # noqa: E402
    EQUITY_EXCLUDED,
    StrategyResult,
    _build_equity_group_map,
    _ensure_benchmark_columns,
    _evaluate_net,
    _load_asset_table,
    _load_daily_universe,
    _rolling_ten_x_stats,
    _run_id,
    _safe_float,
    _select_crypto_tiers,
    _simulate_asset_rule,
    _write_json,
)
from scripts.bench.validation.run_profit_layered_engine_suite import (  # noqa: E402
    StrategyBundle,
    _load_structural_regime_series_local,
    _regime_forward_fill_local,
    _simulate_equity_group_sleeve_v2,
    _simulate_equity_group_sleeve_v3,
    _simulate_equity_trail_switch_bundle,
)


@dataclass(frozen=True)
class AllocationBundle:
    bundle: StrategyBundle
    weights: pd.DataFrame
    source: pd.Series


def _blended_profile(
    crypto_profile: NetAssumptionProfile,
    equity_profile: NetAssumptionProfile,
    *,
    profile_id: str,
    label: str,
) -> NetAssumptionProfile:
    return NetAssumptionProfile(
        profile_id=profile_id,
        label=label,
        jurisdiction="blended",
        transaction_cost_bps_assumed=0.5 * crypto_profile.transaction_cost_bps_assumed
        + 0.5 * equity_profile.transaction_cost_bps_assumed,
        fx_spread_bps_assumed=0.5 * crypto_profile.fx_spread_bps_assumed + 0.5 * equity_profile.fx_spread_bps_assumed,
        capital_gains_tax_rate=0.5 * crypto_profile.capital_gains_tax_rate + 0.5 * equity_profile.capital_gains_tax_rate,
        tax_timing="monthly_positive_proxy",
        dividend_withholding_mode="not_applicable",
        monthly_sales_exemption_modeled=False,
        notes=("blended meta profile",),
    )


def _evaluate_allocation_candidate(
    *,
    candidate_id: str,
    family: str,
    weights: pd.DataFrame,
    crypto_ret: pd.Series,
    equity_ret: pd.Series,
    benchmark_ret: pd.Series,
    profile: NetAssumptionProfile,
    benchmark_profile: NetAssumptionProfile,
    notes: str,
) -> StrategyBundle:
    idx = weights.index.intersection(crypto_ret.index).intersection(equity_ret.index).intersection(benchmark_ret.index)
    w = weights.reindex(idx).fillna(0.0).astype(float)
    sleeve = pd.concat(
        {
            "crypto": pd.to_numeric(crypto_ret.reindex(idx), errors="coerce").fillna(0.0).astype(float),
            "equity": pd.to_numeric(equity_ret.reindex(idx), errors="coerce").fillna(0.0).astype(float),
        },
        axis=1,
    )
    gross = pd.Series((sleeve * w[["crypto", "equity"]]).sum(axis=1), index=idx, dtype=float)
    turnover = w[["crypto", "equity", "cash"]].diff().abs().sum(axis=1).fillna(w[["crypto", "equity", "cash"]].abs().sum(axis=1)) / 2.0
    perf = _evaluate_net(
        gross_ret=gross,
        turnover=turnover,
        profile=profile,
        benchmark_ret=benchmark_ret.reindex(idx).fillna(0.0).astype(float),
        benchmark_profile=benchmark_profile,
    )
    hit5 = _rolling_ten_x_stats(perf["net_ret"], horizon_days=1260)
    wealth = (1.0 + perf["net_ret"]).cumprod()
    hit_full = wealth[wealth >= 10.0]
    years_to_10x = float((hit_full.index[0] - wealth.index[0]).days / 365.25) if not hit_full.empty else float("nan")
    result = StrategyResult(
        suite="regime_simulation",
        candidate_id=candidate_id,
        family=family,
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
        notes=notes,
    )
    return StrategyBundle(
        result=result,
        benchmark_gross_ret=benchmark_ret.reindex(idx).fillna(0.0).astype(float),
        profile=profile,
        benchmark_profile=benchmark_profile,
    )


def _build_meta_v1_allocation(
    *,
    crypto_bundle: StrategyBundle,
    equity_bundle: StrategyBundle,
    btc_prices: pd.Series,
    spy_prices: pd.Series,
    profile: NetAssumptionProfile,
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
    crypto_signal_ret = crypto_ret.shift(1).fillna(0.0)
    equity_signal_ret = equity_ret.shift(1).fillna(0.0)
    crypto_trail = (1.0 + crypto_signal_ret).rolling(63, min_periods=21).apply(np.prod, raw=True) - 1.0
    equity_trail = (1.0 + equity_signal_ret).rolling(63, min_periods=21).apply(np.prod, raw=True) - 1.0
    weights = pd.DataFrame(0.0, index=idx, columns=["crypto", "equity", "cash"], dtype=float)
    source = pd.Series(index=idx, dtype=object)
    for dt in idx:
        prefer_crypto = bool(btc_ok.loc[dt]) and _safe_float(crypto_trail.loc[dt], -1.0) > _safe_float(equity_trail.loc[dt], -1.0)
        if not bool(spy_ok.loc[dt]) and not bool(btc_ok.loc[dt]):
            source.loc[dt] = "cash"
            weights.loc[dt, "cash"] = 1.0
        elif prefer_crypto:
            source.loc[dt] = "crypto"
            weights.loc[dt, "crypto"] = 1.0
        else:
            source.loc[dt] = "equity"
            weights.loc[dt, "equity"] = 1.0
    btc_bench = btc.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    spy_bench = spy.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    bench = 0.5 * btc_bench + 0.5 * spy_bench
    bundle = _evaluate_allocation_candidate(
        candidate_id="meta_v1__btc63_vs_equity",
        family="meta_switch",
        weights=weights,
        crypto_ret=crypto_ret,
        equity_ret=equity_ret,
        benchmark_ret=bench,
        profile=profile,
        benchmark_profile=profile,
        notes="crypto if BTC risk-on and trailing 63d beats equities; else equities; cash if both BTC and SPY below MM200.",
    )
    return AllocationBundle(bundle=bundle, weights=weights, source=source)


def _build_meta_hrp_allocation(
    *,
    crypto_bundle: StrategyBundle,
    equity_bundle: StrategyBundle,
    btc_prices: pd.Series,
    spy_prices: pd.Series,
    profile: NetAssumptionProfile,
    lookback: int = 63,
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
    crypto_signal_ret = crypto_ret.shift(1).fillna(0.0)
    equity_signal_ret = equity_ret.shift(1).fillna(0.0)
    crypto_trail = (1.0 + crypto_signal_ret).rolling(63, min_periods=21).apply(np.prod, raw=True) - 1.0
    equity_trail = (1.0 + equity_signal_ret).rolling(63, min_periods=21).apply(np.prod, raw=True) - 1.0
    sleeves = pd.concat({"crypto": crypto_signal_ret, "equity": equity_signal_ret}, axis=1)
    weights = pd.DataFrame(0.0, index=idx, columns=["crypto", "equity", "cash"], dtype=float)
    source = pd.Series(index=idx, dtype=object)
    for pos, dt in enumerate(idx):
        active_crypto = bool(btc_ok.loc[dt]) and _safe_float(crypto_trail.loc[dt], -1.0) > 0.0
        active_equity = bool(spy_ok.loc[dt]) and _safe_float(equity_trail.loc[dt], -1.0) > 0.0
        if not active_crypto and not active_equity:
            weights.loc[dt, "cash"] = 1.0
            source.loc[dt] = "cash"
            continue
        if active_crypto and not active_equity:
            weights.loc[dt, "crypto"] = 1.0
            source.loc[dt] = "crypto"
            continue
        if active_equity and not active_crypto:
            weights.loc[dt, "equity"] = 1.0
            source.loc[dt] = "equity"
            continue
        hist = sleeves.iloc[max(0, pos - int(lookback)) : pos].dropna(how="any")
        if len(hist) < 20:
            weights.loc[dt, ["crypto", "equity"]] = [0.5, 0.5]
        else:
            cov = hist.cov(min_periods=2).fillna(0.0)
            corr = hist.corr().fillna(0.0)
            w = hrp_weights(cov, corr=corr)
            weights.loc[dt, "crypto"] = float(w.get("crypto", 0.5))
            weights.loc[dt, "equity"] = float(w.get("equity", 0.5))
        source.loc[dt] = "hrp_blend"
    btc_bench = btc.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    spy_bench = spy.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    bench = 0.5 * btc_bench + 0.5 * spy_bench
    bundle = _evaluate_allocation_candidate(
        candidate_id="meta_hrp__btc63_equity63",
        family="meta_switch_hrp",
        weights=weights,
        crypto_ret=crypto_ret,
        equity_ret=equity_ret,
        benchmark_ret=bench,
        profile=profile,
        benchmark_profile=profile,
        notes="when crypto and equities are both active, allocate by rolling HRP between sleeves; otherwise keep the active sleeve or cash.",
    )
    return AllocationBundle(bundle=bundle, weights=weights, source=source)


def _apply_mc_guard(
    *,
    candidate_id: str,
    base: AllocationBundle,
    returns: pd.DataFrame,
    regime: pd.Series,
    profile: NetAssumptionProfile,
    lookback: int = 252,
    horizon: int = 21,
    n_paths: int = 400,
    step: int = 42,
) -> tuple[AllocationBundle, pd.DataFrame]:
    weights = base.weights[["crypto", "equity"]].copy()
    ret_frame = returns.reindex(base.weights.index).dropna(how="all")
    reg = pd.Series(regime, index=returns.index).reindex(ret_frame.index).ffill().fillna("stable")
    summary = rolling_regime_conditioned_summary(
        ret_frame,
        reg,
        weights,
        lookback=int(lookback),
        horizon=int(horizon),
        n_paths=int(n_paths),
        step=int(step),
        min_obs=20,
        random_state=17,
    )
    if summary.empty:
        guarded = base.weights.copy()
        bundle = _evaluate_allocation_candidate(
            candidate_id=candidate_id,
            family="meta_switch_mc_guard",
            weights=guarded,
            crypto_ret=returns["crypto"],
            equity_ret=returns["equity"],
            benchmark_ret=base.bundle.benchmark_gross_ret,
            profile=profile,
            benchmark_profile=profile,
            notes=f"fallback_mc_guard;base={base.bundle.result.candidate_id}",
        )
        return AllocationBundle(bundle=bundle, weights=guarded, source=base.source), summary
    scale = pd.Series(1.0, index=base.weights.index, dtype=float)
    scale = scale.where(summary["terminal_p05"] > -0.30, 0.25)
    scale = scale.where(summary["terminal_p05"] > -0.20, 0.50)
    scale = scale.where(summary["terminal_p05"] > -0.12, 0.75)
    stress_cap = np.where(pd.Series(regime, index=base.weights.index).astype(str).str.lower().eq("stress"), 0.70, 1.0)
    scale = pd.Series(np.minimum(scale.to_numpy(dtype=float), stress_cap), index=scale.index, dtype=float)
    guarded = base.weights.copy()
    guarded[["crypto", "equity"]] = guarded[["crypto", "equity"]].mul(scale, axis=0)
    guarded["cash"] = 1.0 - guarded[["crypto", "equity"]].sum(axis=1)
    guarded["cash"] = guarded["cash"].clip(lower=0.0, upper=1.0)
    bundle = _evaluate_allocation_candidate(
        candidate_id=candidate_id,
        family="meta_switch_mc_guard",
        weights=guarded,
        crypto_ret=returns["crypto"],
        equity_ret=returns["equity"],
        benchmark_ret=base.bundle.benchmark_gross_ret,
        profile=profile,
        benchmark_profile=profile,
        notes=f"regime_conditioned_mc_guard;base={base.bundle.result.candidate_id};horizon={horizon};step={step}",
    )
    source = pd.Series(np.where(scale < 1.0, "guarded", base.source.reindex(scale.index).fillna("cash")), index=scale.index, dtype=object)
    return AllocationBundle(bundle=bundle, weights=guarded, source=source), summary


def _build_hmm_meta_allocation(
    *,
    candidate_id: str,
    crypto_bundle: StrategyBundle,
    equity_bundle: StrategyBundle,
    btc_prices: pd.Series,
    spy_prices: pd.Series,
    profile: NetAssumptionProfile,
) -> tuple[AllocationBundle, Any]:
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
    features = build_hmm_feature_frame(
        primary_ret=btc.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0),
        secondary_ret=spy.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0),
        volatility_window=21,
    )
    hmm = fit_hmm_challenger(
        features,
        n_states=3,
        train_end=pd.Timestamp("2021-12-31"),
        random_state=19,
    )
    labels = hmm.regime_label.reindex(idx).ffill().fillna("neutral")
    prob = hmm.risk_on_probability.reindex(idx).ffill().fillna(0.0)
    weights = pd.DataFrame(0.0, index=idx, columns=["crypto", "equity", "cash"], dtype=float)
    source = pd.Series(index=idx, dtype=object)
    for dt in idx:
        label = str(labels.loc[dt]).lower()
        p_on = float(prob.loc[dt])
        if label == "risk_off":
            weights.loc[dt, "cash"] = 1.0
            source.loc[dt] = "cash"
        elif label == "risk_on" and bool(btc_ok.loc[dt]):
            weights.loc[dt, "crypto"] = 1.0
            source.loc[dt] = "crypto"
        elif bool(spy_ok.loc[dt]):
            weights.loc[dt, "equity"] = 1.0
            source.loc[dt] = "equity"
        elif bool(btc_ok.loc[dt]) and p_on >= 0.55:
            weights.loc[dt, "crypto"] = 0.75
            weights.loc[dt, "cash"] = 0.25
            source.loc[dt] = "crypto_partial"
        else:
            weights.loc[dt, "cash"] = 1.0
            source.loc[dt] = "cash"
    btc_bench = btc.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    spy_bench = spy.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    bench = 0.5 * btc_bench + 0.5 * spy_bench
    bundle = _evaluate_allocation_candidate(
        candidate_id=candidate_id,
        family="meta_switch_hmm",
        weights=weights,
        crypto_ret=crypto_ret,
        equity_ret=equity_ret,
        benchmark_ret=bench,
        profile=profile,
        benchmark_profile=profile,
        notes="HMM challenger over BTC/SPY features: risk_on prefers crypto, risk_off goes to cash, otherwise equities if SPY is healthy.",
    )
    return AllocationBundle(bundle=bundle, weights=weights, source=source), hmm


def _scenario_summary_row(
    *,
    candidate_id: str,
    weights: pd.DataFrame,
    history: pd.DataFrame,
    regime_history: pd.Series,
    states: list[str],
    transition: np.ndarray,
    moments: dict[str, Any],
    horizon: int = 21,
    n_paths: int = 2000,
    seed: int = 29,
) -> dict[str, Any]:
    aligned_weights = weights.reindex(history.index).fillna(0.0).astype(float)
    risky_sum = aligned_weights[["crypto", "equity"]].sum(axis=1)
    dt = risky_sum[risky_sum > 1e-8].index[-1] if (risky_sum > 1e-8).any() else aligned_weights.index[-1]
    current_state = str(pd.Series(regime_history).reindex(aligned_weights.index).ffill().fillna("stable").loc[dt])
    sim, _ = simulate_regime_conditioned_paths(
        regime_moments=moments,
        transition_matrix=transition,
        states=states,
        start_state=current_state,
        horizon=int(horizon),
        n_paths=int(n_paths),
        random_state=int(seed),
    )
    summary = summarize_portfolio_distribution(sim, aligned_weights.loc[dt, ["crypto", "equity"]].to_numpy(dtype=float))
    return {
        "candidate_id": candidate_id,
        "scenario_date": str(pd.Timestamp(dt).date()),
        "current_state": current_state,
        "history_rows": int(len(history)),
        "regime_rows": int(len(regime_history)),
        **summary,
    }


def _detection_row(
    *,
    predictor_name: str,
    target_name: str,
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, Any]:
    report = classification_report_binary(y_true, y_pred)
    return {"predictor": predictor_name, "target": target_name, **report}


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
        "hit_rate_10x_5y": result.hit_rate_10x_5y,
        "years_to_10x_full": result.years_to_10x_full,
        "notes": result.notes,
    }


def _calendar_year_rows(
    result: StrategyResult,
    *,
    notional_brl: float = 10000.0,
) -> list[dict[str, Any]]:
    net_ret = pd.to_numeric(result.net_ret, errors="coerce").dropna().astype(float)
    bench_ret = pd.to_numeric(result.benchmark_net_ret, errors="coerce").reindex(net_ret.index).fillna(0.0).astype(float)
    if net_ret.empty:
        return []
    wealth = (1.0 + net_ret).cumprod()
    bench_wealth = (1.0 + bench_ret).cumprod()
    rows: list[dict[str, Any]] = []
    for year, sub in net_ret.groupby(net_ret.index.year):
        idx = sub.index
        bench_sub = bench_ret.loc[idx]
        year_total = float(np.prod(1.0 + sub.to_numpy(dtype=float)) - 1.0)
        bench_total = float(np.prod(1.0 + bench_sub.to_numpy(dtype=float)) - 1.0)
        start_wealth = float(wealth.shift(1).reindex(idx).ffill().iloc[0]) if idx[0] != wealth.index[0] else 1.0
        end_wealth = float(wealth.loc[idx[-1]])
        running_profit = (end_wealth - start_wealth) * float(notional_brl)
        bench_start = float(bench_wealth.shift(1).reindex(idx).ffill().iloc[0]) if idx[0] != bench_wealth.index[0] else 1.0
        bench_end = float(bench_wealth.loc[idx[-1]])
        bench_running_profit = (bench_end - bench_start) * float(notional_brl)
        rows.append(
            {
                "candidate_id": result.candidate_id,
                "suite": result.suite,
                "year": int(year),
                "days": int(len(idx)),
                "year_total_return": year_total,
                "benchmark_total_return": bench_total,
                "edge_total_return": year_total - bench_total,
                "profit_brl_rebased_10000": year_total * float(notional_brl),
                "benchmark_profit_brl_rebased_10000": bench_total * float(notional_brl),
                "running_profit_brl_10000": running_profit,
                "benchmark_running_profit_brl_10000": bench_running_profit,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Compara Cholesky/Monte Carlo/HRP/HMM como camada auxiliar do Eigen Engine.")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--net-assumptions", default="config/profit_net_assumptions.json")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--outdir-root", default="results/validation/profit_regime_simulation_suite")
    args = ap.parse_args()

    outdir = (ROOT / args.outdir_root / _run_id()).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    prices_dir = (ROOT / args.prices_dir).resolve()

    profiles = load_net_assumption_profiles((ROOT / args.net_assumptions).resolve())
    foreign_profile = profiles["profiles"]["foreign_financial_brazil_resident"]
    crypto_profile = profiles["profiles"]["crypto_global_brazil_resident_conservative"]
    blended_profile = _blended_profile(
        crypto_profile,
        foreign_profile,
        profile_id="regime_layer_blended",
        label="Regime layer blended",
    )

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
    equity_group_map = _build_equity_group_map(equity_assets, equity_returns)

    crypto_result = _simulate_asset_rule(
        candidate_id="crypto_all__momvol21_hard15",
        family="crypto",
        allowed_tickers=crypto_tiers["crypto_all"],
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
    if crypto_result is None:
        raise SystemExit("failed to rebuild baseline crypto result")
    crypto_bundle = StrategyBundle(
        result=crypto_result,
        benchmark_gross_ret=pd.to_numeric(crypto_returns[str(args.benchmark_crypto)], errors="coerce").fillna(0.0).astype(float),
        profile=crypto_profile,
        benchmark_profile=crypto_profile,
    )

    eq_agg = _simulate_equity_group_sleeve_v2(
        candidate_id="equities_v2__slow189__g3__a1",
        returns=equity_returns,
        prices=equity_prices,
        asset_table=equity_assets,
        equity_groups=equity_group_map,
        benchmark_ticker=str(args.benchmark_equity),
        group_lookback_fast=63,
        group_lookback_slow=189,
        group_top_k=3,
        assets_per_group=1,
        asset_lookback=126,
        asset_ma_days=200,
        market_ma_days=200,
        profile=foreign_profile,
        benchmark_profile=foreign_profile,
    )
    eq_rob = _simulate_equity_group_sleeve_v3(
        candidate_id="equities_v3__slow252__g3__a2__br30__cap45",
        returns=equity_returns,
        prices=equity_prices,
        asset_table=equity_assets,
        equity_groups=equity_group_map,
        benchmark_ticker=str(args.benchmark_equity),
        group_lookback_fast=63,
        group_lookback_slow=252,
        group_top_k=3,
        assets_per_group=2,
        asset_lookback=126,
        asset_ma_days=200,
        market_ma_days=200,
        min_group_breadth=0.30,
        max_group_weight=0.45,
        profile=foreign_profile,
        benchmark_profile=foreign_profile,
    )
    if eq_agg is None or eq_rob is None:
        raise SystemExit("failed to rebuild baseline equity sleeves")
    regime_series = _load_structural_regime_series_local(ROOT)
    equity_meta = _simulate_equity_trail_switch_bundle(
        candidate_id="equities_meta__trail_switch__a1__r3",
        aggressive_bundle=eq_agg,
        robust_bundle=eq_rob,
        regime_series=regime_series,
        spy_prices=pd.to_numeric(equity_prices[str(args.benchmark_equity)], errors="coerce"),
    )

    btc_prices = pd.to_numeric(crypto_prices[str(args.benchmark_crypto)], errors="coerce")
    spy_prices = pd.to_numeric(equity_prices[str(args.benchmark_equity)], errors="coerce")

    baseline = _build_meta_v1_allocation(
        crypto_bundle=crypto_bundle,
        equity_bundle=equity_meta,
        btc_prices=btc_prices,
        spy_prices=spy_prices,
        profile=blended_profile,
    )
    hrp_bundle = _build_meta_hrp_allocation(
        crypto_bundle=crypto_bundle,
        equity_bundle=equity_meta,
        btc_prices=btc_prices,
        spy_prices=spy_prices,
        profile=blended_profile,
        lookback=63,
    )

    sleeve_returns = pd.concat(
        {
            "crypto": pd.to_numeric(crypto_bundle.result.gross_ret, errors="coerce"),
            "equity": pd.to_numeric(equity_meta.result.gross_ret, errors="coerce"),
        },
        axis=1,
        sort=False,
    ).dropna(how="all")
    aligned_regime = _regime_forward_fill_local(sleeve_returns.index, regime_series)

    mc_guard, mc_summary = _apply_mc_guard(
        candidate_id="meta_mc_guard__regime21",
        base=baseline,
        returns=sleeve_returns,
        regime=aligned_regime,
        profile=blended_profile,
        lookback=252,
        horizon=21,
        n_paths=400,
        step=42,
    )
    hrp_mc_guard, hrp_mc_summary = _apply_mc_guard(
        candidate_id="meta_hrp_mc_guard__regime21",
        base=hrp_bundle,
        returns=sleeve_returns,
        regime=aligned_regime,
        profile=blended_profile,
        lookback=252,
        horizon=21,
        n_paths=400,
        step=42,
    )
    hmm_bundle, hmm = _build_hmm_meta_allocation(
        candidate_id="meta_hmm__btc_spy_challenger",
        crypto_bundle=crypto_bundle,
        equity_bundle=equity_meta,
        btc_prices=btc_prices,
        spy_prices=spy_prices,
        profile=blended_profile,
    )

    candidates = [baseline, hrp_bundle, mc_guard, hrp_mc_guard, hmm_bundle]
    candidate_df = pd.DataFrame([_result_row(c.bundle.result) for c in candidates]).sort_values(
        ["net_sharpe", "net_ann_return", "net_max_drawdown"],
        ascending=[False, False, False],
    )
    candidate_df.to_csv(outdir / "candidate_compare.csv", index=False)
    calendar_year_df = pd.DataFrame(
        [row for candidate in candidates for row in _calendar_year_rows(candidate.bundle.result, notional_brl=10000.0)]
    )
    if not calendar_year_df.empty:
        calendar_year_df = calendar_year_df.sort_values(["year", "edge_total_return", "year_total_return"], ascending=[True, False, False])
    calendar_year_df.to_csv(outdir / "calendar_year_compare.csv", index=False)

    history = sleeve_returns.loc[sleeve_returns.index <= baseline.weights.index.max()].dropna(how="all")
    hist_regime = aligned_regime.reindex(history.index).fillna("stable")
    moments = estimate_regime_moments(history, hist_regime, min_obs=20)
    state_order, transition = estimate_transition_matrix(hist_regime, state_order=sorted(moments))
    scenario_rows = [
        _scenario_summary_row(
            candidate_id=c.bundle.result.candidate_id,
            weights=c.weights,
            history=history,
            regime_history=hist_regime,
            states=state_order,
            transition=transition,
            moments=moments,
            horizon=21,
            n_paths=1200,
            seed=29 + i,
        )
        for i, c in enumerate(candidates)
    ]
    scenario_df = pd.DataFrame(scenario_rows).sort_values("terminal_p05", ascending=False)
    scenario_df.to_csv(outdir / "scenario_compare.csv", index=False)

    benchmark_equity = (1.0 + baseline.bundle.benchmark_gross_ret).cumprod()
    regime_alert = aligned_regime.reindex(baseline.weights.index).fillna("stable").astype(str).str.lower().isin({"transition", "stress"}).astype(int)
    cash_alert = (baseline.weights["cash"] > 0.5).astype(int)
    hmm_alert = hmm.regime_label.reindex(baseline.weights.index).ffill().fillna("neutral").eq("risk_off").astype(int)
    mc_alert = (
        mc_summary.reindex(baseline.weights.index)["terminal_p05"].fillna(0.0).le(-0.08)
        | mc_summary.reindex(baseline.weights.index)["ruin_prob_m10"].fillna(0.0).ge(0.25)
    ).astype(int)
    y_regime = build_regime_future_event_label(aligned_regime.reindex(baseline.weights.index).fillna("stable"), horizon_days=21)
    y_drawdown = build_event_label(equity=benchmark_equity.reindex(baseline.weights.index), horizon_days=21, dd_threshold=0.10)
    detection_rows = []
    for predictor_name, signal in [
        ("eigen_regime_alert", regime_alert),
        ("baseline_cash_signal", cash_alert),
        ("mc_tail_alert", mc_alert),
        ("hmm_risk_off", hmm_alert),
    ]:
        detection_rows.append(_detection_row(predictor_name=predictor_name, target_name="future_regime_21d", y_true=y_regime, y_pred=signal))
        detection_rows.append(_detection_row(predictor_name=predictor_name, target_name="future_drawdown_21d", y_true=y_drawdown, y_pred=signal))
    detection_df = pd.DataFrame(detection_rows).sort_values(["target", "f1", "precision"], ascending=[True, False, False])
    detection_df.to_csv(outdir / "detection_compare.csv", index=False)

    reproduction_reference = json.loads(
        (ROOT / "results/validation/profit_layered_engine_suite/20260307T054325Z/summary.json").read_text(encoding="utf-8")
    )
    ref = reproduction_reference["best_meta_candidate"]
    reproduction_error = {
        "reference_candidate_id": str(ref["candidate_id"]),
        "candidate_id": baseline.bundle.result.candidate_id,
        "delta_net_ann_return": float(baseline.bundle.result.net_ann_return - float(ref["net_ann_return"])),
        "delta_net_sharpe": float(baseline.bundle.result.net_sharpe - float(ref["net_sharpe"])),
        "delta_net_max_drawdown": float(baseline.bundle.result.net_max_drawdown - float(ref["net_max_drawdown"])),
    }

    performance_best = candidate_df.sort_values(["net_ann_return", "net_sharpe"], ascending=[False, False]).iloc[0].to_dict()
    balanced_best = candidate_df.sort_values(["net_sharpe", "net_ann_return", "net_max_drawdown"], ascending=[False, False, False]).iloc[0].to_dict()
    regime_best = detection_df[detection_df["target"] == "future_regime_21d"].sort_values(["f1", "precision"], ascending=[False, False]).iloc[0].to_dict()
    drawdown_best = detection_df[detection_df["target"] == "future_drawdown_21d"].sort_values(["f1", "precision"], ascending=[False, False]).iloc[0].to_dict()

    worth_hrp = bool(
        hrp_bundle.bundle.result.net_sharpe > baseline.bundle.result.net_sharpe
        and hrp_bundle.bundle.result.net_ann_return >= 0.85 * baseline.bundle.result.net_ann_return
    )
    worth_mc = bool(
        mc_guard.bundle.result.net_max_drawdown > baseline.bundle.result.net_max_drawdown
        and mc_guard.bundle.result.net_ann_return >= 0.80 * baseline.bundle.result.net_ann_return
    )
    worth_hmm = bool(
        hmm_bundle.bundle.result.net_ann_return > baseline.bundle.result.net_ann_return
        or (
            drawdown_best["predictor"] == "hmm_risk_off"
            and float(drawdown_best.get("f1", 0.0)) > float(
                detection_df[
                    (detection_df["predictor"] == "eigen_regime_alert") & (detection_df["target"] == "future_drawdown_21d")
                ]["f1"].iloc[0]
            )
        )
    )
    worth_it = {
        "cholesky_regime_mc_layer": {
            "worth_it": worth_mc,
            "candidate": mc_guard.bundle.result.candidate_id,
            "reason": "melhora drawdown/intervalo de confiança sem destruir o retorno" if worth_mc else "melhora a governança de risco, mas não bate o baseline em retorno líquido",
        },
        "hrp_allocation_layer": {
            "worth_it": worth_hrp,
            "candidate": hrp_bundle.bundle.result.candidate_id,
            "reason": "melhora o equilíbrio risco/retorno na alocação entre sleeves" if worth_hrp else "fica mais limpo na alocação, mas o ganho final não justifica trocar o meta atual",
        },
        "hmm_challenger_layer": {
            "worth_it": worth_hmm,
            "candidate": hmm_bundle.bundle.result.candidate_id,
            "reason": "trouxe ganho real de detecção ou performance" if worth_hmm else "útil como challenger e camada de incerteza, não como substituto do core",
        },
    }

    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "outdir": str(outdir),
        "baseline_current": _result_row(baseline.bundle.result),
        "hrp_candidate": _result_row(hrp_bundle.bundle.result),
        "mc_guard_candidate": _result_row(mc_guard.bundle.result),
        "hrp_mc_guard_candidate": _result_row(hrp_mc_guard.bundle.result),
        "hmm_candidate": _result_row(hmm_bundle.bundle.result),
        "best_performance_candidate": performance_best,
        "best_balanced_candidate": balanced_best,
        "best_regime_detector": regime_best,
        "best_drawdown_detector": drawdown_best,
        "reproduction_error_vs_current_layered": reproduction_error,
        "worth_it": worth_it,
        "insights": [
            f"Baseline atual reproduzido como {baseline.bundle.result.candidate_id} com net_ann={baseline.bundle.result.net_ann_return:.4f} e sharpe={baseline.bundle.result.net_sharpe:.4f}.",
            f"HRP entre sleeves gerou {hrp_bundle.bundle.result.candidate_id} com net_ann={hrp_bundle.bundle.result.net_ann_return:.4f} e drawdown={hrp_bundle.bundle.result.net_max_drawdown:.4f}.",
            f"Guard de Monte Carlo condicionado por regime gerou {mc_guard.bundle.result.candidate_id} com net_ann={mc_guard.bundle.result.net_ann_return:.4f} e drawdown={mc_guard.bundle.result.net_max_drawdown:.4f}.",
            f"HMM challenger gerou {hmm_bundle.bundle.result.candidate_id} com net_ann={hmm_bundle.bundle.result.net_ann_return:.4f} e sharpe={hmm_bundle.bundle.result.net_sharpe:.4f}.",
            f"Melhor detector para drawdown futuro: {drawdown_best['predictor']} com f1={_safe_float(drawdown_best.get('f1')):.4f}.",
        ],
        "artifacts": {
            "candidate_compare_csv": str(outdir / "candidate_compare.csv"),
            "calendar_year_compare_csv": str(outdir / "calendar_year_compare.csv"),
            "scenario_compare_csv": str(outdir / "scenario_compare.csv"),
            "detection_compare_csv": str(outdir / "detection_compare.csv"),
            "mc_guard_summary_csv": str(outdir / "mc_guard_summary.csv"),
            "hrp_mc_guard_summary_csv": str(outdir / "hrp_mc_guard_summary.csv"),
            "hmm_state_summary_csv": str(outdir / "hmm_state_summary.csv"),
        },
    }
    mc_summary.to_csv(outdir / "mc_guard_summary.csv", index=True)
    hrp_mc_summary.to_csv(outdir / "hrp_mc_guard_summary.csv", index=True)
    hmm.state_summary.to_csv(outdir / "hmm_state_summary.csv", index=False)
    _write_json(outdir / "summary.json", summary)
    write_run_manifest(
        outdir,
        script="scripts/bench/validation/run_profit_regime_simulation_suite.py",
        params={
            "crypto_asset_groups": args.crypto_asset_groups,
            "equity_asset_groups": args.equity_asset_groups,
            "prices_dir": args.prices_dir,
            "benchmark_equity": args.benchmark_equity,
            "benchmark_crypto": args.benchmark_crypto,
        },
        paths=summary["artifacts"],
        extra={
            "baseline_net_ann_return": baseline.bundle.result.net_ann_return,
            "best_candidate_net_ann_return": _safe_float(performance_best.get("net_ann_return")),
            "best_drawdown_detector_f1": _safe_float(drawdown_best.get("f1")),
            "summary_json": str(outdir / "summary.json"),
        },
    )


if __name__ == "__main__":
    main()
