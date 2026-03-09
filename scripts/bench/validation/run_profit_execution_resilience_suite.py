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
from execution.net_assumptions import NetAssumptionProfile  # noqa: E402
from scripts.bench.validation.run_profit_alpha_hardening_suite import (  # noqa: E402
    _build_alpha_meta_allocation_bundle,
    _build_candidates,
)
from scripts.bench.validation.run_profit_frontier_expansion_suite import (  # noqa: E402
    EQUITY_EXCLUDED,
    StrategyResult,
    _build_equity_group_map,
    _evaluate_net,
    _ensure_benchmark_columns,
    _load_asset_table,
    _load_daily_universe,
    _rolling_ten_x_stats,
    _select_crypto_tiers,
    _simulate_asset_rule,
    _write_json,
)
from scripts.bench.validation.run_profit_investment_yearbook import _calendar_rows  # noqa: E402
from scripts.bench.validation.run_profit_regime_simulation_suite import (  # noqa: E402
    StrategyBundle,
    _evaluate_allocation_candidate,
    _build_meta_v1_allocation,
)
from scripts.bench.validation.run_profit_sector_pressure_suite import _research_row  # noqa: E402
from scripts.bench.validation.run_profit_layered_engine_suite import (  # noqa: E402
    _load_structural_regime_series_local,
    _simulate_equity_group_sleeve_v2,
    _simulate_equity_group_sleeve_v3,
    _simulate_equity_trail_switch_bundle,
)
from scripts.bench.validation.run_profit_sector_pressure_suite import (  # noqa: E402
    _simulate_equity_group_sleeve_v4_sector_pressure,
)


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _human_label(candidate_id: str) -> str:
    mapping = {
        "meta_major8_eq_a2r1": "Modo principal",
        "alpha_attack_major8_equity25": "Modo ataque",
        "meta_major8_eq_a2r1_mc_guard": "Modo principal com guarda",
        "alpha_attack_major8_equity25_mc_guard": "Modo ataque com guarda",
        "pure_crypto_attack": "Cripto puro agressivo",
        "pure_equity_main": "Ações puras",
        "blend_half_attack": "Meio a meio sem troca",
    }
    return mapping.get(str(candidate_id), str(candidate_id))


@dataclass(frozen=True)
class ExecutionProfile:
    name: str
    crypto_capacity_brl: float
    equity_capacity_brl: float
    delay_crypto_days: int
    delay_equity_days: int


@dataclass(frozen=True)
class CandidateModel:
    candidate_id: str
    label: str
    bundle: StrategyBundle
    weights: pd.DataFrame
    returns_frame: pd.DataFrame
    benchmark_ret: pd.Series
    profile: NetAssumptionProfile
    benchmark_profile: NetAssumptionProfile
    exec_profile: ExecutionProfile


def _weights_frame(index: pd.Index, *, crypto: float, equity: float, cash: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "crypto": float(crypto),
            "equity": float(equity),
            "cash": float(cash),
        },
        index=index,
        dtype=float,
    )


def _capacity_profile_for_candidate(candidate_id: str) -> ExecutionProfile:
    cid = str(candidate_id)
    if cid == "alpha_attack_major8_equity25":
        return ExecutionProfile(
            name="attack",
            crypto_capacity_brl=150000.0,
            equity_capacity_brl=900000.0,
            delay_crypto_days=2,
            delay_equity_days=1,
        )
    if cid == "alpha_attack_major8_equity25_mc_guard":
        return ExecutionProfile(
            name="attack_guard",
            crypto_capacity_brl=180000.0,
            equity_capacity_brl=1100000.0,
            delay_crypto_days=2,
            delay_equity_days=1,
        )
    if cid == "meta_major8_eq_a2r1_mc_guard":
        return ExecutionProfile(
            name="main_guard",
            crypto_capacity_brl=280000.0,
            equity_capacity_brl=1500000.0,
            delay_crypto_days=1,
            delay_equity_days=0,
        )
    if cid == "pure_crypto_attack":
        return ExecutionProfile(
            name="pure_crypto",
            crypto_capacity_brl=100000.0,
            equity_capacity_brl=1.0e12,
            delay_crypto_days=2,
            delay_equity_days=0,
        )
    if cid == "pure_equity_main":
        return ExecutionProfile(
            name="pure_equity",
            crypto_capacity_brl=1.0e12,
            equity_capacity_brl=1800000.0,
            delay_crypto_days=0,
            delay_equity_days=0,
        )
    if cid == "blend_half_attack":
        return ExecutionProfile(
            name="blend",
            crypto_capacity_brl=220000.0,
            equity_capacity_brl=1200000.0,
            delay_crypto_days=1,
            delay_equity_days=0,
        )
    return ExecutionProfile(
        name="main",
        crypto_capacity_brl=250000.0,
        equity_capacity_brl=1400000.0,
        delay_crypto_days=1,
        delay_equity_days=0,
    )


def _extra_bps_from_utilization(utilization: float, *, sleeve: str) -> float:
    util = float(max(0.0, utilization))
    if util <= 0.25:
        return 0.0
    if str(sleeve) == "crypto":
        base = 4.0
        slope = 16.0
    else:
        base = 1.5
        slope = 7.0
    if util <= 1.0:
        return base * ((util - 0.25) / 0.75)
    return min(120.0, base + slope * ((util - 1.0) ** 1.25))


def _build_strategy_bundle(
    *,
    candidate_id: str,
    family: str,
    gross_ret: pd.Series,
    turnover: pd.Series,
    benchmark_ret: pd.Series,
    profile: NetAssumptionProfile,
    benchmark_profile: NetAssumptionProfile,
    notes: str,
) -> StrategyBundle:
    idx = gross_ret.index.intersection(benchmark_ret.index)
    gross = pd.to_numeric(gross_ret.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    turn = pd.to_numeric(turnover.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    bench = pd.to_numeric(benchmark_ret.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    perf = _evaluate_net(
        gross_ret=gross,
        turnover=turn,
        profile=profile,
        benchmark_ret=bench,
        benchmark_profile=benchmark_profile,
    )
    hit5 = _rolling_ten_x_stats(perf["net_ret"], horizon_days=1260)
    wealth = (1.0 + perf["net_ret"]).cumprod()
    hit_full = wealth[wealth >= 10.0]
    years_to_10x = float((hit_full.index[0] - wealth.index[0]).days / 365.25) if not hit_full.empty else float("nan")
    result = StrategyResult(
        suite="execution_resilience",
        candidate_id=str(candidate_id),
        family=str(family),
        benchmark_ticker="BTC_SPY_50_50",
        gross_ret=gross,
        turnover=turn,
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
        notes=str(notes),
    )
    return StrategyBundle(
        result=result,
        benchmark_gross_ret=bench,
        profile=profile,
        benchmark_profile=benchmark_profile,
    )


def _build_fixed_sleeve_bundle(
    *,
    candidate_id: str,
    weights: pd.DataFrame,
    returns_frame: pd.DataFrame,
    benchmark_ret: pd.Series,
    profile: NetAssumptionProfile,
) -> StrategyBundle:
    bundle = _evaluate_allocation_candidate(
        candidate_id=str(candidate_id),
        family="ablation",
        weights=weights,
        crypto_ret=returns_frame["crypto"],
        equity_ret=returns_frame["equity"],
        benchmark_ret=benchmark_ret,
        profile=profile,
        benchmark_profile=profile,
        notes="execution resilience ablation bundle",
    )
    return bundle


def _base_candidate_models(built: dict[str, Any]) -> dict[str, CandidateModel]:
    allocations = built["allocations"]
    sleeve_returns = built["sleeve_returns"]

    models: dict[str, CandidateModel] = {}
    for key, candidate_id in [
        ("baseline", "meta_major8_eq_a2r1"),
        ("attack", "alpha_attack_major8_equity25"),
        ("baseline_guard", "meta_major8_eq_a2r1_mc_guard"),
        ("attack_guard", "alpha_attack_major8_equity25_mc_guard"),
    ]:
        alloc = allocations[key]
        models[candidate_id] = CandidateModel(
            candidate_id=candidate_id,
            label=_human_label(candidate_id),
            bundle=alloc.bundle,
            weights=alloc.weights.copy(),
            returns_frame=sleeve_returns[key][["crypto", "equity"]].copy(),
            benchmark_ret=alloc.bundle.benchmark_gross_ret.copy(),
            profile=alloc.bundle.profile,
            benchmark_profile=alloc.bundle.benchmark_profile,
            exec_profile=_capacity_profile_for_candidate(candidate_id),
        )

    attack = models["alpha_attack_major8_equity25"]
    main = models["meta_major8_eq_a2r1"]
    pure_crypto_weights = _weights_frame(attack.weights.index, crypto=1.0, equity=0.0, cash=0.0)
    pure_equity_weights = _weights_frame(main.weights.index, crypto=0.0, equity=1.0, cash=0.0)
    blend_weights = _weights_frame(attack.weights.index, crypto=0.5, equity=0.5, cash=0.0)

    pure_crypto_bundle = _build_fixed_sleeve_bundle(
        candidate_id="pure_crypto_attack",
        weights=pure_crypto_weights,
        returns_frame=attack.returns_frame,
        benchmark_ret=attack.benchmark_ret,
        profile=attack.profile,
    )
    pure_equity_bundle = _build_fixed_sleeve_bundle(
        candidate_id="pure_equity_main",
        weights=pure_equity_weights,
        returns_frame=main.returns_frame,
        benchmark_ret=main.benchmark_ret,
        profile=main.profile,
    )
    blend_bundle = _build_fixed_sleeve_bundle(
        candidate_id="blend_half_attack",
        weights=blend_weights,
        returns_frame=attack.returns_frame,
        benchmark_ret=attack.benchmark_ret,
        profile=attack.profile,
    )

    models["pure_crypto_attack"] = CandidateModel(
        candidate_id="pure_crypto_attack",
        label=_human_label("pure_crypto_attack"),
        bundle=pure_crypto_bundle,
        weights=pure_crypto_weights,
        returns_frame=attack.returns_frame.copy(),
        benchmark_ret=attack.benchmark_ret.copy(),
        profile=attack.profile,
        benchmark_profile=attack.benchmark_profile,
        exec_profile=_capacity_profile_for_candidate("pure_crypto_attack"),
    )
    models["pure_equity_main"] = CandidateModel(
        candidate_id="pure_equity_main",
        label=_human_label("pure_equity_main"),
        bundle=pure_equity_bundle,
        weights=pure_equity_weights,
        returns_frame=main.returns_frame.copy(),
        benchmark_ret=main.benchmark_ret.copy(),
        profile=main.profile,
        benchmark_profile=main.benchmark_profile,
        exec_profile=_capacity_profile_for_candidate("pure_equity_main"),
    )
    models["blend_half_attack"] = CandidateModel(
        candidate_id="blend_half_attack",
        label=_human_label("blend_half_attack"),
        bundle=blend_bundle,
        weights=blend_weights,
        returns_frame=attack.returns_frame.copy(),
        benchmark_ret=attack.benchmark_ret.copy(),
        profile=attack.profile,
        benchmark_profile=attack.benchmark_profile,
        exec_profile=_capacity_profile_for_candidate("blend_half_attack"),
    )
    return models


def _simulate_execution_proxy(
    *,
    model: CandidateModel,
    capital_brl: float,
    apply_liquidity: bool,
    apply_delay: bool,
    scenario_label: str,
) -> tuple[StrategyBundle, pd.Series]:
    idx = model.weights.index.intersection(model.returns_frame.index).intersection(model.benchmark_ret.index).sort_values()
    weights = model.weights.reindex(idx).fillna(0.0).astype(float)
    returns_frame = model.returns_frame.reindex(idx).fillna(0.0).astype(float)
    benchmark = model.benchmark_ret.reindex(idx).fillna(0.0).astype(float)
    delayed = pd.DataFrame(index=idx, dtype=float)
    delayed["crypto"] = returns_frame["crypto"].shift(model.exec_profile.delay_crypto_days if apply_delay else 0).fillna(0.0)
    delayed["equity"] = returns_frame["equity"].shift(model.exec_profile.delay_equity_days if apply_delay else 0).fillna(0.0)

    sleeve_delta = weights[["crypto", "equity"]].diff().abs().fillna(weights[["crypto", "equity"]].abs())
    turnover = weights[["crypto", "equity", "cash"]].diff().abs().sum(axis=1).fillna(weights[["crypto", "equity", "cash"]].abs().sum(axis=1)) / 2.0
    gross = pd.Series(index=idx, dtype=float)
    extra_cost_ret = pd.Series(0.0, index=idx, dtype=float)
    capital = float(capital_brl)

    for dt in idx:
        gross_raw = float(weights.loc[dt, "crypto"]) * float(delayed.loc[dt, "crypto"]) + float(weights.loc[dt, "equity"]) * float(delayed.loc[dt, "equity"])
        extra_cost = 0.0
        if apply_liquidity:
            crypto_notional = float(capital) * float(sleeve_delta.loc[dt, "crypto"])
            equity_notional = float(capital) * float(sleeve_delta.loc[dt, "equity"])
            crypto_util = crypto_notional / float(model.exec_profile.crypto_capacity_brl) if model.exec_profile.crypto_capacity_brl > 0 else 0.0
            equity_util = equity_notional / float(model.exec_profile.equity_capacity_brl) if model.exec_profile.equity_capacity_brl > 0 else 0.0
            crypto_cost = crypto_notional * (_extra_bps_from_utilization(crypto_util, sleeve="crypto") / 10000.0)
            equity_cost = equity_notional * (_extra_bps_from_utilization(equity_util, sleeve="equity") / 10000.0)
            extra_cost = (crypto_cost + equity_cost) / float(capital) if capital > 0 else 0.0
        gross.loc[dt] = gross_raw - extra_cost
        extra_cost_ret.loc[dt] = float(extra_cost)
        capital *= max(0.05, 1.0 + float(gross.loc[dt]))

    bundle = _build_strategy_bundle(
        candidate_id=model.candidate_id,
        family="execution_resilience",
        gross_ret=gross,
        turnover=turnover,
        benchmark_ret=benchmark,
        profile=model.profile,
        benchmark_profile=model.benchmark_profile,
        notes=(
            f"{scenario_label};capital_brl={capital_brl:.2f};"
            f"delay_crypto={model.exec_profile.delay_crypto_days if apply_delay else 0};"
            f"delay_equity={model.exec_profile.delay_equity_days if apply_delay else 0};"
            f"liquidity_proxy={'on' if apply_liquidity else 'off'}"
        ),
    )
    return bundle, extra_cost_ret


def _bundle_row(
    *,
    block: str,
    scenario: str,
    capital_brl: float | None,
    bundle: StrategyBundle,
    extra_cost_ret: pd.Series | None = None,
) -> dict[str, Any]:
    row = {
        "block": str(block),
        "scenario": str(scenario),
        "candidate_id": str(bundle.result.candidate_id),
        "candidate_label": _human_label(bundle.result.candidate_id),
        "capital_brl": _safe_float(capital_brl) if capital_brl is not None else float("nan"),
        "net_ann_return": _safe_float(bundle.result.net_ann_return),
        "net_total_return": _safe_float(bundle.result.net_total_return),
        "net_sharpe": _safe_float(bundle.result.net_sharpe),
        "net_max_drawdown": _safe_float(bundle.result.net_max_drawdown),
        "edge_vs_benchmark": _safe_float(bundle.result.edge_vs_benchmark),
        "avg_turnover_daily": _safe_float(bundle.result.avg_turnover_daily),
        "operation_days_total": int((pd.to_numeric(bundle.result.turnover, errors="coerce").fillna(0.0).abs() > 1e-8).sum()),
        "notes": str(bundle.result.notes),
    }
    if extra_cost_ret is not None and not extra_cost_ret.empty:
        row["avg_extra_liquidity_cost_ret"] = _safe_float(extra_cost_ret.mean())
        row["total_extra_liquidity_cost_ret"] = _safe_float(extra_cost_ret.sum())
    else:
        row["avg_extra_liquidity_cost_ret"] = float("nan")
        row["total_extra_liquidity_cost_ret"] = float("nan")
    return row


def _yearbook_rows(
    *,
    block: str,
    scenario: str,
    capital_brl: float,
    bundle: StrategyBundle,
) -> list[dict[str, Any]]:
    rows = _calendar_rows(result=bundle.result, capital_brl=float(capital_brl))
    for row in rows:
        row["block"] = str(block)
        row["scenario"] = str(scenario)
        row["candidate_label"] = _human_label(bundle.result.candidate_id)
    return rows


def _build_group_scenario_light(
    *,
    context: dict[str, Any],
    prices_dir: Path,
    crypto_groups: Path,
    crypto_meta: Path,
    equity_groups: Path,
    equity_meta: Path,
    benchmark_crypto: str,
    benchmark_equity: str,
    crypto_drop_tickers: list[str] | None = None,
    equity_drop_sectors: list[str] | None = None,
    crypto_allowed_mode: str = "major8",
) -> dict[str, Any]:
    crypto_assets = _load_asset_table(crypto_groups, crypto_meta)
    if crypto_drop_tickers:
        drops = {str(x).strip() for x in crypto_drop_tickers if str(x).strip()}
        crypto_assets = crypto_assets[~crypto_assets["ticker"].astype(str).isin(drops)].copy()
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
    allowed_crypto = list(crypto_tiers["crypto_all"]) if str(crypto_allowed_mode).strip().lower() == "all22" else list(crypto_tiers["crypto_major8"])
    if not allowed_crypto:
        raise SystemExit("universo cripto vazio no cenario de grupos")

    equity_assets = _load_asset_table(equity_groups, equity_meta)
    equity_assets = equity_assets[~equity_assets["asset_group"].astype(str).isin(EQUITY_EXCLUDED)].copy()
    if equity_drop_sectors:
        drops = {str(x).strip() for x in equity_drop_sectors if str(x).strip()}
        equity_assets = equity_assets[~equity_assets["asset_group"].astype(str).isin(drops)].copy()
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

    crypto_base = _simulate_asset_rule(
        candidate_id="crypto_major8__mom_vol_adj_lb021_rb07_k3",
        family="crypto_major8_search",
        allowed_tickers=allowed_crypto,
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
        profile=context["profiles"]["crypto"],
        benchmark_profile=context["profiles"]["crypto"],
    )
    crypto_attack = _simulate_asset_rule(
        candidate_id="crypto_major8__mom_total_lb021_rb07_k1",
        family="crypto_major8_search",
        allowed_tickers=allowed_crypto,
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
        profile=context["profiles"]["crypto"],
        benchmark_profile=context["profiles"]["crypto"],
    )
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
        profile=context["profiles"]["foreign"],
        benchmark_profile=context["profiles"]["foreign"],
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
        profile=context["profiles"]["foreign"],
        benchmark_profile=context["profiles"]["foreign"],
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
        profile=context["profiles"]["foreign"],
        benchmark_profile=context["profiles"]["foreign"],
    )
    if crypto_base is None or crypto_attack is None or eq_a2 is None or eq_r1 is None or eq_sp is None:
        raise SystemExit("falha ao reconstruir cenario leve de grupos")

    eq_base = _simulate_equity_trail_switch_bundle(
        candidate_id="equities_meta__trail_switch__a2__r1",
        aggressive_bundle=eq_a2,
        robust_bundle=eq_r1,
        regime_series=regime_series,
        spy_prices=pd.to_numeric(equity_prices[str(benchmark_equity)], errors="coerce"),
    )
    eq_attack = _simulate_equity_trail_switch_bundle(
        candidate_id="equities_meta__trail_switch__sector_p25",
        aggressive_bundle=eq_sp,
        robust_bundle=eq_r1,
        regime_series=regime_series,
        spy_prices=pd.to_numeric(equity_prices[str(benchmark_equity)], errors="coerce"),
    )

    crypto_base_bundle = StrategyBundle(
        result=crypto_base,
        benchmark_gross_ret=pd.to_numeric(crypto_returns[str(benchmark_crypto)], errors="coerce").reindex(crypto_base.gross_ret.index).fillna(0.0).astype(float),
        profile=context["profiles"]["crypto"],
        benchmark_profile=context["profiles"]["crypto"],
    )
    crypto_attack_bundle = StrategyBundle(
        result=crypto_attack,
        benchmark_gross_ret=pd.to_numeric(crypto_returns[str(benchmark_crypto)], errors="coerce").reindex(crypto_attack.gross_ret.index).fillna(0.0).astype(float),
        profile=context["profiles"]["crypto"],
        benchmark_profile=context["profiles"]["crypto"],
    )
    btc_prices = pd.to_numeric(crypto_prices[str(benchmark_crypto)], errors="coerce")
    spy_prices = pd.to_numeric(equity_prices[str(benchmark_equity)], errors="coerce")
    baseline = _build_meta_v1_allocation(
        crypto_bundle=crypto_base_bundle,
        equity_bundle=eq_base,
        btc_prices=btc_prices,
        spy_prices=spy_prices,
        profile=context["profiles"]["blended"],
    )
    attack = _build_alpha_meta_allocation_bundle(
        candidate_id="alpha_attack_major8_equity25",
        crypto_bundle=crypto_attack_bundle,
        equity_bundle=eq_attack,
        btc_prices=btc_prices,
        spy_prices=spy_prices,
        profile=context["profiles"]["blended"],
        entry_lookback=21,
        exit_lookback=63,
        entry_margin=0.05,
        exit_margin=0.05,
        risk_off_mode="equity25",
        min_crypto_hold_days=0,
    )
    baseline = baseline.__class__(
        bundle=_build_strategy_bundle(
            candidate_id="meta_major8_eq_a2r1",
            family="group_dependency",
            gross_ret=baseline.bundle.result.gross_ret,
            turnover=baseline.bundle.result.turnover,
            benchmark_ret=baseline.bundle.benchmark_gross_ret,
            profile=baseline.bundle.profile,
            benchmark_profile=baseline.bundle.benchmark_profile,
            notes=f"group_dependency_light;crypto_mode={crypto_allowed_mode};drop_sectors={equity_drop_sectors or []};drop_crypto={crypto_drop_tickers or []}",
        ),
        weights=baseline.weights,
        source=baseline.source,
    )
    attack = attack.__class__(
        bundle=_build_strategy_bundle(
            candidate_id="alpha_attack_major8_equity25",
            family="group_dependency",
            gross_ret=attack.bundle.result.gross_ret,
            turnover=attack.bundle.result.turnover,
            benchmark_ret=attack.bundle.benchmark_gross_ret,
            profile=attack.bundle.profile,
            benchmark_profile=attack.bundle.benchmark_profile,
            notes=f"group_dependency_light;crypto_mode={crypto_allowed_mode};drop_sectors={equity_drop_sectors or []};drop_crypto={crypto_drop_tickers or []}",
        ),
        weights=attack.weights,
        source=attack.source,
    )
    return {"baseline": baseline, "attack": attack}


def _research_rows_from_results(rows_df: pd.DataFrame) -> list[dict[str, Any]]:
    keep_ids = {"meta_major8_eq_a2r1", "alpha_attack_major8_equity25"}
    out: list[dict[str, Any]] = []
    subset = rows_df[(rows_df["block"] == "capital_scaling") & (rows_df["scenario"] == "capital_10000")].copy()
    for _, row in subset.iterrows():
        result = StrategyResult(
            suite="execution_resilience",
            candidate_id=str(row["candidate_id"]),
            family="execution_resilience",
            benchmark_ticker="BTC_SPY_50_50",
            gross_ret=pd.Series(dtype=float),
            turnover=pd.Series(dtype=float),
            net_ret=pd.Series(dtype=float),
            benchmark_net_ret=pd.Series(dtype=float),
            net_ann_return=_safe_float(row["net_ann_return"]),
            net_total_return=_safe_float(row["net_total_return"]),
            net_sharpe=_safe_float(row["net_sharpe"]),
            net_max_drawdown=_safe_float(row["net_max_drawdown"]),
            edge_vs_benchmark=_safe_float(row["edge_vs_benchmark"]),
            avg_turnover_daily=_safe_float(row["avg_turnover_daily"]),
            hit_rate_10x_5y=float("nan"),
            years_to_10x_full=float("nan"),
            notes=f"block={row['block']};scenario={row['scenario']}",
        )
        out.append(
            _research_row(
                result,
                outdir=ROOT / "results" / "validation",
                status="keep" if str(row["candidate_id"]) in keep_ids else "watch",
                methodology="execution_resilience_suite",
                label=f"{_human_label(row['candidate_id'])} sob execucao realista",
            )
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Escala de capital, liquidez-proxy, atraso por ativo e dependencia por grupos para os modos finais.")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--outdir-root", default="results/validation/profit_execution_resilience_suite")
    ap.add_argument("--skip-group-dependency", action="store_true")
    args = ap.parse_args()

    outdir = (ROOT / args.outdir_root / _run_id()).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    prices_dir = (ROOT / args.prices_dir).resolve()

    built = _build_candidates(
        prices_dir=prices_dir,
        crypto_groups=(ROOT / args.crypto_asset_groups).resolve(),
        crypto_meta=(ROOT / args.crypto_asset_metadata).resolve(),
        equity_groups=(ROOT / args.equity_asset_groups).resolve(),
        equity_meta=(ROOT / args.equity_asset_metadata).resolve(),
        benchmark_crypto=str(args.benchmark_crypto),
        benchmark_equity=str(args.benchmark_equity),
    )
    models = _base_candidate_models(built)

    compare_rows: list[dict[str, Any]] = []
    yearbook_rows: list[dict[str, Any]] = []

    for capital_brl in [10000.0, 100000.0, 500000.0, 2000000.0]:
        for model in models.values():
            bundle, extra_cost = _simulate_execution_proxy(
                model=model,
                capital_brl=float(capital_brl),
                apply_liquidity=True,
                apply_delay=True,
                scenario_label=f"capital_{int(capital_brl)}",
            )
            compare_rows.append(
                _bundle_row(
                    block="capital_scaling",
                    scenario=f"capital_{int(capital_brl)}",
                    capital_brl=float(capital_brl),
                    bundle=bundle,
                    extra_cost_ret=extra_cost,
                )
            )
            yearbook_rows.extend(
                _yearbook_rows(
                    block="capital_scaling",
                    scenario=f"capital_{int(capital_brl)}",
                    capital_brl=float(capital_brl),
                    bundle=bundle,
                )
            )

    for model in models.values():
        bundle, extra_cost = _simulate_execution_proxy(
            model=model,
            capital_brl=100000.0,
            apply_liquidity=False,
            apply_delay=True,
            scenario_label="delay_only",
        )
        compare_rows.append(
            _bundle_row(
                block="delay_proxy",
                scenario="delay_only",
                capital_brl=100000.0,
                bundle=bundle,
                extra_cost_ret=extra_cost,
            )
        )
        yearbook_rows.extend(
            _yearbook_rows(
                block="delay_proxy",
                scenario="delay_only",
                capital_brl=100000.0,
                bundle=bundle,
            )
        )

    if not bool(args.skip_group_dependency):
        major8 = list(built["context"]["crypto_tiers"]["crypto_major8"])
        group_configs = {
            "base": {},
            "sem_majors": {"crypto_drop_tickers": list(major8), "crypto_allowed_mode": "all22"},
            "sem_technology": {"equity_drop_sectors": ["technology"]},
            "sem_top2_grupos": {"equity_drop_sectors": ["technology", "materials"]},
        }
        for scenario_name, config in group_configs.items():
            payload = _build_group_scenario_light(
                context=built["context"],
                prices_dir=prices_dir,
                crypto_groups=(ROOT / args.crypto_asset_groups).resolve(),
                crypto_meta=(ROOT / args.crypto_asset_metadata).resolve(),
                equity_groups=(ROOT / args.equity_asset_groups).resolve(),
                equity_meta=(ROOT / args.equity_asset_metadata).resolve(),
                benchmark_crypto=str(args.benchmark_crypto),
                benchmark_equity=str(args.benchmark_equity),
                crypto_drop_tickers=config.get("crypto_drop_tickers"),
                equity_drop_sectors=config.get("equity_drop_sectors"),
                crypto_allowed_mode=str(config.get("crypto_allowed_mode", "major8")),
            )
            for key, candidate_id in [("baseline", "meta_major8_eq_a2r1"), ("attack", "alpha_attack_major8_equity25")]:
                alloc = payload[key]
                bundle = alloc.bundle
                compare_rows.append(
                    _bundle_row(
                        block="group_dependency",
                        scenario=str(scenario_name),
                        capital_brl=10000.0,
                        bundle=bundle,
                    )
                )
                yearbook_rows.extend(
                    _yearbook_rows(
                        block="group_dependency",
                        scenario=str(scenario_name),
                        capital_brl=10000.0,
                        bundle=bundle,
                    )
                )

    compare_df = pd.DataFrame(compare_rows).sort_values(
        ["block", "scenario", "net_total_return", "net_ann_return"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)
    yearbook_df = pd.DataFrame(yearbook_rows).sort_values(
        ["block", "scenario", "candidate_id", "year"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    compare_df.to_csv(outdir / "scenario_compare.csv", index=False)
    yearbook_df.to_csv(outdir / "yearbook_reais.csv", index=False)

    base_cap = compare_df[(compare_df["block"] == "capital_scaling") & (compare_df["scenario"] == "capital_10000")].copy()
    high_cap = compare_df[(compare_df["block"] == "capital_scaling") & (compare_df["scenario"] == "capital_2000000")].copy()
    degrade_rows: list[dict[str, Any]] = []
    for candidate_id, sub in base_cap.groupby("candidate_id"):
        high = high_cap[high_cap["candidate_id"] == candidate_id].head(1)
        if sub.empty or high.empty:
            continue
        row = sub.iloc[0]
        row_hi = high.iloc[0]
        degrade_rows.append(
            {
                "candidate_id": str(candidate_id),
                "candidate_label": _human_label(candidate_id),
                "ann_return_delta": _safe_float(row_hi["net_ann_return"]) - _safe_float(row["net_ann_return"]),
                "total_return_delta": _safe_float(row_hi["net_total_return"]) - _safe_float(row["net_total_return"]),
                "drawdown_delta": _safe_float(row_hi["net_max_drawdown"]) - _safe_float(row["net_max_drawdown"]),
                "extra_cost_delta": _safe_float(row_hi["avg_extra_liquidity_cost_ret"]) - _safe_float(row["avg_extra_liquidity_cost_ret"]),
            }
        )
    degrade_df = pd.DataFrame(degrade_rows).sort_values("total_return_delta", ascending=True).reset_index(drop=True)
    degrade_df.to_csv(outdir / "capital_degradation.csv", index=False)

    group_summary = compare_df[compare_df["block"] == "group_dependency"].copy()
    if not group_summary.empty:
        group_base = group_summary[group_summary["scenario"] == "base"][["candidate_id", "net_total_return", "net_ann_return"]].rename(
            columns={"net_total_return": "base_total_return", "net_ann_return": "base_ann_return"}
        )
        group_summary = group_summary.merge(group_base, on="candidate_id", how="left")
        group_summary["total_retention"] = pd.to_numeric(group_summary["net_total_return"], errors="coerce") / pd.to_numeric(group_summary["base_total_return"], errors="coerce")
        group_summary["ann_retention"] = pd.to_numeric(group_summary["net_ann_return"], errors="coerce") / pd.to_numeric(group_summary["base_ann_return"], errors="coerce")
    group_summary.to_csv(outdir / "group_dependency_compare.csv", index=False)

    best_profit = base_cap.sort_values("net_total_return", ascending=False).head(1)
    best_scaled = high_cap.sort_values("net_total_return", ascending=False).head(1)
    most_sensitive = degrade_df.head(1)
    worst_group = group_summary[group_summary["scenario"] != "base"].sort_values("total_retention", ascending=True).head(1) if not group_summary.empty else pd.DataFrame()

    summary = {
        "run_id": outdir.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "questions_answered": {
            "capital_scaling": True,
            "liquidity_proxy": True,
            "per_asset_delay_proxy": True,
            "group_dependency": not group_summary.empty,
            "all_years_yearbook_generated": bool(not yearbook_df.empty),
        },
        "capital_scaling": {
            "capitals_tested_brl": [10000, 100000, 500000, 2000000],
            "best_profit_at_small_capital": best_profit.iloc[0].to_dict() if not best_profit.empty else {},
            "best_profit_at_large_capital": best_scaled.iloc[0].to_dict() if not best_scaled.empty else {},
            "most_scale_sensitive": most_sensitive.iloc[0].to_dict() if not most_sensitive.empty else {},
        },
        "group_dependency": {
            "worst_retention_case": worst_group.iloc[0].to_dict() if not worst_group.empty else {},
        },
        "artifacts": {
            "scenario_compare_csv": str((outdir / "scenario_compare.csv").resolve()),
            "yearbook_reais_csv": str((outdir / "yearbook_reais.csv").resolve()),
            "capital_degradation_csv": str((outdir / "capital_degradation.csv").resolve()),
            "group_dependency_compare_csv": str((outdir / "group_dependency_compare.csv").resolve()),
            "summary_json": str((outdir / "summary.json").resolve()),
        },
    }
    _write_json(outdir / "summary.json", summary)
    research_rows = _research_rows_from_results(compare_df)
    _write_json(outdir / "profit_research_rows.json", {"rows": research_rows})
    write_run_manifest(
        outdir,
        script="scripts/bench/validation/run_profit_execution_resilience_suite.py",
        params={
            "skip_group_dependency": bool(args.skip_group_dependency),
        },
        paths=summary["artifacts"],
        extra={
            "suite": "profit_execution_resilience_suite",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


if __name__ == "__main__":
    main()
