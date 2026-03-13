#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.portfolio import (  # noqa: E402
    AsymmetricPolicyConfig,
    CryptoConcentrationConfig,
    PeriodLossGuardConfig,
    YearDefenseConfig,
    combine_guard_actions,
    compute_ytd_stress,
    crypto_concentration_risk,
    map_risk_state_to_exposure,
    monthly_loss_guard,
    next_mode_state,
    quarterly_loss_guard,
    year_bad_state_trigger,
)
from engine.structural.run_manifest import write_run_manifest  # noqa: E402
from scripts.bench.validation.run_profit_alpha_hardening_suite import _build_candidates  # noqa: E402
from scripts.bench.validation.run_profit_frontier_expansion_suite import (  # noqa: E402
    StrategyResult,
    _evaluate_net,
    _safe_float,
    _write_json,
)
from scripts.bench.validation.run_profit_investment_yearbook import _calendar_rows  # noqa: E402
from scripts.bench.validation.run_profit_regime_simulation_suite import StrategyBundle  # noqa: E402


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _human_label(candidate_id: str) -> str:
    mapping = {
        "alpha_attack_major8_equity25": "Ataque atual",
        "meta_major8_eq_a2r1_mc_guard": "Protegido atual",
        "period_loss_guards_light": "Travas mensal e trimestral leves",
        "period_loss_guards_default": "Travas mensal e trimestral",
        "crypto_cap_default": "Cap cripto condicional",
        "asymmetric_exit_default": "Saída assimétrica",
        "regime_allocator_default": "Migração gradual de risco",
        "year_defense_default": "Defesa de ano ruim",
        "full_defense_light": "Stack defensivo leve",
        "full_defense_default": "Stack defensivo completo",
    }
    return mapping.get(str(candidate_id), str(candidate_id))


def _regime_to_stress(regime: str) -> float:
    key = str(regime or "").strip().lower()
    if key in {"stress", "estresse"}:
        return 0.85
    if key in {"transition", "transicao"}:
        return 0.62
    if key in {"stable", "estavel"}:
        return 0.30
    if key in {"dispersion", "dispersao"}:
        return 0.16
    return 0.45


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _normalize_mix(attack_fraction: float, protection_fraction: float, cash_fraction: float) -> tuple[float, float, float]:
    attack = max(0.0, float(attack_fraction))
    protect = max(0.0, float(protection_fraction))
    cash = max(0.0, float(cash_fraction))
    total = attack + protect + cash
    if total <= 1e-12:
        return 0.0, 0.0, 1.0
    return attack / total, protect / total, cash / total


def _period_state(period_action: str) -> tuple[float, float, float]:
    key = str(period_action or "NORMAL").upper()
    if key == "REDUCED_ATTACK":
        return 0.60, 0.30, 0.10
    if key == "PROTECTED":
        return 0.20, 0.70, 0.10
    if key == "CASH_HEAVY":
        return 0.05, 0.35, 0.60
    return 1.0, 0.0, 0.0


def _recovery_days(net_ret: pd.Series) -> int:
    values = pd.to_numeric(net_ret, errors="coerce").fillna(0.0).astype(float)
    if values.empty:
        return 0
    wealth = (1.0 + values).cumprod()
    peak = wealth.cummax()
    underwater = wealth < peak
    longest = 0
    current = 0
    for flag in underwater:
        if bool(flag):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _bundle_from_paths(
    *,
    candidate_id: str,
    gross_ret: pd.Series,
    turnover: pd.Series,
    benchmark_ret: pd.Series,
    profile,
    benchmark_profile,
    notes: str,
) -> StrategyBundle:
    idx = gross_ret.index.intersection(turnover.index).intersection(benchmark_ret.index)
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
    result = StrategyResult(
        suite="bad_year_defense",
        candidate_id=str(candidate_id),
        family="defense_layers",
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
        hit_rate_10x_5y=float("nan"),
        years_to_10x_full=float("nan"),
        notes=str(notes),
    )
    return StrategyBundle(
        result=result,
        benchmark_gross_ret=bench,
        profile=profile,
        benchmark_profile=benchmark_profile,
    )


def _year_metrics(bundle: StrategyBundle, capital_brl: float) -> pd.DataFrame:
    rows = _calendar_rows(result=bundle.result, capital_brl=float(capital_brl))
    return pd.DataFrame(rows)


def _signal_frame(context: dict[str, Any], attack_alloc, protect_alloc) -> pd.DataFrame:
    idx = (
        attack_alloc.bundle.result.gross_ret.index.intersection(protect_alloc.bundle.result.gross_ret.index)
        .intersection(context["attack_score_exogenous"].index)
        .intersection(context["regime_series"].index)
        .intersection(context["exogenous_panel"].index)
        .sort_values()
    )
    panel = pd.DataFrame(index=idx)
    panel["confidence_score"] = pd.to_numeric(context["attack_score_exogenous"].reindex(idx), errors="coerce").fillna(0.5).clip(0.0, 1.0)
    panel["regime"] = pd.Series(context["regime_series"], index=context["regime_series"].index).reindex(idx).ffill().bfill().astype(str)
    panel["structural_stress"] = panel["regime"].map(_regime_to_stress).fillna(0.45).astype(float)
    exo = context["exogenous_panel"].reindex(idx).copy()
    for col in ["liquidation", "breadth", "crowding", "critical_slowing_down", "crypto_dependency_risk", "macro_stress", "exogenous_risk"]:
        panel[col] = pd.to_numeric(exo.get(col), errors="coerce").fillna(0.5).clip(0.0, 1.0).astype(float)
    crypto_ret = pd.to_numeric(context["crypto_returns"].get(context["benchmark_crypto"]), errors="coerce").reindex(idx).fillna(0.0).astype(float)
    panel["crypto_volatility"] = crypto_ret.rolling(21, min_periods=10).std(ddof=0).fillna(0.0).clip(0.0, 1.0)
    return panel.shift(1).ffill().fillna(
        {
            "confidence_score": 0.5,
            "structural_stress": 0.45,
            "liquidation": 0.5,
            "breadth": 0.5,
            "crowding": 0.5,
            "critical_slowing_down": 0.5,
            "crypto_dependency_risk": 0.5,
            "macro_stress": 0.5,
            "exogenous_risk": 0.5,
            "crypto_volatility": 0.0,
        }
    )


def _apply_top_level_crypto_cap(
    *,
    attack_fraction: float,
    protection_fraction: float,
    cash_fraction: float,
    attack_crypto_weight: float,
    protect_crypto_weight: float,
    signal_bundle: dict[str, Any],
    config: CryptoConcentrationConfig,
) -> tuple[float, float, float, float]:
    attack_fraction, protection_fraction, cash_fraction = _normalize_mix(attack_fraction, protection_fraction, cash_fraction)
    current_crypto_share = attack_fraction * float(attack_crypto_weight) + protection_fraction * float(protect_crypto_weight)
    risk_score = crypto_concentration_risk(
        signal_bundle,
        pd.DataFrame(
            [
                {
                    "crypto": current_crypto_share,
                    "equity": max(0.0, 1.0 - current_crypto_share - cash_fraction),
                    "cash": cash_fraction,
                }
            ]
        ),
    )
    cap = (
        float(config.max_crypto_weight_stressed)
        if float(risk_score) >= float(config.crypto_risk_threshold)
        else float(config.max_crypto_weight_normal)
    )
    if current_crypto_share <= cap + 1e-9:
        return attack_fraction, protection_fraction, cash_fraction, risk_score

    overflow = current_crypto_share - cap
    attack_contrib = attack_fraction * float(attack_crypto_weight)
    protect_contrib = protection_fraction * float(protect_crypto_weight)

    if attack_contrib > 1e-9 and float(attack_crypto_weight) > 1e-9:
        reduce_attack_contrib = min(attack_contrib, overflow)
        delta_attack = reduce_attack_contrib / float(attack_crypto_weight)
        attack_fraction = max(0.0, attack_fraction - delta_attack)
        protection_fraction += delta_attack * 0.35
        cash_fraction += delta_attack * 0.65
        overflow -= reduce_attack_contrib

    if overflow > 1e-9 and protect_contrib > 1e-9 and float(protect_crypto_weight) > 1e-9:
        reduce_protect_contrib = min(protect_contrib, overflow)
        delta_protect = reduce_protect_contrib / float(protect_crypto_weight)
        protection_fraction = max(0.0, protection_fraction - delta_protect)
        cash_fraction += delta_protect

    return (*_normalize_mix(attack_fraction, protection_fraction, cash_fraction), risk_score)


def _simulate_candidate(
    *,
    candidate_id: str,
    attack_alloc,
    protect_alloc,
    signal_frame: pd.DataFrame,
    period_config: PeriodLossGuardConfig | None,
    concentration_config: CryptoConcentrationConfig | None,
    asym_config: AsymmetricPolicyConfig | None,
    allocator_config: dict[str, Any] | None,
    year_config: YearDefenseConfig | None,
    capital_brl: float,
) -> tuple[StrategyBundle, pd.DataFrame]:
    idx = (
        attack_alloc.bundle.result.gross_ret.index.intersection(protect_alloc.bundle.result.gross_ret.index)
        .intersection(attack_alloc.weights.index)
        .intersection(protect_alloc.weights.index)
        .intersection(signal_frame.index)
        .sort_values()
    )
    attack_gross = pd.to_numeric(attack_alloc.bundle.result.gross_ret.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    protect_gross = pd.to_numeric(protect_alloc.bundle.result.gross_ret.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    benchmark = pd.to_numeric(attack_alloc.bundle.benchmark_gross_ret.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    attack_weights = attack_alloc.weights.reindex(idx).fillna(0.0).astype(float)
    protect_weights = protect_alloc.weights.reindex(idx).fillna(0.0).astype(float)

    current_state = "ATTACK"
    month_key = None
    quarter_key = None
    year_key = None
    month_prod = 1.0
    quarter_prod = 1.0
    ytd_prod = 1.0
    ytd_wealth = 1.0
    ytd_peak = 1.0
    bad_days = 0

    weights_rows: list[dict[str, Any]] = []
    gross_rows: list[float] = []

    for dt in idx:
        timestamp = pd.Timestamp(dt)
        this_month = timestamp.to_period("M")
        this_quarter = timestamp.to_period("Q")
        this_year = int(timestamp.year)
        if month_key != this_month:
            month_key = this_month
            month_prod = 1.0
        if quarter_key != this_quarter:
            quarter_key = this_quarter
            quarter_prod = 1.0
        if year_key != this_year:
            year_key = this_year
            ytd_prod = 1.0
            ytd_wealth = 1.0
            ytd_peak = 1.0
            bad_days = 0

        row = signal_frame.loc[dt]
        signal_bundle = {
            "confidence_score": _clip01(row.get("confidence_score", 0.5)),
            "structural_stress": _clip01(row.get("structural_stress", 0.45)),
            "liquidation": _clip01(row.get("liquidation", 0.5)),
            "breadth": _clip01(row.get("breadth", 0.5)),
            "crowding": _clip01(row.get("crowding", 0.5)),
            "critical_slowing_down": _clip01(row.get("critical_slowing_down", 0.5)),
            "crypto_dependency_risk": _clip01(row.get("crypto_dependency_risk", 0.5)),
            "crypto_volatility": _clip01(row.get("crypto_volatility", 0.0)),
        }

        month_return = month_prod - 1.0
        quarter_return = quarter_prod - 1.0
        ytd_return = ytd_prod - 1.0
        ytd_drawdown = (ytd_wealth / ytd_peak) - 1.0 if ytd_peak > 0.0 else 0.0
        year_stress = compute_ytd_stress(pd.Series([ytd_return]), pd.Series([ytd_drawdown]), signal_bundle)

        monthly_action = monthly_loss_guard(month_return, period_config) if period_config is not None else "NORMAL"
        quarterly_action = quarterly_loss_guard(quarter_return, period_config) if period_config is not None else "NORMAL"
        period_action = combine_guard_actions(monthly_action, quarterly_action)

        year_bad = False
        if year_config is not None:
            year_bad = year_bad_state_trigger(
                {
                    "ytd_return": ytd_return,
                    "ytd_drawdown": ytd_drawdown,
                    "year_stress": year_stress,
                    "bad_days": bad_days,
                },
                year_config,
            )

        if asym_config is not None:
            current_state = next_mode_state(
                current_state=current_state,
                attack_signal=float(signal_bundle["confidence_score"]),
                defense_signal=max(float(signal_bundle["structural_stress"]), float(signal_bundle["liquidation"]), float(year_stress)),
                config=asym_config,
            )
        else:
            current_state = "ATTACK"

        if allocator_config is not None:
            confidence_for_allocator = float(signal_bundle["confidence_score"])
            if current_state == "PROTECT":
                confidence_for_allocator = min(confidence_for_allocator, 0.42)
            exposure = map_risk_state_to_exposure(
                {
                    "confidence_score": confidence_for_allocator,
                    "structural_stress": max(float(signal_bundle["structural_stress"]), float(year_stress)),
                },
                {
                    "period_action": period_action,
                    "year_bad_state": year_bad,
                },
                allocator_config,
            )
            attack_fraction, protection_fraction, cash_fraction = _normalize_mix(
                exposure.attack_fraction,
                exposure.protection_fraction,
                exposure.cash_fraction,
            )
        else:
            if current_state == "ATTACK":
                attack_fraction, protection_fraction, cash_fraction = 1.0, 0.0, 0.0
            else:
                attack_fraction, protection_fraction, cash_fraction = 0.0, 1.0, 0.0
            p_attack, p_protect, p_cash = _period_state(period_action)
            if period_config is not None:
                attack_fraction = min(attack_fraction, p_attack)
                protection_fraction = max(protection_fraction, p_protect)
                cash_fraction = max(cash_fraction, p_cash)
            if year_bad:
                attack_fraction = min(attack_fraction, 0.20)
                protection_fraction = max(protection_fraction, 0.55)
                cash_fraction = max(cash_fraction, 0.25)
            attack_fraction, protection_fraction, cash_fraction = _normalize_mix(
                attack_fraction,
                protection_fraction,
                cash_fraction,
            )

        concentration_risk = 0.0
        if concentration_config is not None:
            attack_fraction, protection_fraction, cash_fraction, concentration_risk = _apply_top_level_crypto_cap(
                attack_fraction=attack_fraction,
                protection_fraction=protection_fraction,
                cash_fraction=cash_fraction,
                attack_crypto_weight=float(attack_weights.loc[dt, "crypto"]),
                protect_crypto_weight=float(protect_weights.loc[dt, "crypto"]),
                signal_bundle=signal_bundle,
                config=concentration_config,
            )

        gross = (
            attack_fraction * float(attack_gross.loc[dt])
            + protection_fraction * float(protect_gross.loc[dt])
        )
        blended_weights = attack_weights.loc[dt] * attack_fraction + protect_weights.loc[dt] * protection_fraction
        blended_weights["cash"] = float(blended_weights.get("cash", 0.0)) + cash_fraction
        total = float(blended_weights.sum())
        if total > 1e-12:
            blended_weights = blended_weights / total
        gross_rows.append(gross)
        weights_rows.append(
            {
                "date": timestamp,
                "crypto": float(blended_weights.get("crypto", 0.0)),
                "equity": float(blended_weights.get("equity", 0.0)),
                "cash": float(blended_weights.get("cash", 0.0)),
                "attack_fraction": float(attack_fraction),
                "protection_fraction": float(protection_fraction),
                "cash_fraction": float(cash_fraction),
                "confidence_score": float(signal_bundle["confidence_score"]),
                "structural_stress": float(signal_bundle["structural_stress"]),
                "liquidation": float(signal_bundle["liquidation"]),
                "year_stress": float(year_stress),
                "period_action": str(period_action),
                "year_bad_state": int(year_bad),
                "mode_state": str(current_state),
                "concentration_risk": float(concentration_risk),
            }
        )

        month_prod *= 1.0 + gross
        quarter_prod *= 1.0 + gross
        ytd_prod *= 1.0 + gross
        ytd_wealth *= 1.0 + gross
        ytd_peak = max(ytd_peak, ytd_wealth)
        if gross < 0.0:
            bad_days += 1

    weights_df = pd.DataFrame(weights_rows).set_index("date").astype(
        {
            "crypto": float,
            "equity": float,
            "cash": float,
            "attack_fraction": float,
            "protection_fraction": float,
            "cash_fraction": float,
            "confidence_score": float,
            "structural_stress": float,
            "liquidation": float,
            "year_stress": float,
            "period_action": str,
            "year_bad_state": int,
            "mode_state": str,
            "concentration_risk": float,
        }
    )
    turnover = (
        weights_df[["crypto", "equity", "cash"]]
        .diff()
        .abs()
        .sum(axis=1)
        .fillna(weights_df[["crypto", "equity", "cash"]].abs().sum(axis=1))
        / 2.0
    )
    bundle = _bundle_from_paths(
        candidate_id=candidate_id,
        gross_ret=pd.Series(gross_rows, index=idx, dtype=float),
        turnover=turnover,
        benchmark_ret=benchmark,
        profile=attack_alloc.bundle.profile,
        benchmark_profile=attack_alloc.bundle.benchmark_profile,
        notes=(
            f"period={period_config is not None};concentration={concentration_config is not None};"
            f"asymmetric={asym_config is not None};allocator={allocator_config is not None};year={year_config is not None}"
        ),
    )
    return bundle, weights_df


def _candidate_row(
    *,
    bundle: StrategyBundle,
    baseline: StrategyBundle,
    capital_brl: float,
) -> dict[str, Any]:
    yearbook = _year_metrics(bundle, capital_brl=capital_brl)
    base_yearbook = _year_metrics(baseline, capital_brl=capital_brl)
    worst_year = yearbook.sort_values("profit_brl", ascending=True).head(1)
    years_negative = int((yearbook["profit_brl"] < 0.0).sum()) if not yearbook.empty else 0
    base_years_negative = int((base_yearbook["profit_brl"] < 0.0).sum()) if not base_yearbook.empty else 0
    total_retention = (
        _safe_float(bundle.result.net_total_return) / _safe_float(baseline.result.net_total_return)
        if abs(_safe_float(baseline.result.net_total_return)) > 1e-12
        else float("nan")
    )
    return {
        "candidate_id": str(bundle.result.candidate_id),
        "candidate_label": _human_label(bundle.result.candidate_id),
        "net_ann_return": _safe_float(bundle.result.net_ann_return),
        "net_total_return": _safe_float(bundle.result.net_total_return),
        "net_sharpe": _safe_float(bundle.result.net_sharpe),
        "net_max_drawdown": _safe_float(bundle.result.net_max_drawdown),
        "avg_turnover_daily": _safe_float(bundle.result.avg_turnover_daily),
        "capital_final_brl": float(capital_brl * (1.0 + _safe_float(bundle.result.net_total_return))),
        "profit_total_brl": float(capital_brl * _safe_float(bundle.result.net_total_return)),
        "years_negative": years_negative,
        "baseline_years_negative": base_years_negative,
        "worst_year": int(worst_year.iloc[0]["year"]) if not worst_year.empty else None,
        "worst_year_profit_brl": float(worst_year.iloc[0]["profit_brl"]) if not worst_year.empty else None,
        "max_recovery_days": _recovery_days(bundle.result.net_ret),
        "total_return_retention": total_retention,
        "worth_keeping": bool(
            years_negative <= base_years_negative
            and (_safe_float(bundle.result.net_total_return) >= 0.75 * _safe_float(baseline.result.net_total_return))
            and abs(_safe_float(bundle.result.net_max_drawdown)) <= abs(_safe_float(baseline.result.net_max_drawdown))
        ),
        "notes": str(bundle.result.notes or ""),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Testa defesas para reduzir anos ruins sem trocar o core do ataque.")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--capital-brl", type=float, default=10000.0)
    ap.add_argument("--outdir-root", default="results/validation/profit_bad_year_defense_suite")
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
    attack_alloc = built["allocations"]["attack"]
    protect_alloc = built["allocations"]["baseline_guard"]
    signal_frame = _signal_frame(built["context"], attack_alloc, protect_alloc)

    candidates: dict[str, StrategyBundle] = {
        "alpha_attack_major8_equity25": attack_alloc.bundle,
        "meta_major8_eq_a2r1_mc_guard": protect_alloc.bundle,
    }
    guard_logs: dict[str, pd.DataFrame] = {}

    sim_specs = [
        (
            "period_loss_guards_light",
            dict(
                period_config=PeriodLossGuardConfig(
                    monthly_reduce_threshold=-0.04,
                    monthly_protect_threshold=-0.07,
                    monthly_cash_threshold=-0.12,
                    quarterly_reduce_threshold=-0.06,
                    quarterly_protect_threshold=-0.10,
                    quarterly_cash_threshold=-0.16,
                ),
                concentration_config=None,
                asym_config=None,
                allocator_config=None,
                year_config=None,
            ),
        ),
        (
            "period_loss_guards_default",
            dict(
                period_config=PeriodLossGuardConfig(),
                concentration_config=None,
                asym_config=None,
                allocator_config=None,
                year_config=None,
            ),
        ),
        (
            "crypto_cap_default",
            dict(
                period_config=None,
                concentration_config=CryptoConcentrationConfig(),
                asym_config=None,
                allocator_config=None,
                year_config=None,
            ),
        ),
        (
            "asymmetric_exit_default",
            dict(
                period_config=None,
                concentration_config=None,
                asym_config=AsymmetricPolicyConfig(),
                allocator_config=None,
                year_config=None,
            ),
        ),
        (
            "regime_allocator_default",
            dict(
                period_config=None,
                concentration_config=None,
                asym_config=AsymmetricPolicyConfig(
                    enter_attack_threshold=0.72,
                    stay_attack_threshold=0.58,
                    defense_threshold=0.52,
                    release_threshold=0.64,
                ),
                allocator_config={
                    "neutral_confidence_threshold": 0.56,
                    "attack_full_confidence_threshold": 0.74,
                    "attack_full_stress_threshold": 0.42,
                    "attack_full_crypto_cap": 0.90,
                    "attack_partial_crypto_cap": 0.76,
                    "neutral_crypto_cap": 0.52,
                    "protected_crypto_cap": 0.32,
                },
                year_config=None,
            ),
        ),
        (
            "year_defense_default",
            dict(
                period_config=None,
                concentration_config=None,
                asym_config=None,
                allocator_config=None,
                year_config=YearDefenseConfig(),
            ),
        ),
        (
            "full_defense_light",
            dict(
                period_config=PeriodLossGuardConfig(
                    monthly_reduce_threshold=-0.045,
                    monthly_protect_threshold=-0.075,
                    monthly_cash_threshold=-0.12,
                    quarterly_reduce_threshold=-0.07,
                    quarterly_protect_threshold=-0.11,
                    quarterly_cash_threshold=-0.17,
                ),
                concentration_config=CryptoConcentrationConfig(
                    max_crypto_weight_normal=0.88,
                    max_crypto_weight_stressed=0.62,
                    crypto_risk_threshold=0.68,
                ),
                asym_config=AsymmetricPolicyConfig(
                    enter_attack_threshold=0.74,
                    stay_attack_threshold=0.60,
                    defense_threshold=0.54,
                    release_threshold=0.65,
                ),
                allocator_config={
                    "neutral_confidence_threshold": 0.57,
                    "attack_full_confidence_threshold": 0.76,
                    "attack_full_stress_threshold": 0.44,
                    "attack_full_crypto_cap": 0.88,
                    "attack_partial_crypto_cap": 0.74,
                    "neutral_crypto_cap": 0.50,
                    "protected_crypto_cap": 0.30,
                },
                year_config=YearDefenseConfig(
                    ytd_return_floor=-0.12,
                    ytd_drawdown_floor=-0.16,
                    stress_trigger=0.66,
                    min_bad_days=20,
                ),
            ),
        ),
        (
            "full_defense_default",
            dict(
                period_config=PeriodLossGuardConfig(),
                concentration_config=CryptoConcentrationConfig(),
                asym_config=AsymmetricPolicyConfig(),
                allocator_config={},
                year_config=YearDefenseConfig(),
            ),
        ),
    ]

    for candidate_id, kwargs in sim_specs:
        bundle, guard_log = _simulate_candidate(
            candidate_id=candidate_id,
            attack_alloc=attack_alloc,
            protect_alloc=protect_alloc,
            signal_frame=signal_frame,
            capital_brl=float(args.capital_brl),
            **kwargs,
        )
        candidates[candidate_id] = bundle
        guard_logs[candidate_id] = guard_log

    baseline = candidates["alpha_attack_major8_equity25"]
    compare_rows = [_candidate_row(bundle=bundle, baseline=baseline, capital_brl=float(args.capital_brl)) for bundle in candidates.values()]
    compare_df = pd.DataFrame(compare_rows).sort_values(["worth_keeping", "net_total_return", "net_sharpe"], ascending=[False, False, False])
    compare_df.to_csv(outdir / "candidate_compare.csv", index=False)

    yearbook_rows: list[dict[str, Any]] = []
    for bundle in candidates.values():
        yearbook_rows.extend(_calendar_rows(result=bundle.result, capital_brl=float(args.capital_brl)))
    yearbook_df = pd.DataFrame(yearbook_rows)
    yearbook_df.to_csv(outdir / "yearbook_reais.csv", index=False)

    year_improvement_df = yearbook_df.merge(
        yearbook_df[yearbook_df["candidate_id"] == "alpha_attack_major8_equity25"][["year", "profit_brl"]].rename(columns={"profit_brl": "baseline_profit_brl"}),
        on="year",
        how="left",
    )
    year_improvement_df["profit_delta_vs_attack_brl"] = pd.to_numeric(year_improvement_df["profit_brl"], errors="coerce") - pd.to_numeric(year_improvement_df["baseline_profit_brl"], errors="coerce")
    year_improvement_df.to_csv(outdir / "year_improvement.csv", index=False)

    for candidate_id, log_df in guard_logs.items():
        log_df.to_csv(outdir / f"{candidate_id}_guard_log.csv", index=True)

    best_profit = compare_df.sort_values(["net_total_return", "net_sharpe"], ascending=False).iloc[0].to_dict()
    keep_df = compare_df[compare_df["worth_keeping"] == True]  # noqa: E712
    best_bad_year = compare_df.sort_values(["years_negative", "worst_year_profit_brl", "max_recovery_days"], ascending=[True, False, True]).iloc[0].to_dict()
    summary = {
        "baseline_attack": compare_df[compare_df["candidate_id"] == "alpha_attack_major8_equity25"].head(1).to_dict(orient="records")[0],
        "protected_reference": compare_df[compare_df["candidate_id"] == "meta_major8_eq_a2r1_mc_guard"].head(1).to_dict(orient="records")[0],
        "best_profit_candidate": best_profit,
        "best_bad_year_candidate": best_bad_year,
        "worth_keeping_candidates": keep_df[["candidate_id", "candidate_label", "net_total_return", "years_negative", "worst_year_profit_brl"]].to_dict(orient="records"),
        "notes": [
            "As defesas so valem a pena se reduzirem anos ruins sem destruir o lucro acumulado.",
            "A comparacao principal e sempre contra o ataque atual.",
        ],
    }
    _write_json(outdir / "summary.json", summary)
    write_run_manifest(
        outdir,
        script="scripts/bench/validation/run_profit_bad_year_defense_suite.py",
        params={
            "capital_brl": float(args.capital_brl),
            "crypto_asset_groups": str(args.crypto_asset_groups),
            "crypto_asset_metadata": str(args.crypto_asset_metadata),
            "equity_asset_groups": str(args.equity_asset_groups),
            "equity_asset_metadata": str(args.equity_asset_metadata),
            "prices_dir": str(args.prices_dir),
        },
        paths={
            "summary": str(outdir / "summary.json"),
            "candidate_compare": str(outdir / "candidate_compare.csv"),
            "yearbook_reais": str(outdir / "yearbook_reais.csv"),
        },
        extra={
            "suite": "profit_bad_year_defense_suite",
            "candidates": list(candidates.keys()),
            "period_config_default": asdict(PeriodLossGuardConfig()),
            "concentration_config_default": asdict(CryptoConcentrationConfig()),
            "asymmetric_config_default": asdict(AsymmetricPolicyConfig()),
            "year_config_default": asdict(YearDefenseConfig()),
        },
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
