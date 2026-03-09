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

from engine.structural.run_manifest import write_run_manifest  # noqa: E402
from scripts.bench.validation.run_profit_alpha_hardening_suite import (  # noqa: E402
    AllocationBundle,
    _build_alpha_meta_allocation_bundle,
    _build_candidates,
)
from scripts.bench.validation.run_profit_crypto_resolution_suite import (  # noqa: E402
    _blend_crypto_bundles,
    _crypto_rule_bundle,
)
from scripts.bench.validation.run_profit_frontier_expansion_suite import (  # noqa: E402
    StrategyResult,
    _evaluate_net,
    _safe_float,
    _write_json,
)
from scripts.bench.validation.run_profit_layered_engine_suite import (  # noqa: E402
    StrategyBundle,
    _apply_breadth_overlay_to_bundle,
    _build_breadth_signal,
)
from scripts.bench.validation.run_profit_regime_error_suite import (  # noqa: E402
    _combine_candidate,
    _selector_with_inertia,
)
from scripts.bench.validation.run_profit_sleeve_sizing_synthetic_suite import (  # noqa: E402
    _bundle_from_sleeves,
)


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _tail_return(series: pd.Series, lookback: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    min_periods = max(10, int(lookback) // 3)
    return (1.0 + values).rolling(int(lookback), min_periods=min_periods).apply(np.prod, raw=True) - 1.0


def _human_label(candidate_id: str) -> str:
    mapping = {
        "alpha_attack_major8_equity25": "Ataque atual",
        "crypto_layers__blend60_major8": "Cripto em camadas",
        "confidence_attack__thr60": "Ataque só com confiança alta",
        "less_anxious_exit__hold5_confirm3": "Saída menos ansiosa",
        "crypto_selection__all22_breadth": "Seleção cripto com breadth",
        "equity_leg__base_eq25": "Perna de ações mais forte",
        "confidence_sizing__high100_mid70_low20": "Sizing por confiança",
        "dead_periods__eq50_entry0_exit3": "Redução de períodos mortos",
    }
    return mapping.get(str(candidate_id), str(candidate_id))


def _percent_change(base: float, value: float) -> float:
    if not np.isfinite(float(base)) or abs(float(base)) <= 1e-12:
        return float("nan")
    return float((float(value) - float(base)) / abs(float(base)) * 100.0)


def _result_row(result: StrategyResult, *, baseline: StrategyResult) -> dict[str, Any]:
    return {
        "candidate_id": str(result.candidate_id),
        "candidate_label": _human_label(result.candidate_id),
        "net_ann_return": _safe_float(result.net_ann_return),
        "net_total_return": _safe_float(result.net_total_return),
        "net_sharpe": _safe_float(result.net_sharpe),
        "net_max_drawdown": _safe_float(result.net_max_drawdown),
        "edge_vs_benchmark": _safe_float(result.edge_vs_benchmark),
        "avg_turnover_daily": _safe_float(result.avg_turnover_daily),
        "ann_return_improvement_pct": _percent_change(baseline.net_ann_return, result.net_ann_return),
        "total_return_improvement_pct": _percent_change(baseline.net_total_return, result.net_total_return),
        "sharpe_improvement_pct": _percent_change(baseline.net_sharpe, result.net_sharpe),
        "drawdown_change_pct": _percent_change(abs(float(baseline.net_max_drawdown)), abs(float(result.net_max_drawdown))),
        "worth_keeping_alpha": bool(_safe_float(result.net_total_return) > _safe_float(baseline.net_total_return)),
        "notes": str(result.notes or ""),
    }


def _strategy_bundle_from_result(result: StrategyResult, benchmark_gross_ret: pd.Series, profile) -> StrategyBundle:
    return StrategyBundle(
        result=result,
        benchmark_gross_ret=benchmark_gross_ret.reindex(result.gross_ret.index).fillna(0.0).astype(float),
        profile=profile,
        benchmark_profile=profile,
    )


def _build_confidence_score(context: dict[str, Any], breadth_signal: pd.Series, attack_returns: pd.DataFrame) -> pd.Series:
    idx = (
        breadth_signal.index.intersection(attack_returns.index)
        .intersection(context["btc_prices"].index)
        .intersection(context["spy_prices"].index)
        .intersection(context["regime_series"].index)
    )
    breadth = pd.to_numeric(breadth_signal.reindex(idx), errors="coerce").fillna(0.0).clip(0.0, 1.0).astype(float)
    crypto_ret = pd.to_numeric(attack_returns["crypto"].reindex(idx), errors="coerce").fillna(0.0).astype(float)
    equity_ret = pd.to_numeric(attack_returns["equity"].reindex(idx), errors="coerce").fillna(0.0).astype(float)
    crypto_fast = _tail_return(crypto_ret, 21)
    equity_fast = _tail_return(equity_ret, 21)
    btc = pd.to_numeric(context["btc_prices"].reindex(idx), errors="coerce").astype(float)
    spy = pd.to_numeric(context["spy_prices"].reindex(idx), errors="coerce").astype(float)
    btc_ok = (btc.shift(1) > btc.shift(1).rolling(200, min_periods=100).mean()).fillna(False).astype(float)
    spy_ok = (spy.shift(1) > spy.shift(1).rolling(200, min_periods=100).mean()).fillna(False).astype(float)
    regime = (
        pd.Series(context["regime_series"], index=context["regime_series"].index)
        .reindex(idx)
        .ffill()
        .bfill()
        .astype(str)
        .str.lower()
    )
    structural_clean = regime.isin(["stable", "dispersion"]).astype(float)
    spread = ((crypto_fast - equity_fast + 0.08) / 0.16).clip(0.0, 1.0)
    score = 0.10 + 0.30 * breadth + 0.24 * structural_clean + 0.18 * btc_ok + 0.08 * spy_ok + 0.10 * spread
    return score.clip(0.0, 1.0).astype(float)


def _blend_allocations(
    *,
    candidate_id: str,
    attack_alloc: AllocationBundle,
    protect_alloc: AllocationBundle,
    attack_weight: pd.Series,
) -> StrategyBundle:
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
        suite="alpha_improvement",
        candidate_id=str(candidate_id),
        family="alpha_improvement_blend",
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
        notes="blend ataque/protecao por peso historico de confianca",
    )
    return _strategy_bundle_from_result(result, benchmark, attack_alloc.bundle.profile)


def _best_by_total_return(candidates: dict[str, StrategyResult]) -> StrategyResult:
    return max(candidates.values(), key=lambda result: (_safe_float(result.net_total_return), _safe_float(result.net_sharpe)))


def _build_family_candidates(context: dict[str, Any], built: dict[str, Any]) -> tuple[dict[str, StrategyResult], dict[str, list[dict[str, Any]]]]:
    attack_alloc = built["allocations"]["attack"]
    protect_alloc = built["allocations"]["baseline_guard"]
    attack_returns = built["sleeve_returns"]["attack"]
    profiles = context["profiles"]

    crypto_tiers = context["crypto_tiers"]
    attack_top1 = _crypto_rule_bundle(
        candidate_id="attack_major8_k1",
        allowed_tickers=crypto_tiers["crypto_major8"],
        score_mode="mom_total",
        top_k=1,
        context=context,
    )
    crypto_major8_k2 = _crypto_rule_bundle(
        candidate_id="attack_major8_k2",
        allowed_tickers=crypto_tiers["crypto_major8"],
        score_mode="mom_total",
        top_k=2,
        context=context,
    )
    crypto_major8_k3 = _crypto_rule_bundle(
        candidate_id="div_major8_k3",
        allowed_tickers=crypto_tiers["crypto_major8"],
        score_mode="mom_vol_adj",
        top_k=3,
        context=context,
    )
    crypto_all22_k3 = _crypto_rule_bundle(
        candidate_id="div_all22_k3",
        allowed_tickers=crypto_tiers["crypto_all"],
        score_mode="mom_vol_adj",
        top_k=3,
        context=context,
    )
    breadth_signal = _build_breadth_signal(
        returns=context["crypto_returns"],
        prices=context["crypto_prices"],
        tickers=crypto_tiers["crypto_all"],
        lookback_days=21,
        ma_days=200,
    )
    crypto_all22_breadth = _apply_breadth_overlay_to_bundle(
        candidate_id="div_all22_k3_breadth",
        bundle=crypto_all22_k3,
        breadth_signal=breadth_signal,
        low_threshold=0.38,
        high_threshold=0.62,
        mode="scale",
    )
    confidence_score = _build_confidence_score(context, breadth_signal, attack_returns)

    family_variants: dict[str, list[dict[str, Any]]] = {}
    family_winners: dict[str, StrategyResult] = {}

    # 1. Cripto em camadas
    layered_variants: dict[str, StrategyResult] = {}
    for weight, satellite in [
        (0.80, crypto_major8_k3),
        (0.70, crypto_major8_k3),
        (0.60, crypto_major8_k3),
        (0.70, crypto_all22_breadth),
        (0.60, crypto_all22_breadth),
    ]:
        bundle = _blend_crypto_bundles(
            candidate_id=f"crypto_layers__w{int(weight*100):02d}",
            primary=attack_top1,
            secondary=satellite,
            primary_weight=weight,
        )
        result = _build_alpha_meta_allocation_bundle(
            candidate_id=f"crypto_layers__blend{int(weight*100):02d}_{satellite.result.candidate_id}",
            crypto_bundle=bundle,
            equity_bundle=context["equity_attack"],
            btc_prices=context["btc_prices"],
            spy_prices=context["spy_prices"],
            profile=profiles["blended"],
            entry_lookback=21,
            exit_lookback=63,
            entry_margin=0.05,
            exit_margin=0.05,
            risk_off_mode="equity25",
            min_crypto_hold_days=0,
        ).bundle.result
        layered_variants[result.candidate_id] = result
    family_winners["crypto_layers"] = _best_by_total_return(layered_variants)
    family_variants["crypto_layers"] = list(layered_variants.values())

    # 2. Ataque com confiança alta
    high_conf_variants: dict[str, StrategyResult] = {}
    for threshold in [0.58, 0.60, 0.62, 0.65]:
        attack_choice = (confidence_score >= float(threshold)).map({True: "attack", False: "protect"})
        result = _combine_candidate(
            candidate_id=f"confidence_attack__thr{int(round(threshold*100))}",
            choice=attack_choice,
            attack_alloc=attack_alloc,
            protect_alloc=protect_alloc,
            profile=profiles["blended"],
        )
        high_conf_variants[result["candidate_id"]] = StrategyResult(
            suite="alpha_improvement",
            candidate_id=result["candidate_id"],
            family="confidence_attack",
            benchmark_ticker=attack_alloc.bundle.result.benchmark_ticker,
            gross_ret=result["gross_ret"],
            turnover=result["turnover"],
            net_ret=result["net_ret"],
            benchmark_net_ret=result["benchmark_net_ret"],
            net_ann_return=result["net_ann_return"],
            net_total_return=result["net_total_return"],
            net_sharpe=result["net_sharpe"],
            net_max_drawdown=result["net_max_drawdown"],
            edge_vs_benchmark=result["edge_vs_benchmark"],
            avg_turnover_daily=result["avg_turnover_daily"],
            hit_rate_10x_5y=float("nan"),
            years_to_10x_full=float("nan"),
            notes=f"attack_only_if_confidence_ge_{threshold:.2f}",
        )
    family_winners["confidence_attack"] = _best_by_total_return(high_conf_variants)
    family_variants["confidence_attack"] = list(high_conf_variants.values())

    # 3. Saída menos ansiosa
    less_anxious_variants: dict[str, StrategyResult] = {}
    base_choice = (confidence_score >= 0.60).map({True: "attack", False: "protect"})
    for hold, confirm in [(3, 2), (5, 2), (5, 3), (7, 3)]:
        inertia_choice = _selector_with_inertia(base_choice, min_hold_days=hold, confirm_days=confirm)
        result = _combine_candidate(
            candidate_id=f"less_anxious_exit__hold{hold}_confirm{confirm}",
            choice=inertia_choice,
            attack_alloc=attack_alloc,
            protect_alloc=protect_alloc,
            profile=profiles["blended"],
        )
        less_anxious_variants[result["candidate_id"]] = StrategyResult(
            suite="alpha_improvement",
            candidate_id=result["candidate_id"],
            family="less_anxious_exit",
            benchmark_ticker=attack_alloc.bundle.result.benchmark_ticker,
            gross_ret=result["gross_ret"],
            turnover=result["turnover"],
            net_ret=result["net_ret"],
            benchmark_net_ret=result["benchmark_net_ret"],
            net_ann_return=result["net_ann_return"],
            net_total_return=result["net_total_return"],
            net_sharpe=result["net_sharpe"],
            net_max_drawdown=result["net_max_drawdown"],
            edge_vs_benchmark=result["edge_vs_benchmark"],
            avg_turnover_daily=result["avg_turnover_daily"],
            hit_rate_10x_5y=float("nan"),
            years_to_10x_full=float("nan"),
            notes=f"hold={hold};confirm={confirm}",
        )
    family_winners["less_anxious_exit"] = _best_by_total_return(less_anxious_variants)
    family_variants["less_anxious_exit"] = list(less_anxious_variants.values())

    # 4. Melhor seleção dentro do cripto
    crypto_selection_variants: dict[str, StrategyResult] = {}
    for crypto_bundle, label in [
        (crypto_major8_k2, "major8_k2"),
        (crypto_major8_k3, "major8_k3"),
        (crypto_all22_k3, "all22_k3"),
        (crypto_all22_breadth, "all22_breadth"),
    ]:
        result = _build_alpha_meta_allocation_bundle(
            candidate_id=f"crypto_selection__{label}",
            crypto_bundle=crypto_bundle,
            equity_bundle=context["equity_attack"],
            btc_prices=context["btc_prices"],
            spy_prices=context["spy_prices"],
            profile=profiles["blended"],
            entry_lookback=21,
            exit_lookback=63,
            entry_margin=0.05,
            exit_margin=0.05,
            risk_off_mode="equity25",
            min_crypto_hold_days=0,
        ).bundle.result
        crypto_selection_variants[result.candidate_id] = result
    family_winners["crypto_selection"] = _best_by_total_return(crypto_selection_variants)
    family_variants["crypto_selection"] = list(crypto_selection_variants.values())

    # 5. Perna de ações mais forte
    equity_leg_variants: dict[str, StrategyResult] = {}
    for equity_bundle, risk_off_mode, entry_margin, exit_margin, label in [
        (context["equity_base"], "equity25", 0.05, 0.05, "base_eq25"),
        (context["equity_base"], "equity50", 0.05, 0.05, "base_eq50"),
        (context["equity_base"], "equity25", 0.02, 0.05, "base_eq25_fast"),
    ]:
        result = _build_alpha_meta_allocation_bundle(
            candidate_id=f"equity_leg__{label}",
            crypto_bundle=attack_top1,
            equity_bundle=equity_bundle,
            btc_prices=context["btc_prices"],
            spy_prices=context["spy_prices"],
            profile=profiles["blended"],
            entry_lookback=21,
            exit_lookback=63,
            entry_margin=entry_margin,
            exit_margin=exit_margin,
            risk_off_mode=risk_off_mode,
            min_crypto_hold_days=0,
        ).bundle.result
        equity_leg_variants[result.candidate_id] = result
    family_winners["equity_leg"] = _best_by_total_return(equity_leg_variants)
    family_variants["equity_leg"] = list(equity_leg_variants.values())

    # 6. Sizing por confiança
    confidence_sizing_variants: dict[str, StrategyResult] = {}
    for high, medium, low, hi_th, med_th, label in [
        (1.0, 0.70, 0.20, 0.65, 0.50, "high100_mid70_low20"),
        (1.0, 0.65, 0.15, 0.62, 0.48, "high100_mid65_low15"),
        (0.90, 0.60, 0.20, 0.60, 0.45, "high90_mid60_low20"),
    ]:
        attack_weight = pd.Series(low, index=confidence_score.index, dtype=float)
        attack_weight.loc[confidence_score >= med_th] = medium
        attack_weight.loc[confidence_score >= hi_th] = high
        bundle = _blend_allocations(
            candidate_id=f"confidence_sizing__{label}",
            attack_alloc=attack_alloc,
            protect_alloc=protect_alloc,
            attack_weight=attack_weight,
        )
        confidence_sizing_variants[bundle.result.candidate_id] = bundle.result
    family_winners["confidence_sizing"] = _best_by_total_return(confidence_sizing_variants)
    family_variants["confidence_sizing"] = list(confidence_sizing_variants.values())

    # 7. Redução de períodos mortos
    dead_period_variants: dict[str, StrategyResult] = {}
    for risk_off_mode, entry_margin, exit_margin, hold, label in [
        ("equity50", 0.00, 0.03, 0, "eq50_entry0_exit3"),
        ("equity50", 0.02, 0.03, 0, "eq50_entry2_exit3"),
        ("equity25", 0.00, 0.03, 0, "eq25_entry0_exit3"),
        ("equity50", 0.00, 0.05, 3, "eq50_entry0_hold3"),
    ]:
        result = _build_alpha_meta_allocation_bundle(
            candidate_id=f"dead_periods__{label}",
            crypto_bundle=attack_top1,
            equity_bundle=context["equity_attack"],
            btc_prices=context["btc_prices"],
            spy_prices=context["spy_prices"],
            profile=profiles["blended"],
            entry_lookback=21,
            exit_lookback=63,
            entry_margin=entry_margin,
            exit_margin=exit_margin,
            risk_off_mode=risk_off_mode,
            min_crypto_hold_days=hold,
        ).bundle.result
        dead_period_variants[result.candidate_id] = result
    family_winners["dead_periods"] = _best_by_total_return(dead_period_variants)
    family_variants["dead_periods"] = list(dead_period_variants.values())

    return family_winners, family_variants


def main() -> None:
    ap = argparse.ArgumentParser(description="Compara 7 frentes para tentar aumentar alpha/lucro final do motor.")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--outdir-root", default="results/validation/profit_alpha_improvement_suite")
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
    context = dict(built["context"])
    baseline_attack = built["attack"].result
    family_winners, family_variants = _build_family_candidates(context, built)

    rows = [_result_row(result, baseline=baseline_attack) for result in family_winners.values()]
    compare_df = pd.DataFrame(rows).sort_values(
        ["net_total_return", "net_sharpe"],
        ascending=[False, False],
    ).reset_index(drop=True)
    compare_df.to_csv(outdir / "candidate_compare.csv", index=False)

    variant_rows: list[dict[str, Any]] = []
    for family, variants in family_variants.items():
        for result in variants:
            row = _result_row(result, baseline=baseline_attack)
            row["family"] = str(family)
            variant_rows.append(row)
    variants_df = pd.DataFrame(variant_rows).sort_values(
        ["family", "net_total_return", "net_sharpe"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    variants_df.to_csv(outdir / "family_variants.csv", index=False)

    top = compare_df.iloc[0].to_dict() if not compare_df.empty else {}
    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_attack": _result_row(baseline_attack, baseline=baseline_attack),
        "best_family_winner": top,
        "family_winners": rows,
        "insights": [
            "A comparação usa o modo de ataque atual como referência para medir ganho percentual de lucro, Sharpe e drawdown.",
            "Cada frente foi testada com pequenas variações internas e a melhor de cada família foi escolhida pelo lucro final.",
            "Worth_keeping_alpha marca só quem realmente melhorou o lucro final contra o ataque atual.",
        ],
        "artifacts": {
            "candidate_compare_csv": str(outdir / "candidate_compare.csv"),
            "family_variants_csv": str(outdir / "family_variants.csv"),
        },
    }
    _write_json(outdir / "summary.json", summary)
    write_run_manifest(
        outdir / "RUN_MANIFEST.json",
        script="scripts/bench/validation/run_profit_alpha_improvement_suite.py",
        params={
            "crypto_asset_groups": str(args.crypto_asset_groups),
            "crypto_asset_metadata": str(args.crypto_asset_metadata),
            "equity_asset_groups": str(args.equity_asset_groups),
            "equity_asset_metadata": str(args.equity_asset_metadata),
            "prices_dir": str(args.prices_dir),
            "benchmark_crypto": str(args.benchmark_crypto),
            "benchmark_equity": str(args.benchmark_equity),
        },
        paths={
            "summary_json": str(outdir / "summary.json"),
            "candidate_compare_csv": str(outdir / "candidate_compare.csv"),
            "family_variants_csv": str(outdir / "family_variants.csv"),
        },
        extra={
            "suite": "profit_alpha_improvement_suite",
            "baseline_candidate_id": str(baseline_attack.candidate_id),
            "family_count": int(len(family_winners)),
            "top_candidate_id": str(top.get("candidate_id", "")),
        },
        repo_root=ROOT,
    )
    print(json.dumps({"status": "ok", "outdir": str(outdir), "top_candidate_id": top.get("candidate_id", "")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
