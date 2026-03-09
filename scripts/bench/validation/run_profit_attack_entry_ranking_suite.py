#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
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
from scripts.bench.validation.run_profit_alpha_improvement_suite import (  # noqa: E402
    _blend_allocations,
    _build_confidence_score,
    _safe_float,
    _write_json,
)
from scripts.bench.validation.run_profit_confidence_calibration_suite import _rolling_percentile  # noqa: E402
from scripts.bench.validation.run_profit_frontier_expansion_suite import (  # noqa: E402
    StrategyResult,
    _simulate_asset_rule,
)
from scripts.bench.validation.run_profit_layered_engine_suite import (  # noqa: E402
    StrategyBundle,
    _apply_breadth_overlay_to_bundle,
    _build_breadth_signal,
)
from scripts.bench.validation.run_profit_sector_pressure_suite import _research_row  # noqa: E402


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _result_row(result: StrategyResult, *, baseline: StrategyResult, family: str, label: str) -> dict[str, Any]:
    return {
        "candidate_id": str(result.candidate_id),
        "candidate_label": str(label),
        "family": str(family),
        "net_ann_return": _safe_float(result.net_ann_return),
        "net_total_return": _safe_float(result.net_total_return),
        "net_sharpe": _safe_float(result.net_sharpe),
        "net_max_drawdown": _safe_float(result.net_max_drawdown),
        "edge_vs_benchmark": _safe_float(result.edge_vs_benchmark),
        "avg_turnover_daily": _safe_float(result.avg_turnover_daily),
        "ann_return_improvement_pct": _pct_change(result.net_ann_return, baseline.net_ann_return),
        "total_return_improvement_pct": _pct_change(result.net_total_return, baseline.net_total_return),
        "sharpe_improvement_pct": _pct_change(result.net_sharpe, baseline.net_sharpe),
        "drawdown_change_pct": _pct_change(abs(float(result.net_max_drawdown)), abs(float(baseline.net_max_drawdown))),
        "worth_keeping_alpha": bool(_safe_float(result.net_total_return) > _safe_float(baseline.net_total_return)),
        "notes": str(result.notes or ""),
    }


def _pct_change(value: Any, base: Any) -> float:
    base_f = _safe_float(base)
    value_f = _safe_float(value)
    if not np.isfinite(base_f) or abs(base_f) <= 1e-12 or not np.isfinite(value_f):
        return float("nan")
    return float((value_f - base_f) / abs(base_f) * 100.0)


def _best_by_total_return(results: dict[str, StrategyResult]) -> StrategyResult:
    return max(results.values(), key=lambda result: (_safe_float(result.net_total_return), _safe_float(result.net_sharpe)))


def _bundle_from_result(result: StrategyResult, benchmark_gross_ret: pd.Series, profile) -> StrategyBundle:
    return StrategyBundle(
        result=result,
        benchmark_gross_ret=benchmark_gross_ret.reindex(result.gross_ret.index).fillna(0.0).astype(float),
        profile=profile,
        benchmark_profile=profile,
    )


def _weight_from_current_champion(score: pd.Series) -> pd.Series:
    transformed = _rolling_percentile(score, 126).fillna(score)
    weight = pd.Series(0.15, index=transformed.index, dtype=float)
    weight.loc[transformed >= 0.48] = 0.75
    weight.loc[transformed >= 0.63] = 1.00
    return weight.clip(0.0, 1.0)


def _wrap_with_current_confidence(
    *,
    candidate_id: str,
    context: dict[str, Any],
    crypto_bundle: StrategyBundle,
    attack_alloc: AllocationBundle,
    protect_alloc: AllocationBundle,
) -> StrategyResult:
    breadth_signal = _build_breadth_signal(
        returns=context["crypto_returns"],
        prices=context["crypto_prices"],
        tickers=context["crypto_tiers"]["crypto_all"],
        lookback_days=21,
        ma_days=200,
    )
    attack_returns = pd.concat(
        {
            "crypto": pd.to_numeric(crypto_bundle.result.gross_ret, errors="coerce"),
            "equity": pd.to_numeric(context["equity_attack"].result.gross_ret, errors="coerce"),
        },
        axis=1,
        sort=False,
    ).dropna(how="all")
    raw_score = _build_confidence_score(context, breadth_signal, attack_returns).clip(0.0, 1.0)
    attack_weight = _weight_from_current_champion(raw_score)
    bundle = _blend_allocations(
        candidate_id=str(candidate_id),
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        attack_weight=attack_weight,
    )
    return bundle.result


def _make_crypto_bundle(
    *,
    candidate_id: str,
    context: dict[str, Any],
    allowed_tickers: list[str],
    score_mode: str,
    lookback_days: int,
    rebalance_days: int,
    top_k: int,
    relative_to_benchmark: bool = False,
    skip_recent_days: int = 0,
) -> StrategyBundle:
    result = _simulate_asset_rule(
        candidate_id=f"{candidate_id}__crypto",
        family="attack_entry_ranking",
        allowed_tickers=list(allowed_tickers),
        returns=context["crypto_returns"],
        prices=context["crypto_prices"],
        asset_table=context["crypto_assets"],
        benchmark_ticker=context["benchmark_crypto"],
        fallback_ticker=context["benchmark_crypto"],
        score_mode=str(score_mode),
        lookback_days=int(lookback_days),
        rebalance_days=int(rebalance_days),
        top_k=int(top_k),
        asset_ma_days=0,
        market_ma_days=200,
        relative_to_benchmark=bool(relative_to_benchmark),
        skip_recent_days=int(skip_recent_days),
        trailing_stop_dd=None,
        hard_stop_loss=None,
        stop_to_cash=True,
        profile=context["profiles"]["crypto"],
        benchmark_profile=context["profiles"]["crypto"],
    )
    if result is None:
        raise SystemExit(f"falha ao simular sleeve cripto: {candidate_id}")
    benchmark = pd.to_numeric(
        context["crypto_returns"][str(context["benchmark_crypto"])], errors="coerce"
    ).reindex(result.gross_ret.index).fillna(0.0).astype(float)
    return _bundle_from_result(result, benchmark, context["profiles"]["crypto"])


def _build_attack_allocation(
    *,
    candidate_id: str,
    context: dict[str, Any],
    crypto_bundle: StrategyBundle,
    entry_lookback: int,
    exit_lookback: int,
    entry_margin: float,
    exit_margin: float,
    min_crypto_hold_days: int,
) -> AllocationBundle:
    return _build_alpha_meta_allocation_bundle(
        candidate_id=str(candidate_id),
        crypto_bundle=crypto_bundle,
        equity_bundle=context["equity_attack"],
        btc_prices=context["btc_prices"],
        spy_prices=context["spy_prices"],
        profile=context["profiles"]["blended"],
        entry_lookback=int(entry_lookback),
        exit_lookback=int(exit_lookback),
        entry_margin=float(entry_margin),
        exit_margin=float(exit_margin),
        risk_off_mode="equity25",
        min_crypto_hold_days=int(min_crypto_hold_days),
    )


def _family_ranking(*, context: dict[str, Any], protect_alloc: AllocationBundle) -> tuple[StrategyResult, list[dict[str, Any]], dict[str, StrategyBundle]]:
    major8 = context["crypto_tiers"]["crypto_major8"]
    all22 = context["crypto_tiers"]["crypto_all"]
    bundles: dict[str, StrategyBundle] = {}

    base_major8_k1 = _make_crypto_bundle(
        candidate_id="rank_major8_mom_total_lb21_rb07_k1",
        context=context,
        allowed_tickers=major8,
        score_mode="mom_total",
        lookback_days=21,
        rebalance_days=7,
        top_k=1,
    )
    bundles["rank_major8_mom_total_lb21_rb07_k1"] = base_major8_k1
    bundles["rank_major8_mom_total_lb21_rb07_k2"] = _make_crypto_bundle(
        candidate_id="rank_major8_mom_total_lb21_rb07_k2",
        context=context,
        allowed_tickers=major8,
        score_mode="mom_total",
        lookback_days=21,
        rebalance_days=7,
        top_k=2,
    )
    bundles["rank_major8_mom_vol_lb21_rb07_k2"] = _make_crypto_bundle(
        candidate_id="rank_major8_mom_vol_lb21_rb07_k2",
        context=context,
        allowed_tickers=major8,
        score_mode="mom_vol_adj",
        lookback_days=21,
        rebalance_days=7,
        top_k=2,
    )
    bundles["rank_major8_mom_total_lb21_rb05_k1"] = _make_crypto_bundle(
        candidate_id="rank_major8_mom_total_lb21_rb05_k1",
        context=context,
        allowed_tickers=major8,
        score_mode="mom_total",
        lookback_days=21,
        rebalance_days=5,
        top_k=1,
    )
    bundles["rank_major8_mom_total_lb42_rb07_k1"] = _make_crypto_bundle(
        candidate_id="rank_major8_mom_total_lb42_rb07_k1",
        context=context,
        allowed_tickers=major8,
        score_mode="mom_total",
        lookback_days=42,
        rebalance_days=7,
        top_k=1,
    )
    bundles["rank_major8_mom_total_lb21_rb14_k1"] = _make_crypto_bundle(
        candidate_id="rank_major8_mom_total_lb21_rb14_k1",
        context=context,
        allowed_tickers=major8,
        score_mode="mom_total",
        lookback_days=21,
        rebalance_days=14,
        top_k=1,
    )
    bundles["rank_major8_mom_total_lb21_rb07_k1_relbtc"] = _make_crypto_bundle(
        candidate_id="rank_major8_mom_total_lb21_rb07_k1_relbtc",
        context=context,
        allowed_tickers=major8,
        score_mode="mom_total",
        lookback_days=21,
        rebalance_days=7,
        top_k=1,
        relative_to_benchmark=True,
    )
    bundles["rank_major8_mom_total_lb21_rb07_k1_skip5"] = _make_crypto_bundle(
        candidate_id="rank_major8_mom_total_lb21_rb07_k1_skip5",
        context=context,
        allowed_tickers=major8,
        score_mode="mom_total",
        lookback_days=21,
        rebalance_days=7,
        top_k=1,
        skip_recent_days=5,
    )
    bundles["rank_all22_mom_total_lb21_rb07_k1"] = _make_crypto_bundle(
        candidate_id="rank_all22_mom_total_lb21_rb07_k1",
        context=context,
        allowed_tickers=all22,
        score_mode="mom_total",
        lookback_days=21,
        rebalance_days=7,
        top_k=1,
    )
    all22_k3 = _make_crypto_bundle(
        candidate_id="rank_all22_mom_vol_lb21_rb07_k3",
        context=context,
        allowed_tickers=all22,
        score_mode="mom_vol_adj",
        lookback_days=21,
        rebalance_days=7,
        top_k=3,
    )
    bundles["rank_all22_mom_vol_lb21_rb07_k3"] = all22_k3
    breadth_signal = _build_breadth_signal(
        returns=context["crypto_returns"],
        prices=context["crypto_prices"],
        tickers=all22,
        lookback_days=21,
        ma_days=200,
    )
    bundles["rank_all22_mom_vol_lb21_rb07_k3_breadth"] = _apply_breadth_overlay_to_bundle(
        candidate_id="rank_all22_mom_vol_lb21_rb07_k3_breadth",
        bundle=all22_k3,
        breadth_signal=breadth_signal,
        low_threshold=0.38,
        high_threshold=0.62,
        mode="scale",
    )

    baseline_attack_alloc = _build_attack_allocation(
        candidate_id="ranking_baseline_attack",
        context=context,
        crypto_bundle=base_major8_k1,
        entry_lookback=21,
        exit_lookback=63,
        entry_margin=0.05,
        exit_margin=0.05,
        min_crypto_hold_days=0,
    )
    baseline_current = _wrap_with_current_confidence(
        candidate_id="current_champion",
        context=context,
        crypto_bundle=base_major8_k1,
        attack_alloc=baseline_attack_alloc,
        protect_alloc=protect_alloc,
    )

    rows: list[dict[str, Any]] = []
    results: dict[str, StrategyResult] = {}
    for cid, crypto_bundle in bundles.items():
        attack_alloc = _build_attack_allocation(
            candidate_id=f"{cid}__attack",
            context=context,
            crypto_bundle=crypto_bundle,
            entry_lookback=21,
            exit_lookback=63,
            entry_margin=0.05,
            exit_margin=0.05,
            min_crypto_hold_days=0,
        )
        result = _wrap_with_current_confidence(
            candidate_id=f"{cid}__wrapped",
            context=context,
            crypto_bundle=crypto_bundle,
            attack_alloc=attack_alloc,
            protect_alloc=protect_alloc,
        )
        rows.append(_result_row(result, baseline=baseline_current, family="ranking", label=cid))
        results[cid] = result

    return baseline_current, rows, bundles


def _family_entry(
    *,
    context: dict[str, Any],
    protect_alloc: AllocationBundle,
    base_crypto_bundle: StrategyBundle,
) -> tuple[list[dict[str, Any]], dict[str, StrategyResult]]:
    baseline_attack_alloc = _build_attack_allocation(
        candidate_id="entry_baseline_attack",
        context=context,
        crypto_bundle=base_crypto_bundle,
        entry_lookback=21,
        exit_lookback=63,
        entry_margin=0.05,
        exit_margin=0.05,
        min_crypto_hold_days=0,
    )
    baseline_current = _wrap_with_current_confidence(
        candidate_id="current_champion",
        context=context,
        crypto_bundle=base_crypto_bundle,
        attack_alloc=baseline_attack_alloc,
        protect_alloc=protect_alloc,
    )
    configs = [
        ("entry_fast14_exit63_m2_h0", 14, 63, 0.02, 0.05, 0),
        ("entry_fast14_exit63_m2_h3", 14, 63, 0.02, 0.05, 3),
        ("entry_fast14_exit84_m2_h3", 14, 84, 0.02, 0.05, 3),
        ("entry_fast14_exit84_m3_h0", 14, 84, 0.03, 0.05, 0),
        ("entry_21_exit84_m2_h0", 21, 84, 0.02, 0.05, 0),
        ("entry_21_exit84_m2_h3", 21, 84, 0.02, 0.05, 3),
        ("entry_21_exit63_m2_h0", 21, 63, 0.02, 0.03, 0),
        ("entry_21_exit63_m3_h0", 21, 63, 0.03, 0.03, 0),
        ("entry_42_exit84_m2_h0", 42, 84, 0.02, 0.05, 0),
        ("entry_42_exit84_m5_h0", 42, 84, 0.05, 0.05, 0),
    ]
    rows: list[dict[str, Any]] = []
    results: dict[str, StrategyResult] = {}
    for cid, entry_lb, exit_lb, entry_margin, exit_margin, hold_days in configs:
        attack_alloc = _build_attack_allocation(
            candidate_id=f"{cid}__attack",
            context=context,
            crypto_bundle=base_crypto_bundle,
            entry_lookback=entry_lb,
            exit_lookback=exit_lb,
            entry_margin=entry_margin,
            exit_margin=exit_margin,
            min_crypto_hold_days=hold_days,
        )
        result = _wrap_with_current_confidence(
            candidate_id=f"{cid}__wrapped",
            context=context,
            crypto_bundle=base_crypto_bundle,
            attack_alloc=attack_alloc,
            protect_alloc=protect_alloc,
        )
        rows.append(_result_row(result, baseline=baseline_current, family="entry", label=cid))
        results[cid] = result
    return rows, results


def _family_combo(
    *,
    context: dict[str, Any],
    protect_alloc: AllocationBundle,
    bundles: dict[str, StrategyBundle],
) -> tuple[list[dict[str, Any]], dict[str, StrategyResult]]:
    base_crypto_bundle = bundles["rank_major8_mom_total_lb21_rb07_k1"]
    baseline_attack_alloc = _build_attack_allocation(
        candidate_id="combo_baseline_attack",
        context=context,
        crypto_bundle=base_crypto_bundle,
        entry_lookback=21,
        exit_lookback=63,
        entry_margin=0.05,
        exit_margin=0.05,
        min_crypto_hold_days=0,
    )
    baseline_current = _wrap_with_current_confidence(
        candidate_id="current_champion",
        context=context,
        crypto_bundle=base_crypto_bundle,
        attack_alloc=baseline_attack_alloc,
        protect_alloc=protect_alloc,
    )
    combos = [
        ("combo_rank_k2_fast14", "rank_major8_mom_total_lb21_rb07_k2", 14, 63, 0.02, 0.05, 0),
        ("combo_rank_k2_hold3", "rank_major8_mom_total_lb21_rb07_k2", 14, 84, 0.02, 0.05, 3),
        ("combo_rank_lb42_exit84", "rank_major8_mom_total_lb42_rb07_k1", 21, 84, 0.02, 0.05, 0),
        ("combo_rank_rb14_exit84", "rank_major8_mom_total_lb21_rb14_k1", 21, 84, 0.02, 0.05, 0),
        ("combo_rank_relbtc_fast", "rank_major8_mom_total_lb21_rb07_k1_relbtc", 14, 63, 0.02, 0.05, 0),
        ("combo_rank_skip5_fast", "rank_major8_mom_total_lb21_rb07_k1_skip5", 14, 63, 0.02, 0.05, 0),
        ("combo_all22_breadth_fast", "rank_all22_mom_vol_lb21_rb07_k3_breadth", 14, 63, 0.02, 0.05, 0),
    ]
    rows: list[dict[str, Any]] = []
    results: dict[str, StrategyResult] = {}
    for cid, bundle_key, entry_lb, exit_lb, entry_margin, exit_margin, hold_days in combos:
        crypto_bundle = bundles[bundle_key]
        attack_alloc = _build_attack_allocation(
            candidate_id=f"{cid}__attack",
            context=context,
            crypto_bundle=crypto_bundle,
            entry_lookback=entry_lb,
            exit_lookback=exit_lb,
            entry_margin=entry_margin,
            exit_margin=exit_margin,
            min_crypto_hold_days=hold_days,
        )
        result = _wrap_with_current_confidence(
            candidate_id=f"{cid}__wrapped",
            context=context,
            crypto_bundle=crypto_bundle,
            attack_alloc=attack_alloc,
            protect_alloc=protect_alloc,
        )
        rows.append(_result_row(result, baseline=baseline_current, family="combo", label=cid))
        results[cid] = result
    return rows, results


def main() -> None:
    ap = argparse.ArgumentParser(description="Compara melhorias de ranking e entrada no ataque contra o campeao atual.")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--outdir-root", default="results/validation/profit_attack_entry_ranking_suite")
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
    protect_alloc = built["allocations"]["baseline_guard"]

    baseline_current, ranking_rows, ranking_bundles = _family_ranking(context=context, protect_alloc=protect_alloc)
    entry_rows, entry_results = _family_entry(
        context=context,
        protect_alloc=protect_alloc,
        base_crypto_bundle=ranking_bundles["rank_major8_mom_total_lb21_rb07_k1"],
    )
    combo_rows, combo_results = _family_combo(
        context=context,
        protect_alloc=protect_alloc,
        bundles=ranking_bundles,
    )

    ranking_best = max(ranking_rows, key=lambda row: (row["net_total_return"], row["net_sharpe"]))
    entry_best = max(entry_rows, key=lambda row: (row["net_total_return"], row["net_sharpe"]))
    combo_best = max(combo_rows, key=lambda row: (row["net_total_return"], row["net_sharpe"]))
    best_overall = max([ranking_best, entry_best, combo_best], key=lambda row: (row["net_total_return"], row["net_sharpe"]))

    candidate_rows = [
        _result_row(baseline_current, baseline=baseline_current, family="baseline", label="confidence_current_champion"),
        *ranking_rows,
        *entry_rows,
        *combo_rows,
    ]
    candidate_df = pd.DataFrame(candidate_rows).sort_values(
        ["net_total_return", "net_sharpe"], ascending=[False, False]
    )
    candidate_path = outdir / "candidate_compare.csv"
    candidate_df.to_csv(candidate_path, index=False)

    family_winners = pd.DataFrame([ranking_best, entry_best, combo_best])
    family_winners_path = outdir / "family_winners.csv"
    family_winners.to_csv(family_winners_path, index=False)

    insights = [
        "A busca compara melhorias no ranking do sleeve de ataque e no gatilho de entrada no cripto contra o campeao atual de confianca.",
        "Ranking mexe no que o ataque compra; entrada mexe em quando o meta entra forte no cripto.",
        "Worth_keeping_alpha so marca variantes que melhoram o lucro final contra o campeao atual.",
    ]
    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_current_champion": candidate_rows[0],
        "best_ranking": dict(ranking_best),
        "best_entry": dict(entry_best),
        "best_combo": dict(combo_best),
        "best_overall": dict(best_overall),
        "insights": insights,
        "artifacts": {
            "candidate_compare_csv": str(candidate_path),
            "family_winners_csv": str(family_winners_path),
        },
    }
    _write_json(outdir / "summary.json", summary)

    research_rows = [
        _research_row(
            baseline_current,
            outdir=outdir,
            status="keep",
            methodology="attack_entry_ranking_baseline",
            label="Campeao atual de confianca",
        )
    ]
    for row in [ranking_best, entry_best, combo_best]:
        status = "watch" if bool(row["worth_keeping_alpha"]) else "kill"
        methodology = f"attack_entry_ranking_{row['family']}"
        research_rows.append(
            {
                **_research_row(
                    StrategyResult(
                        suite="attack_entry_ranking",
                        candidate_id=str(row["candidate_id"]),
                        family=str(row["family"]),
                        benchmark_ticker=str(baseline_current.benchmark_ticker),
                        gross_ret=baseline_current.gross_ret,
                        turnover=baseline_current.turnover,
                        net_ret=baseline_current.net_ret,
                        benchmark_net_ret=baseline_current.benchmark_net_ret,
                        net_ann_return=float(row["net_ann_return"]),
                        net_total_return=float(row["net_total_return"]),
                        net_sharpe=float(row["net_sharpe"]),
                        net_max_drawdown=float(row["net_max_drawdown"]),
                        edge_vs_benchmark=float(row["edge_vs_benchmark"]),
                        avg_turnover_daily=float(row["avg_turnover_daily"]),
                        hit_rate_10x_5y=float("nan"),
                        years_to_10x_full=float("nan"),
                        notes=str(row["notes"]),
                    ),
                    outdir=outdir,
                    status=status,
                    methodology=methodology,
                    label=str(row["candidate_label"]),
                ),
                "notes": str(row["notes"]),
            }
        )
    research_df = pd.DataFrame(research_rows)
    research_path = outdir / "research_rows.csv"
    research_df.to_csv(research_path, index=False)
    _write_json(outdir / "research_rows.json", {"rows": research_rows})
    _write_json(outdir / "profit_research_rows.json", research_rows)

    write_run_manifest(
        outdir,
        script="scripts/bench/validation/run_profit_attack_entry_ranking_suite.py",
        params=vars(args),
        paths={
            "summary_json": "summary.json",
            "candidate_compare_csv": "candidate_compare.csv",
            "family_winners_csv": "family_winners.csv",
            "research_rows_csv": "research_rows.csv",
            "profit_research_rows_json": "profit_research_rows.json",
        },
        extra={
            "notes": [
                "Compara melhorias de ranking dentro do ataque e de gatilho de entrada no cripto contra o campeao atual de confianca.",
            ]
        },
    )


if __name__ == "__main__":
    main()
