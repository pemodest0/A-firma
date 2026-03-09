#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from engine.structural.run_manifest import write_run_manifest  # noqa: E402
from scripts.bench.validation.run_profit_alpha_hardening_suite import (  # noqa: E402
    _build_alpha_meta_allocation_bundle,
    _build_candidates,
)
from scripts.bench.validation.run_profit_frontier_expansion_suite import (  # noqa: E402
    StrategyResult,
    _evaluate_net,
    _rolling_ten_x_stats,
    _simulate_asset_rule,
    _write_json,
)
from scripts.bench.validation.run_profit_layered_engine_suite import (  # noqa: E402
    StrategyBundle,
    _apply_breadth_overlay_to_bundle,
    _build_breadth_signal,
)
from scripts.bench.validation.run_profit_sector_pressure_suite import _research_row  # noqa: E402
from scripts.bench.validation.run_profit_universe_resilience_suite import (  # noqa: E402
    _human_label,
    _selection_frequency_for_crypto_rule,
)


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _result_row(result: StrategyResult) -> dict[str, Any]:
    return {
        "candidate_id": str(result.candidate_id),
        "candidate_label": _human_label(str(result.candidate_id)),
        "net_ann_return": _safe_float(result.net_ann_return),
        "net_total_return": _safe_float(result.net_total_return),
        "net_sharpe": _safe_float(result.net_sharpe),
        "net_max_drawdown": _safe_float(result.net_max_drawdown),
        "edge_vs_benchmark": _safe_float(result.edge_vs_benchmark),
        "avg_turnover_daily": _safe_float(result.avg_turnover_daily),
        "notes": str(result.notes or ""),
    }


def _crypto_rule_bundle(
    *,
    candidate_id: str,
    allowed_tickers: list[str],
    score_mode: str,
    top_k: int,
    context: dict[str, Any],
) -> StrategyBundle:
    crypto_result = _simulate_asset_rule(
        candidate_id=f"{candidate_id}__crypto",
        family="crypto_resolution",
        allowed_tickers=list(allowed_tickers),
        returns=context["crypto_returns"],
        prices=context["crypto_prices"],
        asset_table=context["crypto_assets"],
        benchmark_ticker=context["benchmark_crypto"],
        fallback_ticker=context["benchmark_crypto"],
        score_mode=str(score_mode),
        lookback_days=21,
        rebalance_days=7,
        top_k=int(top_k),
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
    if crypto_result is None:
        raise SystemExit(f"falha ao simular sleeve cripto: {candidate_id}")
    benchmark_gross = pd.to_numeric(
        context["crypto_returns"][str(context["benchmark_crypto"])], errors="coerce"
    ).reindex(crypto_result.gross_ret.index).fillna(0.0).astype(float)
    return StrategyBundle(
        result=crypto_result,
        benchmark_gross_ret=benchmark_gross,
        profile=context["profiles"]["crypto"],
        benchmark_profile=context["profiles"]["crypto"],
    )


def _blend_crypto_bundles(
    *,
    candidate_id: str,
    primary: StrategyBundle,
    secondary: StrategyBundle,
    primary_weight: float,
) -> StrategyBundle:
    idx = primary.result.gross_ret.index.intersection(secondary.result.gross_ret.index)
    weight = float(np.clip(primary_weight, 0.0, 1.0))
    gross = (
        weight * pd.to_numeric(primary.result.gross_ret.reindex(idx), errors="coerce").fillna(0.0).astype(float)
        + (1.0 - weight) * pd.to_numeric(secondary.result.gross_ret.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    )
    turnover = (
        weight * pd.to_numeric(primary.result.turnover.reindex(idx), errors="coerce").fillna(0.0).astype(float)
        + (1.0 - weight) * pd.to_numeric(secondary.result.turnover.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    )
    benchmark = primary.benchmark_gross_ret.reindex(idx).fillna(0.0).astype(float)
    perf = _evaluate_net(
        gross_ret=gross,
        turnover=turnover,
        profile=primary.profile,
        benchmark_ret=benchmark,
        benchmark_profile=primary.benchmark_profile,
    )
    hit5 = _rolling_ten_x_stats(perf["net_ret"], horizon_days=1260)
    wealth = (1.0 + perf["net_ret"]).cumprod()
    hit_full = wealth[wealth >= 10.0]
    years_to_10x = float((hit_full.index[0] - wealth.index[0]).days / 365.25) if not hit_full.empty else float("nan")
    result = StrategyResult(
        suite="crypto_resolution",
        candidate_id=f"{candidate_id}__crypto",
        family="crypto_resolution_blend",
        benchmark_ticker=primary.result.benchmark_ticker,
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
        notes=f"blend_primary_weight={weight:.2f};base={primary.result.candidate_id};secondary={secondary.result.candidate_id}",
    )
    return StrategyBundle(
        result=result,
        benchmark_gross_ret=benchmark,
        profile=primary.profile,
        benchmark_profile=primary.benchmark_profile,
    )


def _meta_from_crypto_bundle(
    *,
    candidate_id: str,
    crypto_bundle: StrategyBundle,
    context: dict[str, Any],
) -> StrategyResult:
    allocation = _build_alpha_meta_allocation_bundle(
        candidate_id=str(candidate_id),
        crypto_bundle=crypto_bundle,
        equity_bundle=context["equity_attack"],
        btc_prices=context["btc_prices"],
        spy_prices=context["spy_prices"],
        profile=context["profiles"]["blended"],
        entry_lookback=21,
        exit_lookback=63,
        entry_margin=0.05,
        exit_margin=0.05,
        risk_off_mode="equity25",
        min_crypto_hold_days=0,
    )
    return allocation.bundle.result


def _variant_results(*, context: dict[str, Any], drop_crypto: list[str] | None = None) -> dict[str, StrategyResult]:
    tiers = context["crypto_tiers"]
    blocked = set(drop_crypto or [])
    major8 = [ticker for ticker in tiers["crypto_major8"] if ticker not in blocked]
    all22 = [ticker for ticker in tiers["crypto_all"] if ticker not in blocked]

    attack_top1 = _crypto_rule_bundle(
        candidate_id="attack_major8_k1",
        allowed_tickers=major8,
        score_mode="mom_total",
        top_k=1,
        context=context,
    )
    diversified_major8 = _crypto_rule_bundle(
        candidate_id="div_major8_k3",
        allowed_tickers=major8,
        score_mode="mom_vol_adj",
        top_k=3,
        context=context,
    )
    diversified_all22 = _crypto_rule_bundle(
        candidate_id="div_all22_k3",
        allowed_tickers=all22,
        score_mode="mom_vol_adj",
        top_k=3,
        context=context,
    )

    breadth_signal = _build_breadth_signal(
        returns=context["crypto_returns"],
        prices=context["crypto_prices"],
        tickers=all22,
        lookback_days=21,
        ma_days=200,
    )

    blend_70 = _blend_crypto_bundles(
        candidate_id="blend70_major8_base30",
        primary=attack_top1,
        secondary=diversified_major8,
        primary_weight=0.70,
    )
    blend_60 = _blend_crypto_bundles(
        candidate_id="blend60_major8_base40",
        primary=attack_top1,
        secondary=diversified_major8,
        primary_weight=0.60,
    )
    blend_70_breadth = _apply_breadth_overlay_to_bundle(
        candidate_id="blend70_major8_base30_breadth__crypto",
        bundle=blend_70,
        breadth_signal=breadth_signal,
        low_threshold=0.38,
        high_threshold=0.62,
        mode="scale",
    )
    all22_breadth = _apply_breadth_overlay_to_bundle(
        candidate_id="div_all22_k3_breadth__crypto",
        bundle=diversified_all22,
        breadth_signal=breadth_signal,
        low_threshold=0.38,
        high_threshold=0.62,
        mode="scale",
    )

    results = {
        "alpha_attack_major8_equity25": _meta_from_crypto_bundle(
            candidate_id="alpha_attack_major8_equity25",
            crypto_bundle=attack_top1,
            context=context,
        ),
        "attack_div_major8_k3": _meta_from_crypto_bundle(candidate_id="attack_div_major8_k3", crypto_bundle=diversified_major8, context=context),
        "attack_div_all22_k3": _meta_from_crypto_bundle(candidate_id="attack_div_all22_k3", crypto_bundle=diversified_all22, context=context),
        "attack_blend70_major8": _meta_from_crypto_bundle(candidate_id="attack_blend70_major8", crypto_bundle=blend_70, context=context),
        "attack_blend60_major8": _meta_from_crypto_bundle(candidate_id="attack_blend60_major8", crypto_bundle=blend_60, context=context),
        "attack_blend70_major8_breadth": _meta_from_crypto_bundle(candidate_id="attack_blend70_major8_breadth", crypto_bundle=blend_70_breadth, context=context),
        "attack_div_all22_breadth": _meta_from_crypto_bundle(candidate_id="attack_div_all22_breadth", crypto_bundle=all22_breadth, context=context),
    }
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Resolve a fragilidade do cripto misturando ataque, diversificação e breadth.")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--outdir-root", default="results/validation/profit_crypto_resolution_suite")
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

    top_freq = _selection_frequency_for_crypto_rule(
        allowed_tickers=context["crypto_tiers"]["crypto_major8"],
        returns=context["crypto_returns"],
        prices=context["crypto_prices"],
        benchmark_ticker=str(args.benchmark_crypto),
        lookback_days=21,
        rebalance_days=7,
        top_k=1,
        score_mode="mom_total",
        asset_ma_days=0,
        market_ma_days=200,
    )
    top3 = top_freq["ticker"].head(3).astype(str).tolist()

    base_results = _variant_results(context=context)
    dropped_results = _variant_results(context=context, drop_crypto=top3)

    base_df = pd.DataFrame([_result_row(result) for result in base_results.values()]).sort_values(
        ["net_total_return", "net_sharpe"], ascending=[False, False]
    ).reset_index(drop=True)
    base_df.to_csv(outdir / "candidate_compare.csv", index=False)

    retention_rows: list[dict[str, Any]] = []
    for candidate_id, base_result in base_results.items():
        dropped = dropped_results.get(candidate_id)
        if dropped is None:
            continue
        base_total = _safe_float(base_result.net_total_return)
        base_ann = _safe_float(base_result.net_ann_return)
        drop_total = _safe_float(dropped.net_total_return)
        drop_ann = _safe_float(dropped.net_ann_return)
        total_retention = drop_total / base_total if np.isfinite(base_total) and abs(base_total) > 1e-9 else float("nan")
        ann_retention = drop_ann / base_ann if np.isfinite(base_ann) and abs(base_ann) > 1e-9 else float("nan")
        robustness_score = 0.55 * max(0.0, total_retention) + 0.25 * max(0.0, ann_retention) + 0.20 * max(0.0, 1.0 + _safe_float(dropped.net_max_drawdown))
        retention_rows.append(
            {
                "candidate_id": str(candidate_id),
                "candidate_label": _human_label(str(candidate_id)),
                "drop_crypto": ",".join(top3),
                "base_total_return": base_total,
                "drop_total_return": drop_total,
                "base_ann_return": base_ann,
                "drop_ann_return": drop_ann,
                "base_drawdown": _safe_float(base_result.net_max_drawdown),
                "drop_drawdown": _safe_float(dropped.net_max_drawdown),
                "total_retention": total_retention,
                "ann_retention": ann_retention,
                "robustness_score": robustness_score,
            }
        )
    retention_df = pd.DataFrame(retention_rows).sort_values(
        ["robustness_score", "drop_total_return"], ascending=[False, False]
    ).reset_index(drop=True)
    retention_df.to_csv(outdir / "dependency_retention.csv", index=False)

    baseline_row = base_df[base_df["candidate_id"] == "alpha_attack_major8_equity25"].head(1)
    baseline_ret = retention_df[retention_df["candidate_id"] == "alpha_attack_major8_equity25"].head(1)
    best_profit = base_df.head(1).to_dict(orient="records")
    best_retention = retention_df.head(1).to_dict(orient="records")
    balanced = pd.DataFrame()
    if not base_df.empty and not retention_df.empty:
        merged = base_df.merge(retention_df[["candidate_id", "total_retention", "ann_retention", "robustness_score"]], on="candidate_id", how="left")
        merged["balance_score"] = (
            0.40 * pd.to_numeric(merged["net_total_return"], errors="coerce").rank(pct=True)
            + 0.35 * pd.to_numeric(merged["total_retention"], errors="coerce").rank(pct=True)
            + 0.15 * pd.to_numeric(merged["ann_retention"], errors="coerce").rank(pct=True)
            + 0.10 * (1.0 + pd.to_numeric(merged["net_max_drawdown"], errors="coerce"))
        )
        balanced = merged.sort_values(["balance_score", "net_total_return"], ascending=[False, False]).reset_index(drop=True)
        balanced.to_csv(outdir / "balance_compare.csv", index=False)

    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "top_crypto_frequency": top_freq.head(10).to_dict(orient="records"),
        "top3_removed_for_dependency_test": top3,
        "baseline_attack": baseline_row.iloc[0].to_dict() if not baseline_row.empty else {},
        "baseline_attack_retention": baseline_ret.iloc[0].to_dict() if not baseline_ret.empty else {},
        "best_profit_variant": best_profit[0] if best_profit else {},
        "best_retention_variant": best_retention[0] if best_retention else {},
        "best_balance_variant": balanced.iloc[0].to_dict() if not balanced.empty else {},
        "insights": [
            "A suite tenta manter a perna de acoes igual e mexe so na forma de entrar no cripto.",
            "O objetivo e ver se misturar um cripto agressivo com um cripto mais espalhado melhora a sobrevivencia quando os nomes mais fortes saem da jogada.",
            "O filtro de breadth tenta reduzir dependencia de poucos nomes sem matar completamente a fase boa do sleeve cripto.",
        ],
        "artifacts": {
            "candidate_compare_csv": str(outdir / "candidate_compare.csv"),
            "dependency_retention_csv": str(outdir / "dependency_retention.csv"),
            "balance_compare_csv": str(outdir / "balance_compare.csv"),
        },
    }
    _write_json(outdir / "summary.json", summary)

    research_rows = []
    for result in base_results.values():
        status = "watch"
        if str(result.candidate_id) == str((summary.get("best_balance_variant") or {}).get("candidate_id")):
            status = "keep"
        research_rows.append(
            _research_row(
                result,
                outdir=outdir,
                status=status,
                methodology="crypto_resolution",
                label=_human_label(str(result.candidate_id)),
            )
        )
    (outdir / "profit_research_rows.json").write_text(json.dumps(research_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    write_run_manifest(
        outdir / "RUN_MANIFEST.json",
        script=str(Path(__file__).resolve()),
        params={
            "benchmark_crypto": str(args.benchmark_crypto),
            "benchmark_equity": str(args.benchmark_equity),
            "drop_top3": top3,
        },
        paths={
            "summary_json": str(outdir / "summary.json"),
            "candidate_compare_csv": str(outdir / "candidate_compare.csv"),
            "dependency_retention_csv": str(outdir / "dependency_retention.csv"),
        },
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
