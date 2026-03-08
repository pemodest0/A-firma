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
from execution.net_assumptions import NetAssumptionProfile, load_net_assumption_profiles  # noqa: E402
from scripts.bench.validation.run_profit_frontier_expansion_suite import (  # noqa: E402
    EQUITY_EXCLUDED,
    StrategyResult,
    _ensure_benchmark_columns,
    _evaluate_net,
    _load_asset_table,
    _load_daily_universe,
    _result_row,
    _run_id,
    _safe_float,
    _write_json,
)
from scripts.bench.validation.run_profit_layered_engine_suite import (  # noqa: E402
    StrategyBundle,
    _profile_scaled,
    _research_rows,
    _simulate_equity_group_sleeve_v2,
    _simulate_equity_group_sleeve_v3,
    _stress_bundle,
    _walkforward_rows,
)
from scripts.bench.validation.run_profit_drawdown_control_suite import (  # noqa: E402
    _candidate_diag,
    _load_structural_regime_series,
    _regime_forward_fill,
)


def _load_equity_universe(
    *,
    prices_dir: Path,
    asset_groups: Path,
    asset_metadata: Path,
    benchmark_ticker: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    asset_table = _load_asset_table(asset_groups, asset_metadata)
    asset_table = asset_table[~asset_table["asset_group"].astype(str).isin(EQUITY_EXCLUDED)].copy()
    returns, prices, _ = _load_daily_universe(
        prices_dir=prices_dir,
        asset_table=asset_table,
        min_history_days=1200,
        max_abs_daily_return=0.8,
    )
    returns, prices = _ensure_benchmark_columns(returns, prices, prices_dir, [benchmark_ticker])
    group_map: dict[str, list[str]] = {}
    for group, sub in asset_table.groupby("asset_group", sort=True):
        tickers = [ticker for ticker in sub["ticker"].astype(str).tolist() if ticker in returns.columns]
        if len(tickers) >= 6:
            group_map[str(group)] = tickers
    return asset_table, returns, prices, group_map


def _regime_scale_bundle(
    *,
    candidate_id: str,
    bundle: StrategyBundle,
    regime_series: pd.Series,
    mapping: dict[str, float],
    notes: str,
) -> StrategyBundle:
    idx = bundle.result.gross_ret.index
    scale = _regime_forward_fill(idx, regime_series).map({str(k).lower(): float(v) for k, v in mapping.items()}).fillna(1.0).astype(float)
    gross = pd.to_numeric(bundle.result.gross_ret.reindex(idx), errors="coerce").fillna(0.0).astype(float) * scale
    turnover = pd.to_numeric(bundle.result.turnover.reindex(idx), errors="coerce").fillna(0.0).astype(float) * scale
    turnover = turnover + scale.diff().abs().fillna(scale.abs()) * 0.35
    perf = _evaluate_net(
        gross_ret=gross,
        turnover=turnover,
        profile=bundle.profile,
        benchmark_ret=bundle.benchmark_gross_ret.reindex(idx).fillna(0.0).astype(float),
        benchmark_profile=bundle.benchmark_profile,
    )
    result = StrategyResult(
        suite="equities_regime_scale",
        candidate_id=candidate_id,
        family="equities_regime_scale",
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
        hit_rate_10x_5y=float("nan"),
        years_to_10x_full=float("nan"),
        notes=notes,
    )
    return StrategyBundle(result=result, benchmark_gross_ret=bundle.benchmark_gross_ret, profile=bundle.profile, benchmark_profile=bundle.benchmark_profile)


def _equity_trailing_switch_bundle(
    *,
    candidate_id: str,
    aggressive_bundle: StrategyBundle,
    robust_bundle: StrategyBundle,
    regime_series: pd.Series,
    spy_prices: pd.Series,
    mode: str,
) -> StrategyBundle:
    idx = aggressive_bundle.result.gross_ret.index.intersection(robust_bundle.result.gross_ret.index).intersection(spy_prices.index)
    agg_ret = pd.to_numeric(aggressive_bundle.result.gross_ret.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    rob_ret = pd.to_numeric(robust_bundle.result.gross_ret.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    benchmark = aggressive_bundle.benchmark_gross_ret.reindex(idx).fillna(0.0).astype(float)
    spy = pd.to_numeric(spy_prices.reindex(idx), errors="coerce").astype(float)
    market_ok = (spy.shift(1) > spy.shift(1).rolling(200, min_periods=100).mean()).fillna(False)
    agg_trail = (1.0 + agg_ret).rolling(63, min_periods=21).apply(np.prod, raw=True) - 1.0
    rob_trail = (1.0 + rob_ret).rolling(63, min_periods=21).apply(np.prod, raw=True) - 1.0
    reg = _regime_forward_fill(idx, regime_series)

    gross = pd.Series(np.zeros(len(idx), dtype=float), index=idx, dtype=float)
    turnover = pd.Series(np.zeros(len(idx), dtype=float), index=idx, dtype=float)
    prev_weights: dict[str, float] = {"cash": 1.0}
    for dt in idx:
        regime = str(reg.loc[dt]).lower()
        agg = _safe_float(agg_trail.loc[dt], 0.0)
        rob = _safe_float(rob_trail.loc[dt], 0.0)
        m_ok = bool(market_ok.loc[dt])
        weights: dict[str, float]
        if mode == "regime_switch":
            if regime == "stress":
                weights = {"cash": 1.0}
            elif regime == "transition":
                weights = {"robust": 1.0} if m_ok else {"cash": 1.0}
            else:
                weights = {"aggressive": 1.0} if m_ok else {"robust": 1.0}
        elif mode == "regime_blend":
            if regime == "stress":
                weights = {"robust": 0.35, "cash": 0.65} if m_ok else {"cash": 1.0}
            elif regime == "transition":
                weights = {"aggressive": 0.35, "robust": 0.65} if m_ok else {"robust": 0.60, "cash": 0.40}
            elif regime == "dispersion":
                weights = {"aggressive": 0.85, "robust": 0.15} if m_ok else {"robust": 0.70, "cash": 0.30}
            else:
                weights = {"aggressive": 0.75, "robust": 0.25} if m_ok else {"robust": 0.70, "cash": 0.30}
        else:
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
        hit_rate_10x_5y=float("nan"),
        years_to_10x_full=float("nan"),
        notes=f"mode={mode};agg={aggressive_bundle.result.candidate_id};rob={robust_bundle.result.candidate_id}",
    )
    return StrategyBundle(result=result, benchmark_gross_ret=benchmark, profile=aggressive_bundle.profile, benchmark_profile=aggressive_bundle.benchmark_profile)


def _stress_and_wf(
    bundles: list[StrategyBundle],
    *,
    hard_profile: NetAssumptionProfile,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stress_rows: list[dict[str, Any]] = []
    wf_rows: list[dict[str, Any]] = []
    wf_blocks = [
        ("test_2022", "2022-01-01", "2022-12-31"),
        ("test_2023_2024", "2023-01-01", "2024-12-31"),
        ("test_2025_now", "2025-01-01", str(pd.Timestamp.now("UTC").date())),
    ]
    for bundle in bundles:
        stress_rows.append(_stress_bundle(bundle, delay_days=0, profile=bundle.profile, benchmark_profile=bundle.benchmark_profile, label="base"))
        stress_rows.append(_stress_bundle(bundle, delay_days=1, profile=bundle.profile, benchmark_profile=bundle.benchmark_profile, label="delay_d1"))
        stress_rows.append(_stress_bundle(bundle, delay_days=0, profile=hard_profile, benchmark_profile=hard_profile, label="hard_cost"))
        wf_rows.extend(_walkforward_rows(bundle, wf_blocks))
    return pd.DataFrame(stress_rows), pd.DataFrame(wf_rows)


def _equity_score_row(
    *,
    bundle: StrategyBundle,
    baseline: StrategyBundle,
    stress_df: pd.DataFrame,
    wf_df: pd.DataFrame,
) -> dict[str, Any]:
    diag = _candidate_diag(bundle)
    base_diag = _candidate_diag(baseline)
    stress_sub = stress_df[stress_df["candidate_id"].astype(str) == str(bundle.result.candidate_id)]
    wf_sub = wf_df[wf_df["candidate_id"].astype(str) == str(bundle.result.candidate_id)]
    hard_cost = stress_sub[stress_sub["stress_label"].astype(str) == "hard_cost"]
    base_ann = max(float(baseline.result.net_ann_return), 1e-9)
    base_mdd = max(abs(float(baseline.result.net_max_drawdown)), 1e-9)
    ann_retention = float(bundle.result.net_ann_return / base_ann)
    dd_closure = float((base_mdd - abs(float(bundle.result.net_max_drawdown))) / base_mdd)
    positive_test_share = float((pd.to_numeric(wf_sub.get("edge_vs_benchmark_net_total_return"), errors="coerce") > 0.0).mean()) if not wf_sub.empty else 0.0
    mean_test_edge = float(pd.to_numeric(wf_sub.get("edge_vs_benchmark_net_total_return"), errors="coerce").dropna().mean()) if not wf_sub.empty else float("nan")
    hard_cost_retention = max(0.0, _safe_float(hard_cost["net_ann_return"].iloc[0], 0.0) / base_ann) if not hard_cost.empty else 0.0
    balanced_score = (
        0.35 * ann_retention
        + 0.25 * max(dd_closure, -0.5)
        + 0.20 * float(bundle.result.net_sharpe)
        + 0.10 * positive_test_share
        + 0.10 * hard_cost_retention
    )
    worth_it = bool((dd_closure >= 0.15 and ann_retention >= 0.85) or (bundle.result.net_sharpe >= baseline.result.net_sharpe + 0.10 and dd_closure >= 0.10))
    return {
        **_result_row(bundle.result),
        **diag,
        "ann_retention_vs_baseline": ann_retention,
        "drawdown_closure_vs_baseline": dd_closure,
        "ulcer_improvement_vs_baseline": float((_safe_float(base_diag.get("ulcer_index"), 0.0) - _safe_float(diag.get("ulcer_index"), 0.0)) / max(_safe_float(base_diag.get("ulcer_index"), 1.0), 1e-9)),
        "mean_test_edge": mean_test_edge,
        "positive_test_share": positive_test_share,
        "hard_cost_ann_return": _safe_float(hard_cost["net_ann_return"].iloc[0], float("nan")) if not hard_cost.empty else float("nan"),
        "balanced_score": balanced_score,
        "worth_it": worth_it,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Melhora dirigida do sleeve de equities.")
    ap.add_argument("--asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--benchmark", default="SPY")
    ap.add_argument("--net-assumptions", default="config/profit_net_assumptions.json")
    ap.add_argument("--outdir-root", default="results/validation/profit_equity_improvement_suite")
    args = ap.parse_args()

    outdir = (ROOT / args.outdir_root / _run_id()).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    prices_dir = (ROOT / args.prices_dir).resolve()

    profiles = load_net_assumption_profiles((ROOT / args.net_assumptions).resolve())
    foreign_profile = profiles["profiles"]["foreign_financial_brazil_resident"]
    hard_profile = _profile_scaled(
        foreign_profile,
        profile_id="foreign_hard",
        label="Foreign hard frictions",
        transaction_cost_bps=20.0,
        fx_spread_bps=45.0,
        capital_gains_tax_rate=0.15,
        tax_timing="annual_positive_proxy",
    )

    asset_table, returns, prices, group_map = _load_equity_universe(
        prices_dir=prices_dir,
        asset_groups=(ROOT / args.asset_groups).resolve(),
        asset_metadata=(ROOT / args.asset_metadata).resolve(),
        benchmark_ticker=str(args.benchmark),
    )
    regime_series = _load_structural_regime_series(ROOT)
    spy_prices = pd.to_numeric(prices[str(args.benchmark)], errors="coerce")

    pure_candidates: list[StrategyBundle] = []
    v2_specs = [
        ("equity_v2__slow189__g3__a2", 63, 189, 3, 2, 126, 200, 200),
        ("equity_v2__slow189__g3__a1", 63, 189, 3, 1, 126, 200, 200),
        ("equity_v2__slow252__g3__a1", 63, 252, 3, 1, 126, 200, 200),
        ("equity_v2__slow252__g3__a2_m150", 63, 252, 3, 2, 126, 150, 150),
        ("equity_v2__slow189__g4__a1", 63, 189, 4, 1, 126, 200, 200),
    ]
    for cid, gf, gs, gk, apg, alb, ama, mma in v2_specs:
        bundle = _simulate_equity_group_sleeve_v2(
            candidate_id=cid,
            returns=returns,
            prices=prices,
            asset_table=asset_table,
            equity_groups=group_map,
            benchmark_ticker=str(args.benchmark),
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
            pure_candidates.append(bundle)
    v3_specs = [
        ("equity_v3__slow189__g3__a2__br35__cap40", 63, 189, 3, 2, 126, 200, 200, 0.35, 0.40),
        ("equity_v3__slow189__g3__a2__br30__cap45", 63, 189, 3, 2, 126, 200, 200, 0.30, 0.45),
        ("equity_v3__slow252__g3__a2__br30__cap45", 63, 252, 3, 2, 126, 200, 200, 0.30, 0.45),
        ("equity_v3__slow252__g2__a2__br35__cap40", 63, 252, 2, 2, 126, 200, 200, 0.35, 0.40),
    ]
    for cid, gf, gs, gk, apg, alb, ama, mma, br, cap in v3_specs:
        bundle = _simulate_equity_group_sleeve_v3(
            candidate_id=cid,
            returns=returns,
            prices=prices,
            asset_table=asset_table,
            equity_groups=group_map,
            benchmark_ticker=str(args.benchmark),
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
            pure_candidates.append(bundle)
    if not pure_candidates:
        raise SystemExit("no equity candidates")

    pure_df = pd.DataFrame([_result_row(b.result) for b in pure_candidates]).sort_values(["net_ann_return", "net_sharpe"], ascending=[False, False]).reset_index(drop=True)
    baseline_id = "equity_v2__slow189__g3__a2"
    pure_map = {b.result.candidate_id: b for b in pure_candidates}
    baseline = pure_map.get(baseline_id, pure_map[str(pure_df.iloc[0]["candidate_id"])])

    robust_df = pure_df.copy()
    robust_df["robust_score"] = 0.45 * pd.to_numeric(robust_df["net_ann_return"], errors="coerce").fillna(0.0) + 0.35 * pd.to_numeric(robust_df["net_sharpe"], errors="coerce").fillna(0.0) + 0.20 * (1.0 + pd.to_numeric(robust_df["net_max_drawdown"], errors="coerce").fillna(-1.0))
    best_ann_bundle = pure_map[str(pure_df.iloc[0]["candidate_id"])]
    best_robust_bundle = pure_map[str(robust_df.sort_values(["robust_score", "net_sharpe"], ascending=[False, False]).iloc[0]["candidate_id"])]

    combined_candidates = [
        _regime_scale_bundle(
            candidate_id="equity_regime_scale__stress00_transition65",
            bundle=best_ann_bundle,
            regime_series=regime_series,
            mapping={"stress": 0.0, "transition": 0.65, "stable": 1.0, "dispersion": 1.0},
            notes="escala por regime estrutural; agressivo quando o pano de fundo ajuda",
        ),
        _regime_scale_bundle(
            candidate_id="equity_regime_scale__stress20_transition80",
            bundle=best_ann_bundle,
            regime_series=regime_series,
            mapping={"stress": 0.20, "transition": 0.80, "stable": 1.0, "dispersion": 1.0},
            notes="escala leve por regime estrutural",
        ),
        _equity_trailing_switch_bundle(
            candidate_id="equity_meta__regime_switch",
            aggressive_bundle=best_ann_bundle,
            robust_bundle=best_robust_bundle,
            regime_series=regime_series,
            spy_prices=spy_prices,
            mode="regime_switch",
        ),
        _equity_trailing_switch_bundle(
            candidate_id="equity_meta__regime_blend",
            aggressive_bundle=best_ann_bundle,
            robust_bundle=best_robust_bundle,
            regime_series=regime_series,
            spy_prices=spy_prices,
            mode="regime_blend",
        ),
        _equity_trailing_switch_bundle(
            candidate_id="equity_meta__trail_switch",
            aggressive_bundle=best_ann_bundle,
            robust_bundle=best_robust_bundle,
            regime_series=regime_series,
            spy_prices=spy_prices,
            mode="trail_switch",
        ),
    ]
    ann_pool = [pure_map[str(cid)] for cid in pure_df.head(4)["candidate_id"].astype(str).tolist()]
    robust_pool = [pure_map[str(cid)] for cid in robust_df.sort_values(["robust_score", "net_sharpe"], ascending=[False, False]).head(4)["candidate_id"].astype(str).tolist()]
    for agg_rank, agg_bundle in enumerate(ann_pool, start=1):
        for rob_rank, rob_bundle in enumerate(robust_pool, start=1):
            if agg_bundle.result.candidate_id == rob_bundle.result.candidate_id:
                continue
            for mode in ("trail_switch", "regime_blend"):
                combined_candidates.append(
                    _equity_trailing_switch_bundle(
                        candidate_id=f"equity_meta_search__{mode}__a{agg_rank}__r{rob_rank}",
                        aggressive_bundle=agg_bundle,
                        robust_bundle=rob_bundle,
                        regime_series=regime_series,
                        spy_prices=spy_prices,
                        mode=mode,
                    )
                )
    all_bundles = pure_candidates + combined_candidates
    stress_df, wf_df = _stress_and_wf(all_bundles, hard_profile=hard_profile)
    compare_rows = [_equity_score_row(bundle=b, baseline=baseline, stress_df=stress_df, wf_df=wf_df) for b in all_bundles]
    compare_df = pd.DataFrame(compare_rows).sort_values(["worth_it", "balanced_score", "net_ann_return"], ascending=[False, False, False]).reset_index(drop=True)

    pure_df.to_csv(outdir / "pure_candidate_compare.csv", index=False)
    compare_df.to_csv(outdir / "candidate_compare.csv", index=False)
    stress_df.to_csv(outdir / "stress_compare.csv", index=False)
    wf_df.to_csv(outdir / "walkforward_blocks.csv", index=False)

    worthwhile = compare_df[compare_df["worth_it"].fillna(False)].copy()
    best_overall = compare_df.iloc[0].to_dict()
    best_bundle = next(b for b in all_bundles if b.result.candidate_id == str(best_overall["candidate_id"]))

    status_map = {b.result.candidate_id: "kill" for b in all_bundles}
    status_map[baseline.result.candidate_id] = "keep"
    for cid in worthwhile["candidate_id"].astype(str).tolist():
        if cid != baseline.result.candidate_id:
            status_map[cid] = "watch"

    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "outdir": str(outdir),
        "baseline_candidate": _equity_score_row(bundle=baseline, baseline=baseline, stress_df=stress_df, wf_df=wf_df),
        "best_pure_ann_candidate": _result_row(best_ann_bundle.result),
        "best_pure_robust_candidate": _result_row(best_robust_bundle.result),
        "best_overall_candidate": best_overall,
        "worth_it_candidates": worthwhile.to_dict(orient="records"),
        "insights": [
            f"Baseline equities atual: {baseline.result.candidate_id} com ann={baseline.result.net_ann_return:.4f}, sharpe={baseline.result.net_sharpe:.4f}, mdd={baseline.result.net_max_drawdown:.4f}.",
            f"Melhor puro em retorno: {best_ann_bundle.result.candidate_id}.",
            f"Melhor puro em robustez: {best_robust_bundle.result.candidate_id}.",
            ("Nenhuma combinacao bateu o baseline no custo-beneficio." if str(best_overall["candidate_id"]) == str(baseline.result.candidate_id) else f"A melhor melhora do sleeve de equities foi {best_overall['candidate_id']}."),
        ],
        "artifacts": {
            "pure_candidate_compare_csv": str(outdir / "pure_candidate_compare.csv"),
            "candidate_compare_csv": str(outdir / "candidate_compare.csv"),
            "stress_compare_csv": str(outdir / "stress_compare.csv"),
            "walkforward_blocks_csv": str(outdir / "walkforward_blocks.csv"),
        },
    }
    summary_path = outdir / "summary.json"
    _write_json(summary_path, summary)

    rows = _research_rows([b.result for b in all_bundles], outdir=outdir, summary_path=summary_path, status_map=status_map)
    (outdir / "profit_research_rows.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    write_run_manifest(
        outdir,
        script="scripts/bench/validation/run_profit_equity_improvement_suite.py",
        params={"benchmark": str(args.benchmark)},
        paths={
            "summary_json": str(summary_path),
            "candidate_compare_csv": str(outdir / "candidate_compare.csv"),
            "stress_compare_csv": str(outdir / "stress_compare.csv"),
            "walkforward_blocks_csv": str(outdir / "walkforward_blocks.csv"),
            "profit_research_rows_json": str(outdir / "profit_research_rows.json"),
        },
        extra={"notes": ["Suite focada em melhorar o sleeve de equities do stack."]},
    )
    print(json.dumps({"status": "ok", "outdir": str(outdir), "winner": str(best_overall["candidate_id"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
