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

from engine.portfolio import MetaModeSelectorConfig, monthly_last, monthly_total_return, run_causal_meta_mode_selector  # noqa: E402
from engine.structural.run_manifest import write_run_manifest  # noqa: E402
from scripts.bench.validation.run_profit_alpha_hardening_suite import (  # noqa: E402
    AllocationBundle,
    _blend_allocation_bundles,
    _build_candidates,
    _confidence_weight_from_score,
)
from scripts.bench.validation.run_profit_alpha_improvement_suite import _safe_float, _write_json  # noqa: E402
from scripts.bench.validation.run_profit_frontier_expansion_suite import StrategyResult, _evaluate_net  # noqa: E402
from scripts.bench.validation.run_profit_investment_yearbook import _calendar_rows, _result_row  # noqa: E402
from scripts.bench.validation.run_profit_layered_engine_suite import StrategyBundle  # noqa: E402
from scripts.bench.validation.run_profit_marketmode_criticality_suite import _build_structure_layers  # noqa: E402
from scripts.bench.validation.run_profit_sector_pressure_suite import _research_row  # noqa: E402


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    if values.empty:
        return values
    min_periods = max(6, int(window) // 3)

    def _pct(arr: np.ndarray) -> float:
        arr = arr[np.isfinite(arr)]
        if arr.size <= 1:
            return float("nan")
        return float(np.mean(arr <= float(arr[-1])))

    return values.rolling(int(window), min_periods=min_periods).apply(_pct, raw=True)


def _build_criticality_free_energy_bundle(
    *,
    built: dict[str, Any],
) -> tuple[AllocationBundle, pd.DataFrame]:
    context = dict(built["context"])
    attack_alloc: AllocationBundle = built["allocations"]["attack"]
    protect_alloc: AllocationBundle = built["allocations"]["baseline_guard"]

    base_score = pd.to_numeric(context["attack_score_exogenous"], errors="coerce").fillna(0.0).clip(0.0, 1.0).astype(float)
    structure_daily, _spectral_panel, criticality, _structural_stress = _build_structure_layers(context)

    criticality_pct = _rolling_percentile(
        pd.to_numeric(criticality, errors="coerce").reindex(base_score.index).fillna(0.5),
        126,
    ).fillna(0.5)
    market_pct = (
        pd.to_numeric(structure_daily.get("market_mode_share_pct"), errors="coerce")
        .reindex(base_score.index)
        .fillna(0.5)
    )
    rel_penalty = (
        0.22 * ((criticality_pct - 0.55).clip(lower=0.0) / 0.45)
        + 0.06 * ((market_pct - 0.70).clip(lower=0.0) / 0.30)
    ).clip(0.0, 0.35)
    criticality_rel_score = (base_score - rel_penalty).clip(0.0, 1.0)

    instability = (
        0.60 * pd.to_numeric(criticality, errors="coerce").reindex(base_score.index).fillna(0.5)
        + 0.40 * pd.to_numeric(structure_daily.get("market_mode_share_pct"), errors="coerce").reindex(base_score.index).fillna(0.5)
    ).clip(0.0, 1.0)
    turnover = pd.to_numeric(attack_alloc.bundle.result.turnover, errors="coerce").reindex(base_score.index).fillna(0.0).astype(float)
    free_rel_score = (
        criticality_rel_score
        - 0.06 * turnover.clip(lower=0.0, upper=1.0)
        - 0.08 * instability.clip(lower=0.0, upper=1.0)
    ).clip(0.0, 1.0)

    free_rel_weight = _confidence_weight_from_score(free_rel_score)
    bundle = _blend_allocation_bundles(
        candidate_id="criticality_free_energy_attack",
        notes="combina criticidade relativa com penalidade leve de reorganizacao",
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        attack_weight=free_rel_weight,
    )
    aux = pd.DataFrame(
        {
            "criticality": pd.to_numeric(criticality, errors="coerce").reindex(base_score.index),
            "market_mode_share_pct": pd.to_numeric(structure_daily.get("market_mode_share_pct"), errors="coerce").reindex(base_score.index),
            "criticality_score": pd.to_numeric(criticality_rel_score, errors="coerce").reindex(base_score.index),
            "free_rel_score": pd.to_numeric(free_rel_score, errors="coerce").reindex(base_score.index),
        },
        index=base_score.index,
    )
    return bundle, aux


def _candidate_library(built: dict[str, Any]) -> tuple[dict[str, AllocationBundle], pd.DataFrame]:
    criticality_bundle, criticality_aux = _build_criticality_free_energy_bundle(built=built)
    bundles: dict[str, AllocationBundle] = {
        "principal": built["allocations"]["baseline"],
        "ataque": built["allocations"]["attack"],
        "protegido": built["allocations"]["baseline_guard"],
        "ataque_guard": built["allocations"]["attack_guard"],
        "criticidade": criticality_bundle,
    }
    return bundles, criticality_aux


def _build_feature_frame(*, built: dict[str, Any], criticality_aux: pd.DataFrame, candidate_returns: dict[str, pd.Series]) -> pd.DataFrame:
    context = dict(built["context"])
    idx = context["attack_score_exogenous"].index
    exogenous = context["exogenous_panel"].reindex(idx).copy()
    attack_score = pd.to_numeric(context["attack_score_exogenous"], errors="coerce").reindex(idx).fillna(0.0).astype(float)
    feature_daily = pd.DataFrame(index=idx)
    feature_daily["attack_score"] = attack_score
    for col in ["liquidation", "breadth", "crypto_dependency_risk", "macro_stress"]:
        feature_daily[col] = pd.to_numeric(exogenous.get(col), errors="coerce").reindex(idx)
    for col in criticality_aux.columns:
        feature_daily[col] = pd.to_numeric(criticality_aux[col], errors="coerce").reindex(idx)
    for name, series in candidate_returns.items():
        s = pd.to_numeric(series, errors="coerce").reindex(idx).fillna(0.0).astype(float)
        feature_daily[f"{name}_tail21"] = (1.0 + s).rolling(21, min_periods=7).apply(np.prod, raw=True) - 1.0
        feature_daily[f"{name}_tail63"] = (1.0 + s).rolling(63, min_periods=21).apply(np.prod, raw=True) - 1.0
    monthly_features = monthly_last(feature_daily).shift(1)
    return monthly_features.apply(pd.to_numeric, errors="coerce")


def _selector_daily_bundle(
    *,
    selection: pd.DataFrame,
    bundles: dict[str, AllocationBundle],
    capital_brl: float,
) -> tuple[StrategyBundle, pd.DataFrame]:
    first_bundle = next(iter(bundles.values())).bundle
    all_idx = first_bundle.result.gross_ret.index
    month_map = pd.Series(selection["selected_mode"], index=selection.index).dropna().astype(str)
    month_end_index = all_idx.to_period("M").to_timestamp(how="end").normalize()
    selected_mode_daily = month_map.reindex(month_end_index).astype(object)
    selected_mode_daily.index = all_idx
    selected_mode_daily = selected_mode_daily.ffill().bfill()

    gross = pd.Series(0.0, index=all_idx, dtype=float)
    turnover = pd.Series(0.0, index=all_idx, dtype=float)
    benchmark = pd.to_numeric(first_bundle.benchmark_gross_ret.reindex(all_idx), errors="coerce").fillna(0.0).astype(float)
    weights = pd.DataFrame(0.0, index=all_idx, columns=["crypto", "equity", "cash"], dtype=float)

    previous_weights: pd.Series | None = None
    previous_mode: str | None = None
    for mode, alloc in bundles.items():
        mask = selected_mode_daily.eq(mode)
        if not bool(mask.any()):
            continue
        gross.loc[mask] = pd.to_numeric(alloc.bundle.result.gross_ret.reindex(all_idx), errors="coerce").fillna(0.0).loc[mask]
        turnover.loc[mask] = pd.to_numeric(alloc.bundle.result.turnover.reindex(all_idx), errors="coerce").fillna(0.0).loc[mask]
        weights.loc[mask, ["crypto", "equity", "cash"]] = alloc.weights.reindex(all_idx).fillna(0.0).loc[mask, ["crypto", "equity", "cash"]]

    for dt in all_idx:
        mode = str(selected_mode_daily.loc[dt]) if pd.notna(selected_mode_daily.loc[dt]) else ""
        current_weights = weights.loc[dt, ["crypto", "equity", "cash"]].astype(float)
        if previous_weights is not None and mode != previous_mode:
            turnover.loc[dt] += float((current_weights - previous_weights).abs().sum() / 2.0)
        previous_weights = current_weights
        previous_mode = mode

    profile = first_bundle.profile
    benchmark_profile = first_bundle.benchmark_profile
    perf = _evaluate_net(
        gross_ret=gross,
        turnover=turnover,
        profile=profile,
        benchmark_ret=benchmark,
        benchmark_profile=benchmark_profile,
    )
    result = StrategyResult(
        suite="meta_mode_selector",
        candidate_id="meta_mode_selector",
        family="meta_selector",
        benchmark_ticker=first_bundle.result.benchmark_ticker,
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
        notes="meta-selector causal de modos com treino mensal usando apenas historico anterior",
    )
    summary = pd.DataFrame(
        {
            "date": all_idx,
            "selected_mode": selected_mode_daily.values,
            "gross_ret": gross.values,
            "turnover": turnover.values,
            "crypto_weight": weights["crypto"].values,
            "equity_weight": weights["equity"].values,
            "cash_weight": weights["cash"].values,
        }
    )
    return StrategyBundle(result=result, benchmark_gross_ret=benchmark, profile=profile, benchmark_profile=benchmark_profile), summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Seletor causal de modos sobre candidatos congelados do laboratorio.")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--capital-brl", type=float, default=10000.0)
    ap.add_argument("--outdir-root", default="results/validation/profit_meta_mode_selector_suite")
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

    bundles, criticality_aux = _candidate_library(built)
    candidate_returns = {name: alloc.bundle.result.net_ret.copy() for name, alloc in bundles.items()}
    monthly_candidates = pd.DataFrame({name: monthly_total_return(series) for name, series in candidate_returns.items()}).sort_index()
    benchmark_monthly = monthly_total_return(next(iter(bundles.values())).bundle.result.benchmark_net_ret.copy()).sort_index()
    feature_frame = _build_feature_frame(built=built, criticality_aux=criticality_aux, candidate_returns=candidate_returns)
    monthly_idx = monthly_candidates.index.intersection(feature_frame.index).intersection(benchmark_monthly.index).sort_values()
    monthly_candidates = monthly_candidates.reindex(monthly_idx)
    feature_frame = feature_frame.reindex(monthly_idx)
    benchmark_monthly = benchmark_monthly.reindex(monthly_idx)

    selector = run_causal_meta_mode_selector(
        feature_frame=feature_frame,
        candidate_returns=monthly_candidates,
        benchmark_returns=benchmark_monthly,
        config=MetaModeSelectorConfig(
            training_months=36,
            min_training_months=24,
            neighbor_months=12,
            downside_penalty=1.10,
            underperformance_penalty=0.50,
            tail_penalty=0.30,
            switch_penalty=0.015,
            min_neighbors=6,
            fallback_mode="best_recent",
        ),
    )
    selector_bundle, selector_daily = _selector_daily_bundle(selection=selector, bundles=bundles, capital_brl=float(args.capital_brl))

    summary_rows = [_result_row(selector_bundle.result)]
    summary_rows.extend(_result_row(alloc.bundle.result) for alloc in bundles.values())
    candidate_df = pd.DataFrame(summary_rows).sort_values(
        ["net_total_return", "net_ann_return", "net_sharpe"],
        ascending=[False, False, False],
    )
    calendar_rows = _calendar_rows(result=selector_bundle.result, capital_brl=float(args.capital_brl))
    for alloc in bundles.values():
        calendar_rows.extend(_calendar_rows(result=alloc.bundle.result, capital_brl=float(args.capital_brl)))
    calendar_df = pd.DataFrame(calendar_rows).sort_values(["year", "profit_brl"], ascending=[True, False])

    selector_modes = (
        selector["selected_mode"]
        .dropna()
        .value_counts()
        .rename_axis("selected_mode")
        .reset_index(name="months_selected")
    )

    candidate_df.to_csv(outdir / "candidate_compare.csv", index=False)
    selector.to_csv(outdir / "selection_timeline.csv", index=True)
    selector_daily.to_csv(outdir / "selection_daily.csv", index=False)
    calendar_df.to_csv(outdir / "yearbook_reais.csv", index=False)
    selector_modes.to_csv(outdir / "selection_summary.csv", index=False)

    research_rows = [
        _research_row(
            selector_bundle.result,
            outdir=outdir,
            status="watch",
            methodology="causal_meta_mode_selector",
            label="Meta-seletor causal de modos",
        )
    ]
    (outdir / "profit_research_rows.json").write_text(
        json.dumps(research_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    best = candidate_df.iloc[0].to_dict() if not candidate_df.empty else {}
    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "outdir": str(outdir),
        "best_candidate_id": str(best.get("candidate_id", "")),
        "best_candidate": best,
        "selector_months": selector_modes.to_dict(orient="records"),
        "selector_confidence_avg": _safe_float(pd.to_numeric(selector["selection_confidence"], errors="coerce").mean()),
        "insights": [
            "O meta-seletor escolhe entre poucos modos congelados usando apenas historico anterior ao mes corrente.",
            "O teste responde se a adaptacao causal entre ataque, protecao e challengers bate os modos fixos.",
            "A selecao e treinada por blocos mensais e evita usar informacao do proprio mes escolhido.",
        ],
        "artifacts": {
            "candidate_compare_csv": str(outdir / "candidate_compare.csv"),
            "selection_timeline_csv": str(outdir / "selection_timeline.csv"),
            "selection_daily_csv": str(outdir / "selection_daily.csv"),
            "yearbook_reais_csv": str(outdir / "yearbook_reais.csv"),
            "selection_summary_csv": str(outdir / "selection_summary.csv"),
            "profit_research_rows_json": str(outdir / "profit_research_rows.json"),
        },
    }
    _write_json(outdir / "summary.json", summary)
    write_run_manifest(
        outdir,
        script="scripts/bench/validation/run_profit_meta_mode_selector_suite.py",
        params={
            "benchmark_crypto": args.benchmark_crypto,
            "benchmark_equity": args.benchmark_equity,
            "capital_brl": args.capital_brl,
            "candidate_library": sorted(bundles.keys()),
        },
        paths=summary["artifacts"],
        extra={"summary_json": str(outdir / "summary.json")},
    )


if __name__ == "__main__":
    main()
