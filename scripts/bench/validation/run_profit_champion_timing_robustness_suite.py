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
    _blend_allocation_bundles,
    _build_alpha_meta_allocation_bundle,
    _build_candidates,
    _build_promoted_attack_confidence_score,
    _confidence_weight_from_score,
)
from scripts.bench.validation.run_profit_alpha_improvement_suite import _safe_float, _write_json  # noqa: E402
from scripts.bench.validation.run_profit_attack_entry_ranking_suite import _result_row  # noqa: E402
from scripts.bench.validation.run_profit_champion_extension_suite import (  # noqa: E402
    _build_criticality_free_energy_bundle,
)
from scripts.bench.validation.run_profit_confidence_refinement_suite import (  # noqa: E402
    _dynamic_crypto_bundle,
)
from scripts.bench.validation.run_profit_crypto_resolution_suite import (  # noqa: E402
    _blend_crypto_bundles,
    _crypto_rule_bundle,
)
from scripts.bench.validation.run_profit_investment_yearbook import _calendar_rows  # noqa: E402
from scripts.bench.validation.run_profit_layered_engine_suite import (  # noqa: E402
    StrategyBundle,
    _apply_breadth_overlay_to_bundle,
    _build_breadth_signal,
)
from scripts.bench.validation.run_profit_marketmode_criticality_suite import (  # noqa: E402
    _build_structure_layers,
    _rolling_percentile,
)
from scripts.bench.validation.run_profit_pbo_suite import _pbo_for_metric, _pbo_verdict  # noqa: E402
from scripts.bench.validation.run_profit_sector_pressure_suite import _research_row  # noqa: E402
from scripts.bench.validation.run_profit_universe_resilience_suite import _selection_frequency_for_crypto_rule  # noqa: E402


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _clip01(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float).clip(0.0, 1.0)


def _align(series: pd.Series | Any, index: pd.Index, default: float = 0.0) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce").reindex(index).fillna(default).astype(float)
    return pd.Series(default, index=index, dtype=float)


def _underperform_prob_rolling(
    candidate_ret: pd.Series,
    benchmark_ret: pd.Series,
    *,
    horizon: int = 63,
) -> float:
    idx = candidate_ret.index.intersection(benchmark_ret.index)
    if len(idx) < max(8, int(horizon)):
        return float("nan")
    cand = pd.to_numeric(candidate_ret.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    bench = pd.to_numeric(benchmark_ret.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    cand_roll = (1.0 + cand).rolling(int(horizon), min_periods=int(horizon)).apply(np.prod, raw=True) - 1.0
    bench_roll = (1.0 + bench).rolling(int(horizon), min_periods=int(horizon)).apply(np.prod, raw=True) - 1.0
    valid = cand_roll.notna() & bench_roll.notna()
    if int(valid.sum()) == 0:
        return float("nan")
    return float((cand_roll[valid] < bench_roll[valid]).mean())


def _monthly_returns(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if x.empty:
        return pd.Series(dtype=float)
    monthly = x.groupby(pd.to_datetime(x.index).to_period("M")).apply(lambda s: float(np.prod(1.0 + s.to_numpy(dtype=float)) - 1.0))
    monthly.index = monthly.index.astype(str)
    return monthly.astype(float)


def _common_monthly_matrix(results: dict[str, Any]) -> pd.DataFrame:
    monthly_map = {cid: _monthly_returns(result.net_ret) for cid, result in results.items()}
    common = None
    for series in monthly_map.values():
        idx = pd.Index(series.index)
        common = idx if common is None else common.intersection(idx)
    if common is None or len(common) == 0:
        return pd.DataFrame()
    common = pd.Index(sorted(common.astype(str).tolist()))
    data = {cid: series.reindex(common).astype(float) for cid, series in monthly_map.items()}
    return pd.DataFrame(data, index=common).dropna(how="any")


def _candidate_pbo_profile(detail_df: pd.DataFrame, *, metric: str) -> pd.DataFrame:
    if detail_df.empty:
        return pd.DataFrame(columns=["candidate_id", "metric", "pbo_win_splits", "pbo_below_median_rate", "pbo_median_oos_rank"])
    rows: list[dict[str, Any]] = []
    for cid, sub in detail_df.groupby("winner_candidate_id"):
        rows.append(
            {
                "candidate_id": str(cid),
                "metric": str(metric),
                "pbo_win_splits": int(sub.shape[0]),
                "pbo_below_median_rate": float(pd.to_numeric(sub["winner_below_median"], errors="coerce").mean()),
                "pbo_median_oos_rank": float(pd.to_numeric(sub["winner_oos_rank_desc"], errors="coerce").median()),
            }
        )
    return pd.DataFrame(rows)


def _build_structural_openness(
    *,
    base_score: pd.Series,
    criticality: pd.Series,
    structural_stress: pd.Series,
    market_mode_share_pct: pd.Series,
    liquidation: pd.Series | None = None,
) -> pd.Series:
    idx = base_score.index
    score = _clip01(base_score)
    crit_pct = _rolling_percentile(_clip01(criticality).reindex(idx).fillna(0.5), 126).fillna(0.5)
    stress_pct = _rolling_percentile(_clip01(structural_stress).reindex(idx).fillna(0.5), 126).fillna(0.5)
    market = _clip01(market_mode_share_pct).reindex(idx).fillna(0.5)
    liq = _clip01(liquidation).reindex(idx).fillna(0.5) if isinstance(liquidation, pd.Series) else pd.Series(0.5, index=idx, dtype=float)
    openness = (
        0.40 * score
        + 0.20 * (1.0 - crit_pct)
        + 0.20 * (1.0 - stress_pct)
        + 0.10 * (1.0 - market)
        + 0.10 * (1.0 - liq)
    ).clip(0.0, 1.0)
    return openness.shift(1).ffill().fillna(0.5).clip(0.0, 1.0)


def _selective_score(base_score: pd.Series, openness: pd.Series) -> pd.Series:
    idx = base_score.index.intersection(openness.index)
    score = _clip01(base_score.reindex(idx))
    open_ = _clip01(openness.reindex(idx))
    scale = pd.Series(0.28, index=idx, dtype=float)
    scale.loc[open_ >= 0.54] = 0.72
    scale.loc[open_ >= 0.66] = 1.02
    return (score * scale).clip(0.0, 1.0)


def _crypto_relief_risk(
    *,
    context: dict[str, Any],
    top1_bundle: StrategyBundle,
    broad_bundle: StrategyBundle,
    breadth_signal: pd.Series,
) -> pd.Series:
    attack_returns = pd.concat(
        {
            "crypto": pd.to_numeric(top1_bundle.result.gross_ret, errors="coerce"),
            "equity": pd.to_numeric(context["equity_attack"].result.gross_ret, errors="coerce"),
        },
        axis=1,
        sort=False,
    ).dropna(how="all")
    confidence = _clip01(
        _build_promoted_attack_confidence_score(
            {
                "btc_prices": context["btc_prices"],
                "spy_prices": context["spy_prices"],
                "regime_series": context["regime_series"],
            },
            attack_returns,
        )
    )
    idx = (
        pd.to_numeric(top1_bundle.result.gross_ret, errors="coerce").dropna().index
        .intersection(pd.to_numeric(broad_bundle.result.gross_ret, errors="coerce").dropna().index)
        .intersection(confidence.index)
        .intersection(breadth_signal.index)
    )
    top1 = pd.to_numeric(top1_bundle.result.gross_ret.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    broad = pd.to_numeric(broad_bundle.result.gross_ret.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    breadth = pd.to_numeric(breadth_signal.reindex(idx), errors="coerce").fillna(0.0).clip(0.0, 1.0).astype(float)
    dominance = _rolling_percentile((top1 - broad).clip(lower=0.0), 63).reindex(idx).fillna(0.5)
    low_conf = 1.0 - pd.to_numeric(confidence.reindex(idx), errors="coerce").fillna(0.5).clip(0.0, 1.0)
    low_breadth = 1.0 - breadth
    return (0.45 * low_breadth + 0.35 * dominance + 0.20 * low_conf).clip(0.0, 1.0)


def _build_conditional_crypto_bundle(
    *,
    context: dict[str, Any],
    drop_tickers: list[str] | None = None,
) -> StrategyBundle:
    blocked = set(drop_tickers or [])
    tiers = context["crypto_tiers"]
    major8 = [ticker for ticker in tiers["crypto_major8"] if ticker not in blocked]
    all22 = [ticker for ticker in tiers["crypto_all"] if ticker not in blocked]
    top1 = _crypto_rule_bundle(
        candidate_id="timing_robust_top1",
        allowed_tickers=major8,
        score_mode="mom_total",
        top_k=1,
        context=context,
    )
    broad = _crypto_rule_bundle(
        candidate_id="timing_robust_broad",
        allowed_tickers=all22,
        score_mode="mom_vol_adj",
        top_k=3,
        context=context,
    )
    major8_div = _crypto_rule_bundle(
        candidate_id="timing_robust_major8_div",
        allowed_tickers=major8,
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
    broad_breadth = _apply_breadth_overlay_to_bundle(
        candidate_id="timing_robust_broad_breadth",
        bundle=broad,
        breadth_signal=breadth_signal,
        low_threshold=0.38,
        high_threshold=0.62,
        mode="scale",
    )
    blend70 = _blend_crypto_bundles(
        candidate_id="timing_robust_blend70",
        primary=top1,
        secondary=major8_div,
        primary_weight=0.70,
    )
    risk = _crypto_relief_risk(
        context=context,
        top1_bundle=top1,
        broad_bundle=broad_breadth,
        breadth_signal=breadth_signal,
    )
    return _dynamic_crypto_bundle(
        candidate_id="timing_robust_conditional",
        high_bundle=top1,
        mid_bundle=blend70,
        low_bundle=broad_breadth,
        score=1.0 - risk,
    )


def _build_raw_attack_alloc_and_score(
    *,
    candidate_id: str,
    crypto_bundle: StrategyBundle,
    equity_bundle: StrategyBundle,
    context: dict[str, Any],
    entry_lookback: int,
    exit_lookback: int,
    entry_margin: float,
    exit_margin: float,
) -> tuple[AllocationBundle, pd.Series]:
    raw_attack = _build_alpha_meta_allocation_bundle(
        candidate_id=f"{candidate_id}__raw",
        crypto_bundle=crypto_bundle,
        equity_bundle=equity_bundle,
        btc_prices=context["btc_prices"],
        spy_prices=context["spy_prices"],
        profile=context["profiles"]["blended"],
        entry_lookback=entry_lookback,
        exit_lookback=exit_lookback,
        entry_margin=entry_margin,
        exit_margin=exit_margin,
        risk_off_mode="equity25",
        min_crypto_hold_days=0,
    )
    sleeve_returns = pd.concat(
        {
            "crypto": pd.to_numeric(crypto_bundle.result.gross_ret, errors="coerce"),
            "equity": pd.to_numeric(equity_bundle.result.gross_ret, errors="coerce"),
        },
        axis=1,
        sort=False,
    ).dropna(how="all")
    score = _build_promoted_attack_confidence_score(
        {
            "btc_prices": context["btc_prices"],
            "spy_prices": context["spy_prices"],
            "regime_series": context["regime_series"],
        },
        sleeve_returns,
    )
    liquidation = context["exogenous_panel"].get("liquidation")
    if isinstance(liquidation, pd.Series):
        score = score - 0.14 * _align(liquidation, score.index, default=0.0)
    return raw_attack, _clip01(score)


def _build_baseline_top_frequency(context: dict[str, Any], benchmark_crypto: str) -> pd.DataFrame:
    return _selection_frequency_for_crypto_rule(
        allowed_tickers=context["crypto_tiers"]["crypto_major8"],
        returns=context["crypto_returns"],
        prices=context["crypto_prices"],
        benchmark_ticker=str(benchmark_crypto),
        lookback_days=21,
        rebalance_days=7,
        top_k=1,
        score_mode="mom_total",
        asset_ma_days=0,
        market_ma_days=200,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Ataca curto prazo, dependência cripto e robustez do campeão com challengers mais seletivos e simples.")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--capital-brl", type=float, default=10000.0)
    ap.add_argument("--outdir-root", default="results/validation/profit_champion_timing_robustness_suite")
    args = ap.parse_args()

    outdir = (ROOT / args.outdir_root / _run_id()).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[timing_robustness] outdir={outdir}", flush=True)

    built = _build_candidates(
        prices_dir=(ROOT / args.prices_dir).resolve(),
        crypto_groups=(ROOT / args.crypto_asset_groups).resolve(),
        crypto_meta=(ROOT / args.crypto_asset_metadata).resolve(),
        equity_groups=(ROOT / args.equity_asset_groups).resolve(),
        equity_meta=(ROOT / args.equity_asset_metadata).resolve(),
        benchmark_crypto=str(args.benchmark_crypto),
        benchmark_equity=str(args.benchmark_equity),
    )
    print("[timing_robustness] base candidates ready", flush=True)
    context = dict(built["context"])
    attack_alloc: AllocationBundle = built["allocations"]["attack"]
    protect_alloc: AllocationBundle = built["allocations"]["baseline_guard"]

    structure_daily, _spectral_panel, criticality, structural_stress = _build_structure_layers(context)
    base_score = _clip01(context["attack_score_exogenous"])
    criticality_aligned = _align(criticality, base_score.index, default=0.5)
    structural_stress_aligned = _align(structural_stress, base_score.index, default=0.5)
    market_mode_share_pct = _align(structure_daily.get("market_mode_share_pct"), base_score.index, default=0.5)
    openness = _build_structural_openness(
        base_score=base_score,
        criticality=criticality_aligned,
        structural_stress=structural_stress_aligned,
        market_mode_share_pct=market_mode_share_pct,
        liquidation=context["exogenous_panel"].get("liquidation"),
    )
    selective_base_score = _selective_score(base_score, openness)

    champion_bundle, _champion_score, _champion_weight = _build_criticality_free_energy_bundle(
        candidate_id="criticality_free_energy_attack",
        notes="campeão atual: criticidade com reorganização leve",
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        base_score=base_score,
        structure_daily=structure_daily,
        criticality=criticality_aligned,
    )
    print("[timing_robustness] champion baseline ready", flush=True)

    selective_bundle, selective_score, _selective_weight = _build_criticality_free_energy_bundle(
        candidate_id="champion_selective_convexity",
        notes="ataque só abre de verdade quando estrutura e confiança concordam; usa o ataque mais como convexidade seletiva",
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        base_score=selective_base_score,
        structure_daily=structure_daily,
        criticality=criticality_aligned,
    )
    print("[timing_robustness] selective convexity ready", flush=True)

    top1_crypto = _crypto_rule_bundle(
        candidate_id="timing_robust_top1_major8",
        allowed_tickers=context["crypto_tiers"]["crypto_major8"],
        score_mode="mom_total",
        top_k=1,
        context=context,
    )
    conditional_crypto = _build_conditional_crypto_bundle(
        context=context,
        drop_tickers=None,
    )
    conditional_attack_alloc, conditional_raw_score = _build_raw_attack_alloc_and_score(
        candidate_id="champion_conditional_crypto",
        crypto_bundle=conditional_crypto,
        equity_bundle=context["equity_attack"],
        context=context,
        entry_lookback=14,
        exit_lookback=63,
        entry_margin=0.02,
        exit_margin=0.04,
    )
    conditional_score = _selective_score(conditional_raw_score, openness)
    conditional_bundle, _conditional_score_final, _conditional_weight = _build_criticality_free_energy_bundle(
        candidate_id="champion_conditional_crypto",
        notes="usa alívio condicional do sleeve cripto e gate seletivo para reduzir atraso e dependência do top1",
        attack_alloc=conditional_attack_alloc,
        protect_alloc=protect_alloc,
        base_score=conditional_score,
        structure_daily=structure_daily,
        criticality=criticality_aligned,
    )
    print("[timing_robustness] conditional crypto ready", flush=True)

    equity_base_attack_alloc, equity_base_raw_score = _build_raw_attack_alloc_and_score(
        candidate_id="champion_equity_base_selective",
        crypto_bundle=top1_crypto,
        equity_bundle=context["equity_base"],
        context=context,
        entry_lookback=14,
        exit_lookback=63,
        entry_margin=0.02,
        exit_margin=0.05,
    )
    equity_base_score = _selective_score(equity_base_raw_score, openness)
    equity_base_bundle, _equity_base_score_final, _equity_base_weight = _build_criticality_free_energy_bundle(
        candidate_id="champion_equity_base_selective",
        notes="mantém o top1 cripto, mas troca a perna de ações por uma versão mais robusta para ajudar nos anos normais",
        attack_alloc=equity_base_attack_alloc,
        protect_alloc=protect_alloc,
        base_score=equity_base_score,
        structure_daily=structure_daily,
        criticality=criticality_aligned,
    )
    print("[timing_robustness] equity base selective ready", flush=True)

    simple_weight = _confidence_weight_from_score(selective_base_score)
    simple_bundle = _blend_allocation_bundles(
        candidate_id="champion_simple_selective",
        notes="versão mais simples do campeão: gate seletivo e blend direto, sem penalidade de reorganização",
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        attack_weight=simple_weight,
    )
    print("[timing_robustness] simple selective ready", flush=True)

    results = {
        champion_bundle.bundle.result.candidate_id: champion_bundle.bundle.result,
        selective_bundle.bundle.result.candidate_id: selective_bundle.bundle.result,
        conditional_bundle.bundle.result.candidate_id: conditional_bundle.bundle.result,
        equity_base_bundle.bundle.result.candidate_id: equity_base_bundle.bundle.result,
        simple_bundle.bundle.result.candidate_id: simple_bundle.bundle.result,
    }

    top_freq = _build_baseline_top_frequency(context, str(args.benchmark_crypto))
    top1_removed = top_freq["ticker"].head(1).astype(str).tolist()
    top3_removed = top_freq["ticker"].head(3).astype(str).tolist()

    drop_results_top1: dict[str, Any] = {}
    drop_results_top3: dict[str, Any] = {}
    for dropped, holder in [(top1_removed, drop_results_top1), (top3_removed, drop_results_top3)]:
        print(f"[timing_robustness] dependency rerun drop={','.join(dropped)}", flush=True)
        dropped_set = set(dropped)
        reduced_major8 = [ticker for ticker in context["crypto_tiers"]["crypto_major8"] if ticker not in dropped_set]
        reduced_top1 = _crypto_rule_bundle(
            candidate_id="timing_robust_drop_top",
            allowed_tickers=reduced_major8,
            score_mode="mom_total",
            top_k=1,
            context=context,
        )
        reduced_conditional = _build_conditional_crypto_bundle(context=context, drop_tickers=dropped)
        reduced_selective_attack_alloc, _ = _build_raw_attack_alloc_and_score(
            candidate_id="drop_selective",
            crypto_bundle=reduced_top1,
            equity_bundle=context["equity_attack"],
            context=context,
            entry_lookback=14,
            exit_lookback=63,
            entry_margin=0.02,
            exit_margin=0.05,
        )
        reduced_equity_base_alloc, _ = _build_raw_attack_alloc_and_score(
            candidate_id="drop_equity_base",
            crypto_bundle=reduced_top1,
            equity_bundle=context["equity_base"],
            context=context,
            entry_lookback=14,
            exit_lookback=63,
            entry_margin=0.02,
            exit_margin=0.05,
        )
        reduced_conditional_alloc, _ = _build_raw_attack_alloc_and_score(
            candidate_id="drop_conditional",
            crypto_bundle=reduced_conditional,
            equity_bundle=context["equity_attack"],
            context=context,
            entry_lookback=14,
            exit_lookback=63,
            entry_margin=0.02,
            exit_margin=0.04,
        )
        holder["criticality_free_energy_attack"] = _build_criticality_free_energy_bundle(
            candidate_id="drop_baseline",
            notes="drop crypto leaders baseline",
            attack_alloc=reduced_selective_attack_alloc,
            protect_alloc=protect_alloc,
            base_score=base_score,
            structure_daily=structure_daily,
            criticality=criticality_aligned,
        )[0].bundle.result
        holder["champion_selective_convexity"] = _build_criticality_free_energy_bundle(
            candidate_id="drop_selective",
            notes="drop crypto leaders selective",
            attack_alloc=reduced_selective_attack_alloc,
            protect_alloc=protect_alloc,
            base_score=selective_base_score,
            structure_daily=structure_daily,
            criticality=criticality_aligned,
        )[0].bundle.result
        holder["champion_conditional_crypto"] = _build_criticality_free_energy_bundle(
            candidate_id="drop_conditional",
            notes="drop crypto leaders conditional",
            attack_alloc=reduced_conditional_alloc,
            protect_alloc=protect_alloc,
            base_score=conditional_score,
            structure_daily=structure_daily,
            criticality=criticality_aligned,
        )[0].bundle.result
        holder["champion_equity_base_selective"] = _build_criticality_free_energy_bundle(
            candidate_id="drop_equity_base",
            notes="drop crypto leaders equity base",
            attack_alloc=reduced_equity_base_alloc,
            protect_alloc=protect_alloc,
            base_score=equity_base_score,
            structure_daily=structure_daily,
            criticality=criticality_aligned,
        )[0].bundle.result
        holder["champion_simple_selective"] = _blend_allocation_bundles(
            candidate_id="drop_simple",
            notes="drop crypto leaders simple",
            attack_alloc=reduced_selective_attack_alloc,
            protect_alloc=protect_alloc,
            attack_weight=simple_weight,
        ).bundle.result
    print("[timing_robustness] dependency reruns ready", flush=True)

    monthly_matrix = _common_monthly_matrix(results)
    pbo_metric_summary: dict[str, Any] = {}
    pbo_profile_frames: list[pd.DataFrame] = []
    pbo_split_frames: list[pd.DataFrame] = []
    if not monthly_matrix.empty and len(monthly_matrix.columns) >= 2:
        for metric in ("total_return", "sharpe"):
            split_df, metric_summary = _pbo_for_metric(monthly_matrix, metric=metric, n_slices=8)
            metric_summary["verdict"] = _pbo_verdict(float(metric_summary.get("pbo", float("nan"))))
            pbo_metric_summary[str(metric)] = metric_summary
            if not split_df.empty:
                split_df["metric"] = str(metric)
                pbo_split_frames.append(split_df)
                pbo_profile_frames.append(_candidate_pbo_profile(split_df, metric=str(metric)))

    pbo_profile_df = pd.concat(pbo_profile_frames, axis=0, ignore_index=True) if pbo_profile_frames else pd.DataFrame()
    pbo_split_df = pd.concat(pbo_split_frames, axis=0, ignore_index=True) if pbo_split_frames else pd.DataFrame()
    if not pbo_profile_df.empty:
        pbo_candidate_summary = (
            pbo_profile_df.groupby("candidate_id", as_index=False)
            .agg(
                pbo_win_splits=("pbo_win_splits", "sum"),
                pbo_below_median_rate=("pbo_below_median_rate", "mean"),
                pbo_median_oos_rank=("pbo_median_oos_rank", "mean"),
            )
        )
    else:
        pbo_candidate_summary = pd.DataFrame(columns=["candidate_id", "pbo_win_splits", "pbo_below_median_rate", "pbo_median_oos_rank"])

    rows: list[dict[str, Any]] = []
    for cid, result in results.items():
        base_row = _result_row(result, baseline=champion_bundle.bundle.result, family="champion_timing_robustness", label=cid)
        base_row["underperform_prob_63"] = _underperform_prob_rolling(result.net_ret, result.benchmark_net_ret, horizon=63)
        drop1 = drop_results_top1.get(cid)
        drop3 = drop_results_top3.get(cid)
        base_total = _safe_float(result.net_total_return)
        base_ann = _safe_float(result.net_ann_return)
        drop1_total = _safe_float(drop1.net_total_return) if drop1 is not None else float("nan")
        drop3_total = _safe_float(drop3.net_total_return) if drop3 is not None else float("nan")
        drop1_ann = _safe_float(drop1.net_ann_return) if drop1 is not None else float("nan")
        drop3_ann = _safe_float(drop3.net_ann_return) if drop3 is not None else float("nan")
        base_row["top1_total_retention"] = drop1_total / base_total if np.isfinite(base_total) and abs(base_total) > 1e-9 and np.isfinite(drop1_total) else float("nan")
        base_row["top3_total_retention"] = drop3_total / base_total if np.isfinite(base_total) and abs(base_total) > 1e-9 and np.isfinite(drop3_total) else float("nan")
        base_row["top1_ann_retention"] = drop1_ann / base_ann if np.isfinite(base_ann) and abs(base_ann) > 1e-9 and np.isfinite(drop1_ann) else float("nan")
        base_row["top3_ann_retention"] = drop3_ann / base_ann if np.isfinite(base_ann) and abs(base_ann) > 1e-9 and np.isfinite(drop3_ann) else float("nan")
        base_row["complexity_rank"] = {
            "criticality_free_energy_attack": 4,
            "champion_selective_convexity": 5,
            "champion_conditional_crypto": 6,
            "champion_equity_base_selective": 5,
            "champion_simple_selective": 2,
        }.get(cid, 5)
        rows.append(base_row)

    compare_df = pd.DataFrame(rows)
    compare_df = compare_df.merge(pbo_candidate_summary, on="candidate_id", how="left")
    compare_df["pbo_below_median_rate"] = pd.to_numeric(compare_df["pbo_below_median_rate"], errors="coerce").fillna(1.0)
    compare_df["pbo_median_oos_rank"] = pd.to_numeric(compare_df["pbo_median_oos_rank"], errors="coerce").fillna(float(compare_df.shape[0]))
    compare_df["robustness_score"] = (
        0.28 * pd.to_numeric(compare_df["net_total_return"], errors="coerce").rank(pct=True)
        + 0.18 * pd.to_numeric(compare_df["net_sharpe"], errors="coerce").rank(pct=True)
        + 0.18 * (-pd.to_numeric(compare_df["underperform_prob_63"], errors="coerce")).rank(pct=True)
        + 0.16 * pd.to_numeric(compare_df["top3_total_retention"], errors="coerce").rank(pct=True)
        + 0.12 * (-pd.to_numeric(compare_df["pbo_below_median_rate"], errors="coerce")).rank(pct=True)
        + 0.08 * (-pd.to_numeric(compare_df["complexity_rank"], errors="coerce")).rank(pct=True)
    )
    compare_df = compare_df.sort_values(["robustness_score", "net_total_return"], ascending=[False, False]).reset_index(drop=True)
    compare_df.to_csv(outdir / "candidate_compare.csv", index=False)

    calendar_rows: list[dict[str, Any]] = []
    for result in results.values():
        calendar_rows.extend(_calendar_rows(result=result, capital_brl=float(args.capital_brl)))
    calendar_df = pd.DataFrame(calendar_rows).sort_values(["year", "candidate_id"])
    calendar_df.to_csv(outdir / "yearbook_reais.csv", index=False)
    if not calendar_df.empty:
        base_year = calendar_df[calendar_df["candidate_id"] == champion_bundle.bundle.result.candidate_id][["year", "profit_brl", "year_total_return"]].rename(
            columns={"profit_brl": "baseline_profit_brl", "year_total_return": "baseline_year_total_return"}
        )
        year_cmp = calendar_df.merge(base_year, on="year", how="left")
        year_cmp["profit_brl_diff"] = pd.to_numeric(year_cmp["profit_brl"], errors="coerce") - pd.to_numeric(year_cmp["baseline_profit_brl"], errors="coerce")
        year_cmp["year_total_return_diff"] = pd.to_numeric(year_cmp["year_total_return"], errors="coerce") - pd.to_numeric(year_cmp["baseline_year_total_return"], errors="coerce")
        year_cmp.to_csv(outdir / "year_improvement.csv", index=False)

    if not pbo_split_df.empty:
        pbo_split_df.to_csv(outdir / "pbo_split_results.csv", index=False)
    if not pbo_profile_df.empty:
        pbo_profile_df.to_csv(outdir / "pbo_candidate_profile.csv", index=False)

    baseline_row = compare_df.loc[compare_df["candidate_id"] == "criticality_free_energy_attack"].iloc[0].to_dict()
    best_row = compare_df.iloc[0].to_dict() if not compare_df.empty else {}
    best_id = str(best_row.get("candidate_id", "criticality_free_energy_attack"))
    worth_keeping = bool(
        best_id != "criticality_free_energy_attack"
        and _safe_float(best_row.get("underperform_prob_63")) <= _safe_float(baseline_row.get("underperform_prob_63")) - 0.03
        and _safe_float(best_row.get("top3_total_retention")) >= _safe_float(baseline_row.get("top3_total_retention")) + 0.05
        and _safe_float(best_row.get("net_total_return")) >= _safe_float(baseline_row.get("net_total_return")) - 0.15 * max(1.0, abs(_safe_float(baseline_row.get("net_total_return"))))
    )
    worth_promoting = bool(
        worth_keeping
        and _safe_float(best_row.get("net_sharpe")) >= _safe_float(baseline_row.get("net_sharpe"))
    )

    pbo_overall = {
        metric: {
            **payload,
            "verdict": str(payload.get("verdict") or ""),
        }
        for metric, payload in pbo_metric_summary.items()
    }
    summary = {
        "suite": "profit_champion_timing_robustness_suite",
        "baseline_candidate": "criticality_free_energy_attack",
        "best_candidate": best_id,
        "worth_keeping": worth_keeping,
        "worth_promoting": worth_promoting,
        "baseline_total_return": _safe_float(baseline_row.get("net_total_return")),
        "baseline_ann_return": _safe_float(baseline_row.get("net_ann_return")),
        "baseline_sharpe": _safe_float(baseline_row.get("net_sharpe")),
        "baseline_mdd": _safe_float(baseline_row.get("net_max_drawdown")),
        "baseline_underperform_prob_63": _safe_float(baseline_row.get("underperform_prob_63")),
        "baseline_top3_total_retention": _safe_float(baseline_row.get("top3_total_retention")),
        "best_total_return": _safe_float(best_row.get("net_total_return")),
        "best_ann_return": _safe_float(best_row.get("net_ann_return")),
        "best_sharpe": _safe_float(best_row.get("net_sharpe")),
        "best_mdd": _safe_float(best_row.get("net_max_drawdown")),
        "best_underperform_prob_63": _safe_float(best_row.get("underperform_prob_63")),
        "best_top3_total_retention": _safe_float(best_row.get("top3_total_retention")),
        "pbo_overall": pbo_overall,
        "top1_removed_for_dependency_test": top1_removed,
        "top3_removed_for_dependency_test": top3_removed,
    }
    _write_json(outdir / "summary.json", summary)

    research_rows = [
        _research_row(champion_bundle.bundle.result, outdir=outdir, status="keep", methodology="criticality_plus_free_energy", label="Campeão atual"),
        _research_row(selective_bundle.bundle.result, outdir=outdir, status="watch", methodology="champion_selective_convexity", label="Campeão com timing seletivo"),
        _research_row(conditional_bundle.bundle.result, outdir=outdir, status="watch", methodology="champion_conditional_crypto", label="Campeão com cripto condicional"),
        _research_row(equity_base_bundle.bundle.result, outdir=outdir, status="watch", methodology="champion_equity_base_selective", label="Campeão com equities mais robustas"),
        _research_row(simple_bundle.bundle.result, outdir=outdir, status="watch", methodology="champion_simple_selective", label="Campeão simplificado"),
    ]
    _write_json(outdir / "profit_research_rows.json", research_rows)

    write_run_manifest(
        outdir,
        script="scripts/bench/validation/run_profit_champion_timing_robustness_suite.py",
        params=vars(args),
        extra={
            "suite": "profit_champion_timing_robustness_suite",
            "baseline_candidate": "criticality_free_energy_attack",
            "best_candidate": best_id,
            "worth_keeping": worth_keeping,
            "worth_promoting": worth_promoting,
        },
    )
    print("[timing_robustness] completed", flush=True)


if __name__ == "__main__":
    main()
