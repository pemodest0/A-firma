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

from engine.portfolio import (  # noqa: E402
    apply_free_energy_penalty,
    build_attractor_persistence_score,
    build_criticality_score,
    build_direction_gradient_score,
    build_market_mode_structure_panel,
    build_state_curvature_score,
)
from engine.portfolio.asymmetric_state_policy import AsymmetricPolicyConfig, next_mode_state  # noqa: E402
from engine.portfolio.exogenous_features import (  # noqa: E402
    build_critical_slowing_down_signal,
    build_structural_stress_signal,
    feature_spectral_extremes,
)
from engine.structural.run_manifest import write_run_manifest  # noqa: E402
from scripts.bench.validation.run_profit_alpha_hardening_suite import (  # noqa: E402
    AllocationBundle,
    _blend_allocation_bundles,
    _build_candidates,
    _confidence_weight_from_score,
)
from scripts.bench.validation.run_profit_alpha_improvement_suite import _safe_float, _write_json  # noqa: E402
from scripts.bench.validation.run_profit_attack_entry_ranking_suite import _result_row  # noqa: E402
from scripts.bench.validation.run_profit_investment_yearbook import _calendar_rows  # noqa: E402
from scripts.bench.validation.run_profit_sector_pressure_suite import _research_row  # noqa: E402


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    if values.empty:
        return values
    min_periods = max(10, int(window) // 3)

    def _pct(arr: np.ndarray) -> float:
        arr = arr[np.isfinite(arr)]
        if arr.size <= 1:
            return float("nan")
        return float(np.mean(arr <= float(arr[-1])))

    return values.rolling(int(window), min_periods=min_periods).apply(_pct, raw=True)


def _build_structure_inputs(context: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, str]]:
    crypto_cols = [
        c
        for c in context["crypto_tiers"]["crypto_major8"]
        if c in context["crypto_returns"].columns
    ]
    equity_returns = context["equity_returns"].copy()
    equity_assets = context["equity_assets"].copy()
    if "ticker" not in equity_assets.columns:
        equity_assets["ticker"] = equity_assets["asset_id"].astype(str)
    if "asset_group" in equity_assets.columns:
        sector_col = "asset_group"
    elif "sector_internal" in equity_assets.columns:
        sector_col = "sector_internal"
    else:
        sector_col = "sector_gics"
    equity_assets[sector_col] = equity_assets[sector_col].fillna("unknown").astype(str)
    selected_equity: list[str] = []
    sector_map: dict[str, str] = {}
    for sector, sub in equity_assets.groupby(sector_col):
        tickers = sorted(set(sub["ticker"].astype(str)))
        kept = [ticker for ticker in tickers if ticker in equity_returns.columns][:2]
        for ticker in kept:
            selected_equity.append(ticker)
            sector_map[ticker] = str(sector)
    selected_equity = sorted(set(selected_equity))
    returns = pd.concat(
        [
            context["crypto_returns"][crypto_cols],
            equity_returns[selected_equity],
        ],
        axis=1,
        sort=False,
    ).dropna(how="all")
    for ticker in crypto_cols:
        sector_map[ticker] = "crypto"
    return returns, sector_map


def _build_structure_layers(context: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    structure_returns, sector_map = _build_structure_inputs(context)
    structure_panel = build_market_mode_structure_panel(
        returns=structure_returns,
        sector_map=sector_map,
        window=120,
        step=5,
    )
    if structure_panel.empty:
        empty_idx = context["attack_score_exogenous"].index
        return (
            pd.DataFrame(index=empty_idx),
            pd.DataFrame(index=empty_idx),
            pd.Series(np.nan, index=empty_idx, dtype=float),
            pd.Series(np.nan, index=empty_idx, dtype=float),
        )
    spectral_panel = pd.DataFrame(
        {
            "lambda1": pd.to_numeric(structure_panel["market_mode_share"], errors="coerce")
            * pd.to_numeric(structure_panel["n_assets"], errors="coerce"),
            "n_assets": pd.to_numeric(structure_panel["n_assets"], errors="coerce"),
            "deff": pd.to_numeric(structure_panel["deff_ratio"], errors="coerce")
            * pd.to_numeric(structure_panel["n_assets"], errors="coerce"),
            "avg_abs_corr": pd.to_numeric(structure_panel["avg_abs_corr"], errors="coerce"),
            "p1": pd.to_numeric(structure_panel["market_mode_share"], errors="coerce"),
        },
        index=structure_panel.index,
    )
    csd = build_critical_slowing_down_signal(
        returns=structure_returns,
        benchmark_col="BTC-USD" if "BTC-USD" in structure_returns.columns else structure_returns.columns[0],
        window=63,
    )
    base_index = context["attack_score_exogenous"].index
    structural_stress = build_structural_stress_signal(
        spectral_panel=spectral_panel,
        index=base_index,
    )
    criticality = build_criticality_score(
        structure_panel=structure_panel,
        critical_slowing_down=csd,
        structural_stress=structural_stress,
        index=base_index,
    )
    structure_daily = structure_panel.reindex(base_index.union(structure_panel.index)).sort_index().ffill().reindex(base_index)
    structure_daily = structure_daily.apply(pd.to_numeric, errors="coerce")
    structure_daily["criticality"] = pd.to_numeric(criticality, errors="coerce").astype(float)
    structure_daily["structural_stress"] = pd.to_numeric(structural_stress, errors="coerce").astype(float)
    return structure_daily, spectral_panel, criticality, structural_stress


def _market_mode_score(base_score: pd.Series, structure_daily: pd.DataFrame) -> pd.Series:
    score = pd.to_numeric(base_score, errors="coerce").fillna(0.0).astype(float)
    mm = pd.to_numeric(structure_daily.get("market_mode_share_pct"), errors="coerce").reindex(score.index).fillna(0.5)
    rotation = pd.to_numeric(structure_daily.get("sector_rotation_score"), errors="coerce").reindex(score.index).fillna(0.5)
    residual = pd.to_numeric(structure_daily.get("residual_dispersion"), errors="coerce").reindex(score.index).fillna(0.5)
    adjusted = score - 0.18 * mm + 0.08 * (rotation - 0.5) + 0.05 * (residual - 0.5)
    return adjusted.clip(0.0, 1.0).astype(float)


def _asymmetric_weight(
    *,
    base_weight: pd.Series,
    attack_signal: pd.Series,
    defense_signal: pd.Series,
    config: AsymmetricPolicyConfig,
) -> pd.Series:
    idx = base_weight.index.intersection(attack_signal.index).intersection(defense_signal.index)
    weights = pd.Series(index=idx, dtype=float)
    state = "PROTECT"
    for dt in idx:
        state = next_mode_state(
            current_state=state,
            attack_signal=float(pd.to_numeric(attack_signal.loc[dt], errors="coerce")),
            defense_signal=float(pd.to_numeric(defense_signal.loc[dt], errors="coerce")),
            config=config,
        )
        weights.loc[dt] = float(base_weight.loc[dt]) if state == "ATTACK" else 0.0
    return weights.reindex(base_weight.index).ffill().fillna(0.0).clip(0.0, 1.0)


def _state_ecosystem_compare(structure_daily: pd.DataFrame, weight: pd.Series, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    idx = structure_daily.index.intersection(weight.index)
    if idx.empty:
        return rows
    frame = structure_daily.reindex(idx)
    w = pd.to_numeric(weight.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    for state, mask in {
        "full_attack": w >= 0.95,
        "mid_attack": (w >= 0.50) & (w < 0.95),
        "protected": w < 0.50,
    }.items():
        sub = frame.loc[mask]
        if sub.empty:
            continue
        rows.append(
            {
                "candidate": label,
                "state": state,
                "market_mode_share": float(pd.to_numeric(sub["market_mode_share"], errors="coerce").mean()),
                "deff_ratio": float(pd.to_numeric(sub["deff_ratio"], errors="coerce").mean()),
                "avg_abs_corr": float(pd.to_numeric(sub["avg_abs_corr"], errors="coerce").mean()),
                "criticality": float(pd.to_numeric(sub["criticality"], errors="coerce").mean()),
                "n_points": int(len(sub)),
            }
        )
    return rows


def build_official_mode_allocations(
    *,
    prices_dir: Path,
    crypto_groups: Path,
    crypto_meta: Path,
    equity_groups: Path,
    equity_meta: Path,
    benchmark_crypto: str,
    benchmark_equity: str,
) -> dict[str, Any]:
    built = _build_candidates(
        prices_dir=prices_dir,
        crypto_groups=crypto_groups,
        crypto_meta=crypto_meta,
        equity_groups=equity_groups,
        equity_meta=equity_meta,
        benchmark_crypto=str(benchmark_crypto),
        benchmark_equity=str(benchmark_equity),
    )
    context = dict(built["context"])
    attack_alloc: AllocationBundle = built["allocations"]["attack"]
    protect_alloc: AllocationBundle = built["allocations"]["baseline_guard"]

    base_score = pd.to_numeric(context["attack_score_exogenous"], errors="coerce").fillna(0.0).clip(0.0, 1.0).astype(float)
    structure_daily, _spectral_panel, criticality, _structural_stress = _build_structure_layers(context)
    criticality_pct = _rolling_percentile(criticality, 120).reindex(base_score.index).fillna(0.5).clip(0.0, 1.0)
    market_mode_share_pct = pd.to_numeric(
        structure_daily.get("market_mode_share_pct"),
        errors="coerce",
    ).reindex(base_score.index).fillna(0.5)
    criticality_rel_score = (
        base_score
        - 0.18 * criticality_pct
        - 0.05 * market_mode_share_pct
    ).clip(0.0, 1.0)
    criticality_rel_weight = _confidence_weight_from_score(criticality_rel_score)
    criticality_bundle = _blend_allocation_bundles(
        candidate_id="criticality_guard_attack",
        notes="reduz a mao quando a criticidade e a concentracao estrutural entram no pior bloco recente",
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        attack_weight=criticality_rel_weight,
    )

    instability = (
        0.60 * pd.to_numeric(criticality, errors="coerce").reindex(base_score.index).fillna(0.5)
        + 0.40 * market_mode_share_pct
    ).clip(0.0, 1.0)
    free_rel_score = apply_free_energy_penalty(
        base_score=criticality_rel_score,
        turnover=attack_alloc.bundle.result.turnover,
        instability=instability,
        gamma=0.06,
        eta=0.08,
    )
    free_rel_weight = _confidence_weight_from_score(free_rel_score)
    free_rel_bundle = _blend_allocation_bundles(
        candidate_id="criticality_free_energy_attack",
        notes="combina criticidade relativa com penalidade leve de reorganizacao",
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        attack_weight=free_rel_weight,
    )

    return {
        "context": context,
        "built": built,
        "official_attack": free_rel_bundle,
        "official_attack_guard": criticality_bundle,
        "official_main": built["allocations"]["baseline"],
        "official_main_guard": built["allocations"]["baseline_guard"],
        "official_notes": {
            "attack": "Ataque com criticidade estrutural e reorganizacao leve.",
            "attack_guard": "Ataque com freio de criticidade mais direto.",
            "main": "Modo principal equilibrado.",
            "main_guard": "Modo principal com protecao reforcada.",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Testa market mode, criticality, assimetria e energia livre sobre o melhor ataque atual.")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--capital-brl", type=float, default=10000.0)
    ap.add_argument("--outdir-root", default="results/validation/profit_marketmode_criticality_suite")
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
    attack_alloc: AllocationBundle = built["allocations"]["attack"]
    protect_alloc: AllocationBundle = built["allocations"]["baseline_guard"]
    baseline_result = attack_alloc.bundle.result
    protected_result = protect_alloc.bundle.result

    base_score = pd.to_numeric(context["attack_score_exogenous"], errors="coerce").fillna(0.0).clip(0.0, 1.0).astype(float)
    base_weight = _confidence_weight_from_score(base_score)

    structure_daily, spectral_panel, criticality, structural_stress = _build_structure_layers(context)
    market_score = _market_mode_score(base_score, structure_daily)
    market_weight = _confidence_weight_from_score(market_score)
    market_bundle = _blend_allocation_bundles(
        candidate_id="market_mode_decomp_attack",
        notes="penaliza modo de mercado dominante e privilegia rotação setorial e dispersão residual",
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        attack_weight=market_weight,
    )

    criticality_score = (base_score - 0.20 * pd.to_numeric(criticality, errors="coerce").reindex(base_score.index).fillna(0.5)).clip(0.0, 1.0)
    criticality_weight = _confidence_weight_from_score(criticality_score)
    criticality_bundle = _blend_allocation_bundles(
        candidate_id="criticality_guard_attack",
        notes="reduz ataque quando a criticidade estrutural sobe",
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        attack_weight=criticality_weight,
    )

    direction_score = build_direction_gradient_score(
        structure_panel=structure_daily,
        criticality=criticality,
        structural_stress=structural_stress,
        index=base_score.index,
    )
    persistence_score = build_attractor_persistence_score(
        direction_score=direction_score,
        criticality=criticality,
        index=base_score.index,
        window=21,
    )
    curvature_score = build_state_curvature_score(
        direction_score=direction_score,
        criticality=criticality,
        index=base_score.index,
    )

    criticality_pct = _rolling_percentile(
        pd.to_numeric(criticality, errors="coerce").reindex(base_score.index).fillna(0.5),
        126,
    ).fillna(0.5)
    market_pct = pd.to_numeric(structure_daily.get("market_mode_share_pct"), errors="coerce").reindex(base_score.index).fillna(0.5)

    rel_penalty = (
        0.22 * ((criticality_pct - 0.55).clip(lower=0.0) / 0.45)
        + 0.06 * ((market_pct - 0.70).clip(lower=0.0) / 0.30)
    ).clip(0.0, 0.35)
    criticality_rel_score = (base_score - rel_penalty).clip(0.0, 1.0)
    criticality_rel_weight = _confidence_weight_from_score(criticality_rel_score)
    criticality_rel_bundle = _blend_allocation_bundles(
        candidate_id="criticality_relative_attack",
        notes="usa criticidade relativa ao proprio historico recente e freia mais so quando o estado fica extremo",
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        attack_weight=criticality_rel_weight,
    )

    extreme_scale = pd.Series(1.0, index=base_score.index, dtype=float)
    extreme_scale.loc[criticality_pct >= 0.60] = 0.85
    extreme_scale.loc[criticality_pct >= 0.78] = 0.62
    extreme_scale.loc[criticality_pct >= 0.90] = 0.35
    extreme_weight = (base_weight * extreme_scale).clip(0.0, 1.0)
    extreme_bundle = _blend_allocation_bundles(
        candidate_id="criticality_extreme_guard_attack",
        notes="so reduz de forma relevante quando a criticidade entra no pior bloco do historico recente",
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        attack_weight=extreme_weight,
    )

    instability = (
        0.60 * pd.to_numeric(criticality, errors="coerce").reindex(base_score.index).fillna(0.5)
        + 0.40 * pd.to_numeric(structure_daily.get("market_mode_share_pct"), errors="coerce").reindex(base_score.index).fillna(0.5)
    ).clip(0.0, 1.0)
    free_score = apply_free_energy_penalty(
        base_score=base_score,
        turnover=attack_alloc.bundle.result.turnover,
        instability=instability,
        gamma=0.10,
        eta=0.14,
    )
    free_weight = _confidence_weight_from_score(free_score)
    free_bundle = _blend_allocation_bundles(
        candidate_id="free_energy_attack",
        notes="penaliza estados caros: giro alto e instabilidade estrutural alta",
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        attack_weight=free_weight,
    )

    free_rel_score = apply_free_energy_penalty(
        base_score=criticality_rel_score,
        turnover=attack_alloc.bundle.result.turnover,
        instability=instability,
        gamma=0.06,
        eta=0.08,
    )
    free_rel_weight = _confidence_weight_from_score(free_rel_score)
    free_rel_bundle = _blend_allocation_bundles(
        candidate_id="criticality_free_energy_attack",
        notes="combina criticidade relativa com penalidade leve de reorganizacao",
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        attack_weight=free_rel_weight,
    )

    direction_attack_score = (
        free_rel_score + 0.10 * (direction_score - 0.5)
    ).clip(0.0, 1.0)
    direction_attack_weight = _confidence_weight_from_score(direction_attack_score)
    direction_attack_bundle = _blend_allocation_bundles(
        candidate_id="direction_gradient_attack",
        notes="usa a direcao recente do estado estrutural para aumentar ou reduzir ataque",
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        attack_weight=direction_attack_weight,
    )

    attractor_attack_score = (
        free_rel_score
        + 0.08 * (direction_score - 0.5)
        + 0.06 * (persistence_score - 0.5)
    ).clip(0.0, 1.0)
    attractor_attack_weight = _confidence_weight_from_score(attractor_attack_score)
    attractor_attack_bundle = _blend_allocation_bundles(
        candidate_id="attractor_persistence_attack",
        notes="amplifica o ataque quando a direcao favoravel tambem parece estavel",
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        attack_weight=attractor_attack_weight,
    )

    curvature_attack_score = (
        free_rel_score
        + 0.08 * (direction_score - 0.5)
        + 0.08 * (curvature_score - 0.5)
    ).clip(0.0, 1.0)
    curvature_attack_weight = _confidence_weight_from_score(curvature_attack_score)
    curvature_attack_bundle = _blend_allocation_bundles(
        candidate_id="curvature_attack",
        notes="usa aceleracao estrutural para antecipar reforco ou perda de tracao do ataque",
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        attack_weight=curvature_attack_weight,
    )

    directional_combo_score = (
        free_rel_score
        + 0.08 * (direction_score - 0.5)
        + 0.05 * (persistence_score - 0.5)
        + 0.05 * (curvature_score - 0.5)
    ).clip(0.0, 1.0)
    directional_combo_weight = _confidence_weight_from_score(directional_combo_score)
    weak_state = ((persistence_score < 0.42) & (curvature_score < 0.40)).reindex(base_score.index).fillna(False)
    directional_combo_weight = directional_combo_weight.where(~weak_state, directional_combo_weight * 0.65).clip(0.0, 1.0)
    directional_combo_bundle = _blend_allocation_bundles(
        candidate_id="directional_dynamics_attack",
        notes="combina direcao, persistencia e aceleracao do estado para dosar o ataque",
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        attack_weight=directional_combo_weight,
    )

    defense_signal = np.maximum(
        pd.to_numeric(criticality, errors="coerce").reindex(base_score.index).fillna(0.5).to_numpy(dtype=float),
        pd.to_numeric(structure_daily.get("market_mode_share_pct"), errors="coerce").reindex(base_score.index).fillna(0.5).to_numpy(dtype=float),
    )
    asym_weight = _asymmetric_weight(
        base_weight=base_weight,
        attack_signal=base_score,
        defense_signal=pd.Series(defense_signal, index=base_score.index, dtype=float),
        config=AsymmetricPolicyConfig(
            enter_attack_threshold=0.74,
            stay_attack_threshold=0.60,
            defense_threshold=0.58,
            release_threshold=0.64,
        ),
    )
    asym_bundle = _blend_allocation_bundles(
        candidate_id="asymmetric_policy_attack",
        notes="entra devagar e sai rápido quando criticidade ou modo de mercado pioram",
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        attack_weight=asym_weight,
    )

    combined_score = _market_mode_score(base_score, structure_daily)
    combined_score = (combined_score - 0.16 * pd.to_numeric(criticality, errors="coerce").reindex(base_score.index).fillna(0.5)).clip(0.0, 1.0)
    combined_score = apply_free_energy_penalty(
        base_score=combined_score,
        turnover=attack_alloc.bundle.result.turnover,
        instability=instability,
        gamma=0.08,
        eta=0.12,
    )
    combined_weight = _confidence_weight_from_score(combined_score)
    combined_weight = _asymmetric_weight(
        base_weight=combined_weight,
        attack_signal=combined_score,
        defense_signal=pd.Series(defense_signal, index=base_score.index, dtype=float),
        config=AsymmetricPolicyConfig(
            enter_attack_threshold=0.72,
            stay_attack_threshold=0.58,
            defense_threshold=0.56,
            release_threshold=0.62,
        ),
    )
    combined_bundle = _blend_allocation_bundles(
        candidate_id="combined_structural_attack",
        notes="junta separação do modo de mercado, criticidade, custo de reorganização e política assimétrica",
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        attack_weight=combined_weight,
    )

    results = {
        baseline_result.candidate_id: baseline_result,
        protected_result.candidate_id: protected_result,
        market_bundle.bundle.result.candidate_id: market_bundle.bundle.result,
        criticality_bundle.bundle.result.candidate_id: criticality_bundle.bundle.result,
        criticality_rel_bundle.bundle.result.candidate_id: criticality_rel_bundle.bundle.result,
        direction_attack_bundle.bundle.result.candidate_id: direction_attack_bundle.bundle.result,
        attractor_attack_bundle.bundle.result.candidate_id: attractor_attack_bundle.bundle.result,
        curvature_attack_bundle.bundle.result.candidate_id: curvature_attack_bundle.bundle.result,
        directional_combo_bundle.bundle.result.candidate_id: directional_combo_bundle.bundle.result,
        extreme_bundle.bundle.result.candidate_id: extreme_bundle.bundle.result,
        free_bundle.bundle.result.candidate_id: free_bundle.bundle.result,
        free_rel_bundle.bundle.result.candidate_id: free_rel_bundle.bundle.result,
        asym_bundle.bundle.result.candidate_id: asym_bundle.bundle.result,
        combined_bundle.bundle.result.candidate_id: combined_bundle.bundle.result,
    }

    compare_rows = [
        _result_row(result, baseline=baseline_result, family="marketmode_criticality", label=result.candidate_id)
        for result in results.values()
    ]
    compare_df = pd.DataFrame(compare_rows).sort_values(
        by=["net_total_return", "net_sharpe"],
        ascending=[False, False],
    )
    compare_df.to_csv(outdir / "candidate_compare.csv", index=False)

    calendar_rows: list[dict[str, Any]] = []
    for result in results.values():
        calendar_rows.extend(_calendar_rows(result=result, capital_brl=float(args.capital_brl)))
    calendar_df = pd.DataFrame(calendar_rows).sort_values(["year", "candidate_id"])
    calendar_df.to_csv(outdir / "yearbook_reais.csv", index=False)

    if not calendar_df.empty:
        base_year = calendar_df[calendar_df["candidate_id"] == baseline_result.candidate_id][["year", "profit_brl", "year_total_return"]].rename(
            columns={"profit_brl": "baseline_profit_brl", "year_total_return": "baseline_year_total_return"}
        )
        year_cmp = calendar_df.merge(base_year, on="year", how="left")
        year_cmp["profit_brl_diff"] = pd.to_numeric(year_cmp["profit_brl"], errors="coerce") - pd.to_numeric(year_cmp["baseline_profit_brl"], errors="coerce")
        year_cmp["year_total_return_diff"] = pd.to_numeric(year_cmp["year_total_return"], errors="coerce") - pd.to_numeric(year_cmp["baseline_year_total_return"], errors="coerce")
        year_cmp.to_csv(outdir / "year_improvement.csv", index=False)

    feature_panel = pd.DataFrame(
        {
            "market_mode_share_pct": pd.to_numeric(structure_daily.get("market_mode_share_pct"), errors="coerce"),
            "sector_rotation_score": pd.to_numeric(structure_daily.get("sector_rotation_score"), errors="coerce"),
            "residual_dispersion": pd.to_numeric(structure_daily.get("residual_dispersion"), errors="coerce"),
            "criticality": pd.to_numeric(criticality, errors="coerce").reindex(base_score.index),
            "structural_stress": pd.to_numeric(structural_stress, errors="coerce").reindex(base_score.index),
            "direction_gradient": pd.to_numeric(direction_score, errors="coerce").reindex(base_score.index),
            "attractor_persistence": pd.to_numeric(persistence_score, errors="coerce").reindex(base_score.index),
            "state_curvature": pd.to_numeric(curvature_score, errors="coerce").reindex(base_score.index),
        },
        index=base_score.index,
    )
    spectral_daily = pd.DataFrame(
        {
            "p1": pd.to_numeric(structure_daily.get("market_mode_share"), errors="coerce"),
            "deff": pd.to_numeric(structure_daily.get("deff_ratio"), errors="coerce") * pd.to_numeric(structure_daily.get("n_assets"), errors="coerce"),
            "avg_abs_corr": pd.to_numeric(structure_daily.get("avg_abs_corr"), errors="coerce"),
            "lambda1": pd.to_numeric(structure_daily.get("market_mode_share"), errors="coerce") * pd.to_numeric(structure_daily.get("n_assets"), errors="coerce"),
        },
        index=base_score.index,
    )
    spectral_effects = feature_spectral_extremes(
        feature_panel=feature_panel,
        spectral_panel=spectral_daily,
        feature_cols=list(feature_panel.columns),
    )
    spectral_effects.to_csv(outdir / "feature_spectral_effects.csv", index=False)

    ecosystem_rows: list[dict[str, Any]] = []
    ecosystem_rows.extend(_state_ecosystem_compare(structure_daily, base_weight, "baseline_attack"))
    ecosystem_rows.extend(_state_ecosystem_compare(structure_daily, market_weight, "market_mode_decomp"))
    ecosystem_rows.extend(_state_ecosystem_compare(structure_daily, criticality_weight, "criticality_guard"))
    ecosystem_rows.extend(_state_ecosystem_compare(structure_daily, criticality_rel_weight, "criticality_relative"))
    ecosystem_rows.extend(_state_ecosystem_compare(structure_daily, direction_attack_weight, "direction_gradient"))
    ecosystem_rows.extend(_state_ecosystem_compare(structure_daily, attractor_attack_weight, "attractor_persistence"))
    ecosystem_rows.extend(_state_ecosystem_compare(structure_daily, curvature_attack_weight, "curvature"))
    ecosystem_rows.extend(_state_ecosystem_compare(structure_daily, directional_combo_weight, "directional_dynamics"))
    ecosystem_rows.extend(_state_ecosystem_compare(structure_daily, extreme_weight, "criticality_extreme_guard"))
    ecosystem_rows.extend(_state_ecosystem_compare(structure_daily, free_rel_weight, "criticality_free_energy"))
    ecosystem_rows.extend(_state_ecosystem_compare(structure_daily, asym_weight, "asymmetric_policy"))
    ecosystem_rows.extend(_state_ecosystem_compare(structure_daily, combined_weight, "combined"))
    ecosystem_df = pd.DataFrame(ecosystem_rows)
    ecosystem_df.to_csv(outdir / "ecosystem_compare.csv", index=False)

    research_rows = [
        _research_row(baseline_result, outdir=outdir, status="keep", methodology="marketmode_criticality_baseline", label="Ataque atual"),
        _research_row(protected_result, outdir=outdir, status="watch", methodology="marketmode_criticality_protected", label="Protecao atual"),
        _research_row(market_bundle.bundle.result, outdir=outdir, status="watch", methodology="market_mode_decomposition", label="Separacao do modo de mercado"),
        _research_row(criticality_bundle.bundle.result, outdir=outdir, status="watch", methodology="criticality_guard", label="Freio por criticidade"),
        _research_row(criticality_rel_bundle.bundle.result, outdir=outdir, status="watch", methodology="criticality_relative_guard", label="Freio por criticidade relativa"),
        _research_row(direction_attack_bundle.bundle.result, outdir=outdir, status="watch", methodology="direction_gradient", label="Direcao do estado"),
        _research_row(attractor_attack_bundle.bundle.result, outdir=outdir, status="watch", methodology="attractor_persistence", label="Persistencia do atrator"),
        _research_row(curvature_attack_bundle.bundle.result, outdir=outdir, status="watch", methodology="state_curvature", label="Curvatura do estado"),
        _research_row(directional_combo_bundle.bundle.result, outdir=outdir, status="watch", methodology="directional_dynamics", label="Direcao + persistencia + curvatura"),
        _research_row(extreme_bundle.bundle.result, outdir=outdir, status="watch", methodology="criticality_extreme_guard", label="Freio so em criticidade extrema"),
        _research_row(free_bundle.bundle.result, outdir=outdir, status="kill", methodology="free_energy_penalty", label="Penalidade de reorganizacao"),
        _research_row(free_rel_bundle.bundle.result, outdir=outdir, status="watch", methodology="criticality_plus_free_energy", label="Criticidade com reorganizacao leve"),
        _research_row(asym_bundle.bundle.result, outdir=outdir, status="watch", methodology="asymmetric_policy", label="Entrada devagar, saida rapida"),
        _research_row(combined_bundle.bundle.result, outdir=outdir, status="watch", methodology="market_mode_plus_criticality", label="Camada estrutural combinada"),
    ]
    (outdir / "profit_research_rows.json").write_text(json_dumps(research_rows), encoding="utf-8")

    best = compare_df.iloc[0].to_dict() if not compare_df.empty else {}
    best_non_baseline = compare_df[compare_df["candidate_id"] != baseline_result.candidate_id]
    best_challenger = best_non_baseline.iloc[0].to_dict() if not best_non_baseline.empty else {}
    summary = {
        "run_id": outdir.name,
        "baseline_attack_candidate": baseline_result.candidate_id,
        "best_candidate": best.get("candidate_id"),
        "best_challenger": best_challenger.get("candidate_id"),
        "worth_promoting": bool(best_challenger.get("net_total_return", float("-inf")) > _safe_float(baseline_result.net_total_return)),
        "baseline_net_ann_return": _safe_float(baseline_result.net_ann_return),
        "baseline_net_total_return": _safe_float(baseline_result.net_total_return),
        "baseline_net_sharpe": _safe_float(baseline_result.net_sharpe),
        "baseline_net_max_drawdown": _safe_float(baseline_result.net_max_drawdown),
        "best_challenger_net_ann_return": _safe_float(best_challenger.get("net_ann_return")),
        "best_challenger_net_total_return": _safe_float(best_challenger.get("net_total_return")),
        "best_challenger_net_sharpe": _safe_float(best_challenger.get("net_sharpe")),
        "best_challenger_net_max_drawdown": _safe_float(best_challenger.get("net_max_drawdown")),
        "compare_file": str(outdir / "candidate_compare.csv"),
        "yearbook_file": str(outdir / "yearbook_reais.csv"),
        "ecosystem_file": str(outdir / "ecosystem_compare.csv"),
        "spectral_effects_file": str(outdir / "feature_spectral_effects.csv"),
    }
    _write_json(outdir / "summary.json", summary)
    write_run_manifest(
        outdir,
        script="scripts/bench/validation/run_profit_marketmode_criticality_suite.py",
        params={
            "benchmark_crypto": args.benchmark_crypto,
            "benchmark_equity": args.benchmark_equity,
            "capital_brl": args.capital_brl,
        },
        paths={
            "summary_json": str(outdir / "summary.json"),
            "candidate_compare_csv": str(outdir / "candidate_compare.csv"),
            "yearbook_reais_csv": str(outdir / "yearbook_reais.csv"),
            "ecosystem_compare_csv": str(outdir / "ecosystem_compare.csv"),
            "feature_spectral_effects_csv": str(outdir / "feature_spectral_effects.csv"),
        },
        extra={
            "suite": "profit_marketmode_criticality_suite",
            "best_candidate": best.get("candidate_id"),
            "best_challenger": best_challenger.get("candidate_id"),
        },
    )


def json_dumps(payload: Any) -> str:
    import json

    return json.dumps(payload, ensure_ascii=True, indent=2, default=str)


if __name__ == "__main__":
    main()
