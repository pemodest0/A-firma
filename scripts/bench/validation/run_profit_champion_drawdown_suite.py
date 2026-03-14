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

from engine.portfolio.regime_allocator import map_risk_state_to_exposure  # noqa: E402
from engine.structural.run_manifest import write_run_manifest  # noqa: E402
from scripts.bench.validation.run_profit_alpha_hardening_suite import (  # noqa: E402
    _blend_allocation_bundles,
    _build_candidates,
    _confidence_weight_from_score,
)
from scripts.bench.validation.run_profit_alpha_improvement_suite import _safe_float, _write_json  # noqa: E402
from scripts.bench.validation.run_profit_attack_entry_ranking_suite import _result_row  # noqa: E402
from scripts.bench.validation.run_profit_champion_extension_suite import (  # noqa: E402
    _build_criticality_free_energy_bundle,
)
from scripts.bench.validation.run_profit_investment_yearbook import _calendar_rows  # noqa: E402
from scripts.bench.validation.run_profit_marketmode_criticality_suite import (  # noqa: E402
    _build_structure_layers,
    _rolling_percentile,
)
from scripts.bench.validation.run_profit_sector_pressure_suite import _research_row  # noqa: E402


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _clip01(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float).clip(0.0, 1.0)


def _align(series: pd.Series | pd.Index | Any, index: pd.Index, default: float = 0.0) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce").reindex(index).fillna(default).astype(float)
    return pd.Series(default, index=index, dtype=float)


def _hazard_component(series: pd.Series, *, threshold: float, max_weight: float) -> pd.Series:
    denom = max(1e-9, 1.0 - float(threshold))
    return max_weight * ((series - float(threshold)).clip(lower=0.0) / denom)


def _build_early_stress_scale(
    *,
    criticality: pd.Series,
    structural_stress: pd.Series,
    market_mode_share_pct: pd.Series,
) -> pd.Series:
    crit = _clip01(criticality)
    stress = _clip01(structural_stress)
    market = _clip01(market_mode_share_pct)
    crit_pct = _rolling_percentile(crit, 126).fillna(0.5)
    stress_pct = _rolling_percentile(stress, 126).fillna(0.5)
    hazard = (
        _hazard_component(crit_pct, threshold=0.58, max_weight=0.38)
        + _hazard_component(stress_pct, threshold=0.55, max_weight=0.34)
        + _hazard_component(crit, threshold=0.57, max_weight=0.18)
        + _hazard_component(stress, threshold=0.60, max_weight=0.18)
        + _hazard_component(market, threshold=0.68, max_weight=0.10)
    ).clip(0.0, 0.72)
    return (1.0 - hazard).clip(0.18, 1.0)


def _build_regime_cap(
    *,
    criticality: pd.Series,
    structural_stress: pd.Series,
    market_mode_share_pct: pd.Series,
) -> pd.Series:
    crit = _clip01(criticality)
    stress = _clip01(structural_stress)
    market = _clip01(market_mode_share_pct)
    cap = pd.Series(1.0, index=crit.index, dtype=float)
    transition_mask = (stress >= 0.58) | (crit >= 0.56) | (market >= 0.68)
    hard_mask = (stress >= 0.68) | (crit >= 0.63) | (market >= 0.76)
    stress_mask = (stress >= 0.76) | (crit >= 0.72) | (market >= 0.84)
    cap.loc[transition_mask] = 0.55
    cap.loc[hard_mask] = 0.32
    cap.loc[stress_mask] = 0.14
    return cap.clip(0.10, 1.0)


def _build_gradual_posture_cap(
    *,
    base_score: pd.Series,
    criticality: pd.Series,
    structural_stress: pd.Series,
) -> pd.Series:
    idx = base_score.index
    score = _clip01(base_score)
    crit = _clip01(criticality)
    stress = _clip01(structural_stress)
    caps = pd.Series(index=idx, dtype=float)
    for dt in idx:
        period_action = "NORMAL"
        if float(stress.loc[dt]) >= 0.72 or float(crit.loc[dt]) >= 0.70:
            period_action = "PROTECTED"
        elif float(stress.loc[dt]) >= 0.58 or float(crit.loc[dt]) >= 0.60:
            period_action = "REDUCED_ATTACK"
        profile = map_risk_state_to_exposure(
            signal_bundle={
                "confidence_score": float(score.loc[dt]),
                "structural_stress": float(stress.loc[dt]),
            },
            guards={
                "year_bad_state": False,
                "period_action": period_action,
            },
            config={
                "attack_full_confidence_threshold": 0.74,
                "attack_full_stress_threshold": 0.44,
                "neutral_confidence_threshold": 0.60,
                "protected_stress_threshold": 0.70,
            },
        )
        caps.loc[dt] = float(profile.attack_fraction)
    return caps.reindex(idx).ffill().fillna(0.10).clip(0.10, 1.0)


def _summarize_keep_decision(compare_df: pd.DataFrame, baseline_id: str) -> dict[str, Any]:
    if compare_df.empty:
        return {
            "best_candidate": str(baseline_id),
            "worth_keeping": False,
            "worth_promoting": False,
        }
    ranked = compare_df.copy()
    ranked["mdd_abs"] = pd.to_numeric(ranked["net_max_drawdown"], errors="coerce").abs()
    ranked = ranked.sort_values(["mdd_abs", "net_total_return", "net_sharpe"], ascending=[True, False, False])
    best_row = ranked.iloc[0].to_dict()
    best_id = str(best_row.get("candidate_id", baseline_id))
    baseline_row = compare_df.loc[compare_df["candidate_id"] == baseline_id].iloc[0].to_dict()
    baseline_mdd_abs = abs(_safe_float(baseline_row.get("net_max_drawdown")))
    best_mdd_abs = abs(_safe_float(best_row.get("net_max_drawdown")))
    baseline_total = _safe_float(baseline_row.get("net_total_return"))
    best_total = _safe_float(best_row.get("net_total_return"))
    mdd_improvement = baseline_mdd_abs - best_mdd_abs
    total_return_gap = best_total - baseline_total
    worth_keeping = bool(best_id != baseline_id and mdd_improvement >= 0.03 and total_return_gap >= -0.20)
    worth_promoting = bool(best_id != baseline_id and mdd_improvement >= 0.05 and total_return_gap >= -0.10)
    return {
        "best_candidate": best_id,
        "best_row": best_row,
        "mdd_improvement_abs": mdd_improvement,
        "total_return_gap": total_return_gap,
        "worth_keeping": worth_keeping,
        "worth_promoting": worth_promoting,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Ataca o drawdown do campeão atual com freio antecipado, cap dinâmico e postura gradual.")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--capital-brl", type=float, default=10000.0)
    ap.add_argument("--outdir-root", default="results/validation/profit_champion_drawdown_suite")
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
    attack_alloc = built["allocations"]["attack"]
    protect_alloc = built["allocations"]["baseline_guard"]

    structure_daily, _spectral_panel, criticality, structural_stress = _build_structure_layers(context)
    base_score = _clip01(context["attack_score_exogenous"])
    market_mode_share_pct = _align(structure_daily.get("market_mode_share_pct"), base_score.index, default=0.5)
    criticality_aligned = _align(criticality, base_score.index, default=0.5)
    structural_stress_aligned = _align(structural_stress, base_score.index, default=0.5)

    champion_bundle, champion_score, champion_weight = _build_criticality_free_energy_bundle(
        candidate_id="criticality_free_energy_attack",
        notes="campeão atual: criticidade com reorganização leve",
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        base_score=base_score,
        structure_daily=structure_daily,
        criticality=criticality_aligned,
    )

    early_scale = _build_early_stress_scale(
        criticality=criticality_aligned,
        structural_stress=structural_stress_aligned,
        market_mode_share_pct=market_mode_share_pct,
    )
    regime_cap = _build_regime_cap(
        criticality=criticality_aligned,
        structural_stress=structural_stress_aligned,
        market_mode_share_pct=market_mode_share_pct,
    )
    gradual_cap = _build_gradual_posture_cap(
        base_score=champion_score,
        criticality=criticality_aligned,
        structural_stress=structural_stress_aligned,
    )

    early_weight = (champion_weight * early_scale).clip(0.0, 1.0)
    early_bundle = _blend_allocation_bundles(
        candidate_id="champion_early_stress_throttle",
        notes="reduz exposicao mais cedo quando criticidade, stress e market mode sobem juntos",
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        attack_weight=early_weight,
    )

    regime_weight = np.minimum(champion_weight.to_numpy(dtype=float), regime_cap.to_numpy(dtype=float))
    regime_weight = pd.Series(regime_weight, index=champion_weight.index, dtype=float).clip(0.0, 1.0)
    regime_bundle = _blend_allocation_bundles(
        candidate_id="champion_dynamic_regime_cap",
        notes="coloca teto dinamico no ataque quando a estrutura entra em transicao ou stress",
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        attack_weight=regime_weight,
    )

    gradual_weight = np.minimum(champion_weight.to_numpy(dtype=float), gradual_cap.to_numpy(dtype=float))
    gradual_weight = pd.Series(gradual_weight, index=champion_weight.index, dtype=float).clip(0.0, 1.0)
    gradual_bundle = _blend_allocation_bundles(
        candidate_id="champion_gradual_posture_cap",
        notes="usa uma postura mais gradual entre ataque e quase caixa com caps progressivos de exposicao",
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        attack_weight=gradual_weight,
    )

    combo_weight = (champion_weight * early_scale).clip(0.0, 1.0)
    combo_weight = pd.Series(
        np.minimum(combo_weight.to_numpy(dtype=float), regime_cap.to_numpy(dtype=float)),
        index=combo_weight.index,
        dtype=float,
    )
    combo_weight = pd.Series(
        np.minimum(combo_weight.to_numpy(dtype=float), gradual_cap.to_numpy(dtype=float)),
        index=combo_weight.index,
        dtype=float,
    ).clip(0.0, 1.0)
    combo_bundle = _blend_allocation_bundles(
        candidate_id="champion_drawdown_combo",
        notes="combina freio antecipado, cap dinamico por regime e postura gradual antes de quase caixa",
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        attack_weight=combo_weight,
    )

    results = {
        champion_bundle.bundle.result.candidate_id: champion_bundle.bundle.result,
        early_bundle.bundle.result.candidate_id: early_bundle.bundle.result,
        regime_bundle.bundle.result.candidate_id: regime_bundle.bundle.result,
        gradual_bundle.bundle.result.candidate_id: gradual_bundle.bundle.result,
        combo_bundle.bundle.result.candidate_id: combo_bundle.bundle.result,
    }

    compare_rows = [
        _result_row(result, baseline=champion_bundle.bundle.result, family="champion_drawdown", label=result.candidate_id)
        for result in results.values()
    ]
    compare_df = pd.DataFrame(compare_rows).sort_values(by=["net_max_drawdown", "net_total_return"], ascending=[False, False])
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

    decision = _summarize_keep_decision(compare_df, champion_bundle.bundle.result.candidate_id)
    research_rows = [
        _research_row(champion_bundle.bundle.result, outdir=outdir, status="keep", methodology="criticality_plus_free_energy", label="Campeão atual"),
        _research_row(early_bundle.bundle.result, outdir=outdir, status="watch", methodology="champion_early_stress_throttle", label="Campeão com freio antecipado por stress"),
        _research_row(regime_bundle.bundle.result, outdir=outdir, status="watch", methodology="champion_dynamic_regime_cap", label="Campeão com teto dinâmico por regime"),
        _research_row(gradual_bundle.bundle.result, outdir=outdir, status="watch", methodology="champion_gradual_posture_cap", label="Campeão com postura gradual"),
        _research_row(combo_bundle.bundle.result, outdir=outdir, status="watch", methodology="champion_drawdown_combo", label="Campeão com pacote completo de drawdown"),
    ]
    _write_json(outdir / "profit_research_rows.json", research_rows)

    summary = {
        "suite": "profit_champion_drawdown_suite",
        "baseline_candidate": champion_bundle.bundle.result.candidate_id,
        "best_drawdown_candidate": decision["best_candidate"],
        "worth_keeping": bool(decision["worth_keeping"]),
        "worth_promoting": bool(decision["worth_promoting"]),
        "baseline_total_return": _safe_float(champion_bundle.bundle.result.net_total_return),
        "baseline_ann_return": _safe_float(champion_bundle.bundle.result.net_ann_return),
        "baseline_sharpe": _safe_float(champion_bundle.bundle.result.net_sharpe),
        "baseline_mdd": _safe_float(champion_bundle.bundle.result.net_max_drawdown),
        "best_total_return": _safe_float((decision.get("best_row") or {}).get("net_total_return")),
        "best_ann_return": _safe_float((decision.get("best_row") or {}).get("net_ann_return")),
        "best_sharpe": _safe_float((decision.get("best_row") or {}).get("net_sharpe")),
        "best_mdd": _safe_float((decision.get("best_row") or {}).get("net_max_drawdown")),
        "mdd_improvement_abs": _safe_float(decision.get("mdd_improvement_abs")),
        "total_return_gap": _safe_float(decision.get("total_return_gap")),
    }
    _write_json(outdir / "summary.json", summary)
    write_run_manifest(
        outdir,
        script="scripts/bench/validation/run_profit_champion_drawdown_suite.py",
        params=vars(args),
        extra={
            "suite": "profit_champion_drawdown_suite",
            "best_drawdown_candidate": decision["best_candidate"],
            "worth_keeping": bool(decision["worth_keeping"]),
            "worth_promoting": bool(decision["worth_promoting"]),
        },
    )


if __name__ == "__main__":
    main()
