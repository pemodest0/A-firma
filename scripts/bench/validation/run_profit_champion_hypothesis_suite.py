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
    PeriodLossGuardConfig,
    apply_free_energy_penalty,
    build_attractor_persistence_score,
    build_direction_gradient_score,
    quarterly_loss_guard,
    monthly_loss_guard,
    combine_guard_actions,
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
from scripts.bench.validation.run_profit_marketmode_criticality_suite import (  # noqa: E402
    _build_structure_layers,
    _rolling_percentile,
)
from scripts.bench.validation.run_profit_sector_pressure_suite import _research_row  # noqa: E402


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _period_guard_scale(net_ret: pd.Series, config: PeriodLossGuardConfig) -> pd.Series:
    idx = net_ret.index
    scale = pd.Series(1.0, index=idx, dtype=float)
    values = pd.to_numeric(net_ret, errors="coerce").fillna(0.0).astype(float)
    lagged = values.shift(1).fillna(0.0)
    for dt in idx:
        history = lagged.loc[:dt]
        month_start = dt.replace(day=1)
        quarter_month = 3 * ((int(dt.month) - 1) // 3) + 1
        quarter_start = dt.replace(month=quarter_month, day=1)
        month_ret = float((1.0 + history.loc[month_start:]).prod() - 1.0) if not history.loc[month_start:].empty else 0.0
        quarter_ret = float((1.0 + history.loc[quarter_start:]).prod() - 1.0) if not history.loc[quarter_start:].empty else 0.0
        action = combine_guard_actions(
            monthly_loss_guard(month_ret, config),
            quarterly_loss_guard(quarter_ret, config),
        )
        if action == "REDUCED_ATTACK":
            scale.loc[dt] = 0.80
        elif action == "PROTECTED":
            scale.loc[dt] = 0.45
        elif action == "CASH_HEAVY":
            scale.loc[dt] = 0.15
    return scale


def _build_champion_weights(
    context: dict[str, Any],
    attack_alloc: AllocationBundle,
) -> tuple[pd.Series, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    base_score = pd.to_numeric(context["attack_score_exogenous"], errors="coerce").fillna(0.0).clip(0.0, 1.0).astype(float)
    structure_daily, _spectral_panel, criticality, structural_stress = _build_structure_layers(context)
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
    instability = (
        0.60 * pd.to_numeric(criticality, errors="coerce").reindex(base_score.index).fillna(0.5)
        + 0.40 * pd.to_numeric(structure_daily.get("market_mode_share_pct"), errors="coerce").reindex(base_score.index).fillna(0.5)
    ).clip(0.0, 1.0)
    free_rel_score = apply_free_energy_penalty(
        base_score=criticality_rel_score,
        turnover=attack_alloc.bundle.result.turnover,
        instability=instability,
        gamma=0.06,
        eta=0.08,
    )
    champion_weight = _confidence_weight_from_score(free_rel_score)
    return champion_weight, structure_daily, criticality, direction_score, persistence_score, free_rel_score


def main() -> None:
    ap = argparse.ArgumentParser(description="Ataca o campeao atual com hipoteses leves de direcao, persistencia e travas curtas.")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--capital-brl", type=float, default=10000.0)
    ap.add_argument("--outdir-root", default="results/validation/profit_champion_hypothesis_suite")
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

    champion_weight, structure_daily, criticality, direction_score, persistence_score, free_rel_score = _build_champion_weights(
        context,
        attack_alloc,
    )
    champion_bundle = _blend_allocation_bundles(
        candidate_id="criticality_free_energy_attack",
        notes="campeao atual: criticidade com reorganizacao leve",
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        attack_weight=champion_weight,
    )

    direction_score_light = (free_rel_score + 0.035 * (direction_score - 0.5)).clip(0.0, 1.0)
    direction_weight = _confidence_weight_from_score(direction_score_light)
    direction_bundle = _blend_allocation_bundles(
        candidate_id="champion_direction_light",
        notes="campeao atual com leve reforco pela direcao do estado",
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        attack_weight=direction_weight,
    )

    persistence_score_light = (free_rel_score + 0.030 * (persistence_score - 0.5)).clip(0.0, 1.0)
    persistence_weight = _confidence_weight_from_score(persistence_score_light)
    persistence_bundle = _blend_allocation_bundles(
        candidate_id="champion_persistence_light",
        notes="campeao atual com reforco leve quando o estado favoravel parece persistente",
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        attack_weight=persistence_weight,
    )

    combo_score = (
        free_rel_score
        + 0.040 * (direction_score - 0.5)
        + 0.030 * (persistence_score - 0.5)
    ).clip(0.0, 1.0)
    combo_weight = _confidence_weight_from_score(combo_score)
    weak_state = ((pd.to_numeric(direction_score, errors="coerce") < 0.46) & (pd.to_numeric(persistence_score, errors="coerce") < 0.46)).reindex(combo_weight.index).fillna(False)
    combo_weight = combo_weight.where(~weak_state, combo_weight * 0.80).clip(0.0, 1.0)
    combo_bundle = _blend_allocation_bundles(
        candidate_id="champion_direction_persistence",
        notes="campeao atual com direcao e persistencia, reduzindo um pouco quando ambos enfraquecem",
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        attack_weight=combo_weight,
    )

    light_guard_cfg = PeriodLossGuardConfig(
        monthly_reduce_threshold=-0.045,
        monthly_protect_threshold=-0.075,
        monthly_cash_threshold=-0.11,
        quarterly_reduce_threshold=-0.07,
        quarterly_protect_threshold=-0.11,
        quarterly_cash_threshold=-0.17,
    )
    guard_scale = _period_guard_scale(champion_bundle.bundle.result.net_ret, light_guard_cfg)
    guard_weight = (champion_weight * guard_scale).clip(0.0, 1.0)
    guard_bundle = _blend_allocation_bundles(
        candidate_id="champion_light_period_guard",
        notes="campeao atual com trava leve por mes e trimestre",
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        attack_weight=guard_weight,
    )

    combo_guard_weight = (combo_weight * guard_scale).clip(0.0, 1.0)
    combo_guard_bundle = _blend_allocation_bundles(
        candidate_id="champion_direction_persistence_guard",
        notes="campeao atual com direcao, persistencia e trava leve por periodo",
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        attack_weight=combo_guard_weight,
    )

    results = {
        baseline_result.candidate_id: baseline_result,
        protected_result.candidate_id: protected_result,
        champion_bundle.bundle.result.candidate_id: champion_bundle.bundle.result,
        direction_bundle.bundle.result.candidate_id: direction_bundle.bundle.result,
        persistence_bundle.bundle.result.candidate_id: persistence_bundle.bundle.result,
        combo_bundle.bundle.result.candidate_id: combo_bundle.bundle.result,
        guard_bundle.bundle.result.candidate_id: guard_bundle.bundle.result,
        combo_guard_bundle.bundle.result.candidate_id: combo_guard_bundle.bundle.result,
    }

    compare_rows = [
        _result_row(result, baseline=champion_bundle.bundle.result, family="champion_hypothesis", label=result.candidate_id)
        for result in results.values()
    ]
    compare_df = pd.DataFrame(compare_rows).sort_values(by=["net_total_return", "net_sharpe"], ascending=[False, False])
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

    research_rows = [
        _research_row(champion_bundle.bundle.result, outdir=outdir, status="keep", methodology="criticality_plus_free_energy", label="Campeao atual"),
        _research_row(direction_bundle.bundle.result, outdir=outdir, status="watch", methodology="champion_direction_light", label="Campeao + direcao leve"),
        _research_row(persistence_bundle.bundle.result, outdir=outdir, status="watch", methodology="champion_persistence_light", label="Campeao + persistencia leve"),
        _research_row(combo_bundle.bundle.result, outdir=outdir, status="watch", methodology="champion_direction_persistence", label="Campeao + direcao + persistencia"),
        _research_row(guard_bundle.bundle.result, outdir=outdir, status="watch", methodology="champion_light_period_guard", label="Campeao + trava leve por periodo"),
        _research_row(combo_guard_bundle.bundle.result, outdir=outdir, status="watch", methodology="champion_direction_persistence_guard", label="Campeao + direcao + persistencia + trava"),
    ]
    (outdir / "profit_research_rows.json").write_text(json_dumps(research_rows), encoding="utf-8")

    best = compare_df.iloc[0].to_dict() if not compare_df.empty else {}
    best_non_baseline = compare_df[compare_df["candidate_id"] != champion_bundle.bundle.result.candidate_id]
    best_challenger = best_non_baseline.iloc[0].to_dict() if not best_non_baseline.empty else {}
    summary = {
        "run_id": outdir.name,
        "baseline_candidate": champion_bundle.bundle.result.candidate_id,
        "best_candidate": best.get("candidate_id"),
        "best_challenger": best_challenger.get("candidate_id"),
        "worth_promoting": bool(best_challenger.get("net_total_return", float("-inf")) > _safe_float(champion_bundle.bundle.result.net_total_return)),
        "baseline_net_ann_return": _safe_float(champion_bundle.bundle.result.net_ann_return),
        "baseline_net_total_return": _safe_float(champion_bundle.bundle.result.net_total_return),
        "baseline_net_sharpe": _safe_float(champion_bundle.bundle.result.net_sharpe),
        "baseline_net_max_drawdown": _safe_float(champion_bundle.bundle.result.net_max_drawdown),
        "best_challenger_net_ann_return": _safe_float(best_challenger.get("net_ann_return")),
        "best_challenger_net_total_return": _safe_float(best_challenger.get("net_total_return")),
        "best_challenger_net_sharpe": _safe_float(best_challenger.get("net_sharpe")),
        "best_challenger_net_max_drawdown": _safe_float(best_challenger.get("net_max_drawdown")),
        "compare_file": str(outdir / "candidate_compare.csv"),
        "yearbook_file": str(outdir / "yearbook_reais.csv"),
        "year_improvement_file": str(outdir / "year_improvement.csv"),
    }
    _write_json(outdir / "summary.json", summary)
    write_run_manifest(
        outdir,
        script="scripts/bench/validation/run_profit_champion_hypothesis_suite.py",
        params={
            "benchmark_crypto": args.benchmark_crypto,
            "benchmark_equity": args.benchmark_equity,
            "capital_brl": args.capital_brl,
        },
        paths={
            "summary_json": str(outdir / "summary.json"),
            "candidate_compare_csv": str(outdir / "candidate_compare.csv"),
            "yearbook_reais_csv": str(outdir / "yearbook_reais.csv"),
            "year_improvement_csv": str(outdir / "year_improvement.csv"),
        },
        extra={
            "suite": "profit_champion_hypothesis_suite",
            "best_candidate": best.get("candidate_id"),
            "best_challenger": best_challenger.get("candidate_id"),
        },
    )


def json_dumps(payload: Any) -> str:
    import json

    return json.dumps(payload, ensure_ascii=True, indent=2, default=str)


if __name__ == "__main__":
    main()
