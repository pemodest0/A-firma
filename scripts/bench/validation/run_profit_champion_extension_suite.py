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

from engine.portfolio import apply_free_energy_penalty  # noqa: E402
from engine.portfolio.exogenous_features import adjust_confidence_with_feature  # noqa: E402
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
from scripts.bench.validation.run_profit_investment_yearbook import _calendar_rows  # noqa: E402
from scripts.bench.validation.run_profit_marketmode_criticality_suite import (  # noqa: E402
    _build_structure_layers,
    _rolling_percentile,
)
from scripts.bench.validation.run_profit_sector_pressure_suite import _research_row  # noqa: E402
from scripts.bench.validation.run_profit_u800_alpha_suite import (  # noqa: E402
    _build_u800_equity_candidates,
    _crypto_bundle,
)


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _build_criticality_free_energy_bundle(
    *,
    candidate_id: str,
    notes: str,
    attack_alloc: AllocationBundle,
    protect_alloc: AllocationBundle,
    base_score: pd.Series,
    structure_daily: pd.DataFrame,
    criticality: pd.Series,
) -> tuple[AllocationBundle, pd.Series, pd.Series]:
    base_score = pd.to_numeric(base_score, errors="coerce").fillna(0.0).clip(0.0, 1.0).astype(float)
    criticality_aligned = pd.to_numeric(criticality, errors="coerce").reindex(base_score.index).fillna(0.5).clip(0.0, 1.0)
    criticality_pct = _rolling_percentile(criticality_aligned, 126).fillna(0.5)
    market_pct = pd.to_numeric(structure_daily.get("market_mode_share_pct"), errors="coerce").reindex(base_score.index).fillna(0.5)
    rel_penalty = (
        0.22 * ((criticality_pct - 0.55).clip(lower=0.0) / 0.45)
        + 0.06 * ((market_pct - 0.70).clip(lower=0.0) / 0.30)
    ).clip(0.0, 0.35)
    criticality_rel_score = (base_score - rel_penalty).clip(0.0, 1.0)
    instability = (0.60 * criticality_aligned + 0.40 * market_pct).clip(0.0, 1.0)
    free_rel_score = apply_free_energy_penalty(
        base_score=criticality_rel_score,
        turnover=attack_alloc.bundle.result.turnover,
        instability=instability,
        gamma=0.06,
        eta=0.08,
    )
    weight = _confidence_weight_from_score(free_rel_score)
    bundle = _blend_allocation_bundles(
        candidate_id=str(candidate_id),
        notes=str(notes),
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        attack_weight=weight,
    )
    return bundle, free_rel_score, weight


def _fragility_decile_scale(liquidation: pd.Series, *, window: int = 126) -> pd.Series:
    liq = pd.to_numeric(liquidation, errors="coerce").fillna(0.0).clip(0.0, 1.0).astype(float)
    if int(window) < 10:
        min_periods = max(3, int(window) // 2)

        def _pct(arr: np.ndarray) -> float:
            arr = arr[np.isfinite(arr)]
            if arr.size <= 1:
                return float("nan")
            return float(np.mean(arr <= float(arr[-1])))

        pct = liq.rolling(int(window), min_periods=min_periods).apply(_pct, raw=True).fillna(0.5)
    else:
        pct = _rolling_percentile(liq, int(window)).fillna(0.5)
    scale = pd.Series(1.0, index=liq.index, dtype=float)
    scale.loc[pct >= 0.90] = 0.80
    scale.loc[pct >= 0.97] = 0.55
    return scale


def _profit_lock_scale(net_ret: pd.Series) -> pd.Series:
    idx = net_ret.index
    scale = pd.Series(1.0, index=idx, dtype=float)
    values = pd.to_numeric(net_ret, errors="coerce").fillna(0.0).astype(float)
    lagged = values.shift(1).fillna(0.0)
    for dt in idx:
        history = lagged.loc[:dt]
        month_start = dt.replace(day=1)
        year_start = dt.replace(month=1, day=1)
        month_slice = history.loc[month_start:]
        ytd_slice = history.loc[year_start:]
        month_ret = float((1.0 + month_slice).prod() - 1.0) if not month_slice.empty else 0.0
        ytd_ret = float((1.0 + ytd_slice).prod() - 1.0) if not ytd_slice.empty else 0.0
        month_scale = 1.0
        ytd_scale = 1.0
        if month_ret >= 0.15:
            month_scale = 0.88
        if month_ret >= 0.30:
            month_scale = 0.72
        if ytd_ret >= 0.60:
            ytd_scale = 0.82
        if ytd_ret >= 1.20:
            ytd_scale = 0.62
        scale.loc[dt] = min(month_scale, ytd_scale)
    return scale


def main() -> None:
    ap = argparse.ArgumentParser(description="Testa extensoes do campeao atual: apoio do universo 800, reducao so no pior decil de fragilidade e trava parcial de lucro.")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--capital-brl", type=float, default=10000.0)
    ap.add_argument("--outdir-root", default="results/validation/profit_champion_extension_suite")
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

    structure_daily, _spectral_panel, criticality, _structural_stress = _build_structure_layers(context)

    champion_bundle, champion_score, champion_weight = _build_criticality_free_energy_bundle(
        candidate_id="criticality_free_energy_attack",
        notes="campeao atual: criticidade com reorganizacao leve",
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        base_score=context["attack_score_exogenous"],
        structure_daily=structure_daily,
        criticality=criticality,
    )

    crypto_bundle = _crypto_bundle(
        candidate_id="major8_fast_support",
        allowed_tickers=list(context["crypto_tiers"]["crypto_major8"]),
        context=context,
    )
    u800_equities = _build_u800_equity_candidates(
        prices_dir=(ROOT / args.prices_dir).resolve(),
        asset_groups=(ROOT / args.equity_asset_groups).resolve(),
        asset_metadata=(ROOT / args.equity_asset_metadata).resolve(),
        benchmark_ticker=str(args.benchmark_equity),
        profile=context["profiles"]["foreign"],
    )
    u800_equity_bundle = u800_equities["a2r1"]
    raw_u800_attack = _build_alpha_meta_allocation_bundle(
        candidate_id="champion_u800_support_raw",
        crypto_bundle=crypto_bundle,
        equity_bundle=u800_equity_bundle,
        btc_prices=context["btc_prices"],
        spy_prices=context["spy_prices"],
        profile=context["profiles"]["blended"],
        entry_lookback=14,
        exit_lookback=63,
        entry_margin=0.02,
        exit_margin=0.05,
        risk_off_mode="equity25",
        min_crypto_hold_days=0,
    )
    u800_base_score = _build_promoted_attack_confidence_score(
        {
            "btc_prices": context["btc_prices"],
            "spy_prices": context["spy_prices"],
            "regime_series": context["regime_series"],
        },
        pd.concat(
            {
                "crypto": pd.to_numeric(crypto_bundle.result.gross_ret, errors="coerce"),
                "equity": pd.to_numeric(u800_equity_bundle.result.gross_ret, errors="coerce"),
            },
            axis=1,
            sort=False,
        ).dropna(how="all"),
    )
    u800_base_score = adjust_confidence_with_feature(
        base_score=u800_base_score,
        feature=context["exogenous_panel"].get("liquidation"),
        mode="penalty",
        weight=0.14,
    )
    u800_bundle, u800_score, u800_weight = _build_criticality_free_energy_bundle(
        candidate_id="champion_u800_support",
        notes="troca o bloco nao cripto pelo melhor apoio atual do universo 800",
        attack_alloc=raw_u800_attack,
        protect_alloc=protect_alloc,
        base_score=u800_base_score,
        structure_daily=structure_daily,
        criticality=criticality,
    )

    liquidation = pd.to_numeric(context["exogenous_panel"].get("liquidation"), errors="coerce").reindex(champion_weight.index).fillna(0.0)
    fragility_weight = (champion_weight * _fragility_decile_scale(liquidation)).clip(0.0, 1.0)
    fragility_bundle = _blend_allocation_bundles(
        candidate_id="champion_fragility_decile",
        notes="reduz o tamanho so quando a fragilidade do cripto entra no pior decil recente",
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        attack_weight=fragility_weight,
    )

    profit_lock_weight = (champion_weight * _profit_lock_scale(champion_bundle.bundle.result.net_ret)).clip(0.0, 1.0)
    profit_lock_bundle = _blend_allocation_bundles(
        candidate_id="champion_profit_lock_partial",
        notes="trava parcial de lucro depois de meses e anos muito fortes, usando apenas historico ja realizado",
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        attack_weight=profit_lock_weight,
    )

    results = {
        champion_bundle.bundle.result.candidate_id: champion_bundle.bundle.result,
        u800_bundle.bundle.result.candidate_id: u800_bundle.bundle.result,
        fragility_bundle.bundle.result.candidate_id: fragility_bundle.bundle.result,
        profit_lock_bundle.bundle.result.candidate_id: profit_lock_bundle.bundle.result,
    }

    compare_rows = [
        _result_row(result, baseline=champion_bundle.bundle.result, family="champion_extension", label=result.candidate_id)
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

    best_id = str(compare_df.iloc[0]["candidate_id"]) if not compare_df.empty else champion_bundle.bundle.result.candidate_id
    best_row = compare_df.iloc[0].to_dict() if not compare_df.empty else {}
    worth_promoting = bool(best_id != champion_bundle.bundle.result.candidate_id and _safe_float(best_row.get("net_total_return", 0.0)) > _safe_float(champion_bundle.bundle.result.net_total_return))

    research_rows = [
        _research_row(champion_bundle.bundle.result, outdir=outdir, status="keep", methodology="criticality_plus_free_energy", label="Campeao atual"),
        _research_row(u800_bundle.bundle.result, outdir=outdir, status="watch", methodology="champion_u800_support", label="Campeao com apoio das acoes do universo 800"),
        _research_row(fragility_bundle.bundle.result, outdir=outdir, status="watch", methodology="champion_fragility_decile", label="Campeao com redutor so no pior decil de fragilidade"),
        _research_row(profit_lock_bundle.bundle.result, outdir=outdir, status="watch", methodology="champion_profit_lock_partial", label="Campeao com trava parcial de lucro"),
    ]
    _write_json(outdir / "profit_research_rows.json", research_rows)

    summary = {
        "suite": "profit_champion_extension_suite",
        "baseline_candidate": champion_bundle.bundle.result.candidate_id,
        "best_candidate": best_id,
        "worth_promoting": bool(worth_promoting),
        "baseline_total_return": _safe_float(champion_bundle.bundle.result.net_total_return),
        "best_total_return": _safe_float(best_row.get("net_total_return")),
        "baseline_ann_return": _safe_float(champion_bundle.bundle.result.net_ann_return),
        "best_ann_return": _safe_float(best_row.get("net_ann_return")),
        "baseline_sharpe": _safe_float(champion_bundle.bundle.result.net_sharpe),
        "best_sharpe": _safe_float(best_row.get("net_sharpe")),
        "baseline_mdd": _safe_float(champion_bundle.bundle.result.net_max_drawdown),
        "best_mdd": _safe_float(best_row.get("net_max_drawdown")),
    }
    _write_json(outdir / "summary.json", summary)
    write_run_manifest(
        outdir,
        script="scripts/bench/validation/run_profit_champion_extension_suite.py",
        params=vars(args),
        extra={
            "suite": "profit_champion_extension_suite",
            "best_candidate": best_id,
            "worth_promoting": bool(worth_promoting),
        },
    )


if __name__ == "__main__":
    main()
