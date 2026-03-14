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

from engine.portfolio.exogenous_features import adjust_confidence_with_feature  # noqa: E402
from engine.structural.run_manifest import write_run_manifest  # noqa: E402
from scripts.bench.validation.run_profit_alpha_hardening_suite import (  # noqa: E402
    AllocationBundle,
    _blend_allocation_bundles,
    _build_alpha_meta_allocation_bundle,
    _build_candidates,
    _build_promoted_attack_confidence_score,
)
from scripts.bench.validation.run_profit_alpha_improvement_suite import _safe_float, _write_json  # noqa: E402
from scripts.bench.validation.run_profit_attack_entry_ranking_suite import _result_row  # noqa: E402
from scripts.bench.validation.run_profit_champion_drawdown_suite import _summarize_keep_decision  # noqa: E402
from scripts.bench.validation.run_profit_champion_extension_suite import (  # noqa: E402
    _build_criticality_free_energy_bundle,
)
from scripts.bench.validation.run_profit_equity_improvement_suite import (  # noqa: E402
    _equity_trailing_switch_bundle,
    _load_equity_universe,
)
from scripts.bench.validation.run_profit_frontier_expansion_suite import _simulate_asset_rule  # noqa: E402
from scripts.bench.validation.run_profit_investment_yearbook import _calendar_rows  # noqa: E402
from scripts.bench.validation.run_profit_layered_engine_suite import (  # noqa: E402
    StrategyBundle,
    _simulate_equity_group_sleeve_v2,
    _simulate_equity_group_sleeve_v3,
)
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


def _build_offensive_share(
    *,
    base_score: pd.Series,
    criticality: pd.Series,
    structural_stress: pd.Series,
    market_mode_share_pct: pd.Series,
) -> pd.Series:
    idx = base_score.index
    score = _clip01(base_score)
    crit = _clip01(criticality).reindex(idx).fillna(0.5)
    stress = _clip01(structural_stress).reindex(idx).fillna(0.5)
    market = _clip01(market_mode_share_pct).reindex(idx).fillna(0.5)
    crit_pct = _rolling_percentile(crit, 126).reindex(idx).fillna(0.5)
    stress_pct = _rolling_percentile(stress, 126).reindex(idx).fillna(0.5)
    danger = (
        0.35 * crit_pct
        + 0.30 * stress_pct
        + 0.20 * market
        + 0.15 * (1.0 - score)
    ).clip(0.0, 1.0)
    share = 1.0 - 0.90 * ((danger - 0.56).clip(lower=0.0) / 0.34).clip(0.0, 1.0)
    transition_mask = (crit >= 0.58) | (stress >= 0.60) | (market >= 0.70)
    hard_mask = (crit >= 0.67) | (stress >= 0.70) | (market >= 0.79)
    panic_mask = (crit >= 0.76) | (stress >= 0.79) | (market >= 0.88)
    share.loc[transition_mask] = np.minimum(share.loc[transition_mask], 0.72)
    share.loc[hard_mask] = np.minimum(share.loc[hard_mask], 0.38)
    share.loc[panic_mask] = 0.0
    # Apply one-day lag so the rotation only reacts after the stressed close is known.
    return share.shift(1).ffill().fillna(1.0).clip(0.0, 1.0)


def _blend_scores(
    *,
    offensive_score: pd.Series,
    defensive_score: pd.Series,
    offensive_share: pd.Series,
) -> pd.Series:
    idx = offensive_score.index.intersection(defensive_score.index).intersection(offensive_share.index)
    offense = pd.to_numeric(offensive_score.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    defense = pd.to_numeric(defensive_score.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    share = pd.to_numeric(offensive_share.reindex(idx), errors="coerce").fillna(1.0).astype(float).clip(0.0, 1.0)
    return (share * offense + (1.0 - share) * defense).clip(0.0, 1.0)


def _build_crypto_bundle(
    *,
    candidate_id: str,
    context: dict[str, Any],
    score_mode: str,
    top_k: int,
) -> StrategyBundle:
    result = _simulate_asset_rule(
        candidate_id=candidate_id,
        family="champion_selection_rotation_crypto",
        allowed_tickers=list(context["crypto_tiers"]["crypto_major8"]),
        returns=context["crypto_returns"],
        prices=context["crypto_prices"],
        asset_table=context["crypto_assets"],
        benchmark_ticker=str(context["benchmark_crypto"]),
        fallback_ticker=str(context["benchmark_crypto"]),
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
    if result is None:
        raise SystemExit(f"falha ao simular sleeve cripto {candidate_id}")
    benchmark = (
        pd.to_numeric(context["crypto_returns"][str(context["benchmark_crypto"])], errors="coerce")
        .reindex(result.gross_ret.index)
        .fillna(0.0)
        .astype(float)
    )
    return StrategyBundle(
        result=result,
        benchmark_gross_ret=benchmark,
        profile=context["profiles"]["crypto"],
        benchmark_profile=context["profiles"]["crypto"],
    )


def _build_raw_attack_alloc_and_score(
    *,
    candidate_id: str,
    crypto_bundle: StrategyBundle,
    equity_bundle: StrategyBundle,
    context: dict[str, Any],
) -> tuple[AllocationBundle, pd.Series]:
    raw_attack = _build_alpha_meta_allocation_bundle(
        candidate_id=f"{candidate_id}__raw",
        crypto_bundle=crypto_bundle,
        equity_bundle=equity_bundle,
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
    sleeve_returns = (
        pd.concat(
            {
                "crypto": pd.to_numeric(crypto_bundle.result.gross_ret, errors="coerce"),
                "equity": pd.to_numeric(equity_bundle.result.gross_ret, errors="coerce"),
            },
            axis=1,
            sort=False,
        )
        .dropna(how="all")
    )
    score = _build_promoted_attack_confidence_score(
        {
            "btc_prices": context["btc_prices"],
            "spy_prices": context["spy_prices"],
            "regime_series": context["regime_series"],
        },
        sleeve_returns,
    )
    score = adjust_confidence_with_feature(
        base_score=score,
        feature=context["exogenous_panel"].get("liquidation"),
        mode="penalty",
        weight=0.14,
    )
    return raw_attack, _clip01(score)


def _build_u800_equity_support(
    *,
    prices_dir: Path,
    asset_groups: Path,
    asset_metadata: Path,
    benchmark_ticker: str,
    regime_series: pd.Series,
    profile: Any,
) -> StrategyBundle:
    asset_table, returns, prices, group_map = _load_equity_universe(
        prices_dir=prices_dir,
        asset_groups=asset_groups,
        asset_metadata=asset_metadata,
        benchmark_ticker=benchmark_ticker,
    )
    spy_prices = pd.to_numeric(prices[str(benchmark_ticker)], errors="coerce")
    eq_a2 = _simulate_equity_group_sleeve_v2(
        candidate_id="selection_rotation_u800_v2_a2",
        returns=returns,
        prices=prices,
        asset_table=asset_table,
        equity_groups=group_map,
        benchmark_ticker=benchmark_ticker,
        group_lookback_fast=63,
        group_lookback_slow=189,
        group_top_k=4,
        assets_per_group=1,
        asset_lookback=126,
        asset_ma_days=200,
        market_ma_days=200,
        profile=profile,
        benchmark_profile=profile,
    )
    eq_r1 = _simulate_equity_group_sleeve_v3(
        candidate_id="selection_rotation_u800_v3_r1",
        returns=returns,
        prices=prices,
        asset_table=asset_table,
        equity_groups=group_map,
        benchmark_ticker=benchmark_ticker,
        group_lookback_fast=63,
        group_lookback_slow=189,
        group_top_k=3,
        assets_per_group=2,
        asset_lookback=126,
        asset_ma_days=200,
        market_ma_days=200,
        min_group_breadth=0.35,
        max_group_weight=0.40,
        profile=profile,
        benchmark_profile=profile,
    )
    if eq_a2 is None or eq_r1 is None:
        raise SystemExit("falha ao reconstruir apoio de equities do universo 800")
    return _equity_trailing_switch_bundle(
        candidate_id="selection_rotation_u800_support_a2r1",
        aggressive_bundle=eq_a2,
        robust_bundle=eq_r1,
        regime_series=regime_series,
        spy_prices=spy_prices,
        mode="trail_switch",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Ataca drawdown do campeão trocando seleção de ativos e sleeves em regime ruim, sem dado futuro.")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--capital-brl", type=float, default=10000.0)
    ap.add_argument("--outdir-root", default="results/validation/profit_champion_selection_rotation_suite")
    args = ap.parse_args()

    outdir = (ROOT / args.outdir_root / _run_id()).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[selection_rotation] outdir={outdir}", flush=True)

    built = _build_candidates(
        prices_dir=(ROOT / args.prices_dir).resolve(),
        crypto_groups=(ROOT / args.crypto_asset_groups).resolve(),
        crypto_meta=(ROOT / args.crypto_asset_metadata).resolve(),
        equity_groups=(ROOT / args.equity_asset_groups).resolve(),
        equity_meta=(ROOT / args.equity_asset_metadata).resolve(),
        benchmark_crypto=str(args.benchmark_crypto),
        benchmark_equity=str(args.benchmark_equity),
    )
    print("[selection_rotation] base candidates ready", flush=True)
    context = dict(built["context"])
    attack_alloc: AllocationBundle = built["allocations"]["attack"]
    protect_alloc: AllocationBundle = built["allocations"]["baseline_guard"]

    structure_daily, _spectral_panel, criticality, structural_stress = _build_structure_layers(context)
    base_score = _clip01(context["attack_score_exogenous"])
    criticality_aligned = _align(criticality, base_score.index, default=0.5)
    structural_stress_aligned = _align(structural_stress, base_score.index, default=0.5)
    market_mode_share_pct = _align(structure_daily.get("market_mode_share_pct"), base_score.index, default=0.5)

    champion_bundle, champion_score, _champion_weight = _build_criticality_free_energy_bundle(
        candidate_id="criticality_free_energy_attack",
        notes="campeão atual: criticidade com reorganização leve",
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        base_score=base_score,
        structure_daily=structure_daily,
        criticality=criticality_aligned,
    )
    print("[selection_rotation] champion baseline ready", flush=True)

    attack_crypto_bundle = _build_crypto_bundle(
        candidate_id="champion_rotation_crypto_attack",
        context=context,
        score_mode="mom_total",
        top_k=1,
    )
    defensive_crypto_bundle = _build_crypto_bundle(
        candidate_id="champion_rotation_crypto_defensive",
        context=context,
        score_mode="mom_vol_adj",
        top_k=3,
    )
    u800_equity_bundle = _build_u800_equity_support(
        prices_dir=(ROOT / args.prices_dir).resolve(),
        asset_groups=(ROOT / args.equity_asset_groups).resolve(),
        asset_metadata=(ROOT / args.equity_asset_metadata).resolve(),
        benchmark_ticker=str(args.benchmark_equity),
        regime_series=context["regime_series"],
        profile=context["profiles"]["foreign"],
    )
    print("[selection_rotation] u800 support ready", flush=True)

    offensive_raw_alloc, offensive_raw_score = _build_raw_attack_alloc_and_score(
        candidate_id="champion_rotation_offensive",
        crypto_bundle=attack_crypto_bundle,
        equity_bundle=context["equity_attack"],
        context=context,
    )
    crypto_only_defensive_alloc, crypto_only_defensive_score = _build_raw_attack_alloc_and_score(
        candidate_id="champion_rotation_crypto_only_defensive",
        crypto_bundle=defensive_crypto_bundle,
        equity_bundle=context["equity_attack"],
        context=context,
    )
    core_defensive_alloc, core_defensive_score = _build_raw_attack_alloc_and_score(
        candidate_id="champion_rotation_core_defensive",
        crypto_bundle=defensive_crypto_bundle,
        equity_bundle=context["equity_base"],
        context=context,
    )
    u800_defensive_alloc, u800_defensive_score = _build_raw_attack_alloc_and_score(
        candidate_id="champion_rotation_u800_defensive",
        crypto_bundle=defensive_crypto_bundle,
        equity_bundle=u800_equity_bundle,
        context=context,
    )
    print("[selection_rotation] offensive and defensive raw sleeves ready", flush=True)

    offensive_share = _build_offensive_share(
        base_score=offensive_raw_score,
        criticality=criticality_aligned,
        structural_stress=structural_stress_aligned,
        market_mode_share_pct=market_mode_share_pct,
    )

    crypto_rotation_attack = _blend_allocation_bundles(
        candidate_id="champion_crypto_rotation_attack_raw",
        notes="troca apenas o sleeve de cripto por um mais diversificado quando o regime piora",
        attack_alloc=offensive_raw_alloc,
        protect_alloc=crypto_only_defensive_alloc,
        attack_weight=offensive_share,
    )
    crypto_rotation_score = _blend_scores(
        offensive_score=offensive_raw_score,
        defensive_score=crypto_only_defensive_score,
        offensive_share=offensive_share,
    )
    crypto_rotation_bundle, _crypto_rotation_final_score, _crypto_rotation_weight = _build_criticality_free_energy_bundle(
        candidate_id="champion_crypto_rotation",
        notes="mantém o ataque original em regime limpo e troca o sleeve cripto para major8 vol-adjusted quando a estrutura aperta",
        attack_alloc=crypto_rotation_attack,
        protect_alloc=protect_alloc,
        base_score=crypto_rotation_score,
        structure_daily=structure_daily,
        criticality=criticality_aligned,
    )
    print("[selection_rotation] crypto rotation ready", flush=True)

    core_rotation_attack = _blend_allocation_bundles(
        candidate_id="champion_core_rotation_attack_raw",
        notes="troca o sleeve cripto e o sleeve de equities por versões mais robustas do núcleo quando o regime piora",
        attack_alloc=offensive_raw_alloc,
        protect_alloc=core_defensive_alloc,
        attack_weight=offensive_share,
    )
    core_rotation_score = _blend_scores(
        offensive_score=offensive_raw_score,
        defensive_score=core_defensive_score,
        offensive_share=offensive_share,
    )
    core_rotation_bundle, _core_rotation_final_score, _core_rotation_weight = _build_criticality_free_energy_bundle(
        candidate_id="champion_core_rotation",
        notes="em stress, sai do top1 cripto e do sleeve agressivo de equities para uma combinacao mais robusta do nucleo",
        attack_alloc=core_rotation_attack,
        protect_alloc=protect_alloc,
        base_score=core_rotation_score,
        structure_daily=structure_daily,
        criticality=criticality_aligned,
    )
    print("[selection_rotation] core rotation ready", flush=True)

    u800_rotation_attack = _blend_allocation_bundles(
        candidate_id="champion_u800_rotation_attack_raw",
        notes="em stress, troca a perna nao cripto pelo melhor apoio atual do universo 800 e amplia o sleeve cripto",
        attack_alloc=offensive_raw_alloc,
        protect_alloc=u800_defensive_alloc,
        attack_weight=offensive_share,
    )
    u800_rotation_score = _blend_scores(
        offensive_score=offensive_raw_score,
        defensive_score=u800_defensive_score,
        offensive_share=offensive_share,
    )
    u800_rotation_bundle, _u800_rotation_final_score, _u800_rotation_weight = _build_criticality_free_energy_bundle(
        candidate_id="champion_u800_rotation",
        notes="em stress, troca top1 cripto por major8 diversificado e leva a perna de ações para o trail switch do universo 800",
        attack_alloc=u800_rotation_attack,
        protect_alloc=protect_alloc,
        base_score=u800_rotation_score,
        structure_daily=structure_daily,
        criticality=criticality_aligned,
    )
    print("[selection_rotation] u800 rotation ready", flush=True)

    results = {
        champion_bundle.bundle.result.candidate_id: champion_bundle.bundle.result,
        crypto_rotation_bundle.bundle.result.candidate_id: crypto_rotation_bundle.bundle.result,
        core_rotation_bundle.bundle.result.candidate_id: core_rotation_bundle.bundle.result,
        u800_rotation_bundle.bundle.result.candidate_id: u800_rotation_bundle.bundle.result,
    }

    compare_rows = [
        _result_row(result, baseline=champion_bundle.bundle.result, family="champion_selection_rotation", label=result.candidate_id)
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

    rotation_df = pd.DataFrame(
        {
            "date": offensive_share.index,
            "offensive_share": offensive_share.to_numpy(dtype=float),
            "criticality": criticality_aligned.reindex(offensive_share.index).to_numpy(dtype=float),
            "structural_stress": structural_stress_aligned.reindex(offensive_share.index).to_numpy(dtype=float),
            "market_mode_share_pct": market_mode_share_pct.reindex(offensive_share.index).to_numpy(dtype=float),
            "base_score": offensive_raw_score.reindex(offensive_share.index).to_numpy(dtype=float),
        }
    )
    rotation_df.to_csv(outdir / "rotation_profile.csv", index=False)

    decision = _summarize_keep_decision(compare_df, champion_bundle.bundle.result.candidate_id)
    research_rows = [
        _research_row(champion_bundle.bundle.result, outdir=outdir, status="keep", methodology="criticality_plus_free_energy", label="Campeão atual"),
        _research_row(crypto_rotation_bundle.bundle.result, outdir=outdir, status="watch", methodology="champion_crypto_rotation", label="Rotação só do sleeve cripto"),
        _research_row(core_rotation_bundle.bundle.result, outdir=outdir, status="watch", methodology="champion_core_rotation", label="Rotação defensiva do núcleo"),
        _research_row(u800_rotation_bundle.bundle.result, outdir=outdir, status="watch", methodology="champion_u800_rotation", label="Rotação defensiva com apoio do universo 800"),
    ]
    _write_json(outdir / "profit_research_rows.json", research_rows)

    summary = {
        "suite": "profit_champion_selection_rotation_suite",
        "baseline_candidate": champion_bundle.bundle.result.candidate_id,
        "best_candidate": decision["best_candidate"],
        "worth_keeping": bool(decision["worth_keeping"]),
        "worth_promoting": bool(decision["worth_promoting"]),
        "baseline_total_return": _safe_float(champion_bundle.bundle.result.net_total_return),
        "baseline_ann_return": _safe_float(champion_bundle.bundle.result.net_ann_return),
        "baseline_sharpe": _safe_float(champion_bundle.bundle.result.net_sharpe),
        "baseline_mdd": _safe_float(champion_bundle.bundle.result.net_max_drawdown),
        "best_total_return": _safe_float(decision.get("best_row", {}).get("net_total_return")),
        "best_ann_return": _safe_float(decision.get("best_row", {}).get("net_ann_return")),
        "best_sharpe": _safe_float(decision.get("best_row", {}).get("net_sharpe")),
        "best_mdd": _safe_float(decision.get("best_row", {}).get("net_max_drawdown")),
        "mdd_improvement_abs": _safe_float(decision.get("mdd_improvement_abs")),
        "total_return_gap": _safe_float(decision.get("total_return_gap")),
    }
    _write_json(outdir / "summary.json", summary)

    write_run_manifest(
        outdir,
        script="scripts/bench/validation/run_profit_champion_selection_rotation_suite.py",
        params=vars(args),
        extra={
            "suite": "profit_champion_selection_rotation_suite",
            "baseline_candidate": champion_bundle.bundle.result.candidate_id,
            "best_candidate": decision["best_candidate"],
            "worth_keeping": bool(decision["worth_keeping"]),
            "worth_promoting": bool(decision["worth_promoting"]),
        },
    )
    print("[selection_rotation] completed", flush=True)


if __name__ == "__main__":
    main()
