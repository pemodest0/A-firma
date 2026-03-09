#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    AllocationBundle,
    _build_alpha_meta_allocation_bundle,
    _build_candidates,
)
from scripts.bench.validation.run_profit_alpha_improvement_suite import (  # noqa: E402
    _blend_allocations,
    _build_confidence_score,
    _result_row,
    _safe_float,
    _write_json,
)
from scripts.bench.validation.run_profit_confidence_calibration_suite import _rolling_percentile  # noqa: E402
from scripts.bench.validation.run_profit_crypto_resolution_suite import (  # noqa: E402
    _blend_crypto_bundles,
    _crypto_rule_bundle,
)
from scripts.bench.validation.run_profit_frontier_expansion_suite import StrategyResult  # noqa: E402
from scripts.bench.validation.run_profit_layered_engine_suite import (  # noqa: E402
    StrategyBundle,
    _apply_breadth_overlay_to_bundle,
    _build_breadth_signal,
)


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _best_by_total_return(candidates: dict[str, StrategyResult]) -> StrategyResult:
    return max(candidates.values(), key=lambda result: (_safe_float(result.net_total_return), _safe_float(result.net_sharpe)))


def _make_bundle_result(
    *,
    candidate_id: str,
    family: str,
    notes: str,
    attack_alloc,
    protect_alloc,
    attack_weight: pd.Series,
) -> StrategyResult:
    bundle = _blend_allocations(
        candidate_id=str(candidate_id),
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        attack_weight=attack_weight,
    )
    return StrategyResult(
        suite="confidence_refinement",
        candidate_id=str(candidate_id),
        family=str(family),
        benchmark_ticker=bundle.result.benchmark_ticker,
        gross_ret=bundle.result.gross_ret,
        turnover=bundle.result.turnover,
        net_ret=bundle.result.net_ret,
        benchmark_net_ret=bundle.result.benchmark_net_ret,
        net_ann_return=bundle.result.net_ann_return,
        net_total_return=bundle.result.net_total_return,
        net_sharpe=bundle.result.net_sharpe,
        net_max_drawdown=bundle.result.net_max_drawdown,
        edge_vs_benchmark=bundle.result.edge_vs_benchmark,
        avg_turnover_daily=bundle.result.avg_turnover_daily,
        hit_rate_10x_5y=float("nan"),
        years_to_10x_full=float("nan"),
        notes=str(notes),
    )


def _score_to_weight(score: pd.Series, *, low: float = 0.15, medium: float = 0.75, high: float = 1.0, med_th: float = 0.48, hi_th: float = 0.63) -> pd.Series:
    weight = pd.Series(float(low), index=score.index, dtype=float)
    weight.loc[score >= float(med_th)] = float(medium)
    weight.loc[score >= float(hi_th)] = float(high)
    return weight.clip(0.0, 1.0)


def _weight_to_state(weight: pd.Series) -> pd.Series:
    states = pd.Series(index=weight.index, dtype=object)
    states.loc[weight >= 0.95] = "high"
    states.loc[(weight >= 0.50) & (weight < 0.95)] = "mid"
    states.loc[weight < 0.50] = "low"
    return states.fillna("low").astype(str)


def _state_to_weight(state: pd.Series) -> pd.Series:
    mapping = {"high": 1.0, "mid": 0.75, "low": 0.15}
    return state.map(mapping).fillna(0.15).astype(float)


def _apply_state_inertia(state: pd.Series, *, min_hold_days: int, confirm_days: int) -> pd.Series:
    base = state.astype(str)
    if base.empty:
        return base
    hold = max(0, int(min_hold_days))
    confirm = max(1, int(confirm_days))
    out = pd.Series(index=base.index, dtype=object)
    current = str(base.iloc[0])
    candidate = current
    candidate_streak = 0
    held_days = 0
    for dt, proposed in base.items():
        proposed = str(proposed)
        if proposed == current:
            candidate = current
            candidate_streak = 0
            held_days += 1
            out.loc[dt] = current
            continue
        if proposed != candidate:
            candidate = proposed
            candidate_streak = 1
        else:
            candidate_streak += 1
        if held_days >= hold and candidate_streak >= confirm:
            current = proposed
            held_days = 1
            candidate = current
            candidate_streak = 0
        else:
            held_days += 1
        out.loc[dt] = current
    return out.astype(str)


def _dynamic_crypto_bundle(
    *,
    candidate_id: str,
    high_bundle: StrategyBundle,
    mid_bundle: StrategyBundle,
    low_bundle: StrategyBundle,
    score: pd.Series,
) -> StrategyBundle:
    idx = (
        high_bundle.result.gross_ret.index.intersection(mid_bundle.result.gross_ret.index)
        .intersection(low_bundle.result.gross_ret.index)
        .intersection(score.index)
    )
    choice = pd.Series("low", index=idx, dtype=object)
    choice.loc[pd.to_numeric(score.reindex(idx), errors="coerce").fillna(0.0) >= 0.48] = "mid"
    choice.loc[pd.to_numeric(score.reindex(idx), errors="coerce").fillna(0.0) >= 0.63] = "high"

    high_gross = pd.to_numeric(high_bundle.result.gross_ret.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    mid_gross = pd.to_numeric(mid_bundle.result.gross_ret.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    low_gross = pd.to_numeric(low_bundle.result.gross_ret.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    high_turn = pd.to_numeric(high_bundle.result.turnover.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    mid_turn = pd.to_numeric(mid_bundle.result.turnover.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    low_turn = pd.to_numeric(low_bundle.result.turnover.reindex(idx), errors="coerce").fillna(0.0).astype(float)

    gross = pd.Series(np.where(choice.eq("high"), high_gross, np.where(choice.eq("mid"), mid_gross, low_gross)), index=idx, dtype=float)
    turnover = pd.Series(np.where(choice.eq("high"), high_turn, np.where(choice.eq("mid"), mid_turn, low_turn)), index=idx, dtype=float)
    benchmark = pd.to_numeric(high_bundle.benchmark_gross_ret.reindex(idx), errors="coerce").fillna(0.0).astype(float)

    result = StrategyResult(
        suite="confidence_refinement",
        candidate_id=f"{candidate_id}__crypto",
        family="dynamic_crypto_bundle",
        benchmark_ticker=high_bundle.result.benchmark_ticker,
        gross_ret=gross,
        turnover=turnover,
        net_ret=gross,
        benchmark_net_ret=benchmark,
        net_ann_return=float("nan"),
        net_total_return=float("nan"),
        net_sharpe=float("nan"),
        net_max_drawdown=float("nan"),
        edge_vs_benchmark=float("nan"),
        avg_turnover_daily=float(turnover.mean()) if len(turnover) else float("nan"),
        hit_rate_10x_5y=float("nan"),
        years_to_10x_full=float("nan"),
        notes="crypto_concentration_variable_by_confidence",
    )
    return StrategyBundle(
        result=result,
        benchmark_gross_ret=benchmark,
        profile=high_bundle.profile,
        benchmark_profile=high_bundle.benchmark_profile,
    )


def _build_attack_allocations(context: dict[str, Any], score: pd.Series) -> dict[str, AllocationBundle]:
    tiers = context["crypto_tiers"]
    top1 = _crypto_rule_bundle(
        candidate_id="attack_major8_k1",
        allowed_tickers=tiers["crypto_major8"],
        score_mode="mom_total",
        top_k=1,
        context=context,
    )
    major8_k3 = _crypto_rule_bundle(
        candidate_id="div_major8_k3",
        allowed_tickers=tiers["crypto_major8"],
        score_mode="mom_vol_adj",
        top_k=3,
        context=context,
    )
    all22_k3 = _crypto_rule_bundle(
        candidate_id="div_all22_k3",
        allowed_tickers=tiers["crypto_all"],
        score_mode="mom_vol_adj",
        top_k=3,
        context=context,
    )
    breadth_signal = _build_breadth_signal(
        returns=context["crypto_returns"],
        prices=context["crypto_prices"],
        tickers=tiers["crypto_all"],
        lookback_days=21,
        ma_days=200,
    )
    all22_breadth = _apply_breadth_overlay_to_bundle(
        candidate_id="div_all22_k3_breadth__crypto",
        bundle=all22_k3,
        breadth_signal=breadth_signal,
        low_threshold=0.38,
        high_threshold=0.62,
        mode="scale",
    )
    blend70 = _blend_crypto_bundles(
        candidate_id="blend70_major8_base30",
        primary=top1,
        secondary=major8_k3,
        primary_weight=0.70,
    )
    dynamic_bundle = _dynamic_crypto_bundle(
        candidate_id="dynamic_confidence_crypto",
        high_bundle=top1,
        mid_bundle=blend70,
        low_bundle=all22_breadth,
        score=score,
    )
    dynamic_attack = _build_alpha_meta_allocation_bundle(
        candidate_id="dynamic_confidence_crypto_attack",
        crypto_bundle=dynamic_bundle,
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
    baseline_attack = context["built_attack_alloc"]
    return {"baseline": baseline_attack, "dynamic_crypto": dynamic_attack}


def _family_inertia(score: pd.Series, attack_alloc, protect_alloc) -> tuple[StrategyResult, list[StrategyResult]]:
    base_weight = _score_to_weight(score)
    base_state = _weight_to_state(base_weight)
    candidates: dict[str, StrategyResult] = {}
    for hold, confirm in [(3, 2), (5, 2), (5, 3), (7, 3)]:
        state = _apply_state_inertia(base_state, min_hold_days=hold, confirm_days=confirm)
        weight = _state_to_weight(state)
        cid = f"inertia__hold{hold}_confirm{confirm}"
        candidates[cid] = _make_bundle_result(
            candidate_id=cid,
            family="inertia",
            notes=f"hold={hold};confirm={confirm}",
            attack_alloc=attack_alloc,
            protect_alloc=protect_alloc,
            attack_weight=weight,
        )
    return _best_by_total_return(candidates), list(candidates.values())


def _family_realistic_confidence(score: pd.Series, context: dict[str, Any], attack_alloc, protect_alloc) -> tuple[StrategyResult, list[StrategyResult]]:
    turnover_pen = _rolling_percentile(
        pd.to_numeric(attack_alloc.bundle.result.turnover, errors="coerce").fillna(0.0).astype(float),
        63,
    ).reindex(score.index).fillna(0.5)
    breadth_signal = _build_breadth_signal(
        returns=context["crypto_returns"],
        prices=context["crypto_prices"],
        tickers=context["crypto_tiers"]["crypto_all"],
        lookback_days=21,
        ma_days=200,
    ).reindex(score.index).fillna(0.0)
    top1_fast = pd.to_numeric(context["built_attack_alloc"].bundle.result.gross_ret.reindex(score.index), errors="coerce").fillna(0.0).astype(float)
    broad_fast = pd.to_numeric(context["built_baseline_alloc"].bundle.result.gross_ret.reindex(score.index), errors="coerce").fillna(0.0).astype(float)
    dominance_spread = _rolling_percentile((top1_fast - broad_fast).clip(lower=0.0), 63).reindex(score.index).fillna(0.5)
    breadth_pen = (1.0 - breadth_signal.clip(0.0, 1.0)).astype(float)

    candidates: dict[str, StrategyResult] = {}
    for turn_w, dom_w, breadth_w in [(0.15, 0.15, 0.10), (0.20, 0.20, 0.10), (0.15, 0.20, 0.15)]:
        adjusted = score * (1.0 - turn_w * turnover_pen - dom_w * dominance_spread - breadth_w * breadth_pen)
        adjusted = adjusted.clip(0.05, 1.0)
        weight = _score_to_weight(adjusted)
        cid = f"realistic_conf__t{int(turn_w*100)}_d{int(dom_w*100)}_b{int(breadth_w*100)}"
        candidates[cid] = _make_bundle_result(
            candidate_id=cid,
            family="realistic_confidence",
            notes=f"turnover_pen={turn_w:.2f};dominance_pen={dom_w:.2f};breadth_pen={breadth_w:.2f}",
            attack_alloc=attack_alloc,
            protect_alloc=protect_alloc,
            attack_weight=weight,
        )
    return _best_by_total_return(candidates), list(candidates.values())


def _family_dynamic_crypto(score: pd.Series, context: dict[str, Any], protect_alloc) -> tuple[StrategyResult, list[StrategyResult]]:
    attack_variants = _build_attack_allocations(context, score)
    candidates: dict[str, StrategyResult] = {}
    for high, medium, low, hi_th, med_th, label in [
        (1.0, 0.75, 0.15, 0.63, 0.48, "champion"),
        (1.0, 0.80, 0.10, 0.68, 0.50, "piecewise_more_attack"),
        (1.0, 0.70, 0.20, 0.65, 0.50, "baseline_conf"),
    ]:
        weight = _score_to_weight(score, low=low, medium=medium, high=high, med_th=med_th, hi_th=hi_th)
        cid = f"dynamic_crypto__{label}"
        candidates[cid] = _make_bundle_result(
            candidate_id=cid,
            family="dynamic_crypto",
            notes=f"attack_alloc=dynamic_crypto;high={high:.2f};medium={medium:.2f};low={low:.2f}",
            attack_alloc=attack_variants["dynamic_crypto"],
            protect_alloc=protect_alloc,
            attack_weight=weight,
        )
    return _best_by_total_return(candidates), list(candidates.values())


def _family_combo(score: pd.Series, context: dict[str, Any], protect_alloc) -> tuple[StrategyResult, list[StrategyResult]]:
    attack_variants = _build_attack_allocations(context, score)
    turnover_pen = _rolling_percentile(
        pd.to_numeric(attack_variants["dynamic_crypto"].bundle.result.turnover, errors="coerce").fillna(0.0).astype(float),
        63,
    ).reindex(score.index).fillna(0.5)
    breadth_signal = _build_breadth_signal(
        returns=context["crypto_returns"],
        prices=context["crypto_prices"],
        tickers=context["crypto_tiers"]["crypto_all"],
        lookback_days=21,
        ma_days=200,
    ).reindex(score.index).fillna(0.0)
    adjusted = (score * (1.0 - 0.15 * turnover_pen - 0.10 * (1.0 - breadth_signal))).clip(0.05, 1.0)
    base_state = _weight_to_state(_score_to_weight(adjusted))
    candidates: dict[str, StrategyResult] = {}
    for hold, confirm in [(3, 2), (5, 2), (5, 3)]:
        state = _apply_state_inertia(base_state, min_hold_days=hold, confirm_days=confirm)
        weight = _state_to_weight(state)
        cid = f"combo__hold{hold}_confirm{confirm}"
        candidates[cid] = _make_bundle_result(
            candidate_id=cid,
            family="combo",
            notes=f"dynamic_crypto + realistic_conf + inertia hold={hold} confirm={confirm}",
            attack_alloc=attack_variants["dynamic_crypto"],
            protect_alloc=protect_alloc,
            attack_weight=weight,
        )
    return _best_by_total_return(candidates), list(candidates.values())


def main() -> None:
    ap = argparse.ArgumentParser(description="Refina o campeao de confianca com inercia, confianca realista e concentracao cripto variavel.")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--outdir-root", default="results/validation/profit_confidence_refinement_suite")
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
    context["built_attack_alloc"] = built["allocations"]["attack"]
    context["built_baseline_alloc"] = built["allocations"]["baseline"]
    baseline_attack = built["allocations"]["attack"].bundle.result
    protect_alloc = built["allocations"]["baseline_guard"]

    breadth_signal = _build_breadth_signal(
        returns=context["crypto_returns"],
        prices=context["crypto_prices"],
        tickers=context["crypto_tiers"]["crypto_all"],
        lookback_days=21,
        ma_days=200,
    )
    raw_score = _build_confidence_score(context, breadth_signal, built["sleeve_returns"]["attack"]).clip(0.0, 1.0)
    score = _rolling_percentile(raw_score, 126).fillna(raw_score)

    current_champion = _make_bundle_result(
        candidate_id="confidence_calibrated_current_champion",
        family="baseline_current",
        notes="pct126 + high100_mid75_low15",
        attack_alloc=built["allocations"]["attack"],
        protect_alloc=protect_alloc,
        attack_weight=_score_to_weight(score),
    )

    families: dict[str, tuple[StrategyResult, list[StrategyResult]]] = {
        "inertia": _family_inertia(score, built["allocations"]["attack"], protect_alloc),
        "realistic_confidence": _family_realistic_confidence(score, context, built["allocations"]["attack"], protect_alloc),
        "dynamic_crypto": _family_dynamic_crypto(score, context, protect_alloc),
        "combo": _family_combo(score, context, protect_alloc),
    }

    winner_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    best_result = current_champion
    for family, (winner, variants) in families.items():
        row = _result_row(winner, baseline=current_champion)
        row["family"] = family
        winner_rows.append(row)
        if (_safe_float(winner.net_total_return), _safe_float(winner.net_sharpe)) > (
            _safe_float(best_result.net_total_return),
            _safe_float(best_result.net_sharpe),
        ):
            best_result = winner
        for result in variants:
            row = _result_row(result, baseline=current_champion)
            row["family"] = family
            all_rows.append(row)

    winners_df = pd.DataFrame(winner_rows).sort_values(["net_total_return", "net_sharpe"], ascending=[False, False]).reset_index(drop=True)
    winners_df.to_csv(outdir / "family_winners.csv", index=False)
    variants_df = pd.DataFrame(all_rows).sort_values(["family", "net_total_return", "net_sharpe"], ascending=[True, False, False]).reset_index(drop=True)
    variants_df.to_csv(outdir / "all_variants.csv", index=False)

    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_current_champion": _result_row(current_champion, baseline=current_champion),
        "best_overall": _result_row(best_result, baseline=current_champion),
        "family_winners": winner_rows,
        "insights": [
            "A busca refinou o campeao atual por tres frentes: inercia curta, confianca mais realista e concentracao cripto variavel.",
            "Os resultados sao comparados contra o campeao atual de sizing por confianca calibrado por percentile 126.",
            "Worth_keeping_alpha marca so as variantes que melhoraram o lucro final contra o campeao atual.",
        ],
        "artifacts": {
            "family_winners_csv": str(outdir / "family_winners.csv"),
            "all_variants_csv": str(outdir / "all_variants.csv"),
        },
    }
    _write_json(outdir / "summary.json", summary)

    write_run_manifest(
        outdir,
        script="scripts/bench/validation/run_profit_confidence_refinement_suite.py",
        params={
            "crypto_asset_groups": str(args.crypto_asset_groups),
            "crypto_asset_metadata": str(args.crypto_asset_metadata),
            "equity_asset_groups": str(args.equity_asset_groups),
            "equity_asset_metadata": str(args.equity_asset_metadata),
            "prices_dir": str(args.prices_dir),
            "benchmark_crypto": str(args.benchmark_crypto),
            "benchmark_equity": str(args.benchmark_equity),
        },
        paths={
            "summary_json": str(outdir / "summary.json"),
            "family_winners_csv": str(outdir / "family_winners.csv"),
            "all_variants_csv": str(outdir / "all_variants.csv"),
        },
        extra={
            "suite": "profit_confidence_refinement_suite",
            "best_candidate_id": str(best_result.candidate_id),
            "best_total_return": _safe_float(best_result.net_total_return),
            "best_ann_return": _safe_float(best_result.net_ann_return),
            "best_sharpe": _safe_float(best_result.net_sharpe),
            "best_max_drawdown": _safe_float(best_result.net_max_drawdown),
            "baseline_total_return": _safe_float(current_champion.net_total_return),
        },
        repo_root=ROOT,
    )
    print(outdir)


if __name__ == "__main__":
    main()
