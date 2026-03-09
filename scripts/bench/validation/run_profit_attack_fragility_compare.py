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
from scripts.bench.validation.run_profit_alpha_hardening_suite import _build_candidates  # noqa: E402
from scripts.bench.validation.run_profit_attack_entry_ranking_suite import (  # noqa: E402
    _build_attack_allocation,
    _make_crypto_bundle,
    _wrap_with_current_confidence,
)
from scripts.bench.validation.run_profit_crypto_resolution_suite import _safe_float  # noqa: E402
from scripts.bench.validation.run_profit_frontier_expansion_suite import StrategyResult  # noqa: E402
from scripts.bench.validation.run_profit_sector_pressure_suite import _research_row  # noqa: E402
from scripts.bench.validation.run_profit_universe_resilience_suite import (  # noqa: E402
    _human_label,
    _selection_frequency_for_crypto_rule,
)


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _result_row(
    *,
    result: StrategyResult,
    drop_top1: StrategyResult | None,
    drop_top3: StrategyResult | None,
) -> dict[str, Any]:
    base_total = _safe_float(result.net_total_return)
    base_ann = _safe_float(result.net_ann_return)
    top1_total = _safe_float(drop_top1.net_total_return) if drop_top1 is not None else float("nan")
    top3_total = _safe_float(drop_top3.net_total_return) if drop_top3 is not None else float("nan")
    top1_ann = _safe_float(drop_top1.net_ann_return) if drop_top1 is not None else float("nan")
    top3_ann = _safe_float(drop_top3.net_ann_return) if drop_top3 is not None else float("nan")
    top1_ret = top1_total / base_total if np.isfinite(base_total) and abs(base_total) > 1e-9 and np.isfinite(top1_total) else float("nan")
    top3_ret = top3_total / base_total if np.isfinite(base_total) and abs(base_total) > 1e-9 and np.isfinite(top3_total) else float("nan")
    top1_ann_ret = top1_ann / base_ann if np.isfinite(base_ann) and abs(base_ann) > 1e-9 and np.isfinite(top1_ann) else float("nan")
    top3_ann_ret = top3_ann / base_ann if np.isfinite(base_ann) and abs(base_ann) > 1e-9 and np.isfinite(top3_ann) else float("nan")
    fragility_adjusted = (
        0.45 * max(0.0, base_total)
        + 0.20 * max(0.0, _safe_float(result.net_sharpe))
        + 0.20 * max(0.0, top1_ret if np.isfinite(top1_ret) else 0.0)
        + 0.15 * max(0.0, top3_ret if np.isfinite(top3_ret) else 0.0)
    )
    return {
        "candidate_id": str(result.candidate_id),
        "candidate_label": _human_label(str(result.candidate_id)),
        "net_ann_return": base_ann,
        "net_total_return": base_total,
        "net_sharpe": _safe_float(result.net_sharpe),
        "net_max_drawdown": _safe_float(result.net_max_drawdown),
        "avg_turnover_daily": _safe_float(result.avg_turnover_daily),
        "drop_top1_total_return": top1_total,
        "drop_top3_total_return": top3_total,
        "drop_top1_ann_return": top1_ann,
        "drop_top3_ann_return": top3_ann,
        "top1_total_retention": top1_ret,
        "top3_total_retention": top3_ret,
        "top1_ann_retention": top1_ann_ret,
        "top3_ann_retention": top3_ann_ret,
        "fragility_adjusted_score": fragility_adjusted,
        "notes": str(result.notes or ""),
    }


def _candidate_from_bundle(
    *,
    candidate_id: str,
    context: dict[str, Any],
    protect_alloc,
    allowed_tickers: list[str],
    entry_lookback: int,
    exit_lookback: int,
    entry_margin: float,
    exit_margin: float,
    hold_days: int,
) -> StrategyResult:
    crypto_bundle = _make_crypto_bundle(
        candidate_id=candidate_id,
        context=context,
        allowed_tickers=allowed_tickers,
        score_mode="mom_total",
        lookback_days=21,
        rebalance_days=7,
        top_k=1,
    )
    attack_alloc = _build_attack_allocation(
        candidate_id=f"{candidate_id}__attack",
        context=context,
        crypto_bundle=crypto_bundle,
        entry_lookback=entry_lookback,
        exit_lookback=exit_lookback,
        entry_margin=entry_margin,
        exit_margin=exit_margin,
        min_crypto_hold_days=hold_days,
    )
    return _wrap_with_current_confidence(
        candidate_id=str(candidate_id),
        context=context,
        crypto_bundle=crypto_bundle,
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
    )


def _build_variant_results(
    *,
    context: dict[str, Any],
    protect_alloc,
    drop_tickers: list[str] | None = None,
) -> dict[str, StrategyResult]:
    major8 = [ticker for ticker in context["crypto_tiers"]["crypto_major8"] if ticker not in set(drop_tickers or [])]
    results = {
        "current_champion": _candidate_from_bundle(
            candidate_id="current_champion",
            context=context,
            protect_alloc=protect_alloc,
            allowed_tickers=major8,
            entry_lookback=21,
            exit_lookback=63,
            entry_margin=0.05,
            exit_margin=0.05,
            hold_days=0,
        ),
        "entry_fast14_exit63_m2_h0": _candidate_from_bundle(
            candidate_id="entry_fast14_exit63_m2_h0",
            context=context,
            protect_alloc=protect_alloc,
            allowed_tickers=major8,
            entry_lookback=14,
            exit_lookback=63,
            entry_margin=0.02,
            exit_margin=0.05,
            hold_days=0,
        ),
        "baseline_guard": protect_alloc.bundle.result,
    }
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Compara alpha do novo gatilho de entrada contra fragilidade cripto.")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--outdir-root", default="results/validation/profit_attack_fragility_compare")
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
    protect_alloc = built["allocations"]["baseline_guard"]

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
    top1 = top_freq["ticker"].head(1).astype(str).tolist()
    top3 = top_freq["ticker"].head(3).astype(str).tolist()

    base = _build_variant_results(context=context, protect_alloc=protect_alloc)
    drop1 = _build_variant_results(context=context, protect_alloc=protect_alloc, drop_tickers=top1)
    drop3 = _build_variant_results(context=context, protect_alloc=protect_alloc, drop_tickers=top3)

    rows = [
        _result_row(result=result, drop_top1=drop1.get(cid), drop_top3=drop3.get(cid))
        for cid, result in base.items()
    ]
    compare_df = pd.DataFrame(rows).sort_values(
        ["fragility_adjusted_score", "net_total_return", "net_sharpe"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    compare_path = outdir / "candidate_compare.csv"
    compare_df.to_csv(compare_path, index=False)

    baseline = compare_df[compare_df["candidate_id"] == "current_champion"].head(1)
    fast_entry = compare_df[compare_df["candidate_id"] == "entry_fast14_exit63_m2_h0"].head(1)
    guard = compare_df[compare_df["candidate_id"] == "baseline_guard"].head(1)

    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "top_crypto_frequency": top_freq.head(8).to_dict(orient="records"),
        "top1_removed_for_dependency_test": top1,
        "top3_removed_for_dependency_test": top3,
        "baseline_current_champion": baseline.iloc[0].to_dict() if not baseline.empty else {},
        "fast_entry_candidate": fast_entry.iloc[0].to_dict() if not fast_entry.empty else {},
        "protect_mode_reference": guard.iloc[0].to_dict() if not guard.empty else {},
        "best_fragility_adjusted": compare_df.head(1).to_dict(orient="records")[0] if not compare_df.empty else {},
        "insights": [
            "A suite compara o campeao atual, o novo gatilho de entrada rapida e o modo protegido medindo lucro e dependencia dos nomes cripto mais fortes.",
            "Se o novo gatilho ganhar muito, mas desabar quando tiramos os principais nomes, ele nao deve ser promovido sem ressalva.",
            "O score final tenta equilibrar lucro, qualidade e retencao apos remover o topo do sleeve cripto.",
        ],
        "artifacts": {
            "candidate_compare_csv": str(compare_path),
        },
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    research_rows = []
    for cid, result in base.items():
        state = "watch"
        if cid == str(summary["best_fragility_adjusted"].get("candidate_id", "")):
            state = "keep"
        research_rows.append(
            _research_row(
                result,
                outdir=outdir,
                status=state,
                methodology="attack_fragility_compare",
                label=_human_label(str(cid)),
            )
        )
    (outdir / "research_rows.json").write_text(json.dumps({"rows": research_rows}, indent=2, ensure_ascii=False), encoding="utf-8")

    write_run_manifest(
        outdir,
        script="scripts/bench/validation/run_profit_attack_fragility_compare.py",
        params=vars(args),
        paths={
            "summary_json": "summary.json",
            "candidate_compare_csv": "candidate_compare.csv",
            "research_rows_json": "research_rows.json",
        },
        extra={
            "notes": [
                "Compara lucro do novo gatilho de entrada contra fragilidade dos principais nomes cripto.",
            ]
        },
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
