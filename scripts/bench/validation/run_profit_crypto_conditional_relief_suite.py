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
from scripts.bench.validation.run_profit_alpha_improvement_suite import _build_confidence_score  # noqa: E402
from scripts.bench.validation.run_profit_attack_entry_ranking_suite import (  # noqa: E402
    _build_attack_allocation,
    _make_crypto_bundle,
    _wrap_with_current_confidence,
)
from scripts.bench.validation.run_profit_confidence_calibration_suite import _rolling_percentile  # noqa: E402
from scripts.bench.validation.run_profit_confidence_refinement_suite import (  # noqa: E402
    _dynamic_crypto_bundle,
)
from scripts.bench.validation.run_profit_crypto_resolution_suite import (  # noqa: E402
    _blend_crypto_bundles,
    _safe_float,
)
from scripts.bench.validation.run_profit_layered_engine_suite import (  # noqa: E402
    _apply_breadth_overlay_to_bundle,
    _build_breadth_signal,
)
from scripts.bench.validation.run_profit_sector_pressure_suite import _research_row  # noqa: E402
from scripts.bench.validation.run_profit_universe_resilience_suite import (  # noqa: E402
    _human_label,
    _selection_frequency_for_crypto_rule,
)


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _risk_score(
    *,
    context: dict[str, Any],
    top1_bundle,
    broad_bundle,
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
    confidence = _build_confidence_score(context, breadth_signal, attack_returns).clip(0.0, 1.0)
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
    risk = (0.45 * low_breadth + 0.35 * dominance + 0.20 * low_conf).clip(0.0, 1.0)
    return risk.astype(float)


def _result_row(*, result, drop_top1, drop_top3) -> dict[str, Any]:
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
    score = (
        0.50 * max(0.0, base_total)
        + 0.20 * max(0.0, _safe_float(result.net_sharpe))
        + 0.15 * max(0.0, top1_ret if np.isfinite(top1_ret) else 0.0)
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
        "fragility_adjusted_score": score,
        "notes": str(result.notes or ""),
    }


def _candidate_results(
    *,
    context: dict[str, Any],
    protect_alloc,
    drop_tickers: list[str] | None = None,
) -> dict[str, Any]:
    blocked = set(drop_tickers or [])
    major8 = [ticker for ticker in context["crypto_tiers"]["crypto_major8"] if ticker not in blocked]
    all22 = [ticker for ticker in context["crypto_tiers"]["crypto_all"] if ticker not in blocked]

    top1 = _make_crypto_bundle(
        candidate_id="major8_top1",
        context=context,
        allowed_tickers=major8,
        score_mode="mom_total",
        lookback_days=21,
        rebalance_days=7,
        top_k=1,
    )
    broad = _make_crypto_bundle(
        candidate_id="all22_broad_k3",
        context=context,
        allowed_tickers=all22,
        score_mode="mom_vol_adj",
        lookback_days=21,
        rebalance_days=7,
        top_k=3,
    )
    breadth_signal = _build_breadth_signal(
        returns=context["crypto_returns"],
        prices=context["crypto_prices"],
        tickers=all22,
        lookback_days=21,
        ma_days=200,
    )
    broad_breadth = _apply_breadth_overlay_to_bundle(
        candidate_id="all22_broad_k3_breadth",
        bundle=broad,
        breadth_signal=breadth_signal,
        low_threshold=0.38,
        high_threshold=0.62,
        mode="scale",
    )
    blend70 = _blend_crypto_bundles(
        candidate_id="blend70_major8_base30",
        primary=top1,
        secondary=broad,
        primary_weight=0.70,
    )
    risk = _risk_score(context=context, top1_bundle=top1, broad_bundle=broad_breadth, breadth_signal=breadth_signal)
    conditional_bundle = _dynamic_crypto_bundle(
        candidate_id="conditional_relief_crypto",
        high_bundle=top1,
        mid_bundle=blend70,
        low_bundle=broad_breadth,
        score=1.0 - risk,
    )

    baseline_attack_alloc = _build_attack_allocation(
        candidate_id="current_champion__attack",
        context=context,
        crypto_bundle=top1,
        entry_lookback=21,
        exit_lookback=63,
        entry_margin=0.05,
        exit_margin=0.05,
        min_crypto_hold_days=0,
    )
    fast_attack_alloc = _build_attack_allocation(
        candidate_id="entry_fast14_exit63_m2_h0__attack",
        context=context,
        crypto_bundle=top1,
        entry_lookback=14,
        exit_lookback=63,
        entry_margin=0.02,
        exit_margin=0.05,
        min_crypto_hold_days=0,
    )
    conditional_attack_alloc = _build_attack_allocation(
        candidate_id="conditional_relief_fast14__attack",
        context=context,
        crypto_bundle=conditional_bundle,
        entry_lookback=14,
        exit_lookback=63,
        entry_margin=0.02,
        exit_margin=0.05,
        min_crypto_hold_days=0,
    )

    return {
        "current_champion": _wrap_with_current_confidence(
            candidate_id="current_champion",
            context=context,
            crypto_bundle=top1,
            attack_alloc=baseline_attack_alloc,
            protect_alloc=protect_alloc,
        ),
        "fast_entry": _wrap_with_current_confidence(
            candidate_id="entry_fast14_exit63_m2_h0",
            context=context,
            crypto_bundle=top1,
            attack_alloc=fast_attack_alloc,
            protect_alloc=protect_alloc,
        ),
        "conditional_relief": _wrap_with_current_confidence(
            candidate_id="conditional_relief_fast14",
            context=context,
            crypto_bundle=conditional_bundle,
            attack_alloc=conditional_attack_alloc,
            protect_alloc=protect_alloc,
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Testa alivio condicional da concentracao cripto sem diversificar sempre.")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--outdir-root", default="results/validation/profit_crypto_conditional_relief_suite")
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

    base = _candidate_results(context=context, protect_alloc=protect_alloc)
    drop1 = _candidate_results(context=context, protect_alloc=protect_alloc, drop_tickers=top1)
    drop3 = _candidate_results(context=context, protect_alloc=protect_alloc, drop_tickers=top3)

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

    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "top_crypto_frequency": top_freq.head(8).to_dict(orient="records"),
        "top1_removed_for_dependency_test": top1,
        "top3_removed_for_dependency_test": top3,
        "current_champion": compare_df[compare_df["candidate_id"] == "current_champion"].head(1).to_dict(orient="records")[0],
        "fast_entry": compare_df[compare_df["candidate_id"] == "entry_fast14_exit63_m2_h0"].head(1).to_dict(orient="records")[0],
        "conditional_relief": compare_df[compare_df["candidate_id"] == "conditional_relief_fast14"].head(1).to_dict(orient="records")[0],
        "best_fragility_adjusted": compare_df.head(1).to_dict(orient="records")[0],
        "insights": [
            "O alivio condicional tenta manter o ataque atual e so espalhar o cripto quando breadth baixo, dominancia alta e confianca fraca indicarem dependencia perigosa.",
            "A ideia e aliviar concentracao so quando o risco de ficar dependente de poucos nomes estiver alto.",
            "Se funcionar, o alpha cai pouco e a retencao apos remover os nomes principais melhora.",
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
                methodology="crypto_conditional_relief",
                label=_human_label(str(cid)),
            )
        )
    (outdir / "research_rows.json").write_text(json.dumps({"rows": research_rows}, indent=2, ensure_ascii=False), encoding="utf-8")

    write_run_manifest(
        outdir,
        script="scripts/bench/validation/run_profit_crypto_conditional_relief_suite.py",
        params=vars(args),
        paths={
            "summary_json": "summary.json",
            "candidate_compare_csv": "candidate_compare.csv",
            "research_rows_json": "research_rows.json",
        },
        extra={
            "notes": [
                "Alivia concentracao cripto so quando o risco de dependencia fica alto.",
            ]
        },
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
