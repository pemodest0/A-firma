#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.structural.run_manifest import write_run_manifest  # noqa: E402
from scripts.bench.validation.run_profit_alpha_hardening_suite import (  # noqa: E402
    _build_alpha_meta_allocation_bundle,
    _build_candidates,
)
from scripts.bench.validation.run_profit_frontier_expansion_suite import _simulate_asset_rule, _write_json  # noqa: E402
from scripts.bench.validation.run_profit_sector_pressure_suite import _research_row  # noqa: E402


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _result_row(result) -> dict[str, Any]:
    return {
        "candidate_id": str(result.candidate_id),
        "family": str(result.family),
        "benchmark_ticker": str(result.benchmark_ticker),
        "net_ann_return": float(result.net_ann_return),
        "net_total_return": float(result.net_total_return),
        "net_sharpe": float(result.net_sharpe),
        "net_max_drawdown": float(result.net_max_drawdown),
        "edge_vs_benchmark": float(result.edge_vs_benchmark),
        "avg_turnover_daily": float(result.avg_turnover_daily),
        "notes": str(result.notes or ""),
    }


def _candidate_from_crypto(
    *,
    candidate_id: str,
    score_mode: str,
    allowed_tickers: list[str],
    top_k: int,
    risk_off_mode: str,
    context: dict[str, Any],
):
    crypto_result = _simulate_asset_rule(
        candidate_id=f"{candidate_id}__crypto",
        family="crypto_dependency_relief",
        allowed_tickers=list(allowed_tickers),
        returns=context["crypto_returns"],
        prices=context["crypto_prices"],
        asset_table=context["crypto_assets"],
        benchmark_ticker=context["benchmark_crypto"],
        fallback_ticker=context["benchmark_crypto"],
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
    if crypto_result is None:
        raise SystemExit(f"falha ao simular sleeve cripto: {candidate_id}")

    benchmark_gross_ret = pd.to_numeric(
        context["crypto_returns"][str(context["benchmark_crypto"])], errors="coerce"
    ).reindex(crypto_result.gross_ret.index).fillna(0.0).astype(float)

    from scripts.bench.validation.run_profit_layered_engine_suite import StrategyBundle  # noqa: E402

    crypto_bundle = StrategyBundle(
        result=crypto_result,
        benchmark_gross_ret=benchmark_gross_ret,
        profile=context["profiles"]["crypto"],
        benchmark_profile=context["profiles"]["crypto"],
    )
    allocation = _build_alpha_meta_allocation_bundle(
        candidate_id=str(candidate_id),
        crypto_bundle=crypto_bundle,
        equity_bundle=context["equity_attack"],
        btc_prices=context["btc_prices"],
        spy_prices=context["spy_prices"],
        profile=context["profiles"]["blended"],
        entry_lookback=21,
        exit_lookback=63,
        entry_margin=0.05,
        exit_margin=0.05,
        risk_off_mode=str(risk_off_mode),
        min_crypto_hold_days=0,
    )
    return allocation.bundle.result


def main() -> None:
    ap = argparse.ArgumentParser(description="Busca simples para reduzir dependencia de poucos nomes cripto sem matar lucro.")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--outdir-root", default="results/validation/profit_crypto_dependency_relief_suite")
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
    ctx = built["context"]
    tiers = ctx["crypto_tiers"]

    candidates = [
        built["attack"].result,
        built["baseline"].result,
        _candidate_from_crypto(
            candidate_id="attack_major8_k2",
            score_mode="mom_total",
            allowed_tickers=tiers["crypto_major8"],
            top_k=2,
            risk_off_mode="equity25",
            context=ctx,
        ),
        _candidate_from_crypto(
            candidate_id="attack_major8_k3",
            score_mode="mom_total",
            allowed_tickers=tiers["crypto_major8"],
            top_k=3,
            risk_off_mode="equity25",
            context=ctx,
        ),
        _candidate_from_crypto(
            candidate_id="attack_major8_momvol_k2",
            score_mode="mom_vol_adj",
            allowed_tickers=tiers["crypto_major8"],
            top_k=2,
            risk_off_mode="equity25",
            context=ctx,
        ),
        _candidate_from_crypto(
            candidate_id="attack_all22_k2",
            score_mode="mom_total",
            allowed_tickers=tiers["crypto_all"],
            top_k=2,
            risk_off_mode="equity25",
            context=ctx,
        ),
        _candidate_from_crypto(
            candidate_id="attack_major8_k2_equity50",
            score_mode="mom_total",
            allowed_tickers=tiers["crypto_major8"],
            top_k=2,
            risk_off_mode="equity50",
            context=ctx,
        ),
        _candidate_from_crypto(
            candidate_id="attack_major8_k2_cash",
            score_mode="mom_total",
            allowed_tickers=tiers["crypto_major8"],
            top_k=2,
            risk_off_mode="cash",
            context=ctx,
        ),
    ]

    rows = [_result_row(c) for c in candidates]
    df = pd.DataFrame(rows).sort_values(["net_total_return", "net_sharpe"], ascending=[False, False]).reset_index(drop=True)
    df.to_csv(outdir / "candidate_compare.csv", index=False)

    best_profit = df.iloc[0].to_dict() if not df.empty else {}
    baseline_row = df[df["candidate_id"] == "alpha_attack_major8_equity25"].iloc[0].to_dict()
    better_balance = None
    filtered = df[(df["net_total_return"] >= 0.85 * float(baseline_row["net_total_return"])) & (df["net_max_drawdown"] > float(baseline_row["net_max_drawdown"]))]
    if not filtered.empty:
        better_balance = filtered.sort_values(["net_sharpe", "net_total_return"], ascending=[False, False]).iloc[0].to_dict()

    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_attack": baseline_row,
        "best_profit_variant": best_profit,
        "best_balance_variant": better_balance,
        "insights": [
            "A suíte tenta reduzir a dependência de um nome só no cripto distribuindo a seleção entre 2 ou 3 moedas.",
            "As variantes mantêm a mesma perna de ações e a mesma lógica de meta-switch, mudando só a forma de entrar no sleeve cripto.",
            "O foco aqui é descobrir se dá para perder pouca explosão de lucro e ganhar robustez contra concentração excessiva.",
        ],
    }
    _write_json(outdir / "summary.json", summary)
    _write_json(
        outdir / "profit_research_rows.json",
        [
            _research_row(c, outdir=outdir, status=("keep" if c.candidate_id == str(best_profit.get("candidate_id")) else "watch"), methodology="crypto_dependency_relief", label=str(c.candidate_id))
            for c in candidates
        ],
    )
    write_run_manifest(
        outdir / "RUN_MANIFEST.json",
        script=str(Path(__file__).resolve()),
        params={
            "benchmark_crypto": str(args.benchmark_crypto),
            "benchmark_equity": str(args.benchmark_equity),
        },
        paths={
            "summary_json": str(outdir / "summary.json"),
            "candidate_compare_csv": str(outdir / "candidate_compare.csv"),
        },
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
