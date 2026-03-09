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
from scripts.bench.validation.run_profit_alpha_hardening_suite import _build_candidates  # noqa: E402
from scripts.bench.validation.run_profit_layered_engine_suite import _walkforward_rows  # noqa: E402
from scripts.bench.validation.run_profit_sector_pressure_suite import _research_row  # noqa: E402
from scripts.bench.validation.run_profit_frontier_expansion_suite import _write_json  # noqa: E402


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _build_blocks() -> list[tuple[str, str, str]]:
    return [
        ("2016", "2016-01-01", "2016-12-31"),
        ("2017", "2017-01-01", "2017-12-31"),
        ("2018", "2018-01-01", "2018-12-31"),
        ("2019", "2019-01-01", "2019-12-31"),
        ("2020", "2020-01-01", "2020-12-31"),
        ("2021", "2021-01-01", "2021-12-31"),
        ("2022", "2022-01-01", "2022-12-31"),
        ("2023", "2023-01-01", "2023-12-31"),
        ("2024", "2024-01-01", "2024-12-31"),
        ("2025", "2025-01-01", "2025-12-31"),
        ("2026_ytd", "2026-01-01", "2026-12-31"),
        ("oos_2020_2021", "2020-01-01", "2021-12-31"),
        ("oos_2022", "2022-01-01", "2022-12-31"),
        ("oos_2023_2024", "2023-01-01", "2024-12-31"),
        ("oos_2025_2026", "2025-01-01", "2026-12-31"),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Torneio walk-forward congelado dos melhores modos de lucro.")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--outdir-root", default="results/validation/profit_alpha_walkforward_tournament")
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
    bundles = [
        built["baseline"],
        built["attack"],
        built["baseline_guard"],
        built["attack_guard"],
    ]
    blocks = _build_blocks()
    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        rows.extend(_walkforward_rows(bundle, blocks))
    df = pd.DataFrame(rows).sort_values(["block", "net_total_return"], ascending=[True, False]).reset_index(drop=True)
    df.to_csv(outdir / "walkforward_blocks.csv", index=False)

    winners = (
        df.sort_values(["block", "net_total_return", "net_sharpe"], ascending=[True, False, False])
        .groupby("block", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    winners.to_csv(outdir / "block_winners.csv", index=False)

    score = (
        winners.groupby("candidate_id")
        .agg(
            block_wins=("block", "count"),
            mean_block_edge=("edge_vs_benchmark_net_total_return", "mean"),
            mean_block_return=("net_total_return", "mean"),
            mean_block_sharpe=("net_sharpe", "mean"),
        )
        .reset_index()
        .sort_values(["block_wins", "mean_block_edge", "mean_block_return"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    score.to_csv(outdir / "tournament_score.csv", index=False)

    recent = df[df["block"].astype(str).isin(["oos_2020_2021", "oos_2022", "oos_2023_2024", "oos_2025_2026"])].copy()
    recent_score = (
        recent.groupby("candidate_id")
        .agg(
            recent_blocks=("block", "count"),
            mean_recent_edge=("edge_vs_benchmark_net_total_return", "mean"),
            mean_recent_return=("net_total_return", "mean"),
            mean_recent_sharpe=("net_sharpe", "mean"),
            positive_recent_blocks=("edge_vs_benchmark_net_total_return", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0.0) > 0.0).sum())),
        )
        .reset_index()
        .sort_values(["positive_recent_blocks", "mean_recent_edge", "mean_recent_return"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    recent_score.to_csv(outdir / "recent_oos_score.csv", index=False)

    winner = score.iloc[0].to_dict() if not score.empty else {}
    recent_winner = recent_score.iloc[0].to_dict() if not recent_score.empty else {}
    research_rows = []
    winner_id = str(winner.get("candidate_id", ""))
    recent_id = str(recent_winner.get("candidate_id", ""))
    for bundle in bundles:
        status = "watch"
        if bundle.result.candidate_id == winner_id:
            status = "keep"
        if bundle.result.candidate_id == recent_id:
            status = "keep"
        research_rows.append(
            _research_row(
                bundle.result,
                outdir=outdir,
                status=status,
                methodology="alpha_walkforward_tournament",
                label="Torneio walk-forward congelado",
            )
        )
    (outdir / "profit_research_rows.json").write_text(json.dumps(research_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "outdir": str(outdir),
        "frozen_walkforward_winner": winner,
        "recent_oos_winner": recent_winner,
        "block_winners": winners.to_dict(orient="records"),
        "insights": [
            "Todos os modos foram avaliados com os mesmos parâmetros fixos, sem retuning no meio do caminho.",
            "O torneio mede quem mais vence em blocos sucessivos e quem continua aparecendo bem na parte recente.",
            "A escolha final privilegia consistência de bloco, não só o melhor trecho isolado.",
        ],
        "artifacts": {
            "walkforward_blocks_csv": str(outdir / "walkforward_blocks.csv"),
            "block_winners_csv": str(outdir / "block_winners.csv"),
            "tournament_score_csv": str(outdir / "tournament_score.csv"),
            "recent_oos_score_csv": str(outdir / "recent_oos_score.csv"),
            "profit_research_rows_json": str(outdir / "profit_research_rows.json"),
        },
    }
    _write_json(outdir / "summary.json", summary)
    write_run_manifest(
        outdir,
        script="scripts/bench/validation/run_profit_alpha_walkforward_tournament.py",
        params={
            "benchmark_crypto": args.benchmark_crypto,
            "benchmark_equity": args.benchmark_equity,
            "blocks": [b[0] for b in blocks],
        },
        paths=summary["artifacts"],
        extra={"summary_json": str(outdir / "summary.json")},
    )


if __name__ == "__main__":
    main()
