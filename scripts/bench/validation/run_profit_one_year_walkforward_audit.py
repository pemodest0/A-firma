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

from engine.structural.run_manifest import write_run_manifest  # noqa: E402
from scripts.bench.validation.run_profit_alpha_improvement_suite import _write_json  # noqa: E402
from scripts.bench.validation.run_profit_one_year_payoff_audit import (  # noqa: E402
    DEFAULT_CRYPTO_CANDIDATES,
    _forward_path_frame,
    _parse_rule_candidate_id,
    _payoff_row_from_frame,
    _simulate_crypto_candidate_series,
)
from scripts.bench.validation.run_profit_10x_rule_search import (  # noqa: E402
    _ensure_benchmark_columns,
    _load_asset_table,
    _load_daily_universe,
    _precompute_scores,
)
from execution.net_assumptions import load_net_assumption_profiles  # noqa: E402


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _year_return(net_ret: pd.Series, year: int) -> float:
    ret = pd.to_numeric(net_ret, errors="coerce").fillna(0.0).astype(float)
    mask = pd.to_datetime(ret.index).year == int(year)
    block = ret[mask]
    if block.empty:
        return float("nan")
    return float(np.prod(1.0 + block.to_numpy(dtype=float)) - 1.0)


def _train_score(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row.get("hit_rate_6x_252d", float("-inf"))),
        float(row.get("median_return_252d", float("-inf"))),
        -float(row.get("end_below_50_252d", float("inf"))),
        -float(row.get("touch_loss_50_252d", float("inf"))),
    )


def _subset_frame_for_start_year(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    out["start_date"] = pd.to_datetime(out["start_date"], errors="coerce")
    out = out[out["start_date"].dt.year.eq(int(year))].copy()
    out["start_date"] = out["start_date"].dt.strftime("%Y-%m-%d")
    return out.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward serio de payoff 252d: treina ate o ano anterior e executa no ano seguinte.")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid.csv")
    ap.add_argument("--net-assumptions-config", default="config/profit_net_assumptions.json")
    ap.add_argument("--benchmark-ticker", default="BTC-USD")
    ap.add_argument("--fallback-ticker", default="BTC-USD")
    ap.add_argument("--horizon-days", type=int, default=252)
    ap.add_argument("--years", default="2023,2024,2025")
    ap.add_argument("--candidate-ids", default=",".join(DEFAULT_CRYPTO_CANDIDATES))
    ap.add_argument("--outdir-root", default="results/validation/profit_one_year_walkforward_audit")
    args = ap.parse_args()

    outdir = (ROOT / args.outdir_root / _run_id()).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    years = [int(token.strip()) for token in str(args.years).split(",") if token.strip()]
    candidate_ids = [token.strip() for token in str(args.candidate_ids).split(",") if token.strip()]
    prices_dir = (ROOT / args.prices_dir).resolve()
    asset_groups = (ROOT / args.crypto_asset_groups).resolve()
    asset_metadata = (ROOT / args.crypto_asset_metadata).resolve()
    net_assumptions_config = (ROOT / args.net_assumptions_config).resolve()

    asset_table = _load_asset_table(asset_groups, asset_metadata)
    returns, prices, _viability = _load_daily_universe(
        prices_dir=prices_dir,
        asset_table=asset_table,
        min_history_days=252,
        max_abs_daily_return=2.0,
    )
    returns, prices = _ensure_benchmark_columns(returns, prices, prices_dir, [str(args.benchmark_ticker)])
    returns = returns[returns.index >= pd.Timestamp("2016-02-18")].copy()
    prices = prices.reindex(returns.index).copy()

    lookbacks = sorted({int(cid.split("__")[1].replace("lb", "")) for cid in candidate_ids})
    ma_days = sorted(
        {
            int(cid.split("__")[5].replace("ama", ""))
            for cid in candidate_ids
        }
        | {
            int(cid.split("__")[6].replace("mma", ""))
            for cid in candidate_ids
        }
        | {0}
    )
    score_map, asset_ma_filters, benchmark_filters = _precompute_scores(
        returns,
        prices,
        lookbacks=lookbacks,
        asset_ma_days_list=ma_days,
        benchmark_ticker=str(args.benchmark_ticker),
    )
    net_profiles = load_net_assumption_profiles(net_assumptions_config)
    all_groups = tuple(sorted(asset_table["asset_group"].astype(str).unique().tolist()))

    series_by_candidate: dict[str, pd.Series] = {}
    base_metrics_by_candidate: dict[str, dict[str, Any]] = {}
    for candidate_id in candidate_ids:
        cfg = _parse_rule_candidate_id(candidate_id, groups=all_groups)
        row, net_ret = _simulate_crypto_candidate_series(
            cfg=cfg,
            returns=returns,
            prices=prices,
            asset_table=asset_table,
            score_map=score_map,
            asset_ma_filters=asset_ma_filters,
            benchmark_filters=benchmark_filters,
            benchmark_ticker=str(args.benchmark_ticker),
            fallback_ticker=str(args.fallback_ticker),
            net_profiles=net_profiles,
        )
        if net_ret.empty:
            continue
        series_by_candidate[str(candidate_id)] = net_ret
        base_metrics_by_candidate[str(candidate_id)] = {
            "full_net_ann_return": float(row.get("net_ann_return", float("nan"))),
            "full_net_total_return": float(row.get("net_total_return", float("nan"))),
            "full_net_max_drawdown": float(row.get("net_max_drawdown", float("nan"))),
            "avg_turnover_daily": float(row.get("avg_turnover_daily", float("nan"))),
        }

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []

    for year in years:
        train_end = pd.Timestamp(year=year - 1, month=12, day=31)
        best_train_row: dict[str, Any] | None = None
        best_candidate_id: str | None = None

        for candidate_id, net_ret in series_by_candidate.items():
            train_ret = net_ret[net_ret.index <= train_end].copy()
            train_frame = _forward_path_frame(train_ret, horizon_days=int(args.horizon_days), monthly_start=False)
            train_row = _payoff_row_from_frame(
                scenario=f"train_to_{year-1}",
                candidate_id=candidate_id,
                frame=train_frame,
                base_metrics=base_metrics_by_candidate.get(candidate_id, {}),
            )
            train_row["eval_year"] = int(year)
            train_row["train_end"] = str(train_end.date())
            train_row["selection_rank_key"] = list(_train_score(train_row))
            train_rows.append(train_row)

            if best_train_row is None or _train_score(train_row) > _train_score(best_train_row):
                best_train_row = train_row
                best_candidate_id = candidate_id

        if best_candidate_id is None or best_train_row is None:
            continue

        full_frame = _forward_path_frame(series_by_candidate[best_candidate_id], horizon_days=int(args.horizon_days), monthly_start=False)
        eval_frame = _subset_frame_for_start_year(full_frame, year)
        eval_row = _payoff_row_from_frame(
            scenario=f"eval_{year}",
            candidate_id=best_candidate_id,
            frame=eval_frame,
            base_metrics=base_metrics_by_candidate.get(best_candidate_id, {}),
        )
        eval_row["eval_year"] = int(year)
        eval_row["train_end"] = str(train_end.date())
        eval_row["selected_from_train_candidate_id"] = str(best_candidate_id)
        eval_row["train_hit_rate_6x_252d"] = float(best_train_row.get("hit_rate_6x_252d", float("nan")))
        eval_row["train_median_return_252d"] = float(best_train_row.get("median_return_252d", float("nan")))
        eval_row["train_end_below_50_252d"] = float(best_train_row.get("end_below_50_252d", float("nan")))
        eval_row["calendar_year_return"] = _year_return(series_by_candidate[best_candidate_id], year)
        eval_rows.append(eval_row)

    train_df = pd.DataFrame(train_rows).sort_values(["eval_year", "hit_rate_6x_252d", "median_return_252d", "end_below_50_252d"], ascending=[True, False, False, True]).reset_index(drop=True)
    eval_df = pd.DataFrame(eval_rows).sort_values(["eval_year"]).reset_index(drop=True)
    train_df.to_csv(outdir / "train_candidate_compare.csv", index=False)
    eval_df.to_csv(outdir / "eval_year_compare.csv", index=False)

    summary = {
        "suite": "profit_one_year_walkforward_audit",
        "years": years,
        "horizon_days": int(args.horizon_days),
        "selection_rule": [
            "1. maior hit_rate_6x_252d no treino",
            "2. maior mediana de retorno em 252d no treino",
            "3. menor probabilidade de terminar abaixo de -50% no treino",
            "4. menor probabilidade de tocar -50% no caminho no treino",
        ],
        "notes": [
            "Daily-start usa todos os dias uteis como ponto de partida.",
            "A janela forward de 252 dias pode cruzar para o ano seguinte; o criterio de recorte e pelo ano do start, nao pelo ano do encerramento.",
            "Para 2025, a cobertura depende de quantos starts ainda cabem dentro do historico ate 2026.",
        ],
        "train_winners_by_year": eval_df[["eval_year", "selected_from_train_candidate_id", "train_hit_rate_6x_252d", "train_median_return_252d", "train_end_below_50_252d"]].to_dict("records") if not eval_df.empty else [],
        "eval_results": eval_df.to_dict("records"),
    }
    _write_json(outdir / "summary.json", summary)

    write_run_manifest(
        outdir,
        script="scripts/bench/validation/run_profit_one_year_walkforward_audit.py",
        params={
            "prices_dir": str(prices_dir),
            "crypto_asset_groups": str(asset_groups),
            "crypto_asset_metadata": str(asset_metadata),
            "net_assumptions_config": str(net_assumptions_config),
            "benchmark_ticker": str(args.benchmark_ticker),
            "fallback_ticker": str(args.fallback_ticker),
            "horizon_days": int(args.horizon_days),
            "years": years,
            "candidate_ids": candidate_ids,
        },
        extra={"suite": "profit_one_year_walkforward_audit"},
    )
    print(str(outdir))


if __name__ == "__main__":
    main()
