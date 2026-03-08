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

from execution.net_assumptions import (  # noqa: E402
    apply_net_assumptions,
    blend_profiles,
    load_net_assumption_profiles,
    summarize_net_series,
)


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _latest_suite_dir(root: Path) -> Path:
    dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not dirs:
        raise FileNotFoundError(f"no suite dirs in {root}")
    return dirs[-1]


def _eval_slice(
    candidate_monthly: pd.DataFrame,
    spy_monthly: pd.Series,
    profiles: dict[str, Any],
) -> dict[str, Any]:
    monthly = candidate_monthly.copy()
    if "ym" in monthly.columns:
        monthly["ym"] = monthly["ym"].astype(str)
        monthly = monthly.drop_duplicates(subset=["ym"], keep="last").set_index("ym", drop=True)
    gross = pd.to_numeric(monthly["gross_ret"], errors="coerce").dropna().astype(float)
    if gross.empty:
        return {}
    turnover = pd.to_numeric(monthly["turnover"], errors="coerce").reindex(gross.index).fillna(0.0).astype(float)
    foreign_share = float(pd.to_numeric(monthly["foreign_share"], errors="coerce").fillna(0.0).mean())
    foreign = profiles["profiles"]["foreign_financial_brazil_resident"]
    local = profiles["profiles"]["br_local_equity"]
    blended = blend_profiles(foreign_share, foreign_profile=foreign, br_profile=local)
    net = apply_net_assumptions(gross, turnover, profile=blended, periods_index=gross.index)
    spy = pd.to_numeric(spy_monthly, errors="coerce").reindex(gross.index).dropna().astype(float)
    if spy.empty:
        return {}
    gross = gross.reindex(spy.index).dropna().astype(float)
    turnover = turnover.reindex(gross.index).fillna(0.0).astype(float)
    net = apply_net_assumptions(gross, turnover, profile=blended, periods_index=gross.index)
    spy = spy.reindex(gross.index).astype(float)
    spy_net = apply_net_assumptions(
        spy,
        pd.Series(np.zeros(len(spy), dtype=float), index=spy.index, dtype=float),
        profile=foreign,
        periods_index=spy.index,
    )
    net_summary = summarize_net_series(net, periods_per_year=12)
    spy_summary = summarize_net_series(spy_net, periods_per_year=12)
    return {
        "gross_ann_return": _safe_float(net_summary["gross"].get("annualized_return")),
        "net_ann_return": _safe_float(net_summary["net"].get("annualized_return")),
        "net_sharpe": _safe_float(net_summary["net"].get("sharpe")),
        "net_max_drawdown": _safe_float(net_summary["net"].get("max_drawdown")),
        "spy_net_ann_return": _safe_float(spy_summary["net"].get("annualized_return")),
        "edge_vs_spy_net_total_return": _safe_float(net_summary["net"].get("total_return")) - _safe_float(spy_summary["net"].get("total_return")),
        "months": int(gross.shape[0]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Valida OOS da pesquisa de grupos de lucro.")
    ap.add_argument("--suite-dir", default="")
    ap.add_argument("--suite-root", default="results/validation/profit_group_methodology_suite")
    ap.add_argument("--net-assumptions-config", default="config/profit_net_assumptions.json")
    ap.add_argument("--splits", default="2020-12,2021-12,2022-12")
    ap.add_argument("--top-n-train", type=int, default=10)
    ap.add_argument("--outdir-root", default="results/validation/profit_group_oos_validation")
    args = ap.parse_args()

    suite_dir = Path(args.suite_dir).resolve() if str(args.suite_dir).strip() else _latest_suite_dir((ROOT / args.suite_root).resolve())
    candidate_compare = pd.read_csv(suite_dir / "candidate_compare.csv")
    candidate_monthly = pd.read_csv(suite_dir / "candidate_monthly_returns.csv")
    benchmark_monthly = pd.read_csv(suite_dir / "benchmark_monthly_returns.csv")
    profiles = load_net_assumption_profiles((ROOT / args.net_assumptions_config).resolve())
    outdir = (ROOT / args.outdir_root / _run_id()).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    benchmark_monthly["ym"] = benchmark_monthly["ym"].astype(str)
    candidate_monthly["ym"] = candidate_monthly["ym"].astype(str)
    spy_monthly = pd.Series(pd.to_numeric(benchmark_monthly["SPY"], errors="coerce").to_numpy(dtype=float), index=benchmark_monthly["ym"], dtype=float)

    split_rows: list[dict[str, Any]] = []
    for split in [x.strip() for x in str(args.splits).split(",") if x.strip()]:
        train_rows = []
        for row in candidate_compare.to_dict(orient="records"):
            cid = str(row["candidate_id"])
            monthly = candidate_monthly[candidate_monthly["candidate_id"].astype(str) == cid].copy()
            monthly = monthly.sort_values("ym").reset_index(drop=True)
            train = monthly[monthly["ym"].astype(str) <= str(split)].copy()
            test = monthly[monthly["ym"].astype(str) > str(split)].copy()
            if train.shape[0] < 24 or test.shape[0] < 12:
                continue
            train_eval = _eval_slice(train, spy_monthly, profiles)
            test_eval = _eval_slice(test, spy_monthly, profiles)
            if not train_eval or not test_eval:
                continue
            train_rows.append(
                {
                    "split_ym": str(split),
                    "candidate_id": cid,
                    "kind": str(row["kind"]),
                    "groups": str(row.get("groups", "")),
                    "train_net_ann_return": _safe_float(train_eval.get("net_ann_return")),
                    "train_net_sharpe": _safe_float(train_eval.get("net_sharpe")),
                    "test_net_ann_return": _safe_float(test_eval.get("net_ann_return")),
                    "test_net_sharpe": _safe_float(test_eval.get("net_sharpe")),
                    "test_edge_vs_spy_net_total_return": _safe_float(test_eval.get("edge_vs_spy_net_total_return")),
                    "test_months": int(test_eval.get("months", 0)),
                }
            )
        if not train_rows:
            continue
        split_df = pd.DataFrame(train_rows).sort_values(
            ["train_net_ann_return", "train_net_sharpe"],
            ascending=[False, False],
        ).reset_index(drop=True)
        split_df["train_rank"] = np.arange(1, split_df.shape[0] + 1, dtype=int)
        top = split_df.head(int(max(1, args.top_n_train))).copy()
        top.to_csv(outdir / f"split_{split.replace('-', '')}_top_train.csv", index=False)
        split_rows.extend(top.to_dict(orient="records"))

    split_results = pd.DataFrame(split_rows)
    if split_results.empty:
        raise SystemExit("no OOS rows produced")
    split_results.to_csv(outdir / "oos_candidate_results.csv", index=False)

    top1 = split_results[split_results["train_rank"] == 1].copy()
    top1 = top1.sort_values("split_ym").reset_index(drop=True)
    best_consistent = (
        split_results.groupby("candidate_id", as_index=False)
        .agg(
            appearances=("candidate_id", "count"),
            mean_test_net_ann_return=("test_net_ann_return", "mean"),
            mean_test_net_sharpe=("test_net_sharpe", "mean"),
            mean_test_edge_vs_spy=("test_edge_vs_spy_net_total_return", "mean"),
        )
        .sort_values(["mean_test_net_ann_return", "mean_test_edge_vs_spy"], ascending=[False, False])
        .reset_index(drop=True)
    )
    best_consistent.to_csv(outdir / "oos_consistency.csv", index=False)

    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite_dir": str(suite_dir),
        "splits": [x.strip() for x in str(args.splits).split(",") if x.strip()],
        "top_n_train": int(args.top_n_train),
        "top1_by_split": top1.to_dict(orient="records"),
        "best_consistent_candidates": best_consistent.head(15).to_dict(orient="records"),
        "insights": [
            f"Melhor candidato consistente em teste: {best_consistent.iloc[0]['candidate_id']} com media de {float(best_consistent.iloc[0]['mean_test_net_ann_return']):.4f} a.a. nos blocos OOS.",
            f"Topo do primeiro split: {top1.iloc[0]['candidate_id']} com teste de {float(top1.iloc[0]['test_net_ann_return']):.4f} a.a.",
        ],
        "artifacts": {
            "oos_candidate_results_csv": str(outdir / "oos_candidate_results.csv"),
            "oos_consistency_csv": str(outdir / "oos_consistency.csv"),
        },
    }
    _write_json(outdir / "summary.json", summary)
    print(json.dumps({"status": "ok", "outdir": str(outdir), "summary_json": str(outdir / "summary.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
