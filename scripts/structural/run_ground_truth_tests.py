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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.structural.ground_truth import (
    build_event_label,
    build_regime_future_event_label,
    classification_report_binary,
    threshold_from_train,
)
from engine.structural.run_manifest import write_run_manifest


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _latest_lab_run() -> Path:
    base = ROOT / "results" / "lab_corr_macro"
    if not base.exists():
        raise FileNotFoundError(f"missing base dir: {base}")
    runs = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
    for d in runs:
        if (d / "diagnostics_structural_score_daily.csv").exists() and (d / "backtest_regime_T120.csv").exists():
            return d
    raise FileNotFoundError("no run with structural score + backtest file found")


def _parse_horizons(text: str) -> list[int]:
    out: list[int] = []
    for token in str(text).split(","):
        t = token.strip()
        if not t:
            continue
        out.append(int(t))
    out = sorted(set(int(max(1, x)) for x in out))
    return out or [5, 10, 20]


def _resolve_train_mask(dates: pd.Series, train_end: str) -> pd.Series:
    d = pd.to_datetime(dates, errors="coerce")
    if str(train_end).strip():
        cutoff = pd.Timestamp(str(train_end).strip())
        m = d <= cutoff
        if int(m.sum()) >= 20 and int((~m).sum()) >= 20:
            return pd.Series(m, index=dates.index)
    pos = int(max(20, min(len(d) - 1, int(0.7 * len(d)))))
    cutoff = pd.Timestamp(d.iloc[pos])
    return pd.Series(d <= cutoff, index=dates.index)


def _build_predictions(df: pd.DataFrame, score_thr: float) -> pd.DataFrame:
    out = df.copy()
    out["pred_score"] = ((pd.to_numeric(out["score"], errors="coerce") >= float(score_thr))).astype("Int64")
    out["pred_regime"] = out["regime"].astype(str).str.lower().isin({"stress", "transition"}).astype("Int64")
    out["pred_combined"] = ((out["pred_score"].fillna(0).astype(int) == 1) | (out["pred_regime"].fillna(0).astype(int) == 1)).astype("Int64")
    return out


def _evaluate_one_horizon(df: pd.DataFrame, *, horizon: int, dd_threshold: float, score_thr: float, train_mask: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
    x = df.copy()
    x["y_event_drawdown"] = build_event_label(equity=x["benchmark_equity"], horizon_days=int(horizon), dd_threshold=float(dd_threshold))
    x["y_event_regime_entry"] = build_regime_future_event_label(
        regime=x["regime"],
        horizon_days=int(horizon),
        target_regimes={"stress", "transition"},
    )

    test_mask = ~train_mask
    valid_drawdown = test_mask & x["y_event_drawdown"].notna()
    valid_regime_entry = test_mask & x["y_event_regime_entry"].notna()

    metrics_drawdown = {
        "score_only": classification_report_binary(x.loc[valid_drawdown, "y_event_drawdown"], x.loc[valid_drawdown, "pred_score"]),
        "regime_only": classification_report_binary(x.loc[valid_drawdown, "y_event_drawdown"], x.loc[valid_drawdown, "pred_regime"]),
        "combined_or": classification_report_binary(x.loc[valid_drawdown, "y_event_drawdown"], x.loc[valid_drawdown, "pred_combined"]),
    }
    metrics_regime_entry = {
        "score_only": classification_report_binary(x.loc[valid_regime_entry, "y_event_regime_entry"], x.loc[valid_regime_entry, "pred_score"]),
        "regime_only": classification_report_binary(x.loc[valid_regime_entry, "y_event_regime_entry"], x.loc[valid_regime_entry, "pred_regime"]),
        "combined_or": classification_report_binary(x.loc[valid_regime_entry, "y_event_regime_entry"], x.loc[valid_regime_entry, "pred_combined"]),
    }

    summary = {
        "horizon_days": int(horizon),
        "dd_threshold": float(dd_threshold),
        "score_threshold": float(score_thr),
        "test_metrics": {
            "ground_truth_drawdown": metrics_drawdown,
            "ground_truth_regime_entry": metrics_regime_entry,
        },
        "counts": {
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "test_rows_labeled_drawdown": int(valid_drawdown.sum()),
            "test_rows_labeled_regime_entry": int(valid_regime_entry.sum()),
        },
    }
    return x, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Ground truth suite for structural diagnostics.")
    ap.add_argument("--run-dir", type=str, default="")
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--horizons", type=str, default="5,10,20")
    ap.add_argument("--drawdown-threshold", type=float, default=0.05)
    ap.add_argument("--score-quantile", type=float, default=0.85)
    ap.add_argument("--train-end", type=str, default="2024-12-31")
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if str(args.run_dir).strip() else _latest_lab_run()
    outdir = (ROOT / str(args.outdir).strip()) if str(args.outdir).strip() else (ROOT / "results" / f"structural_ground_truth_{_run_id()}")
    outdir.mkdir(parents=True, exist_ok=True)

    score_path = run_dir / "diagnostics_structural_score_daily.csv"
    bt_path = run_dir / "backtest_regime_T120.csv"
    if not score_path.exists():
        raise SystemExit(f"missing: {score_path}")
    if not bt_path.exists():
        raise SystemExit(f"missing: {bt_path}")

    score_df = pd.read_csv(score_path)
    bt_df = pd.read_csv(bt_path)
    need_score = {"date", "score", "phi", "deff", "ac1_phi", "forman_mean", "flags_valid"}
    need_bt = {"date", "regime", "benchmark_equity"}
    if not need_score.issubset(set(score_df.columns)):
        miss = sorted(need_score - set(score_df.columns))
        raise SystemExit(f"missing columns in score file: {miss}")
    if not need_bt.issubset(set(bt_df.columns)):
        miss = sorted(need_bt - set(bt_df.columns))
        raise SystemExit(f"missing columns in backtest file: {miss}")

    score_df["date"] = pd.to_datetime(score_df["date"], errors="coerce")
    bt_df["date"] = pd.to_datetime(bt_df["date"], errors="coerce")
    bt_df["benchmark_equity"] = pd.to_numeric(bt_df["benchmark_equity"], errors="coerce")

    df = (
        score_df.merge(bt_df[["date", "regime", "benchmark_equity"]], on="date", how="inner")
        .dropna(subset=["date", "benchmark_equity"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    if df.empty:
        raise SystemExit("empty merged dataset")

    train_mask = _resolve_train_mask(df["date"], train_end=str(args.train_end))
    score_thr = threshold_from_train(
        score=df["score"],
        train_mask=train_mask & (pd.to_numeric(df["flags_valid"], errors="coerce").fillna(0).astype(int) == 1),
        q=float(args.score_quantile),
    )
    pred_base = _build_predictions(df, score_thr=score_thr)

    horizon_summaries: list[dict[str, Any]] = []
    out_daily = pred_base[["date", "score", "phi", "deff", "ac1_phi", "forman_mean", "flags_valid", "regime", "pred_score", "pred_regime", "pred_combined"]].copy()
    out_daily["date"] = pd.DatetimeIndex(out_daily["date"]).strftime("%Y-%m-%d")

    horizons = _parse_horizons(args.horizons)
    for h in horizons:
        one_df, one_summary = _evaluate_one_horizon(
            pred_base,
            horizon=int(h),
            dd_threshold=float(args.drawdown_threshold),
            score_thr=float(score_thr),
            train_mask=train_mask,
        )
        horizon_summaries.append(one_summary)
        out_daily[f"y_event_drawdown_h{int(h)}"] = one_df["y_event_drawdown"]
        out_daily[f"y_event_regime_entry_h{int(h)}"] = one_df["y_event_regime_entry"]

    out_daily.to_csv(outdir / "ground_truth_daily.csv", index=False)

    summary = {
        "status": "ok",
        "run_dir": str(run_dir),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "params": {
            "horizons": horizons,
            "drawdown_threshold": float(args.drawdown_threshold),
            "score_quantile": float(args.score_quantile),
            "score_threshold": float(score_thr),
            "train_end": str(args.train_end),
        },
        "counts": {
            "rows_merged": int(df.shape[0]),
            "flags_valid_rows": int(pd.to_numeric(df["flags_valid"], errors="coerce").fillna(0).astype(int).sum()),
        },
        "horizon_summaries": horizon_summaries,
        "files": {
            "ground_truth_daily_csv": str(outdir / "ground_truth_daily.csv"),
            "ground_truth_summary_json": str(outdir / "ground_truth_summary.json"),
        },
    }
    (outdir / "ground_truth_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    latest_dir = ROOT / "results" / "ops" / "ai_knowledge"
    latest_dir.mkdir(parents=True, exist_ok=True)
    (latest_dir / "latest_ground_truth.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "generated_at_utc": summary["generated_at_utc"],
                "source_run_dir": str(run_dir),
                "summary_path": str(outdir / "ground_truth_summary.json"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    write_run_manifest(
        outdir,
        script="scripts/structural/run_ground_truth_tests.py",
        params={
            "run_dir": str(run_dir),
            "horizons": horizons,
            "drawdown_threshold": float(args.drawdown_threshold),
            "score_quantile": float(args.score_quantile),
            "train_end": str(args.train_end),
        },
        paths={
            "ground_truth_daily_csv": str(outdir / "ground_truth_daily.csv"),
            "ground_truth_summary_json": str(outdir / "ground_truth_summary.json"),
            "latest_ground_truth_json": str(latest_dir / "latest_ground_truth.json"),
        },
        gates={
            "score_threshold_finite": bool(np.isfinite(score_thr)),
            "merged_rows_nonempty": bool(df.shape[0] > 0),
            "summary_written": bool((outdir / "ground_truth_summary.json").exists()),
        },
    )

    print(json.dumps({"status": "ok", "outdir": str(outdir), "score_threshold": float(score_thr)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
