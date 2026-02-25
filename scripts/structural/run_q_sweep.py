#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.structural.run_manifest import write_run_manifest  # noqa: E402


PY = sys.executable
RUN_ID_RE = re.compile(r"^\d{8}T\d{6}Z$")


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_int_list(text: str, *, default: list[int], min_value: int = 1) -> list[int]:
    out: list[int] = []
    for token in str(text).split(","):
        t = token.strip()
        if not t:
            continue
        out.append(max(min_value, int(t)))
    vals = sorted(set(out))
    return vals or default


def _safe_float(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x if np.isfinite(x) else float("nan")


def _run(cmd: list[str], *, cwd: Path, timeout_sec: int) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=float(timeout_sec))
        return int(p.returncode), str(p.stdout or ""), str(p.stderr or "")
    except subprocess.TimeoutExpired as exc:
        out = str(exc.stdout or "")
        err = str(exc.stderr or "")
        return 124, out, err


def _list_run_dirs(base: Path) -> list[Path]:
    if not base.exists():
        return []
    out = []
    for d in base.iterdir():
        if not d.is_dir():
            continue
        if RUN_ID_RE.match(d.name):
            out.append(d)
    return sorted(out, key=lambda p: p.name)


def _pick_new_run_dir(base: Path, before: set[str]) -> Path:
    after = _list_run_dirs(base)
    new_dirs = [d for d in after if d.name not in before]
    if new_dirs:
        return new_dirs[-1]
    if after:
        return after[-1]
    raise FileNotFoundError(f"no run dir produced under {base}")


def _q_stats(run_dir: Path, official_window: int) -> dict[str, float]:
    p = run_dir / f"macro_timeseries_T{int(official_window)}.csv"
    if not p.exists():
        return {
            "n_rows": 0.0,
            "n_rows_sufficient": 0.0,
            "n_used_median": float("nan"),
            "q_min": float("nan"),
            "q_median": float("nan"),
            "q_max": float("nan"),
        }
    df = pd.read_csv(p)
    if df.empty:
        return {
            "n_rows": 0.0,
            "n_rows_sufficient": 0.0,
            "n_used_median": float("nan"),
            "q_min": float("nan"),
            "q_median": float("nan"),
            "q_max": float("nan"),
        }
    if "insufficient_universe" in df.columns:
        suff = df[~pd.to_numeric(df["insufficient_universe"], errors="coerce").fillna(1).astype(bool)].copy()
    else:
        suff = df.copy()
    if suff.empty:
        suff = df.copy()
    n_used = pd.to_numeric(suff.get("N_used"), errors="coerce").astype(float)
    q = pd.to_numeric(suff.get("Q"), errors="coerce").astype(float)
    if q.notna().sum() == 0:
        q = float(official_window) / n_used.clip(lower=1.0)
    return {
        "n_rows": float(df.shape[0]),
        "n_rows_sufficient": float(suff.shape[0]),
        "n_used_median": _safe_float(n_used.median()),
        "q_min": _safe_float(q.min()),
        "q_median": _safe_float(q.median()),
        "q_max": _safe_float(q.max()),
    }


def _best_row(curve: pd.DataFrame, *, horizon: int, ground_truth: str) -> dict[str, float]:
    x = curve[
        (curve["model"] == "score_only")
        & (curve["horizon_days"].astype(int) == int(horizon))
        & (curve["ground_truth"] == str(ground_truth))
    ].copy()
    if x.empty:
        return {
            "best_quantile": float("nan"),
            "best_threshold": float("nan"),
            "best_f1": float("nan"),
            "best_recall": float("nan"),
            "best_precision": float("nan"),
            "best_alert_rate": float("nan"),
            "best_lift_precision": float("nan"),
        }
    x["f1_num"] = pd.to_numeric(x["f1"], errors="coerce")
    x = x.sort_values(["f1_num", "quantile"], ascending=[False, True]).reset_index(drop=True)
    r = x.iloc[0]
    return {
        "best_quantile": _safe_float(r.get("quantile")),
        "best_threshold": _safe_float(r.get("threshold")),
        "best_f1": _safe_float(r.get("f1")),
        "best_recall": _safe_float(r.get("recall")),
        "best_precision": _safe_float(r.get("precision")),
        "best_alert_rate": _safe_float(r.get("alert_rate")),
        "best_lift_precision": _safe_float(r.get("lift_precision_vs_random")),
    }


def _objective(*, best_f1: float, best_lift_precision: float, q_median: float) -> float:
    f1 = _safe_float(best_f1)
    lift = _safe_float(best_lift_precision)
    qmed = _safe_float(q_median)
    score = -1.0 if not np.isfinite(f1) else float(f1)
    if np.isfinite(lift):
        score += 0.10 * max(0.0, float(lift) - 1.0)
    if np.isfinite(qmed):
        score -= 0.20 * max(0.0, 0.80 - float(qmed))
    return float(score)


def _to_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep Q (T/N) by varying official window and core universe cap.")
    ap.add_argument("--official-windows", type=str, default="120,252")
    ap.add_argument("--max-core-assets", type=str, default="180,250,320,470")
    ap.add_argument("--coverage-core", type=float, default=0.95)
    ap.add_argument("--coverage-window", type=float, default=0.98)
    ap.add_argument("--min-assets", type=int, default=25)
    ap.add_argument("--start", type=str, default="2018-01-01")
    ap.add_argument("--end", type=str, default="2026-02-12")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--noise-step", type=int, default=5)
    ap.add_argument("--bootstrap-block", type=int, default=10)
    ap.add_argument("--overlap-step", type=int, default=5)
    ap.add_argument("--horizons", type=str, default="5,10")
    ap.add_argument("--quantiles", type=str, default="0.75,0.80,0.85,0.90,0.95")
    ap.add_argument("--split-cutoffs", type=str, default="2023-12-31,2024-06-30")
    ap.add_argument("--drawdown-threshold", type=float, default=0.05)
    ap.add_argument("--random-iters", type=int, default=200)
    ap.add_argument("--rank-horizon", type=int, default=10)
    ap.add_argument("--max-combos", type=int, default=0, help="0 = run all combos")
    ap.add_argument("--timeout-sec-per-run", type=int, default=7200)
    ap.add_argument("--outdir", type=str, default="")
    args = ap.parse_args()

    windows = _parse_int_list(str(args.official_windows), default=[120, 252], min_value=20)
    max_assets_list = _parse_int_list(str(args.max_core_assets), default=[180, 250, 320, 470], min_value=1)

    combos: list[tuple[int, int]] = []
    for w in windows:
        for n in max_assets_list:
            combos.append((int(w), int(n)))
    if int(args.max_combos) > 0:
        combos = combos[: int(args.max_combos)]

    if str(args.outdir).strip():
        base_out = ROOT / str(args.outdir).strip()
    else:
        base_out = ROOT / "results" / f"q_sweep_{_ts()}"
    base_out.mkdir(parents=True, exist_ok=True)
    runs_root = base_out / "lab_runs"
    epi_root = base_out / "epistemic"
    runs_root.mkdir(parents=True, exist_ok=True)
    epi_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for i, (official_window, max_core) in enumerate(combos, start=1):
        tag = f"T{official_window}_N{max_core}"
        combo_runs = runs_root / tag
        combo_runs.mkdir(parents=True, exist_ok=True)
        before = {d.name for d in _list_run_dirs(combo_runs)}

        run_cmd = [
            PY,
            "scripts/lab/run_corr_macro_offline.py",
            "--apply-policy",
            "0",
            "--strict-checks",
            "0",
            "--update-release-pointer",
            "0",
            "--enable-structural-v1",
            "1",
            "--calibrate-exposure-grid",
            "0",
            "--apply-grid-best",
            "0",
            "--seed",
            str(int(args.seed)),
            "--start",
            str(args.start),
            "--end",
            str(args.end),
            "--coverage-core",
            str(float(args.coverage_core)),
            "--coverage-window",
            str(float(args.coverage_window)),
            "--min-assets",
            str(int(args.min_assets)),
            "--max-core-assets",
            str(int(max_core)),
            "--noise-step",
            str(int(args.noise_step)),
            "--bootstrap-block",
            str(int(args.bootstrap_block)),
            "--overlap-step",
            str(int(args.overlap_step)),
            "--official-window",
            str(int(official_window)),
            "--windows",
            str(int(official_window)),
            "--out-base",
            _to_rel(combo_runs),
        ]
        code, out, err = _run(run_cmd, cwd=ROOT, timeout_sec=int(args.timeout_sec_per_run))
        if code != 0:
            rows.append(
                {
                    "combo": tag,
                    "official_window": int(official_window),
                    "max_core_assets": int(max_core),
                    "status": "fail",
                    "error_code": int(code),
                    "error_tail": str((err or out)[-500:]),
                }
            )
            continue

        try:
            run_dir = _pick_new_run_dir(combo_runs, before=before)
        except FileNotFoundError as exc:
            rows.append(
                {
                    "combo": tag,
                    "official_window": int(official_window),
                    "max_core_assets": int(max_core),
                    "status": "fail",
                    "error_code": 99,
                    "error_tail": str(exc),
                }
            )
            continue

        epi_out_rel = _to_rel(epi_root / tag)
        epi_cmd = [
            PY,
            "scripts/structural/run_epistemic_diagnostics.py",
            "--run-dir",
            _to_rel(run_dir),
            "--outdir",
            epi_out_rel,
            "--horizons",
            str(args.horizons),
            "--quantiles",
            str(args.quantiles),
            "--split-cutoffs",
            str(args.split_cutoffs),
            "--drawdown-threshold",
            str(float(args.drawdown_threshold)),
            "--random-iters",
            str(int(args.random_iters)),
            "--seed",
            str(int(args.seed)),
        ]
        code2, out2, err2 = _run(epi_cmd, cwd=ROOT, timeout_sec=int(args.timeout_sec_per_run))
        if code2 != 0:
            rows.append(
                {
                    "combo": tag,
                    "official_window": int(official_window),
                    "max_core_assets": int(max_core),
                    "status": "fail",
                    "error_code": int(code2),
                    "error_tail": str((err2 or out2)[-500:]),
                    "run_dir": _to_rel(run_dir),
                }
            )
            continue

        epi_curve = pd.read_csv(ROOT / epi_out_rel / "epistemic_diagnostics_curve.csv")
        qstats = _q_stats(run_dir=run_dir, official_window=int(official_window))
        reg = _best_row(epi_curve, horizon=int(args.rank_horizon), ground_truth="regime_entry")
        dd = _best_row(epi_curve, horizon=int(args.rank_horizon), ground_truth="drawdown")

        obj = _objective(
            best_f1=reg["best_f1"],
            best_lift_precision=reg["best_lift_precision"],
            q_median=qstats["q_median"],
        )
        rows.append(
            {
                "combo": tag,
                "status": "ok",
                "order": int(i),
                "official_window": int(official_window),
                "max_core_assets": int(max_core),
                "run_dir": _to_rel(run_dir),
                "epistemic_outdir": epi_out_rel,
                "n_rows": qstats["n_rows"],
                "n_rows_sufficient": qstats["n_rows_sufficient"],
                "n_used_median": qstats["n_used_median"],
                "q_min": qstats["q_min"],
                "q_median": qstats["q_median"],
                "q_max": qstats["q_max"],
                "regime_entry_best_quantile_h": reg["best_quantile"],
                "regime_entry_best_threshold_h": reg["best_threshold"],
                "regime_entry_best_f1_h": reg["best_f1"],
                "regime_entry_best_recall_h": reg["best_recall"],
                "regime_entry_best_precision_h": reg["best_precision"],
                "regime_entry_best_alert_rate_h": reg["best_alert_rate"],
                "regime_entry_best_lift_precision_h": reg["best_lift_precision"],
                "drawdown_best_f1_h": dd["best_f1"],
                "drawdown_best_recall_h": dd["best_recall"],
                "drawdown_best_precision_h": dd["best_precision"],
                "drawdown_best_lift_precision_h": dd["best_lift_precision"],
                "objective_score": float(obj),
            }
        )

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(["status", "objective_score"], ascending=[True, False]).reset_index(drop=True)
    out_csv = base_out / "q_sweep_results.csv"
    table.to_csv(out_csv, index=False)

    best: dict[str, Any] = {}
    ok_rows = table[table["status"] == "ok"].copy() if not table.empty else pd.DataFrame()
    if not ok_rows.empty:
        ok_rows = ok_rows.sort_values("objective_score", ascending=False).reset_index(drop=True)
        best = ok_rows.iloc[0].to_dict()

    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "params": {
            "official_windows": windows,
            "max_core_assets": max_assets_list,
            "coverage_core": float(args.coverage_core),
            "coverage_window": float(args.coverage_window),
            "min_assets": int(args.min_assets),
            "start": str(args.start),
            "end": str(args.end),
            "seed": int(args.seed),
            "horizons": str(args.horizons),
            "quantiles": str(args.quantiles),
            "split_cutoffs": str(args.split_cutoffs),
            "drawdown_threshold": float(args.drawdown_threshold),
            "random_iters": int(args.random_iters),
            "rank_horizon": int(args.rank_horizon),
            "objective": "best_f1_regime_entry + 0.10*max(0,lift_precision-1) - 0.20*max(0,0.80-q_median)",
        },
        "counts": {
            "combos_total": int(len(combos)),
            "combos_ok": int((table["status"] == "ok").sum()) if not table.empty else 0,
            "combos_fail": int((table["status"] != "ok").sum()) if not table.empty else 0,
        },
        "files": {
            "results_csv": _to_rel(out_csv),
            "summary_json": _to_rel(base_out / "q_sweep_summary.json"),
        },
        "best": best,
    }
    summary_path = base_out / "q_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    write_run_manifest(
        base_out,
        script="scripts/structural/run_q_sweep.py",
        params=summary["params"],
        paths=summary["files"],
        gates={
            "results_table_written": bool(out_csv.exists()),
            "summary_written": bool(summary_path.exists()),
            "at_least_one_ok_combo": bool(not ok_rows.empty),
        },
        extra={"best": best},
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "outdir": _to_rel(base_out),
                "combos_total": int(len(combos)),
                "combos_ok": int(summary["counts"]["combos_ok"]),
                "best_combo": best.get("combo", ""),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
