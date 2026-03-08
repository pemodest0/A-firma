#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.portfolio.risk_gates import (  # noqa: E402
    TailRiskThresholds,
    apply_tail_gate,
    evaluate_tail_risk,
)
from engine.structural.run_manifest import write_run_manifest  # noqa: E402


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _latest_attack25_dir() -> Path:
    base = ROOT / "results" / "portfolio_sim"
    runs = sorted([p for p in base.iterdir() if p.is_dir() and "ultra_return_compact_attack25" in p.name], key=lambda p: p.name, reverse=True)
    for run in runs:
        if (run / "monthly_systematic_eval.csv").exists():
            return run
    raise FileNotFoundError("attack25 profile not found")


def main() -> None:
    ap = argparse.ArgumentParser(description="Avalia gate formal de cauda para um perfil mensal.")
    ap.add_argument("--profile-dir", default="", help="Diretório do perfil (default: latest attack25)")
    ap.add_argument("--outdir", default="", help="Diretório de saída (default: <profile-dir>/tail_risk_gate_<run_id>)")
    ap.add_argument("--n-paths", type=int, default=10000)
    ap.add_argument("--block-len", type=int, default=3)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--ruin-threshold", type=float, default=-0.50)
    ap.add_argument("--max-drawdown-floor", type=float, default=-0.35)
    ap.add_argument("--cvar95-floor", type=float, default=-0.12)
    ap.add_argument("--max-prob-ruin", type=float, default=0.05)
    ap.add_argument("--max-prob-dd50", type=float, default=0.05)
    ap.add_argument("--min-prob-positive", type=float, default=0.60)
    args = ap.parse_args()

    profile_dir = Path(args.profile_dir).resolve() if str(args.profile_dir).strip() else _latest_attack25_dir()
    monthly_csv = profile_dir / "monthly_systematic_eval.csv"
    if not monthly_csv.exists():
        raise FileNotFoundError(f"missing monthly file: {monthly_csv}")
    outdir = Path(args.outdir).resolve() if str(args.outdir).strip() else (profile_dir / f"tail_risk_gate_{_run_id()}")
    outdir.mkdir(parents=True, exist_ok=True)

    monthly = pd.read_csv(monthly_csv)
    ret = pd.to_numeric(monthly.get("ret"), errors="coerce").dropna().astype(float)
    metrics = evaluate_tail_risk(
        ret,
        n_paths=int(args.n_paths),
        block_len=int(args.block_len),
        seed=int(args.seed),
        alpha=0.95,
        ruin_threshold=float(args.ruin_threshold),
    )
    thresholds = TailRiskThresholds(
        max_drawdown_floor=float(args.max_drawdown_floor),
        cvar95_floor=float(args.cvar95_floor),
        max_prob_total_below_ruin=float(args.max_prob_ruin),
        max_prob_dd_worse_50=float(args.max_prob_dd50),
        min_prob_total_positive=float(args.min_prob_positive),
    )
    gate = apply_tail_gate(metrics, thresholds)

    payload = {
        "status": "ok",
        "profile_dir": str(profile_dir),
        "monthly_csv": str(monthly_csv),
        "tail_metrics": metrics,
        "gate_result": gate,
    }
    report_path = outdir / "tail_risk_gate_report.json"
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    write_run_manifest(
        outdir=outdir,
        script="scripts/ops/run_profile_tail_risk_gate.py",
        params={
            "profile_dir": str(profile_dir),
            "n_paths": int(args.n_paths),
            "block_len": int(args.block_len),
            "seed": int(args.seed),
            "ruin_threshold": float(args.ruin_threshold),
            "max_drawdown_floor": float(args.max_drawdown_floor),
            "cvar95_floor": float(args.cvar95_floor),
            "max_prob_ruin": float(args.max_prob_ruin),
            "max_prob_dd50": float(args.max_prob_dd50),
            "min_prob_positive": float(args.min_prob_positive),
        },
        paths={"report_json": str(report_path)},
        gates={"tail_gate_passed": bool(gate.get("passed", False)), "report_created": report_path.exists()},
    )

    print(json.dumps({"status": "ok", "outdir": str(outdir), "passed": bool(gate.get("passed", False))}))


if __name__ == "__main__":
    main()
