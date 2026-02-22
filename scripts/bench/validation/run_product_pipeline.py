#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PY = sys.executable


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, out.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run validation suite and build product-ready validated artifacts.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-assets", type=int, default=30)
    parser.add_argument("--asset-timeout-sec", type=float, default=180.0)
    parser.add_argument("--max-points", type=int, default=1200)
    args = parser.parse_args()

    steps = [
        [
            PY,
            "scripts/realestate/fetch_realestate_sources.py",
            "--outdir",
            "data/raw/realestate",
            "--bcb-start-year",
            "2000",
        ],
        [
            PY,
            "scripts/realestate/build_realestate_offline_core.py",
            "--normalized-dir",
            "data/realestate/normalized",
            "--raw-dir",
            "data/raw/realestate",
            "--outdir",
            "data/realestate/core",
            "--validation-outdir",
            "results/validation/realestate_offline",
            "--min-points",
            "60",
            "--seed",
            str(args.seed),
        ],
        [
            PY,
            "scripts/realestate/run_realestate_hmm.py",
            "--input-dir",
            "data/realestate/normalized",
            "--outdir",
            "results/realestate/hmm",
            "--states",
            "4",
            "--min-samples",
            "60",
        ],
        [
            PY,
            "scripts/bench/validation/18_hmm_fallback_sector_check.py",
            "--outdir",
            "results/validation/hmm_sector_check",
            "--max-per-sector",
            "20",
            "--min-samples",
            "120",
            "--states",
            "4",
            "--seed",
            str(args.seed),
        ],
        [
            PY,
            "scripts/bench/validation/16_data_adequacy_gate.py",
            "--outdir",
            "results/validation/data_adequacy",
            "--allow-partial",
        ],
        [PY, "scripts/bench/validation/01_sanity_run.py", "--outdir", "results/validation/sanity", "--seed", str(args.seed)],
        [
            PY,
            "scripts/bench/validation/02_robustness_sweep.py",
            "--dataset",
            "data/raw/finance/yfinance_daily/^VIX.csv",
            "--value-col",
            "close",
            "--date-col",
            "date",
            "--outdir",
            "results/validation/robustness",
            "--timeframe",
            "daily",
        ],
        [
            PY,
            "scripts/bench/validation/03_placebo_tests.py",
            "--dataset",
            "data/raw/finance/yfinance_daily/^VIX.csv",
            "--value-col",
            "close",
            "--date-col",
            "date",
            "--outdir",
            "results/validation/placebo",
            "--seed",
            str(args.seed),
            "--timeframe",
            "daily",
            "--include-phase-random",
        ],
        [
            PY,
            "scripts/bench/validation/04_universe_mini.py",
            "--outdir",
            "results/validation/universe_mini",
            "--seed",
            str(args.seed),
            "--max-assets",
            str(args.max_assets),
            "--asset-timeout-sec",
            str(args.asset_timeout_sec),
            "--max-points",
            str(args.max_points),
        ],
        [
            PY,
            "scripts/bench/validation/07_historical_regime_shifts.py",
            "--outdir",
            "results/validation/historical_shifts",
            "--seed",
            str(args.seed),
            "--calibrate",
        ],
        [
            PY,
            "scripts/bench/validation/09_synthetic_false_alarm_stress.py",
            "--outdir",
            "results/validation/synthetic_false_alarm_stress",
            "--seed",
            str(args.seed),
        ],
        [
            PY,
            "scripts/bench/validation/13_ablation_formal.py",
            "--outdir",
            "results/validation/ablation",
            "--seed",
            str(args.seed),
        ],
        [
            PY,
            "scripts/bench/validation/14_adaptive_status_gates.py",
            "--outdir",
            "results/validation/adaptive_gates",
        ],
        [
            PY,
            "scripts/bench/validation/15_risk_utility_metrics.py",
            "--outdir",
            "results/validation/risk_utility",
        ],
        [
            PY,
            "scripts/bench/validation/17_hybrid_ews_var_garch.py",
            "--outdir",
            "results/validation/hybrid_risk",
            "--seed",
            str(args.seed),
            "--max-assets",
            str(args.max_assets),
        ],
        [
            PY,
            "scripts/bench/validation/12_risk_truth_panel.py",
            "--root",
            "results/validation",
            "--out",
            "results/validation/risk_truth_panel.json",
        ],
        [PY, "scripts/bench/validation/05_status.py", "--root", "results/validation"],
        [
            PY,
            "scripts/bench/validation/00_build_validated_artifacts.py",
            "--outdir",
            "results/validated/latest",
            "--gate-config",
            "config/validation_gates.json",
        ],
    ]

    logs = []
    status = "ok"
    for i, cmd in enumerate(steps, start=1):
        code, out = _run(cmd)
        logs.append({"step": i, "cmd": " ".join(cmd), "code": code, "tail": out[-1500:]})
        if code != 0:
            status = "fail"
            break

    out_path = ROOT / "results" / "validated" / "latest" / "pipeline_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"status": status, "steps": logs}, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[product_pipeline] status={status} steps={len(logs)}/{len(steps)} summary={out_path}")
    if status != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
