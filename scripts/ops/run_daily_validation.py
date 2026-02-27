#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable


def _run(cmd: list[str], timeout_sec: float) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout_sec)
        msg = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        return proc.returncode, msg.strip()
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        err = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        msg = (out + ("\n" + err if err else "") + f"\n[timeout] {int(timeout_sec)}s").strip()
        return 124, msg


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily validation orchestrator for production gating.")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--max-assets", type=int, default=80)
    parser.add_argument("--run-id", type=str, default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    parser.add_argument("--outdir", type=str, default="results/ops/daily")
    parser.add_argument("--step-timeout-sec", type=float, default=900.0)
    parser.add_argument("--asset-timeout-sec", type=float, default=180.0)
    parser.add_argument("--max-points", type=int, default=1200)
    args = parser.parse_args()

    run_dir = ROOT / args.outdir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    steps = [
        [
            PY,
            "scripts/bench/validation/run_product_pipeline.py",
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
            "scripts/bench/validation/04_universe_mini.py",
            "--outdir",
            "results/validation/universe_mini_full",
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
            "scripts/bench/validation/10_calibrate_switch_targets.py",
            "--universe-dir",
            "results/validation/universe_mini_full",
            "--outdir",
            "results/validation/calibration_full",
        ],
        [
            PY,
            "scripts/bench/validation/11_metric_uncertainty_ci.py",
            "--outdir",
            "results/validation/uncertainty_full",
            "--seed",
            str(args.seed),
        ],
    ]

    logs: list[dict[str, Any]] = []
    ok = True
    for i, cmd in enumerate(steps, start=1):
        code, tail = _run(cmd, float(args.step_timeout_sec))
        logs.append({"step": i, "cmd": " ".join(cmd), "code": code, "tail": tail[-2000:]})
        if code != 0:
            ok = False
            break

    (run_dir / "execution_log.json").write_text(json.dumps(logs, indent=2, ensure_ascii=False), encoding="utf-8")

    global_status = _read_json(ROOT / "results/validation/STATUS.json")
    universe = _read_json(ROOT / "results/validation/universe_mini_full/universe_report.json")
    uncertainty = _read_json(ROOT / "results/validation/uncertainty_full/summary.json")
    validated = _read_json(ROOT / "results/validated/latest/summary.json")
    ablation = _read_json(ROOT / "results/validation/ablation/ablation_report.json")
    adaptive = _read_json(ROOT / "results/validation/adaptive_gates/summary.json")
    risk_utility = _read_json(ROOT / "results/validation/risk_utility/summary.json")
    adequacy = _read_json(ROOT / "results/validation/data_adequacy/summary.json")
    hist_metrics = _read_json(ROOT / "results/validation/historical_shifts/metrics.json")
    historical_status = _read_json(ROOT / "results/validation/historical_shifts/STATUS.json")
    realestate_offline = _read_json(ROOT / "results/validation/realestate_offline/summary.json")

    success_rate = 0.0
    counts = (universe.get("counts") or {})
    total = float(counts.get("total") or 0)
    if total > 0:
        success_rate = float((counts.get("ok") or 0) / total)

    protocol = _read_json(ROOT / "config/v1_protocol.json")
    goals = protocol.get("goals") or {}
    min_success = float(goals.get("min_universe_success_rate", 0.8))
    min_stability = float(goals.get("min_stability_score", 0.55))
    stability = float((global_status.get("scores") or {}).get("stability_score") or 0.0)

    checks = {
        "pipeline_ok": ok,
        "status_pass": str(global_status.get("status", "")).lower() == "pass",
        "universe_success_ok": success_rate >= min_success,
        "stability_ok": stability >= min_stability,
        "ablation_ok": str(ablation.get("status", "")).lower() == "ok",
        "adaptive_gates_ok": str(adaptive.get("status", "")).lower() == "ok",
        "risk_utility_ok": str(risk_utility.get("status", "")).lower() == "ok",
        "data_adequacy_ok": str(adequacy.get("status", "")).lower() == "ok",
        "pseudo_bifurcation_ok": not bool(hist_metrics.get("pseudo_bifurcation_flag", False)),
        "historical_shifts_ok": str(historical_status.get("status", "")).lower() in {"pass", "neutral"},
        "realestate_offline_ok": str(realestate_offline.get("status", "")).lower() == "ok",
    }
    status = "ok" if all(checks.values()) else "fail"

    summary = {
        "status": status,
        "run_id": args.run_id,
        "seed": args.seed,
        "max_assets": args.max_assets,
        "asset_timeout_sec": float(args.asset_timeout_sec),
        "max_points": int(args.max_points),
        "checks": checks,
        "metrics": {
            "stability_score": stability,
            "universe_success_rate": success_rate,
            "validated_ratio": validated.get("validated_ratio"),
            "conservative_ci_confidence": ((uncertainty.get("conservative_overall_metrics") or {}).get("mean_confidence") or {}).get("conservative_value"),
            "conservative_ci_quality": ((uncertainty.get("conservative_overall_metrics") or {}).get("mean_quality") or {}).get("conservative_value"),
            "ablation_status": ablation.get("status"),
            "adaptive_validated_count": ((adaptive.get("counts") or {}).get("validated")),
            "risk_utility_mean_dd_avoidance": risk_utility.get("mean_drawdown_avoidance"),
            "data_adequacy_ok_count": ((adequacy.get("counts") or {}).get("ok")),
            "pseudo_bifurcation_flag": bool(hist_metrics.get("pseudo_bifurcation_flag", False)),
            "historical_status": historical_status.get("status"),
            "realestate_assets_ok": realestate_offline.get("ok"),
            "realestate_assets_fail": realestate_offline.get("fail"),
        },
        "sources": {
            "status": "results/validation/STATUS.json",
            "universe_report": "results/validation/universe_mini_full/universe_report.json",
            "uncertainty": "results/validation/uncertainty_full/summary.json",
            "validated_summary": "results/validated/latest/summary.json",
            "realestate_offline": "results/validation/realestate_offline/summary.json",
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[daily_validation] status={status} "
        f"stability={stability:.3f} success_rate={success_rate:.3f} "
        f"validated_ratio={float(validated.get('validated_ratio') or 0.0):.3f}"
    )
    if status != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
