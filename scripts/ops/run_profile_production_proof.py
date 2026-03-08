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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _f(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="Orquestra prova 1-4 para perfil attack25.")
    ap.add_argument("--profile-dir", default="", help="Diretório do perfil (default: latest attack25)")
    ap.add_argument("--outdir", default="", help="Diretório de saída (default: <profile-dir>/production_proof_<run_id>)")
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()

    profile_dir = Path(args.profile_dir).resolve() if str(args.profile_dir).strip() else _latest_attack25_dir()
    outdir = Path(args.outdir).resolve() if str(args.outdir).strip() else (profile_dir / f"production_proof_{_run_id()}")
    outdir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    cmd_baselines = [
        py,
        str(ROOT / "scripts" / "bench" / "portfolio" / "run_baselines_for_profile.py"),
        "--profile-dir",
        str(profile_dir),
        "--outdir",
        str(outdir / "01_baselines"),
        "--seed",
        str(int(args.seed)),
    ]
    cmd_tail = [
        py,
        str(ROOT / "scripts" / "ops" / "run_profile_tail_risk_gate.py"),
        "--profile-dir",
        str(profile_dir),
        "--outdir",
        str(outdir / "02_tail_risk"),
        "--seed",
        str(int(args.seed)),
    ]
    cmd_cost = [
        py,
        str(ROOT / "scripts" / "ops" / "run_profile_cost_net.py"),
        "--profile-dir",
        str(profile_dir),
        "--outdir",
        str(outdir / "03_cost_net"),
    ]
    cmd_stability = [
        py,
        str(ROOT / "scripts" / "ops" / "run_profile_stability_gate.py"),
        "--profile-dir",
        str(profile_dir),
        "--outdir",
        str(outdir / "04_stability"),
    ]

    for cmd in [cmd_baselines, cmd_tail, cmd_cost, cmd_stability]:
        subprocess.run(cmd, cwd=ROOT, check=True)

    base = _load_json(outdir / "01_baselines" / "baselines_report.json")
    tail = _load_json(outdir / "02_tail_risk" / "tail_risk_gate_report.json")
    cost = _load_json(outdir / "03_cost_net" / "cost_net_report.json")
    stab = _load_json(outdir / "04_stability" / "stability_gate_report.json")

    cmp_eqw = (base.get("comparisons_vs_strategy") or {}).get("equal_weight_same_budget", {})
    cmp_mom = (base.get("comparisons_vs_strategy") or {}).get("momentum_same_budget", {})
    rnd = base.get("random_distribution", {})
    baseline_gate = (
        _f(cmp_eqw.get("outperformance_year_win_rate")) >= 0.70
        and _f(cmp_eqw.get("outperformance_semester_win_rate")) >= 0.60
        and _f(cmp_mom.get("outperformance_year_win_rate")) >= 0.60
        and _f(rnd.get("prob_strategy_beats_random_total")) >= 0.70
    )

    tail_gate = bool((tail.get("gate_result") or {}).get("passed", False))
    stability_gate = bool((stab.get("gate_result") or {}).get("passed", False))
    m = cost.get("metrics", {})
    net_us_total = _f((m.get("net_strategy_US") or {}).get("total_return"))
    net_br_total = _f((m.get("net_strategy_BR") or {}).get("total_return"))
    eqw_total = _f((m.get("eqw") or {}).get("total_return"))
    cost_gate = net_us_total > eqw_total and net_br_total > eqw_total

    overall = bool(baseline_gate and tail_gate and cost_gate and stability_gate)
    summary = {
        "status": "ok",
        "profile_dir": str(profile_dir),
        "overall_passed": overall,
        "gates": {
            "baseline_gate": bool(baseline_gate),
            "tail_gate": bool(tail_gate),
            "cost_gate": bool(cost_gate),
            "stability_gate": bool(stability_gate),
        },
        "key_metrics": {
            "prob_strategy_beats_random_total": _f(rnd.get("prob_strategy_beats_random_total")),
            "eqw_budget_year_win_rate": _f(cmp_eqw.get("outperformance_year_win_rate")),
            "eqw_budget_semester_win_rate": _f(cmp_eqw.get("outperformance_semester_win_rate")),
            "momentum_year_win_rate": _f(cmp_mom.get("outperformance_year_win_rate")),
            "net_us_total_return": net_us_total,
            "net_br_total_return": net_br_total,
            "eqw_total_return": eqw_total,
            "tail_gate_passed": bool(tail_gate),
            "stability_gate_passed": bool(stability_gate),
        },
        "paths": {
            "baselines_report": str(outdir / "01_baselines" / "baselines_report.json"),
            "tail_risk_report": str(outdir / "02_tail_risk" / "tail_risk_gate_report.json"),
            "cost_net_report": str(outdir / "03_cost_net" / "cost_net_report.json"),
            "stability_report": str(outdir / "04_stability" / "stability_gate_report.json"),
        },
    }
    summary_path = outdir / "production_proof_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    write_run_manifest(
        outdir=outdir,
        script="scripts/ops/run_profile_production_proof.py",
        params={"profile_dir": str(profile_dir), "seed": int(args.seed)},
        paths={"summary_json": str(summary_path)},
        gates={
            "baseline_gate": bool(baseline_gate),
            "tail_gate": bool(tail_gate),
            "cost_gate": bool(cost_gate),
            "stability_gate": bool(stability_gate),
            "overall_passed": bool(overall),
        },
    )
    print(json.dumps({"status": "ok", "outdir": str(outdir), "overall_passed": overall}))


if __name__ == "__main__":
    main()

