#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


SUITES: list[tuple[str, str]] = [
    ("historical_closure", "scripts/bench/validation/run_profit_historical_closure_suite.py"),
    ("pbo", "scripts/bench/validation/run_profit_pbo_suite.py"),
    ("execution_phase", "scripts/bench/validation/run_profit_execution_phase_suite.py"),
    ("universe_resilience", "scripts/bench/validation/run_profit_universe_resilience_suite.py"),
    ("bad_year_defense", "scripts/bench/validation/run_profit_bad_year_defense_suite.py"),
    ("u800_alpha", "scripts/bench/validation/run_profit_u800_alpha_suite.py"),
    ("marketmode_criticality", "scripts/bench/validation/run_profit_marketmode_criticality_suite.py"),
    ("meta_mode_selector", "scripts/bench/validation/run_profit_meta_mode_selector_suite.py"),
]


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main() -> None:
    ap = argparse.ArgumentParser(description="Reroda do zero as suítes pesadas do laboratório e reconsolida o board.")
    ap.add_argument("--outdir-root", default="results/validation/profit_hypothesis_lab_full_rerun")
    ap.add_argument("--publish-ops", action="store_true")
    args = ap.parse_args()

    outdir = (ROOT / args.outdir_root / _run_id()).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    suite_runs: list[dict[str, str]] = []

    for suite_id, rel_script in SUITES:
        script = (ROOT / rel_script).resolve()
        cmd = [sys.executable, str(script)]
        started_at = datetime.now(timezone.utc).isoformat()
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            capture_output=True,
        )
        log_path = outdir / f"{suite_id}.log"
        log_path.write_text(
            "\n".join(
                [
                    f"cmd={' '.join(cmd)}",
                    f"returncode={proc.returncode}",
                    "",
                    "--- stdout ---",
                    proc.stdout,
                    "",
                    "--- stderr ---",
                    proc.stderr,
                ]
            ),
            encoding="utf-8",
        )
        if proc.returncode != 0:
            raise SystemExit(f"suite {suite_id} falhou; ver {log_path}")
        suite_runs.append(
            {
                "suite_id": suite_id,
                "script": rel_script,
                "started_at_utc": started_at,
                "log_path": str(log_path),
            }
        )

    consolidate_cmd = [
        sys.executable,
        str((ROOT / "scripts/bench/validation/run_profit_hypothesis_lab_suite.py").resolve()),
    ]
    if args.publish_ops:
        consolidate_cmd.append("--publish-ops")
    consolidate = subprocess.run(
        consolidate_cmd,
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )
    consolidate_log = outdir / "consolidation.log"
    consolidate_log.write_text(
        "\n".join(
            [
                f"cmd={' '.join(consolidate_cmd)}",
                f"returncode={consolidate.returncode}",
                "",
                "--- stdout ---",
                consolidate.stdout,
                "",
                "--- stderr ---",
                consolidate.stderr,
            ]
        ),
        encoding="utf-8",
    )
    if consolidate.returncode != 0:
        raise SystemExit(f"consolidação falhou; ver {consolidate_log}")

    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "outdir": str(outdir),
        "suite_runs": suite_runs,
        "consolidation_log": str(consolidate_log),
        "consolidation_stdout": consolidate.stdout.strip(),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
