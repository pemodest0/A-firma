#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution.live_ops import append_execution_ledger, reconcile_execution, write_json  # noqa: E402
from execution.live_tax import build_live_tax_summary, write_json as write_tax_json, write_monthly_summary_csv  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Reconcilia o plano emitido com a execucao real local.")
    ap.add_argument("--plan-json", default="results/ops/execution_live/latest_execution_plan.json")
    ap.add_argument("--execution-report-json", default="data/live_execution/execution_report.json")
    ap.add_argument("--outdir-root", default="results/ops/execution_live")
    args = ap.parse_args()

    plan = _read_json((ROOT / args.plan_json).resolve())
    report = _read_json((ROOT / args.execution_report_json).resolve())
    if not plan:
        raise SystemExit("missing live execution plan")
    if not report:
        raise SystemExit("missing execution report")

    reconciled = reconcile_execution(plan, report)
    run_id = str(plan.get("run_id") or "latest")
    outdir = (ROOT / args.outdir_root / run_id).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    write_json(outdir / "reconciliation.json", reconciled)
    write_json((ROOT / args.outdir_root / "latest_reconciliation.json").resolve(), reconciled)

    ledger_rows = []
    for row in reconciled.get("reconciled_rows", []):
        if not isinstance(row, dict):
            continue
        ledger_rows.append(
            {
                "plan_run_id": str(reconciled.get("plan_run_id") or ""),
                **row,
            }
        )
    if ledger_rows:
        append_execution_ledger(ROOT / args.outdir_root / "execution_ledger.csv", ledger_rows)
    tax_summary = build_live_tax_summary(
        ledger_csv=ROOT / args.outdir_root / "execution_ledger.csv",
        net_assumptions_config=ROOT / "config" / "profit_net_assumptions.json",
    )
    write_tax_json(outdir / "tax_summary.json", tax_summary)
    write_tax_json((ROOT / args.outdir_root / "latest_tax_summary.json").resolve(), tax_summary)
    write_monthly_summary_csv(outdir / "tax_monthly_summary.csv", tax_summary.get("monthly_rows", []))
    print(json.dumps(reconciled, ensure_ascii=False))


if __name__ == "__main__":
    main()
