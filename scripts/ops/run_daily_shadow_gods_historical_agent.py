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

from execution.shadow_gods import load_shadow_gods_profile  # noqa: E402
from execution.shadow_gods_historical import build_shadow_gods_historical_summary  # noqa: E402
from scripts.ops.agent_guides import attach_agent_guide  # noqa: E402
from scripts.ops.cycle_context import attach_cycle_context, resolve_cycle_run_id, utc_now_iso, utc_run_id  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _can_reuse(previous: dict[str, Any], *, as_of_date: str) -> bool:
    return str(previous.get("status") or "").strip().lower() == "ok" and str(previous.get("as_of_date") or "").strip() == str(as_of_date).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Agente diário do replay histórico dos shadow gods.")
    ap.add_argument("--config", default="config/shadow_gods_portfolios.json")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--outdir-root", default="results/ops/agents/daily_shadow_gods_historical")
    ap.add_argument("--start-date", default="2023-01-01")
    ap.add_argument("--end-date", default="2025-12-31")
    ap.add_argument("--cycle-run-id", default="")
    args = ap.parse_args()

    agent_run_id = utc_run_id()
    cycle_run_id = resolve_cycle_run_id(args.cycle_run_id)
    outroot = (ROOT / args.outdir_root).resolve()
    outdir = outroot / agent_run_id
    outdir.mkdir(parents=True, exist_ok=True)
    previous = _read_json(outroot / "latest_summary.json")

    if _can_reuse(previous, as_of_date=str(args.end_date)):
        summary = dict(previous)
        summary["generated_at_utc"] = utc_now_iso()
        summary["reuse_reason"] = "same_as_of_date"
        summary["reused_previous_run"] = True
        summary["agent_run_id"] = agent_run_id
    else:
        profile = load_shadow_gods_profile(ROOT / args.config)
        summary = build_shadow_gods_historical_summary(
            repo_root=ROOT,
            profile=profile,
            prices_dir=(ROOT / args.prices_dir).resolve(),
            start_date=str(args.start_date),
            end_date=str(args.end_date),
        )
        summary["config_path"] = str((ROOT / args.config).resolve())
        summary["prices_dir"] = str((ROOT / args.prices_dir).resolve())
        summary["reused_previous_run"] = False

    summary = attach_agent_guide(
        attach_cycle_context(summary, cycle_run_id=cycle_run_id, agent_run_id=agent_run_id),
        "daily-shadow-gods-historical-agent",
    )
    _write_json(outdir / "summary.json", summary)
    _write_json(outroot / "latest_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
