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

from scripts.ops.agent_guides import attach_agent_guide
from scripts.ops.cycle_context import attach_cycle_context, resolve_cycle_run_id, utc_now_iso, utc_run_id


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


def _parse_dt(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _run_step(cmd: list[str], *, timeout_sec: float) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    stdout = (proc.stdout or "").strip()
    parsed = {}
    if stdout:
        last_line = stdout.splitlines()[-1]
        try:
            parsed = json.loads(last_line)
        except json.JSONDecodeError:
            parsed = {}
    return {
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "stdout": stdout[-2000:],
        "stderr": (proc.stderr or "")[-2000:],
        "ok": proc.returncode == 0,
        "parsed": parsed,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Watchdog diário dos agentes operacionais com retry controlado.")
    ap.add_argument("--outdir-root", default="results/ops/agents/daily_watchdog")
    ap.add_argument("--cycle-run-id", default="")
    args = ap.parse_args()
    agent_run_id = utc_run_id()
    cycle_run_id = resolve_cycle_run_id(args.cycle_run_id)

    summaries = {
        "daily_ingestion": _read_json(ROOT / "results/ops/agents/daily_ingestion/latest_summary.json"),
        "daily_backfill": _read_json(ROOT / "results/ops/agents/daily_backfill/latest_summary.json"),
        "daily_operation": _read_json(ROOT / "results/ops/agents/daily_operation/latest_summary.json"),
        "daily_shadow_gods": _read_json(ROOT / "results/ops/agents/daily_shadow_gods/latest_summary.json"),
        "daily_vigilance": _read_json(ROOT / "results/ops/agents/daily_vigilance/latest_summary.json"),
        "daily_data_quality": _read_json(ROOT / "results/ops/agents/daily_data_quality/latest_summary.json"),
        "daily_publish": _read_json(ROOT / "results/ops/agents/daily_publish/latest_summary.json"),
        "daily_smoke_test": _read_json(ROOT / "results/ops/agents/daily_smoke_test/latest_summary.json"),
    }

    retries: list[dict[str, Any]] = []
    checks: list[dict[str, str]] = []

    ingestion = summaries["daily_ingestion"]
    data_quality = summaries["daily_data_quality"]
    publish = summaries["daily_publish"]
    vigilance = summaries["daily_vigilance"]
    smoke = summaries["daily_smoke_test"]

    if str(ingestion.get("status") or "").lower() == "fail":
        retries.append(_run_step([sys.executable, "scripts/ops/run_daily_ingestion_agent.py", "--cycle-run-id", cycle_run_id], timeout_sec=1800.0))
    if int(data_quality.get("critical_stale_assets") or 0) > 0 or str(summaries["daily_backfill"].get("status") or "").lower() == "fail":
        retries.append(_run_step([sys.executable, "scripts/ops/run_daily_backfill_agent.py", "--cycle-run-id", cycle_run_id], timeout_sec=2400.0))
    operation_dt = _parse_dt(summaries["daily_operation"].get("generated_at_utc"))
    vigilance_dt = _parse_dt(vigilance.get("generated_at_utc"))
    if not vigilance or str(vigilance.get("status") or "").lower() == "fail" or (
        operation_dt is not None and vigilance_dt is not None and vigilance_dt < operation_dt
    ):
        retries.append(_run_step([sys.executable, "scripts/ops/run_daily_vigilance_agent.py", "--cycle-run-id", cycle_run_id], timeout_sec=1200.0))
    if not summaries["daily_shadow_gods"] or str(summaries["daily_shadow_gods"].get("status") or "").lower() == "fail":
        retries.append(_run_step([sys.executable, "scripts/ops/run_daily_shadow_gods_agent.py", "--cycle-run-id", cycle_run_id], timeout_sec=1200.0))
    if not publish or str(publish.get("status") or "").lower() == "fail":
        retries.append(_run_step(["bash", "scripts/ops/run_daily_publish_site.sh"], timeout_sec=7200.0))
    if str(smoke.get("status") or "").lower() == "fail":
        retries.append(_run_step([sys.executable, "scripts/ops/run_daily_operation_agent.py", "--cycle-run-id", cycle_run_id], timeout_sec=2400.0))
        retries.append(_run_step([sys.executable, "scripts/ops/run_daily_data_quality_agent.py", "--cycle-run-id", cycle_run_id], timeout_sec=1200.0))
        retries.append(_run_step([sys.executable, "scripts/ops/build_site_finance_snapshot.py", "--cycle-run-id", cycle_run_id], timeout_sec=1200.0))
        retries.append(_run_step([sys.executable, "scripts/ops/run_daily_smoke_test_agent.py", "--cycle-run-id", cycle_run_id], timeout_sec=1200.0))

    refreshed = {
        "daily_ingestion": _read_json(ROOT / "results/ops/agents/daily_ingestion/latest_summary.json"),
        "daily_backfill": _read_json(ROOT / "results/ops/agents/daily_backfill/latest_summary.json"),
        "daily_operation": _read_json(ROOT / "results/ops/agents/daily_operation/latest_summary.json"),
        "daily_shadow_gods": _read_json(ROOT / "results/ops/agents/daily_shadow_gods/latest_summary.json"),
        "daily_vigilance": _read_json(ROOT / "results/ops/agents/daily_vigilance/latest_summary.json"),
        "daily_data_quality": _read_json(ROOT / "results/ops/agents/daily_data_quality/latest_summary.json"),
        "daily_publish": _read_json(ROOT / "results/ops/agents/daily_publish/latest_summary.json"),
        "daily_smoke_test": _read_json(ROOT / "results/ops/agents/daily_smoke_test/latest_summary.json"),
    }

    if int((refreshed["daily_data_quality"].get("critical_stale_assets") or 0)) > 0:
        checks.append({"level": "fail", "code": "critical_assets_still_stale", "message": "O watchdog não conseguiu zerar os ativos críticos atrasados."})
    if str(refreshed["daily_smoke_test"].get("status") or "").lower() == "fail":
        checks.append({"level": "fail", "code": "smoke_still_failing", "message": "O smoke test continuou falhando após o retry controlado."})
    if str(refreshed["daily_ingestion"].get("status") or "").lower() == "fail":
        checks.append({"level": "fail", "code": "ingestion_still_failing", "message": "A ingestão diária continuou falhando após o retry."})
    if str(refreshed["daily_publish"].get("status") or "").lower() == "fail" or not refreshed["daily_publish"]:
        checks.append({"level": "fail", "code": "publish_still_failing", "message": "O publish diário continuou falhando ou sem resumo."})
    if str(refreshed["daily_shadow_gods"].get("status") or "").lower() == "fail" or not refreshed["daily_shadow_gods"]:
        checks.append({"level": "fail", "code": "shadow_gods_still_failing", "message": "O agente dos shadow gods continuou falhando após o retry."})
    refreshed_operation_dt = _parse_dt(refreshed["daily_operation"].get("generated_at_utc"))
    refreshed_vigilance_dt = _parse_dt(refreshed["daily_vigilance"].get("generated_at_utc"))
    if refreshed_operation_dt is not None and refreshed_vigilance_dt is not None and refreshed_vigilance_dt < refreshed_operation_dt:
        checks.append({"level": "warn", "code": "vigilance_still_older_than_operation", "message": "A vigilância continuou mais velha que a operação após o retry."})
    if int((refreshed["daily_data_quality"].get("core_stale_assets") or 0)) > 0:
        checks.append({"level": "warn", "code": "core_still_stale", "message": "O watchdog resolveu o crítico, mas o núcleo ainda tem ativos atrasados."})

    status = "ok"
    if any(item["level"] == "fail" for item in checks):
        status = "fail"
    elif any(item["level"] == "warn" for item in checks):
        status = "warn"

    summary = attach_agent_guide(
        attach_cycle_context({
            "status": status,
            "generated_at_utc": utc_now_iso(),
            "retries_attempted": len(retries),
            "retries": retries,
            "checks": checks,
            "notes": [
                "O watchdog só faz retry controlado. Ele não substitui o papel dos agentes principais.",
                "Se a falha persistir depois do retry, o estado fica explícito como fail.",
            ],
        }, cycle_run_id=cycle_run_id, agent_run_id=agent_run_id),
        "daily-watchdog-agent",
    )

    outroot = (ROOT / args.outdir_root).resolve()
    ts_dir = outroot / agent_run_id
    _write_json(ts_dir / "summary.json", summary)
    _write_json(outroot / "latest_summary.json", summary)
    _write_json(outroot / "latest_watchdog.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    if status == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
