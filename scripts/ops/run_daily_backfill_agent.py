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
from scripts.ops.data_review_policy import DEFERRED_REVIEW_TICKERS
from scripts.ops.run_daily_ingestion_agent import REMOTE_FALLBACK_TICKERS
from scripts.ops.run_daily_data_quality_agent import (
    CRITICAL_TICKERS,
    _fresh_tolerance_days,
    _latest_csv_date,
    _load_universe_assets,
    _parse_date,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _collect_stale_core_and_critical(prices_dir: Path, reference_date) -> list[dict[str, Any]]:
    universe_assets = _load_universe_assets(
        [
            ROOT / "data" / "asset_groups_crypto_top_liquid_plus.csv",
            ROOT / "data" / "asset_groups_target_800_clean_plus.csv",
        ]
    )
    rows: list[dict[str, Any]] = []
    for path in sorted(prices_dir.glob("*.csv")):
        latest_date = _latest_csv_date(path)
        if latest_date is None:
            continue
        stale_days = max((reference_date - latest_date).days, 0)
        ticker = path.stem
        role = "peripheral"
        if ticker in CRITICAL_TICKERS:
            role = "critical"
        elif ticker in universe_assets:
            role = "core"
        if role not in {"critical", "core"}:
            continue
        if stale_days <= _fresh_tolerance_days(ticker):
            continue
        rows.append(
            {
                "ticker": ticker,
                "role": role,
                "latest_date": latest_date.isoformat(),
                "stale_days": stale_days,
                "deferred_review": ticker in DEFERRED_REVIEW_TICKERS or ticker not in REMOTE_FALLBACK_TICKERS,
            }
        )
    rows.sort(key=lambda row: (0 if row["role"] == "critical" else 1, -int(row["stale_days"]), str(row["ticker"])))
    return rows


def _chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


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
    ap = argparse.ArgumentParser(description="Reprocessa ativos críticos e do núcleo que ficaram atrasados após a ingestão principal.")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--ingestion-summary", default="results/ops/agents/daily_ingestion/latest_summary.json")
    ap.add_argument("--outdir-root", default="results/ops/agents/daily_backfill")
    ap.add_argument("--chunk-size", type=int, default=25)
    ap.add_argument("--max-targets", type=int, default=80)
    ap.add_argument("--cycle-run-id", default="")
    args = ap.parse_args()
    agent_run_id = utc_run_id()
    cycle_run_id = resolve_cycle_run_id(args.cycle_run_id)

    prices_dir = (ROOT / args.prices_dir).resolve()
    ingestion = _read_json((ROOT / args.ingestion_summary).resolve())
    reference_date = _parse_date(ingestion.get("max_latest_date")) or datetime.now(timezone.utc).date()
    stale_rows = _collect_stale_core_and_critical(prices_dir, reference_date)
    deferred_rows = [row for row in stale_rows if row["deferred_review"]]
    targeted_rows = [row for row in stale_rows if not row["deferred_review"]][: max(0, int(args.max_targets))]
    targeted_tickers = [str(row["ticker"]) for row in targeted_rows]

    retry_runs: list[dict[str, Any]] = []
    aggregate_provider_counts: dict[str, int] = {}
    updated_assets = 0
    refreshed_assets = 0
    unresolved_targets: list[str] = []

    for chunk in _chunked(targeted_tickers, max(1, int(args.chunk_size))):
        step = _run_step(
            [
                sys.executable,
                "scripts/ops/run_daily_ingestion_agent.py",
                "--tickers",
                ",".join(chunk),
                "--cycle-run-id",
                cycle_run_id,
            ],
            timeout_sec=1800.0,
        )
        retry_runs.append(step)
        parsed = step.get("parsed") if isinstance(step.get("parsed"), dict) else {}
        updated_assets += int(parsed.get("updated_assets") or 0)
        refreshed_assets += int(parsed.get("refreshed_assets") or 0)
        for provider, count in (parsed.get("provider_counts") or {}).items():
            aggregate_provider_counts[str(provider)] = aggregate_provider_counts.get(str(provider), 0) + int(count or 0)
        if not step["ok"]:
            unresolved_targets.extend(chunk)

    quality_step = _run_step([sys.executable, "scripts/ops/run_daily_data_quality_agent.py", "--cycle-run-id", cycle_run_id], timeout_sec=1200.0)
    snapshot_step = _run_step([sys.executable, "scripts/ops/build_site_finance_snapshot.py", "--cycle-run-id", cycle_run_id], timeout_sec=1200.0)
    quality_after = quality_step.get("parsed") if isinstance(quality_step.get("parsed"), dict) else {}

    alerts: list[dict[str, str]] = []
    if unresolved_targets:
        alerts.append(
            {
                "level": "warn",
                "code": "backfill_unresolved_targets",
                "message": f"{len(unresolved_targets)} ativos seguem sem refresh após a rodada de reconciliação.",
            }
        )
    if deferred_rows:
        alerts.append(
            {
                "level": "info",
                "code": "deferred_review_assets",
                "message": f"{len(deferred_rows)} ativos do núcleo ficaram fora do backfill automático e pedem revisão manual de universo.",
            }
        )
    if not quality_step["ok"] or not snapshot_step["ok"]:
        alerts.append(
            {
                "level": "fail",
                "code": "post_refresh_rebuild_failed",
                "message": "A reconciliação atualizou preços, mas não conseguiu fechar qualidade e snapshot depois disso.",
            }
        )

    status = "ok"
    if any(alert["level"] == "fail" for alert in alerts):
        status = "fail"
    elif any(alert["level"] == "warn" for alert in alerts):
        status = "warn"

    summary = attach_agent_guide(
        attach_cycle_context({
            "status": status,
            "generated_at_utc": utc_now_iso(),
            "reference_data_date": reference_date.isoformat(),
            "targeted_assets": len(targeted_tickers),
            "updated_assets": updated_assets,
            "refreshed_assets": refreshed_assets,
            "provider_counts": aggregate_provider_counts,
            "sample_targeted_tickers": targeted_tickers[:20],
            "deferred_review_tickers": [row["ticker"] for row in deferred_rows],
            "unresolved_targets": unresolved_targets[:20],
            "quality_after": {
                "status": quality_after.get("status"),
                "critical_stale_assets": quality_after.get("critical_stale_assets"),
                "core_stale_assets": quality_after.get("core_stale_assets"),
                "peripheral_stale_assets": quality_after.get("peripheral_stale_assets"),
            },
            "steps": {
                "retry_runs": retry_runs,
                "data_quality": quality_step,
                "site_snapshot": snapshot_step,
            },
            "alerts": alerts,
            "notes": [
                "Este agente não substitui a ingestão principal; ele só reconcilia o que ficou para trás.",
                "Ativos marcados para revisão ficam fora do retry automático para não reintroduzir ruído no núcleo.",
            ],
        }, cycle_run_id=cycle_run_id, agent_run_id=agent_run_id),
        "daily-backfill-agent",
    )

    outroot = (ROOT / args.outdir_root).resolve()
    ts_dir = outroot / agent_run_id
    _write_json(ts_dir / "summary.json", summary)
    _write_json(outroot / "latest_summary.json", summary)
    _write_json(outroot / "latest_backfill.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    if status == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
