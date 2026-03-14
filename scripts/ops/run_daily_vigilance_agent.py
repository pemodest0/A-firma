#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


def _parse_dt(text: Any) -> datetime | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _days_old(text: Any) -> float | None:
    dt = _parse_dt(text)
    if dt is None:
        return None
    now = datetime.now(timezone.utc)
    return max(0.0, (now - dt.astimezone(timezone.utc)).total_seconds() / 86400.0)


def _push_alert(alerts: list[dict[str, Any]], *, level: str, code: str, message: str) -> None:
    alerts.append({"level": str(level), "code": str(code), "message": str(message)})


def main() -> None:
    ap = argparse.ArgumentParser(description="Agente diario de vigilancia do motor e do site.")
    ap.add_argument("--operation-summary", default="results/ops/agents/daily_operation/latest_summary.json")
    ap.add_argument("--site-snapshot", default="results/ops/site_data/latest_site_snapshot.json")
    ap.add_argument("--profit-registry", default="results/ops/profit_research/latest_registry.json")
    ap.add_argument("--pbo-summary", default="results/validation/profit_pbo_suite/20260309T023026Z/summary.json")
    ap.add_argument("--outdir-root", default="results/ops/agents/daily_vigilance")
    ap.add_argument("--cycle-run-id", default="")
    args = ap.parse_args()
    agent_run_id = utc_run_id()
    cycle_run_id = resolve_cycle_run_id(args.cycle_run_id)

    operation = _read_json((ROOT / args.operation_summary).resolve())
    snapshot = _read_json((ROOT / args.site_snapshot).resolve())
    registry = _read_json((ROOT / args.profit_registry).resolve())
    pbo = _read_json((ROOT / args.pbo_summary).resolve())

    alerts: list[dict[str, Any]] = []
    op_age = _days_old(operation.get("generated_at_utc"))
    snapshot_date = str(snapshot.get("as_of_date") or "").strip()
    pbo_verdict = str(pbo.get("overall_verdict") or "").strip() or "desconhecido"

    if not operation:
        _push_alert(alerts, level="fail", code="operation_missing", message="Agente de operação sem resumo publicado.")
    elif op_age is not None and op_age > 2.0:
        _push_alert(alerts, level="warn", code="operation_stale", message="Resumo operacional está velho para uso diário.")

    if not snapshot:
        _push_alert(alerts, level="fail", code="snapshot_missing", message="Snapshot do site não foi gerado.")
    elif not snapshot_date:
        _push_alert(alerts, level="warn", code="snapshot_date_missing", message="Snapshot do site saiu sem data-base clara.")

    if not registry:
        _push_alert(alerts, level="fail", code="registry_missing", message="Registro de pesquisa está ausente.")

    if pbo_verdict not in {"robusto", "aceitavel"}:
        _push_alert(alerts, level="warn", code="pbo_soft", message="Teste de overfit não está no nível mais forte.")

    attack = operation.get("mode_attack", {}) if isinstance(operation.get("mode_attack"), dict) else {}
    main_mode = operation.get("mode_main", {}) if isinstance(operation.get("mode_main"), dict) else {}
    attack_dd = attack.get("net_max_drawdown")
    main_dd = main_mode.get("net_max_drawdown")

    try:
        if float(attack_dd) <= -0.75:
            _push_alert(alerts, level="warn", code="attack_deep_drawdown", message="Modo ataque continua muito agressivo no histórico.")
    except (TypeError, ValueError):
        _push_alert(alerts, level="warn", code="attack_missing_drawdown", message="Modo ataque sem leitura clara de drawdown.")

    try:
        if float(main_dd) <= -0.65:
            _push_alert(alerts, level="warn", code="main_deep_drawdown", message="Modo principal ainda tem tombo histórico pesado.")
    except (TypeError, ValueError):
        _push_alert(alerts, level="warn", code="main_missing_drawdown", message="Modo principal sem leitura clara de drawdown.")

    if not alerts:
        status = "ok"
    elif any(a["level"] == "fail" for a in alerts):
        status = "fail"
    else:
        status = "warn"

    summary = attach_agent_guide(attach_cycle_context({
        "status": status,
        "generated_at_utc": utc_now_iso(),
        "operation_age_days": op_age,
        "snapshot_as_of_date": snapshot_date,
        "research_rows_total": registry.get("rows_total"),
        "pbo_verdict": pbo_verdict,
        "alerts": alerts,
        "notes": [
            "Este agente não muda parâmetros do motor. Ele só vigia frescor, fragilidade e integridade.",
            "O foco é impedir que o site e a trilha operacional pareçam saudáveis quando os artefatos envelhecem ou falham.",
        ],
    }, cycle_run_id=cycle_run_id, agent_run_id=agent_run_id), "daily-vigilance-agent")

    outroot = (ROOT / args.outdir_root).resolve()
    ts_dir = outroot / agent_run_id
    _write_json(ts_dir / "summary.json", summary)
    _write_json(outroot / "latest_summary.json", summary)
    _write_json(outroot / "latest_vigilance.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
