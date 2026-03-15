#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

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


def _parse_date(text: Any):
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw).date()
    except ValueError:
        return None


def _parse_dt(text: Any):
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _push(checks: list[dict[str, Any]], *, level: str, code: str, message: str) -> None:
    checks.append({"level": str(level), "code": str(code), "message": str(message)})


def _http_check(base_url: str, path: str) -> dict[str, Any]:
    url = base_url.rstrip("/") + path
    try:
        with urlopen(url, timeout=20) as response:  # noqa: S310
            code = int(getattr(response, "status", 200))
            return {"url": url, "status_code": code, "ok": 200 <= code < 400}
    except URLError as exc:
        return {"url": url, "status_code": None, "ok": False, "error": str(exc)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke test diário de artefatos operacionais e snapshot do site.")
    ap.add_argument("--operation-summary", default="results/ops/agents/daily_operation/latest_summary.json")
    ap.add_argument("--shadow-gods-summary", default="results/ops/agents/daily_shadow_gods/latest_summary.json")
    ap.add_argument("--quality-summary", default="results/ops/agents/daily_data_quality/latest_summary.json")
    ap.add_argument("--vigilance-summary", default="results/ops/agents/daily_vigilance/latest_summary.json")
    ap.add_argument("--publish-summary", default="results/ops/agents/daily_publish/latest_summary.json")
    ap.add_argument("--site-snapshot", default="results/ops/site_data/latest_site_snapshot.json")
    ap.add_argument("--public-site-snapshot", default="website-ui/public/data/site/latest_site_snapshot.json")
    ap.add_argument("--profit-registry", default="results/ops/profit_research/latest_registry.json")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--outdir-root", default="results/ops/agents/daily_smoke_test")
    ap.add_argument("--cycle-run-id", default="")
    args = ap.parse_args()
    agent_run_id = utc_run_id()
    cycle_run_id = resolve_cycle_run_id(args.cycle_run_id)

    operation = _read_json((ROOT / args.operation_summary).resolve())
    shadow_gods = _read_json((ROOT / args.shadow_gods_summary).resolve())
    quality = _read_json((ROOT / args.quality_summary).resolve())
    vigilance = _read_json((ROOT / args.vigilance_summary).resolve())
    publish = _read_json((ROOT / args.publish_summary).resolve())
    snapshot = _read_json((ROOT / args.site_snapshot).resolve())
    public_snapshot = _read_json((ROOT / args.public_site_snapshot).resolve())
    registry = _read_json((ROOT / args.profit_registry).resolve())

    checks: list[dict[str, Any]] = []
    finance_last = str((snapshot.get("finance") or {}).get("data_last_date") or snapshot.get("as_of_date") or "").strip()
    attack = operation.get("mode_attack") if isinstance(operation.get("mode_attack"), dict) else {}
    attack_latest = str(attack.get("latest_date") or "").strip()
    op_candidate = str(attack.get("candidate_id") or "").strip()
    reg_candidate = str(
        ((registry.get("official_attack_candidate") or {}).get("candidate_id"))
        or ((registry.get("top_candidate") or {}).get("candidate_id"))
        or ""
    ).strip()

    if not snapshot:
        _push(checks, level="fail", code="snapshot_missing", message="Snapshot principal do site não existe.")
    if not public_snapshot:
        _push(checks, level="fail", code="public_snapshot_missing", message="Snapshot público do site não existe.")
    if snapshot and public_snapshot:
        if snapshot.get("as_of_date") != public_snapshot.get("as_of_date"):
            _push(checks, level="fail", code="snapshot_as_of_mismatch", message="Snapshot interno e público divergiram na data-base.")
        if (snapshot.get("data_quality") or {}).get("quality_core_stale_assets") != (public_snapshot.get("data_quality") or {}).get("quality_core_stale_assets"):
            _push(checks, level="warn", code="snapshot_quality_mismatch", message="Snapshot interno e público divergiram na leitura de qualidade.")
    if not operation:
        _push(checks, level="fail", code="operation_missing", message="Resumo operacional ausente.")
    if not shadow_gods:
        _push(checks, level="fail", code="shadow_gods_missing", message="Resumo diário dos shadow gods ausente.")
    if not vigilance:
        _push(checks, level="fail", code="vigilance_missing", message="Resumo de vigilância ausente.")
    if not publish:
        _push(checks, level="fail", code="publish_missing", message="Resumo do publish diário ausente.")
    elif str(publish.get("status") or "").lower() != "ok":
        _push(checks, level="fail", code="publish_failed", message="O publish diário não fechou com status ok.")
    cycle_values = {
        "operation": str(operation.get("cycle_run_id") or "").strip(),
        "shadow_gods": str(shadow_gods.get("cycle_run_id") or "").strip(),
        "quality": str(quality.get("cycle_run_id") or "").strip(),
        "vigilance": str(vigilance.get("cycle_run_id") or "").strip(),
        "publish": str(publish.get("cycle_run_id") or "").strip(),
        "snapshot": str(snapshot.get("cycle_run_id") or "").strip(),
    }
    distinct_cycles = sorted({value for value in cycle_values.values() if value})
    if len(distinct_cycles) > 1:
        _push(checks, level="fail", code="cycle_run_mismatch", message="Os artefatos do ciclo diário foram publicados com run_ids diferentes.")
    if op_candidate and reg_candidate and op_candidate != reg_candidate:
        _push(checks, level="fail", code="candidate_mismatch", message="Modo ataque operacional divergiu do campeão registrado.")
    if finance_last and attack_latest:
        finance_dt = _parse_date(finance_last)
        attack_dt = _parse_date(attack_latest)
        if finance_dt and attack_dt:
            lag = max((finance_dt - attack_dt).days, 0)
            if lag > 3:
                _push(checks, level="warn", code="operation_attack_stale", message=f"O modo ataque ficou {lag} dias atrás da base publicada.")
    op_generated = _parse_dt(operation.get("generated_at_utc"))
    vig_generated = _parse_dt(vigilance.get("generated_at_utc"))
    if op_generated and vig_generated and vig_generated < op_generated:
        _push(checks, level="warn", code="vigilance_older_than_operation", message="A vigilância diária está mais velha que a leitura operacional.")
    if int(quality.get("critical_stale_assets") or 0) > 0:
        _push(checks, level="fail", code="critical_assets_stale", message="Ainda há ativos críticos atrasados após o fechamento diário.")
    if int(quality.get("core_stale_assets") or 0) > 0:
        _push(checks, level="warn", code="core_assets_stale", message="O núcleo do motor ainda tem ativos atrasados.")
    if not (operation.get("publish_ready") is True):
        _push(checks, level="warn", code="publish_not_marked_ready", message="O agente de operação não marcou a publicação como pronta.")
    if snapshot and not isinstance(snapshot.get("shadow_gods"), dict):
        _push(checks, level="fail", code="snapshot_shadow_gods_missing", message="O snapshot não publicou o bloco principal dos shadow gods.")
    elif snapshot:
        gods = (snapshot.get("shadow_gods") or {}).get("gods")
        if not isinstance(gods, list) or len(gods) != 4:
            _push(checks, level="warn", code="snapshot_shadow_gods_unexpected_count", message="O snapshot publicou uma contagem inesperada de shadow gods.")

    base_url = str(args.base_url or "").strip()
    http_checks: list[dict[str, Any]] = []
    if base_url:
        for path in ["/api/platform/latest", "/app/dashboard", "/app/shadow-mode"]:
            result = _http_check(base_url, path)
            http_checks.append(result)
            if not result.get("ok"):
                _push(checks, level="fail", code="http_smoke_failed", message=f"Falha no smoke HTTP para {path}.")

    status = "ok"
    if any(item["level"] == "fail" for item in checks):
        status = "fail"
    elif any(item["level"] == "warn" for item in checks):
        status = "warn"

    summary = attach_agent_guide(
        attach_cycle_context({
            "status": status,
            "generated_at_utc": utc_now_iso(),
            "finance_last_date": finance_last,
            "operation_attack_latest_date": attack_latest,
            "operation_candidate_id": op_candidate,
            "registry_candidate_id": reg_candidate,
            "cycle_values": cycle_values,
            "http_checks": http_checks,
            "checks": checks,
            "notes": [
                "Este agente compara a verdade operacional, o snapshot interno e o snapshot público.",
                "Smoke HTTP só roda quando uma base URL é fornecida explicitamente.",
            ],
        }, cycle_run_id=cycle_run_id, agent_run_id=agent_run_id),
        "daily-smoke-test-agent",
    )

    outroot = (ROOT / args.outdir_root).resolve()
    ts_dir = outroot / agent_run_id
    _write_json(ts_dir / "summary.json", summary)
    _write_json(outroot / "latest_summary.json", summary)
    _write_json(outroot / "latest_smoke_test.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    if status == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
