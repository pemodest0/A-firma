from __future__ import annotations

import os
from datetime import datetime, timezone


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_cycle_run_id(value: str | None = None) -> str:
    raw = str(value or os.environ.get("ASSYNTRAX_CYCLE_RUN_ID") or "").strip()
    return raw or utc_run_id()


def attach_cycle_context(payload: dict, *, cycle_run_id: str, agent_run_id: str) -> dict:
    payload["cycle_run_id"] = str(cycle_run_id)
    payload["agent_run_id"] = str(agent_run_id)
    return payload
