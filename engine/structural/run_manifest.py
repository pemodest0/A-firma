from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_hash(repo_root: Path | None = None) -> str:
    root = repo_root or ROOT
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        h = str(proc.stdout).strip()
        return h or "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _normalize_gates(gates: dict[str, Any] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in (gates or {}).items():
        ok = bool(v)
        out[str(k)] = "pass" if ok else "fail"
    return out


def write_run_manifest(
    outdir: Path,
    *,
    script: str,
    params: dict[str, Any] | None = None,
    paths: dict[str, Any] | None = None,
    gates: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    repo_root: Path | None = None,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "version": "run_manifest.v1",
        "timestamp_utc": utc_timestamp(),
        "git_hash": git_hash(repo_root=repo_root),
        "script": str(script),
        "params": params or {},
        "paths": paths or {},
        "gates": _normalize_gates(gates),
    }
    if extra:
        payload["extra"] = extra
    path = outdir / "RUN_MANIFEST.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
