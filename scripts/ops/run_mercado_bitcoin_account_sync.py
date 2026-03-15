#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution.broker_mercado_bitcoin import (  # noqa: E402
    fetch_mercado_bitcoin_account_snapshot,
    write_mercado_bitcoin_snapshot,
)
from execution.live_ops import load_live_execution_profile, write_json  # noqa: E402


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Sincroniza a carteira real da Mercado Bitcoin para o fluxo live.")
    ap.add_argument("--config", default="config/live_execution_profile.json")
    ap.add_argument("--outdir-root", default="results/ops/execution_live")
    args = ap.parse_args()

    profile = load_live_execution_profile(ROOT / args.config)
    adapter = profile.get("broker_adapter", {}) if isinstance(profile.get("broker_adapter"), dict) else {}
    paths = profile.get("paths", {}) if isinstance(profile.get("paths"), dict) else {}
    adapter_paths = adapter.get("paths", {}) if isinstance(adapter.get("paths"), dict) else {}
    portfolio_path = (ROOT / str(paths.get("portfolio_state_json") or "data/live_execution/portfolio_state.json")).resolve()
    latest_snapshot_path = (ROOT / str(adapter_paths.get("latest_account_snapshot_json") or "results/ops/execution_live/latest_mercado_bitcoin_account_snapshot.json")).resolve()

    current = _read_json(portfolio_path)
    snapshot = fetch_mercado_bitcoin_account_snapshot(profile, portfolio_state=current)
    write_mercado_bitcoin_snapshot(snapshot, json_path=latest_snapshot_path)
    if snapshot.get("status") == "ok" and isinstance(snapshot.get("portfolio_state"), dict):
        write_json(portfolio_path, snapshot["portfolio_state"])

    run_id = str(snapshot.get("generated_at_utc") or "latest").replace(":", "").replace("-", "")
    outdir = (ROOT / args.outdir_root / run_id).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    write_mercado_bitcoin_snapshot(snapshot, json_path=outdir / "mercado_bitcoin_account_sync.json")
    print(json.dumps(snapshot, ensure_ascii=False))


if __name__ == "__main__":
    main()
