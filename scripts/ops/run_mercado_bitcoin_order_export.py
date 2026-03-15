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

from execution.broker_mercado_bitcoin import (  # noqa: E402
    build_mercado_bitcoin_preview,
    write_mercado_bitcoin_preview,
)
from execution.live_ops import load_live_execution_profile  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Exporta preview de ordens da Mercado Bitcoin a partir do latest_execution_plan.")
    ap.add_argument("--config", default="config/live_execution_profile.json")
    ap.add_argument("--plan-json", default="results/ops/execution_live/latest_execution_plan.json")
    args = ap.parse_args()

    profile = load_live_execution_profile(ROOT / args.config)
    plan = _read_json((ROOT / args.plan_json).resolve())
    adapter = profile.get("broker_adapter", {}) if isinstance(profile.get("broker_adapter"), dict) else {}
    paths = adapter.get("paths", {}) if isinstance(adapter.get("paths"), dict) else {}
    json_path = ROOT / str(paths.get("latest_preview_json") or "results/ops/execution_live/latest_mercado_bitcoin_order_preview.json")
    csv_path = ROOT / str(paths.get("latest_preview_csv") or "results/ops/execution_live/latest_mercado_bitcoin_order_preview.csv")

    preview = build_mercado_bitcoin_preview(plan, profile)
    write_mercado_bitcoin_preview(preview, json_path=json_path, csv_path=csv_path)
    print(json.dumps(preview, ensure_ascii=False))


if __name__ == "__main__":
    main()
