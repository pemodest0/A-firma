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
    submit_mercado_bitcoin_orders,
    write_mercado_bitcoin_submission,
)
from execution.live_ops import load_live_execution_profile  # noqa: E402


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Submete manualmente o preview de ordens para a Mercado Bitcoin.")
    ap.add_argument("--config", default="config/live_execution_profile.json")
    ap.add_argument("--preview-json", default="results/ops/execution_live/latest_mercado_bitcoin_order_preview.json")
    ap.add_argument("--submit", action="store_true", help="Sem este flag, apenas gera um resumo bloqueado.")
    ap.add_argument("--confirm", default="", help="Confirme com MB_SUBMIT para liberar o envio.")
    args = ap.parse_args()

    profile = load_live_execution_profile(ROOT / args.config)
    adapter = profile.get("broker_adapter", {}) if isinstance(profile.get("broker_adapter"), dict) else {}
    adapter_paths = adapter.get("paths", {}) if isinstance(adapter.get("paths"), dict) else {}
    preview = _read_json((ROOT / args.preview_json).resolve())
    latest_json = (ROOT / str(adapter_paths.get("latest_submit_json") or "results/ops/execution_live/latest_mercado_bitcoin_submit.json")).resolve()
    latest_csv = (ROOT / str(adapter_paths.get("latest_submit_csv") or "results/ops/execution_live/latest_mercado_bitcoin_submit.csv")).resolve()

    if not preview:
        payload = {
            "status": "missing_preview",
            "broker": str(adapter.get("name") or "mercado_bitcoin"),
        }
    elif not args.submit:
        payload = {
            "status": "blocked",
            "broker": str(adapter.get("name") or "mercado_bitcoin"),
            "reason": "missing_submit_flag",
            "notes": ["Use --submit --confirm MB_SUBMIT para liberar o envio manual assistido."],
        }
    elif str(args.confirm).strip() != "MB_SUBMIT":
        payload = {
            "status": "blocked",
            "broker": str(adapter.get("name") or "mercado_bitcoin"),
            "reason": "missing_confirmation_token",
            "notes": ["Confirmacao invalida. Use --confirm MB_SUBMIT para liberar o envio."],
        }
    else:
        payload = submit_mercado_bitcoin_orders(preview, profile)

    write_mercado_bitcoin_submission(payload, json_path=latest_json, csv_path=latest_csv)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
