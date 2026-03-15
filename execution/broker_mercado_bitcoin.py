from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if out != out:
        return float(default)
    return float(out)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else [
        "client_order_id",
        "market",
        "side",
        "order_type",
        "notional_brl",
        "quantity_estimate",
        "source_ticker",
        "reason",
        "status",
    ]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_mercado_bitcoin_preview(plan: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    adapter = profile.get("broker_adapter", {}) if isinstance(profile.get("broker_adapter"), dict) else {}
    supported_pairs = adapter.get("supported_pairs", {}) if isinstance(adapter.get("supported_pairs"), dict) else {}
    tickets = plan.get("tickets", []) if isinstance(plan.get("tickets"), list) else []
    export_rows: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []

    for raw in tickets:
        if not isinstance(raw, dict):
            continue
        source_ticker = str(raw.get("ticker") or "").strip()
        if not source_ticker or source_ticker == "CASH-BRL":
            continue
        market = str(supported_pairs.get(source_ticker) or "").strip()
        if not market:
            unsupported.append(
                {
                    "ticket_id": str(raw.get("ticket_id") or ""),
                    "ticker": source_ticker,
                    "reason": "unsupported_market",
                }
            )
            continue
        export_rows.append(
            {
                "client_order_id": str(raw.get("ticket_id") or ""),
                "market": market,
                "side": str(raw.get("side") or "").lower(),
                "order_type": str(raw.get("preferred_order_type") or adapter.get("default_order_type") or "market"),
                "notional_brl": round(_safe_float(raw.get("notional_brl")), 2),
                "quantity_estimate": _safe_float(raw.get("quantity_estimate"), default=0.0),
                "price_reference_brl": _safe_float(raw.get("price_brl_reference"), default=0.0),
                "source_ticker": source_ticker,
                "reason": str(raw.get("reason") or ""),
                "status": "preview",
            }
        )

    notes = [
        "Preview only: ainda nao envia ordem automatica para a Mercado Bitcoin.",
        "Revise market, side e notional_brl antes de colar ou integrar na corretora.",
        "CASH-BRL nao gera ordem; SHY e outros ativos fora do mapa ficam marcados como unsupported_market.",
    ]
    if unsupported:
        notes.append("Existem tickets nao suportados pela Mercado Bitcoin e eles ficaram fora do preview.")

    return {
        "status": "ok" if export_rows else "no_action",
        "broker": str(adapter.get("name") or "mercado_bitcoin"),
        "mode": str(adapter.get("mode") or "preview_only"),
        "submit_enabled": adapter.get("submit_enabled") is True,
        "generated_at_utc": str(plan.get("generated_at_utc") or ""),
        "plan_run_id": str(plan.get("run_id") or ""),
        "cycle_run_id": str(plan.get("cycle_run_id") or ""),
        "selected_mode": plan.get("selected_mode", {}),
        "order_count": len(export_rows),
        "unsupported_count": len(unsupported),
        "orders": export_rows,
        "unsupported": unsupported,
        "manual_steps": [
            "Abrir a Mercado Bitcoin e conferir saldo em BRL e posicoes antes de qualquer ordem.",
            "Comparar latest_execution_plan.json com este preview antes de agir.",
            "Executar manualmente apenas as ordens que continuarem fazendo sentido no book real.",
        ],
        "notes": notes,
    }


def write_mercado_bitcoin_preview(
    preview: dict[str, Any],
    *,
    json_path: str | Path,
    csv_path: str | Path,
) -> None:
    _write_json(Path(json_path).resolve(), preview)
    rows = preview.get("orders", []) if isinstance(preview.get("orders"), list) else []
    write_csv(csv_path, [row for row in rows if isinstance(row, dict)])
