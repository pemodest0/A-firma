from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from execution.net_assumptions import load_net_assumption_profiles


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if pd.isna(out):
        return float(default)
    return float(out)


def _resolve_progressive_rate(gain_brl: float, brackets: list[tuple[float, float]], fallback_rate: float) -> float:
    gain = float(max(0.0, gain_brl))
    if not brackets:
        return float(max(0.0, fallback_rate))
    for up_to_brl, rate in brackets:
        if gain <= float(up_to_brl):
            return float(max(0.0, rate))
    return float(max(0.0, brackets[-1][1]))


def _read_ledger(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    if "executed_at" in frame.columns:
        frame["executed_at"] = pd.to_datetime(frame["executed_at"], errors="coerce")
    return frame


def build_live_tax_summary(
    *,
    ledger_csv: str | Path,
    net_assumptions_config: str | Path,
    profile_id: str = "crypto_global_brazil_resident_conservative",
) -> dict[str, Any]:
    ledger_path = Path(ledger_csv).resolve()
    ledger = _read_ledger(ledger_path)
    profiles = load_net_assumption_profiles(net_assumptions_config)
    profile = profiles["profiles"][profile_id]
    brackets = [(float(a), float(b)) for a, b in profile.capital_gains_brackets]

    if ledger.empty:
        return {
            "status": "no_fills_yet",
            "profile_id": profile_id,
            "ledger_path": str(ledger_path),
            "monthly_rows": [],
            "inventory": {},
            "alerts": ["Nenhuma execucao preenchida ainda no ledger."],
        }

    inventory: dict[str, dict[str, float]] = {}
    monthly_rows: list[dict[str, Any]] = []
    carry_loss_brl = 0.0
    current_month: str | None = None
    month_buy_brl = 0.0
    month_sales_brl = 0.0
    month_realized_gain_brl = 0.0

    def flush_month() -> None:
        nonlocal carry_loss_brl, month_buy_brl, month_sales_brl, month_realized_gain_brl, current_month
        if current_month is None:
            return
        taxable_brl = 0.0
        if not (profile.monthly_sales_exemption_brl > 0.0 and month_sales_brl <= float(profile.monthly_sales_exemption_brl)):
            if profile.loss_compensation_enabled:
                effective = month_realized_gain_brl + carry_loss_brl
                taxable_brl = float(max(0.0, effective))
                carry_loss_brl = float(min(0.0, effective))
            else:
                taxable_brl = float(max(0.0, month_realized_gain_brl))
        elif profile.loss_compensation_enabled and month_realized_gain_brl < 0.0:
            carry_loss_brl += month_realized_gain_brl

        tax_rate = _resolve_progressive_rate(taxable_brl, brackets, float(profile.capital_gains_tax_rate))
        tax_due = taxable_brl * tax_rate
        monthly_rows.append(
            {
                "ym": str(current_month),
                "buy_notional_brl": float(month_buy_brl),
                "sell_notional_brl": float(month_sales_brl),
                "realized_gain_brl": float(month_realized_gain_brl),
                "taxable_gain_brl": float(taxable_brl),
                "estimated_tax_due_brl": float(tax_due),
                "estimated_tax_rate": float(tax_rate),
                "sales_exemption_limit_brl": float(profile.monthly_sales_exemption_brl),
                "sales_above_exemption": bool(month_sales_brl > float(profile.monthly_sales_exemption_brl)),
                "carry_loss_brl_end": float(carry_loss_brl),
            }
        )
        month_buy_brl = 0.0
        month_sales_brl = 0.0
        month_realized_gain_brl = 0.0

    for _, row in ledger.iterrows():
        status = str(row.get("status") or "").strip().lower()
        if status not in {"filled", "partial"}:
            continue
        side = str(row.get("side") or "").strip().lower()
        ticker = str(row.get("ticker") or "").strip()
        if not ticker or side not in {"buy", "sell"}:
            continue

        executed_at = pd.to_datetime(row.get("executed_at"), errors="coerce")
        month = executed_at.strftime("%Y-%m") if pd.notna(executed_at) else "unknown"
        if current_month is None:
            current_month = month
        elif month != current_month:
            flush_month()
            current_month = month

        filled_notional_brl = _safe_float(row.get("filled_notional_brl"))
        avg_price_brl = _safe_float(row.get("avg_price_brl"))
        fee_brl = _safe_float(row.get("fee_brl"))
        filled_quantity = _safe_float(row.get("filled_quantity"))
        if filled_quantity <= 0.0 and avg_price_brl > 0.0:
            filled_quantity = filled_notional_brl / avg_price_brl
        if filled_notional_brl <= 0.0 or filled_quantity <= 0.0:
            continue

        lot = inventory.setdefault(ticker, {"quantity": 0.0, "cost_basis_brl": 0.0, "avg_cost_brl": 0.0})
        if side == "buy":
            lot["quantity"] += filled_quantity
            lot["cost_basis_brl"] += filled_notional_brl + fee_brl
            lot["avg_cost_brl"] = lot["cost_basis_brl"] / max(lot["quantity"], 1e-9)
            month_buy_brl += filled_notional_brl
        else:
            qty_to_sell = min(filled_quantity, lot["quantity"])
            if qty_to_sell <= 0.0:
                continue
            avg_cost = lot["avg_cost_brl"] if lot["quantity"] > 0.0 else 0.0
            cost_basis_sold = avg_cost * qty_to_sell
            proceeds_net = max(0.0, filled_notional_brl - fee_brl)
            realized_gain = proceeds_net - cost_basis_sold
            lot["quantity"] = max(0.0, lot["quantity"] - qty_to_sell)
            lot["cost_basis_brl"] = max(0.0, lot["cost_basis_brl"] - cost_basis_sold)
            lot["avg_cost_brl"] = lot["cost_basis_brl"] / max(lot["quantity"], 1e-9) if lot["quantity"] > 0.0 else 0.0
            month_sales_brl += filled_notional_brl
            month_realized_gain_brl += realized_gain

    flush_month()
    alerts: list[str] = []
    if any(bool(row["sales_above_exemption"]) for row in monthly_rows):
        alerts.append("Houve mes com vendas acima da faixa de isencao; conferir DARF e ganho estimado.")
    if any(float(row["estimated_tax_due_brl"]) > 0.0 for row in monthly_rows):
        alerts.append("Existe imposto estimado positivo em pelo menos um mes.")
    if not alerts:
        alerts.append("Ate agora nao apareceu imposto estimado positivo no ledger preenchido.")
    return {
        "status": "ok",
        "profile_id": profile_id,
        "ledger_path": str(ledger_path),
        "monthly_rows": monthly_rows,
        "inventory": inventory,
        "alerts": alerts,
    }


def write_monthly_summary_csv(path: str | Path, monthly_rows: list[dict[str, Any]]) -> None:
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if not monthly_rows:
        out.write_text("", encoding="utf-8")
        return
    frame = pd.DataFrame(monthly_rows)
    frame.to_csv(out, index=False)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
