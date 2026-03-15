from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if pd.isna(out):
        return float(default)
    return float(out)


@dataclass(frozen=True)
class PortfolioPosition:
    ticker: str
    quantity: float
    market_value_brl: float
    avg_price_brl: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PortfolioState:
    as_of_date: str
    base_currency: str
    cash_brl: float
    fx_rates: dict[str, float]
    positions: list[PortfolioPosition]
    open_orders: list[dict[str, Any]]

    @property
    def nav_brl(self) -> float:
        return float(self.cash_brl + sum(float(pos.market_value_brl) for pos in self.positions))

    def position_map(self) -> dict[str, PortfolioPosition]:
        return {str(pos.ticker): pos for pos in self.positions}

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of_date": self.as_of_date,
            "base_currency": self.base_currency,
            "cash_brl": float(self.cash_brl),
            "fx_rates": {str(k): float(v) for k, v in self.fx_rates.items()},
            "positions": [pos.to_dict() for pos in self.positions],
            "open_orders": list(self.open_orders),
            "nav_brl": float(self.nav_brl),
        }


@dataclass(frozen=True)
class OrderTicket:
    ticket_id: str
    ticker: str
    side: str
    notional_brl: float
    current_notional_brl: float
    target_notional_brl: float
    quantity_estimate: float | None
    price_brl_reference: float | None
    preferred_order_type: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_live_execution_profile(path: str | Path) -> dict[str, Any]:
    return _read_json(Path(path).resolve())


def load_portfolio_state(path: str | Path) -> PortfolioState:
    payload = _read_json(Path(path).resolve())
    positions: list[PortfolioPosition] = []
    for raw in payload.get("positions", []) if isinstance(payload.get("positions"), list) else []:
        if not isinstance(raw, dict):
            continue
        positions.append(
            PortfolioPosition(
                ticker=str(raw.get("ticker") or "").strip(),
                quantity=_safe_float(raw.get("quantity")),
                market_value_brl=_safe_float(raw.get("market_value_brl")),
                avg_price_brl=_safe_float(raw.get("avg_price_brl")),
            )
        )
    fx_rates = payload.get("fx_rates", {}) if isinstance(payload.get("fx_rates"), dict) else {}
    return PortfolioState(
        as_of_date=str(payload.get("as_of_date") or ""),
        base_currency=str(payload.get("base_currency") or "BRL"),
        cash_brl=_safe_float(payload.get("cash_brl")),
        fx_rates={str(k): _safe_float(v) for k, v in fx_rates.items()},
        positions=positions,
        open_orders=payload.get("open_orders", []) if isinstance(payload.get("open_orders"), list) else [],
    )


def portfolio_template_payload() -> dict[str, Any]:
    return {
        "as_of_date": "",
        "base_currency": "BRL",
        "cash_brl": 0.0,
        "fx_rates": {"USD_BRL": 0.0},
        "positions": [{"ticker": "BTC-USD", "quantity": 0.0, "market_value_brl": 0.0, "avg_price_brl": 0.0}],
        "open_orders": [],
    }


def load_last_prices(prices_dir: str | Path, tickers: list[str], *, fx_rates: dict[str, float] | None = None) -> dict[str, dict[str, float | None]]:
    prices: dict[str, dict[str, float | None]] = {}
    fx = fx_rates or {}
    usd_brl = _safe_float(fx.get("USD_BRL"), default=0.0)
    for ticker in sorted(set(str(t) for t in tickers if str(t).strip())):
        path = Path(prices_dir).resolve() / f"{ticker}.csv"
        if not path.exists():
            prices[ticker] = {"price_native": None, "price_brl": None}
            continue
        try:
            frame = pd.read_csv(path)
        except Exception:
            prices[ticker] = {"price_native": None, "price_brl": None}
            continue
        if frame.empty or "price" not in frame.columns:
            prices[ticker] = {"price_native": None, "price_brl": None}
            continue
        native = _safe_float(pd.to_numeric(frame["price"], errors="coerce").dropna().iloc[-1], default=0.0)
        price_brl = native
        if ticker.endswith("-USD"):
            price_brl = native * usd_brl if usd_brl > 0.0 else None
        prices[ticker] = {"price_native": native if native > 0.0 else None, "price_brl": price_brl if price_brl and price_brl > 0.0 else None}
    return prices


def select_live_mode(operation: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    architecture = profile.get("live_architecture", {}) if isinstance(profile.get("live_architecture"), dict) else {}
    rules = profile.get("selection_rules", {}) if isinstance(profile.get("selection_rules"), dict) else {}
    guardrails = profile.get("risk_guardrails", {}) if isinstance(profile.get("risk_guardrails"), dict) else {}
    recommended = ((operation.get("recommended_live_mode") or {}).get("mode") or "").strip().lower()
    confidence = _safe_float(((operation.get("recommended_live_mode") or {}).get("confidence_score")), default=_safe_float(((operation.get("mode_confidence") or {}).get("confidence_score")), 0.0))
    vigilance = str((((operation.get("mode_confidence") or {}).get("metrics") or {}).get("vigilance_status") or "")).strip().lower()
    attack_payload = operation.get("mode_attack", {}) if isinstance(operation.get("mode_attack"), dict) else {}
    attack_gross = _safe_float(attack_payload.get("gross_exposure"))
    attack_conf_floor = max(
        _safe_float(rules.get("attack_min_confidence_score"), 0.62),
        _safe_float(guardrails.get("block_new_attack_when_confidence_below"), 0.55),
    )
    if recommended == str(guardrails.get("force_cash_when_recommended_mode") or "").strip().lower():
        mode_key = "guard"
    elif vigilance in {str(x).strip().lower() for x in guardrails.get("block_new_attack_vigilance_statuses", []) if str(x).strip()}:
        mode_key = "guard"
    elif confidence >= attack_conf_floor and attack_gross >= _safe_float(rules.get("attack_min_gross_exposure"), 0.3):
        mode_key = "attack"
    else:
        mode_key = "core"
    selected = architecture.get(mode_key, {}) if isinstance(architecture.get(mode_key), dict) else {}
    return {
        "mode_key": mode_key,
        "candidate_id": str(selected.get("candidate_id") or ""),
        "alias": str(selected.get("alias") or mode_key),
        "label": str(selected.get("label") or mode_key),
        "confidence_score": confidence,
        "recommended_mode": recommended or "proteção",
        "vigilance_status": vigilance or "unknown",
        "attack_gross_exposure": attack_gross,
    }


def build_target_allocation(operation: dict[str, Any], profile: dict[str, Any], portfolio: PortfolioState) -> dict[str, float]:
    exec_profile = profile.get("execution_profile", {}) if isinstance(profile.get("execution_profile"), dict) else {}
    rules = profile.get("selection_rules", {}) if isinstance(profile.get("selection_rules"), dict) else {}
    guardrails = profile.get("risk_guardrails", {}) if isinstance(profile.get("risk_guardrails"), dict) else {}
    selected = select_live_mode(operation, profile)
    nav_brl = float(max(0.0, portfolio.nav_brl))
    attack_payload = operation.get("mode_attack", {}) if isinstance(operation.get("mode_attack"), dict) else {}
    raw_gross = _safe_float(attack_payload.get("gross_exposure"))
    if selected["mode_key"] == "guard":
        target_gross = 0.0
        proxy_weights = exec_profile.get("guard_proxy_weights", {"CASH-BRL": 1.0})
    elif selected["mode_key"] == "attack":
        target_gross = min(
            _safe_float(guardrails.get("max_gross_exposure"), 0.8),
            max(_safe_float(rules.get("attack_min_gross_exposure"), 0.3), raw_gross),
            _safe_float(rules.get("attack_max_gross_exposure"), 0.8),
        )
        proxy_weights = exec_profile.get("attack_proxy_weights", {"BTC-USD": 0.7, "ETH-USD": 0.3})
    else:
        if raw_gross < _safe_float(exec_profile.get("min_actionable_gross_exposure"), 0.15):
            target_gross = 0.0
            proxy_weights = exec_profile.get("guard_proxy_weights", {"CASH-BRL": 1.0})
        else:
            target_gross = min(
                _safe_float(guardrails.get("max_gross_exposure"), 0.8),
                max(_safe_float(rules.get("core_floor_gross_exposure"), 0.2), raw_gross),
                _safe_float(rules.get("core_max_gross_exposure"), 0.45),
            )
            proxy_weights = exec_profile.get("core_proxy_weights", {"BTC-USD": 0.85, "ETH-USD": 0.15})

    risk_capital = nav_brl * target_gross
    min_two_asset_risk = _safe_float(exec_profile.get("min_two_asset_risk_brl"), 180.0)
    if risk_capital < min_two_asset_risk:
        proxy_weights = {"BTC-USD": 1.0} if target_gross > 0.0 else {"CASH-BRL": 1.0}

    normalized: dict[str, float] = {}
    total = sum(_safe_float(v) for v in proxy_weights.values())
    for ticker, weight in proxy_weights.items():
        normalized[str(ticker)] = _safe_float(weight) / max(total, 1e-9)

    target: dict[str, float] = {}
    for ticker, weight in normalized.items():
        if ticker == "CASH-BRL":
            continue
        target[ticker] = float(nav_brl * target_gross * weight)
    target["CASH-BRL"] = float(max(0.0, nav_brl - sum(target.values())))
    return target


def compile_order_tickets(operation: dict[str, Any], profile: dict[str, Any], portfolio: PortfolioState, prices: dict[str, dict[str, float | None]]) -> dict[str, Any]:
    exec_profile = profile.get("execution_profile", {}) if isinstance(profile.get("execution_profile"), dict) else {}
    guardrails = profile.get("risk_guardrails", {}) if isinstance(profile.get("risk_guardrails"), dict) else {}
    selected = select_live_mode(operation, profile)
    target = build_target_allocation(operation, profile, portfolio)
    nav_brl = max(0.0, portfolio.nav_brl)
    current_map = portfolio.position_map()
    current_notional: dict[str, float] = {
        str(ticker): float(pos.market_value_brl) for ticker, pos in current_map.items()
    }
    current_notional["CASH-BRL"] = float(portfolio.cash_brl)

    if portfolio.open_orders:
        return {
            "status": "blocked_open_orders",
            "selected_mode": selected,
            "target_notional_brl": target,
            "tickets": [],
            "notes": ["Existem ordens abertas; limpe ou reconcilie antes de emitir novo plano."],
        }

    raw_deltas: dict[str, float] = {}
    for ticker in sorted(set(current_notional) | set(target)):
        raw_deltas[ticker] = float(target.get(ticker, 0.0) - current_notional.get(ticker, 0.0))
    total_turnover = sum(abs(value) for key, value in raw_deltas.items() if key != "CASH-BRL") / 2.0
    max_turnover = float(nav_brl * _safe_float(guardrails.get("max_turnover_fraction_of_nav"), 0.65))
    scale = 1.0
    if total_turnover > max_turnover > 0.0:
        scale = max_turnover / max(total_turnover, 1e-9)

    min_order_brl = _safe_float(exec_profile.get("min_order_brl"), 40.0)
    min_position_brl = _safe_float(exec_profile.get("min_position_brl"), 80.0)
    tickets: list[OrderTicket] = []
    blocked: list[dict[str, Any]] = []
    next_target: dict[str, float] = dict(current_notional)
    for ticker, delta in raw_deltas.items():
        if ticker == "CASH-BRL":
            continue
        scaled_delta = float(delta * scale)
        current_value = float(current_notional.get(ticker, 0.0))
        target_value = float(current_value + scaled_delta)
        if abs(scaled_delta) < min_order_brl:
            next_target[ticker] = current_value
            continue
        if target_value > 0.0 and target_value < min_position_brl:
            blocked.append(
                {
                    "ticker": ticker,
                    "reason": "target_below_min_position",
                    "target_notional_brl": target_value,
                    "current_notional_brl": current_value,
                }
            )
            next_target[ticker] = current_value
            continue
        price_brl = (prices.get(ticker) or {}).get("price_brl")
        qty = float(abs(scaled_delta) / price_brl) if price_brl and price_brl > 0.0 else None
        tickets.append(
            OrderTicket(
                ticket_id=str(uuid4()),
                ticker=ticker,
                side="buy" if scaled_delta > 0.0 else "sell",
                notional_brl=abs(scaled_delta),
                current_notional_brl=current_value,
                target_notional_brl=target_value,
                quantity_estimate=qty,
                price_brl_reference=float(price_brl) if price_brl and price_brl > 0.0 else None,
                preferred_order_type="market_notional",
                reason=f"align_to_{selected['mode_key']}",
            )
        )
        next_target[ticker] = target_value
    next_target["CASH-BRL"] = float(nav_brl - sum(value for key, value in next_target.items() if key != "CASH-BRL"))

    actionable_assets = [ticker for ticker, value in next_target.items() if ticker != "CASH-BRL" and value >= min_position_brl]
    if len(actionable_assets) > int(_safe_float(exec_profile.get("max_assets"), 2)):
        actionable_assets = sorted(actionable_assets, key=lambda t: next_target[t], reverse=True)[: int(_safe_float(exec_profile.get("max_assets"), 2))]
        blocked.append({"ticker": "MULTI", "reason": "max_assets_exceeded", "kept": actionable_assets})

    return {
        "status": "ok",
        "selected_mode": selected,
        "target_notional_brl": target,
        "target_notional_after_caps_brl": next_target,
        "turnover_requested_brl": total_turnover,
        "turnover_cap_brl": max_turnover,
        "turnover_scale_applied": scale,
        "tickets": [ticket.to_dict() for ticket in tickets],
        "blocked": blocked,
        "notes": [
            "O plano live usa proxy executavel em poucos ativos, nao replica literalmente toda a cesta teorica do laboratorio.",
            "A emissao continua semi-automatica: o sistema sugere notional e lado, e a revisao humana ainda fica obrigatoria.",
        ],
    }


def append_execution_ledger(csv_path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(csv_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def reconcile_execution(plan: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    plan_tickets = plan.get("tickets", []) if isinstance(plan.get("tickets"), list) else []
    exec_rows = report.get("executions", []) if isinstance(report.get("executions"), list) else []
    execution_by_ticket = {str(row.get("ticket_id") or ""): row for row in exec_rows if isinstance(row, dict)}
    reconciled: list[dict[str, Any]] = []
    for ticket in plan_tickets:
        ticket_id = str(ticket.get("ticket_id") or "")
        fill = execution_by_ticket.get(ticket_id, {})
        filled = _safe_float(fill.get("filled_notional_brl"))
        planned = _safe_float(ticket.get("notional_brl"))
        reconciled.append(
            {
                "ticket_id": ticket_id,
                "ticker": str(ticket.get("ticker") or ""),
                "side": str(ticket.get("side") or ""),
                "planned_notional_brl": planned,
                "filled_notional_brl": filled,
                "filled_quantity": _safe_float(fill.get("filled_quantity")),
                "fill_ratio": filled / planned if planned > 0.0 else 0.0,
                "avg_price_brl": _safe_float(fill.get("avg_price_brl")),
                "fee_brl": _safe_float(fill.get("fee_brl")),
                "executed_at": str(fill.get("executed_at") or ""),
                "status": str(fill.get("status") or "missing"),
            }
        )
    return {
        "status": "ok",
        "plan_run_id": str(plan.get("run_id") or ""),
        "ticket_count": len(plan_tickets),
        "execution_count": len(exec_rows),
        "reconciled_rows": reconciled,
        "notes": [
            "Diferencas entre planned_notional_brl e filled_notional_brl medem o slippage operacional real da conta.",
            "A reconciliacao ainda depende de um report local da corretora/exchange; API automatica fica para a proxima etapa.",
        ],
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    _write_json(Path(path).resolve(), payload)
