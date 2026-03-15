from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from execution.live_ops import (
    PortfolioPosition,
    PortfolioState,
    append_execution_ledger,
    compile_target_order_tickets,
    load_last_prices,
)


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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def load_shadow_gods_profile(path: str | Path) -> dict[str, Any]:
    return _read_json(Path(path).resolve())


def capital_block(profile: dict[str, Any], capital_brl: float) -> dict[str, Any]:
    blocks = profile.get("capital_blocks", []) if isinstance(profile.get("capital_blocks"), list) else []
    rounded = round(float(capital_brl), 2)
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if round(_safe_float(block.get("capital_brl")), 2) == rounded:
            return block
    return {
        "capital_brl": float(capital_brl),
        "max_assets": 2,
        "min_order_brl": 40.0,
        "min_position_brl": 80.0,
    }


def _portfolio_payload(portfolio: PortfolioState, *, label: str) -> dict[str, Any]:
    return {
        "label": label,
        "as_of_date": portfolio.as_of_date,
        "base_currency": portfolio.base_currency,
        "cash_brl": float(portfolio.cash_brl),
        "fx_rates": {str(k): float(v) for k, v in portfolio.fx_rates.items()},
        "positions": [pos.to_dict() for pos in portfolio.positions],
        "open_orders": list(portfolio.open_orders),
        "nav_brl": float(portfolio.nav_brl),
    }


def load_or_create_shadow_portfolio(
    path: str | Path,
    *,
    capital_brl: float,
    as_of_date: str,
    base_currency: str,
    fx_rates: dict[str, float],
) -> PortfolioState:
    state_path = Path(path).resolve()
    payload = _read_json(state_path)
    if payload:
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
        return PortfolioState(
            as_of_date=str(payload.get("as_of_date") or as_of_date),
            base_currency=str(payload.get("base_currency") or base_currency),
            cash_brl=_safe_float(payload.get("cash_brl"), capital_brl),
            fx_rates={str(k): _safe_float(v) for k, v in (payload.get("fx_rates") or fx_rates).items()},
            positions=positions,
            open_orders=payload.get("open_orders", []) if isinstance(payload.get("open_orders"), list) else [],
        )
    portfolio = PortfolioState(
        as_of_date=as_of_date,
        base_currency=base_currency,
        cash_brl=float(capital_brl),
        fx_rates={str(k): float(v) for k, v in fx_rates.items()},
        positions=[],
        open_orders=[],
    )
    _write_json(state_path, _portfolio_payload(portfolio, label="initial"))
    return portfolio


def revalue_portfolio(portfolio: PortfolioState, prices: dict[str, dict[str, float | None]], *, as_of_date: str) -> PortfolioState:
    revalued_positions: list[PortfolioPosition] = []
    for pos in portfolio.positions:
        price_brl = _safe_float((prices.get(pos.ticker) or {}).get("price_brl"), pos.avg_price_brl)
        revalued_positions.append(
            PortfolioPosition(
                ticker=pos.ticker,
                quantity=float(pos.quantity),
                market_value_brl=float(pos.quantity * price_brl),
                avg_price_brl=float(pos.avg_price_brl),
            )
        )
    return PortfolioState(
        as_of_date=as_of_date,
        base_currency=portfolio.base_currency,
        cash_brl=float(portfolio.cash_brl),
        fx_rates=dict(portfolio.fx_rates),
        positions=revalued_positions,
        open_orders=[],
    )


def infer_shadow_market_state(operation: dict[str, Any], vigilance: dict[str, Any], *, role: str) -> tuple[str, list[str]]:
    recommended_mode = str(((operation.get("recommended_live_mode") or {}).get("mode")) or "").strip().lower()
    confidence = _safe_float(((operation.get("recommended_live_mode") or {}).get("confidence_score")), _safe_float(((operation.get("mode_confidence") or {}).get("confidence_score")), 0.0))
    vigilance_status = str((((operation.get("mode_confidence") or {}).get("metrics") or {}).get("vigilance_status")) or (vigilance.get("status") or "")).strip().lower()
    structural_regime = str(((operation.get("official_structural_regime") or {}).get("regime")) or "").strip().lower()
    posture = str((((operation.get("recommended_live_mode") or {}).get("current_posture")) or {}).get("posture") or "").strip().lower()
    notes: list[str] = []

    if recommended_mode == "proteção" or vigilance_status in {"warn", "fail"} or structural_regime == "stress":
        notes.append("Mercado em proteção, vigilância ruim ou stress estrutural.")
        state = "defense"
    elif structural_regime == "dispersion":
        notes.append("Dispersion libera busca mais oportunística.")
        state = "opportunistic"
    elif structural_regime == "transition":
        notes.append("Transition pede postura intermediária.")
        state = "balanced"
    elif confidence >= 0.68 and posture in {"ataque_pleno", "ataque_parcial"}:
        notes.append("Confiança e postura permitem risk-on.")
        state = "risk_on"
    else:
        notes.append("Sem gatilho forte; fica em balanced.")
        state = "balanced"

    if role == "turbo_attack_shadow" and state == "risk_on" and confidence < 0.75:
        notes.append("Hermes perde um degrau porque a confiança ainda não é alta o bastante.")
        state = "balanced"
    if role == "structural_anchor" and state == "defense" and structural_regime == "transition":
        notes.append("Apollo segura balanced em vez de zerar totalmente por ser âncora estrutural.")
        state = "balanced"
    return state, notes


def _normalized_weights(weights: dict[str, float]) -> dict[str, float]:
    clean = {str(k): max(0.0, _safe_float(v)) for k, v in weights.items() if _safe_float(v) > 0.0}
    total = sum(clean.values())
    if total <= 0.0:
        return {"CASH-BRL": 1.0}
    return {ticker: value / total for ticker, value in clean.items()}


def _trim_weights_for_block(weights: dict[str, float], *, max_assets: int) -> dict[str, float]:
    normalized = _normalized_weights(weights)
    cash_weight = float(normalized.get("CASH-BRL", 0.0))
    risk_weights = [(ticker, weight) for ticker, weight in normalized.items() if ticker != "CASH-BRL"]
    if len(risk_weights) <= max_assets:
        return normalized
    kept = sorted(risk_weights, key=lambda item: item[1], reverse=True)[: int(max_assets)]
    kept_sum = sum(weight for _, weight in kept)
    risk_total = sum(weight for _, weight in risk_weights)
    trimmed: dict[str, float] = {}
    if kept_sum > 0.0:
        for ticker, weight in kept:
            trimmed[ticker] = (weight / kept_sum) * risk_total
    if cash_weight > 0.0:
        trimmed["CASH-BRL"] = cash_weight
    return _normalized_weights(trimmed)


def build_shadow_target_weights(god: dict[str, Any], *, market_state: str, max_assets: int, prices: dict[str, dict[str, float | None]]) -> tuple[dict[str, float], list[str]]:
    allocations = god.get("allocations", {}) if isinstance(god.get("allocations"), dict) else {}
    requested = allocations.get(market_state, {}) if isinstance(allocations.get(market_state), dict) else {}
    notes: list[str] = []
    dropped_for_price: list[str] = []
    filtered: dict[str, float] = {}
    for ticker, weight in requested.items():
        if ticker == "CASH-BRL":
            filtered[ticker] = _safe_float(weight)
            continue
        price_brl = _safe_float((prices.get(str(ticker)) or {}).get("price_brl"))
        if price_brl <= 0.0:
            dropped_for_price.append(str(ticker))
            continue
        filtered[str(ticker)] = _safe_float(weight)
    if dropped_for_price:
        notes.append(f"Tickers sem preço válido caíram para caixa: {','.join(dropped_for_price)}")
        filtered["CASH-BRL"] = _safe_float(filtered.get("CASH-BRL"), 0.0) + sum(
            _safe_float(requested.get(ticker), 0.0) for ticker in dropped_for_price
        )
    trimmed = _trim_weights_for_block(filtered, max_assets=max_assets)
    return trimmed, notes


def build_target_notional(nav_brl: float, weights: dict[str, float]) -> dict[str, float]:
    return {str(ticker): float(nav_brl * _safe_float(weight)) for ticker, weight in weights.items()}


def _asset_class(profile: dict[str, Any], ticker: str) -> str:
    mapping = profile.get("ticker_asset_class", {}) if isinstance(profile.get("ticker_asset_class"), dict) else {}
    return str(mapping.get(ticker) or "international")


def simulate_shadow_fills(
    tickets: list[dict[str, Any]],
    *,
    profile: dict[str, Any],
    portfolio: PortfolioState,
    prices: dict[str, dict[str, float | None]],
    as_of_date: str,
) -> list[dict[str, Any]]:
    fee_map = profile.get("fee_bps_by_asset_class", {}) if isinstance(profile.get("fee_bps_by_asset_class"), dict) else {}
    cash_available = float(portfolio.cash_brl)
    fills: list[dict[str, Any]] = []
    ordered = sorted(tickets, key=lambda row: 0 if str(row.get("side") or "").lower() == "sell" else 1)
    for ticket in ordered:
        ticker = str(ticket.get("ticker") or "")
        side = str(ticket.get("side") or "")
        price_brl = _safe_float(ticket.get("price_brl_reference"), _safe_float((prices.get(ticker) or {}).get("price_brl"), 0.0))
        if price_brl <= 0.0:
            fills.append(
                {
                    "ticket_id": str(ticket.get("ticket_id") or ""),
                    "ticker": ticker,
                    "side": side,
                    "filled_notional_brl": 0.0,
                    "filled_quantity": 0.0,
                    "avg_price_brl": None,
                    "fee_brl": 0.0,
                    "status": "no_price",
                    "executed_at": f"{as_of_date}T00:00:00+00:00",
                }
            )
            continue
        requested_notional = _safe_float(ticket.get("notional_brl"), 0.0)
        fee_rate = _safe_float(fee_map.get(_asset_class(profile, ticker)), 0.0) / 10000.0
        filled_notional = requested_notional
        if side == "buy":
            max_notional = cash_available / max(1.0 + fee_rate, 1e-9)
            filled_notional = min(requested_notional, max_notional)
        fee_brl = float(filled_notional * fee_rate)
        if side == "buy":
            cash_available -= filled_notional + fee_brl
        else:
            cash_available += filled_notional - fee_brl
        fills.append(
            {
                "ticket_id": str(ticket.get("ticket_id") or ""),
                "ticker": ticker,
                "side": side,
                "filled_notional_brl": float(filled_notional),
                "filled_quantity": float(filled_notional / price_brl) if filled_notional > 0.0 else 0.0,
                "avg_price_brl": float(price_brl),
                "fee_brl": fee_brl,
                "status": "filled" if filled_notional > 0.0 else "skipped_cash",
                "executed_at": f"{as_of_date}T00:00:00+00:00",
            }
        )
    return fills


def apply_shadow_fills(portfolio: PortfolioState, fills: list[dict[str, Any]], *, as_of_date: str) -> PortfolioState:
    cash_brl = float(portfolio.cash_brl)
    positions = {pos.ticker: pos for pos in portfolio.positions}
    for fill in fills:
        if str(fill.get("status") or "") != "filled":
            continue
        ticker = str(fill.get("ticker") or "")
        qty = _safe_float(fill.get("filled_quantity"), 0.0)
        notional = _safe_float(fill.get("filled_notional_brl"), 0.0)
        avg_price = _safe_float(fill.get("avg_price_brl"), 0.0)
        fee_brl = _safe_float(fill.get("fee_brl"), 0.0)
        side = str(fill.get("side") or "")
        current = positions.get(ticker)
        current_qty = float(current.quantity) if current else 0.0
        current_avg = float(current.avg_price_brl) if current else 0.0
        if side == "buy":
            cash_brl -= notional + fee_brl
            new_qty = current_qty + qty
            total_cost = (current_qty * current_avg) + notional + fee_brl
            new_avg = total_cost / max(new_qty, 1e-9)
            positions[ticker] = PortfolioPosition(
                ticker=ticker,
                quantity=new_qty,
                market_value_brl=float(new_qty * avg_price),
                avg_price_brl=float(new_avg),
            )
        else:
            cash_brl += notional - fee_brl
            new_qty = max(0.0, current_qty - qty)
            if new_qty <= 1e-12:
                positions.pop(ticker, None)
            else:
                positions[ticker] = PortfolioPosition(
                    ticker=ticker,
                    quantity=new_qty,
                    market_value_brl=float(new_qty * avg_price),
                    avg_price_brl=float(current_avg),
                )
    return PortfolioState(
        as_of_date=as_of_date,
        base_currency=portfolio.base_currency,
        cash_brl=float(max(cash_brl, 0.0)),
        fx_rates=dict(portfolio.fx_rates),
        positions=sorted(list(positions.values()), key=lambda pos: pos.ticker),
        open_orders=[],
    )


def snapshot_holdings(portfolio: PortfolioState) -> list[dict[str, Any]]:
    holdings = []
    nav = max(portfolio.nav_brl, 1e-9)
    for pos in sorted(portfolio.positions, key=lambda row: row.market_value_brl, reverse=True):
        holdings.append(
            {
                "ticker": pos.ticker,
                "quantity": float(pos.quantity),
                "market_value_brl": float(pos.market_value_brl),
                "avg_price_brl": float(pos.avg_price_brl),
                "weight": float(pos.market_value_brl / nav),
            }
        )
    return holdings


def write_shadow_rows(csv_path: str | Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        append_execution_ledger(csv_path, rows)


def recent_csv_rows(csv_path: str | Path, *, limit: int = 5) -> list[dict[str, Any]]:
    path = Path(csv_path).resolve()
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-limit:]


def write_shadow_state(path: str | Path, portfolio: PortfolioState, *, label: str) -> None:
    _write_json(Path(path).resolve(), _portfolio_payload(portfolio, label=label))


def run_shadow_scenario(
    *,
    repo_root: Path,
    god: dict[str, Any],
    profile: dict[str, Any],
    operation: dict[str, Any],
    vigilance: dict[str, Any],
    prices_dir: Path,
    capital_brl: float,
    cycle_run_id: str,
    agent_run_id: str,
    as_of_date: str,
    state_root: Path,
    history_root: Path,
) -> dict[str, Any]:
    cap_block = capital_block(profile, capital_brl)
    alias = str(god.get("alias") or "unknown")
    role = str(god.get("role") or "")
    scenario_id = f"{alias.lower()}_{int(round(capital_brl))}"
    scenario_state_dir = state_root / alias / str(int(round(capital_brl)))
    scenario_history_dir = history_root / alias / str(int(round(capital_brl)))
    state_path = scenario_state_dir / "portfolio_state.json"
    requests_csv = scenario_history_dir / "requests.csv"
    fills_csv = scenario_history_dir / "fills.csv"
    history_csv = scenario_history_dir / "history.csv"

    portfolio = load_or_create_shadow_portfolio(
        state_path,
        capital_brl=capital_brl,
        as_of_date=as_of_date,
        base_currency=str(profile.get("base_currency") or "BRL"),
        fx_rates={str(k): float(v) for k, v in (profile.get("default_fx_rates") or {}).items()},
    )
    price_tickers = sorted({ticker for state_name in (god.get("allocations") or {}).values() if isinstance(state_name, dict) for ticker in state_name.keys() if ticker != "CASH-BRL"})
    prices = load_last_prices(prices_dir, price_tickers, fx_rates=portfolio.fx_rates)
    revalued = revalue_portfolio(portfolio, prices, as_of_date=as_of_date)
    market_state, market_notes = infer_shadow_market_state(operation, vigilance, role=role)
    target_weights, target_notes = build_shadow_target_weights(
        god,
        market_state=market_state,
        max_assets=int(_safe_float(cap_block.get("max_assets"), 2)),
        prices=prices,
    )
    target_notional = build_target_notional(revalued.nav_brl, target_weights)
    selected = {
        "mode_key": market_state,
        "candidate_id": str(god.get("candidate_id") or ""),
        "alias": alias,
        "label": str(god.get("thesis") or alias),
    }
    exec_profile = {
        "min_order_brl": _safe_float(cap_block.get("min_order_brl"), 40.0),
        "min_position_brl": _safe_float(cap_block.get("min_position_brl"), 80.0),
        "max_assets": int(_safe_float(cap_block.get("max_assets"), 2)),
    }
    guardrails = {
        "max_turnover_fraction_of_nav": 1.0,
    }
    compiled = compile_target_order_tickets(
        target=target_notional,
        selected=selected,
        exec_profile=exec_profile,
        guardrails=guardrails,
        portfolio=revalued,
        prices=prices,
        reason_prefix=f"shadow_{alias.lower()}",
        notes=[
            "Shadow diário com execução simulada automática para comparar os quatro deuses congelados.",
            "As ordens abaixo servem como solicitação operacional simulada; podem resultar em no-trade quando o estado pedir caixa.",
        ],
    )
    request_rows = []
    for ticket in compiled.get("tickets", []):
        request_rows.append(
            {
                "cycle_run_id": cycle_run_id,
                "agent_run_id": agent_run_id,
                "as_of_date": as_of_date,
                "scenario_id": scenario_id,
                "alias": alias,
                "capital_brl": capital_brl,
                "market_state": market_state,
                "ticket_id": ticket.get("ticket_id"),
                "ticker": ticket.get("ticker"),
                "side": ticket.get("side"),
                "notional_brl": ticket.get("notional_brl"),
                "reason": ticket.get("reason"),
            }
        )
    fills = simulate_shadow_fills(
        compiled.get("tickets", []),
        profile=profile,
        portfolio=revalued,
        prices=prices,
        as_of_date=as_of_date,
    )
    updated = apply_shadow_fills(revalued, fills, as_of_date=as_of_date)
    updated = revalue_portfolio(updated, prices, as_of_date=as_of_date)
    fill_rows = []
    for fill in fills:
        row = dict(fill)
        row.update(
            {
                "cycle_run_id": cycle_run_id,
                "agent_run_id": agent_run_id,
                "as_of_date": as_of_date,
                "scenario_id": scenario_id,
                "alias": alias,
                "capital_brl": capital_brl,
                "market_state": market_state,
            }
        )
        fill_rows.append(row)
    write_shadow_rows(requests_csv, request_rows)
    write_shadow_rows(fills_csv, fill_rows)
    history_row = {
        "cycle_run_id": cycle_run_id,
        "agent_run_id": agent_run_id,
        "as_of_date": as_of_date,
        "scenario_id": scenario_id,
        "alias": alias,
        "capital_brl": capital_brl,
        "market_state": market_state,
        "nav_before_brl": float(revalued.nav_brl),
        "nav_after_brl": float(updated.nav_brl),
        "cash_after_brl": float(updated.cash_brl),
        "order_count": len(compiled.get("tickets", [])),
        "fill_count": len([row for row in fills if str(row.get("status") or "") == "filled"]),
        "selected_assets": ",".join([row["ticker"] for row in snapshot_holdings(updated)]),
    }
    write_shadow_rows(history_csv, [history_row])
    write_shadow_state(state_path, updated, label=f"{alias}_{int(round(capital_brl))}")
    request_path = scenario_history_dir / f"{agent_run_id}_requests.json"
    fills_path = scenario_history_dir / f"{agent_run_id}_fills.json"
    _write_json(
        request_path,
        {
            "status": "ok",
            "cycle_run_id": cycle_run_id,
            "agent_run_id": agent_run_id,
            "scenario_id": scenario_id,
            "requests": request_rows,
            "compiled": compiled,
        },
    )
    _write_json(
        fills_path,
        {
            "status": "ok",
            "cycle_run_id": cycle_run_id,
            "agent_run_id": agent_run_id,
            "scenario_id": scenario_id,
            "fills": fill_rows,
        },
    )
    return {
        "scenario_id": scenario_id,
        "capital_brl": float(capital_brl),
        "market_state": market_state,
        "market_notes": market_notes + target_notes,
        "nav_before_brl": float(revalued.nav_brl),
        "nav_after_brl": float(updated.nav_brl),
        "cash_after_brl": float(updated.cash_brl),
        "target_weights": target_weights,
        "target_notional_brl": target_notional,
        "orders": compiled.get("tickets", []),
        "blocked": compiled.get("blocked", []),
        "fills": fills,
        "holdings": snapshot_holdings(updated),
        "order_count": len(compiled.get("tickets", [])),
        "fill_count": len([row for row in fills if str(row.get("status") or "") == "filled"]),
        "artifacts": {
            "state_path": str(state_path),
            "requests_csv": str(requests_csv),
            "fills_csv": str(fills_csv),
            "history_csv": str(history_csv),
            "request_json": str(request_path),
            "fills_json": str(fills_path),
        },
        "history_tail": recent_csv_rows(history_csv, limit=5),
        "request_tail": recent_csv_rows(requests_csv, limit=5),
        "fills_tail": recent_csv_rows(fills_csv, limit=5),
    }


def build_shadow_gods_summary(
    *,
    repo_root: Path,
    profile: dict[str, Any],
    operation: dict[str, Any],
    vigilance: dict[str, Any],
    prices_dir: Path,
    cycle_run_id: str,
    agent_run_id: str,
    as_of_date: str,
) -> dict[str, Any]:
    state_root = repo_root / "results" / "ops" / "shadow_gods" / "state"
    history_root = repo_root / "results" / "ops" / "shadow_gods" / "history"
    capital_blocks = profile.get("capital_blocks", []) if isinstance(profile.get("capital_blocks"), list) else []
    gods = profile.get("gods", []) if isinstance(profile.get("gods"), list) else []
    god_rows = []
    for god in gods:
        if not isinstance(god, dict):
            continue
        scenarios = []
        for block in capital_blocks:
            if not isinstance(block, dict):
                continue
            scenarios.append(
                run_shadow_scenario(
                    repo_root=repo_root,
                    god=god,
                    profile=profile,
                    operation=operation,
                    vigilance=vigilance,
                    prices_dir=prices_dir,
                    capital_brl=_safe_float(block.get("capital_brl"), 0.0),
                    cycle_run_id=cycle_run_id,
                    agent_run_id=agent_run_id,
                    as_of_date=as_of_date,
                    state_root=state_root,
                    history_root=history_root,
                )
            )
        god_rows.append(
            {
                "alias": str(god.get("alias") or ""),
                "candidate_id": str(god.get("candidate_id") or ""),
                "role": str(god.get("role") or ""),
                "thesis": str(god.get("thesis") or ""),
                "scenarios": scenarios,
                "order_count_total": sum(int(row.get("order_count") or 0) for row in scenarios),
                "fill_count_total": sum(int(row.get("fill_count") or 0) for row in scenarios),
                "latest_states": {str(int(round(row.get("capital_brl") or 0))): str(row.get("market_state") or "") for row in scenarios},
            }
        )
    return {
        "status": "ok",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "cycle_run_id": cycle_run_id,
        "agent_run_id": agent_run_id,
        "as_of_date": as_of_date,
        "gods": god_rows,
        "overview": {
            "god_count": len(god_rows),
            "scenario_count": sum(len(row.get("scenarios", [])) for row in god_rows),
            "order_count_total": sum(int(row.get("order_count_total") or 0) for row in god_rows),
            "fill_count_total": sum(int(row.get("fill_count_total") or 0) for row in god_rows),
        },
    }
