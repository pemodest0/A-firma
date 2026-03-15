from __future__ import annotations

import csv
import json
import os
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable


JsonRequestFn = Callable[..., dict[str, Any]]


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


def _adapter(profile: dict[str, Any]) -> dict[str, Any]:
    adapter = profile.get("broker_adapter", {}) if isinstance(profile.get("broker_adapter"), dict) else {}
    return adapter


def _join_url(base_url: str, path: str) -> str:
    return urllib.parse.urljoin(str(base_url).rstrip("/") + "/", str(path).lstrip("/"))


def _request_json(
    *,
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
    timeout_sec: float = 20.0,
) -> dict[str, Any]:
    body = None
    req_headers = {"Accept": "application/json"}
    if headers:
        req_headers.update({str(k): str(v) for k, v in headers.items()})
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        req_headers.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=body, headers=req_headers, method=str(method).upper())
    with urllib.request.urlopen(req, timeout=float(timeout_sec)) as response:
        raw = response.read().decode("utf-8")
    if not raw.strip():
        return {}
    decoded = json.loads(raw)
    return decoded if isinstance(decoded, dict) else {"data": decoded}


def _response_list(payload: dict[str, Any] | list[Any] | None) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("data", "results", "items", "balances", "orders", "accounts"):
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    return []


def _read_env(profile: dict[str, Any], env: dict[str, str] | None = None) -> dict[str, Any]:
    adapter = _adapter(profile)
    auth = adapter.get("auth", {}) if isinstance(adapter.get("auth"), dict) else {}
    env_map = auth.get("env", {}) if isinstance(auth.get("env"), dict) else {}
    store = env if env is not None else os.environ
    values: dict[str, str] = {}
    missing: list[str] = []
    for field, env_name in env_map.items():
        raw = str(store.get(str(env_name), "")).strip()
        if raw:
            values[str(field)] = raw
        else:
            missing.append(str(field))
    account_id = str(store.get(str(auth.get("account_id_env") or "MB_ACCOUNT_ID"), "")).strip()
    if account_id:
        values["account_id"] = account_id
    return {
        "status": "ok" if not missing else "missing_credentials",
        "values": values,
        "missing_fields": missing,
        "account_id": values.get("account_id", ""),
    }


def load_mercado_bitcoin_credentials(profile: dict[str, Any], env: dict[str, str] | None = None) -> dict[str, Any]:
    out = _read_env(profile, env=env)
    out["broker"] = str(_adapter(profile).get("name") or "mercado_bitcoin")
    return out


def _extract_token(payload: dict[str, Any], adapter: dict[str, Any]) -> str:
    auth = adapter.get("auth", {}) if isinstance(adapter.get("auth"), dict) else {}
    token_paths = auth.get("token_paths", [])
    if not isinstance(token_paths, list):
        token_paths = []
    candidates = token_paths or ["access_token", "token", "data.access_token", "data.token"]
    for path in candidates:
        node: Any = payload
        ok = True
        for part in str(path).split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                ok = False
                break
        if ok and str(node or "").strip():
            return str(node).strip()
    return ""


def authorize_mercado_bitcoin(
    profile: dict[str, Any],
    *,
    env: dict[str, str] | None = None,
    request_json: JsonRequestFn = _request_json,
) -> dict[str, Any]:
    adapter = _adapter(profile)
    creds = _read_env(profile, env=env)
    if creds["status"] != "ok":
        return {
            "status": "missing_credentials",
            "broker": str(adapter.get("name") or "mercado_bitcoin"),
            "missing_fields": creds["missing_fields"],
        }
    auth = adapter.get("auth", {}) if isinstance(adapter.get("auth"), dict) else {}
    base_url = str(auth.get("base_url") or adapter.get("base_url") or "").strip()
    authorize_endpoint = str(auth.get("authorize_endpoint") or "").strip()
    if not base_url or not authorize_endpoint:
        return {
            "status": "misconfigured",
            "broker": str(adapter.get("name") or "mercado_bitcoin"),
            "reason": "missing_base_url_or_authorize_endpoint",
        }
    payload_fields = auth.get("payload_fields", {}) if isinstance(auth.get("payload_fields"), dict) else {}
    payload: dict[str, Any] = {}
    for field, source_key in payload_fields.items():
        value = creds["values"].get(str(source_key), "")
        if str(value).strip():
            payload[str(field)] = str(value).strip()
    if not payload:
        payload = dict(creds["values"])
    response = request_json(
        method="POST",
        url=_join_url(base_url, authorize_endpoint),
        payload=payload,
        timeout_sec=_safe_float(auth.get("timeout_sec"), 20.0),
    )
    token = _extract_token(response, adapter)
    if not token:
        return {
            "status": "auth_failed",
            "broker": str(adapter.get("name") or "mercado_bitcoin"),
            "response_excerpt": json.dumps(response, ensure_ascii=False)[:1000],
        }
    return {
        "status": "ok",
        "broker": str(adapter.get("name") or "mercado_bitcoin"),
        "token": token,
        "account_id": creds.get("account_id", ""),
    }


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {str(token).strip()}"}


def _first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row.get(key) is not None:
            return row.get(key)
    return None


def _reverse_supported_pairs(adapter: dict[str, Any]) -> dict[str, str]:
    supported_pairs = adapter.get("supported_pairs", {}) if isinstance(adapter.get("supported_pairs"), dict) else {}
    reverse: dict[str, str] = {}
    for ticker, market in supported_pairs.items():
        market_s = str(market)
        base_asset = market_s.split("-")[0].strip().upper()
        if base_asset:
            reverse[base_asset] = str(ticker)
    return reverse


def _parse_accounts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _response_list(payload)
    if rows:
        return rows
    if isinstance(payload, dict) and any(key in payload for key in ("id", "account_id", "uuid")):
        return [payload]
    return []


def _resolve_account_id(accounts: list[dict[str, Any]], preferred_account_id: str) -> str:
    preferred = str(preferred_account_id or "").strip()
    if preferred:
        for row in accounts:
            rid = str(_first_present(row, "id", "account_id", "uuid") or "").strip()
            if rid == preferred:
                return rid
    for row in accounts:
        rid = str(_first_present(row, "id", "account_id", "uuid") or "").strip()
        if rid:
            return rid
    return preferred


def _parse_balances(payload: dict[str, Any], adapter: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    reverse_pairs = _reverse_supported_pairs(adapter)
    rows = _response_list(payload)
    cash_brl = 0.0
    positions: list[dict[str, Any]] = []
    for row in rows:
        currency = str(_first_present(row, "currency", "symbol", "asset", "coin", "currency_code") or "").strip().upper()
        total = _safe_float(_first_present(row, "total", "balance", "amount", "quantity", "available_balance"), 0.0)
        available = _safe_float(_first_present(row, "available", "free", "available_balance", "amount_available"), total)
        if total <= 0.0 and available <= 0.0:
            continue
        if currency == "BRL":
            cash_brl += max(total, available)
            continue
        ticker = reverse_pairs.get(currency)
        if not ticker:
            continue
        positions.append(
            {
                "ticker": ticker,
                "asset_code": currency,
                "quantity": max(total, available),
                "available_quantity": available,
            }
        )
    return float(cash_brl), positions


def _extract_orderbook_price(payload: dict[str, Any]) -> float | None:
    bids = payload.get("bids") if isinstance(payload.get("bids"), list) else []
    asks = payload.get("asks") if isinstance(payload.get("asks"), list) else []

    def _price(levels: list[Any]) -> float | None:
        if not levels:
            return None
        first = levels[0]
        if isinstance(first, list) and first:
            return _safe_float(first[0], 0.0) or None
        if isinstance(first, dict):
            return _safe_float(_first_present(first, "price", "rate", "value"), 0.0) or None
        return None

    bid = _price(bids)
    ask = _price(asks)
    if bid and ask:
        return float((bid + ask) / 2.0)
    return ask or bid


def _public_market_price(
    *,
    adapter: dict[str, Any],
    market: str,
    request_json: JsonRequestFn,
) -> float | None:
    public_cfg = adapter.get("public_market_data", {}) if isinstance(adapter.get("public_market_data"), dict) else {}
    base_url = str(public_cfg.get("base_url") or adapter.get("base_url") or "").strip()
    template = str(public_cfg.get("orderbook_endpoint_template") or "").strip()
    if not base_url or not template:
        return None
    payload = request_json(
        method="GET",
        url=_join_url(base_url, template.format(market=str(market))),
        timeout_sec=_safe_float(public_cfg.get("timeout_sec"), 10.0),
    )
    return _extract_orderbook_price(payload)


def _fetch_open_orders(
    *,
    adapter: dict[str, Any],
    base_url: str,
    token: str,
    account_id: str,
    request_json: JsonRequestFn,
) -> list[dict[str, Any]]:
    endpoints = adapter.get("private_endpoints", {}) if isinstance(adapter.get("private_endpoints"), dict) else {}
    template = str(endpoints.get("market_orders_endpoint_template") or "").strip()
    if not template or not account_id:
        return []
    supported_pairs = adapter.get("supported_pairs", {}) if isinstance(adapter.get("supported_pairs"), dict) else {}
    open_orders: list[dict[str, Any]] = []
    for ticker, market in supported_pairs.items():
        try:
            payload = request_json(
                method="GET",
                url=_join_url(base_url, template.format(account_id=account_id, market=str(market))),
                headers=_auth_headers(token),
                timeout_sec=_safe_float(endpoints.get("timeout_sec"), 20.0),
            )
        except Exception as exc:  # noqa: BLE001
            open_orders.append(
                {
                    "ticker": str(ticker),
                    "market": str(market),
                    "status": "fetch_failed",
                    "error": str(exc),
                }
            )
            continue
        rows = _response_list(payload)
        if not rows and isinstance(payload, dict):
            state = str(_first_present(payload, "status", "state") or "").strip().lower()
            if state:
                rows = [payload]
        for row in rows:
            status = str(_first_present(row, "status", "state") or "").strip().lower()
            if status and status not in {"open", "pending", "partially_filled", "new"}:
                continue
            open_orders.append(
                {
                    "ticker": str(ticker),
                    "market": str(market),
                    "broker_order_id": str(_first_present(row, "id", "order_id", "uuid") or "").strip(),
                    "side": str(_first_present(row, "side", "type") or "").strip().lower(),
                    "status": status or "open",
                    "price": _safe_float(_first_present(row, "price", "limit_price", "rate"), 0.0),
                    "quantity": _safe_float(_first_present(row, "quantity", "amount", "executed_quantity"), 0.0),
                    "raw": row,
                }
            )
    return open_orders


def fetch_mercado_bitcoin_account_snapshot(
    profile: dict[str, Any],
    *,
    portfolio_state: dict[str, Any] | None = None,
    env: dict[str, str] | None = None,
    request_json: JsonRequestFn = _request_json,
) -> dict[str, Any]:
    adapter = _adapter(profile)
    auth_out = authorize_mercado_bitcoin(profile, env=env, request_json=request_json)
    if auth_out.get("status") != "ok":
        return auth_out

    auth = adapter.get("auth", {}) if isinstance(adapter.get("auth"), dict) else {}
    private_endpoints = adapter.get("private_endpoints", {}) if isinstance(adapter.get("private_endpoints"), dict) else {}
    base_url = str(auth.get("base_url") or adapter.get("base_url") or "").strip()
    accounts_endpoint = str(private_endpoints.get("accounts_endpoint") or "").strip()
    balances_template = str(private_endpoints.get("balances_endpoint_template") or "").strip()
    if not base_url or not accounts_endpoint or not balances_template:
        return {
            "status": "misconfigured",
            "broker": str(adapter.get("name") or "mercado_bitcoin"),
            "reason": "missing_private_endpoints",
        }

    accounts_payload = request_json(
        method="GET",
        url=_join_url(base_url, accounts_endpoint),
        headers=_auth_headers(str(auth_out.get("token") or "")),
        timeout_sec=_safe_float(private_endpoints.get("timeout_sec"), 20.0),
    )
    accounts = _parse_accounts(accounts_payload)
    account_id = _resolve_account_id(accounts, str(auth_out.get("account_id") or ""))
    if not account_id:
        return {
            "status": "no_account",
            "broker": str(adapter.get("name") or "mercado_bitcoin"),
            "accounts_response": accounts_payload,
        }

    balances_payload = request_json(
        method="GET",
        url=_join_url(base_url, balances_template.format(account_id=account_id)),
        headers=_auth_headers(str(auth_out.get("token") or "")),
        timeout_sec=_safe_float(private_endpoints.get("timeout_sec"), 20.0),
    )
    cash_brl, positions = _parse_balances(balances_payload, adapter)
    supported_pairs = adapter.get("supported_pairs", {}) if isinstance(adapter.get("supported_pairs"), dict) else {}
    existing_positions = {
        str(row.get("ticker") or ""): row
        for row in (portfolio_state or {}).get("positions", [])
        if isinstance(row, dict) and str(row.get("ticker") or "").strip()
    }
    priced_positions: list[dict[str, Any]] = []
    for pos in positions:
        ticker = str(pos["ticker"])
        market = str(supported_pairs.get(ticker) or "")
        market_price_brl = _public_market_price(adapter=adapter, market=market, request_json=request_json) if market else None
        existing = existing_positions.get(ticker, {})
        avg_price_brl = _safe_float(existing.get("avg_price_brl"), _safe_float(market_price_brl, 0.0))
        price_brl = _safe_float(market_price_brl, avg_price_brl)
        priced_positions.append(
            {
                "ticker": ticker,
                "quantity": float(pos["quantity"]),
                "market_value_brl": float(pos["quantity"]) * float(price_brl) if price_brl > 0.0 else 0.0,
                "avg_price_brl": float(avg_price_brl),
                "available_quantity": float(pos["available_quantity"]),
                "price_brl_reference": float(price_brl) if price_brl > 0.0 else 0.0,
            }
        )

    open_orders = _fetch_open_orders(
        adapter=adapter,
        base_url=base_url,
        token=str(auth_out.get("token") or ""),
        account_id=account_id,
        request_json=request_json,
    )
    fx_rates = (portfolio_state or {}).get("fx_rates", {}) if isinstance((portfolio_state or {}).get("fx_rates"), dict) else {}
    return {
        "status": "ok",
        "broker": str(adapter.get("name") or "mercado_bitcoin"),
        "mode": str(adapter.get("mode") or "preview_only"),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "account_id": str(account_id),
        "account_count": int(len(accounts)),
        "portfolio_state": {
            "as_of_date": datetime.now(UTC).date().isoformat(),
            "base_currency": "BRL",
            "cash_brl": float(cash_brl),
            "fx_rates": {str(k): float(v) for k, v in fx_rates.items()},
            "positions": priced_positions,
            "open_orders": open_orders,
        },
        "balances_raw_count": int(len(_response_list(balances_payload))),
        "positions_count": int(len(priced_positions)),
        "open_orders_count": int(len(open_orders)),
    }


def build_mercado_bitcoin_preview(plan: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    adapter = _adapter(profile)
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
        "Revise market, side e notional_brl antes de enviar qualquer ordem.",
        "CASH-BRL nao gera ordem; ativos fora do mapa ficam como unsupported_market.",
    ]
    if str(adapter.get("mode") or "").strip().lower() != "preview_only":
        notes.append("O adapter ja consegue sincronizar conta real; submit ainda depende de confirmacao explicita.")
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
        "portfolio_source": plan.get("portfolio_source", {}),
        "order_count": len(export_rows),
        "unsupported_count": len(unsupported),
        "orders": export_rows,
        "unsupported": unsupported,
        "manual_steps": [
            "Conferir se o sync da carteira da corretora foi recente antes de agir.",
            "Comparar latest_execution_plan.json com este preview antes de enviar.",
            "Executar manualmente ou via submit assistido apenas as ordens que continuarem fazendo sentido no book real.",
        ],
        "notes": notes,
    }


def _submission_payload(order: dict[str, Any], adapter: dict[str, Any]) -> dict[str, Any]:
    order_fields = adapter.get("order_fields", {}) if isinstance(adapter.get("order_fields"), dict) else {}
    payload: dict[str, Any] = {}
    side_field = str(order_fields.get("side_field") or "side")
    type_field = str(order_fields.get("type_field") or "type")
    client_field = str(order_fields.get("client_order_id_field") or "external_id")
    quantity_field = str(order_fields.get("quantity_field") or "quantity")
    notional_field = str(order_fields.get("notional_field") or "cost")

    payload[client_field] = str(order.get("client_order_id") or "")
    payload[side_field] = str(order.get("side") or "").lower()
    raw_type = str(order.get("order_type") or "").lower()
    payload[type_field] = "market" if "market" in raw_type else raw_type or "market"

    quantity = _safe_float(order.get("quantity_estimate"), 0.0)
    notional_brl = _safe_float(order.get("notional_brl"), 0.0)
    prefer_notional_for_buy = bool((adapter.get("order_fields") or {}).get("prefer_notional_for_buy", True))
    if payload[side_field] == "buy" and prefer_notional_for_buy and notional_brl > 0.0:
        payload[notional_field] = round(notional_brl, 2)
    elif quantity > 0.0:
        payload[quantity_field] = quantity
    elif notional_brl > 0.0:
        payload[notional_field] = round(notional_brl, 2)
    return payload


def submit_mercado_bitcoin_orders(
    preview: dict[str, Any],
    profile: dict[str, Any],
    *,
    env: dict[str, str] | None = None,
    request_json: JsonRequestFn = _request_json,
) -> dict[str, Any]:
    adapter = _adapter(profile)
    if adapter.get("submit_enabled") is not True:
        return {
            "status": "blocked",
            "broker": str(adapter.get("name") or "mercado_bitcoin"),
            "reason": "submit_disabled_in_profile",
        }
    auth_out = authorize_mercado_bitcoin(profile, env=env, request_json=request_json)
    if auth_out.get("status") != "ok":
        return auth_out

    base_url = str((adapter.get("auth") or {}).get("base_url") or adapter.get("base_url") or "").strip()
    private_endpoints = adapter.get("private_endpoints", {}) if isinstance(adapter.get("private_endpoints"), dict) else {}
    orders_template = str(private_endpoints.get("market_orders_endpoint_template") or "").strip()
    if not base_url or not orders_template:
        return {
            "status": "misconfigured",
            "broker": str(adapter.get("name") or "mercado_bitcoin"),
            "reason": "missing_market_orders_endpoint_template",
        }

    account_snapshot = fetch_mercado_bitcoin_account_snapshot(profile, request_json=request_json)
    if account_snapshot.get("status") != "ok":
        return account_snapshot
    account_id = str(account_snapshot.get("account_id") or "")
    rows: list[dict[str, Any]] = []
    for order in preview.get("orders", []) if isinstance(preview.get("orders"), list) else []:
        if not isinstance(order, dict):
            continue
        market = str(order.get("market") or "").strip()
        if not market:
            continue
        payload = _submission_payload(order, adapter)
        try:
            response = request_json(
                method="POST",
                url=_join_url(base_url, orders_template.format(account_id=account_id, market=market)),
                headers=_auth_headers(str(auth_out.get("token") or "")),
                payload=payload,
                timeout_sec=_safe_float(private_endpoints.get("timeout_sec"), 20.0),
            )
            rows.append(
                {
                    "client_order_id": str(order.get("client_order_id") or ""),
                    "market": market,
                    "side": str(order.get("side") or ""),
                    "status": "submitted",
                    "response": response,
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "client_order_id": str(order.get("client_order_id") or ""),
                    "market": market,
                    "side": str(order.get("side") or ""),
                    "status": "submit_failed",
                    "error": str(exc),
                }
            )
    ok_count = sum(1 for row in rows if str(row.get("status") or "") == "submitted")
    return {
        "status": "ok" if ok_count == len(rows) and rows else ("partial" if ok_count else "failed"),
        "broker": str(adapter.get("name") or "mercado_bitcoin"),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "account_id": account_id,
        "submitted_count": ok_count,
        "order_count": len(rows),
        "rows": rows,
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


def write_mercado_bitcoin_snapshot(snapshot: dict[str, Any], *, json_path: str | Path) -> None:
    _write_json(Path(json_path).resolve(), snapshot)


def write_mercado_bitcoin_submission(
    payload: dict[str, Any],
    *,
    json_path: str | Path,
    csv_path: str | Path,
) -> None:
    _write_json(Path(json_path).resolve(), payload)
    rows = payload.get("rows", []) if isinstance(payload.get("rows"), list) else []
    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        flat_rows.append(
            {
                "client_order_id": str(row.get("client_order_id") or ""),
                "market": str(row.get("market") or ""),
                "side": str(row.get("side") or ""),
                "status": str(row.get("status") or ""),
                "error": str(row.get("error") or ""),
            }
        )
    write_csv(csv_path, flat_rows)
