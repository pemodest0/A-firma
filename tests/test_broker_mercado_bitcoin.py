from __future__ import annotations

from execution.broker_mercado_bitcoin import (
    build_mercado_bitcoin_preview,
    fetch_mercado_bitcoin_account_snapshot,
    load_mercado_bitcoin_credentials,
    submit_mercado_bitcoin_orders,
)


def _profile(submit_enabled: bool = False):
    return {
        "broker_adapter": {
            "name": "mercado_bitcoin",
            "mode": "manual_assisted",
            "submit_enabled": submit_enabled,
            "base_url": "https://api.mercadobitcoin.net",
            "supported_pairs": {
                "BTC-USD": "BTC-BRL",
                "ETH-USD": "ETH-BRL",
                "SOL-USD": "SOL-BRL",
            },
            "auth": {
                "base_url": "https://api.mercadobitcoin.net",
                "authorize_endpoint": "/api/v4/authorize/",
                "env": {
                    "api_key": "MB_API_KEY",
                    "password": "MB_API_PASSWORD",
                },
                "payload_fields": {
                    "api_key": "api_key",
                    "password": "password",
                },
                "token_paths": ["access_token"],
            },
            "private_endpoints": {
                "accounts_endpoint": "/api/v4/accounts/",
                "balances_endpoint_template": "/api/v4/accounts/{account_id}/balances/",
                "market_orders_endpoint_template": "/api/v4/accounts/{account_id}/{market}/orders/",
            },
            "public_market_data": {
                "base_url": "https://api.mercadobitcoin.net",
                "orderbook_endpoint_template": "/api/v4/{market}/orderbook/",
            },
            "order_fields": {
                "client_order_id_field": "external_id",
                "side_field": "side",
                "type_field": "type",
                "quantity_field": "quantity",
                "notional_field": "cost",
                "prefer_notional_for_buy": True,
            },
        }
    }


def test_build_mercado_bitcoin_preview_maps_supported_orders():
    plan = {
        "run_id": "RUN1",
        "cycle_run_id": "CYCLE1",
        "generated_at_utc": "2026-03-15T12:00:00Z",
        "selected_mode": {"alias": "Ares"},
        "tickets": [
            {
                "ticket_id": "t1",
                "ticker": "BTC-USD",
                "side": "buy",
                "notional_brl": 220.0,
                "quantity_estimate": 0.0011,
                "price_brl_reference": 200000.0,
                "preferred_order_type": "market_notional",
                "reason": "align_to_attack",
            }
        ],
    }
    preview = build_mercado_bitcoin_preview(plan, _profile())
    assert preview["status"] == "ok"
    assert preview["order_count"] == 1
    assert preview["orders"][0]["market"] == "BTC-BRL"
    assert preview["orders"][0]["source_ticker"] == "BTC-USD"


def test_build_mercado_bitcoin_preview_marks_unsupported_tickers():
    plan = {
        "run_id": "RUN2",
        "tickets": [
            {
                "ticket_id": "t2",
                "ticker": "SHY",
                "side": "buy",
                "notional_brl": 300.0,
            }
        ],
    }
    preview = build_mercado_bitcoin_preview(plan, _profile())
    assert preview["status"] == "no_action"
    assert preview["order_count"] == 0
    assert preview["unsupported_count"] == 1
    assert preview["unsupported"][0]["reason"] == "unsupported_market"


def test_load_mercado_bitcoin_credentials_reads_env_mapping():
    creds = load_mercado_bitcoin_credentials(
        _profile(),
        env={
            "MB_API_KEY": "key-1",
            "MB_API_PASSWORD": "pw-1",
            "MB_ACCOUNT_ID": "acc-9",
        },
    )
    assert creds["status"] == "ok"
    assert creds["values"]["api_key"] == "key-1"
    assert creds["values"]["password"] == "pw-1"
    assert creds["account_id"] == "acc-9"


def test_fetch_mercado_bitcoin_account_snapshot_builds_portfolio_state():
    calls: list[str] = []

    def fake_request_json(*, method, url, headers=None, payload=None, timeout_sec=0):
        calls.append(f"{method} {url}")
        if url.endswith("/authorize/"):
            return {"access_token": "token-1"}
        if url.endswith("/accounts/"):
            return {"accounts": [{"id": "acc-1"}]}
        if url.endswith("/accounts/acc-1/balances/"):
            return {
                "balances": [
                    {"currency": "BRL", "available": "123.45"},
                    {"currency": "BTC", "total": "0.01", "available": "0.01"},
                ]
            }
        if url.endswith("/api/v4/BTC-BRL/orderbook/"):
            return {"bids": [[200000.0, 0.5]], "asks": [[202000.0, 0.4]]}
        if url.endswith("/api/v4/ETH-BRL/orderbook/"):
            return {"bids": [[10000.0, 1.0]], "asks": [[10100.0, 1.0]]}
        if url.endswith("/api/v4/SOL-BRL/orderbook/"):
            return {"bids": [[700.0, 5.0]], "asks": [[705.0, 5.0]]}
        if url.endswith("/accounts/acc-1/BTC-BRL/orders/"):
            return {"orders": []}
        if url.endswith("/accounts/acc-1/ETH-BRL/orders/"):
            return {"orders": []}
        if url.endswith("/accounts/acc-1/SOL-BRL/orders/"):
            return {"orders": []}
        raise AssertionError(url)

    snapshot = fetch_mercado_bitcoin_account_snapshot(
        _profile(),
        portfolio_state={"fx_rates": {"USD_BRL": 5.0}, "positions": [], "open_orders": []},
        env={"MB_API_KEY": "key", "MB_API_PASSWORD": "pw"},
        request_json=fake_request_json,
    )
    assert snapshot["status"] == "ok"
    assert snapshot["account_id"] == "acc-1"
    assert snapshot["portfolio_state"]["cash_brl"] == 123.45
    assert snapshot["portfolio_state"]["positions"][0]["ticker"] == "BTC-USD"
    assert snapshot["portfolio_state"]["positions"][0]["market_value_brl"] > 0.0
    assert any(part.endswith("/balances/") for part in calls)


def test_submit_mercado_bitcoin_orders_stays_blocked_when_disabled():
    preview = {
        "orders": [
            {
                "client_order_id": "oid-1",
                "market": "BTC-BRL",
                "side": "buy",
                "order_type": "market_notional",
                "notional_brl": 150.0,
                "quantity_estimate": 0.001,
            }
        ]
    }
    payload = submit_mercado_bitcoin_orders(preview, _profile(submit_enabled=False))
    assert payload["status"] == "blocked"
    assert payload["reason"] == "submit_disabled_in_profile"
