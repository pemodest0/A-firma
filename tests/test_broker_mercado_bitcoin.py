from __future__ import annotations

from execution.broker_mercado_bitcoin import build_mercado_bitcoin_preview


def _profile():
    return {
        "broker_adapter": {
            "name": "mercado_bitcoin",
            "mode": "preview_only",
            "submit_enabled": False,
            "supported_pairs": {
                "BTC-USD": "BTC-BRL",
                "ETH-USD": "ETH-BRL",
                "SOL-USD": "SOL-BRL",
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
