from __future__ import annotations

from execution.live_ops import (
    PortfolioPosition,
    PortfolioState,
    build_target_allocation,
    compile_order_tickets,
    reconcile_execution,
    select_live_mode,
)


def _profile() -> dict:
    return {
        "live_architecture": {
            "core": {"candidate_id": "core", "alias": "Athena"},
            "guard": {"candidate_id": "guard", "alias": "Hestia"},
            "attack": {"candidate_id": "attack", "alias": "Ares"},
        },
        "execution_profile": {
            "allowed_tickers": ["BTC-USD", "ETH-USD", "SOL-USD"],
            "cash_ticker": "CASH-BRL",
            "min_order_brl": 40.0,
            "min_position_brl": 80.0,
            "min_actionable_gross_exposure": 0.15,
            "min_two_asset_risk_brl": 180.0,
            "max_assets": 2,
            "core_proxy_weights": {"BTC-USD": 0.85, "ETH-USD": 0.15},
            "attack_proxy_weights": {"BTC-USD": 0.7, "ETH-USD": 0.3},
            "guard_proxy_weights": {"CASH-BRL": 1.0},
        },
        "selection_rules": {
            "attack_min_confidence_score": 0.62,
            "attack_min_gross_exposure": 0.3,
            "core_floor_gross_exposure": 0.2,
            "core_max_gross_exposure": 0.45,
            "attack_max_gross_exposure": 0.8,
        },
        "risk_guardrails": {
            "max_gross_exposure": 0.8,
            "max_assets": 2,
            "max_turnover_fraction_of_nav": 0.65,
            "block_new_attack_vigilance_statuses": ["warn", "fail"],
            "block_new_attack_when_confidence_below": 0.55,
            "force_cash_when_recommended_mode": "proteção",
        },
    }


def _portfolio() -> PortfolioState:
    return PortfolioState(
        as_of_date="2026-03-15",
        base_currency="BRL",
        cash_brl=400.0,
        fx_rates={"USD_BRL": 5.0},
        positions=[],
        open_orders=[],
    )


def _attack_operation() -> dict:
    return {
        "recommended_live_mode": {"mode": "ataque", "confidence_score": 0.7},
        "mode_confidence": {"metrics": {"vigilance_status": "ok"}, "confidence_score": 0.7},
        "mode_attack": {"gross_exposure": 0.5},
    }


def test_selects_guard_when_recommended_protection() -> None:
    operation = {
        "recommended_live_mode": {"mode": "proteção", "confidence_score": 0.8},
        "mode_confidence": {"metrics": {"vigilance_status": "ok"}, "confidence_score": 0.8},
        "mode_attack": {"gross_exposure": 0.5},
    }
    selected = select_live_mode(operation, _profile())
    assert selected["mode_key"] == "guard"


def test_simplifies_to_btc_only_for_small_risk_capital() -> None:
    operation = {
        "recommended_live_mode": {"mode": "ataque", "confidence_score": 0.61},
        "mode_confidence": {"metrics": {"vigilance_status": "ok"}, "confidence_score": 0.61},
        "mode_attack": {"gross_exposure": 0.2},
    }
    target = build_target_allocation(operation, _profile(), _portfolio())
    assert target["BTC-USD"] > 0.0
    assert target.get("ETH-USD", 0.0) == 0.0


def test_compiles_buy_orders_against_cash() -> None:
    prices = {"BTC-USD": {"price_brl": 500000.0}, "ETH-USD": {"price_brl": 20000.0}}
    compiled = compile_order_tickets(_attack_operation(), _profile(), _portfolio(), prices)
    assert compiled["status"] == "ok"
    assert compiled["tickets"]
    assert all(ticket["side"] == "buy" for ticket in compiled["tickets"])


def test_reconcile_matches_fill_ratio() -> None:
    plan = {
        "run_id": "abc",
        "tickets": [
            {
                "ticket_id": "t1",
                "ticker": "BTC-USD",
                "side": "buy",
                "notional_brl": 100.0,
            }
        ],
    }
    report = {
        "executions": [
            {
                "ticket_id": "t1",
                "ticker": "BTC-USD",
                "side": "buy",
                "filled_notional_brl": 90.0,
                "avg_price_brl": 510000.0,
                "fee_brl": 1.0,
                "status": "filled",
            }
        ]
    }
    reconciled = reconcile_execution(plan, report)
    assert reconciled["ticket_count"] == 1
    assert reconciled["reconciled_rows"][0]["fill_ratio"] == 0.9
