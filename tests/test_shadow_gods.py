from __future__ import annotations

import json
from pathlib import Path

from execution.shadow_gods import (
    build_shadow_target_weights,
    infer_shadow_market_state,
    load_shadow_gods_profile,
    run_shadow_scenario,
)


def _write_price_csv(path: Path, price: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"date,price\n2026-03-14,{price}\n", encoding="utf-8")


def test_infer_shadow_market_state_prefers_defense_on_protection() -> None:
    operation = {
        "recommended_live_mode": {"mode": "proteção", "confidence_score": 0.7},
        "mode_confidence": {"metrics": {"vigilance_status": "warn"}},
        "official_structural_regime": {"regime": "stress"},
    }
    state, notes = infer_shadow_market_state(operation, {}, role="turbo_attack_shadow")
    assert state == "defense"
    assert notes


def test_build_shadow_target_weights_trims_to_max_assets() -> None:
    god = {
        "allocations": {
            "risk_on": {
                "SPY": 0.22,
                "QQQ": 0.13,
                "BTC-USD": 0.20,
                "ETH-USD": 0.08,
                "PETR4.SA": 0.10,
                "VALE3.SA": 0.10,
                "ITUB4.SA": 0.07,
                "SHY": 0.10,
            }
        }
    }
    prices = {ticker: {"price_brl": 100.0} for ticker in ["SPY", "QQQ", "BTC-USD", "ETH-USD", "PETR4.SA", "VALE3.SA", "ITUB4.SA", "SHY"]}
    weights, _ = build_shadow_target_weights(god, market_state="risk_on", max_assets=2, prices=prices)
    non_cash = [ticker for ticker in weights if ticker != "CASH-BRL"]
    assert len(non_cash) == 2
    assert round(sum(weights.values()), 8) == 1.0


def test_run_shadow_scenario_persists_orders_and_state(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    profile_path = repo / "config" / "shadow_gods_portfolios.json"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "base_currency": "BRL",
        "default_fx_rates": {"USD_BRL": 5.0},
        "capital_blocks": [{"capital_brl": 200.0, "max_assets": 2, "min_order_brl": 40.0, "min_position_brl": 80.0}],
        "fee_bps_by_asset_class": {"crypto": 25.0, "international": 10.0, "brazil": 15.0, "cash": 0.0},
        "ticker_asset_class": {"BTC-USD": "crypto", "ETH-USD": "crypto", "CASH-BRL": "cash"},
        "gods": [
            {
                "alias": "Hermes",
                "candidate_id": "all_assets__lb021__rb07__k3__mom_vol_adj__ama000__mma000__relshy",
                "role": "turbo_attack_shadow",
                "thesis": "Turbo",
                "allocations": {
                    "risk_on": {"BTC-USD": 0.7, "ETH-USD": 0.3},
                    "balanced": {"BTC-USD": 0.4, "CASH-BRL": 0.6},
                    "opportunistic": {"BTC-USD": 0.5, "ETH-USD": 0.5},
                    "defense": {"CASH-BRL": 1.0},
                },
            }
        ],
    }
    profile_path.write_text(json.dumps(config), encoding="utf-8")
    profile = load_shadow_gods_profile(profile_path)
    prices_dir = repo / "data/raw/finance/yfinance_daily"
    _write_price_csv(prices_dir / "BTC-USD.csv", 50000.0)
    _write_price_csv(prices_dir / "ETH-USD.csv", 2000.0)
    operation = {
        "recommended_live_mode": {"mode": "ataque", "confidence_score": 0.8, "current_posture": {"posture": "ataque_pleno"}},
        "mode_confidence": {"metrics": {"vigilance_status": "ok"}},
        "official_structural_regime": {"regime": "stable"},
    }
    summary = run_shadow_scenario(
        repo_root=repo,
        god=profile["gods"][0],
        profile=profile,
        operation=operation,
        vigilance={},
        prices_dir=prices_dir,
        capital_brl=200.0,
        cycle_run_id="20260315T000000Z",
        agent_run_id="20260315T010000Z",
        as_of_date="2026-03-15",
        state_root=repo / "results/ops/shadow_gods/state",
        history_root=repo / "results/ops/shadow_gods/history",
    )
    assert summary["scenario_id"] == "hermes_200"
    assert summary["nav_after_brl"] > 0.0
    assert Path(summary["artifacts"]["state_path"]).exists()
    assert Path(summary["artifacts"]["history_csv"]).exists()
