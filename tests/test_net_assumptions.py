from __future__ import annotations

from pathlib import Path

import pandas as pd

from execution.net_assumptions import apply_net_assumptions, blend_profiles, load_net_assumption_profiles


def test_load_net_assumption_profiles_reads_config() -> None:
    payload = load_net_assumption_profiles(Path("config/profit_net_assumptions.json"))

    assert payload["version"] == "profit_net_assumptions_v3"
    assert "foreign_financial_brazil_resident" in payload["profiles"]
    assert "br_local_equity" in payload["profiles"]


def test_apply_net_assumptions_supports_annual_positive_proxy() -> None:
    payload = load_net_assumption_profiles(Path("config/profit_net_assumptions.json"))
    profile = payload["profiles"]["foreign_financial_brazil_resident"]
    gross = pd.Series([0.10, -0.05, 0.02], index=["2025-01", "2025-02", "2026-01"], dtype=float)
    turnover = pd.Series([0.5, 0.0, 0.5], index=gross.index, dtype=float)

    out = apply_net_assumptions(gross, turnover, profile=profile, periods_index=gross.index)

    assert out["transaction_cost_ret"].sum() > 0.0
    assert out.loc["2025-01", "tax_ret"] > 0.0
    assert out.loc["2025-02", "tax_ret"] > 0.0
    assert out.loc["2026-01", "tax_ret"] > 0.0


def test_blend_profiles_uses_weighted_costs() -> None:
    payload = load_net_assumption_profiles(Path("config/profit_net_assumptions.json"))
    foreign = payload["profiles"]["foreign_financial_brazil_resident"]
    local = payload["profiles"]["br_local_equity"]

    blended = blend_profiles(0.75, foreign_profile=foreign, br_profile=local)

    assert blended.jurisdiction == "blended"
    assert blended.total_cost_bps_assumed > local.total_cost_bps_assumed
    assert blended.total_cost_bps_assumed < foreign.total_cost_bps_assumed


def test_apply_net_assumptions_adds_cash_yield_when_fully_in_cash() -> None:
    payload = load_net_assumption_profiles(Path("config/profit_net_assumptions.json"))
    profile = payload["profiles"]["foreign_financial_brazil_resident"]
    gross = pd.Series([0.0, 0.0, 0.0], index=pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]))
    turnover = pd.Series([0.0, 0.0, 0.0], index=gross.index, dtype=float)
    cash_weight = pd.Series([1.0, 1.0, 1.0], index=gross.index, dtype=float)

    out = apply_net_assumptions(gross, turnover, profile=profile, periods_index=gross.index, cash_weight=cash_weight)

    assert out["cash_ret"].sum() > 0.0
    assert out["net_ret"].sum() > 0.0


def test_apply_net_assumptions_honors_monthly_sales_exemption_proxy() -> None:
    payload = load_net_assumption_profiles(Path("config/profit_net_assumptions.json"))
    profile = payload["profiles"]["br_local_equity"]
    gross = pd.Series([0.02, 0.01], index=pd.to_datetime(["2025-01-02", "2025-01-03"]))
    turnover = pd.Series([0.10, 0.10], index=gross.index, dtype=float)

    out = apply_net_assumptions(
        gross,
        turnover,
        profile=profile,
        periods_index=gross.index,
        initial_capital_brl=10000.0,
    )

    assert float(out["tax_ret"].sum()) == 0.0
    assert out["withholding_ret"].sum() > 0.0


def test_apply_net_assumptions_inventory_proxy_taxes_only_realized_gain() -> None:
    payload = load_net_assumption_profiles(Path("config/profit_net_assumptions.json"))
    profile = payload["profiles"]["br_local_equity"]
    gross = pd.Series([0.05, 0.04], index=pd.to_datetime(["2025-02-03", "2025-02-04"]))
    turnover = pd.Series([0.0, 0.0], index=gross.index, dtype=float)

    out = apply_net_assumptions(
        gross,
        turnover,
        profile=profile,
        periods_index=gross.index,
        initial_capital_brl=100000.0,
    )

    assert float(out["tax_ret"].sum()) == 0.0
    assert float(out["withholding_ret"].sum()) == 0.0


def test_apply_net_assumptions_inventory_proxy_tax_applies_above_exemption() -> None:
    payload = load_net_assumption_profiles(Path("config/profit_net_assumptions.json"))
    profile = payload["profiles"]["br_local_equity"]
    gross = pd.Series([0.08, 0.03], index=pd.to_datetime(["2025-03-03", "2025-03-04"]))
    turnover = pd.Series([0.9, 0.9], index=gross.index, dtype=float)

    out = apply_net_assumptions(
        gross,
        turnover,
        profile=profile,
        periods_index=gross.index,
        initial_capital_brl=100000.0,
    )

    assert float(out["withholding_ret"].sum()) > 0.0
    assert float(out["tax_ret"].sum()) > 0.0
