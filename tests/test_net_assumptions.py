from __future__ import annotations

from pathlib import Path

import pandas as pd

from execution.net_assumptions import apply_net_assumptions, blend_profiles, load_net_assumption_profiles


def test_load_net_assumption_profiles_reads_config() -> None:
    payload = load_net_assumption_profiles(Path("config/profit_net_assumptions.json"))

    assert payload["version"] == "profit_net_assumptions_v1"
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
