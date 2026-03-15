from __future__ import annotations

import pandas as pd

from execution.net_assumptions import NetAssumptionProfile
from scripts.bench.validation.run_profit_small_account_dca import _simulate_dca_path, _window_slices


def _profile() -> NetAssumptionProfile:
    return NetAssumptionProfile(
        profile_id="test",
        label="test",
        jurisdiction="test",
        transaction_cost_bps_assumed=0.0,
        fx_spread_bps_assumed=0.0,
        capital_gains_tax_rate=0.0,
        tax_timing="monthly_inventory_proxy",
        dividend_withholding_mode="not_modeled",
    )


def test_window_slices_use_exact_month_count() -> None:
    idx = pd.bdate_range("2024-01-01", "2024-05-31")
    windows = _window_slices(pd.DatetimeIndex(idx), horizon_months=3)
    assert windows
    assert windows[0]["start_period"] == "2024-01"
    assert windows[0]["end_period"] == "2024-03"
    assert len(windows[0]["contribution_positions"]) == 2


def test_dca_path_without_cost_or_tax_matches_contributions_when_returns_zero() -> None:
    idx = pd.bdate_range("2024-01-01", "2024-03-29")
    gross = pd.Series(0.0, index=idx, dtype=float)
    turnover = pd.Series(0.0, index=idx, dtype=float)
    result = _simulate_dca_path(
        gross_ret=gross,
        turnover=turnover,
        profile=_profile(),
        initial_capital_brl=400.0,
        monthly_contribution_brl=100.0,
        contribution_positions=[idx.get_loc(pd.Timestamp("2024-02-01")), idx.get_loc(pd.Timestamp("2024-03-01"))],
    )
    assert result.total_contributed_brl == 600.0
    assert result.final_value_brl == 600.0
    assert result.profit_brl == 0.0
    assert result.contribution_count == 3
