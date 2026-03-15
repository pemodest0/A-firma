from __future__ import annotations

import pandas as pd

from execution.net_assumptions import load_net_assumption_profiles
from scripts.bench.validation.run_profit_crypto_capital_scenarios import (
    Scenario,
    _scenario_row,
    _simulate_fixed_basket,
)


def test_simulate_fixed_basket_has_entry_and_exit_turnover() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    returns = pd.DataFrame(
        {
            "BTC-USD": [0.01, 0.0, 0.0, 0.0],
            "ETH-USD": [0.0, 0.01, 0.0, 0.0],
        },
        index=idx,
    )
    scenario = Scenario("pair", "fixed_basket", ("BTC-USD", "ETH-USD"), (0.5, 0.5), 0, "")
    gross, turnover = _simulate_fixed_basket(scenario=scenario, returns=returns)
    assert gross.shape[0] == 4
    assert float(turnover.iloc[0]) == 1.0
    assert float(turnover.iloc[-1]) == 1.0


def test_scenario_row_builds_capital_outputs() -> None:
    idx = pd.date_range("2024-01-01", periods=260, freq="B")
    gross = pd.Series(0.001, index=idx, dtype=float)
    turnover = pd.Series(0.0, index=idx, dtype=float)
    benchmark = pd.Series(0.0, index=idx, dtype=float)
    profile = load_net_assumption_profiles("config/profit_net_assumptions.json")["profiles"]["crypto_global_brazil_resident_conservative"]
    scenario = Scenario("btc_hold", "single_asset", ("BTC-USD",), (1.0,), 0, "")
    row = _scenario_row(
        scenario=scenario,
        capital_brl=200.0,
        gross=gross,
        turnover=turnover,
        benchmark_ret=benchmark,
        profile=profile,
    )
    assert row["capital_brl"] == 200.0
    assert row["median_end_value_252d_brl"] > 200.0
    assert row["transaction_cost_brl"] == 0.0
