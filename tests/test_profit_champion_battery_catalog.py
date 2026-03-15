from scripts.bench.validation.run_profit_champion_battery_catalog import (
    _candidate_name,
    _metric_sort_tuple,
)


def test_candidate_name_prefers_candidate_id() -> None:
    row = {
        "candidate_id": "official_mode",
        "scenario_id": "fallback_scenario",
    }
    assert _candidate_name(row) == "official_mode"


def test_metric_sort_tuple_uses_negative_inf_for_missing_values() -> None:
    row = {"net_total_return": "12.5"}
    tup = _metric_sort_tuple(row, ("net_total_return", "net_sharpe"))
    assert tup[0] == 12.5
    assert tup[1] == float("-inf")
