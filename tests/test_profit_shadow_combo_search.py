from __future__ import annotations

import pandas as pd

from scripts.bench.validation.run_profit_shadow_combo_search import (
    _alpha_from_rule,
    _candidate_grid,
    _cfg_id,
    _parse_csv_tokens,
)


def test_alpha_from_rule_dynamic_moves_toward_challenger_when_spread_positive() -> None:
    main_ret = pd.Series([0.01, 0.01, 0.01, 0.01], dtype=float)
    challenger_ret = pd.Series([0.05, 0.05, 0.05, 0.05], dtype=float)

    alpha = _alpha_from_rule(
        rule="dynamic",
        pos=4,
        ym="2026-05",
        market_state_by_month={"2026-04": "bull"},
        main_ret=main_ret,
        challenger_ret=challenger_ret,
        lookback_months=3,
        dynamic_base_alpha_main=0.5,
        dynamic_strength=1.0,
    )

    assert 0.0 <= alpha < 0.5


def test_cfg_id_is_stable_and_includes_rules() -> None:
    cfg = {
        "lookback_months": 3,
        "bull_rule": "challenger",
        "recovery_rule": "challenger",
        "bear_rule": "main",
        "sideways_rule": "dynamic",
        "dynamic_base_alpha_main": 0.5,
        "dynamic_strength": 1.0,
    }

    cid = _cfg_id(cfg)

    assert cid.startswith("lb03_")
    assert "_s-dynamic_" in cid


def test_parse_csv_tokens_and_grid_filters_work() -> None:
    lookbacks = _parse_csv_tokens("2,4", [1, 2, 3], cast=int)
    bull_rules = _parse_csv_tokens("challenger,dynamic", ["main"], cast=str)
    dynamic_bases = _parse_csv_tokens("0.25,0.75", [0.5], cast=float)
    dynamic_strengths = _parse_csv_tokens("1.5", [1.0], cast=float)

    grid = _candidate_grid(
        lookbacks=lookbacks,
        bull_rules=bull_rules,
        recovery_rules=["challenger"],
        bear_rules=["main"],
        sideways_rules=["trailing_best"],
        dynamic_bases=dynamic_bases,
        dynamic_strengths=dynamic_strengths,
    )

    assert {cfg["lookback_months"] for cfg in grid} == {2, 4}
    assert {cfg["bull_rule"] for cfg in grid} == {"challenger", "dynamic"}
    assert {cfg["dynamic_base_alpha_main"] for cfg in grid if cfg["bull_rule"] == "dynamic"} == {0.25, 0.75}
    assert {cfg["dynamic_strength"] for cfg in grid if cfg["bull_rule"] == "dynamic"} == {1.5}
