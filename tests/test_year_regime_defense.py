import pandas as pd

from engine.portfolio.year_regime_defense import (
    YearDefenseConfig,
    compute_ytd_stress,
    year_bad_state_trigger,
)


def test_compute_ytd_stress_rises_with_bad_path():
    calm = compute_ytd_stress(
        pd.Series([0.03]),
        pd.Series([-0.02]),
        {"structural_stress": 0.1, "liquidation": 0.1, "breadth": 0.8},
    )
    bad = compute_ytd_stress(
        pd.Series([-0.12]),
        pd.Series([-0.20]),
        {"structural_stress": 0.8, "liquidation": 0.8, "breadth": 0.2},
    )
    assert bad > calm


def test_year_bad_state_trigger_requires_both_damage_and_persistence():
    cfg = YearDefenseConfig()
    assert not year_bad_state_trigger(
        {"ytd_return": -0.15, "ytd_drawdown": -0.2, "year_stress": 0.8, "bad_days": 10},
        cfg,
    )
    assert year_bad_state_trigger(
        {"ytd_return": -0.15, "ytd_drawdown": -0.2, "year_stress": 0.8, "bad_days": 30},
        cfg,
    )
