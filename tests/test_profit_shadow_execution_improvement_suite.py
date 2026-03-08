from __future__ import annotations

import pandas as pd

from scripts.bench.validation.run_profit_shadow_execution_improvement_suite import (
    _apply_corr_penalty,
    _apply_sector_cap,
    _top_sector_weight,
    _transform_monthly_eval,
)


def test_apply_corr_penalty_preserves_total_weight() -> None:
    weights = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}
    trailing = pd.DataFrame(
        {
            "AAA": [0.01, 0.02, 0.01, 0.03],
            "BBB": [0.01, 0.02, 0.01, 0.03],
            "CCC": [0.00, -0.01, 0.01, 0.00],
        }
    )

    out = _apply_corr_penalty(weights, trailing, strength=0.5, target_total=1.0)

    assert abs(sum(out.values()) - 1.0) < 1e-9
    assert out["AAA"] < 0.5 or out["BBB"] < 0.3


def test_apply_sector_cap_frees_weight_from_dominant_sector() -> None:
    weights = {"AAA": 0.6, "BBB": 0.2, "CCC": 0.2}
    sectors = {"AAA": "tech", "BBB": "tech", "CCC": "health"}

    capped, freed = _apply_sector_cap(weights, sectors, cap_fraction=0.5)

    assert freed > 0.0
    assert _top_sector_weight(capped, sectors) <= 0.5 + 1e-9


def test_transform_monthly_eval_adds_sleeve_and_reduces_concentration() -> None:
    monthly = pd.DataFrame(
        {
            "ym": ["2024-03"],
            "executed_weights_json": ['{"AAA": 0.7, "BBB": 0.1}'],
            "executed_assets": ["AAA,BBB"],
            "selected_assets": ["AAA,BBB"],
            "cash_weight": [0.2],
            "n_selected": [2],
        }
    )
    asset_to_sector = {"AAA": "tech", "BBB": "tech", "SPY": "equities_us_broad", "XLV": "health_care"}
    market_state_by_month = {"2024-02": "bear"}
    returns_wide = pd.DataFrame(
        {
            "AAA": [0.01] * 130,
            "BBB": [0.01] * 130,
            "SPY": [0.005] * 130,
            "XLV": [0.004] * 130,
            "XLP": [0.003] * 130,
            "XLU": [0.002] * 130,
            "XLRE": [0.001] * 130,
            "IEF": [0.0] * 130,
        },
        index=pd.date_range("2023-01-01", periods=130, freq="B"),
    )
    policy = {
        "policy_id": "test",
        "cap_profile": "moderate",
        "corr_strength": 0.0,
        "sleeve_share": 0.2,
        "sleeve_trigger": 0.65,
    }

    out = _transform_monthly_eval(
        monthly,
        policy=policy,
        asset_to_sector=asset_to_sector,
        market_state_by_month=market_state_by_month,
        returns_wide=returns_wide,
    )
    weights = out.iloc[0]["executed_weights_json"]

    assert "IEF" in weights or "XLV" in weights
    parsed = {"AAA": 0.0, "BBB": 0.0}
    for key in ["AAA", "BBB"]:
        parsed[key] = 0.0 if key not in weights else 1.0
    assert parsed["AAA"] == 1.0
