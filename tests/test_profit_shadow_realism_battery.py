from __future__ import annotations

import pandas as pd

from scripts.bench.validation.run_profit_shadow_realism_battery import (
    apply_concentration_caps,
    build_daily_replay_with_delay,
    classify_market_slices,
)


def test_build_daily_replay_with_delay_uses_previous_month_for_initial_days() -> None:
    dates = pd.to_datetime(["2026-01-30", "2026-01-31", "2026-02-03", "2026-02-04"])
    returns = pd.DataFrame({"AAA": [0.01, 0.01, 0.10, 0.00], "BBB": [0.00, 0.00, 0.00, 0.02]}, index=dates)
    bench = pd.Series([0.0, 0.0, 0.0, 0.0], index=dates, dtype=float)
    monthly = pd.DataFrame(
        [
            {"ym": "2026-01", "risk_bucket": "stable", "executed_assets": "AAA", "executed_weights_json": "{\"AAA\": 1.0}", "cash_weight": 0.0, "hedge_weight": 0.0},
            {"ym": "2026-02", "risk_bucket": "dispersion", "executed_assets": "BBB", "executed_weights_json": "{\"BBB\": 1.0}", "cash_weight": 0.0, "hedge_weight": 0.0},
        ]
    )

    out = build_daily_replay_with_delay(
        monthly_eval=monthly,
        returns_wide=returns,
        benchmark_returns=bench,
        initial_capital=1000.0,
        execution_delay_days=1,
    )

    feb = out[out["ym"] == "2026-02"].reset_index(drop=True)
    assert round(float(feb.iloc[0]["portfolio_return"]), 6) == 0.10
    assert round(float(feb.iloc[1]["portfolio_return"]), 6) == 0.02


def test_apply_concentration_caps_enforces_asset_and_gross_caps() -> None:
    monthly = pd.DataFrame(
        [
            {
                "ym": "2026-02",
                "executed_weights_json": "{\"AAA\": 0.50, \"BBB\": 0.40, \"CCC\": 0.30}",
                "executed_assets": "AAA,BBB,CCC",
                "selected_assets": "AAA,BBB,CCC",
                "cash_weight": 0.0,
                "hedge_weight": 0.0,
            }
        ]
    )
    out = apply_concentration_caps(
        monthly,
        asset_to_sector={"AAA": "tech", "BBB": "tech", "CCC": "industrial"},
        asset_cap=0.25,
        sector_cap=0.30,
        gross_cap=0.60,
    )

    weights = out.iloc[0]["executed_weights_json"]
    assert "\"AAA\": 0.15" in weights or "\"BBB\": 0.15" in weights
    assert float(out.iloc[0]["cash_weight"]) >= 0.40


def test_classify_market_slices_marks_bear_and_recovery() -> None:
    idx = pd.date_range("2026-01-01", periods=120, freq="B")
    ret = pd.Series([0.001] * 40 + [-0.01] * 30 + [0.008] * 50, index=idx, dtype=float)

    labels = classify_market_slices(ret)

    assert "bear" in set(labels.values)
    assert "recovery" in set(labels.values) or "bull" in set(labels.values)
