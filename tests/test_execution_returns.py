from __future__ import annotations

import math

import pandas as pd

from execution.returns import compound_simple_returns, daily_simple_to_monthly, load_return_frame_csv


def test_load_return_frame_csv_converts_log_returns_to_simple(tmp_path) -> None:
    path = tmp_path / "SPY.csv"
    pd.DataFrame(
        {
            "date": ["2026-01-02", "2026-01-05"],
            "r": [0.01, -0.02],
        }
    ).to_csv(path, index=False)

    out = load_return_frame_csv(path, source_kind="log", target_kind="simple")

    assert out["r"].round(8).tolist() == [round(math.expm1(0.01), 8), round(math.expm1(-0.02), 8)]


def test_daily_simple_to_monthly_compounds_simple_returns() -> None:
    dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-02-03"])
    simple = pd.Series([math.expm1(0.01), math.expm1(-0.02), math.expm1(0.03)], index=dates, dtype=float)

    monthly = daily_simple_to_monthly(simple)

    assert monthly.index.tolist() == ["2026-01", "2026-02"]
    assert monthly.round(8).tolist() == [round(math.expm1(-0.01), 8), round(math.expm1(0.03), 8)]


def test_compound_simple_returns_matches_log_identity() -> None:
    simple = pd.Series([math.expm1(0.01), math.expm1(-0.02), math.expm1(0.03)], dtype=float)

    total = compound_simple_returns(simple)

    assert round(total, 8) == round(math.expm1(0.02), 8)
