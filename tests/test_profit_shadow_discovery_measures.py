from __future__ import annotations

import pandas as pd

from scripts.bench.validation.run_profit_shadow_discovery_measures import (
    _capture_ratio,
    _drawdown_duration_stats,
    _monthly_structure,
    _rolling_outperformance_share,
)


def test_capture_ratio_prefers_stronger_upside_and_smaller_downside() -> None:
    strategy = pd.Series([0.02, 0.01, -0.01, -0.005], dtype=float)
    benchmark = pd.Series([0.01, 0.01, -0.02, -0.01], dtype=float)

    upside = _capture_ratio(strategy, benchmark, positive=True)
    downside = _capture_ratio(strategy, benchmark, positive=False)

    assert upside > 1.0
    assert 0.0 < downside < 1.0


def test_drawdown_duration_stats_counts_consecutive_underwater_days() -> None:
    returns = pd.Series([0.1, -0.05, -0.02, 0.0, 0.1], dtype=float)

    stats = _drawdown_duration_stats(returns)

    assert stats["max_drawdown_duration_days"] == 3.0
    assert stats["ulcer_index"] > 0.0


def test_monthly_structure_summarizes_weights_and_turnover() -> None:
    monthly = pd.DataFrame(
        {
            "ym": ["2024-01", "2024-02"],
            "executed_weights_json": ['{"AAA": 0.6, "BBB": 0.2}', '{"AAA": 0.2, "CCC": 0.5}'],
            "cash_weight": [0.2, 0.3],
        }
    )
    asset_to_sector = {"AAA": "tech", "BBB": "tech", "CCC": "health"}

    out = _monthly_structure(monthly, "case", asset_to_sector)

    assert list(out["profile"]) == ["case", "case"]
    assert out.loc[0, "max_asset_weight"] == 0.6
    assert out.loc[0, "sector_count"] == 1
    assert out.loc[1, "sector_count"] == 2
    assert out.loc[1, "turnover_l1"] > 0.0


def test_rolling_outperformance_share_detects_better_strategy() -> None:
    strategy = pd.Series([0.01] * 10, dtype=float)
    benchmark = pd.Series([0.005] * 10, dtype=float)

    share = _rolling_outperformance_share(strategy, benchmark, window=5)

    assert share == 1.0
