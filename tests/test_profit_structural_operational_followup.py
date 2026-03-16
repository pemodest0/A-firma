from __future__ import annotations

import pandas as pd

from scripts.bench.validation.run_profit_structural_operational_followup import _build_subset_masks, _subset_stats


def test_build_subset_masks_creates_holdout_and_attack_views() -> None:
    idx = pd.to_datetime(["2022-12-31", "2023-01-01", "2024-01-01", "2025-01-01"])
    attack = pd.Series([0, 1, 0, 1], index=idx, dtype=int)
    masks = _build_subset_masks(idx, attack)
    assert set(masks) == {
        "all",
        "attack_only",
        "holdout_2023",
        "holdout_2024",
        "holdout_2025",
        "attack_2023",
        "attack_2024",
        "attack_2025",
    }
    assert int(masks["attack_only"].sum()) == 2
    assert int(masks["holdout_2023"].sum()) == 1
    assert int(masks["attack_2024"].sum()) == 0


def test_subset_stats_reports_coverage() -> None:
    idx = pd.date_range("2025-01-01", periods=4, freq="D")
    mask = pd.Series([1, 0, 1, 0], index=idx, dtype=int)
    stats = _subset_stats(mask)
    assert stats["active_days"] == 2
    assert abs(float(stats["coverage"]) - 0.5) < 1e-9
