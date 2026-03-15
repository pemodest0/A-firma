from __future__ import annotations

import pandas as pd

from scripts.bench.validation.run_profit_one_year_walkforward_audit import (
    _subset_frame_for_start_year,
    _train_score,
)


def test_subset_frame_for_start_year_filters_correctly() -> None:
    frame = pd.DataFrame(
        {
            "start_date": ["2023-01-02", "2024-01-03", "2024-12-20", "2025-01-05"],
            "terminal_multiple": [1.1, 2.0, 0.7, 3.0],
            "max_multiple": [1.2, 2.5, 0.9, 3.5],
            "min_multiple": [0.9, 0.8, 0.6, 0.7],
        }
    )
    out = _subset_frame_for_start_year(frame, 2024)
    assert out["start_date"].tolist() == ["2024-01-03", "2024-12-20"]


def test_train_score_prioritizes_6x_then_median_then_loss() -> None:
    a = {"hit_rate_6x_252d": 0.10, "median_return_252d": 0.80, "end_below_50_252d": 0.05, "touch_loss_50_252d": 0.10}
    b = {"hit_rate_6x_252d": 0.12, "median_return_252d": 0.40, "end_below_50_252d": 0.20, "touch_loss_50_252d": 0.30}
    c = {"hit_rate_6x_252d": 0.12, "median_return_252d": 0.60, "end_below_50_252d": 0.30, "touch_loss_50_252d": 0.40}
    assert _train_score(b) > _train_score(a)
    assert _train_score(c) > _train_score(b)
