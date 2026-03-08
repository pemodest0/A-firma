from __future__ import annotations

import pandas as pd

from scripts.bench.validation.run_profit_10x_rule_search import _rolling_ten_x_stats, _top_k_indices


def test_top_k_indices_respects_valid_mask_and_ranks_descending() -> None:
    score = pd.Series([0.1, 0.8, 0.4, 0.6], dtype=float).to_numpy()
    valid = pd.Series([False, True, True, False], dtype=bool).to_numpy()

    out = _top_k_indices(score, valid, top_k=2)

    assert out == [1, 2]


def test_rolling_ten_x_stats_detects_fast_hit() -> None:
    idx = pd.date_range("2024-01-01", periods=200, freq="B")
    ret = pd.Series([0.03] * len(idx), index=idx, dtype=float)

    out = _rolling_ten_x_stats(ret, horizon_days=252 * 2, target_multiple=10.0)

    assert out["starts_considered"] > 0
    assert out["hit_rate"] > 0.0
    assert out["best_years"] < 1.0
