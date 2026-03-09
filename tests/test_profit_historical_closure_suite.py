from __future__ import annotations

import pandas as pd

from scripts.bench.validation.run_profit_historical_closure_suite import _longest_negative_streak


def test_longest_negative_streak_counts_consecutive_losses() -> None:
    alpha = pd.Series([0.1, -0.2, -0.1, 0.0, -0.3, -0.4, -0.5, 0.2], dtype=float)
    assert _longest_negative_streak(alpha) == 3
