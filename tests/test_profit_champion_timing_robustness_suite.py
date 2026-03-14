from __future__ import annotations

import pandas as pd
import pytest

from scripts.bench.validation.run_profit_champion_timing_robustness_suite import (
    _build_structural_openness,
    _selective_score,
    _underperform_prob_rolling,
)


def test_structural_openness_is_lagged_and_bounded() -> None:
    idx = pd.date_range("2025-01-01", periods=5, freq="D")
    base_score = pd.Series([0.8, 0.75, 0.72, 0.70, 0.68], index=idx)
    criticality = pd.Series([0.35, 0.40, 0.62, 0.75, 0.80], index=idx)
    stress = pd.Series([0.30, 0.35, 0.60, 0.73, 0.82], index=idx)
    market = pd.Series([0.25, 0.30, 0.68, 0.80, 0.90], index=idx)
    liquidation = pd.Series([0.20, 0.22, 0.30, 0.44, 0.60], index=idx)
    openness = _build_structural_openness(
        base_score=base_score,
        criticality=criticality,
        structural_stress=stress,
        market_mode_share_pct=market,
        liquidation=liquidation,
    )
    assert float(openness.iloc[0]) == pytest.approx(0.5)
    assert float(openness.min()) >= 0.0
    assert float(openness.max()) <= 1.0
    assert float(openness.iloc[-1]) < float(openness.iloc[1])


def test_selective_score_and_underperform_probability() -> None:
    idx = pd.date_range("2025-01-01", periods=70, freq="D")
    base_score = pd.Series([0.8] * 35 + [0.6] * 35, index=idx)
    openness = pd.Series([0.8] * 35 + [0.3] * 35, index=idx)
    selective = _selective_score(base_score, openness)
    assert float(selective.iloc[0]) > float(selective.iloc[-1])

    candidate = pd.Series([0.01] * 63 + [-0.01] * 7, index=idx)
    benchmark = pd.Series([0.0] * 70, index=idx)
    prob = _underperform_prob_rolling(candidate, benchmark, horizon=63)
    assert 0.0 <= float(prob) <= 1.0
