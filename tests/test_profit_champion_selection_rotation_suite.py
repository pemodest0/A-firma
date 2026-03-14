from __future__ import annotations

import pandas as pd
import pytest

from scripts.bench.validation.run_profit_champion_selection_rotation_suite import (
    _blend_scores,
    _build_offensive_share,
)


def test_offensive_share_is_lagged_and_shrinks_in_stress() -> None:
    idx = pd.date_range("2025-01-01", periods=6, freq="D")
    base_score = pd.Series([0.80, 0.78, 0.76, 0.74, 0.72, 0.70], index=idx)
    criticality = pd.Series([0.35, 0.38, 0.62, 0.74, 0.78, 0.82], index=idx)
    structural_stress = pd.Series([0.32, 0.36, 0.64, 0.72, 0.80, 0.84], index=idx)
    market_mode = pd.Series([0.30, 0.34, 0.68, 0.76, 0.88, 0.92], index=idx)
    share = _build_offensive_share(
        base_score=base_score,
        criticality=criticality,
        structural_stress=structural_stress,
        market_mode_share_pct=market_mode,
    )
    assert float(share.iloc[0]) == 1.0
    assert float(share.iloc[2]) <= 1.0
    assert float(share.iloc[4]) < float(share.iloc[2])
    assert float(share.iloc[-1]) <= 0.38


def test_blend_scores_respects_offensive_share() -> None:
    idx = pd.date_range("2025-01-01", periods=3, freq="D")
    offensive = pd.Series([0.8, 0.6, 0.4], index=idx)
    defensive = pd.Series([0.3, 0.3, 0.3], index=idx)
    share = pd.Series([1.0, 0.5, 0.0], index=idx)
    blended = _blend_scores(
        offensive_score=offensive,
        defensive_score=defensive,
        offensive_share=share,
    )
    assert float(blended.iloc[0]) == pytest.approx(0.8)
    assert float(blended.iloc[1]) == pytest.approx(0.45)
    assert float(blended.iloc[2]) == pytest.approx(0.3)
