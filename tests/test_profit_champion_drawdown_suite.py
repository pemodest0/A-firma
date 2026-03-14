from __future__ import annotations

import pandas as pd

from scripts.bench.validation.run_profit_champion_drawdown_suite import (
    _build_early_stress_scale,
    _build_gradual_posture_cap,
    _build_regime_cap,
)


def test_regime_cap_shrinks_when_stress_rises() -> None:
    idx = pd.date_range("2025-01-01", periods=4, freq="D")
    criticality = pd.Series([0.30, 0.58, 0.66, 0.80], index=idx)
    structural_stress = pd.Series([0.32, 0.60, 0.70, 0.82], index=idx)
    market_mode = pd.Series([0.25, 0.55, 0.72, 0.88], index=idx)
    cap = _build_regime_cap(
        criticality=criticality,
        structural_stress=structural_stress,
        market_mode_share_pct=market_mode,
    )
    assert float(cap.iloc[0]) == 1.0
    assert float(cap.iloc[1]) <= 0.55
    assert float(cap.iloc[2]) <= 0.32
    assert float(cap.iloc[3]) <= 0.14


def test_early_stress_scale_stays_bounded() -> None:
    idx = pd.date_range("2025-01-01", periods=160, freq="D")
    criticality = pd.Series([0.35] * 80 + [0.78] * 80, index=idx)
    structural_stress = pd.Series([0.40] * 80 + [0.82] * 80, index=idx)
    market_mode = pd.Series([0.30] * 80 + [0.86] * 80, index=idx)
    scale = _build_early_stress_scale(
        criticality=criticality,
        structural_stress=structural_stress,
        market_mode_share_pct=market_mode,
    )
    assert float(scale.min()) >= 0.18
    assert float(scale.max()) <= 1.0
    assert float(scale.iloc[-1]) < float(scale.iloc[40])


def test_gradual_posture_cap_reacts_to_higher_stress() -> None:
    idx = pd.date_range("2025-01-01", periods=4, freq="D")
    base_score = pd.Series([0.82, 0.80, 0.78, 0.76], index=idx)
    criticality = pd.Series([0.30, 0.50, 0.62, 0.74], index=idx)
    structural_stress = pd.Series([0.20, 0.45, 0.63, 0.78], index=idx)
    cap = _build_gradual_posture_cap(
        base_score=base_score,
        criticality=criticality,
        structural_stress=structural_stress,
    )
    assert float(cap.iloc[0]) >= 0.78
    assert float(cap.iloc[2]) <= 0.52
    assert float(cap.iloc[3]) <= 0.25
