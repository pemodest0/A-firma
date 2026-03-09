from __future__ import annotations

import pandas as pd

from scripts.bench.validation.run_profit_regime_error_suite import (
    _delay_regime,
    _flip_regime,
    _selector_choice,
    _selector_with_inertia,
)


def test_delay_regime_preserves_shape_and_front_fill():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    series = pd.Series(["stable", "transition", "stress", "dispersion"], index=idx)
    delayed = _delay_regime(series, 2)
    assert list(delayed.index) == list(idx)
    assert delayed.iloc[0] == "stable"
    assert delayed.iloc[1] == "stable"
    assert delayed.iloc[2] == "stable"


def test_flip_regime_changes_only_known_states():
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    series = pd.Series(["stable", "transition", "stress", "dispersion", "stable", "stress"], index=idx)
    flipped = _flip_regime(series, 1.0, 17)
    assert set(flipped.unique()) <= {"stable", "transition", "stress", "dispersion"}
    assert all(value != original for value, original in zip(flipped.tolist(), series.tolist()))


def test_selector_choice_prefers_attack_in_risk_on():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    regime = pd.Series(["stable", "dispersion", "transition", "stress"], index=idx)
    selected = _selector_choice(regime)
    assert selected.tolist() == ["attack", "attack", "protect", "protect"]


def test_selector_with_inertia_holds_short_flip():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    base = pd.Series(["attack", "protect", "protect", "protect", "protect"], index=idx)
    selected = _selector_with_inertia(base, min_hold_days=3, confirm_days=1)
    assert selected.tolist()[:3] == ["attack", "attack", "attack"]


def test_selector_with_inertia_requires_confirmation():
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    base = pd.Series(["attack", "protect", "attack", "protect", "protect", "protect"], index=idx)
    selected = _selector_with_inertia(base, min_hold_days=0, confirm_days=2)
    assert selected.iloc[1] == "attack"
    assert selected.iloc[-1] == "protect"
