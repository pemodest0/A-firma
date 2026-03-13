from __future__ import annotations

import pandas as pd

from scripts.bench.validation.run_profit_champion_extension_suite import (
    _fragility_decile_scale,
    _profit_lock_scale,
)


def test_fragility_decile_scale_only_reduces_in_extremes() -> None:
    idx = pd.date_range("2024-01-01", periods=8, freq="D")
    liquidation = pd.Series([0.1, 0.2, 0.2, 0.3, 0.25, 0.4, 0.9, 1.0], index=idx, dtype=float)
    scale = _fragility_decile_scale(liquidation, window=8)
    assert float(scale.iloc[-1]) <= 0.80
    assert float(scale.iloc[0]) == 1.0


def test_profit_lock_scale_uses_only_lagged_returns() -> None:
    idx = pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01"])
    net_ret = pd.Series([0.00, 0.35, 0.00], index=idx, dtype=float)
    scale = _profit_lock_scale(net_ret)
    assert float(scale.loc[pd.Timestamp("2024-01-31")]) == 1.0
    assert float(scale.loc[pd.Timestamp("2024-02-01")]) < 1.0
