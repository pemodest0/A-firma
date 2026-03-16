from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.bench.validation.run_profit_crypto_universe_promotion_suite import (
    _classify_regime_series,
    _timing_diagnostics,
)


def test_classify_regime_series_respects_thresholds() -> None:
    frame = pd.DataFrame(
        {
            "criticality": [0.3, 0.6, 0.75],
            "structural_stress": [0.3, 0.55, 0.8],
            "market_mode_share_pct": [0.3, 0.66, 0.83],
        },
        index=pd.date_range("2026-01-01", periods=3, freq="D"),
    )
    out = _classify_regime_series(frame)
    assert list(out.astype(str)) == ["dispersion", "transition", "stress"]


def test_timing_diagnostics_returns_separation_metrics() -> None:
    idx = pd.date_range("2026-01-01", periods=40, freq="D")
    net = pd.Series([0.01] * 20 + [-0.01] * 20, index=idx, dtype=float)
    bench = pd.Series([0.0] * 40, index=idx, dtype=float)
    structure = pd.DataFrame(
        {
            "criticality": [0.2] * 20 + [0.8] * 20,
            "structural_stress": [0.2] * 20 + [0.8] * 20,
            "market_mode_share_pct": [0.2] * 20 + [0.8] * 20,
        },
        index=idx,
    )
    summary, regime_df = _timing_diagnostics(
        net_ret=net,
        benchmark_net_ret=bench,
        structure_daily=structure,
        horizon=5,
    )
    assert "regime_separation_21d" in summary
    assert not regime_df.empty


def test_expanded_crypto_metadata_has_more_assets_than_plus() -> None:
    root = Path(__file__).resolve().parents[1]
    plus_df = pd.read_csv(root / "data" / "asset_metadata_crypto_top_liquid_plus.csv")
    expanded_df = pd.read_csv(root / "data" / "asset_metadata_crypto_top_liquid_expanded.csv")
    assert expanded_df["ticker"].nunique() > plus_df["ticker"].nunique()
