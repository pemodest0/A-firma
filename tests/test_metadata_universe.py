from __future__ import annotations

from pathlib import Path

import pandas as pd

from engine.core.universe import select_global_universe, select_sector_universe
from engine.ops.metadata import load_asset_metadata


def test_load_asset_metadata_requires_unique_asset_id(tmp_path: Path) -> None:
    p = tmp_path / "asset_metadata.csv"
    df = pd.DataFrame(
        {
            "asset_id": ["A", "A"],
            "ticker": ["A", "A2"],
            "sector_gics": ["tech", "tech"],
            "sector_internal": ["tech_int", "tech_int"],
        }
    )
    df.to_csv(p, index=False)
    try:
        load_asset_metadata(p)
        assert False, "expected duplicate validation error"
    except ValueError as exc:
        assert "unique" in str(exc)


def test_universe_selection_is_deterministic() -> None:
    idx = pd.date_range("2026-01-01", periods=6, freq="D")
    returns = pd.DataFrame(
        {
            "A": [0.1, 0.2, 0.1, 0.1, 0.0, 0.1],
            "B": [0.1, 0.2, None, 0.1, 0.0, 0.1],
            "C": [0.1, None, None, 0.1, 0.0, 0.1],
        },
        index=idx,
    )
    metadata = pd.DataFrame(
        {
            "asset_id": ["A", "B", "C"],
            "ticker": ["A", "B", "C"],
            "sector_gics": ["tech", "tech", "energy"],
            "sector_internal": ["alpha", "alpha", "beta"],
            "liquidity_proxy": [30, 20, 10],
        }
    )
    g = select_global_universe(returns, metadata, n_global=2, min_coverage=0.5)
    assert g == ["A", "B"]

    s = select_sector_universe(returns, metadata, sector_name="tech", n_sector=2, min_coverage=0.5)
    assert s == ["A", "B"]


def test_sector_selection_respects_sector_filter() -> None:
    idx = pd.date_range("2026-01-01", periods=6, freq="D")
    returns = pd.DataFrame(
        {
            "AAA": [0.1, 0.2, 0.1, 0.1, 0.0, 0.1],  # tech, lower liquidity
            "BBB": [0.1, 0.2, 0.1, 0.1, 0.0, 0.1],  # tech, medium liquidity
            "ZZZ": [0.1, 0.2, 0.1, 0.1, 0.0, 0.1],  # energy, highest liquidity
        },
        index=idx,
    )
    metadata = pd.DataFrame(
        {
            "asset_id": ["AAA", "BBB", "ZZZ"],
            "ticker": ["AAA", "BBB", "ZZZ"],
            "sector_gics": ["tech", "tech", "energy"],
            "sector_internal": ["alpha", "alpha", "beta"],
            "liquidity_proxy": [10, 20, 999],
        }
    )
    s = select_sector_universe(returns, metadata, sector_name="tech", n_sector=2, min_coverage=0.9)
    assert s == ["BBB", "AAA"]
