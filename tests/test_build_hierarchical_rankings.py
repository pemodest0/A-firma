from __future__ import annotations

import pandas as pd

from scripts.build_hierarchical_rankings import _build_payloads


def test_build_payloads_contains_required_sections() -> None:
    asset_global_daily = pd.DataFrame(
        {
            "date": ["2026-01-02", "2026-01-02", "2026-01-02"],
            "asset_id": ["A", "B", "C"],
            "ticker": ["AAA", "BBB", "CCC"],
            "impact_global": [0.5, 0.3, 0.2],
            "sector_gics": ["tech", "financials", "tech"],
            "sector_internal": ["alpha", "beta", "alpha"],
        }
    )
    overlap_daily = pd.DataFrame(
        {
            "date": ["2026-01-02", "2026-01-02"],
            "sector_kind": ["gics", "internal"],
            "sector": ["tech", "alpha"],
            "overlap_sector_global": [0.91, 0.88],
        }
    )
    global_state_daily = pd.DataFrame(
        {
            "date": ["2026-01-02"],
            "score": [1.25],
            "phi": [0.33],
            "deff": [9.7],
            "Q": [1.2],
            "N_used": [100],
        }
    )
    out = _build_payloads(
        asset_global_daily=asset_global_daily,
        overlap_daily=overlap_daily,
        global_state_daily=global_state_daily,
        top_assets=2,
        top_sectors=3,
    )
    assert len(out) == 1
    row = out[0]
    assert row["date"] == "2026-01-02"
    assert row["top_assets_global_mode"][0]["asset_id"] == "A"
    assert "top_sectors_global_mode" in row
    assert "sector_global_overlap" in row
    assert abs(float(row["global_state"]["q"]) - 1.2) < 1e-9
