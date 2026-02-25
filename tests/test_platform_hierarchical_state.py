from __future__ import annotations

import pandas as pd

from scripts.ops.run_platform_hierarchical_state import _build_daily_payloads


def test_build_daily_payloads_ranks_sectors() -> None:
    global_scores = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02"],
            "score": [0.2, 0.4],
        }
    )
    sector_scores = pd.DataFrame(
        {
            "date": ["2026-01-02", "2026-01-02", "2026-01-02"],
            "sector": ["tech", "energy", "banks"],
            "kind": ["gics", "gics", "gics"],
            "score": [0.9, 0.8, 0.3],
        }
    )
    cross_scores = pd.DataFrame(
        {
            "date": ["2026-01-02", "2026-01-02", "2026-01-02"],
            "sector": ["tech", "energy", "banks"],
            "kind": ["gics", "gics", "gics"],
            "loading_sector_on_global": [0.7, 0.6, 0.2],
            "overlap_sector_global": [0.9, 0.8, 0.5],
        }
    )
    payloads, latest = _build_daily_payloads(
        global_scores=global_scores,
        sector_scores=sector_scores,
        cross_scores=cross_scores,
        top_k=2,
    )
    assert len(payloads) == 2
    assert latest["date"] == "2026-01-02"
    assert latest["top_sectors_by_score"][0]["sector"] == "tech"
    assert latest["top_sectors_by_loading"][0]["sector"] == "tech"
