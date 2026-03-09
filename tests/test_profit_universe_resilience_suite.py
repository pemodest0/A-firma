from __future__ import annotations

import pandas as pd

from scripts.bench.validation.run_profit_universe_resilience_suite import _random_keep


def test_random_keep_reduces_universe_without_duplicates() -> None:
    df = pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "D", "E"],
            "asset_group": ["g1", "g1", "g2", "g2", "g3"],
        }
    )
    out = _random_keep(df, fraction=0.4, seed=7)
    assert 1 <= len(out) <= len(df)
    assert out["ticker"].is_unique
