from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.structural.run_epistemic_diagnostics import _discover_universes, _median_lead_time_drawdown


def test_median_lead_time_drawdown_computes_first_breach() -> None:
    equity = pd.Series([1.00, 0.99, 0.97, 0.96, 0.95])
    pred = pd.Series([1, 0, 1, 0, 0], dtype="Int64")
    valid = pd.Series([True, True, True, True, True])
    lead = _median_lead_time_drawdown(
        equity=equity,
        pred=pred,
        valid_mask=valid,
        horizon=3,
        dd_threshold=0.03,
    )
    assert abs(float(lead) - 2.0) < 1e-9


def test_discover_universes_all_reads_hierarchical_layout(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    hier = run_dir / "hierarchical"
    uni = hier / "universes"
    uni.mkdir(parents=True, exist_ok=True)

    (hier / "diagnostics_global_score_daily.csv").write_text("date,score,flags_valid\n2026-01-01,0.1,1\n", encoding="utf-8")
    (hier / "diagnostics_sector_gics_tech_score_daily.csv").write_text("date,score,flags_valid\n2026-01-01,0.2,1\n", encoding="utf-8")
    (hier / "diagnostics_sector_internal_alpha_score_daily.csv").write_text("date,score,flags_valid\n2026-01-01,0.3,1\n", encoding="utf-8")
    pd.DataFrame(
        [
            {"kind": "gics", "sector": "Technology", "slug": "tech", "n_assets": 2},
            {"kind": "internal", "sector": "Alpha", "slug": "alpha", "n_assets": 2},
        ]
    ).to_csv(uni / "sector_universe_index.csv", index=False)

    xs = _discover_universes(run_dir=run_dir, selector="all")
    names = sorted([str(x["name"]) for x in xs])
    assert names == ["gics:Technology", "global", "internal:Alpha"]
