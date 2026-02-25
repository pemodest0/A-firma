from __future__ import annotations

import pandas as pd

from engine.structural.impact import (
    compute_asset_global_impact,
    compute_asset_sector_impact,
    compute_sector_pair_overlap,
    merge_asset_sector_global_impacts,
)
from scripts.structural.run_structural_impact_learning import (
    _build_label_direction_sanity,
    _build_walkforward_compare_rows,
    _month_starts_between,
)


def test_asset_global_impact_normalizes_per_date() -> None:
    g = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"],
            "asset_id": ["A", "B", "A", "B"],
            "weight": [0.6, 0.8, 0.3, 0.4],
        }
    )
    out = compute_asset_global_impact(g)
    sums = out.groupby("date")["impact_global"].sum().to_dict()
    assert abs(float(sums["2026-01-01"]) - 1.0) < 1e-9
    assert abs(float(sums["2026-01-02"]) - 1.0) < 1e-9


def test_merge_asset_sector_global_impacts_adds_cross_metrics() -> None:
    g = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-01"],
            "asset_id": ["A", "B"],
            "weight": [0.7, 0.3],
        }
    )
    s_map = {
        ("gics", "Tech"): pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-01"],
                "asset_id": ["A", "B"],
                "weight": [0.8, 0.6],
            }
        )
    }
    cross = pd.DataFrame(
        {
            "date": ["2026-01-01"],
            "sector": ["Tech"],
            "sector_kind": ["gics"],
            "loading_sector_on_global": [0.75],
            "overlap_sector_global": [0.85],
        }
    )
    out = merge_asset_sector_global_impacts(
        asset_global=compute_asset_global_impact(g),
        asset_sector=compute_asset_sector_impact(s_map),
        cross_daily=cross,
        metadata=pd.DataFrame({"asset_id": ["A", "B"], "ticker": ["AAA", "BBB"]}),
    )
    assert sorted(out["ticker"].unique().tolist()) == ["AAA", "BBB"]
    assert abs(float(out["sector_loading"].dropna().iloc[0]) - 0.75) < 1e-9
    assert abs(float(out["overlap_sector_global"].dropna().iloc[0]) - 0.85) < 1e-9


def test_sector_pair_overlap_identical_vectors_is_one() -> None:
    s_map = {
        ("gics", "A"): pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-01"],
                "asset_id": ["X", "Y"],
                "weight": [0.6, 0.8],
            }
        ),
        ("gics", "B"): pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-01"],
                "asset_id": ["X", "Y"],
                "weight": [0.6, 0.8],
            }
        ),
    }
    out = compute_sector_pair_overlap(s_map)
    assert int(out.shape[0]) == 1
    assert abs(float(out.iloc[0]["overlap_ab"]) - 1.0) < 1e-9


def test_month_starts_between_bounds() -> None:
    out = _month_starts_between(pd.Timestamp("2024-01-15"), pd.Timestamp("2024-04-10"))
    vals = [str(x.date()) for x in out]
    assert vals == ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]


def test_build_walkforward_compare_rows_has_expected_columns() -> None:
    wf = pd.DataFrame(
        {
            "month": ["2024-01", "2024-01"],
            "mode": ["fixed", "expanding"],
            "target": ["drawdown_label", "drawdown_label"],
            "model": ["linear", "linear"],
            "event_rate": [0.2, 0.25],
            "test_rows": [100, 120],
            "alert_rate": [0.15, 0.16],
            "precision": [0.4, 0.35],
            "recall": [0.2, 0.22],
            "f1": [0.2666, 0.2734],
            "lift_precision_vs_random": [1.5, 1.3],
        }
    )
    out = _build_walkforward_compare_rows(wf_df=wf, horizon_days=10)
    assert out.columns.tolist() == [
        "month",
        "cv_mode",
        "horizon",
        "label",
        "model",
        "alert_rate",
        "precision",
        "recall",
        "f1",
        "lift",
        "n_events",
    ]
    assert out["horizon"].unique().tolist() == [10]
    assert sorted(out["n_events"].tolist()) == [20.0, 30.0]


def test_label_direction_sanity_reports_ratio() -> None:
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    loading = [0.01 * (i + 1) for i in range(20)]
    overlap = [0.02 * (i + 1) for i in range(20)]
    score = [0.03 * (i + 1) for i in range(20)]
    loading[14] = 0.95
    loading[18] = 0.96
    loading[19] = 0.98
    overlap[14] = 0.91
    overlap[18] = 0.92
    overlap[19] = 0.93
    score[14] = 0.85
    score[18] = 0.86
    score[19] = 0.89
    draw = [0] * 20
    draw[14] = 1
    draw[18] = 1
    draw[19] = 1
    ds = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "sector_loading": loading,
            "overlap_sector_global": overlap,
            "global_score": score,
            "drawdown_label": draw,
        }
    )
    out = _build_label_direction_sanity(ds, train_end="2024-01-15", target_col="drawdown_label")
    assert out["status"] == "ok"
    ratio_train = float(out["features"]["sector_loading"]["splits"]["train"]["ratio_high_vs_base"])
    assert ratio_train >= 1.0
