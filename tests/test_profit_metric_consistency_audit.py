from __future__ import annotations

import math

import pandas as pd

from scripts.bench.validation.run_profit_metric_consistency_audit import _compare_scalar, _series_metrics


class _DummyResult:
    def __init__(self) -> None:
        idx = pd.date_range("2025-01-01", periods=5, freq="D")
        self.net_ret = pd.Series([0.01, -0.02, 0.03, 0.00, 0.01], index=idx, dtype=float)
        self.benchmark_net_ret = pd.Series([0.0, -0.01, 0.01, 0.00, 0.0], index=idx, dtype=float)
        self.turnover = pd.Series([0.0, 0.1, 0.0, 0.2, 0.0], index=idx, dtype=float)


def test_compare_scalar_accepts_small_difference() -> None:
    row = _compare_scalar(scope="x", metric="y", expected=1.0, actual=1.0 + 1e-12, tolerance=1e-9)
    assert row["within_tolerance"] is True
    assert float(row["abs_diff"]) < 1e-9


def test_series_metrics_returns_basic_fields() -> None:
    out = _series_metrics(_DummyResult())
    assert set(out) == {
        "net_total_return",
        "net_ann_return",
        "net_max_drawdown",
        "net_sharpe",
        "edge_vs_benchmark",
        "avg_turnover_daily",
        "underperform_prob_63",
    }
    assert math.isfinite(float(out["net_total_return"]))
    assert math.isfinite(float(out["avg_turnover_daily"]))
