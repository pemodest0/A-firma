from __future__ import annotations

import pandas as pd

from scripts.ops.build_profit_shadow_scorecard import _perf_from_simple_returns, _window_total
from scripts.ops.run_profit_shadow_canonical import _candidate_run_id


def test_window_total_compounds_last_rows_only() -> None:
    series = pd.Series([0.10, -0.05, 0.02, 0.03], dtype=float)

    out = _window_total(series, 2)

    assert round(float(out), 6) == round((1.02 * 1.03) - 1.0, 6)


def test_perf_from_simple_returns_reports_drawdown() -> None:
    series = pd.Series([0.10, -0.20, 0.05], dtype=float)

    perf = _perf_from_simple_returns(series)

    assert perf["total_return"] < 0.0
    assert perf["max_drawdown"] < -0.1999


def test_candidate_run_id_appends_candidate_name() -> None:
    assert _candidate_run_id("20260306T120000Z", "attack_v2") == "20260306T120000Z_attack_v2"
