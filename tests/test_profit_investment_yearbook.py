from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.bench.validation.run_profit_frontier_expansion_suite import StrategyResult
from scripts.bench.validation.run_profit_investment_yearbook import _calendar_rows


def test_calendar_rows_include_profit_and_operation_days() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2025-01-02"])
    ret = pd.Series([0.10, -0.05, 0.20], index=idx, dtype=float)
    bench = pd.Series([0.02, 0.01, 0.03], index=idx, dtype=float)
    turnover = pd.Series([0.0, 0.5, 1.0], index=idx, dtype=float)
    result = StrategyResult(
        suite="test",
        candidate_id="candidate",
        family="test",
        benchmark_ticker="TEST",
        gross_ret=ret,
        turnover=turnover,
        net_ret=ret,
        benchmark_net_ret=bench,
        net_ann_return=0.0,
        net_total_return=0.0,
        net_sharpe=0.0,
        net_max_drawdown=0.0,
        edge_vs_benchmark=0.0,
        avg_turnover_daily=float(turnover.mean()),
        hit_rate_10x_5y=float("nan"),
        years_to_10x_full=float("nan"),
        notes="",
    )
    rows = _calendar_rows(result=result, capital_brl=10000.0)
    assert len(rows) == 2
    first = rows[0]
    assert first["year"] == 2024
    assert np.isfinite(first["profit_brl"])
    assert first["operation_days"] == 1
