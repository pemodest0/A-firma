from __future__ import annotations

import pandas as pd

from scripts.ops.build_profit_shadow_benchmarks import _equal_weight_return, _momentum_topk_return


def test_equal_weight_return_uses_present_sector_columns() -> None:
    idx = pd.to_datetime(["2026-01-02", "2026-01-05"])
    returns = pd.DataFrame({"XLK": [0.01, 0.02], "XLF": [0.00, 0.01]}, index=idx)

    out, used = _equal_weight_return(returns, ["XLK", "XLF", "XLV"])

    assert used == ["XLK", "XLF"]
    assert round(float(out.iloc[0]), 6) == 0.005
    assert round(float(out.iloc[1]), 6) == 0.015


def test_momentum_topk_return_rotates_to_best_asset_after_lookback() -> None:
    idx = pd.date_range("2026-01-01", periods=80, freq="B")
    returns = pd.DataFrame({"SPY": 0.0001, "QQQ": 0.0020, "IEF": -0.0002}, index=idx)

    out, picks = _momentum_topk_return(returns, tickers=["SPY", "QQQ", "IEF"], lookback_days=21, top_k=1, fallback_ticker="SPY")

    assert out.shape[0] == len(idx)
    assert picks.iloc[-1] == "QQQ"
    assert float(out.iloc[-1]) == float(returns.iloc[-1]["QQQ"])
