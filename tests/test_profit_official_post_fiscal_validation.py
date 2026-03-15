from __future__ import annotations

import pandas as pd

from scripts.bench.validation.run_profit_official_post_fiscal_validation import (
    _forward_return_distribution,
    _leave_one_year_out,
    _topk_crypto_share,
)


def test_leave_one_year_out_produces_one_row_per_year() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-06-03", "2025-01-02", "2025-06-03"])
    ret = pd.Series([0.10, 0.00, 0.05, 0.00], index=idx, dtype=float)

    out = _leave_one_year_out(ret)

    assert out["excluded_year"].tolist() == [2024, 2025]
    assert out.shape[0] == 2


def test_forward_return_distribution_uses_forward_window_only() -> None:
    idx = pd.date_range("2025-01-01", periods=5, freq="D")
    ret = pd.Series([0.10, 0.00, 0.10, 0.00, 0.10], index=idx, dtype=float)

    out = _forward_return_distribution(ret, horizon=2)

    assert out.index[0] == idx[0]
    assert round(float(out.iloc[0]), 6) == round((1.0 * 1.10) - 1.0, 6)


def test_topk_crypto_share_reports_internal_crypto_concentration() -> None:
    weights = pd.DataFrame(
        {
            "BTC-USD": [0.40, 0.20],
            "ETH-USD": [0.10, 0.10],
            "SOL-USD": [0.05, 0.20],
            "cash": [0.45, 0.50],
        },
        index=pd.to_datetime(["2025-01-01", "2025-01-02"]),
    )

    mean_share, max_share = _topk_crypto_share(weights, crypto_tickers=["BTC-USD", "ETH-USD", "SOL-USD"], k=1)

    assert 0.0 < mean_share <= 1.0
    assert 0.0 < max_share <= 1.0
