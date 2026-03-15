from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.bench.validation.run_profit_country_compare_suite import (
    _filter_brazil_equities,
    _write_synthetic_benchmark,
)


def test_filter_brazil_equities_keeps_only_brazil_rows(tmp_path: Path) -> None:
    groups = tmp_path / "groups.csv"
    meta = tmp_path / "meta.csv"
    pd.DataFrame(
        [
            {"asset": "PETR4.SA", "group": "equities_br_bluechips"},
            {"asset": "VALE3.SA", "group": "equities_br_bluechips"},
            {"asset": "SPY", "group": "broad_equity"},
        ]
    ).to_csv(groups, index=False)
    pd.DataFrame(
        [
            {"asset_id": "PETR4.SA", "ticker": "PETR4.SA", "sector_gics": "equities_br_bluechips", "sector_internal": "equities_br_bluechips", "liquidity_proxy": 1000},
            {"asset_id": "VALE3.SA", "ticker": "VALE3.SA", "sector_gics": "equities_br_bluechips", "sector_internal": "equities_br_bluechips", "liquidity_proxy": 1000},
            {"asset_id": "SPY", "ticker": "SPY", "sector_gics": "broad_equity", "sector_internal": "broad_equity", "liquidity_proxy": 1000},
        ]
    ).to_csv(meta, index=False)

    br_groups, br_meta, tickers = _filter_brazil_equities(equity_groups=groups, equity_meta=meta, outdir=tmp_path)

    assert tickers == ["PETR4.SA", "VALE3.SA"]
    assert pd.read_csv(br_groups)["asset"].tolist() == ["PETR4.SA", "VALE3.SA"]
    assert pd.read_csv(br_meta)["ticker"].tolist() == ["PETR4.SA", "VALE3.SA"]


def test_write_synthetic_benchmark_creates_price_series(tmp_path: Path) -> None:
    prices_dir = tmp_path / "prices"
    prices_dir.mkdir()
    dates = pd.date_range("2024-01-01", periods=260, freq="B")
    for ticker, scale in [("PETR4.SA", 1.0), ("VALE3.SA", 1.5)]:
        price = pd.Series(100.0 + scale * pd.RangeIndex(len(dates)), index=dates, dtype=float)
        log_price = np.log(price.astype(float))
        frame = pd.DataFrame(
            {
                "date": dates,
                "price": price.to_numpy(dtype=float),
                "log_price": log_price.to_numpy(dtype=float),
                "r": log_price.diff().fillna(0.0).to_numpy(dtype=float),
            }
        )
        frame.to_csv(prices_dir / f"{ticker}.csv", index=False)

    out = _write_synthetic_benchmark(
        prices_dir=prices_dir,
        tickers=["PETR4.SA", "VALE3.SA"],
        outdir=prices_dir,
        benchmark_ticker="BR_SYNTH",
    )

    bench = pd.read_csv(out)
    assert out.exists()
    assert list(bench.columns) == ["date", "price", "log_price", "r"]
    assert bench["price"].iloc[-1] > bench["price"].iloc[0]
