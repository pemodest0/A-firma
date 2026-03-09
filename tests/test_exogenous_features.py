from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from engine.portfolio.exogenous_features import build_exogenous_feature_panel


def test_build_exogenous_feature_panel_outputs_expected_columns(tmp_path: Path) -> None:
    idx = pd.date_range("2022-01-01", periods=260, freq="D")
    def _make_price(path: Path, level: float, drift: float) -> None:
        r = np.full(len(idx), drift, dtype=float)
        price = level * np.exp(np.cumsum(r))
        df = pd.DataFrame({"date": idx.strftime("%Y-%m-%d"), "price": price, "r": r})
        df.to_csv(path, index=False)

    prices_dir = tmp_path / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)
    for ticker, level, drift in [
        ("^VIX", 20.0, 0.0005),
        ("UUP", 25.0, 0.0002),
        ("HYG", 80.0, 0.0003),
        ("LQD", 110.0, 0.0001),
        ("TLT", 120.0, -0.0001),
        ("SHY", 85.0, 0.00005),
        ("TIP", 105.0, 0.00008),
    ]:
        _make_price(prices_dir / f"{ticker}.csv", level, drift)

    cols = ["BTC-USD", "ETH-USD", "ADA-USD"]
    crypto_r = pd.DataFrame(
        {
            "BTC-USD": np.full(len(idx), 0.001, dtype=float),
            "ETH-USD": np.full(len(idx), 0.0008, dtype=float),
            "ADA-USD": np.full(len(idx), 0.0006, dtype=float),
        },
        index=idx,
    )
    crypto_p = 100.0 * np.exp(crypto_r.cumsum())

    out = build_exogenous_feature_panel(
        prices_dir=prices_dir,
        crypto_returns=crypto_r,
        crypto_prices=crypto_p,
        benchmark_crypto="BTC-USD",
    )
    expected = {
        "funding",
        "open_interest",
        "liquidation",
        "btc_dominance",
        "breadth",
        "crypto_dependency_risk",
        "VIX",
        "credit_spreads",
        "rates",
        "dollar",
        "liquidity",
        "macro_stress",
        "exogenous_risk",
    }
    assert expected.issubset(set(out.panel.columns))
    assert ((out.panel[list(expected)] >= 0.0) & (out.panel[list(expected)] <= 1.0)).all().all()
