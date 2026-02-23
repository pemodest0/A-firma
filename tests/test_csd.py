from __future__ import annotations

import numpy as np
import pandas as pd

from engine.structural.csd import ews_pack, rolling_ac1, rolling_variance


def _ar1_series(n: int = 1200, rho: float = 0.8, seed: int = 23) -> pd.Series:
    rng = np.random.default_rng(int(seed))
    eps = rng.normal(0.0, 1.0, size=int(n))
    x = np.zeros(int(n), dtype=float)
    for i in range(1, int(n)):
        x[i] = float(rho) * x[i - 1] + eps[i]
    idx = pd.date_range("2020-01-01", periods=int(n), freq="D")
    return pd.Series(x, index=idx)


def test_rolling_variance_non_negative() -> None:
    s = _ar1_series()
    rv = rolling_variance(s, window=60)
    assert (rv.dropna() >= 0.0).all()


def test_rolling_ac1_approximates_ar1_rho() -> None:
    rho = 0.75
    s = _ar1_series(rho=rho)
    ac1 = rolling_ac1(s, window=200)
    m = float(ac1.dropna().tail(200).mean())
    assert abs(m - rho) < 0.15


def test_ews_pack_outputs_expected_keys() -> None:
    s = _ar1_series()
    out = ews_pack(s, window=80, train_end="2021-12-31")
    assert {"var", "ac1", "z_var", "z_ac1"}.issubset(out.keys())
