from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _to_series(series: pd.Series | np.ndarray | list[float]) -> pd.Series:
    if isinstance(series, pd.Series):
        s = series.copy()
    else:
        s = pd.Series(np.asarray(series, dtype=float))
    s = pd.to_numeric(s, errors="coerce")
    return s


def rolling_variance(series: pd.Series | np.ndarray | list[float], window: int) -> pd.Series:
    s = _to_series(series)
    w = int(max(2, window))
    return s.rolling(window=w, min_periods=w).var(ddof=0)


def _ac1_window(x: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    a = np.asarray(x[:-1], dtype=float)
    b = np.asarray(x[1:], dtype=float)
    ma = float(np.mean(a))
    mb = float(np.mean(b))
    da = a - ma
    db = b - mb
    va = float(np.sum(da**2))
    vb = float(np.sum(db**2))
    denom = float(np.sqrt(va * vb))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(da * db) / denom)


def rolling_ac1(series: pd.Series | np.ndarray | list[float], window: int) -> pd.Series:
    s = _to_series(series)
    w = int(max(3, window))

    def _fn(arr: np.ndarray) -> float:
        return _ac1_window(np.asarray(arr, dtype=float))

    return s.rolling(window=w, min_periods=w).apply(_fn, raw=True)


def _zscore_against_train(series: pd.Series, train_mask: pd.Series) -> pd.Series:
    tr = pd.to_numeric(series[train_mask], errors="coerce")
    mu = float(tr.mean(skipna=True))
    if not np.isfinite(mu):
        mu = 0.0
    sd = float(tr.std(ddof=0, skipna=True))
    if (not np.isfinite(sd)) or sd <= 1e-12:
        sd = 1.0
    z = (pd.to_numeric(series, errors="coerce") - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan)


def ews_pack(
    series: pd.Series | np.ndarray | list[float],
    window: int,
    train_end: Any | None = None,
) -> dict[str, pd.Series]:
    s = _to_series(series)
    rv = rolling_variance(s, window=window)
    ac1 = rolling_ac1(s, window=window)

    if train_end is None:
        train_mask = pd.Series(True, index=s.index)
    else:
        if isinstance(s.index, pd.DatetimeIndex):
            cutoff = pd.Timestamp(train_end)
            train_mask = pd.Series(s.index <= cutoff, index=s.index)
        else:
            if isinstance(train_end, int):
                train_mask = pd.Series(np.arange(len(s)) <= int(train_end), index=s.index)
            else:
                train_mask = pd.Series(True, index=s.index)

    z_var = _zscore_against_train(rv, train_mask=train_mask)
    z_ac1 = _zscore_against_train(ac1, train_mask=train_mask)
    return {
        "var": rv,
        "ac1": ac1,
        "z_var": z_var,
        "z_ac1": z_ac1,
    }
