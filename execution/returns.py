from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


VALID_RETURN_KINDS = {"log", "simple"}


def _normalize_return_kind(kind: str) -> str:
    value = str(kind or "").strip().lower()
    if value not in VALID_RETURN_KINDS:
        raise ValueError(f"unsupported return kind: {kind}")
    return value


def convert_return_series(
    values: pd.Series | np.ndarray | list[float],
    *,
    source_kind: str = "log",
    target_kind: str = "simple",
) -> pd.Series:
    src = _normalize_return_kind(source_kind)
    tgt = _normalize_return_kind(target_kind)
    out = pd.to_numeric(pd.Series(values, copy=True), errors="coerce").astype(float)
    if src == tgt:
        return out
    if src == "log" and tgt == "simple":
        return np.expm1(out)
    if src == "simple" and tgt == "log":
        return np.log1p(out)
    raise ValueError(f"unsupported conversion: {src} -> {tgt}")


def load_return_frame_csv(
    path: Path,
    *,
    date_col: str = "date",
    return_col: str = "r",
    source_kind: str = "log",
    target_kind: str = "simple",
    business_days_only: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    if date_col not in df.columns or return_col not in df.columns:
        raise ValueError(f"missing required columns [{date_col}, {return_col}] in {path}")
    out = df[[date_col, return_col]].copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[return_col] = convert_return_series(out[return_col], source_kind=source_kind, target_kind=target_kind)
    out = out.dropna(subset=[date_col, return_col]).sort_values(date_col).drop_duplicates(subset=[date_col], keep="last")
    if business_days_only:
        out = out[out[date_col].dt.dayofweek < 5]
    return out.reset_index(drop=True)


def load_return_series_csv(
    path: Path,
    *,
    date_col: str = "date",
    return_col: str = "r",
    source_kind: str = "log",
    target_kind: str = "simple",
    business_days_only: bool = False,
    series_name: str | None = None,
) -> pd.Series:
    out = load_return_frame_csv(
        path,
        date_col=date_col,
        return_col=return_col,
        source_kind=source_kind,
        target_kind=target_kind,
        business_days_only=business_days_only,
    )
    series = out.set_index(date_col)[return_col].astype(float).sort_index()
    if series_name is not None:
        series = series.rename(str(series_name))
    return series


def compound_simple_returns(simple_returns: pd.Series | np.ndarray | list[float]) -> float:
    x = pd.to_numeric(pd.Series(simple_returns, copy=True), errors="coerce").dropna().astype(float)
    if x.empty:
        return float("nan")
    return float(np.prod(1.0 + x.to_numpy(dtype=float)) - 1.0)


def daily_simple_to_monthly(simple_returns: pd.Series) -> pd.Series:
    s = pd.to_numeric(pd.Series(simple_returns, copy=True), errors="coerce").dropna().astype(float)
    if s.empty:
        return pd.Series(dtype=float)
    idx = pd.to_datetime(s.index, errors="coerce")
    s = s[idx.notna()].copy()
    s.index = pd.DatetimeIndex(idx[idx.notna()])
    monthly = s.groupby(s.index.to_period("M")).apply(compound_simple_returns)
    monthly.index = monthly.index.astype(str)
    return monthly.astype(float)
