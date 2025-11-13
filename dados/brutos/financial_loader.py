from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except Exception:  # pragma: no cover - yfinance might be absent
    yf = None  # type: ignore
    YFINANCE_AVAILABLE = False

__all__ = [
    "YFINANCE_AVAILABLE",
    "PriceSeries",
    "ReturnWindow",
    "download_price_series",
    "load_price_csv",
    "load_value_csv",
    "prepare_returns",
    "discretize_returns",
    "generate_return_windows",
]


@dataclass(frozen=True)
class PriceSeries:
    """Container holding raw price data and associated returns."""

    data: pd.DataFrame
    price_column: str
    return_column: str


@dataclass(frozen=True)
class ReturnWindow:
    """Sliding window over returns mapped to a discrete distribution."""

    start: pd.Timestamp
    end: pd.Timestamp
    returns: pd.Series
    distribution: np.ndarray


def download_price_series(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1d",
    price_column: str = "Adj Close",
) -> pd.DataFrame:
    """
    Download price data using yfinance.

    Parameters
    ----------
    symbol:
        Ticker identifier understood by Yahoo! Finance (e.g. 'SPY', '^BVSP').
    start, end:
        Optional ISO8601 date strings. yfinance falls back to maximum range when omitted.
    interval:
        Sampling interval, defaults to daily ('1d').
    price_column:
        Column to keep from the downloaded frame. Defaults to adjusted close.
    """
    if not YFINANCE_AVAILABLE:  # pragma: no cover - runtime guard
        raise RuntimeError("yfinance is required for download_price_series but is not installed.")

    df = yf.download(
        symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    if df.empty:
        raise ValueError(f"No data returned for symbol {symbol} (start={start}, end={end}, interval={interval}).")
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(symbol, axis=1, level=-1, drop_level=True)
        except (KeyError, IndexError):
            df.columns = [
                " ".join(str(part) for part in col if part not in ("", symbol))
                if isinstance(col, tuple)
                else str(col)
                for col in df.columns
            ]
    df.columns = [str(col) for col in df.columns]
    if price_column not in df.columns:
        candidate = None
        symbolized = f"{price_column} {symbol}"
        if symbolized in df.columns:
            candidate = symbolized
        elif price_column.lower() == "adj close" and "Adj Close" in df.columns:
            candidate = "Adj Close"
        elif price_column.lower() == "adj close" and "Close" in df.columns:
            candidate = "Close"
        if candidate is None:
            raise KeyError(
                f"Column '{price_column}' not found in downloaded dataset. "
                f"Available columns: {', '.join(df.columns)}"
            )
        price_column = candidate
    series = (
        df[[price_column]]
        .rename(columns={price_column: "price"})
        .dropna()
    )
    series = series.asfreq("D")
    series = series.ffill()
    series.reset_index(inplace=True)
    series.rename(columns={"Date": "date"}, inplace=True)
    return series


def load_value_csv(path: Path, date_column: str = "date", value_column: str = "price") -> pd.DataFrame:
    """
    Load a CSV file containing a date column and a value column.

    Parameters
    ----------
    path:
        Location of the CSV file.
    date_column:
        Column containing the timestamps.
    value_column:
        Column containing the raw values/prices.
    """
    df = pd.read_csv(path)
    if date_column not in df.columns:
        raise KeyError(f"CSV must contain a '{date_column}' column.")
    if value_column not in df.columns:
        raise KeyError(f"CSV must contain a '{value_column}' column.")
    frame = df[[date_column, value_column]].copy()
    frame.rename(columns={date_column: "date", value_column: "price"}, inplace=True)
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame.sort_values("date", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def load_price_csv(path: Path, price_column: str = "price", date_column: str = "date") -> pd.DataFrame:
    """Backward compatible wrapper for financial CSV files."""
    return load_value_csv(path, date_column=date_column, value_column=price_column)


def prepare_returns(frame: pd.DataFrame, method: str = "log") -> PriceSeries:
    """
    Append a return column to the price frame.

    Parameters
    ----------
    frame:
        DataFrame with columns 'date' and 'price'.
    method:
        'log' for logarithmic returns, 'simple' for percentage change.
    """
    if "date" not in frame or "price" not in frame:
        raise KeyError("Input frame must contain 'date' and 'price' columns.")

    df = frame.copy()
    df["price"] = df["price"].astype(float)
    df["price"] = df["price"].replace(0.0, np.nan)
    df.dropna(subset=["price"], inplace=True)

    if method == "log":
        df["return"] = np.log(df["price"]).diff()
    elif method == "simple":
        df["return"] = df["price"].pct_change()
    elif method == "diff":
        df["return"] = df["price"].diff()
    else:
        raise ValueError("method must be 'log', 'simple', or 'diff'.")

    df.dropna(subset=["return"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return PriceSeries(data=df, price_column="price", return_column="return")


def discretize_returns(
    returns: pd.Series,
    num_bins: int = 51,
    method: str = "quantile",
    clip: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert continuous returns into discrete bins.

    Parameters
    ----------
    returns:
        Series of return values.
    num_bins:
        Total number of discrete bins/nodes. Should be an odd number to keep symmetry.
    method:
        'quantile' for equal-mass bins, 'linear' for equally-spaced bins.
    clip:
        Optional (min, max) to clip extreme returns before binning.

    Returns
    -------
    bin_edges, bin_centers : np.ndarray
        Arrays describing the binning used to map real returns to discrete indices.
    """
    if clip is not None:
        low, high = clip
        values = returns.clip(low, high).to_numpy()
    else:
        values = returns.to_numpy(copy=True)

    if method == "quantile":
        quantiles = np.linspace(0.0, 1.0, num_bins + 1)
        edges = np.quantile(values, quantiles)
        edges[0] = values.min()
        edges[-1] = values.max()
    elif method == "linear":
        edges = np.linspace(values.min(), values.max(), num_bins + 1)
    else:
        raise ValueError("method must be 'quantile' or 'linear'.")

    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def _distribution_from_counts(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    if total == 0:
        return np.full_like(counts, 1.0 / counts.size, dtype=float)
    return counts / total


def _bin_indices(values: Iterable[float], edges: np.ndarray) -> np.ndarray:
    idx = np.digitize(values, edges, right=False) - 1
    return np.clip(idx, 0, edges.size - 2)


def generate_return_windows(
    series: PriceSeries,
    bin_edges: np.ndarray,
    window: int = 30,
    step: int = 5,
) -> List[ReturnWindow]:
    """
    Produce sliding windows of return distributions using the provided binning.

    Parameters
    ----------
    series:
        PriceSeries created by prepare_returns.
    bin_edges:
        Output edges from discretize_returns.
    window:
        Number of consecutive days per window.
    step:
        Hop size between windows.
    """
    returns = series.data[series.return_column]
    timestamps = series.data["date"]

    windows: List[ReturnWindow] = []
    for start in range(0, returns.size - window + 1, step):
        stop = start + window
        chunk = returns.iloc[start:stop]
        indices = _bin_indices(chunk.to_numpy(), bin_edges)
        counts = np.bincount(indices, minlength=bin_edges.size - 1)
        distribution = _distribution_from_counts(counts)
        windows.append(
            ReturnWindow(
                start=timestamps.iloc[start],
                end=timestamps.iloc[stop - 1],
                returns=chunk,
                distribution=distribution,
            )
        )
    return windows
