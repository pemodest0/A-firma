from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, List

import numpy as np
import pandas as pd

from graph_discovery.schemas import load_finance_records
from .utils import read_csv_bundle

__all__ = ["FinanceImportConfig", "load_finance_dataset"]


@dataclass
class FinanceImportConfig:
    sources: Sequence[Path]
    tickers: Optional[Sequence[str]] = None
    min_history: int = 250
    compute_returns: bool = True


def _attach_finance_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame.sort_values(["ticker", "date"], inplace=True)
    frame["return"] = frame.groupby("ticker")["close"].pct_change()
    frame["return_t1"] = frame.groupby("ticker")["return"].shift(-1)
    frame["vol_realized_short"] = (
        frame.groupby("ticker")["return"]
        .rolling(window=5, min_periods=3)
        .std()
        .rename("vol_realized_short")
        .reset_index(level=0, drop=True)
    )
    frame["vol_realized_long"] = (
        frame.groupby("ticker")["return"]
        .rolling(window=20, min_periods=10)
        .std()
        .rename("vol_realized_long")
        .reset_index(level=0, drop=True)
    )
    frame["vol_ratio"] = frame["vol_realized_short"] / frame["vol_realized_long"]
    frame["momentum"] = frame.groupby("ticker")["close"].pct_change(periods=10)
    frame["mom"] = frame["momentum"]
    rolling_max = frame.groupby("ticker")["close"].cummax()
    frame["drawdown"] = frame["close"] / rolling_max - 1.0
    volume_roll = (
        frame.groupby("ticker")["volume"]
        .rolling(window=20, min_periods=5)
        .mean()
        .rename("volume_mean_20")
        .reset_index(level=0, drop=True)
    )
    frame["volume_zscore"] = (
        frame["volume"] - volume_roll
    ) / frame.groupby("ticker")["volume"].transform(
        lambda s: s.rolling(window=20, min_periods=5).std()
    )
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    return frame


def load_finance_dataset(config: FinanceImportConfig) -> pd.DataFrame:
    rows = read_csv_bundle(config.sources)
    records = load_finance_records(rows)
    if config.tickers is not None:
        tickers = set(config.tickers)
        records = [rec for rec in records if rec.ticker in tickers]
    if not records:
        raise ValueError("Nenhum registro financeiro após filtragem.")
    df = pd.DataFrame(
        [
            {
                "date": rec.date,
                "ticker": rec.ticker,
                "close": rec.close,
                "volume": rec.volume,
                "high": rec.high,
                "low": rec.low,
                **rec.features,
                "return_t1": rec.label,
            }
            for rec in records
        ]
    )
    enriched = _attach_finance_features(df)
    counts = enriched.groupby("ticker")["date"].count()
    valid_tickers = counts[counts >= config.min_history].index
    filtered = enriched[enriched["ticker"].isin(valid_tickers)].reset_index(drop=True)
    if config.compute_returns:
        filtered.dropna(subset=["return", "return_t1"], inplace=True)
    return filtered
