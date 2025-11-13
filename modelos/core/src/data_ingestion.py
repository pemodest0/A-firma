from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

DATE_CANDIDATES = (
    "date",
    "Date",
    "DATA",
    "Data",
    "DATE",
    "Fecha",
    "timestamp",
)

PRICE_CANDIDATES = (
    "Adj Close",
    "Adj Close*",
    "Close",
    "Último",
    "Price",
    "Preço",
    "price",
    "close",
)


@dataclass
class NormalizationResult:
    frame: pd.DataFrame
    date_column: str
    price_column: str


def _guess_column(columns, candidates) -> Optional[str]:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _coerce_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    cleaned = (
        series.astype(str)
        .str.replace(r"[^0-9,.\-]", "", regex=True)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def normalize_price_csv(
    input_path: Path,
    output_path: Optional[Path] = None,
    date_column: Optional[str] = None,
    price_column: Optional[str] = None,
    separator: Optional[str] = None,
    decimal: Optional[str] = None,
    thousands: Optional[str] = None,
) -> NormalizationResult:
    """
    Convert arbitrary financial CSV into standard date/price format.
    """
    read_kwargs = {"engine": "python"}
    if separator:
        read_kwargs["sep"] = separator
    else:
        read_kwargs["sep"] = None
    if decimal:
        read_kwargs["decimal"] = decimal
    if thousands:
        read_kwargs["thousands"] = thousands
    df = pd.read_csv(input_path, **read_kwargs)
    if df.empty:
        raise ValueError("input CSV is empty")

    date_col = date_column or _guess_column(df.columns, DATE_CANDIDATES)
    price_col = price_column or _guess_column(df.columns, PRICE_CANDIDATES)
    if date_col is None or price_col is None:
        raise KeyError("Unable to detect date or price columns automatically.")

    # recombine split decimal columns if needed
    extra_cols = [c for c in df.columns if "Unnamed" in c and c != date_col]
    if extra_cols:
        for extra in extra_cols:
            mask = df[extra].notna() & (df[extra].astype(str).str.strip() != "")
            if mask.any() and price_col in df.columns:
                df.loc[mask, price_col] = (
                    df.loc[mask, price_col].astype(str).str.strip()
                    + "."
                    + df.loc[mask, extra].astype(str).str.strip()
                )
        df.drop(columns=extra_cols, inplace=True)

    frame = df[[date_col, price_col]].copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame[price_col] = _coerce_numeric(frame[price_col])
    frame.dropna(subset=[date_col, price_col], inplace=True)
    frame.sort_values(date_col, inplace=True)
    frame.drop_duplicates(subset=[date_col], keep="last", inplace=True)
    frame.rename(columns={date_col: "date", price_col: "price"}, inplace=True)
    frame.reset_index(drop=True, inplace=True)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)

    return NormalizationResult(frame=frame, date_column=date_col, price_column=price_col)
