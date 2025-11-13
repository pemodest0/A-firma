from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class DataQualityReport:
    rows: int
    start_date: Optional[pd.Timestamp]
    end_date: Optional[pd.Timestamp]
    duplicate_dates: int
    duplicate_date_list: List[str]
    non_positive_prices: int
    non_positive_dates: List[str]
    missing_business_days: int
    missing_business_day_list: List[str]
    outlier_returns: int
    outlier_dates: List[str]
    jump_threshold_pct: float

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        # Convert Timestamps to string for JSON serialization
        for key in ("start_date", "end_date"):
            value = payload.get(key)
            if isinstance(value, pd.Timestamp):
                payload[key] = value.isoformat()
        return payload

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)


def _coerce_price(price_series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(price_series):
        return price_series.astype(float)
    cleaned = (
        price_series.astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def analyze_price_series(
    df: pd.DataFrame,
    date_col: str = "date",
    price_col: str = "price",
    jump_threshold_pct: float = 15.0,
    zscore_threshold: float = 4.0,
) -> DataQualityReport:
    if date_col not in df.columns or price_col not in df.columns:
        raise KeyError(f"columns '{date_col}' and '{price_col}' must exist")

    frame = df.copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame.sort_values(date_col, inplace=True)
    frame.dropna(subset=[date_col], inplace=True)
    frame[price_col] = _coerce_price(frame[price_col])

    duplicates_mask = frame[date_col].duplicated(keep=False)
    duplicate_dates = frame.loc[duplicates_mask, date_col].dt.strftime("%Y-%m-%d").tolist()

    non_positive_mask = frame[price_col] <= 0
    non_positive_dates = frame.loc[non_positive_mask, date_col].dt.strftime("%Y-%m-%d").tolist()

    # Missing business days
    if frame.empty:
        business_gaps: List[str] = []
    else:
        bday_index = pd.date_range(
            frame[date_col].min(), frame[date_col].max(), freq="B"
        )
        missing = sorted(set(bday_index.date) - set(frame[date_col].dt.date))
        business_gaps = [pd.Timestamp(day).strftime("%Y-%m-%d") for day in missing]

    # Outlier detection via log-returns
    price_series = frame[price_col]
    returns = np.log(price_series).diff()
    if returns.dropna().empty:
        outlier_dates = []
    else:
        zscores = (returns - returns.mean()) / returns.std(ddof=0)
        outlier_mask = zscores.abs() > zscore_threshold
        outlier_dates = frame.loc[outlier_mask, date_col].dt.strftime("%Y-%m-%d").tolist()

    # Large jumps (absolute pct change > threshold)
    pct_change = price_series.pct_change().abs() * 100.0
    jump_mask = pct_change > jump_threshold_pct
    jump_dates = frame.loc[jump_mask, date_col].dt.strftime("%Y-%m-%d").tolist()

    combined_outliers = sorted(set(outlier_dates + jump_dates))

    report = DataQualityReport(
        rows=int(frame.shape[0]),
        start_date=frame[date_col].min() if not frame.empty else None,
        end_date=frame[date_col].max() if not frame.empty else None,
        duplicate_dates=len(duplicate_dates),
        duplicate_date_list=duplicate_dates,
        non_positive_prices=int(non_positive_mask.sum()),
        non_positive_dates=non_positive_dates,
        missing_business_days=len(business_gaps),
        missing_business_day_list=business_gaps,
        outlier_returns=len(combined_outliers),
        outlier_dates=combined_outliers,
        jump_threshold_pct=float(jump_threshold_pct),
    )
    return report


def summarize_report(report: DataQualityReport) -> str:
    lines = [
        f"Total linhas: {report.rows}",
        f"Início: {report.start_date}, Fim: {report.end_date}",
        f"Datas duplicadas: {report.duplicate_dates}",
        f"Preços não positivos: {report.non_positive_prices}",
        f"Dias Úteis sem cotação: {report.missing_business_days}",
        f"Outliers detectados (> {report.jump_threshold_pct}% ou Z-score alto): {report.outlier_returns}",
    ]
    return "\n".join(lines)

