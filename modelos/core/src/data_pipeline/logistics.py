from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from graph_discovery.schemas import load_logistics_records
from .utils import read_csv_bundle

__all__ = ["LogisticsImportConfig", "load_logistics_dataset"]


@dataclass
class LogisticsImportConfig:
    sources: Sequence[Path]
    status_filter: Optional[Sequence[str]] = None
    min_orders: int = 50


def _attach_logistics_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame.sort_values(["order_id", "day"], inplace=True)
    frame["window_hours"] = (frame["window_end"] - frame["window_start"]).dt.total_seconds() / 3600.0
    if "eta_real" in frame.columns:
        frame["lateness"] = (frame["eta_real"] - frame["eta_pred"]) / 60.0
    else:
        frame["lateness"] = np.nan
    frame["order_count_driver"] = frame.groupby("driver_id")["order_id"].transform("count")
    frame["cost_ratio"] = frame["cost"] / frame.groupby("driver_id")["cost"].transform(
        lambda s: s.rolling(window=30, min_periods=5).mean()
    )
    frame["status_flag"] = frame["status"].isin(["late", "failed"]).astype(int)
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    return frame


def load_logistics_dataset(config: LogisticsImportConfig) -> pd.DataFrame:
    rows = read_csv_bundle(config.sources)
    records = load_logistics_records(rows)
    if config.status_filter is not None:
        status_set = set(config.status_filter)
        records = [rec for rec in records if rec.status in status_set]
    if not records:
        raise ValueError("Nenhum registro logístico após filtragem.")
    df = pd.DataFrame(
        [
            {
                "day": rec.day,
                "order_id": rec.order_id,
                "origin": rec.origin,
                "destination": rec.destination,
                "window_start": rec.window_start,
                "window_end": rec.window_end,
                "driver_id": rec.driver_id,
                "vehicle": rec.vehicle,
                "status": rec.status,
                "eta_pred": rec.eta_pred,
                "eta_real": rec.eta_real,
                "cost": rec.cost,
                "failure_reason": rec.failure_reason,
                **rec.features,
            }
            for rec in records
        ]
    )
    enriched = _attach_logistics_features(df)
    counts = enriched.groupby("driver_id")["order_id"].count()
    valid_drivers = counts[counts >= config.min_orders].index
    filtered = enriched[enriched["driver_id"].isin(valid_drivers)].reset_index(drop=True)
    return filtered
