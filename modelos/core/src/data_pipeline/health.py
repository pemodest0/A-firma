from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from graph_discovery.schemas import load_health_records
from .utils import read_csv_bundle

__all__ = ["HealthImportConfig", "load_health_dataset"]


@dataclass
class HealthImportConfig:
    sources: Sequence[Path]
    metric_filter: Optional[Sequence[str]] = None
    min_history: int = 30
    smoothing_window: int = 7


def _attach_health_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    frame = df.copy()
    frame.sort_values(["entity_id", "metric", "timestamp"], inplace=True)
    group_cols = ["entity_id", "metric"]
    rolling = frame.groupby(group_cols)["value"]
    frame["value_ma"] = rolling.transform(lambda s: s.rolling(window=window, min_periods=max(3, window // 2)).mean())
    frame["value_std"] = rolling.transform(lambda s: s.rolling(window=window, min_periods=max(3, window // 2)).std())
    frame["value_trend"] = rolling.transform(lambda s: s.diff())
    frame["value_zscore"] = (frame["value"] - frame["value_ma"]) / frame["value_std"]
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    return frame


def load_health_dataset(config: HealthImportConfig) -> pd.DataFrame:
    rows = read_csv_bundle(config.sources)
    records = load_health_records(rows)
    if config.metric_filter is not None:
        metrics = set(config.metric_filter)
        records = [rec for rec in records if rec.metric in metrics]
    if not records:
        raise ValueError("Nenhum registro de saúde após filtragem.")
    df = pd.DataFrame(
        [
            {
                "timestamp": rec.timestamp,
                "entity_id": rec.entity_id,
                "metric": rec.metric,
                "value": rec.value,
                **rec.features,
            }
            for rec in records
        ]
    )
    enriched = _attach_health_features(df, window=config.smoothing_window)
    counts = enriched.groupby(["entity_id", "metric"])["timestamp"].count()
    valid_groups = counts[counts >= config.min_history].index
    mask = enriched.set_index(["entity_id", "metric"]).index.isin(valid_groups)
    filtered = enriched[mask].reset_index(drop=True)
    filtered.dropna(subset=["value", "value_ma"], inplace=True)
    return filtered
