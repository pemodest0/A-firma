from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from graph_discovery.schemas import load_physics_records
from .utils import read_csv_bundle

__all__ = ["PhysicsImportConfig", "load_physics_dataset"]


@dataclass
class PhysicsImportConfig:
    sources: Sequence[Path]
    system_filter: Optional[Sequence[str]] = None
    min_steps: int = 50
    smoothing_window: int = 5
    derive_velocity: bool = True


def _attach_physics_features(df: pd.DataFrame, window: int, derive_velocity: bool) -> pd.DataFrame:
    frame = df.copy()
    frame.sort_values(["system_id", "time"], inplace=True)
    state_cols = [col for col in frame.columns if col not in {"system_id", "time"}]
    group = frame.groupby("system_id")
    for col in state_cols:
        rolling = group[col].transform(lambda s: s.rolling(window=window, min_periods=max(3, window // 2)).mean())
        frame[f"{col}_ma"] = rolling
        frame[f"{col}_std"] = group[col].transform(
            lambda s: s.rolling(window=window, min_periods=max(3, window // 2)).std()
        )
        frame[f"{col}_zscore"] = (frame[col] - frame[f"{col}_ma"]) / frame[f"{col}_std"]
    if derive_velocity:
        for col in state_cols:
            diff = group[col].diff()
            delta_t = group["time"].diff().replace(0, np.nan)
            frame[f"{col}_velocity"] = diff / delta_t
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    return frame


def load_physics_dataset(config: PhysicsImportConfig) -> pd.DataFrame:
    rows = read_csv_bundle(config.sources)
    records = load_physics_records(rows)
    if config.system_filter is not None:
        systems = set(config.system_filter)
        records = [rec for rec in records if rec.system_id in systems]
    if not records:
        raise ValueError("Nenhum registro físico após filtragem.")
    feature_keys = sorted({key for rec in records for key in rec.features.keys()})
    df = pd.DataFrame(
        [
            {
                "time": rec.time,
                "system_id": rec.system_id,
                **{key: rec.features.get(key, np.nan) for key in feature_keys},
            }
            for rec in records
        ]
    )
    enriched = _attach_physics_features(df, window=config.smoothing_window, derive_velocity=config.derive_velocity)
    counts = enriched.groupby("system_id")["time"].count()
    valid_systems = counts[counts >= config.min_steps].index
    filtered = enriched[enriched["system_id"].isin(valid_systems)].reset_index(drop=True)
    filtered.dropna(subset=["time"], inplace=True)
    return filtered
