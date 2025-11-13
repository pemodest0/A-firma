from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class FinanceRecord:
    date: pd.Timestamp
    ticker: str
    close: float
    volume: float
    high: float
    low: float
    features: Dict[str, float]
    label: Optional[float] = None


@dataclass
class LogisticsRecord:
    day: pd.Timestamp
    order_id: str
    origin: str
    destination: str
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    driver_id: str
    vehicle: str
    status: str
    eta_pred: float
    eta_real: Optional[float]
    cost: float
    failure_reason: Optional[str]
    features: Dict[str, float]


@dataclass
class HealthRecord:
    timestamp: pd.Timestamp
    entity_id: str
    metric: str
    value: float
    features: Dict[str, float]


@dataclass
class PhysicsRecord:
    time: float
    system_id: str
    features: Dict[str, float]


def load_finance_records(
    table: Iterable[Dict],
    feature_columns: Optional[Sequence[str]] = None,
    label_column: str = "return_t1",
) -> List[FinanceRecord]:
    df = pd.DataFrame(table)
    if df.empty:
        return []
    normalized_cols = {col: col.lower() for col in df.columns}
    rename_map = {}
    for col, lower in normalized_cols.items():
        if lower == "adj close":
            rename_map[col] = "close"
        elif lower == "adj close*":
            rename_map[col] = "close"
        elif lower == "close":
            rename_map[col] = "close"
        elif lower == "open":
            continue
        elif lower == "high":
            rename_map[col] = "high"
        elif lower == "low":
            rename_map[col] = "low"
        elif lower == "volume":
            rename_map[col] = "volume"
        elif lower == "ticker":
            rename_map[col] = "ticker"
    df = df.rename(columns=rename_map)
    if "ticker" not in df.columns and "symbol" in df.columns:
        df = df.rename(columns={"symbol": "ticker"})
    required = {"date", "ticker", "close", "volume", "high", "low"}
    missing = required - set(df.columns)
    if "close" not in df.columns and "price" in df.columns:
        df = df.rename(columns={"price": "close"})
    if "high" not in df.columns:
        df["high"] = df["close"]
    if "low" not in df.columns:
        df["low"] = df["close"]
    if "volume" not in df.columns:
        df["volume"] = np.nan  # type: ignore[name-defined]
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing finance columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"])
    if feature_columns is None:
        numeric_cols = df.select_dtypes(include=["number"]).columns
        feature_columns = [col for col in numeric_cols if col not in {"close", "volume", "high", "low", label_column}]

    records: List[FinanceRecord] = []
    for _, row in df.iterrows():
        features = {col: float(row[col]) for col in feature_columns if pd.notna(row[col])}
        label = float(row[label_column]) if label_column in row and pd.notna(row[label_column]) else None
        records.append(
            FinanceRecord(
                date=row["date"],
                ticker=str(row["ticker"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                high=float(row["high"]),
                low=float(row["low"]),
                features=features,
                label=label,
            )
        )
    return records


def load_logistics_records(
    table: Iterable[Dict],
    feature_columns: Optional[Sequence[str]] = None,
) -> List[LogisticsRecord]:
    df = pd.DataFrame(table)
    required = {
        "day",
        "order_id",
        "origin",
        "destination",
        "window_start",
        "window_end",
        "driver_id",
        "vehicle",
        "status",
        "eta_prev",
        "cost",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing logistics columns: {sorted(missing)}")
    df["day"] = pd.to_datetime(df["day"])
    df["window_start"] = pd.to_datetime(df["window_start"])
    df["window_end"] = pd.to_datetime(df["window_end"])
    df["eta_prev"] = pd.to_numeric(df["eta_prev"], errors="coerce")
    if "eta_real" in df.columns:
        df["eta_real"] = pd.to_numeric(df["eta_real"], errors="coerce")
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce")

    if feature_columns is None:
        numeric_cols = df.select_dtypes(include=["number"]).columns
        feature_columns = [
            col
            for col in numeric_cols
            if col not in {"eta_prev", "eta_real", "cost"}
        ]

    records: List[LogisticsRecord] = []
    for _, row in df.iterrows():
        features = {col: float(row[col]) for col in feature_columns if pd.notna(row[col])}
        records.append(
            LogisticsRecord(
                day=row["day"],
                order_id=str(row["order_id"]),
                origin=str(row["origin"]),
                destination=str(row["destination"]),
                window_start=row["window_start"],
                window_end=row["window_end"],
                driver_id=str(row["driver_id"]),
                vehicle=str(row["vehicle"]),
                status=str(row["status"]),
                eta_pred=float(row["eta_prev"]) if pd.notna(row["eta_prev"]) else float("nan"),
                eta_real=float(row["eta_real"]) if "eta_real" in row and pd.notna(row["eta_real"]) else None,
                cost=float(row["cost"]) if pd.notna(row["cost"]) else float("nan"),
                failure_reason=str(row["failure_reason"]) if "failure_reason" in row and pd.notna(row["failure_reason"]) else None,
                features=features,
            )
        )
    return records


def load_health_records(
    table: Iterable[Dict],
    feature_columns: Optional[Sequence[str]] = None,
    value_column: str = "value",
) -> List[HealthRecord]:
    df = pd.DataFrame(table)
    required = {"timestamp", "entity_id", "metric", value_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing health columns: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df[value_column] = pd.to_numeric(df[value_column], errors="coerce")
    if feature_columns is None:
        numeric_cols = df.select_dtypes(include=["number"]).columns
        feature_columns = [
            col for col in numeric_cols if col not in {value_column}
        ]
    records: List[HealthRecord] = []
    for _, row in df.iterrows():
        features = {col: float(row[col]) for col in feature_columns if pd.notna(row[col])}
        value = float(row[value_column]) if pd.notna(row[value_column]) else float("nan")
        records.append(
            HealthRecord(
                timestamp=row["timestamp"],
                entity_id=str(row["entity_id"]),
                metric=str(row["metric"]),
                value=value,
                features=features,
            )
        )
    return records


def load_physics_records(
    table: Iterable[Dict],
    time_column: str = "time",
    system_column: str = "system_id",
    feature_columns: Optional[Sequence[str]] = None,
) -> List[PhysicsRecord]:
    df = pd.DataFrame(table)
    required = {time_column, system_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing physics columns: {sorted(missing)}")
    df[time_column] = pd.to_numeric(df[time_column], errors="coerce")
    if feature_columns is None:
        numeric_cols = df.select_dtypes(include=["number"]).columns
        feature_columns = [
            col for col in numeric_cols if col not in {time_column}
        ]
    records: List[PhysicsRecord] = []
    for _, row in df.iterrows():
        features = {col: float(row[col]) for col in feature_columns if pd.notna(row[col])}
        records.append(
            PhysicsRecord(
                time=float(row[time_column]),
                system_id=str(row[system_column]),
                features=features,
            )
        )
    return records
