#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

from data_pipeline import (
    FinanceImportConfig,
    LogisticsImportConfig,
    HealthImportConfig,
    load_finance_dataset,
    load_logistics_dataset,
    load_health_dataset,
)

OUTPUT_PATH = Path("results/forecast_metrics.json")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_finance() -> Dict:
    cfg = _load_config(Path("dados/configs/data_pipeline_finance.json"))
    finance_cfg = cfg["finance"]
    dataset = load_finance_dataset(
        FinanceImportConfig(
            sources=[Path(src) for src in finance_cfg["sources"]],
            tickers=finance_cfg.get("tickers"),
            min_history=int(finance_cfg.get("min_history", 250)),
            compute_returns=bool(finance_cfg.get("compute_returns", True)),
        )
    )
    max_rows = finance_cfg.get("max_rows")
    if max_rows:
        dataset = dataset.tail(int(max_rows)).reset_index(drop=True)
    return {
        "data": dataset,
        "label_column": cfg["selection"].get("label_column", "return_t1"),
    }


def _load_logistics() -> Dict:
    cfg = _load_config(Path("dados/configs/data_pipeline_logistics.json"))
    logistics_cfg = cfg["logistics"]
    dataset = load_logistics_dataset(
        LogisticsImportConfig(
            sources=[Path(src) for src in logistics_cfg["sources"]],
            status_filter=logistics_cfg.get("status_filter"),
            min_orders=int(logistics_cfg.get("min_orders", 50)),
        )
    )
    max_rows = logistics_cfg.get("max_rows")
    if max_rows:
        dataset = dataset.tail(int(max_rows)).reset_index(drop=True)
    return {
        "data": dataset,
        "label_column": cfg["selection"].get("label_column", "lateness"),
    }


def _load_health() -> Dict:
    cfg = _load_config(Path("dados/configs/data_pipeline_health.json"))
    health_cfg = cfg["health"]
    dataset = load_health_dataset(
        HealthImportConfig(
            sources=[Path(src) for src in health_cfg["sources"]],
            metric_filter=health_cfg.get("metric_filter"),
            min_history=int(health_cfg.get("min_history", 30)),
            smoothing_window=int(health_cfg.get("smoothing_window", 7)),
        )
    )
    max_rows = health_cfg.get("max_rows")
    if max_rows:
        dataset = dataset.tail(int(max_rows)).reset_index(drop=True)
    return {
        "data": dataset,
        "label_column": cfg["selection"].get("label_column", "value"),
    }


def evaluate_finance() -> Dict[str, float]:
    payload = _load_finance()
    df = payload["data"]
    label_col = payload["label_column"]
    y = df[label_col].to_numpy(dtype=float)

    baseline_zero = np.zeros_like(y)
    mae_zero = float(np.mean(np.abs(y - baseline_zero)))
    rmse_zero = float(np.sqrt(np.mean((y - baseline_zero) ** 2)))

    walk_pred = df[label_col].rolling(5, min_periods=1).mean().shift(1)
    walk_pred.fillna(df[label_col].mean(), inplace=True)
    walk_values = walk_pred.to_numpy(dtype=float)
    mae_walk = float(np.mean(np.abs(y - walk_values)))
    rmse_walk = float(np.sqrt(np.mean((y - walk_values) ** 2)))

    return {
        "mae_zero": mae_zero,
        "rmse_zero": rmse_zero,
        "mae_walk": mae_walk,
        "rmse_walk": rmse_walk,
    }


def evaluate_logistics() -> Dict[str, float]:
    payload = _load_logistics()
    df = payload["data"]
    label_col = payload["label_column"]
    on_time = (df[label_col] <= 0).astype(float)
    df = df.assign(on_time=on_time)

    baseline_prob = 0.5
    brier_zero = float(np.mean((on_time - baseline_prob) ** 2))
    accuracy_zero = float(np.mean(on_time == (baseline_prob >= 0.5)))

    walk_prob = (
        df.groupby("driver_id")["on_time"]
        .transform(lambda s: s.expanding().mean().shift(1))
        .fillna(baseline_prob)
    )
    brier_walk = float(np.mean((on_time - walk_prob) ** 2))
    predictions = (walk_prob >= 0.5).astype(float)
    accuracy_walk = float(np.mean(predictions == on_time))

    return {
        "brier_zero": brier_zero,
        "accuracy_zero": accuracy_zero,
        "brier_walk": brier_walk,
        "accuracy_walk": accuracy_walk,
    }


def evaluate_health() -> Dict[str, float]:
    payload = _load_health()
    df = payload["data"]
    label_col = payload["label_column"]
    values = df[label_col].to_numpy(dtype=float)

    baseline_const = float(np.mean(values))
    mae_zero = float(np.mean(np.abs(values - baseline_const)))
    rmse_zero = float(np.sqrt(np.mean((values - baseline_const) ** 2)))

    walk_pred = (
        df.groupby("entity_id")[label_col]
        .transform(lambda s: s.rolling(3, min_periods=1).mean().shift(1))
        .fillna(baseline_const)
    )
    walk_values = walk_pred.to_numpy(dtype=float)
    mae_walk = float(np.mean(np.abs(values - walk_values)))
    rmse_walk = float(np.sqrt(np.mean((values - walk_values) ** 2)))

    return {
        "mae_zero": mae_zero,
        "rmse_zero": rmse_zero,
        "mae_walk": mae_walk,
        "rmse_walk": rmse_walk,
    }


def main() -> None:
    metrics = {
        "finance": evaluate_finance(),
        "logistics": evaluate_logistics(),
        "health": evaluate_health(),
    }
    OUTPUT_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Métricas salvas em {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
