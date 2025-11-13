from __future__ import annotations

import json
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
import requests

__all__ = [
    "FinanceMetricConfig",
    "HealthMetricConfig",
    "LogisticsMetricConfig",
    "fetch_finance_metrics",
    "fetch_health_metrics",
    "fetch_logistics_metrics",
    "MetricsDownloadError",
]


class MetricsDownloadError(RuntimeError):
    """Erro genérico para falhas nas chamadas de métricas externas."""


def _http_get(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Any:
    response = requests.get(url, params=params, headers=headers, timeout=30)
    if response.status_code != 200:
        raise MetricsDownloadError(f"Falha ao buscar {url}: HTTP {response.status_code}")
    try:
        return response.json()
    except json.JSONDecodeError as exc:
        raise MetricsDownloadError(f"Resposta inválida de {url}: {exc}") from exc


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _sanitize_filename(value: str) -> str:
    safe = value.lower().replace(" ", "_").replace(".", "_").replace("/", "_")
    return "".join(ch for ch in safe if ch.isalnum() or ch in {"_", "-"})


@dataclass
class FinanceMetricConfig:
    tickers: Sequence[str]
    interval: str = "d"  # stooq: d (diário), w (semanal), m (mensal)
    start: Optional[str] = None
    end: Optional[str] = None
    out_dir: Path = Path("dados/brutos/metrics/finance")


@dataclass
class HealthMetricConfig:
    country: str
    indicator: str
    start_year: int
    end_year: int
    out_dir: Path = Path("dados/brutos/metrics/health")


@dataclass
class LogisticsMetricConfig:
    start_date: str
    end_date: str
    out_dir: Path = Path("dados/brutos/metrics/logistics")
    limit: int = 5000


def fetch_finance_metrics(config: FinanceMetricConfig) -> List[Path]:
    base_url = "https://stooq.com/q/d/l/"
    config.out_dir.mkdir(parents=True, exist_ok=True)
    output_paths: List[Path] = []

    for ticker in config.tickers:
        params = {"s": ticker.lower(), "i": config.interval}
        response = requests.get(base_url, params=params, timeout=30)
        if response.status_code != 200:
            raise MetricsDownloadError(f"Falha ao buscar {ticker}: HTTP {response.status_code}")
        text = response.text.strip()
        if text.startswith("No data"):
            continue
        df = pd.read_csv(StringIO(text))
        if df.empty:
            continue
        df.rename(columns=str.lower, inplace=True)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if config.start:
            df = df[df["date"] >= pd.to_datetime(config.start)]
        if config.end:
            df = df[df["date"] <= pd.to_datetime(config.end)]
        df = df.dropna(subset=["date", "close"])
        records = [
            {
                "ticker": ticker,
                "date": row["date"].strftime("%Y-%m-%d"),
                "open": float(row.get("open", float("nan"))),
                "high": float(row.get("high", float("nan"))),
                "low": float(row.get("low", float("nan"))),
                "close": float(row.get("close", float("nan"))),
                "volume": float(row.get("volume", float("nan"))),
            }
            for _, row in df.iterrows()
        ]
        if not records:
            continue
        filename = _sanitize_filename(ticker) + ".jsonl"
        path = config.out_dir / filename
        _write_jsonl(path, records)
        output_paths.append(path)

    if not output_paths:
        raise MetricsDownloadError("Não foi possível baixar métricas financeiras para os tickers informados.")
    return output_paths


def fetch_health_metrics(config: HealthMetricConfig) -> List[Path]:
    base_url = f"https://api.worldbank.org/v2/country/{config.country}/indicator/{config.indicator}"
    params = {
        "format": "json",
        "per_page": 1000,
        "date": f"{config.start_year}:{config.end_year}",
    }
    payload = _http_get(base_url, params=params)
    if not isinstance(payload, list) or len(payload) < 2:
        raise MetricsDownloadError("Resposta inesperada da API do World Bank.")
    records_raw = payload[1] or []
    records = []
    for item in records_raw:
        records.append(
            {
                "country": item.get("country", {}).get("value"),
                "indicator": item.get("indicator", {}).get("value"),
                "date": item.get("date"),
                "value": item.get("value"),
            }
        )
    if not records:
        raise MetricsDownloadError("Nenhuma métrica de saúde encontrada na resposta.")
    config.out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{_sanitize_filename(config.country)}_{_sanitize_filename(config.indicator)}.jsonl"
    path = config.out_dir / filename
    _write_jsonl(path, records)
    return [path]


def fetch_logistics_metrics(config: LogisticsMetricConfig) -> List[Path]:
    base_url = "https://data.cityofnewyork.us/resource/h9gi-nx95.json"
    where_clause = (
        f"crash_date between '{config.start_date}T00:00:00.000' "
        f"and '{config.end_date}T23:59:59.999'"
    )
    params = {
        "$where": where_clause,
        "$limit": config.limit,
    }
    payload = _http_get(base_url, params=params)
    if not isinstance(payload, list):
        raise MetricsDownloadError("Resposta inesperada da API de logística (NYC).")
    records = []
    for item in payload:
        records.append(
            {
                "crash_date": item.get("crash_date"),
                "borough": item.get("borough"),
                "injured_persons": item.get("number_of_persons_injured"),
                "fatalities": item.get("number_of_persons_killed"),
                "vehicle_type": item.get("vehicle_type_code1"),
                "latitude": item.get("latitude"),
                "longitude": item.get("longitude"),
            }
        )
    if not records:
        raise MetricsDownloadError("Nenhuma métrica logística encontrada no período informado.")
    config.out_dir.mkdir(parents=True, exist_ok=True)
    base = _sanitize_filename(f"nyc_collisions_{config.start_date}_{config.end_date}")
    path = config.out_dir / f"{base}.jsonl"
    _write_jsonl(path, records)
    return [path]
