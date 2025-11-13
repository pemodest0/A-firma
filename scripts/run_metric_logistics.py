#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

from data_pipeline.metrics import (
    LogisticsMetricConfig,
    MetricsDownloadError,
    fetch_logistics_metrics,
)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Baixa métricas de logística (NYC collisions dataset).")
    parser.add_argument("--start", required=True, help="Data inicial YYYY-MM-DD.")
    parser.add_argument("--end", required=True, help="Data final YYYY-MM-DD.")
    parser.add_argument("--limit", type=int, default=5000, help="Limite de registros da API.")
    parser.add_argument("--out-dir", type=Path, default=Path("dados/brutos/metrics/logistics"), help="Diretório de saída.")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    cfg = LogisticsMetricConfig(
        start_date=args.start,
        end_date=args.end,
        limit=args.limit,
        out_dir=args.out_dir,
    )
    try:
        paths = fetch_logistics_metrics(cfg)
    except MetricsDownloadError as exc:
        print(f"[logistics] erro: {exc}")
        return
    for path in paths:
        print(f"[logistics] salvo: {path}")


if __name__ == "__main__":
    main()
