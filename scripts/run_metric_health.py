#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

from data_pipeline.metrics import (
    HealthMetricConfig,
    MetricsDownloadError,
    fetch_health_metrics,
)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Baixa indicadores de saúde (World Bank).")
    parser.add_argument("--country", required=True, help="País (ISO Alpha-3).")
    parser.add_argument("--indicator", required=True, help="ID do indicador (ex.: SH.XPD.CHEX.PC.CD).")
    parser.add_argument("--start-year", type=int, default=2000, help="Ano inicial.")
    parser.add_argument("--end-year", type=int, default=2023, help="Ano final.")
    parser.add_argument("--out-dir", type=Path, default=Path("dados/brutos/metrics/health"), help="Diretório de saída.")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    cfg = HealthMetricConfig(
        country=args.country,
        indicator=args.indicator,
        start_year=args.start_year,
        end_year=args.end_year,
        out_dir=args.out_dir,
    )
    try:
        paths = fetch_health_metrics(cfg)
    except MetricsDownloadError as exc:
        print(f"[health] erro: {exc}")
        return
    for path in paths:
        print(f"[health] salvo: {path}")


if __name__ == "__main__":
    main()
