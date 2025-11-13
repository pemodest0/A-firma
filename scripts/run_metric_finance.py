#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

from data_pipeline.metrics import (
    FinanceMetricConfig,
    MetricsDownloadError,
    fetch_finance_metrics,
)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Baixa métricas financeiras via Stooq.")
    parser.add_argument("--tickers", nargs="+", required=True, help="Tickers (formato stooq, ex.: SPY.US).")
    parser.add_argument("--interval", default="d", help="Intervalo (d=diário, w=semanal, m=mensal).")
    parser.add_argument("--start", help="Data inicial YYYY-MM-DD (opcional).")
    parser.add_argument("--end", help="Data final YYYY-MM-DD (opcional).")
    parser.add_argument("--out-dir", type=Path, default=Path("dados/brutos/metrics/finance"), help="Diretório de saída.")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    cfg = FinanceMetricConfig(
        tickers=args.tickers,
        interval=args.interval,
        start=args.start,
        end=args.end,
        out_dir=args.out_dir,
    )
    try:
        paths = fetch_finance_metrics(cfg)
    except MetricsDownloadError as exc:
        print(f"[finance] erro: {exc}")
        return
    for path in paths:
        print(f"[finance] salvo: {path}")


if __name__ == "__main__":
    main()
