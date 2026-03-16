#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.finance.yf_fetch_or_load import CRYPTO_BINANCE_MAP, save_cache, unify_to_daily  # noqa: E402


WINDOW_STARTS_MS = [
    int(pd.Timestamp("2016-01-01", tz="UTC").timestamp() * 1000),
    int(pd.Timestamp("2018-09-27", tz="UTC").timestamp() * 1000),
    int(pd.Timestamp("2021-06-23", tz="UTC").timestamp() * 1000),
    int(pd.Timestamp("2024-03-19", tz="UTC").timestamp() * 1000),
]


def _read_tickers(path: Path) -> list[str]:
    with path.open() as handle:
        return [str(row["ticker"]).strip().upper() for row in csv.DictReader(handle) if row.get("ticker")]


def _fetch_klines(symbol: str, start_ms: int) -> list[list[object]]:
    url = (
        "https://api.binance.com/api/v3/klines"
        f"?symbol={symbol}&interval=1d&limit=1000&startTime={int(start_ms)}"
    )
    proc = subprocess.run(
        ["/usr/bin/curl", "-L", "--fail", "--silent", "--show-error", "--ipv4", url],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    return payload if isinstance(payload, list) else []


def _fetch_symbol_history(symbol: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for start_ms in WINDOW_STARTS_MS:
        for row in _fetch_klines(symbol, start_ms):
            rows.append(
                {
                    "date": pd.to_datetime(row[0], unit="ms", errors="coerce"),
                    "price": pd.to_numeric(row[4], errors="coerce"),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["date", "price"])
    frame = pd.DataFrame(rows).dropna()
    return frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Seed do universo cripto expandido via Binance.")
    ap.add_argument(
        "--metadata",
        default="data/asset_metadata_crypto_top_liquid_expanded.csv",
        help="CSV com o universo expandido.",
    )
    ap.add_argument(
        "--base-dir",
        default=".",
        help="Raiz do repositório para salvar em data/raw/finance/yfinance_daily.",
    )
    args = ap.parse_args()

    metadata = (ROOT / args.metadata).resolve()
    base_dir = (ROOT / args.base_dir).resolve()
    tickers = _read_tickers(metadata)

    written = 0
    missing: list[str] = []
    for ticker in tickers:
        symbols = CRYPTO_BINANCE_MAP.get(ticker) or []
        if not symbols:
            missing.append(ticker)
            continue
        frame = pd.DataFrame()
        for symbol in symbols:
            try:
                frame = _fetch_symbol_history(symbol)
            except Exception:
                frame = pd.DataFrame()
            if not frame.empty:
                break
        if frame.empty:
            missing.append(ticker)
            continue
        daily = unify_to_daily(frame)
        if daily.empty:
            missing.append(ticker)
            continue
        save_cache(daily, base_dir, ticker)
        written += 1

    print(json.dumps({"written": written, "missing": missing}, ensure_ascii=True))


if __name__ == "__main__":
    main()
