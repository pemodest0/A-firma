#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_file_with_metadata(src: Path, dst: Path) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "source": str(src),
        "copy": str(dst),
        "sha256": _file_sha256(src),
        "size_bytes": int(src.stat().st_size),
        "mtime_utc": datetime.fromtimestamp(src.stat().st_mtime, UTC).isoformat(),
    }


def _extract_tickers(table: pd.DataFrame | None) -> list[str]:
    if table is None or table.empty:
        return []
    for col in ("ticker", "asset_id", "asset", "symbol"):
        if col in table.columns:
            return sorted({str(v).strip() for v in table[col].astype(str) if str(v).strip()})
    return []


def snapshot_validation_inputs(
    *,
    outdir: Path,
    label: str,
    prices_dir: Path,
    metadata_files: dict[str, Path] | None = None,
    universe_tables: dict[str, pd.DataFrame] | None = None,
    extra_tickers: list[str] | None = None,
) -> dict[str, Any]:
    root = outdir / "input_snapshot" / str(label)
    prices_out = root / "prices"
    metadata_out = root / "metadata"
    root.mkdir(parents=True, exist_ok=True)

    metadata_records: dict[str, Any] = {}
    for key, path in sorted((metadata_files or {}).items()):
        src = Path(path).resolve()
        if not src.exists():
            metadata_records[str(key)] = {"missing": True, "source": str(src)}
            continue
        metadata_records[str(key)] = _copy_file_with_metadata(src, metadata_out / src.name)

    tickers: set[str] = set()
    for table in (universe_tables or {}).values():
        tickers.update(_extract_tickers(table))
    for ticker in extra_tickers or []:
        clean = str(ticker).strip()
        if clean:
            tickers.add(clean)

    price_records: dict[str, Any] = {}
    missing_tickers: list[str] = []
    for ticker in sorted(tickers):
        src = (Path(prices_dir).resolve() / f"{ticker}.csv").resolve()
        if not src.exists():
            missing_tickers.append(str(ticker))
            price_records[str(ticker)] = {"missing": True, "source": str(src)}
            continue
        price_records[str(ticker)] = _copy_file_with_metadata(src, prices_out / src.name)

    summary = {
        "label": str(label),
        "generated_at": datetime.now(UTC).isoformat(),
        "prices_dir": str(Path(prices_dir).resolve()),
        "ticker_count": int(len(tickers)),
        "tickers_missing": missing_tickers,
        "metadata": metadata_records,
        "prices": price_records,
    }
    (root / "snapshot_manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
