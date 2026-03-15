from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.bench.validation.validation_input_snapshot import snapshot_validation_inputs


def test_snapshot_validation_inputs_copies_selected_files(tmp_path: Path) -> None:
    prices_dir = tmp_path / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)
    for ticker in ("BTC-USD", "SPY"):
        (prices_dir / f"{ticker}.csv").write_text("date,price\n2025-01-01,100\n", encoding="utf-8")
    groups = tmp_path / "groups.csv"
    meta = tmp_path / "meta.csv"
    groups.write_text("group,asset\ng1,BTC-USD\n", encoding="utf-8")
    meta.write_text("ticker\nBTC-USD\n", encoding="utf-8")
    universe = pd.DataFrame({"ticker": ["BTC-USD"]})

    out = snapshot_validation_inputs(
        outdir=tmp_path / "artifact",
        label="demo",
        prices_dir=prices_dir,
        metadata_files={"groups": groups, "meta": meta},
        universe_tables={"crypto": universe},
        extra_tickers=["SPY"],
    )

    snap_dir = tmp_path / "artifact" / "input_snapshot" / "demo"
    assert out["ticker_count"] == 2
    assert (snap_dir / "prices" / "BTC-USD.csv").exists()
    assert (snap_dir / "prices" / "SPY.csv").exists()
    assert (snap_dir / "metadata" / "groups.csv").exists()
    manifest = json.loads((snap_dir / "snapshot_manifest.json").read_text(encoding="utf-8"))
    assert manifest["label"] == "demo"
    assert manifest["tickers_missing"] == []
