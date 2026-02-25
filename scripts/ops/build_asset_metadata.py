#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.structural.run_manifest import write_run_manifest  # noqa: E402

DEFAULT_FINANCE_BASE = ROOT / "results" / "finance_download"


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _latest_finance_run() -> Path:
    if not DEFAULT_FINANCE_BASE.exists():
        raise FileNotFoundError(f"missing finance base: {DEFAULT_FINANCE_BASE}")
    runs = sorted([p for p in DEFAULT_FINANCE_BASE.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
    for d in runs:
        if (d / "universe_fixed.csv").exists():
            return d
    raise FileNotFoundError("no finance run with universe_fixed.csv")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build canonical asset metadata CSV.")
    ap.add_argument("--universe-csv", type=str, default="")
    ap.add_argument("--groups-csv", type=str, default="data/asset_groups_470_enriched.csv")
    ap.add_argument("--out-csv", type=str, default="data/asset_metadata.csv")
    ap.add_argument("--manifest-outdir", type=str, default="")
    args = ap.parse_args()

    if str(args.universe_csv).strip():
        universe_path = ROOT / str(args.universe_csv).strip()
    else:
        universe_path = _latest_finance_run() / "universe_fixed.csv"
    groups_path = ROOT / str(args.groups_csv).strip()
    out_path = ROOT / str(args.out_csv).strip()

    uni = _read_csv(universe_path)
    if uni.empty:
        raise SystemExit(f"empty or missing universe csv: {universe_path}")
    if "ticker" not in uni.columns:
        raise SystemExit(f"universe csv missing ticker column: {universe_path}")
    if "sector" not in uni.columns:
        uni["sector"] = "unknown"

    grp = _read_csv(groups_path)
    if not grp.empty:
        if "asset" not in grp.columns:
            grp = pd.DataFrame()
        else:
            grp = grp.rename(columns={"asset": "ticker"})
            if "group" not in grp.columns:
                grp["group"] = pd.NA

    base = uni.copy()
    base["ticker"] = base["ticker"].astype(str).str.strip()
    base["asset_id"] = base["ticker"]
    base["sector_gics"] = base["sector"].astype(str).str.strip()
    base["liquidity_proxy"] = pd.to_numeric(base.get("n_rows"), errors="coerce")

    if not grp.empty:
        g = grp[["ticker", "group"]].copy()
        g["ticker"] = g["ticker"].astype(str).str.strip()
        g = g.drop_duplicates(subset=["ticker"], keep="first")
        base = base.merge(g, on="ticker", how="left")
        base["sector_internal"] = base["group"].astype(str).str.strip()
        base["sector_internal"] = base["sector_internal"].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    else:
        base["sector_internal"] = pd.NA

    base["sector_internal"] = base["sector_internal"].fillna(base["sector_gics"])
    out = base[["asset_id", "ticker", "sector_gics", "sector_internal", "liquidity_proxy"]].copy()
    out = out.dropna(subset=["asset_id", "ticker"]).drop_duplicates(subset=["asset_id"], keep="first")
    out = out.sort_values(["sector_gics", "ticker"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_universe_csv": str(universe_path),
        "source_groups_csv": str(groups_path),
        "out_csv": str(out_path),
        "n_assets": int(out.shape[0]),
        "n_sector_gics": int(out["sector_gics"].nunique()),
        "n_sector_internal": int(out["sector_internal"].nunique()),
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if str(args.manifest_outdir).strip():
        manifest_outdir = ROOT / str(args.manifest_outdir).strip()
    else:
        manifest_outdir = ROOT / "results" / "ops" / "asset_metadata" / _run_id()
    manifest_outdir.mkdir(parents=True, exist_ok=True)

    write_run_manifest(
        manifest_outdir,
        script="scripts/ops/build_asset_metadata.py",
        params={
            "universe_csv": str(universe_path),
            "groups_csv": str(groups_path),
            "out_csv": str(out_path),
            "manifest_outdir": str(manifest_outdir),
        },
        paths={
            "asset_metadata_csv": str(out_path),
            "asset_metadata_meta_json": str(meta_path),
            "manifest_dir": str(manifest_outdir),
        },
        gates={
            "asset_metadata_nonempty": bool(out.shape[0] > 0),
            "asset_id_unique": bool(not out["asset_id"].duplicated().any()),
        },
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
