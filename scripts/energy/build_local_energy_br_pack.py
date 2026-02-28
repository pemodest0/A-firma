#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _latest_energy_pack(results_dir: Path) -> Path:
    packs = sorted(
        [p for p in results_dir.iterdir() if p.is_dir() and p.name.startswith("local_pack_")],
        key=lambda p: p.name,
        reverse=True,
    )
    for p in packs:
        if (p / "panel_long_sector.csv").exists() and (p / "universe_fixed.csv").exists():
            return p
    raise FileNotFoundError(f"no local_pack_* found in {results_dir}")


def _sector_internal_for_row(ticker: str, sector: str) -> str:
    t = str(ticker).upper()
    s = str(sector).lower().strip()
    if t.startswith("EAR_") or "storage" in s:
        return "reservatorios_ear"
    if t.startswith("CMO_") or "price" in s:
        return "cmo_operacional"
    return "carga_sin"


def _build_metadata(universe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in universe.iterrows():
        ticker = str(r.get("ticker", "")).strip()
        sector = str(r.get("sector", "energy")).strip() or "energy"
        if not ticker:
            continue
        name = f"ONS {ticker}"
        if ticker.startswith("ONS_"):
            name = f"Carga {ticker.replace('ONS_', '')}"
        elif ticker.startswith("EAR_"):
            name = f"EAR {ticker.replace('EAR_', '')}"
        elif ticker.startswith("CMO_"):
            name = f"CMO {ticker.replace('CMO_', '')}"
        rows.append(
            {
                "asset_id": ticker,
                "ticker": ticker,
                "name": name,
                "sector_gics": "energy",
                "sector_internal": _sector_internal_for_row(ticker=ticker, sector=sector),
                "source": "ONS Open Data",
                "license": "Dados abertos ONS",
                "liquidity_proxy": "",
            }
        )
    md = pd.DataFrame(rows).drop_duplicates(subset=["asset_id"], keep="first")
    return md.sort_values(["sector_internal", "ticker"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build local Energia BR daily pack (canonical format for Eigen Engine).")
    ap.add_argument("--energy-pack-dir", type=str, default="")
    ap.add_argument("--energy-pack-results-dir", type=str, default="results/energy_download")
    ap.add_argument("--results-dir", type=str, default="results/energy_br")
    ap.add_argument("--write-canonical", type=int, default=1)
    args = ap.parse_args()

    if str(args.energy_pack_dir).strip():
        src_pack = Path(args.energy_pack_dir)
        if not src_pack.is_absolute():
            src_pack = ROOT / str(args.energy_pack_dir)
    else:
        base = Path(args.energy_pack_results_dir)
        if not base.is_absolute():
            base = ROOT / str(args.energy_pack_results_dir)
        src_pack = _latest_energy_pack(base)

    panel_path = src_pack / "panel_long_sector.csv"
    universe_path = src_pack / "universe_fixed.csv"
    if not panel_path.exists() or not universe_path.exists():
        raise SystemExit(f"missing files in source pack: {src_pack}")

    panel = pd.read_csv(panel_path)
    universe = pd.read_csv(universe_path)
    if panel.empty or universe.empty:
        raise SystemExit("source pack empty")

    required = {"date", "ticker", "sector", "r"}
    if not required.issubset(set(panel.columns)):
        raise SystemExit(f"panel missing columns: {required - set(panel.columns)}")
    panel = panel[["date", "ticker", "sector", "r"]].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.dropna(subset=["date", "ticker", "r"]).copy()
    panel["date"] = panel["date"].dt.strftime("%Y-%m-%d")
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    if panel.empty:
        raise SystemExit("empty panel after cleaning")

    if not {"ticker", "sector"}.issubset(set(universe.columns)):
        universe["sector"] = "energy"
    universe = universe[["ticker", "sector"]].dropna().copy()
    universe["ticker"] = universe["ticker"].astype(str).str.strip()
    universe["sector"] = universe["sector"].astype(str).str.strip().replace("", "energy")
    universe = universe[universe["ticker"] != ""].drop_duplicates(subset=["ticker"], keep="first")
    universe = universe[universe["ticker"].isin(set(panel["ticker"].astype(str)))].copy()
    universe = universe.sort_values(["sector", "ticker"]).reset_index(drop=True)
    if universe.empty:
        raise SystemExit("empty universe after cleaning")

    md = _build_metadata(universe=universe)
    if md.empty:
        raise SystemExit("metadata empty")

    run_id = f"local_pack_{_run_id()}"
    out_base = Path(args.results_dir)
    if not out_base.is_absolute():
        out_base = ROOT / str(args.results_dir)
    outdir = out_base / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    (outdir / "panel_long_sector.csv").write_text(panel.to_csv(index=False), encoding="utf-8")
    (outdir / "panel_long_energy_br.csv").write_text(panel.to_csv(index=False), encoding="utf-8")
    (outdir / "universe_fixed.csv").write_text(universe.to_csv(index=False), encoding="utf-8")
    (outdir / "asset_metadata_energy_br.csv").write_text(md.to_csv(index=False), encoding="utf-8")

    canonical_paths: dict[str, str] = {}
    if bool(int(args.write_canonical)):
        canonical = ROOT / "data" / "processed" / "energy" / "br_daily"
        canonical.mkdir(parents=True, exist_ok=True)
        (canonical / "panel_long_energy_br.csv").write_text(panel.to_csv(index=False), encoding="utf-8")
        (canonical / "universe_fixed.csv").write_text(universe.to_csv(index=False), encoding="utf-8")
        (canonical / "asset_metadata_energy_br.csv").write_text(md.to_csv(index=False), encoding="utf-8")
        canonical_paths = {
            "panel": str(canonical / "panel_long_energy_br.csv"),
            "universe": str(canonical / "universe_fixed.csv"),
            "metadata": str(canonical / "asset_metadata_energy_br.csv"),
        }

    meta = {
        "status": "ok",
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_pack": str(src_pack),
        "outdir": str(outdir),
        "rows_panel": int(panel.shape[0]),
        "assets_ok": int(universe.shape[0]),
        "period_start": str(panel["date"].min()),
        "period_end": str(panel["date"].max()),
        "canonical_paths": canonical_paths,
    }
    (outdir / "build_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
