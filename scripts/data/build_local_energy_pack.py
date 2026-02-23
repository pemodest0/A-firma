#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_csv_auto(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.shape[1] == 1 and ";" in str(df.columns[0]):
        df = pd.read_csv(path, sep=";")
    return df


def _load_ons_rows(paths: list[Path]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in paths:
        try:
            raw = _read_csv_auto(path)
        except Exception:
            continue
        if raw.empty:
            continue

        col_map = {str(c).strip().lower(): str(c) for c in raw.columns}
        req = {
            "id_subsistema": col_map.get("id_subsistema"),
            "nom_subsistema": col_map.get("nom_subsistema"),
            "din_instante": col_map.get("din_instante"),
            "val_cargaenergiamwmed": col_map.get("val_cargaenergiamwmed"),
        }
        if any(v is None for v in req.values()):
            continue

        df = raw[[req["id_subsistema"], req["nom_subsistema"], req["din_instante"], req["val_cargaenergiamwmed"]]].copy()
        df.columns = ["id_subsistema", "nom_subsistema", "din_instante", "val_cargaenergiamwmed"]
        df["din_instante"] = pd.to_datetime(df["din_instante"], errors="coerce")
        df["val_cargaenergiamwmed"] = pd.to_numeric(df["val_cargaenergiamwmed"], errors="coerce")
        df = df.dropna(subset=["din_instante", "val_cargaenergiamwmed", "id_subsistema"]).copy()
        if df.empty:
            continue
        df["source_file"] = str(path)
        parts.append(df)

    if not parts:
        return pd.DataFrame(
            columns=[
                "id_subsistema",
                "nom_subsistema",
                "din_instante",
                "val_cargaenergiamwmed",
                "source_file",
            ]
        )
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["din_instante", "id_subsistema", "source_file"]).drop_duplicates(
        subset=["din_instante", "id_subsistema"], keep="last"
    )
    return out.reset_index(drop=True)


def _returns_from_value(v: pd.Series) -> pd.Series:
    s = pd.to_numeric(v, errors="coerce")
    return np.log(s.clip(lower=1e-9)).diff()


def _build_panel(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    universe_rows: list[dict[str, Any]] = []

    for subsystem, part in df.groupby("id_subsistema", dropna=True):
        x = part.sort_values("din_instante").copy()
        if x.empty:
            continue
        ticker = f"ONS_{str(subsystem).strip().upper()}"
        x["ticker"] = ticker
        x["sector"] = "energy"
        x["date"] = x["din_instante"].dt.date.astype(str)
        x["value"] = pd.to_numeric(x["val_cargaenergiamwmed"], errors="coerce")
        x["r"] = _returns_from_value(x["value"])
        x = x.dropna(subset=["date", "value", "r"]).copy()
        if x.empty:
            continue
        rows.append(x[["date", "ticker", "sector", "value", "r"]])
        universe_rows.append(
            {
                "ticker": ticker,
                "sector": "energy",
                "source_file": str(sorted(set(part["source_file"].astype(str).tolist()))[0]),
                "n_rows": int(x.shape[0]),
                "start": str(x["date"].iloc[0]),
                "end": str(x["date"].iloc[-1]),
            }
        )

    # Aggregate total system load across subsystems as an extra structural node.
    agg = (
        df.groupby("din_instante", as_index=False)["val_cargaenergiamwmed"]
        .sum()
        .rename(columns={"val_cargaenergiamwmed": "value"})
        .sort_values("din_instante")
    )
    if not agg.empty:
        agg["ticker"] = "ONS_BR"
        agg["sector"] = "energy"
        agg["date"] = agg["din_instante"].dt.date.astype(str)
        agg["r"] = _returns_from_value(agg["value"])
        agg = agg.dropna(subset=["date", "value", "r"]).copy()
        if not agg.empty:
            rows.append(agg[["date", "ticker", "sector", "value", "r"]])
            universe_rows.append(
                {
                    "ticker": "ONS_BR",
                    "sector": "energy",
                    "source_file": "aggregate_subsystems",
                    "n_rows": int(agg.shape[0]),
                    "start": str(agg["date"].iloc[0]),
                    "end": str(agg["date"].iloc[-1]),
                }
            )

    panel = pd.concat(rows, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True) if rows else pd.DataFrame()
    universe = pd.DataFrame(universe_rows).sort_values(["sector", "ticker"]).reset_index(drop=True) if universe_rows else pd.DataFrame()
    return panel, universe


def main() -> None:
    ap = argparse.ArgumentParser(description="Build local energy pack (panel_long_sector + universe_fixed) from ONS CSV snapshots.")
    ap.add_argument("--raw-dir", type=str, default="data/raw/ONS/ons_carga_diaria")
    ap.add_argument("--raw-dir-alt", type=str, default="")
    ap.add_argument("--results-dir", type=str, default="results/energy_download")
    ap.add_argument("--business-days-only", type=int, default=0)
    ap.add_argument("--start", type=str, default="2018-01-01")
    ap.add_argument("--end", type=str, default="")
    ap.add_argument("--min-rows", type=int, default=300)
    ap.add_argument("--write-canonical-raw", type=int, default=1, help="Write normalized per-ticker energy CSVs into data/raw/energy/ons")
    args = ap.parse_args()

    raw_dir = ROOT / str(args.raw_dir)
    raw_dir_alt = ROOT / str(args.raw_dir_alt) if str(args.raw_dir_alt).strip() else None
    results_dir = ROOT / str(args.results_dir)
    paths = sorted(raw_dir.glob("*.csv"))
    if raw_dir_alt is not None:
        paths += sorted(raw_dir_alt.glob("*.csv"))
    paths = sorted(set(paths))
    if not paths:
        alt_txt = f" or {raw_dir_alt}" if raw_dir_alt is not None else ""
        raise SystemExit(f"no energy raw csv files found in: {raw_dir}{alt_txt}")

    df_raw = _load_ons_rows(paths)
    if df_raw.empty:
        raise SystemExit("unable to parse ONS data from raw CSV files")

    panel, universe = _build_panel(df_raw)
    if panel.empty or universe.empty:
        raise SystemExit("failed to build non-empty energy panel/universe")

    if str(args.start).strip():
        panel = panel[panel["date"] >= str(args.start).strip()].copy()
    if str(args.end).strip():
        panel = panel[panel["date"] <= str(args.end).strip()].copy()
    if bool(int(args.business_days_only)):
        d = pd.to_datetime(panel["date"], errors="coerce")
        panel = panel[d.dt.dayofweek < 5].copy()
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Recompute universe after filtering.
    keep = panel.groupby("ticker").size().rename("n_rows").reset_index()
    universe = universe.merge(keep, on="ticker", how="inner", suffixes=("", "_new"))
    if "n_rows_new" in universe.columns:
        universe["n_rows"] = pd.to_numeric(universe["n_rows_new"], errors="coerce").fillna(universe["n_rows"]).astype(int)
        universe = universe.drop(columns=["n_rows_new"])
    date_span = panel.groupby("ticker")["date"].agg(["min", "max"]).reset_index()
    universe = universe.merge(date_span, on="ticker", how="left")
    universe["start"] = universe["min"]
    universe["end"] = universe["max"]
    universe = universe.drop(columns=["min", "max"])

    # Enforce min rows threshold.
    min_rows = int(max(1, args.min_rows))
    valid_tickers = set(universe[universe["n_rows"] >= min_rows]["ticker"].astype(str).tolist())
    panel = panel[panel["ticker"].isin(valid_tickers)].copy()
    universe = universe[universe["ticker"].isin(valid_tickers)].copy().sort_values(["sector", "ticker"]).reset_index(drop=True)
    if panel.empty or universe.empty:
        raise SystemExit("no tickers left after min_rows filter")

    run_id = f"local_pack_{_run_id()}"
    outdir = results_dir / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    panel_out = panel[["date", "ticker", "sector", "r"]].copy()
    panel_out.to_csv(outdir / "panel_long_sector.csv", index=False)
    panel.to_csv(outdir / "panel_long_energy.csv", index=False)
    universe.to_csv(outdir / "universe_fixed.csv", index=False)

    canonical_written: list[str] = []
    if bool(int(args.write_canonical_raw)):
        canonical_dir = ROOT / "data" / "raw" / "energy" / "ons"
        canonical_dir.mkdir(parents=True, exist_ok=True)
        for ticker, part in panel.groupby("ticker"):
            one = part[["date", "value"]].copy().sort_values("date")
            out_path = canonical_dir / f"{str(ticker).upper()}.csv"
            one.to_csv(out_path, index=False)
            canonical_written.append(str(out_path))

    meta = {
        "run_id": run_id,
        "outdir": str(outdir),
        "raw_files_used": [str(p) for p in paths],
        "business_days_only": bool(int(args.business_days_only)),
        "start": str(args.start).strip() or None,
        "end": str(args.end).strip() or None,
        "min_rows": min_rows,
        "assets_ok": int(universe.shape[0]),
        "panel_rows": int(panel.shape[0]),
        "period_start": str(panel["date"].min()),
        "period_end": str(panel["date"].max()),
        "canonical_raw_written": canonical_written,
    }
    (outdir / "build_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
