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


def _collect_csvs(path: Path | None) -> list[Path]:
    if path is None:
        return []
    if not path.exists():
        return []
    return sorted([p for p in path.glob("*.csv") if p.is_file()])


def _to_num(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    has_dot = s.str.contains(".", regex=False, na=False)
    has_comma = s.str.contains(",", regex=False, na=False)

    # Formato pt-BR: 1.234,56 -> remove milhar e converte decimal.
    mask_both = has_dot & has_comma
    s = s.mask(mask_both, s.str.replace(".", "", regex=False))

    # Formato com virgula decimal: 12,34 -> 12.34
    mask_comma = has_comma
    s = s.mask(mask_comma, s.str.replace(",", ".", regex=False))

    return pd.to_numeric(s, errors="coerce")


def _pick_column(col_map: dict[str, str], exact: list[str], contains: list[str] | None = None) -> str | None:
    for key in exact:
        if key in col_map:
            return col_map[key]
    if contains:
        for token in contains:
            for k, v in col_map.items():
                if token in k:
                    return v
    return None


def _normalize_rows(
    raw: pd.DataFrame,
    path: Path,
    *,
    id_candidates: list[str],
    date_candidates: list[str],
    value_exact: list[str],
    value_contains: list[str] | None = None,
    id_name_candidates: list[str] | None = None,
) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["id_subsistema", "nom_subsistema", "din_instante", "value", "source_file"])

    col_map = {str(c).strip().lower(): str(c) for c in raw.columns}
    id_col = _pick_column(col_map, id_candidates)
    date_col = _pick_column(col_map, date_candidates)
    val_col = _pick_column(col_map, value_exact, value_contains)
    name_col = _pick_column(col_map, id_name_candidates or []) if id_name_candidates else None
    if id_col is None or date_col is None or val_col is None:
        return pd.DataFrame(columns=["id_subsistema", "nom_subsistema", "din_instante", "value", "source_file"])

    cols = [id_col, date_col, val_col]
    if name_col:
        cols.append(name_col)
    df = raw[cols].copy()
    rename = {
        id_col: "id_subsistema",
        date_col: "din_instante",
        val_col: "value",
    }
    if name_col:
        rename[name_col] = "nom_subsistema"
    df = df.rename(columns=rename)
    if "nom_subsistema" not in df.columns:
        df["nom_subsistema"] = df["id_subsistema"].astype(str)
    df["din_instante"] = pd.to_datetime(df["din_instante"], errors="coerce")
    df["value"] = _to_num(df["value"])
    df = df.dropna(subset=["id_subsistema", "din_instante", "value"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["id_subsistema", "nom_subsistema", "din_instante", "value", "source_file"])
    df["id_subsistema"] = df["id_subsistema"].astype(str).str.strip().str.upper()
    df["nom_subsistema"] = df["nom_subsistema"].astype(str).str.strip()
    df["source_file"] = str(path)
    return df[["id_subsistema", "nom_subsistema", "din_instante", "value", "source_file"]]


def _load_carga_rows(paths: list[Path]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in paths:
        try:
            raw = _read_csv_auto(path)
        except Exception:
            continue
        part = _normalize_rows(
            raw,
            path,
            id_candidates=["id_subsistema", "nom_subsistema"],
            date_candidates=["din_instante", "dat_referencia", "din_referencia"],
            value_exact=["val_cargaenergiamwmed"],
            id_name_candidates=["nom_subsistema"],
        )
        if part.empty:
            continue
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["id_subsistema", "nom_subsistema", "din_instante", "value", "source_file"])
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["din_instante", "id_subsistema", "source_file"]).drop_duplicates(
        subset=["din_instante", "id_subsistema"],
        keep="last",
    )
    return out.reset_index(drop=True)


def _load_ear_rows(paths: list[Path]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in paths:
        try:
            raw = _read_csv_auto(path)
        except Exception:
            continue
        part = _normalize_rows(
            raw,
            path,
            id_candidates=["id_subsistema", "nom_subsistema"],
            date_candidates=["din_instante", "dat_referencia", "din_referencia"],
            value_exact=[
                "val_ear_verificada_percentual",
                "val_earverificada_percentual",
                "val_ear_verificada_mwmes",
                "val_earverificada_mwmes",
                "val_earmax_percentual",
                "val_earmax_mwmes",
            ],
            value_contains=["ear", "reservatorio"],
            id_name_candidates=["nom_subsistema"],
        )
        if part.empty:
            continue
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["id_subsistema", "nom_subsistema", "din_instante", "value", "source_file"])
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["din_instante", "id_subsistema", "source_file"]).drop_duplicates(
        subset=["din_instante", "id_subsistema"],
        keep="last",
    )
    return out.reset_index(drop=True)


def _load_cmo_rows(paths: list[Path]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in paths:
        try:
            raw = _read_csv_auto(path)
        except Exception:
            continue
        part = _normalize_rows(
            raw,
            path,
            id_candidates=["id_subsistema", "nom_subsistema", "nom_subsitema"],
            date_candidates=["din_instante", "dat_referencia", "din_referencia"],
            value_exact=["val_cmo", "val_cmomed", "val_cmomwmed"],
            value_contains=["cmo"],
            id_name_candidates=["nom_subsistema", "nom_subsitema"],
        )
        if part.empty:
            continue
        # Dataset semanal costuma vir com multiplos registros no mesmo dia/subsistema.
        part = (
            part.groupby(["id_subsistema", "nom_subsistema", "din_instante"], as_index=False)
            .agg({"value": "mean", "source_file": "last"})
            .reset_index(drop=True)
        )
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["id_subsistema", "nom_subsistema", "din_instante", "value", "source_file"])
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["din_instante", "id_subsistema", "source_file"]).drop_duplicates(
        subset=["din_instante", "id_subsistema"],
        keep="last",
    )
    return out.reset_index(drop=True)


def _returns_from_value(v: pd.Series) -> pd.Series:
    s = pd.to_numeric(v, errors="coerce")
    return np.log(s.clip(lower=1e-9)).diff()


def _append_block(
    *,
    rows: list[pd.DataFrame],
    universe_rows: list[dict[str, Any]],
    source_df: pd.DataFrame,
    prefix: str,
    sector: str,
    source_label: str,
) -> None:
    if source_df.empty:
        return
    for subsystem, part in source_df.groupby("id_subsistema", dropna=True):
        x = part.sort_values("din_instante").copy()
        if x.empty:
            continue
        ticker = f"{prefix}_{str(subsystem).strip().upper()}"
        x["ticker"] = ticker
        x["sector"] = sector
        x["date"] = x["din_instante"].dt.date.astype(str)
        x["r"] = _returns_from_value(x["value"])
        x = x.dropna(subset=["date", "value", "r"]).copy()
        if x.empty:
            continue
        rows.append(x[["date", "ticker", "sector", "value", "r"]])
        universe_rows.append(
            {
                "ticker": ticker,
                "sector": sector,
                "source_file": source_label,
                "n_rows": int(x.shape[0]),
                "start": str(x["date"].iloc[0]),
                "end": str(x["date"].iloc[-1]),
            }
        )


def _build_panel(df_carga: pd.DataFrame, df_ear: pd.DataFrame, df_cmo: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    universe_rows: list[dict[str, Any]] = []

    _append_block(
        rows=rows,
        universe_rows=universe_rows,
        source_df=df_carga,
        prefix="ONS",
        sector="energy_load",
        source_label="ons_carga_diaria",
    )
    _append_block(
        rows=rows,
        universe_rows=universe_rows,
        source_df=df_ear,
        prefix="EAR",
        sector="energy_storage",
        source_label="ons_ear_subsistema_di",
    )
    _append_block(
        rows=rows,
        universe_rows=universe_rows,
        source_df=df_cmo,
        prefix="CMO",
        sector="energy_price",
        source_label="ons_cmo_semanal",
    )

    # Node global de carga agregada para ancorar acoplamento sistêmico.
    if not df_carga.empty:
        agg = df_carga.groupby("din_instante", as_index=False)["value"].sum().sort_values("din_instante")
        if not agg.empty:
            agg["ticker"] = "ONS_BR"
            agg["sector"] = "energy_load"
            agg["date"] = agg["din_instante"].dt.date.astype(str)
            agg["r"] = _returns_from_value(agg["value"])
            agg = agg.dropna(subset=["date", "value", "r"]).copy()
            if not agg.empty:
                rows.append(agg[["date", "ticker", "sector", "value", "r"]])
                universe_rows.append(
                    {
                        "ticker": "ONS_BR",
                        "sector": "energy_load",
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
    ap.add_argument("--ear-dir", type=str, default="data/raw/ONS/ons_ear_subsistema_di")
    ap.add_argument("--cmo-dir", type=str, default="data/raw/ONS/ons_cmo_semanal")
    ap.add_argument("--results-dir", type=str, default="results/energy_download")
    ap.add_argument("--business-days-only", type=int, default=0)
    ap.add_argument("--start", type=str, default="2018-01-01")
    ap.add_argument("--end", type=str, default="")
    ap.add_argument("--min-rows", type=int, default=300)
    ap.add_argument("--write-canonical-raw", type=int, default=1, help="Write normalized per-ticker energy CSVs into data/raw/energy/ons")
    args = ap.parse_args()

    raw_dir = ROOT / str(args.raw_dir)
    raw_dir_alt = ROOT / str(args.raw_dir_alt) if str(args.raw_dir_alt).strip() else None
    ear_dir = ROOT / str(args.ear_dir) if str(args.ear_dir).strip() else None
    cmo_dir = ROOT / str(args.cmo_dir) if str(args.cmo_dir).strip() else None
    results_dir = ROOT / str(args.results_dir)

    carga_paths = _collect_csvs(raw_dir) + _collect_csvs(raw_dir_alt)
    carga_paths = sorted(set(carga_paths))
    ear_paths = _collect_csvs(ear_dir)
    cmo_paths = _collect_csvs(cmo_dir)
    if not carga_paths and not ear_paths and not cmo_paths:
        raise SystemExit(
            "no energy raw csv files found in configured dirs "
            f"(carga={raw_dir}, ear={ear_dir}, cmo={cmo_dir})"
        )

    df_carga = _load_carga_rows(carga_paths)
    df_ear = _load_ear_rows(ear_paths)
    df_cmo = _load_cmo_rows(cmo_paths)
    if df_carga.empty and df_ear.empty and df_cmo.empty:
        raise SystemExit("unable to parse ONS data from raw CSV files")

    panel, universe = _build_panel(df_carga=df_carga, df_ear=df_ear, df_cmo=df_cmo)
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
        "raw_files_used": [str(p) for p in sorted(set(carga_paths + ear_paths + cmo_paths))],
        "counts_by_block": {
            "carga_rows": int(df_carga.shape[0]),
            "ear_rows": int(df_ear.shape[0]),
            "cmo_rows": int(df_cmo.shape[0]),
        },
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
