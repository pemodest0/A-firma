#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "agro_brasil_monthly_series.json"


MONTH_MAP = {
    "jan": 1,
    "janeiro": 1,
    "fev": 2,
    "fevereiro": 2,
    "mar": 3,
    "marco": 3,
    "abr": 4,
    "abril": 4,
    "mai": 5,
    "maio": 5,
    "jun": 6,
    "junho": 6,
    "jul": 7,
    "julho": 7,
    "ago": 8,
    "agosto": 8,
    "set": 9,
    "setembro": 9,
    "out": 10,
    "outubro": 10,
    "nov": 11,
    "novembro": 11,
    "dez": 12,
    "dezembro": 12,
}


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(text).strip().lower()).strip("_")


def _to_float(v: object) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    if "," in s and "." in s and s.find(".") < s.find(","):
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    try:
        x = float(s)
    except ValueError:
        return None
    return x if np.isfinite(x) else None


def _read_csv_auto(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.shape[1] == 1 and ";" in str(df.columns[0]):
        df = pd.read_csv(path, sep=";")
    return df


def _read_tabular(path: Path) -> list[pd.DataFrame]:
    suf = path.suffix.lower()
    if suf in {".csv", ".txt"}:
        return [_read_csv_auto(path)]
    if suf in {".xlsx", ".xls"}:
        try:
            xls = pd.ExcelFile(path)
        except Exception:
            return []
        out: list[pd.DataFrame] = []
        for sheet in xls.sheet_names:
            try:
                out.append(xls.parse(sheet))
            except Exception:
                continue
        return out
    return []


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [_slug(c) for c in out.columns]
    return out


def _monthly_transform(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    # asinh-diff handles both positive-only and signed macro levels.
    r = np.arcsinh(x).diff()
    return pd.Series(r, index=series.index, dtype=float)


def _build_series_rows(
    *,
    date: pd.Series,
    value: pd.Series,
    ticker: str,
    sector_internal: str,
    source: str,
    license_name: str,
    aggregation: str,
    name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    d = pd.DataFrame({"date": pd.to_datetime(date, errors="coerce"), "value": pd.to_numeric(value, errors="coerce")})
    d = d.dropna(subset=["date", "value"]).sort_values("date").drop_duplicates("date", keep="last")
    if d.empty:
        return pd.DataFrame(), {}
    d["date"] = d["date"].dt.to_period("M").dt.to_timestamp(how="end").dt.normalize()
    d = d.groupby("date", as_index=False)["value"].last().sort_values("date")
    d["r"] = _monthly_transform(d["value"])
    d = d.dropna(subset=["r"]).copy()
    if d.empty:
        return pd.DataFrame(), {}
    d["date"] = d["date"].dt.strftime("%Y-%m-%d")
    d["ticker"] = str(ticker)
    d["sector"] = "agro_brasil"
    d["source"] = str(source)
    d["aggregation"] = str(aggregation)
    meta = {
        "asset_id": str(ticker),
        "ticker": str(ticker),
        "name": str(name),
        "sector_gics": "agro_brasil",
        "sector_internal": str(sector_internal),
        "liquidity_proxy": float(d["date"].nunique()),
        "source": str(source),
        "license": str(license_name),
        "aggregation": str(aggregation),
        "rows": int(d.shape[0]),
        "start": str(d["date"].iloc[0]),
        "end": str(d["date"].iloc[-1]),
    }
    return d[["date", "ticker", "sector", "value", "r", "source", "aggregation"]].copy(), meta


def _load_bcb_series(raw_dir: Path, cfg: dict[str, Any]) -> tuple[list[pd.DataFrame], list[dict[str, Any]], list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    metadata: list[dict[str, Any]] = []
    status: list[dict[str, Any]] = []

    for item in cfg.get("bcb_sgs", []):
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker", item.get("asset_id", ""))).strip()
        if not ticker:
            continue
        p = raw_dir / f"{ticker}.csv"
        rec: dict[str, Any] = {"ticker": ticker, "source_file": str(p)}
        if not p.exists():
            rec["status"] = "missing"
            status.append(rec)
            continue
        try:
            df = pd.read_csv(p)
            if "date" not in df.columns or "value" not in df.columns:
                rec["status"] = "invalid_schema"
                status.append(rec)
                continue
            one, meta = _build_series_rows(
                date=df["date"],
                value=df["value"],
                ticker=ticker,
                sector_internal=str(item.get("sector_internal", "macro")),
                source=str(item.get("source", "BCB/SGS")),
                license_name=str(item.get("license", "Dados abertos BCB")),
                aggregation=str(item.get("aggregation", "last")),
                name=str(item.get("name", ticker)),
            )
            if one.empty:
                rec["status"] = "empty_after_transform"
                status.append(rec)
                continue
            frames.append(one)
            metadata.append(meta)
            rec["status"] = "ok"
            rec["rows"] = int(one.shape[0])
            rec["start"] = str(one["date"].iloc[0])
            rec["end"] = str(one["date"].iloc[-1])
            status.append(rec)
        except Exception as exc:
            rec["status"] = "fail"
            rec["reason"] = str(exc)
            status.append(rec)
    return frames, metadata, status


def _find_column(cols: list[str], patterns: list[str]) -> str:
    for p in patterns:
        rx = re.compile(p)
        for c in cols:
            if rx.search(c):
                return c
    return ""


def _parse_comex_dir(dir_path: Path, cfg: dict[str, Any]) -> tuple[list[pd.DataFrame], list[dict[str, Any]], list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    metadata: list[dict[str, Any]] = []
    status: list[dict[str, Any]] = []
    if not dir_path.exists():
        return frames, metadata, status

    files = sorted([p for p in dir_path.iterdir() if p.is_file()])
    all_rows: list[pd.DataFrame] = []
    for p in files:
        rec: dict[str, Any] = {"file": str(p)}
        try:
            raw = _read_csv_auto(p)
            if raw.empty:
                rec["status"] = "empty"
                status.append(rec)
                continue
            d = _normalize_columns(raw)
            cols = list(d.columns)
            year_col = _find_column(cols, [r"^ano$", r"^year$"])
            month_col = _find_column(cols, [r"^mes$", r"^month$"])
            date_col = _find_column(cols, [r"^data$", r"^date$", r"periodo", r"ref"])
            flow_col = _find_column(cols, [r"flow", r"fluxo", r"tipo", r"operacao"])
            value_col = _find_column(cols, [r"fob", r"valor", r"usd", r"us\$"])
            qty_col = _find_column(cols, [r"kg", r"quant"])

            if date_col:
                d["date"] = pd.to_datetime(d[date_col], errors="coerce")
            elif year_col and month_col:
                yy = pd.to_numeric(d[year_col], errors="coerce")
                mm = pd.to_numeric(d[month_col], errors="coerce")
                d["date"] = pd.to_datetime(
                    {"year": yy.astype("Int64"), "month": mm.astype("Int64"), "day": 1},
                    errors="coerce",
                )
            else:
                rec["status"] = "no_date_columns"
                status.append(rec)
                continue

            d = d.dropna(subset=["date"]).copy()
            if d.empty:
                rec["status"] = "no_valid_dates"
                status.append(rec)
                continue
            d["flow"] = d[flow_col].astype(str).str.lower() if flow_col else "total"
            d["value_fob"] = pd.to_numeric(d[value_col].map(_to_float), errors="coerce") if value_col else np.nan
            d["value_kg"] = pd.to_numeric(d[qty_col].map(_to_float), errors="coerce") if qty_col else np.nan
            d["month"] = d["date"].dt.to_period("M")
            g = d.groupby(["month", "flow"], as_index=False).agg({"value_fob": "sum", "value_kg": "sum"})
            g["date"] = g["month"].dt.to_timestamp(how="end").dt.normalize()
            all_rows.append(g[["date", "flow", "value_fob", "value_kg"]].copy())
            rec["status"] = "ok"
            rec["rows"] = int(g.shape[0])
            status.append(rec)
        except Exception as exc:
            rec["status"] = "fail"
            rec["reason"] = str(exc)
            status.append(rec)

    if not all_rows:
        return frames, metadata, status

    all_df = pd.concat(all_rows, ignore_index=True)
    all_df["flow"] = all_df["flow"].astype(str).str.lower()
    exp_mask = all_df["flow"].str.contains("exp")
    imp_mask = all_df["flow"].str.contains("imp")
    monthly = pd.DataFrame({"date": sorted(all_df["date"].dropna().unique().tolist())})
    exp_fob = all_df[exp_mask].groupby("date")["value_fob"].sum().rename("export_fob")
    imp_fob = all_df[imp_mask].groupby("date")["value_fob"].sum().rename("import_fob")
    exp_kg = all_df[exp_mask].groupby("date")["value_kg"].sum().rename("export_kg")
    imp_kg = all_df[imp_mask].groupby("date")["value_kg"].sum().rename("import_kg")
    monthly = monthly.merge(exp_fob, on="date", how="left").merge(imp_fob, on="date", how="left")
    monthly = monthly.merge(exp_kg, on="date", how="left").merge(imp_kg, on="date", how="left")
    monthly["balance_fob"] = monthly["export_fob"] - monthly["import_fob"]

    def _append_metric(col: str, ticker: str, name: str) -> None:
        if col not in monthly.columns:
            return
        if monthly[col].notna().sum() < 12:
            return
        one, meta = _build_series_rows(
            date=monthly["date"],
            value=monthly[col],
            ticker=ticker,
            sector_internal="fluxo_externo",
            source=str(cfg.get("source", "Comex Stat (MDIC)")),
            license_name=str(cfg.get("license", "Dados abertos federais")),
            aggregation="sum",
            name=name,
        )
        if one.empty:
            return
        frames.append(one)
        metadata.append(meta)

    _append_metric("export_fob", "COMEX_EXPORT_FOB", "Comex exportacao FOB mensal")
    _append_metric("import_fob", "COMEX_IMPORT_FOB", "Comex importacao FOB mensal")
    _append_metric("balance_fob", "COMEX_BALANCE_FOB", "Comex saldo FOB mensal")
    _append_metric("export_kg", "COMEX_EXPORT_KG", "Comex exportacao KG mensal")
    _append_metric("import_kg", "COMEX_IMPORT_KG", "Comex importacao KG mensal")
    return frames, metadata, status


def _extract_from_conab_frame(df_raw: pd.DataFrame, source_name: str) -> list[tuple[str, pd.Series, pd.Series]]:
    out: list[tuple[str, pd.Series, pd.Series]] = []
    if df_raw is None or df_raw.empty:
        return out
    d = _normalize_columns(df_raw)
    cols = list(d.columns)
    year_col = _find_column(cols, [r"^ano$", r"^year$"])
    month_col = _find_column(cols, [r"^mes$", r"^month$"])
    date_col = _find_column(cols, [r"^data$", r"^date$", r"periodo", r"ref"])

    month_name_cols: list[tuple[str, int]] = []
    for c in cols:
        c_norm = c.lower()
        if c_norm in MONTH_MAP:
            month_name_cols.append((c, MONTH_MAP[c_norm]))

    if date_col:
        d["date"] = pd.to_datetime(d[date_col], errors="coerce")
    elif year_col and month_col:
        yy = pd.to_numeric(d[year_col], errors="coerce")
        mm = pd.to_numeric(d[month_col], errors="coerce")
        d["date"] = pd.to_datetime(
            {"year": yy.astype("Int64"), "month": mm.astype("Int64"), "day": 1},
            errors="coerce",
        )
    elif year_col and month_name_cols:
        base = d[[year_col] + [x[0] for x in month_name_cols]].copy()
        base[year_col] = pd.to_numeric(base[year_col], errors="coerce")
        m = base.melt(id_vars=[year_col], var_name="month_col", value_name="value")
        m["month"] = m["month_col"].map({x[0]: x[1] for x in month_name_cols})
        m["date"] = pd.to_datetime(
            {"year": m[year_col].astype("Int64"), "month": m["month"].astype("Int64"), "day": 1},
            errors="coerce",
        )
        out.append((f"{source_name}_principal", m["date"], pd.to_numeric(m["value"].map(_to_float), errors="coerce")))
        return out
    elif year_col:
        yy = pd.to_numeric(d[year_col], errors="coerce")
        # annual points become Dec reference of the year.
        d["date"] = pd.to_datetime({"year": yy.astype("Int64"), "month": 12, "day": 1}, errors="coerce")
    else:
        return out

    d = d.dropna(subset=["date"]).copy()
    if d.empty:
        return out
    id_cols = {c for c in [year_col, month_col, date_col, "date"] if c}
    numeric_cols: list[str] = []
    for c in cols:
        if c in id_cols:
            continue
        vc = pd.to_numeric(d[c].map(_to_float), errors="coerce")
        if vc.notna().sum() >= 12:
            numeric_cols.append(c)
    for c in numeric_cols:
        out.append((f"{source_name}_{_slug(c)}", d["date"], pd.to_numeric(d[c].map(_to_float), errors="coerce")))
    return out


def _parse_conab_dir(dir_path: Path, cfg: dict[str, Any]) -> tuple[list[pd.DataFrame], list[dict[str, Any]], list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    metadata: list[dict[str, Any]] = []
    status: list[dict[str, Any]] = []
    if not dir_path.exists():
        return frames, metadata, status

    files = sorted([p for p in dir_path.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".txt", ".xlsx", ".xls"}])
    for p in files:
        rec: dict[str, Any] = {"file": str(p)}
        try:
            tables = _read_tabular(p)
            if not tables:
                rec["status"] = "unsupported_or_empty"
                status.append(rec)
                continue
            n_series = 0
            src = _slug(p.stem) or "conab"
            for idx, tab in enumerate(tables, start=1):
                extracted = _extract_from_conab_frame(tab, source_name=f"{src}_s{idx}")
                for sid, date, value in extracted:
                    ticker = f"CONAB_{_slug(sid)}".upper()
                    one, meta = _build_series_rows(
                        date=date,
                        value=value,
                        ticker=ticker,
                        sector_internal="safra_estoque_logistica",
                        source=str(cfg.get("source", "CONAB")),
                        license_name=str(cfg.get("license", "Dados governamentais")),
                        aggregation="monthly_last",
                        name=f"CONAB {sid}",
                    )
                    if one.empty:
                        continue
                    frames.append(one)
                    metadata.append(meta)
                    n_series += 1
            rec["status"] = "ok" if n_series > 0 else "no_series"
            rec["series"] = int(n_series)
            status.append(rec)
        except Exception as exc:
            rec["status"] = "fail"
            rec["reason"] = str(exc)
            status.append(rec)
    return frames, metadata, status


def _parse_local_finance_tickers(
    prices_dir: Path,
    cfg: dict[str, Any],
) -> tuple[list[pd.DataFrame], list[dict[str, Any]], list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    metadata: list[dict[str, Any]] = []
    status: list[dict[str, Any]] = []
    if not prices_dir.exists():
        return frames, metadata, status

    items = cfg.get("local_finance_tickers", [])
    if not isinstance(items, list):
        return frames, metadata, status

    for item in items:
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker", "")).strip()
        asset_id = str(item.get("asset_id", ticker)).strip() or ticker
        rec: dict[str, Any] = {"ticker": ticker, "asset_id": asset_id}
        if not ticker:
            rec["status"] = "invalid_ticker"
            status.append(rec)
            continue
        p = prices_dir / f"{ticker}.csv"
        rec["source_file"] = str(p)
        if not p.exists():
            rec["status"] = "missing"
            status.append(rec)
            continue
        try:
            d = pd.read_csv(p)
            if "date" not in d.columns:
                rec["status"] = "missing_date"
                status.append(rec)
                continue
            d["date"] = pd.to_datetime(d["date"], errors="coerce")
            if "r" in d.columns:
                d["r"] = pd.to_numeric(d["r"], errors="coerce")
            elif "price" in d.columns:
                price = pd.to_numeric(d["price"], errors="coerce")
                d["r"] = np.log(price.clip(lower=1e-9)).diff()
            else:
                rec["status"] = "missing_return_or_price"
                status.append(rec)
                continue
            if "price" in d.columns:
                d["price"] = pd.to_numeric(d["price"], errors="coerce")
            else:
                d["price"] = np.nan
            d = d.dropna(subset=["date", "r"]).sort_values("date").copy()
            if d.empty:
                rec["status"] = "empty_after_parse"
                status.append(rec)
                continue
            d["month"] = d["date"].dt.to_period("M")
            m = d.groupby("month", as_index=False).agg({"r": "sum", "price": "last"})
            m["date"] = m["month"].dt.to_timestamp(how="end").dt.normalize().dt.strftime("%Y-%m-%d")
            m["ticker"] = asset_id
            m["sector"] = "agro_brasil"
            m["value"] = m["price"]
            m["source"] = str(item.get("source", "YFinance local cache"))
            m["aggregation"] = "sum_log_returns_monthly"
            m = m.dropna(subset=["r"]).copy()
            if m.empty:
                rec["status"] = "empty_monthly"
                status.append(rec)
                continue
            one = m[["date", "ticker", "sector", "value", "r", "source", "aggregation"]].copy().sort_values("date")
            frames.append(one)
            meta = {
                "asset_id": str(asset_id),
                "ticker": str(asset_id),
                "name": str(item.get("name", ticker)),
                "sector_gics": str(item.get("sector_gics", "agro_brasil")),
                "sector_internal": str(item.get("sector_internal", "mercado")),
                "liquidity_proxy": float(one["date"].nunique()),
                "source": str(item.get("source", "YFinance local cache")),
                "license": str(item.get("license", "Yahoo Finance - uso local de pesquisa")),
                "aggregation": "sum_log_returns_monthly",
                "rows": int(one.shape[0]),
                "start": str(one["date"].iloc[0]),
                "end": str(one["date"].iloc[-1]),
            }
            metadata.append(meta)
            rec["status"] = "ok"
            rec["rows"] = int(one.shape[0])
            rec["start"] = str(one["date"].iloc[0])
            rec["end"] = str(one["date"].iloc[-1])
            status.append(rec)
        except Exception as exc:
            rec["status"] = "fail"
            rec["reason"] = str(exc)
            status.append(rec)
    return frames, metadata, status


def _dedupe_metadata(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "asset_id",
                "ticker",
                "name",
                "sector_gics",
                "sector_internal",
                "liquidity_proxy",
                "source",
                "license",
                "aggregation",
            ]
        )
    md = pd.DataFrame(rows)
    md["asset_id"] = md["asset_id"].astype(str)
    md["ticker"] = md["ticker"].astype(str)
    md["name"] = md["name"].astype(str)
    md["sector_gics"] = md.get("sector_gics", "agro_brasil").fillna("agro_brasil").astype(str)
    md["sector_internal"] = md.get("sector_internal", "agro").fillna("agro").astype(str)
    md["liquidity_proxy"] = pd.to_numeric(md.get("liquidity_proxy"), errors="coerce")
    md["source"] = md.get("source", "unknown").fillna("unknown").astype(str)
    md["license"] = md.get("license", "unknown").fillna("unknown").astype(str)
    md["aggregation"] = md.get("aggregation", "last").fillna("last").astype(str)
    md = md.sort_values(["asset_id", "liquidity_proxy"], ascending=[True, False]).drop_duplicates("asset_id", keep="first")
    return md.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build local Agro Brasil monthly pack (panel_long + metadata).")
    ap.add_argument("--config-path", type=str, default=str(DEFAULT_CONFIG))
    ap.add_argument("--raw-dir", type=str, default="data/raw/agro")
    ap.add_argument("--download-dir", type=str, default="data/download/agro")
    ap.add_argument("--finance-prices-dir", type=str, default="data/raw/finance/yfinance_daily")
    ap.add_argument("--results-dir", type=str, default="results/agro_br")
    ap.add_argument("--min-rows", type=int, default=24)
    ap.add_argument("--write-canonical", type=int, default=1)
    args = ap.parse_args()

    cfg_path = Path(args.config_path)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / str(args.config_path)
    if not cfg_path.exists():
        raise SystemExit(f"missing config file: {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    raw_dir = Path(args.raw_dir)
    if not raw_dir.is_absolute():
        raw_dir = ROOT / str(args.raw_dir)
    download_dir = Path(args.download_dir)
    if not download_dir.is_absolute():
        download_dir = ROOT / str(args.download_dir)
    finance_prices_dir = Path(args.finance_prices_dir)
    if not finance_prices_dir.is_absolute():
        finance_prices_dir = ROOT / str(args.finance_prices_dir)
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = ROOT / str(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"local_pack_{_run_id()}"
    outdir = results_dir / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []
    status: dict[str, Any] = {"bcb": [], "comex": [], "conab": []}

    bcb_frames, bcb_md, bcb_status = _load_bcb_series(raw_dir / "bcb", cfg)
    frames.extend(bcb_frames)
    metadata_rows.extend(bcb_md)
    status["bcb"] = bcb_status

    comex_frames, comex_md, comex_status = _parse_comex_dir(download_dir / "comex", cfg.get("comex", {}))
    frames.extend(comex_frames)
    metadata_rows.extend(comex_md)
    status["comex"] = comex_status

    conab_frames, conab_md, conab_status = _parse_conab_dir(download_dir / "conab", cfg.get("conab", {}))
    frames.extend(conab_frames)
    metadata_rows.extend(conab_md)
    status["conab"] = conab_status

    fin_frames, fin_md, fin_status = _parse_local_finance_tickers(finance_prices_dir, cfg)
    frames.extend(fin_frames)
    metadata_rows.extend(fin_md)
    status["local_finance"] = fin_status

    if not frames:
        raise SystemExit("no series generated for agro pack")

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.dropna(subset=["date", "ticker", "r"]).copy()
    panel["date"] = panel["date"].dt.strftime("%Y-%m-%d")
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

    md = _dedupe_metadata(metadata_rows)
    if md.empty:
        raise SystemExit("metadata empty after parsing")

    # Keep only assets with sufficient monthly history.
    min_rows = int(max(12, args.min_rows))
    rows_by_ticker = panel.groupby("ticker").size().rename("n_rows").reset_index()
    valid = set(rows_by_ticker[rows_by_ticker["n_rows"] >= min_rows]["ticker"].astype(str).tolist())
    panel = panel[panel["ticker"].isin(valid)].copy().sort_values(["date", "ticker"]).reset_index(drop=True)
    md = md[md["ticker"].isin(valid)].copy().sort_values(["sector_internal", "ticker"]).reset_index(drop=True)
    if panel.empty or md.empty:
        raise SystemExit("no data left after min_rows filter")

    universe = (
        panel.groupby("ticker")
        .agg(
            n_rows=("date", "size"),
            start=("date", "min"),
            end=("date", "max"),
        )
        .reset_index()
    )
    universe = universe.merge(md[["ticker", "sector_gics", "source"]], on="ticker", how="left")
    universe = universe.rename(columns={"sector_gics": "sector", "source": "source_file"})
    universe = universe[["ticker", "sector", "source_file", "n_rows", "start", "end"]].sort_values(["sector", "ticker"]).reset_index(drop=True)

    panel_basic = panel[["date", "ticker", "sector", "r"]].copy()
    panel_basic.to_csv(outdir / "panel_long_sector.csv", index=False)
    panel.to_csv(outdir / "panel_long_agro_br.csv", index=False)
    universe.to_csv(outdir / "universe_fixed.csv", index=False)
    md.to_csv(outdir / "asset_metadata_agro_br.csv", index=False)

    if bool(int(args.write_canonical)):
        canonical = ROOT / "data" / "processed" / "agro" / "br_monthly"
        canonical.mkdir(parents=True, exist_ok=True)
        panel.to_csv(canonical / "panel_long_agro_br.csv", index=False)
        md.to_csv(canonical / "asset_metadata_agro_br.csv", index=False)
        universe.to_csv(canonical / "universe_fixed.csv", index=False)
        (canonical / "latest_local_pack.json").write_text(
            json.dumps(
                {
                    "status": "ok",
                    "run_id": run_id,
                    "outdir": str(outdir),
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    meta = {
        "status": "ok",
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(cfg_path),
        "outdir": str(outdir),
        "panel_rows": int(panel.shape[0]),
        "assets_ok": int(md.shape[0]),
        "period_start": str(panel["date"].min()),
        "period_end": str(panel["date"].max()),
        "min_rows": int(min_rows),
        "sources_status": status,
    }
    (outdir / "build_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
