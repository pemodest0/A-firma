#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "agro_brasil_monthly_series.json"
BCB_PROXY_ASSETS = {
    "BCB_CRED_IMOB_PF_JUROS_TOTAL": "BCB_CRED_IMOB_PF_JUROS_MERCADO",
}


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    s = str(value).strip().replace(".", "").replace(",", ".")
    try:
        v = float(s)
    except ValueError:
        return None
    return v


def _fetch_bcb_series_monthly(
    *,
    series_id: int,
    start_year: int,
    end_year: int,
    aggregation: str,
    timeout_sec: float,
    max_retries: int = 3,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    chunk_sizes = [10, 5, 2, 1]
    year = int(start_year)
    while year <= int(end_year):
        resolved = False
        last_exc: Exception | None = None
        for chunk in chunk_sizes:
            stop_year = min(int(end_year), year + int(chunk) - 1)
            start = datetime(year=year, month=1, day=1)
            stop = datetime(year=stop_year, month=12, day=31)
            query = urlencode(
                {
                    "formato": "json",
                    "dataInicial": start.strftime("%d/%m/%Y"),
                    "dataFinal": stop.strftime("%d/%m/%Y"),
                }
            )
            url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{int(series_id)}/dados?{query}"
            for _ in range(max(1, int(max_retries))):
                try:
                    req = Request(url, headers={"User-Agent": "Assyntrax/1.0"})
                    timeout = float(timeout_sec) * (1.5 if int(chunk) <= 2 else 1.0)
                    with urlopen(req, timeout=timeout) as resp:  # noqa: S310 (official public API)
                        payload = json.loads(resp.read().decode("utf-8"))
                    rows.extend(payload if isinstance(payload, list) else [])
                    resolved = True
                    break
                except Exception as exc:  # pragma: no cover - network/runtime variability
                    last_exc = exc
            if resolved:
                year = stop_year + 1
                break
        if not resolved:
            raise RuntimeError(
                f"BCB fetch failed for series {int(series_id)} around year {int(year)}: {last_exc}"
            )

    if not rows:
        return pd.DataFrame(columns=["date", "value"])

    df = pd.DataFrame(rows)
    if "data" not in df.columns or "valor" not in df.columns:
        return pd.DataFrame(columns=["date", "value"])
    df["date"] = pd.to_datetime(df["data"], format="%d/%m/%Y", errors="coerce")
    df["value"] = df["valor"].map(_to_float)
    df = df.dropna(subset=["date", "value"]).sort_values("date").drop_duplicates("date", keep="last")
    if df.empty:
        return pd.DataFrame(columns=["date", "value"])

    df["month"] = df["date"].dt.to_period("M")
    agg = str(aggregation).strip().lower()
    if agg == "sum":
        m = df.groupby("month", as_index=False)["value"].sum()
    elif agg == "mean":
        m = df.groupby("month", as_index=False)["value"].mean()
    else:
        m = df.groupby("month", as_index=False)["value"].last()
    m["date"] = m["month"].dt.to_timestamp(how="end").dt.normalize().dt.strftime("%Y-%m-%d")
    return m[["date", "value"]].copy().sort_values("date").reset_index(drop=True)


def _monthly_aggregate(df: pd.DataFrame, aggregation: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "value"])
    x = df.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x["value"] = pd.to_numeric(x["value"], errors="coerce")
    x = x.dropna(subset=["date", "value"]).sort_values("date")
    if x.empty:
        return pd.DataFrame(columns=["date", "value"])
    x["month"] = x["date"].dt.to_period("M")
    agg = str(aggregation).strip().lower()
    if agg == "sum":
        m = x.groupby("month", as_index=False)["value"].sum()
    elif agg == "mean":
        m = x.groupby("month", as_index=False)["value"].mean()
    else:
        m = x.groupby("month", as_index=False)["value"].last()
    m["date"] = m["month"].dt.to_timestamp(how="end").dt.normalize().dt.strftime("%Y-%m-%d")
    return m[["date", "value"]].copy().sort_values("date").reset_index(drop=True)


def _load_local_bcb_fallback(
    *,
    asset_id: str,
    series_id: int,
    aggregation: str,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    local_dirs = [
        ROOT / "data" / "raw" / "agro" / "bcb",
        ROOT / "data" / "raw" / "realestate" / "bcb",
    ]
    stripped = asset_id[4:] if asset_id.startswith("BCB_") else asset_id
    candidates = [
        f"{asset_id}.csv",
        f"{stripped}.csv",
        f"{stripped}_{int(series_id)}.csv",
        f"SERIE_{int(series_id)}.csv",
    ]

    src: Path | None = None
    for d in local_dirs:
        for name in candidates:
            p = d / name
            if p.exists():
                src = p
                break
        if src is not None:
            break
    if src is None:
        return pd.DataFrame(columns=["date", "value"])

    raw = pd.read_csv(src)
    col_map = {str(c).strip().lower(): str(c) for c in raw.columns}
    date_col = col_map.get("date") or col_map.get("data")
    value_col = col_map.get("value") or col_map.get("valor")
    if not date_col or not value_col:
        return pd.DataFrame(columns=["date", "value"])
    df = raw[[date_col, value_col]].copy()
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    missing_date = df["date"].isna()
    if bool(missing_date.any()):
        df.loc[missing_date, "date"] = pd.to_datetime(df.loc[missing_date, "date"], errors="coerce", dayfirst=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).sort_values("date").drop_duplicates("date", keep="last")
    if df.empty:
        return pd.DataFrame(columns=["date", "value"])
    df = df[(df["date"].dt.year >= int(start_year)) & (df["date"].dt.year <= int(end_year))].copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "value"])
    out = _monthly_aggregate(df, aggregation=aggregation)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out[np.isfinite(out["value"])]
    return out.reset_index(drop=True)


def _load_bcb_proxy_series(
    *,
    proxy_asset_id: str,
    aggregation: str,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    src_candidates = [
        ROOT / "data" / "raw" / "agro" / "bcb" / f"{proxy_asset_id}.csv",
        ROOT / "data" / "download" / "agro" / "bcb" / f"{proxy_asset_id}.csv",
        ROOT / "data" / "raw" / "realestate" / "bcb" / f"{proxy_asset_id}.csv",
    ]
    src = next((p for p in src_candidates if p.exists()), None)
    if src is None:
        return pd.DataFrame(columns=["date", "value"])
    d = pd.read_csv(src)
    if d.empty:
        return pd.DataFrame(columns=["date", "value"])
    col_map = {str(c).strip().lower(): str(c) for c in d.columns}
    date_col = col_map.get("date") or col_map.get("data")
    value_col = col_map.get("value") or col_map.get("valor")
    if not date_col or not value_col:
        return pd.DataFrame(columns=["date", "value"])
    x = d[[date_col, value_col]].copy()
    x.columns = ["date", "value"]
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x["value"] = pd.to_numeric(x["value"], errors="coerce")
    x = x.dropna(subset=["date", "value"]).sort_values("date")
    if x.empty:
        return pd.DataFrame(columns=["date", "value"])
    x = x[(x["date"].dt.year >= int(start_year)) & (x["date"].dt.year <= int(end_year))].copy()
    if x.empty:
        return pd.DataFrame(columns=["date", "value"])
    out = _monthly_aggregate(x, aggregation=aggregation)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out[np.isfinite(out["value"])]
    return out.reset_index(drop=True)


def _download_file(url: str, out_path: Path, timeout_sec: float) -> dict[str, object]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "Assyntrax/1.0"})
    with urlopen(req, timeout=float(timeout_sec)) as resp:  # noqa: S310 (explicit URL input by operator)
        payload = resp.read()
    out_path.write_bytes(payload)
    return {"status": "ok", "file": str(out_path), "bytes": int(len(payload))}


def _parse_urls(text: str) -> list[str]:
    return [u.strip() for u in str(text).split(",") if u.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch Agro Brasil monthly sources (BCB + optional Comex/CONAB raw files).")
    ap.add_argument("--config-path", type=str, default=str(DEFAULT_CONFIG))
    ap.add_argument("--start-year", type=int, default=2000)
    ap.add_argument("--end-year", type=int, default=datetime.now(timezone.utc).year)
    ap.add_argument("--download-dir", type=str, default="data/download/agro")
    ap.add_argument("--raw-dir", type=str, default="data/raw/agro")
    ap.add_argument("--timeout-sec", type=float, default=10.0)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--download-once", type=int, default=1, help="If 1, keep existing files and skip new download.")
    ap.add_argument("--allow-local-fallback", type=int, default=1, help="If 1, when API fails, use local cached BCB files.")
    ap.add_argument("--comex-urls", type=str, default="", help="Comma-separated file URLs for Comex raw files.")
    ap.add_argument("--conab-urls", type=str, default="", help="Comma-separated file URLs for CONAB raw files.")
    args = ap.parse_args()

    cfg_path = Path(args.config_path)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / str(args.config_path)
    if not cfg_path.exists():
        raise SystemExit(f"missing config file: {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    download_root = Path(args.download_dir)
    if not download_root.is_absolute():
        download_root = ROOT / str(args.download_dir)
    raw_root = Path(args.raw_dir)
    if not raw_root.is_absolute():
        raw_root = ROOT / str(args.raw_dir)
    download_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    bcb_download_dir = download_root / "bcb"
    comex_download_dir = download_root / "comex"
    conab_download_dir = download_root / "conab"
    bcb_raw_dir = raw_root / "bcb"
    bcb_download_dir.mkdir(parents=True, exist_ok=True)
    comex_download_dir.mkdir(parents=True, exist_ok=True)
    conab_download_dir.mkdir(parents=True, exist_ok=True)
    bcb_raw_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"fetch_agro_br_{_run_id()}"
    summary: dict[str, object] = {
        "status": "ok",
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(cfg_path),
        "download_root": str(download_root),
        "raw_root": str(raw_root),
        "sources": {"bcb": [], "comex": [], "conab": []},
    }

    bcb_items = cfg.get("bcb_sgs", [])
    if not isinstance(bcb_items, list):
        raise SystemExit("invalid config: bcb_sgs must be a list")

    for item in bcb_items:
        if not isinstance(item, dict):
            continue
        asset_id = str(item.get("asset_id", "")).strip()
        series_id = int(item.get("series_id", 0) or 0)
        aggregation = str(item.get("aggregation", "last")).strip().lower() or "last"
        if not asset_id or series_id <= 0:
            continue
        rec: dict[str, object] = {"asset_id": asset_id, "series_id": series_id, "aggregation": aggregation}
        csv_name = f"{asset_id}.csv"
        dl_path = bcb_download_dir / csv_name
        raw_path = bcb_raw_dir / csv_name
        try:
            if bool(int(args.download_once)) and dl_path.exists() and raw_path.exists():
                m = pd.read_csv(raw_path)
            else:
                m = _fetch_bcb_series_monthly(
                    series_id=series_id,
                    start_year=int(args.start_year),
                    end_year=int(args.end_year),
                    aggregation=aggregation,
                    timeout_sec=float(args.timeout_sec),
                    max_retries=int(args.max_retries),
                )
                m.to_csv(dl_path, index=False)
                m.to_csv(raw_path, index=False)
            rec.update(
                {
                    "status": "ok",
                    "download_csv": str(dl_path),
                    "raw_csv": str(raw_path),
                    "rows": int(m.shape[0]),
                    "start": str(m["date"].iloc[0]) if not m.empty else "",
                    "end": str(m["date"].iloc[-1]) if not m.empty else "",
                }
            )
        except Exception as exc:  # pragma: no cover - network/runtime variability
            if bool(int(args.allow_local_fallback)):
                try:
                    m_local = _load_local_bcb_fallback(
                        asset_id=asset_id,
                        series_id=series_id,
                        aggregation=aggregation,
                        start_year=int(args.start_year),
                        end_year=int(args.end_year),
                    )
                    if not m_local.empty:
                        m_local.to_csv(dl_path, index=False)
                        m_local.to_csv(raw_path, index=False)
                        rec.update(
                            {
                                "status": "ok_local_fallback",
                                "reason": str(exc),
                                "download_csv": str(dl_path),
                                "raw_csv": str(raw_path),
                                "rows": int(m_local.shape[0]),
                                "start": str(m_local["date"].iloc[0]),
                                "end": str(m_local["date"].iloc[-1]),
                            }
                        )
                    else:
                        proxy_asset = BCB_PROXY_ASSETS.get(asset_id)
                        if proxy_asset:
                            m_proxy = _load_bcb_proxy_series(
                                proxy_asset_id=proxy_asset,
                                aggregation=aggregation,
                                start_year=int(args.start_year),
                                end_year=int(args.end_year),
                            )
                            if not m_proxy.empty:
                                m_proxy.to_csv(dl_path, index=False)
                                m_proxy.to_csv(raw_path, index=False)
                                rec.update(
                                    {
                                        "status": "ok_proxy",
                                        "proxy_asset_id": str(proxy_asset),
                                        "reason": str(exc),
                                        "download_csv": str(dl_path),
                                        "raw_csv": str(raw_path),
                                        "rows": int(m_proxy.shape[0]),
                                        "start": str(m_proxy["date"].iloc[0]),
                                        "end": str(m_proxy["date"].iloc[-1]),
                                    }
                                )
                            else:
                                rec.update({"status": "fail", "reason": str(exc)})
                        else:
                            rec.update({"status": "fail", "reason": str(exc)})
                except Exception as exc_local:
                    rec.update({"status": "fail", "reason": f"{exc} | fallback={exc_local}"})
            else:
                rec.update({"status": "fail", "reason": str(exc)})
        summary["sources"]["bcb"].append(rec)

    def _download_list(urls: list[str], outdir: Path) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        for idx, url in enumerate(urls, start=1):
            name = url.split("/")[-1].split("?")[0].strip() or f"file_{idx}.bin"
            target = outdir / name
            if bool(int(args.download_once)) and target.exists():
                out.append({"status": "ok", "file": str(target), "bytes": int(target.stat().st_size), "url": url, "cached": True})
                continue
            try:
                rec = _download_file(url=url, out_path=target, timeout_sec=float(args.timeout_sec))
                rec["url"] = url
                out.append(rec)
            except Exception as exc:  # pragma: no cover - network/runtime variability
                out.append({"status": "fail", "url": url, "file": str(target), "reason": str(exc)})
        return out

    summary["sources"]["comex"] = _download_list(_parse_urls(args.comex_urls), comex_download_dir)
    summary["sources"]["conab"] = _download_list(_parse_urls(args.conab_urls), conab_download_dir)

    out_dir = ROOT / "results" / "agro_br" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fetch_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "ok", "summary_json": str(out_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
