#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path
from typing import Iterable, List, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DEFAULT_FIPEZAP_URLS = [
    ("fipezap_serieshistoricas.xlsx", "https://downloads.fipe.org.br/indices/fipezap/fipezap-serieshistoricas.xlsx"),
    ("fipezap_historico.xlsx", "https://downloads.fipe.org.br/indices/fipezap/historico-fipezap-01.xlsx"),
]

# Core BCB SGS ids for real-estate data layer:
# P(t): price/cost proxies and inflation
# J(t): interest/financing rates
# L(t): liquidity/credit volume proxies
DEFAULT_BCB_IDS = [
    ("SELIC_D", 11),
    ("IPCA_M", 433),
    ("INPC_M", 188),
    ("IGPM_M", 189),
    ("CRED_IMOB_PF_JUROS_MERCADO", 25447),
    ("CRED_IMOB_PF_JUROS_TOTAL", 20038),
    ("INCC_M", 192),
    ("CRED_IMOB_PF_SALDO", 20540),
]


def _chunked_year_ranges(start: int, end: int, span: int = 10) -> Iterable[Tuple[date, date]]:
    year = start
    while year <= end:
        s = date(year, 1, 1)
        e = date(min(year + span - 1, end), 12, 31)
        yield s, e
        year += span


def _fetch_bcb_series(series_id: int, start: int, end: int, timeout_sec: float) -> List[dict]:
    rows: List[dict] = []
    for s, e in _chunked_year_ranges(start, end, span=10):
        query = urlencode(
            {
                "formato": "json",
                "dataInicial": s.strftime("%d/%m/%Y"),
                "dataFinal": e.strftime("%d/%m/%Y"),
            }
        )
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_id}/dados?{query}"
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 (Assyntrax)"})
        try:
            with urlopen(req, timeout=timeout_sec) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception:
            # fallback without date range
            fallback = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_id}/dados?formato=json"
            req_fb = Request(fallback, headers={"User-Agent": "Mozilla/5.0 (Assyntrax)"})
            with urlopen(req_fb, timeout=timeout_sec) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        rows.extend(payload)

    # Keep unique dates only (latest wins)
    by_date = {}
    for row in rows:
        d = row.get("data")
        if d:
            by_date[d] = row.get("valor")
    return [{"data": d, "valor": v} for d, v in sorted(by_date.items())]


def _write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["data", "valor"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"data": row.get("data"), "valor": row.get("valor")})


def _extract_excel_sheets(xlsx_path: Path, outdir: Path) -> None:
    try:
        import pandas as pd
    except Exception:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        xls = pd.ExcelFile(xlsx_path)
    except Exception:
        return
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in sheet).strip("_")
        df.to_csv(outdir / f"{safe}.csv", index=False)


def _download_file(url: str, dest: Path, timeout_sec: float) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (Assyntrax)"})
    with urlopen(req, timeout=timeout_sec) as resp, dest.open("wb") as f:
        f.write(resp.read())


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch official real-estate sources (FipeZAP, BCB SGS, optional ABRAINC files).")
    parser.add_argument("--outdir", default="data/raw/realestate", help="Base output dir for downloads")
    parser.add_argument("--bcb-start-year", type=int, default=2000)
    parser.add_argument("--bcb-end-year", type=int, default=date.today().year)
    parser.add_argument("--timeout-sec", type=float, default=5.0, help="Timeout por request HTTP em segundos")
    parser.add_argument("--bcb-ids", default="", help="Comma-separated list of id:name or id (e.g. 11:SELIC_D,433:IPCA_M)")
    parser.add_argument(
        "--abrainc-urls",
        default="",
        help="Comma-separated direct ABRAINC-FIPE file URLs (.xlsx/.xls/.csv)",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_date": date.today().isoformat(),
        "fipezap": [],
        "bcb": [],
        "abrainc": [],
    }

    # FipeZAP downloads
    fipezap_dir = outdir / "fipezap"
    fipezap_dir.mkdir(parents=True, exist_ok=True)
    for fname, url in DEFAULT_FIPEZAP_URLS:
        dest = fipezap_dir / fname
        try:
            if not dest.exists():
                _download_file(url, dest, float(args.timeout_sec))
            _extract_excel_sheets(dest, fipezap_dir / "sheets")
            manifest["fipezap"].append({"file": str(dest), "status": "ok"})
        except Exception as exc:
            manifest["fipezap"].append({"file": str(dest), "status": "fail", "reason": str(exc)})

    # BCB SGS downloads
    bcb_dir = outdir / "bcb"
    bcb_dir.mkdir(parents=True, exist_ok=True)
    series = list(DEFAULT_BCB_IDS)
    if args.bcb_ids:
        series = []
        for token in args.bcb_ids.split(","):
            token = token.strip()
            if not token:
                continue
            if ":" in token:
                sid, name = token.split(":", 1)
                series.append((name.strip(), int(sid.strip())))
            else:
                series.append((f"SERIE_{token}", int(token)))

    for name, sid in series:
        target = bcb_dir / f"{name}_{sid}.csv"
        try:
            rows = _fetch_bcb_series(sid, args.bcb_start_year, args.bcb_end_year, float(args.timeout_sec))
            _write_csv(target, rows)
            manifest["bcb"].append({"series_id": sid, "name": name, "rows": len(rows), "status": "ok"})
        except Exception as exc:
            manifest["bcb"].append({"series_id": sid, "name": name, "status": "fail", "reason": str(exc)})

    # ABRAINC-FIPE manual URL ingestion
    abrainc_dir = outdir / "abrainc"
    abrainc_dir.mkdir(parents=True, exist_ok=True)
    urls = [u.strip() for u in args.abrainc_urls.split(",") if u.strip()]
    if urls:
        for idx, url in enumerate(urls, start=1):
            fname = url.split("/")[-1].split("?")[0] or f"abrainc_{idx}.bin"
            dest = abrainc_dir / fname
            try:
                _download_file(url, dest, float(args.timeout_sec))
                if dest.suffix.lower() in {".xlsx", ".xls"}:
                    _extract_excel_sheets(dest, abrainc_dir / "sheets")
                manifest["abrainc"].append({"url": url, "file": str(dest), "status": "ok"})
            except Exception as exc:
                manifest["abrainc"].append({"url": url, "file": str(dest), "status": "fail", "reason": str(exc)})
    else:
        manifest["abrainc"].append(
            {
                "status": "skipped",
                "reason": "abrainc_urls_not_provided",
                "note": "Pass --abrainc-urls with direct links to ABRAINC-FIPE files.",
            }
        )

    manifest_path = outdir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote {manifest_path}")


if __name__ == "__main__":
    main()
