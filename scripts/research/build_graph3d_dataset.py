#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

SECTOR_MAP = {
    "SPY": "finance",
    "QQQ": "finance",
    "DIA": "finance",
    "IWM": "finance",
    "VTI": "finance",
    "VT": "finance",
    "RSP": "finance",
    "XLF": "finance",
    "LQD": "finance",
    "HYG": "finance",
    "SHY": "finance",
    "IEF": "finance",
    "TLT": "finance",
    "TIP": "finance",
    "VIX": "finance",
    "^VIX": "finance",
    "GLD": "commodities",
    "SLV": "commodities",
    "USO": "commodities",
    "DBC": "commodities",
    "DBA": "commodities",
    "XLE": "commodities",
    "XOP": "commodities",
    "XLB": "commodities",
    "UUP": "fx",
    "FXE": "fx",
    "FXY": "fx",
    "BTC-USD": "crypto",
    "ETH-USD": "crypto",
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def tail_align(a: List, n: int) -> List:
    if n <= 0:
        return []
    return a[-n:]


def to_week_key(date_str: str) -> str:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    year, week, _ = dt.isocalendar()
    return f"{year}-W{week:02d}"


def build_time_index(dates: List[str], tf: str) -> List[str]:
    if tf == "daily":
        return dates
    out: List[str] = []
    last_key = ""
    for d in dates:
        key = to_week_key(d)
        if key != last_key:
            out.append(d)
            last_key = key
        else:
            out[-1] = d
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf", choices=["daily", "weekly"], default="daily")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--outdir", default=str(RESULTS / "graph_3d"))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    assets_dir = RESULTS / "latest_graph" / "assets"
    assets = sorted({p.name.split("_")[0] for p in assets_dir.glob("*_regimes.csv")})

    data_out: Dict[str, List[Dict[str, object]]] = {}
    sector_out: Dict[str, str] = {}

    for asset in assets:
        tf = args.tf
        reg_path = assets_dir / f"{asset}_{tf}_regimes.csv"
        emb_path = assets_dir / f"{asset}_{tf}_embedding.csv"
        price_path = ROOT / "data" / "raw" / "finance" / "yfinance_daily" / f"{asset}.csv"
        if not (reg_path.exists() and emb_path.exists() and price_path.exists()):
            continue

        regs = read_csv(reg_path)
        emb = read_csv(emb_path)
        prices = read_csv(price_path)

        dates = [r.get("date", "") for r in prices if r.get("date")]
        dates = build_time_index(dates, tf)

        n = min(len(regs), len(emb), len(dates))
        if n <= 5:
            continue

        regs = tail_align(regs, n)
        emb = tail_align(emb, n)
        dates = tail_align(dates, n)
        prices = tail_align(prices, n)

        points: List[Dict[str, object]] = []
        for i in range(0, n, max(1, args.step)):
            reg = regs[i]
            e = emb[i]
            p = prices[i]
            try:
                x = float(e.get("c1") or e.get("x") or 0.0)
                y = float(e.get("c2") or e.get("y") or 0.0)
            except ValueError:
                x = 0.0
                y = 0.0
            try:
                price = float(p.get("price") or 0.0)
            except ValueError:
                price = 0.0
            try:
                ret = float(p.get("r") or 0.0)
            except ValueError:
                ret = 0.0
            try:
                conf = float(reg.get("confidence") or 0.0)
            except ValueError:
                conf = 0.0
            regime = reg.get("regime") or "NOISY"
            points.append(
                {
                    "date": dates[i],
                    "x": x,
                    "y": y,
                    "z": ret,
                    "price": price,
                    "ret": ret,
                    "confidence": conf,
                    "regime": regime,
                }
            )

        data_out[asset] = points
        sector_out[asset] = SECTOR_MAP.get(asset, "unknown")

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "timeframe": args.tf,
        "assets": sorted(data_out.keys()),
        "sectors": sector_out,
        "points": data_out,
    }

    out_path = outdir / f"graph3d_{args.tf}.json"
    out_path.write_text(json.dumps(payload))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
