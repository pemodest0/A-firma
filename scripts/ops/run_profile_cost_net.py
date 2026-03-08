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
from execution.cost_model import (  # noqa: E402
    apply_cost_model,
    default_market_profiles,
    summarize_return_series,
)


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _latest_attack25_dir() -> Path:
    base = ROOT / "results" / "portfolio_sim"
    runs = sorted([p for p in base.iterdir() if p.is_dir() and "ultra_return_compact_attack25" in p.name], key=lambda p: p.name, reverse=True)
    for run in runs:
        if (run / "monthly_systematic_eval.csv").exists():
            return run
    raise FileNotFoundError("attack25 profile not found")


def main() -> None:
    ap = argparse.ArgumentParser(description="Recalcula métricas líquidas BR/US para um perfil mensal.")
    ap.add_argument("--profile-dir", default="", help="Diretório do perfil (default: latest attack25)")
    ap.add_argument("--outdir", default="", help="Diretório de saída (default: <profile-dir>/cost_net_<run_id>)")
    ap.add_argument("--tax-rate", type=float, default=0.15)
    ap.add_argument("--extra-slippage-br-bps", type=float, default=0.0)
    ap.add_argument("--extra-slippage-us-bps", type=float, default=0.0)
    args = ap.parse_args()

    profile_dir = Path(args.profile_dir).resolve() if str(args.profile_dir).strip() else _latest_attack25_dir()
    monthly_csv = profile_dir / "monthly_systematic_eval.csv"
    if not monthly_csv.exists():
        raise FileNotFoundError(f"missing monthly file: {monthly_csv}")
    outdir = Path(args.outdir).resolve() if str(args.outdir).strip() else (profile_dir / f"cost_net_{_run_id()}")
    outdir.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(monthly_csv)
    d["ym"] = d["ym"].astype(str)
    d["ret"] = pd.to_numeric(d.get("ret"), errors="coerce").fillna(0.0).astype(float)
    d["turnover"] = pd.to_numeric(d.get("turnover"), errors="coerce").fillna(0.0).astype(float)
    d["eqw_ret"] = pd.to_numeric(d.get("eqw_ret"), errors="coerce").fillna(0.0).astype(float)
    d["mkt_ret"] = pd.to_numeric(d.get("mkt_ret"), errors="coerce").fillna(0.0).astype(float)

    profiles = default_market_profiles(tax_rate=float(max(0.0, args.tax_rate)))
    br_df = apply_cost_model(
        d["ret"],
        d["turnover"],
        profile=profiles["BR"],
        extra_slippage_bps=float(max(0.0, args.extra_slippage_br_bps)),
    )
    us_df = apply_cost_model(
        d["ret"],
        d["turnover"],
        profile=profiles["US"],
        extra_slippage_bps=float(max(0.0, args.extra_slippage_us_bps)),
    )

    out_monthly = pd.DataFrame(
        {
            "ym": d["ym"].astype(str),
            "ret_gross": d["ret"].astype(float),
            "turnover": d["turnover"].astype(float),
            "ret_net_br": pd.to_numeric(br_df["net_ret"], errors="coerce").fillna(0.0).astype(float),
            "ret_net_us": pd.to_numeric(us_df["net_ret"], errors="coerce").fillna(0.0).astype(float),
            "ret_eqw": d["eqw_ret"].astype(float),
            "ret_market": d["mkt_ret"].astype(float),
        }
    )
    monthly_path = outdir / "cost_net_monthly.csv"
    out_monthly.to_csv(monthly_path, index=False)

    payload = {
        "status": "ok",
        "profile_dir": str(profile_dir),
        "monthly_csv": str(monthly_csv),
        "profiles": {
            "BR": profiles["BR"].to_dict(),
            "US": profiles["US"].to_dict(),
        },
        "extra_slippage_bps": {
            "BR": float(max(0.0, args.extra_slippage_br_bps)),
            "US": float(max(0.0, args.extra_slippage_us_bps)),
        },
        "metrics": {
            "gross_strategy": summarize_return_series(out_monthly["ret_gross"]),
            "net_strategy_BR": summarize_return_series(out_monthly["ret_net_br"]),
            "net_strategy_US": summarize_return_series(out_monthly["ret_net_us"]),
            "eqw": summarize_return_series(out_monthly["ret_eqw"]),
            "market": summarize_return_series(out_monthly["ret_market"]),
        },
        "paths": {
            "cost_net_monthly_csv": str(monthly_path),
        },
    }
    report_path = outdir / "cost_net_report.json"
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    write_run_manifest(
        outdir=outdir,
        script="scripts/ops/run_profile_cost_net.py",
        params={
            "profile_dir": str(profile_dir),
            "tax_rate": float(args.tax_rate),
            "extra_slippage_br_bps": float(args.extra_slippage_br_bps),
            "extra_slippage_us_bps": float(args.extra_slippage_us_bps),
        },
        paths={"report_json": str(report_path), "monthly_csv": str(monthly_path)},
        gates={"report_created": report_path.exists(), "monthly_created": monthly_path.exists()},
    )

    print(json.dumps({"status": "ok", "outdir": str(outdir), "report": str(report_path)}))


if __name__ == "__main__":
    main()

