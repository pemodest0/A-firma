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
from engine.structural.stability_metrics import (  # noqa: E402
    ModeStabilityThresholds,
    apply_mode_stability_gate,
    dominant_mode_series,
    summarize_mode_stability,
)


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _latest_attack25_dir() -> Path:
    base = ROOT / "results" / "portfolio_sim"
    runs = sorted([p for p in base.iterdir() if p.is_dir() and "ultra_return_compact_attack25" in p.name], key=lambda p: p.name, reverse=True)
    for run in runs:
        if (run / "simulation_summary.json").exists() and (run / "monthly_systematic_eval.csv").exists():
            return run
    raise FileNotFoundError("attack25 profile not found")


def _build_monthly_returns(returns_csv: Path) -> pd.DataFrame:
    d = pd.read_csv(returns_csv)
    if "date" not in d.columns:
        raise ValueError(f"missing date column: {returns_csv}")
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    assets = [c for c in d.columns if c != "date"]
    if not assets:
        raise ValueError(f"no assets in {returns_csv}")
    d = d[["date", *assets]].copy()
    d[assets] = d[assets].apply(pd.to_numeric, errors="coerce")
    d = d.copy()
    d["ym"] = d["date"].dt.to_period("M").astype(str)
    rows: list[pd.Series] = []
    labels: list[str] = []
    for ym, g in d.groupby("ym"):
        arr = g[assets].to_numpy(dtype=float)
        s = pd.Series((1.0 + arr).prod(axis=0) - 1.0, index=assets, dtype=float)
        rows.append(s)
        labels.append(str(ym))
    out = pd.DataFrame(rows, index=labels).sort_index()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Avalia estabilidade do modo dominante (correlação) para um perfil.")
    ap.add_argument("--profile-dir", default="", help="Diretório do perfil (default: latest attack25)")
    ap.add_argument("--outdir", default="", help="Diretório de saída (default: <profile-dir>/stability_gate_<run_id>)")
    ap.add_argument("--window-months", type=int, default=12)
    ap.add_argument("--min-assets", type=int, default=20)
    ap.add_argument("--min-obs-asset", type=int, default=8)
    ap.add_argument("--min-median-overlap", type=float, default=0.55)
    ap.add_argument("--min-p10-overlap", type=float, default=0.30)
    ap.add_argument("--max-share-overlap-lt-05", type=float, default=0.35)
    ap.add_argument("--max-max-drift", type=float, default=0.90)
    args = ap.parse_args()

    profile_dir = Path(args.profile_dir).resolve() if str(args.profile_dir).strip() else _latest_attack25_dir()
    sim_json = profile_dir / "simulation_summary.json"
    monthly_csv = profile_dir / "monthly_systematic_eval.csv"
    if not sim_json.exists() or not monthly_csv.exists():
        raise FileNotFoundError(f"missing required files in {profile_dir}")

    sim = json.loads(sim_json.read_text(encoding="utf-8"))
    returns_csv = Path(str(sim.get("returns_csv", ""))).resolve()
    if not returns_csv.exists():
        raise FileNotFoundError(f"returns_csv not found: {returns_csv}")

    outdir = Path(args.outdir).resolve() if str(args.outdir).strip() else (profile_dir / f"stability_gate_{_run_id()}")
    outdir.mkdir(parents=True, exist_ok=True)

    profile_months = pd.read_csv(monthly_csv)["ym"].astype(str).tolist()
    mret = _build_monthly_returns(returns_csv)
    mret = mret.reindex(profile_months).fillna(0.0)

    stab_df, _vectors = dominant_mode_series(
        mret,
        window_months=int(args.window_months),
        min_assets=int(args.min_assets),
        min_obs_asset=int(args.min_obs_asset),
    )
    summary = summarize_mode_stability(stab_df)
    thresholds = ModeStabilityThresholds(
        min_median_overlap=float(args.min_median_overlap),
        min_p10_overlap=float(args.min_p10_overlap),
        max_share_overlap_lt_05=float(args.max_share_overlap_lt_05),
        max_max_drift=float(args.max_max_drift),
    )
    gate = apply_mode_stability_gate(summary, thresholds)

    series_path = outdir / "dominant_mode_stability_series.csv"
    stab_df.to_csv(series_path, index=False)
    payload = {
        "status": "ok",
        "profile_dir": str(profile_dir),
        "returns_csv": str(returns_csv),
        "stability_summary": summary,
        "gate_result": gate,
        "params": {
            "window_months": int(args.window_months),
            "min_assets": int(args.min_assets),
            "min_obs_asset": int(args.min_obs_asset),
        },
        "paths": {"stability_series_csv": str(series_path)},
    }
    report_path = outdir / "stability_gate_report.json"
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    write_run_manifest(
        outdir=outdir,
        script="scripts/ops/run_profile_stability_gate.py",
        params={
            "profile_dir": str(profile_dir),
            "window_months": int(args.window_months),
            "min_assets": int(args.min_assets),
            "min_obs_asset": int(args.min_obs_asset),
        },
        paths={"report_json": str(report_path), "series_csv": str(series_path)},
        gates={"stability_gate_passed": bool(gate.get("passed", False)), "report_created": report_path.exists()},
    )
    print(json.dumps({"status": "ok", "outdir": str(outdir), "passed": bool(gate.get("passed", False))}))


if __name__ == "__main__":
    main()
