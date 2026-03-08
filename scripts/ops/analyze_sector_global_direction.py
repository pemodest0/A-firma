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


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _corr(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    z = pd.concat([x, y], axis=1).dropna()
    if len(z) < 8:
        return float("nan")
    return _safe_float(z.iloc[:, 0].corr(z.iloc[:, 1]))


def _load_panel(impact_csv: Path, sector_kind: str, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    usecols = [
        "date",
        "sector_kind",
        "sector",
        "sector_loading",
        "overlap_sector_global",
        "impact_global",
        "impact_sector",
        "global_score",
    ]
    d = pd.read_csv(impact_csv, usecols=usecols)
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date", "sector"]).copy()
    d["sector_kind"] = d["sector_kind"].astype(str).str.lower()
    d["sector"] = d["sector"].astype(str).str.strip()
    d = d[d["sector"].str.len() > 0].copy()
    d = d[d["sector"].str.lower() != "unknown"].copy()
    d = d[d["sector_kind"] == str(sector_kind).strip().lower()].copy()
    for c in ["sector_loading", "overlap_sector_global", "impact_global", "impact_sector", "global_score"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    if start_date:
        d = d[d["date"] >= pd.to_datetime(start_date, errors="coerce")]
    if end_date:
        d = d[d["date"] <= pd.to_datetime(end_date, errors="coerce")]
    return d.sort_values(["date", "sector"]).reset_index(drop=True)


def _build_daily_sector(d: pd.DataFrame) -> pd.DataFrame:
    sec = (
        d.groupby(["date", "sector"], as_index=False)
        .agg(
            sector_loading=("sector_loading", "mean"),
            overlap_sector_global=("overlap_sector_global", "mean"),
            impact_sector=("impact_sector", "mean"),
            impact_global_sector=("impact_global", "sum"),
            global_score=("global_score", "mean"),
            n_assets=("impact_global", "size"),
        )
        .sort_values(["date", "sector"])
        .reset_index(drop=True)
    )
    tot = sec.groupby("date", as_index=False)["impact_global_sector"].sum().rename(columns={"impact_global_sector": "impact_global_total"})
    sec = sec.merge(tot, on="date", how="left")
    sec["impact_share"] = np.where(
        pd.to_numeric(sec["impact_global_total"], errors="coerce").fillna(0.0) > 0.0,
        pd.to_numeric(sec["impact_global_sector"], errors="coerce").fillna(0.0)
        / pd.to_numeric(sec["impact_global_total"], errors="coerce").fillna(np.nan),
        np.nan,
    )
    return sec


def _direction_metrics(sec: pd.DataFrame, min_obs: int, lead_days: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sector, g in sec.groupby("sector"):
        z = g.sort_values("date").reset_index(drop=True)
        if len(z) < int(min_obs):
            continue
        load = z["sector_loading"]
        ovlp = z["overlap_sector_global"]
        glob = z["global_score"]
        # Sector -> Global (lead): sector_t with global_{t+lead}
        s_to_g_load = _corr(load, glob.shift(-int(lead_days)))
        s_to_g_ovlp = _corr(ovlp, glob.shift(-int(lead_days)))
        # Global -> Sector (lead): global_t with sector_{t+lead}
        g_to_s_load = _corr(glob, load.shift(-int(lead_days)))
        g_to_s_ovlp = _corr(glob, ovlp.shift(-int(lead_days)))
        edge_load = _safe_float(abs(s_to_g_load) - abs(g_to_s_load))
        edge_ovlp = _safe_float(abs(s_to_g_ovlp) - abs(g_to_s_ovlp))
        edge_avg = _safe_float(np.nanmean([edge_load, edge_ovlp]))
        if np.isnan(edge_avg):
            direction = "inconclusivo"
        elif edge_avg > 0.02:
            direction = "setor_lidera_global"
        elif edge_avg < -0.02:
            direction = "global_lidera_setor"
        else:
            direction = "acoplado"
        rows.append(
            {
                "sector": str(sector),
                "obs": int(len(z)),
                "impact_share_mean": _safe_float(z["impact_share"].mean()),
                "impact_share_p90": _safe_float(z["impact_share"].quantile(0.90)),
                "impact_sector_mean": _safe_float(z["impact_sector"].mean()),
                "sector_loading_mean": _safe_float(z["sector_loading"].mean()),
                "overlap_mean": _safe_float(z["overlap_sector_global"].mean()),
                "corr_now_loading_global": _corr(load, glob),
                "corr_now_overlap_global": _corr(ovlp, glob),
                "lead_s_to_g_loading": s_to_g_load,
                "lead_g_to_s_loading": g_to_s_load,
                "lead_s_to_g_overlap": s_to_g_ovlp,
                "lead_g_to_s_overlap": g_to_s_ovlp,
                "lead_edge_loading_abs": edge_load,
                "lead_edge_overlap_abs": edge_ovlp,
                "lead_edge_avg_abs": edge_avg,
                "direction": direction,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["impact_share_mean", "lead_edge_avg_abs"], ascending=[False, False]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze sector->global vs global->sector direction using impact dataset.")
    ap.add_argument("--impact-csv", required=True)
    ap.add_argument("--sector-kind", default="gics")
    ap.add_argument("--start-date", default="")
    ap.add_argument("--end-date", default="")
    ap.add_argument("--lead-days", type=int, default=5)
    ap.add_argument("--min-obs", type=int, default=60)
    ap.add_argument("--outdir", default="")
    args = ap.parse_args()

    impact_csv = Path(args.impact_csv).resolve()
    if not impact_csv.exists():
        raise FileNotFoundError(f"missing file: {impact_csv}")

    run_id = _run_id()
    outdir = Path(args.outdir).resolve() if str(args.outdir).strip() else (ROOT / "results" / "portfolio_sim" / f"{run_id}_sector_global_direction")
    outdir.mkdir(parents=True, exist_ok=True)

    panel = _load_panel(
        impact_csv=impact_csv,
        sector_kind=str(args.sector_kind),
        start_date=(str(args.start_date).strip() or None),
        end_date=(str(args.end_date).strip() or None),
    )
    if panel.empty:
        payload = {
            "status": "empty",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "impact_csv": str(impact_csv),
            "sector_kind": str(args.sector_kind),
            "start_date": (str(args.start_date).strip() or None),
            "end_date": (str(args.end_date).strip() or None),
        }
        out_json = outdir / "sector_global_direction_summary.json"
        out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"status": "empty", "outdir": str(outdir), "summary_json": str(out_json)}, ensure_ascii=False))
        return

    sec = _build_daily_sector(panel)
    met = _direction_metrics(sec, min_obs=int(args.min_obs), lead_days=int(args.lead_days))
    met_csv = outdir / "sector_direction_metrics.csv"
    sec_csv = outdir / "daily_sector_panel.csv"
    sec.to_csv(sec_csv, index=False)
    met.to_csv(met_csv, index=False)

    top_by_share = met.sort_values("impact_share_mean", ascending=False).head(10).to_dict(orient="records") if not met.empty else []
    top_sector_leads = (
        met[met["direction"] == "setor_lidera_global"]
        .sort_values("lead_edge_avg_abs", ascending=False)
        .head(10)
        .to_dict(orient="records")
        if not met.empty
        else []
    )
    top_global_leads = (
        met[met["direction"] == "global_lidera_setor"]
        .sort_values("lead_edge_avg_abs", ascending=True)
        .head(10)
        .to_dict(orient="records")
        if not met.empty
        else []
    )
    payload = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "impact_csv": str(impact_csv),
        "sector_kind": str(args.sector_kind),
        "start_date": (str(args.start_date).strip() or None),
        "end_date": (str(args.end_date).strip() or None),
        "lead_days": int(args.lead_days),
        "min_obs": int(args.min_obs),
        "n_rows_input": int(panel.shape[0]),
        "n_rows_daily": int(sec.shape[0]),
        "n_sectors": int(met.shape[0]),
        "top_by_impact_share": top_by_share,
        "top_sector_leads_global": top_sector_leads,
        "top_global_leads_sector": top_global_leads,
        "artifacts": {
            "daily_sector_panel_csv": str(sec_csv),
            "sector_direction_metrics_csv": str(met_csv),
        },
    }
    out_json = outdir / "sector_global_direction_summary.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "ok", "outdir": str(outdir), "summary_json": str(out_json), "n_sectors": int(met.shape[0])}, ensure_ascii=False))


if __name__ == "__main__":
    main()

