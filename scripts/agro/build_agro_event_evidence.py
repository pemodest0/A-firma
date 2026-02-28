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
DEFAULT_CATALOG = ROOT / "config" / "event_catalog_agro_br.json"


def _safe_float(x: object) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _load_global_series(run_dir: Path) -> pd.DataFrame:
    candidates = [
        run_dir / "hierarchical" / "diagnostics_global_score_daily.csv",
        run_dir / "diagnostics_structural_score_daily.csv",
        run_dir / "diagnostics_structural_daily.csv",
    ]
    for p in candidates:
        if not p.exists():
            continue
        d = pd.read_csv(p)
        if d.empty or "date" not in d.columns:
            continue
        x = d.copy()
        x["date"] = pd.to_datetime(x["date"], errors="coerce")
        x = x.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        for col in ["score", "phi", "p1", "deff", "ac1_phi", "Q", "N_used", "T_window"]:
            if col not in x.columns:
                x[col] = np.nan
            x[col] = pd.to_numeric(x[col], errors="coerce")
        return x
    return pd.DataFrame(columns=["date", "score", "phi", "p1", "deff", "ac1_phi", "Q", "N_used", "T_window"])


def _mean_std(s: pd.Series) -> tuple[float, float]:
    x = pd.to_numeric(s, errors="coerce")
    x = x[np.isfinite(x)]
    if x.empty:
        return float("nan"), float("nan")
    return float(x.mean()), float(x.std(ddof=0))


def _z(value: float, mean: float, std: float) -> float:
    if (not np.isfinite(value)) or (not np.isfinite(mean)) or (not np.isfinite(std)) or std <= 1e-12:
        return float("nan")
    return float((value - mean) / std)


def _event_windows(date: pd.Timestamp) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    return {
        "baseline": (date - pd.DateOffset(months=12), date - pd.DateOffset(months=4)),
        "pre": (date - pd.DateOffset(months=3), date - pd.DateOffset(months=1)),
        "post": (date, date + pd.DateOffset(months=3)),
    }


def _slice(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df[(df["date"] >= start) & (df["date"] <= end)].copy()


def _build_event_row(df: pd.DataFrame, event: dict[str, Any]) -> dict[str, Any]:
    date = pd.to_datetime(event.get("date"), errors="coerce")
    if pd.isna(date):
        return {"id": str(event.get("id", "")), "status": "invalid_date"}
    windows = _event_windows(date)
    base = _slice(df, *windows["baseline"])
    pre = _slice(df, *windows["pre"])
    post = _slice(df, *windows["post"])

    metrics = {}
    for metric in ["score", "phi", "p1", "deff", "ac1_phi"]:
        b_mu, b_sd = _mean_std(base[metric]) if metric in base.columns else (float("nan"), float("nan"))
        pre_mu = _safe_float(pre[metric].mean()) if metric in pre.columns else float("nan")
        post_mu = _safe_float(post[metric].mean()) if metric in post.columns else float("nan")
        metrics[metric] = {
            "baseline_mean": b_mu,
            "baseline_std": b_sd,
            "pre_mean": pre_mu,
            "post_mean": post_mu,
            "z_pre": _z(pre_mu, b_mu, b_sd),
            "z_post": _z(post_mu, b_mu, b_sd),
        }

    z_score_pre = _safe_float(metrics["score"]["z_pre"])
    z_phi_pre = _safe_float(metrics["phi"]["z_pre"])
    z_deff_pre = _safe_float(metrics["deff"]["z_pre"])
    pre_signal = bool(
        (np.isfinite(z_score_pre) and z_score_pre >= 1.0)
        or (np.isfinite(z_phi_pre) and z_phi_pre >= 1.0)
        or (np.isfinite(z_deff_pre) and z_deff_pre <= -1.0)
    )

    return {
        "id": str(event.get("id", "")),
        "title": str(event.get("title", "")),
        "type": str(event.get("type", "")),
        "date": date.strftime("%Y-%m-%d"),
        "status": "ok",
        "samples": {
            "baseline": int(base.shape[0]),
            "pre": int(pre.shape[0]),
            "post": int(post.shape[0]),
        },
        "pre_signal": pre_signal,
        "metrics": metrics,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Agro BR pre/post event structural evidence summary.")
    ap.add_argument("--run-dir", type=str, required=True)
    ap.add_argument("--event-catalog", type=str, default=str(DEFAULT_CATALOG))
    ap.add_argument("--outdir", type=str, default="results/agro_br/latest")
    ap.add_argument("--score-z-threshold", type=float, default=1.0)
    ap.add_argument("--phi-z-threshold", type=float, default=1.0)
    ap.add_argument("--deff-z-threshold", type=float, default=-1.0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / str(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"run dir not found: {run_dir}")

    catalog_path = Path(args.event_catalog)
    if not catalog_path.is_absolute():
        catalog_path = ROOT / str(args.event_catalog)
    if not catalog_path.exists():
        raise SystemExit(f"event catalog not found: {catalog_path}")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    events = catalog.get("events", [])
    if not isinstance(events, list):
        raise SystemExit("invalid event catalog: events must be list")

    df = _load_global_series(run_dir)
    if df.empty:
        raise SystemExit("missing global diagnostics for event evidence")

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = ROOT / str(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = [_build_event_row(df, ev) for ev in events if isinstance(ev, dict)]
    score_thr = float(args.score_z_threshold)
    phi_thr = float(args.phi_z_threshold)
    deff_thr = float(args.deff_z_threshold)
    for row in rows:
        if row.get("status") != "ok":
            continue
        metrics = row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else {}
        z_score_pre = _safe_float((metrics.get("score") or {}).get("z_pre") if isinstance(metrics.get("score"), dict) else float("nan"))
        z_phi_pre = _safe_float((metrics.get("phi") or {}).get("z_pre") if isinstance(metrics.get("phi"), dict) else float("nan"))
        z_deff_pre = _safe_float((metrics.get("deff") or {}).get("z_pre") if isinstance(metrics.get("deff"), dict) else float("nan"))
        pre_signal = bool(
            (np.isfinite(z_score_pre) and z_score_pre >= score_thr)
            or (np.isfinite(z_phi_pre) and z_phi_pre >= phi_thr)
            or (np.isfinite(z_deff_pre) and z_deff_pre <= deff_thr)
        )
        row["pre_signal"] = pre_signal
    valid_rows = [r for r in rows if r.get("status") == "ok"]
    n_pre_signal = int(sum(1 for r in valid_rows if bool(r.get("pre_signal"))))
    pre_signal_rate = float(n_pre_signal / len(valid_rows)) if valid_rows else float("nan")

    latest = df.tail(1).iloc[0]
    summary = {
        "schema_version": "historical_structure_summary_agro_br_v1",
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "last_date": pd.to_datetime(latest["date"]).strftime("%Y-%m-%d"),
        "state_latest": {
            "score": _safe_float(latest.get("score")),
            "phi": _safe_float(latest.get("phi")),
            "p1": _safe_float(latest.get("p1")),
            "deff": _safe_float(latest.get("deff")),
            "Q": _safe_float(latest.get("Q")),
            "N_used": _safe_float(latest.get("N_used")),
            "T_window": _safe_float(latest.get("T_window")),
        },
        "evidence": {
            "events_total": int(len(rows)),
            "events_valid": int(len(valid_rows)),
            "pre_signal_count": int(n_pre_signal),
            "pre_signal_rate": pre_signal_rate,
            "thresholds": {
                "score_z_threshold": score_thr,
                "phi_z_threshold": phi_thr,
                "deff_z_threshold": deff_thr,
            },
            "interpretation": (
                "evidencia_suporte_h1"
                if np.isfinite(pre_signal_rate) and pre_signal_rate >= 0.5
                else "evidencia_moderada_ou_insuficiente"
            ),
        },
        "events": rows,
    }

    out_json = outdir / "historical_structure_summary_agro_br.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    out_csv = outdir / "historical_structure_summary_agro_br_events.csv"
    pd.DataFrame(
        [
            {
                "id": r.get("id"),
                "date": r.get("date"),
                "title": r.get("title"),
                "type": r.get("type"),
                "pre_signal": r.get("pre_signal"),
                "samples_baseline": (r.get("samples") or {}).get("baseline"),
                "samples_pre": (r.get("samples") or {}).get("pre"),
                "samples_post": (r.get("samples") or {}).get("post"),
                "z_score_pre": (((r.get("metrics") or {}).get("score") or {}).get("z_pre")),
                "z_phi_pre": (((r.get("metrics") or {}).get("phi") or {}).get("z_pre")),
                "z_deff_pre": (((r.get("metrics") or {}).get("deff") or {}).get("z_pre")),
            }
            for r in valid_rows
        ]
    ).to_csv(out_csv, index=False)

    print(json.dumps({"status": "ok", "summary_json": str(out_json), "summary_csv": str(out_csv)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
