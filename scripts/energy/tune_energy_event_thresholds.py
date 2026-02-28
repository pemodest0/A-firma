#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = ROOT / "config" / "event_catalog_energy_br.json"


def _parse_grid(text: str) -> list[float]:
    out: list[float] = []
    for part in str(text).split(","):
        s = part.strip()
        if not s:
            continue
        out.append(float(s))
    if not out:
        raise ValueError("grid vazio")
    return out


def _load_series(run_dir: Path) -> pd.DataFrame:
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
        for col in ["score", "phi", "deff"]:
            if col not in x.columns:
                x[col] = np.nan
            x[col] = pd.to_numeric(x[col], errors="coerce")
        return x[["date", "score", "phi", "deff"]].drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    raise FileNotFoundError(f"diagnostics global ausente em {run_dir}")


def _load_events(path: Path) -> list[pd.Timestamp]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("events", [])
    out: list[pd.Timestamp] = []
    for r in rows:
        dt = pd.to_datetime((r or {}).get("date"), errors="coerce")
        if pd.isna(dt):
            continue
        out.append(dt.normalize())
    out = sorted(set(out))
    if not out:
        raise ValueError("catalogo sem datas validas")
    return out


def _causal_zscores(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["z_score"] = np.nan
    x["z_phi"] = np.nan
    x["z_deff"] = np.nan
    for i, row in x.iterrows():
        d = row["date"]
        base = x[(x["date"] >= d - pd.Timedelta(days=365)) & (x["date"] <= d - pd.Timedelta(days=120))].copy()
        if base.shape[0] < 60:
            continue
        zvals: dict[str, float] = {}
        ok = True
        for col in ["score", "phi", "deff"]:
            b = pd.to_numeric(base[col], errors="coerce")
            b = b[np.isfinite(b)]
            v = float(row[col]) if np.isfinite(row[col]) else float("nan")
            if b.shape[0] < 60 or not np.isfinite(v):
                ok = False
                break
            mu = float(b.mean())
            sd = float(b.std(ddof=0))
            zvals[col] = float((v - mu) / sd) if np.isfinite(sd) and sd > 1e-12 else float("nan")
        if not ok:
            continue
        x.at[i, "z_score"] = zvals["score"]
        x.at[i, "z_phi"] = zvals["phi"]
        x.at[i, "z_deff"] = zvals["deff"]
    return x


def _event_hits(dates: pd.Series, alerts: np.ndarray, events: list[pd.Timestamp]) -> tuple[int, int, list[dict[str, object]]]:
    x = pd.DataFrame({"date": dates.values, "alert": alerts})
    details: list[dict[str, object]] = []
    n_valid = 0
    n_hit = 0
    for e in events:
        pre = x[(x["date"] >= e - pd.Timedelta(days=90)) & (x["date"] <= e - pd.Timedelta(days=7))].copy()
        if pre.empty:
            details.append({"event_date": e.strftime("%Y-%m-%d"), "valid": False, "pre_signal": False})
            continue
        n_valid += 1
        hit = bool(pre["alert"].fillna(False).any())
        if hit:
            n_hit += 1
        details.append({"event_date": e.strftime("%Y-%m-%d"), "valid": True, "pre_signal": hit})
    return n_hit, n_valid, details


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep de limiares do Energia BR para maximizar pre-sinal com budget de alerta.")
    ap.add_argument("--run-dir", type=str, required=True)
    ap.add_argument("--event-catalog", type=str, default=str(DEFAULT_CATALOG))
    ap.add_argument("--score-grid", type=str, default="0.0,0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--phi-grid", type=str, default="0.2,0.4,0.6,0.8,1.0,1.2")
    ap.add_argument("--deff-grid", type=str, default="-1.0,-0.5,0.0,0.5")
    ap.add_argument("--target-alert-rate", type=float, default=0.25)
    ap.add_argument("--max-alert-rate", type=float, default=0.45)
    ap.add_argument("--outdir", type=str, default="results/energy_br/latest/threshold_tuning")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / str(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"run dir not found: {run_dir}")

    event_catalog = Path(args.event_catalog)
    if not event_catalog.is_absolute():
        event_catalog = ROOT / str(args.event_catalog)
    if not event_catalog.exists():
        raise SystemExit(f"event catalog not found: {event_catalog}")

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = ROOT / str(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    score_grid = _parse_grid(args.score_grid)
    phi_grid = _parse_grid(args.phi_grid)
    deff_grid = _parse_grid(args.deff_grid)

    series = _load_series(run_dir=run_dir)
    events = _load_events(event_catalog)
    base = _causal_zscores(series)
    eligible = base[np.isfinite(base["z_score"]) | np.isfinite(base["z_phi"]) | np.isfinite(base["z_deff"])].copy()

    rows: list[dict[str, object]] = []
    for score_thr, phi_thr, deff_thr in itertools.product(score_grid, phi_grid, deff_grid):
        alert_mask = (
            ((np.isfinite(base["z_score"])) & (base["z_score"] >= float(score_thr)))
            | ((np.isfinite(base["z_phi"])) & (base["z_phi"] >= float(phi_thr)))
            | ((np.isfinite(base["z_deff"])) & (base["z_deff"] <= float(deff_thr)))
        )
        alert_rate = float(alert_mask[eligible.index].mean()) if not eligible.empty else float("nan")
        n_hit, n_valid, details = _event_hits(dates=base["date"], alerts=alert_mask.values, events=events)
        pre_rate = float(n_hit / n_valid) if n_valid > 0 else float("nan")
        rows.append(
            {
                "score_z_threshold": float(score_thr),
                "phi_z_threshold": float(phi_thr),
                "deff_z_threshold": float(deff_thr),
                "n_events_valid": int(n_valid),
                "pre_signal_count": int(n_hit),
                "pre_signal_rate": pre_rate,
                "alert_rate": alert_rate,
                "budget_ok": bool(np.isfinite(alert_rate) and alert_rate <= float(args.max_alert_rate)),
                "distance_to_target_alert_rate": (
                    abs(alert_rate - float(args.target_alert_rate)) if np.isfinite(alert_rate) else float("inf")
                ),
                "event_details": json.dumps(details, ensure_ascii=False),
            }
        )

    df = pd.DataFrame(rows).sort_values(
        by=["budget_ok", "pre_signal_count", "pre_signal_rate", "distance_to_target_alert_rate"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    best = df.iloc[0].to_dict() if not df.empty else {}
    best_clean = {k: (None if (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in best.items()}

    top_csv = outdir / "threshold_sweep_top.csv"
    full_csv = outdir / "threshold_sweep_full.csv"
    df.head(15).to_csv(top_csv, index=False)
    df.to_csv(full_csv, index=False)

    rec = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "event_catalog": str(event_catalog),
        "target_alert_rate": float(args.target_alert_rate),
        "max_alert_rate": float(args.max_alert_rate),
        "grid_size": int(df.shape[0]),
        "best": best_clean,
        "recommended_cli": (
            f"--score-z-threshold {best_clean.get('score_z_threshold')} "
            f"--phi-z-threshold {best_clean.get('phi_z_threshold')} "
            f"--deff-z-threshold {best_clean.get('deff_z_threshold')}"
            if best_clean
            else ""
        ),
        "artifacts": {
            "top_csv": str(top_csv),
            "full_csv": str(full_csv),
        },
    }
    out_json = outdir / "threshold_sweep_recommendation.json"
    out_json.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "ok", "recommendation_json": str(out_json), "top_csv": str(top_csv)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

