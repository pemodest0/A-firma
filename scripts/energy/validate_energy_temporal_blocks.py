#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = ROOT / "config" / "event_catalog_energy_br.json"


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


def _build_alerts(df: pd.DataFrame, *, score_thr: float, phi_thr: float, deff_thr: float) -> pd.DataFrame:
    x = df.copy()
    x["z_score"] = np.nan
    x["z_phi"] = np.nan
    x["z_deff"] = np.nan
    x["alert"] = False
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
        z_score = zvals["score"]
        z_phi = zvals["phi"]
        z_deff = zvals["deff"]
        alert = bool(
            (np.isfinite(z_score) and z_score >= float(score_thr))
            or (np.isfinite(z_phi) and z_phi >= float(phi_thr))
            or (np.isfinite(z_deff) and z_deff <= float(deff_thr))
        )
        x.at[i, "z_score"] = z_score
        x.at[i, "z_phi"] = z_phi
        x.at[i, "z_deff"] = z_deff
        x.at[i, "alert"] = alert
    return x


def _parse_blocks(text: str) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    out: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for part in str(text).split(","):
        s = part.strip()
        if not s:
            continue
        try:
            a, b = s.split(":")
        except ValueError as exc:
            raise ValueError(f"bloco invalido: {s}") from exc
        start = pd.to_datetime(a.strip(), errors="coerce")
        end = pd.to_datetime(b.strip(), errors="coerce")
        if pd.isna(start) or pd.isna(end) or end < start:
            raise ValueError(f"bloco invalido: {s}")
        label = f"{start.strftime('%Y')}-{end.strftime('%Y')}"
        out.append((label, start.normalize(), end.normalize()))
    if not out:
        raise ValueError("nenhum bloco valido")
    return out


def _days_to_next_event(d: pd.Timestamp, events: list[pd.Timestamp]) -> float:
    deltas = [(e - d).days for e in events if e > d]
    deltas = [float(v) for v in deltas if v > 0]
    return min(deltas) if deltas else float("nan")


def _safe_div(a: float, b: float) -> float:
    if b == 0 or not np.isfinite(b):
        return float("nan")
    return float(a / b)


def _f1(precision: float, recall: float) -> float:
    if not np.isfinite(precision) or not np.isfinite(recall) or (precision + recall) <= 0:
        return float("nan")
    return float(2 * precision * recall / (precision + recall))


def main() -> None:
    ap = argparse.ArgumentParser(description="Validação temporal do Energia BR por blocos com recall/lift.")
    ap.add_argument("--run-dir", type=str, required=True)
    ap.add_argument("--event-catalog", type=str, default=str(DEFAULT_CATALOG))
    ap.add_argument("--score-z-threshold", type=float, default=0.0)
    ap.add_argument("--phi-z-threshold", type=float, default=1.2)
    ap.add_argument("--deff-z-threshold", type=float, default=-1.0)
    ap.add_argument("--horizon-days", type=int, default=30)
    ap.add_argument(
        "--blocks",
        type=str,
        default="2018-01-01:2019-12-31,2020-01-01:2022-12-31,2023-01-01:2026-12-31",
    )
    ap.add_argument("--outdir", type=str, default="results/energy_br/latest/temporal_validation")
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

    blocks = _parse_blocks(args.blocks)
    series = _load_series(run_dir=run_dir)
    events = _load_events(event_catalog)
    data = _build_alerts(
        series,
        score_thr=float(args.score_z_threshold),
        phi_thr=float(args.phi_z_threshold),
        deff_thr=float(args.deff_z_threshold),
    )
    data["days_to_next_event"] = data["date"].map(lambda d: _days_to_next_event(d, events))
    data["y_event"] = (
        (np.isfinite(data["days_to_next_event"]))
        & (data["days_to_next_event"] > 0)
        & (data["days_to_next_event"] <= int(args.horizon_days))
    )
    data["eligible"] = np.isfinite(data["z_score"]) | np.isfinite(data["z_phi"]) | np.isfinite(data["z_deff"])
    eval_df = data[data["eligible"]].copy().reset_index(drop=True)

    summary_rows: list[dict[str, object]] = []
    for label, start, end in blocks:
        d = eval_df[(eval_df["date"] >= start) & (eval_df["date"] <= end)].copy()
        if d.empty:
            summary_rows.append(
                {
                    "block": label,
                    "start": start.strftime("%Y-%m-%d"),
                    "end": end.strftime("%Y-%m-%d"),
                    "n_rows": 0,
                    "n_events": 0,
                    "event_rate": float("nan"),
                    "n_alerts": 0,
                    "alert_rate": float("nan"),
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "f1": float("nan"),
                    "lift_vs_random": float("nan"),
                    "median_lead_days_tp": float("nan"),
                }
            )
            continue
        y = d["y_event"].astype(bool).values
        a = d["alert"].fillna(False).astype(bool).values
        tp = int(np.sum(a & y))
        fp = int(np.sum(a & (~y)))
        fn = int(np.sum((~a) & y))
        precision = _safe_div(float(tp), float(tp + fp))
        recall = _safe_div(float(tp), float(tp + fn))
        f1 = _f1(precision, recall)
        event_rate = float(np.mean(y)) if y.size else float("nan")
        alert_rate = float(np.mean(a)) if a.size else float("nan")
        lift = _safe_div(precision, event_rate) if np.isfinite(precision) and np.isfinite(event_rate) else float("nan")
        tp_leads = d.loc[(d["alert"] == True) & (d["y_event"] == True), "days_to_next_event"]
        median_lead = float(tp_leads.median()) if not tp_leads.empty else float("nan")
        summary_rows.append(
            {
                "block": label,
                "start": start.strftime("%Y-%m-%d"),
                "end": end.strftime("%Y-%m-%d"),
                "n_rows": int(d.shape[0]),
                "n_events": int(np.sum(y)),
                "event_rate": event_rate,
                "n_alerts": int(np.sum(a)),
                "alert_rate": alert_rate,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "lift_vs_random": lift,
                "median_lead_days_tp": median_lead,
            }
        )

    summary = pd.DataFrame(summary_rows)
    details_csv = outdir / "temporal_validation_details.csv"
    summary_csv = outdir / "temporal_validation_summary.csv"
    summary_json = outdir / "temporal_validation_summary.json"
    eval_df.to_csv(details_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    payload = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "thresholds": {
            "score_z_threshold": float(args.score_z_threshold),
            "phi_z_threshold": float(args.phi_z_threshold),
            "deff_z_threshold": float(args.deff_z_threshold),
            "horizon_days": int(args.horizon_days),
        },
        "blocks": summary_rows,
        "artifacts": {
            "summary_csv": str(summary_csv),
            "details_csv": str(details_csv),
        },
    }
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "ok", "summary_json": str(summary_json), "summary_csv": str(summary_csv)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

