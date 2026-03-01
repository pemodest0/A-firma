#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


@dataclass
class EvalConfig:
    train_end: pd.Timestamp
    target_alert_rate: float
    horizon: int
    pre_min: int
    pre_max: int
    n_random: int
    seed: int
    freq: str  # daily | monthly


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        y = float(x)
    except (TypeError, ValueError):
        return float(default)
    return y if np.isfinite(y) else float(default)


def _f1(precision: float, recall: float) -> float:
    if (not np.isfinite(precision)) or (not np.isfinite(recall)) or (precision + recall <= 0):
        return float("nan")
    return float(2.0 * precision * recall / (precision + recall))


def _latest_regime_file(run_dir: Path) -> Path:
    run_meta = run_dir / "run_meta.json"
    off_window: int | None = None
    if run_meta.exists():
        try:
            meta = json.loads(run_meta.read_text(encoding="utf-8"))
            off_window = int(meta.get("official_window"))
        except Exception:
            off_window = None
    if off_window is not None:
        p = run_dir / f"regime_series_T{off_window}.csv"
        if p.exists():
            return p
    cands = sorted(run_dir.glob("regime_series_T*.csv"))
    if not cands:
        raise FileNotFoundError(f"missing regime_series_T*.csv in {run_dir}")
    best = cands[0]
    best_rows = -1
    for c in cands:
        try:
            n = int(sum(1 for _ in c.open("r", encoding="utf-8")) - 1)
        except Exception:
            n = -1
        if n > best_rows:
            best = c
            best_rows = n
    return best


def _load_events(catalog_path: Path) -> list[pd.Timestamp]:
    raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    out: list[pd.Timestamp] = []
    for row in raw.get("events", []):
        dt = pd.to_datetime((row or {}).get("date"), errors="coerce")
        if pd.isna(dt):
            continue
        out.append(pd.Timestamp(dt).normalize())
    out = sorted(set(out))
    if not out:
        raise ValueError(f"no valid events in catalog: {catalog_path}")
    return out


def _infer_freq(dates: pd.Series) -> str:
    d = pd.to_datetime(dates, errors="coerce").dropna().sort_values().drop_duplicates()
    if d.shape[0] < 3:
        return "daily"
    gaps = d.diff().dt.days.dropna()
    if gaps.empty:
        return "daily"
    med = float(gaps.median())
    return "monthly" if med >= 20.0 else "daily"


def _distance_to_next_event(d: pd.Timestamp, events: list[pd.Timestamp], *, freq: str) -> float:
    if freq == "monthly":
        vals: list[float] = []
        for e in events:
            if e <= d:
                continue
            m = int((e.year - d.year) * 12 + (e.month - d.month))
            if m > 0:
                vals.append(float(m))
        return min(vals) if vals else float("nan")
    vals = [float((e - d).days) for e in events if e > d]
    vals = [v for v in vals if v > 0]
    return min(vals) if vals else float("nan")


def _causal_z(
    dates: pd.Series,
    values: pd.Series,
    *,
    freq: str,
) -> pd.Series:
    x = pd.Series(pd.to_numeric(values, errors="coerce"), index=values.index, dtype=float)
    d = pd.to_datetime(dates, errors="coerce")
    out = pd.Series(np.nan, index=x.index, dtype=float)
    if freq == "monthly":
        lookback = pd.DateOffset(months=12)
        gap = pd.DateOffset(months=4)
        min_hist = 6
    else:
        lookback = pd.Timedelta(days=365)
        gap = pd.Timedelta(days=120)
        min_hist = 60
    for i in x.index:
        di = d.loc[i]
        if pd.isna(di):
            continue
        base = x[(d >= di - lookback) & (d <= di - gap)]
        base = base[np.isfinite(base)]
        vi = float(x.loc[i]) if np.isfinite(x.loc[i]) else float("nan")
        if (base.shape[0] < min_hist) or (not np.isfinite(vi)):
            continue
        mu = float(base.mean())
        sd = float(base.std(ddof=0))
        if (not np.isfinite(sd)) or (sd <= 1e-12):
            continue
        out.loc[i] = float((vi - mu) / sd)
    return out


def _score_modes(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    for col in [
        "z_p1",
        "z_deff_collapse",
        "z_eig_instability",
        "z_turnover",
        "z_lambda_gap",
        "z_forman_neg",
    ]:
        if col not in out.columns:
            out[col] = np.nan
    out["mode_concentration"] = (
        pd.concat([out["z_p1"], out["z_deff_collapse"]], axis=1).max(axis=1, skipna=True).astype(float)
    )
    out["mode_rotation"] = (
        pd.concat([out["z_eig_instability"], out["z_turnover"]], axis=1).max(axis=1, skipna=True).astype(float)
    )
    out["mode_spectral_gap"] = (
        0.50 * out["z_p1"].fillna(0.0)
        + 0.30 * out["z_lambda_gap"].fillna(0.0)
        + 0.20 * out["z_forman_neg"].fillna(0.0)
    )
    out["mode_topology_break"] = (
        0.70 * out["z_eig_instability"].fillna(0.0) + 0.30 * out["z_forman_neg"].fillna(0.0)
    )
    arr = np.vstack(
        [
            out["z_p1"].fillna(-np.inf).values,
            out["z_deff_collapse"].fillna(-np.inf).values,
            out["z_eig_instability"].fillna(-np.inf).values,
            out["z_turnover"].fillna(-np.inf).values,
            out["z_lambda_gap"].fillna(-np.inf).values,
        ]
    ).T
    arr = np.sort(arr, axis=1)
    top2 = arr[:, -2:]
    consensus = np.full(shape=(top2.shape[0],), fill_value=np.nan, dtype=float)
    for i in range(top2.shape[0]):
        row = top2[i, :]
        row = row[np.isfinite(row)]
        if row.size > 0:
            consensus[i] = float(np.mean(row))
    out["mode_consensus"] = consensus
    return out


def _fit_threshold(train_scores: pd.Series, target_alert_rate: float) -> float:
    s = pd.to_numeric(train_scores, errors="coerce")
    s = s[np.isfinite(s)]
    if s.empty:
        return float("nan")
    q = float(1.0 - min(0.80, max(0.02, target_alert_rate)))
    return float(np.quantile(s.values, q))


def _event_pre_signal_rate(
    dates: pd.Series,
    alerts: pd.Series,
    events: list[pd.Timestamp],
    *,
    cfg: EvalConfig,
) -> tuple[float, int, int]:
    x = pd.DataFrame(
        {
            "date": pd.to_datetime(dates, errors="coerce"),
            "alert": pd.Series(alerts).fillna(False).astype(bool),
        }
    ).dropna(subset=["date"])
    if x.empty:
        return float("nan"), 0, 0
    valid = 0
    hits = 0
    for e in events:
        if e <= cfg.train_end:
            continue
        if cfg.freq == "monthly":
            start = e - pd.DateOffset(months=int(cfg.pre_max))
            end = e - pd.DateOffset(months=int(cfg.pre_min))
        else:
            start = e - pd.Timedelta(days=int(cfg.pre_max))
            end = e - pd.Timedelta(days=int(cfg.pre_min))
        pre = x[(x["date"] >= start) & (x["date"] <= end)].copy()
        if pre.empty:
            continue
        valid += 1
        if bool(pre["alert"].any()):
            hits += 1
    rate = float(hits / valid) if valid > 0 else float("nan")
    return rate, hits, valid


def _random_metrics(
    y_event: np.ndarray,
    *,
    alert_rate: float,
    n_random: int,
    seed: int,
    dates: pd.Series,
    events: list[pd.Timestamp],
    cfg: EvalConfig,
) -> dict[str, float]:
    n = int(y_event.size)
    if n <= 0 or (not np.isfinite(alert_rate)):
        return {"precision_mean": float("nan"), "pre_signal_rate_mean": float("nan")}
    rng = np.random.default_rng(int(seed))
    p = float(min(0.95, max(0.0, alert_rate)))
    precs: list[float] = []
    pres: list[float] = []
    for _ in range(int(max(10, n_random))):
        ra = rng.random(n) < p
        tp = float(np.sum(ra & y_event))
        fp = float(np.sum(ra & (~y_event)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        pre_rate, _, _ = _event_pre_signal_rate(
            dates=dates,
            alerts=pd.Series(ra),
            events=events,
            cfg=cfg,
        )
        if np.isfinite(precision):
            precs.append(float(precision))
        if np.isfinite(pre_rate):
            pres.append(float(pre_rate))
    return {
        "precision_mean": float(np.mean(precs)) if precs else float("nan"),
        "pre_signal_rate_mean": float(np.mean(pres)) if pres else float("nan"),
    }


def _evaluate_mode(
    d: pd.DataFrame,
    *,
    score_col: str,
    events: list[pd.Timestamp],
    cfg: EvalConfig,
) -> dict[str, Any]:
    x = d.copy()
    score = pd.to_numeric(x.get(score_col), errors="coerce")
    x["score"] = score
    x = x[np.isfinite(x["score"])].copy()
    x = x.sort_values("date").reset_index(drop=True)
    if x.empty:
        return {
            "mode": score_col,
            "status": "no_data",
        }
    train = x[x["date"] <= cfg.train_end].copy()
    test = x[x["date"] > cfg.train_end].copy()
    thr = _fit_threshold(train["score"], target_alert_rate=cfg.target_alert_rate)
    if (not np.isfinite(thr)) or test.empty:
        return {
            "mode": score_col,
            "status": "no_test_or_threshold",
        }
    test["alert"] = test["score"] >= float(thr)
    y = test["y_event"].fillna(False).astype(bool).values
    a = test["alert"].fillna(False).astype(bool).values
    tp = int(np.sum(a & y))
    fp = int(np.sum(a & (~y)))
    fn = int(np.sum((~a) & y))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    f1 = _f1(precision, recall)
    event_rate = float(np.mean(y)) if y.size else float("nan")
    alert_rate = float(np.mean(a)) if a.size else float("nan")
    lift = float(precision / event_rate) if np.isfinite(precision) and np.isfinite(event_rate) and event_rate > 0 else float("nan")
    pre_rate, pre_hits, pre_valid = _event_pre_signal_rate(
        dates=test["date"],
        alerts=test["alert"],
        events=events,
        cfg=cfg,
    )
    rnd = _random_metrics(
        y_event=y,
        alert_rate=alert_rate,
        n_random=cfg.n_random,
        seed=cfg.seed,
        dates=test["date"],
        events=events,
        cfg=cfg,
    )
    lift_prec_vs_rand = (
        float(precision / rnd["precision_mean"])
        if np.isfinite(precision) and np.isfinite(rnd["precision_mean"]) and rnd["precision_mean"] > 0
        else float("nan")
    )
    lift_pre_vs_rand = (
        float(pre_rate / rnd["pre_signal_rate_mean"])
        if np.isfinite(pre_rate) and np.isfinite(rnd["pre_signal_rate_mean"]) and rnd["pre_signal_rate_mean"] > 0
        else float("nan")
    )
    return {
        "mode": score_col,
        "status": "ok",
        "threshold": float(thr),
        "train_rows": int(train.shape[0]),
        "test_rows": int(test.shape[0]),
        "test_event_rate": event_rate,
        "test_alert_rate": alert_rate,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "lift_vs_random": lift,
        "pre_signal_rate": pre_rate,
        "pre_signal_hits": int(pre_hits),
        "pre_signal_valid_events": int(pre_valid),
        "random_precision_mean": float(rnd["precision_mean"]),
        "random_pre_signal_mean": float(rnd["pre_signal_rate_mean"]),
        "lift_precision_vs_random": lift_prec_vs_rand,
        "lift_pre_signal_vs_random": lift_pre_vs_rand,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate event detection modes derived from correlation-matrix structure.")
    ap.add_argument("--run-dir", type=str, required=True)
    ap.add_argument("--event-catalog", type=str, required=True)
    ap.add_argument("--train-end", type=str, default="2022-12-31")
    ap.add_argument("--target-alert-rate", type=float, default=0.20)
    ap.add_argument("--horizon", type=int, default=0, help="0=auto (30d daily, 3m monthly)")
    ap.add_argument("--pre-min", type=int, default=0, help="0=auto (7d daily, 1m monthly)")
    ap.add_argument("--pre-max", type=int, default=0, help="0=auto (90d daily, 3m monthly)")
    ap.add_argument("--n-random", type=int, default=250)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--outdir", type=str, required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / str(args.run_dir)
    event_catalog = Path(args.event_catalog)
    if not event_catalog.is_absolute():
        event_catalog = ROOT / str(args.event_catalog)
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = ROOT / str(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    regime_path = _latest_regime_file(run_dir=run_dir)
    d = pd.read_csv(regime_path)
    if d.empty or ("date" not in d.columns):
        raise SystemExit(f"invalid regime file: {regime_path}")
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if d.empty:
        raise SystemExit("empty regime after date parsing")

    freq = _infer_freq(d["date"])
    horizon = int(args.horizon) if int(args.horizon) > 0 else (3 if freq == "monthly" else 30)
    pre_min = int(args.pre_min) if int(args.pre_min) > 0 else (1 if freq == "monthly" else 7)
    pre_max = int(args.pre_max) if int(args.pre_max) > 0 else (3 if freq == "monthly" else 90)

    cfg = EvalConfig(
        train_end=pd.to_datetime(args.train_end).normalize(),
        target_alert_rate=float(args.target_alert_rate),
        horizon=int(horizon),
        pre_min=int(pre_min),
        pre_max=int(pre_max),
        n_random=int(max(10, args.n_random)),
        seed=int(args.seed),
        freq=str(freq),
    )

    events = _load_events(catalog_path=event_catalog)
    for c in ["p1", "deff", "eigvec_instability_1d", "overlap_instability", "turnover_pair_frac", "lambda1", "lambda2", "forman_share_negative"]:
        if c not in d.columns:
            d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["deff_collapse"] = -1.0 * d["deff"]
    d["eig_instability"] = d["eigvec_instability_1d"]
    miss_eig = ~np.isfinite(d["eig_instability"])
    d.loc[miss_eig, "eig_instability"] = d.loc[miss_eig, "overlap_instability"]
    d["lambda_gap"] = d["lambda1"] / d["lambda2"].replace(0.0, np.nan)
    d["turnover"] = d["turnover_pair_frac"]
    if d["turnover"].notna().sum() == 0:
        d["turnover"] = d["eig_instability"]

    d["z_p1"] = _causal_z(d["date"], d["p1"], freq=freq)
    d["z_deff_collapse"] = _causal_z(d["date"], d["deff_collapse"], freq=freq)
    d["z_eig_instability"] = _causal_z(d["date"], d["eig_instability"], freq=freq)
    d["z_turnover"] = _causal_z(d["date"], d["turnover"], freq=freq)
    d["z_lambda_gap"] = _causal_z(d["date"], d["lambda_gap"], freq=freq)
    d["z_forman_neg"] = _causal_z(d["date"], d["forman_share_negative"], freq=freq)

    d = _score_modes(d)
    d["dist_next_event"] = d["date"].map(lambda x: _distance_to_next_event(x, events, freq=freq))
    d["y_event"] = (
        np.isfinite(d["dist_next_event"])
        & (d["dist_next_event"] > 0)
        & (d["dist_next_event"] <= int(cfg.horizon))
    )

    mode_cols = [
        "mode_concentration",
        "mode_rotation",
        "mode_spectral_gap",
        "mode_topology_break",
        "mode_consensus",
    ]
    rows: list[dict[str, Any]] = []
    for m in mode_cols:
        rows.append(_evaluate_mode(d, score_col=m, events=events, cfg=cfg))
    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        ["status", "lift_pre_signal_vs_random", "lift_precision_vs_random", "recall", "f1"],
        ascending=[True, False, False, False, False],
    ).reset_index(drop=True)

    details_cols = ["date", "y_event", "dist_next_event"] + mode_cols + [f"z_{c}" for c in ["p1", "deff_collapse", "eig_instability", "turnover", "lambda_gap", "forman_neg"]]
    details = d[[c for c in details_cols if c in d.columns]].copy()

    summary_csv = outdir / "corr_event_modes_summary.csv"
    details_csv = outdir / "corr_event_modes_details.csv"
    summary.to_csv(summary_csv, index=False)
    details.to_csv(details_csv, index=False)

    payload = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "regime_file": str(regime_path),
        "event_catalog": str(event_catalog),
        "freq": cfg.freq,
        "train_end": str(cfg.train_end.date()),
        "target_alert_rate": float(cfg.target_alert_rate),
        "horizon": int(cfg.horizon),
        "pre_min": int(cfg.pre_min),
        "pre_max": int(cfg.pre_max),
        "n_random": int(cfg.n_random),
        "seed": int(cfg.seed),
        "best_mode": summary.iloc[0].to_dict() if not summary.empty else {},
        "artifacts": {
            "summary_csv": str(summary_csv),
            "details_csv": str(details_csv),
        },
    }
    out_json = outdir / "corr_event_modes_eval.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "summary_csv": str(summary_csv),
                "eval_json": str(out_json),
                "best_mode": payload.get("best_mode", {}).get("mode", ""),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
