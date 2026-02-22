#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MIN_POINTS = 200
MIN_LONG_POINTS = 2000
OUTDIR_DEFAULT = ROOT / "results" / "validation" / "historical_shifts"


@dataclass
class EventRef:
    name: str
    date: pd.Timestamp


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_status(outdir: Path, payload: dict[str, Any]) -> None:
    _json_dump(outdir / "STATUS.json", payload)


def _detect_dataset(user_path: str | None) -> tuple[Path, str, str]:
    if user_path:
        p = Path(user_path)
        return p, "close", "date"
    candidates = [
        (ROOT / "data" / "raw" / "finance" / "yfinance_daily" / "^VIX.csv", "close", "date"),
        (ROOT / "data" / "raw" / "finance" / "yfinance_daily" / "VIX.csv", "close", "date"),
        (ROOT / "data" / "raw" / "finance" / "yfinance_daily" / "SPY.csv", "close", "date"),
        (ROOT / "data" / "raw" / "finance" / "yfinance_daily" / "QQQ.csv", "close", "date"),
    ]
    for p, vc, dc in candidates:
        if p.exists():
            return p, vc, dc
    root = ROOT / "data" / "raw" / "finance" / "yfinance_daily"
    files = sorted(root.glob("*.csv")) if root.exists() else []
    if not files:
        raise FileNotFoundError("Nenhum dataset financeiro encontrado para historical_shifts")
    best = max(files, key=lambda x: x.stat().st_size)
    return best, "close", "date"


def _resolve_col(df: pd.DataFrame, col: str | None) -> str | None:
    if col is None:
        return None
    if col in df.columns:
        return col
    lower = col.lower()
    for c in df.columns:
        if c.lower() == lower:
            return c
    return None


def _auto_value_col(df: pd.DataFrame) -> str | None:
    for cand in ["close", "price", "adj_close", "value", "valor"]:
        found = _resolve_col(df, cand)
        if found is not None:
            return found
    return None


def _auto_date_col(df: pd.DataFrame) -> str | None:
    for cand in ["date", "data", "datetime", "timestamp", "time"]:
        found = _resolve_col(df, cand)
        if found is not None:
            return found
    return None


def _historical_events() -> list[EventRef]:
    return [
        EventRef("asia_ltcM_1998", pd.Timestamp("1998-09-01")),
        EventRef("dotcom_2001", pd.Timestamp("2001-09-01")),
        EventRef("gfc_2008", pd.Timestamp("2008-10-01")),
        EventRef("euro_debt_2011", pd.Timestamp("2011-08-01")),
        EventRef("covid_2020", pd.Timestamp("2020-03-15")),
        EventRef("tightening_2022", pd.Timestamp("2022-06-15")),
    ]


def _run_motor(values: np.ndarray, seed: int, timeframe: str) -> Any:
    from engine.graph.core import run_graph_engine  # type: ignore
    from engine.graph.embedding import estimate_embedding_params  # type: ignore

    m, tau = estimate_embedding_params(values, max_tau=20, max_m=6)
    result = run_graph_engine(
        values,
        m=m,
        tau=tau,
        n_micro=80,
        n_regimes=4,
        k_nn=5,
        theiler=10,
        alpha=2.0,
        seed=seed,
        timeframe=timeframe,
    )
    return result, m, tau


def _normalize_entropy(ent: pd.Series) -> pd.Series:
    if ent.isna().all():
        return pd.Series(np.zeros(len(ent)), index=ent.index)
    lo = float(ent.min())
    hi = float(ent.max())
    if hi - lo < 1e-12:
        return pd.Series(np.zeros(len(ent)), index=ent.index)
    return ((ent - lo) / (hi - lo)).clip(0, 1)


def _pick_peaks_per_year(df: pd.DataFrame, k: int, min_gap_days: int) -> pd.DataFrame:
    picks: list[pd.Series] = []
    for _, yg in df.groupby(df["date"].dt.year):
        yg = yg.sort_values("delta_instability", ascending=False)
        chosen_dates: list[pd.Timestamp] = []
        for _, row in yg.iterrows():
            d = row["date"]
            if not chosen_dates:
                picks.append(row)
                chosen_dates.append(d)
                if len(chosen_dates) >= k:
                    break
                continue
            if all(abs((d - cd).days) >= min_gap_days for cd in chosen_dates):
                picks.append(row)
                chosen_dates.append(d)
                if len(chosen_dates) >= k:
                    break
    if not picks:
        return df.iloc[0:0].copy()
    return pd.DataFrame(picks).sort_values("date").drop_duplicates(subset=["date"], keep="first")


def _expand_block(df: pd.DataFrame, peak_idx: int, q_expand: float, max_duration: int) -> tuple[int, int, float]:
    n = len(df)
    left_w = max(0, peak_idx - 120)
    right_w = min(n - 1, peak_idx + 120)
    local_vals = df.iloc[left_w : right_w + 1]["delta_instability"].to_numpy(dtype=float)
    thr = float(np.quantile(local_vals, q_expand)) if local_vals.size else float(df.iloc[peak_idx]["delta_instability"])

    left = peak_idx
    steps = 0
    while left - 1 >= 0 and float(df.iloc[left - 1]["delta_instability"]) >= thr and steps < max_duration:
        left -= 1
        steps += 1

    right = peak_idx
    while right + 1 < n and float(df.iloc[right + 1]["delta_instability"]) >= thr and (right - left + 1) < max_duration:
        right += 1

    return left, right, thr


def _build_macro_blocks(
    df: pd.DataFrame,
    micro_switch_idx: np.ndarray,
    k: int,
    peak_min_gap_days: int,
    merge_gap_days: int,
    q_expand: float,
    min_duration: int,
    max_duration: int,
) -> pd.DataFrame:
    peaks = _pick_peaks_per_year(df, k=k, min_gap_days=peak_min_gap_days)
    blocks: list[dict[str, Any]] = []
    for _, p in peaks.iterrows():
        peak_date = pd.Timestamp(p["date"])
        peak_idx = int(p["t"])
        left, right, thr = _expand_block(df, peak_idx, q_expand=q_expand, max_duration=max_duration)
        duration = int(right - left + 1)
        if duration < min_duration:
            right = min(len(df) - 1, left + min_duration - 1)
        blocks.append(
            {
                "start_idx": left,
                "end_idx": right,
                "peak_idx": peak_idx,
                "peak_date": peak_date,
                "peak_delta": float(df.iloc[peak_idx]["delta_instability"]),
                "threshold_expand": float(thr),
            }
        )

    if not blocks:
        return pd.DataFrame(
            columns=[
                "start_date",
                "end_date",
                "duration_days",
                "block_type",
                "peak_date",
                "peak_delta",
                "mean_instability",
                "mean_confidence",
                "mean_quality",
                "n_micro_switches_inside",
            ]
        )

    blocks = sorted(blocks, key=lambda x: (x["start_idx"], x["end_idx"]))
    merged: list[dict[str, Any]] = [blocks[0].copy()]
    for b in blocks[1:]:
        cur = merged[-1]
        gap = int(b["start_idx"] - cur["end_idx"] - 1)
        if gap <= merge_gap_days:
            cur["end_idx"] = max(cur["end_idx"], b["end_idx"])
            if b["peak_delta"] > cur["peak_delta"]:
                cur["peak_idx"] = b["peak_idx"]
                cur["peak_date"] = b["peak_date"]
                cur["peak_delta"] = b["peak_delta"]
            cur["threshold_expand"] = max(float(cur.get("threshold_expand", 0.0)), float(b.get("threshold_expand", 0.0)))
        else:
            merged.append(b.copy())

    rows: list[dict[str, Any]] = []
    for b in merged:
        s = int(b["start_idx"])
        e = int(b["end_idx"])
        g = df.iloc[s : e + 1]
        duration = int((g["date"].iloc[-1] - g["date"].iloc[0]).days) + 1
        block_type = "macro_long_regime" if duration > 365 else "macro_event"
        inside = int(((micro_switch_idx >= s) & (micro_switch_idx <= e)).sum())
        rows.append(
            {
                "start_date": g["date"].iloc[0].strftime("%Y-%m-%d"),
                "end_date": g["date"].iloc[-1].strftime("%Y-%m-%d"),
                "duration_days": duration,
                "block_type": block_type,
                "peak_date": pd.Timestamp(b["peak_date"]).strftime("%Y-%m-%d"),
                "peak_delta": float(b["peak_delta"]),
                "mean_instability": float(g["instability_score"].mean()),
                "mean_confidence": float(g["confidence"].mean()),
                "mean_quality": float(g["quality"].mean()),
                "n_micro_switches_inside": inside,
            }
        )
    return pd.DataFrame(rows)


def _interval_intersects(a0: pd.Timestamp, a1: pd.Timestamp, b0: pd.Timestamp, b1: pd.Timestamp) -> bool:
    return not (a1 < b0 or b1 < a0)


def _calc_density_ratio(macro_events: pd.DataFrame, event_windows: list[tuple[pd.Timestamp, pd.Timestamp]], data_start: pd.Timestamp, data_end: pd.Timestamp) -> float:
    total_days = max(1, int((data_end - data_start).days) + 1)
    mask = pd.Series(False, index=pd.date_range(data_start, data_end, freq="D"))
    for s, e in event_windows:
        s2 = max(s, data_start)
        e2 = min(e, data_end)
        if s2 <= e2:
            mask.loc[s2:e2] = True

    in_days = int(mask.sum())
    out_days = max(1, total_days - in_days)

    in_count = 0
    out_count = 0
    for _, b in macro_events.iterrows():
        s = pd.Timestamp(b["start_date"])
        e = pd.Timestamp(b["end_date"])
        hit = any(_interval_intersects(s, e, w0, w1) for w0, w1 in event_windows)
        if hit:
            in_count += 1
        else:
            out_count += 1

    dens_in = in_count / max(1, in_days)
    dens_out = out_count / max(1, out_days)
    if dens_out <= 0:
        return float("inf") if dens_in > 0 else 0.0
    return float(dens_in / dens_out)


def _metrics_for_blocks(df: pd.DataFrame, blocks: pd.DataFrame, events: list[EventRef]) -> tuple[dict[str, Any], pd.DataFrame]:
    data_start = pd.Timestamp(df["date"].min())
    data_end = pd.Timestamp(df["date"].max())

    macro_events = blocks[blocks["block_type"] == "macro_event"].copy()
    event_windows = [(ev.date - pd.Timedelta(days=90), ev.date + pd.Timedelta(days=90)) for ev in events]

    rows = []
    hits = 0
    deltas = []

    for ev in events:
        nearest_date = None
        nearest_delta = None
        det_type = "none"
        for _, b in macro_events.iterrows():
            s = pd.Timestamp(b["start_date"])
            e = pd.Timestamp(b["end_date"])
            mid = s + (e - s) / 2
            d = int((mid - ev.date).days)
            if nearest_delta is None or abs(d) < abs(nearest_delta):
                nearest_delta = d
                nearest_date = mid
                det_type = "macro_event"
        w0 = ev.date - pd.Timedelta(days=90)
        w1 = ev.date + pd.Timedelta(days=90)
        has_hit = any(
            _interval_intersects(pd.Timestamp(b["start_date"]), pd.Timestamp(b["end_date"]), w0, w1)
            for _, b in macro_events.iterrows()
        )
        if has_hit:
            hits += 1
        if nearest_delta is not None:
            deltas.append(nearest_delta)
        rows.append(
            {
                "event_name": ev.name,
                "event_date": ev.date.strftime("%Y-%m-%d"),
                "nearest_detection_date": nearest_date.strftime("%Y-%m-%d") if nearest_date is not None else "",
                "delta_days": int(nearest_delta) if nearest_delta is not None else None,
                "detection_type": det_type,
            }
        )

    n_years = max(1e-9, (data_end - data_start).days / 365.25)
    density_ratio = _calc_density_ratio(macro_events, event_windows, data_start, data_end)
    lead_rate = float(sum(1 for d in deltas if d < 0) / len(deltas)) if deltas else None
    mean_delta_signed = float(np.mean(deltas)) if deltas else None
    metrics = {
        "n_macro_blocks_total": int(blocks.shape[0]),
        "n_macro_events": int(macro_events.shape[0]),
        "n_macro_long_regimes": int((blocks["block_type"] == "macro_long_regime").sum()) if not blocks.empty else 0,
        "hit_rate_macro": float(hits / len(events)) if events else 0.0,
        "avg_delta_days_macro": float(np.mean(np.abs(deltas))) if deltas else None,
        "mean_delta_signed_days_macro": mean_delta_signed,
        "lead_rate_macro": lead_rate,
        "density_ratio_macro": float(density_ratio),
        "macro_block_rate_per_year": float(macro_events.shape[0] / n_years),
    }
    return metrics, pd.DataFrame(rows)


def _score_candidate(metrics: dict[str, Any]) -> float:
    hit = float(metrics.get("hit_rate_macro") or 0.0)
    dens = float(metrics.get("density_ratio_macro") or 0.0)
    lead = float(metrics.get("lead_rate_macro") or 0.0)
    dens_cap = min(max(dens, 0.0), 2.0) / 2.0
    return 0.45 * hit + 0.35 * dens_cap + 0.20 * lead


def _verdict(m: dict[str, Any], used_k: int) -> dict[str, Any]:
    hit = float(m.get("hit_rate_macro", 0.0))
    dens = float(m.get("density_ratio_macro", 0.0))
    pseudo = bool(m.get("pseudo_bifurcation_flag", False))
    if hit >= 0.5 and dens > 1.0:
        status = "pass"
    elif hit >= 0.35 and dens > 0.8:
        status = "neutral"
    else:
        status = "fail"
    if pseudo and status == "pass":
        status = "neutral"
    return {
        "status": status,
        "rule": "pass if hit_rate_macro >= 0.5 and density_ratio_macro > 1.0",
        "used_K": int(used_k),
        "hit_rate_macro": hit,
        "density_ratio_macro": dens,
        "pseudo_bifurcation_flag": pseudo,
    }


def _working_with_delta(base_working: pd.DataFrame, w: int) -> pd.DataFrame:
    working = base_working.copy()
    inst = working["instability_score"].astype(float)
    rolling_med = inst.rolling(window=w, min_periods=max(10, w // 4)).median()
    rolling_med = rolling_med.fillna(inst.expanding(min_periods=1).median())
    working["delta_instability"] = (inst - rolling_med).abs().to_numpy(dtype=float)
    return working


def _micro_switch_idx(working: pd.DataFrame) -> np.ndarray:
    labels = working["regime_label"].astype(str).to_numpy()
    if labels.size <= 1:
        return np.array([], dtype=int)
    return np.where(labels[1:] != labels[:-1])[0] + 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Historical regime shift validation with calibrated macro-events.")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--value-col", type=str, default="close")
    parser.add_argument("--date-col", type=str, default="date")
    parser.add_argument("--outdir", type=str, default=str(OUTDIR_DEFAULT))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--w", type=int, default=60)
    parser.add_argument("--timeframe", type=str, default="daily", choices=["daily", "weekly"])
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--calibration-ratio", type=float, default=0.70)
    parser.add_argument("--k-grid", type=str, default="1,2,3")
    parser.add_argument("--w-grid", type=str, default="45,60,90")
    parser.add_argument("--q-expand-grid", type=str, default="0.55,0.60,0.65")
    parser.add_argument("--peak-min-gap-days", type=int, default=30)
    parser.add_argument("--merge-gap-days", type=int, default=20)
    parser.add_argument("--min-duration", type=int, default=5)
    parser.add_argument("--max-duration", type=int, default=180)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dataset, vc_auto, dc_auto = _detect_dataset(args.dataset)
    if not dataset.exists():
        _write_status(outdir, {"status": "fail", "reason": f"missing_data: {dataset}"})
        print(f"[fail] dataset not found: {dataset}")
        return

    try:
        df = pd.read_csv(dataset)
    except Exception as exc:
        _write_status(outdir, {"status": "fail", "reason": f"read_error: {exc}"})
        print(f"[fail] read_error: {exc}")
        return

    value_col = _resolve_col(df, args.value_col) or _resolve_col(df, vc_auto) or _auto_value_col(df)
    date_col = _resolve_col(df, args.date_col) or _resolve_col(df, dc_auto) or _auto_date_col(df)
    if value_col is None or date_col is None:
        _write_status(outdir, {"status": "fail", "reason": "missing required columns"})
        print("[fail] missing required columns")
        return

    x = pd.to_numeric(df[value_col], errors="coerce")
    d = pd.to_datetime(df[date_col], errors="coerce")
    valid = x.notna() & d.notna()
    x = x[valid].to_numpy(dtype=float)
    d = d[valid].reset_index(drop=True)

    if x.shape[0] < MIN_POINTS:
        _write_status(outdir, {"status": "fail", "reason": f"insufficient_points<{MIN_POINTS}", "n_points": int(x.shape[0])})
        print(f"[fail] insufficient points: {x.shape[0]}")
        return

    result, m, tau = _run_motor(x, seed=args.seed, timeframe=args.timeframe)
    n = int(result.state_labels.shape[0])
    if n <= 0:
        _write_status(outdir, {"status": "fail", "reason": "empty_motor_output"})
        print("[fail] empty motor output")
        return

    dates = pd.to_datetime(d.iloc[len(d) - n :].to_numpy())
    labels = result.state_labels.astype(str)
    conf = pd.Series(result.confidence.astype(float))
    quality_scalar = float((result.quality or {}).get("score", np.nan))
    quality = pd.Series(np.repeat(quality_scalar, n), dtype=float)
    entropy_scalar = float((result.quality or {}).get("entropy_rate", np.nan))
    entropy = pd.Series(np.repeat(entropy_scalar, n), dtype=float)
    entropy_norm = _normalize_entropy(entropy)

    regime_order: list[str] = []
    for lbl in labels.tolist():
        if lbl not in regime_order:
            regime_order.append(lbl)
    rid = {k: i for i, k in enumerate(regime_order)}

    regimes = pd.DataFrame(
        {
            "t": np.arange(n, dtype=int),
            "date": dates.strftime("%Y-%m-%d"),
            "regime_id": [rid[str(v)] for v in labels],
            "regime_label": labels,
            "confidence": conf,
            "quality": quality,
            "entropy": entropy,
        }
    )

    instability = ((1.0 - regimes["confidence"]) + (1.0 - regimes["quality"]) + entropy_norm).clip(0, 1)
    regimes["instability_score"] = instability

    micro_switch = np.where(labels[1:] != labels[:-1])[0] + 1
    micro_df = pd.DataFrame(
        {
            "date": dates[micro_switch].strftime("%Y-%m-%d"),
            "type": "regime_change",
            "from_label": labels[micro_switch - 1],
            "to_label": labels[micro_switch],
            "delta_instability": np.nan,
        }
    )

    base_working = pd.DataFrame(
        {
            "t": np.arange(n, dtype=int),
            "date": pd.to_datetime(regimes["date"]),
            "regime_label": labels,
            "confidence": conf.to_numpy(dtype=float),
            "quality": quality.to_numpy(dtype=float),
            "instability_score": instability.to_numpy(dtype=float),
        }
    )

    events = _historical_events()
    _json_dump(outdir / "historical_events.json", {"events": [{"event_name": e.name, "event_date": e.date.strftime("%Y-%m-%d")} for e in events]})

    k_grid = [int(x.strip()) for x in args.k_grid.split(",") if x.strip()]
    w_grid = [int(x.strip()) for x in args.w_grid.split(",") if x.strip()]
    q_grid = [float(x.strip()) for x in args.q_expand_grid.split(",") if x.strip()]
    if not args.calibrate:
        k_grid = [2]
        w_grid = [args.w]
        q_grid = [0.60]

    split_date: pd.Timestamp | None = None
    cal_frame = base_working.copy()
    eval_frame = base_working.copy()
    events_cal = list(events)
    events_eval = list(events)
    if args.calibrate:
        ratio = float(np.clip(args.calibration_ratio, 0.50, 0.90))
        split_idx = int(len(base_working) * ratio)
        split_idx = max(252, split_idx)
        split_idx = min(split_idx, max(253, len(base_working) - 120))
        if split_idx < len(base_working):
            split_date = pd.Timestamp(base_working.iloc[split_idx]["date"])
            cal_frame = base_working[base_working["date"] <= split_date].copy()
            eval_frame = base_working[base_working["date"] > split_date].copy()
            events_cal = [ev for ev in events if ev.date <= split_date]
            events_eval = [ev for ev in events if ev.date > split_date]
        # Fallback keeps the run deterministic and avoids empty holdout.
        if cal_frame.shape[0] < 252 or eval_frame.shape[0] < 120 or len(events_cal) < 2 or len(events_eval) < 2:
            if len(events) >= 4:
                split_date = events[(len(events) // 2) - 1].date
                cal_frame = base_working[base_working["date"] <= split_date].copy()
                eval_frame = base_working[base_working["date"] > split_date].copy()
                events_cal = [ev for ev in events if ev.date <= split_date]
                events_eval = [ev for ev in events if ev.date > split_date]
        if cal_frame.empty or eval_frame.empty:
            split_date = None
            cal_frame = base_working.copy()
            eval_frame = base_working.copy()
            events_cal = list(events)
            events_eval = list(events)
        _json_dump(
            outdir / "calibration_split.json",
            {
                "status": "ok",
                "calibrate_mode": True,
                "split_date": split_date.strftime("%Y-%m-%d") if split_date is not None else "",
                "calibration_rows": int(cal_frame.shape[0]),
                "evaluation_rows": int(eval_frame.shape[0]),
                "events_calibration": int(len(events_cal)),
                "events_evaluation": int(len(events_eval)),
            },
        )

    sweep_rows: list[dict[str, Any]] = []
    best_payload: dict[str, Any] | None = None
    best_by_k: dict[int, dict[str, Any]] = {}

    for w in w_grid:
        working_cal = _working_with_delta(cal_frame, w=w)
        micro_switch_cal = _micro_switch_idx(working_cal)

        for k in k_grid:
            for q_expand in q_grid:
                blocks_cal = _build_macro_blocks(
                    working_cal,
                    micro_switch_idx=micro_switch_cal,
                    k=k,
                    peak_min_gap_days=args.peak_min_gap_days,
                    merge_gap_days=args.merge_gap_days,
                    q_expand=q_expand,
                    min_duration=args.min_duration,
                    max_duration=args.max_duration,
                )
                metrics_cal, _ = _metrics_for_blocks(working_cal, blocks_cal, events_cal)
                score_cal = _score_candidate(metrics_cal)
                row = {
                    "w": w,
                    "k": k,
                    "q_expand": q_expand,
                    "score": score_cal,
                    "hit_rate_macro": metrics_cal.get("hit_rate_macro"),
                    "density_ratio_macro": metrics_cal.get("density_ratio_macro"),
                    "lead_rate_macro": metrics_cal.get("lead_rate_macro"),
                    "n_macro_events": metrics_cal.get("n_macro_events"),
                }
                sweep_rows.append(row)
                if best_payload is None or score_cal > float(best_payload["score"]):
                    best_payload = {
                        "w": w,
                        "k": k,
                        "q_expand": q_expand,
                        "score": score_cal,
                        "metrics_cal": metrics_cal.copy(),
                    }
                current = best_by_k.get(int(k))
                if current is None or score_cal > float(current["score"]):
                    best_by_k[int(k)] = {
                        "w": w,
                        "k": k,
                        "q_expand": q_expand,
                        "score": score_cal,
                        "metrics_cal": metrics_cal.copy(),
                    }

    if best_payload is None:
        _write_status(outdir, {"status": "fail", "reason": "no_sweep_candidates"})
        print("[fail] no sweep candidates")
        return

    sweep_df = pd.DataFrame(sweep_rows).sort_values("score", ascending=False)
    sweep_df.to_csv(outdir / "calibration_grid.csv", index=False)

    # Rule: report sensitivity for K=1/2/3, but use K=2 as primary verdict when available.
    chosen = best_by_k.get(2, best_payload)
    selected_w = int(chosen["w"])
    selected_k = int(chosen["k"])
    selected_q = float(chosen["q_expand"])

    working_full = _working_with_delta(base_working, w=selected_w)
    regimes["delta_instability"] = working_full["delta_instability"].to_numpy(dtype=float)
    regimes.to_csv(outdir / "regimes.csv", index=False)

    micro_df["delta_instability"] = regimes.iloc[micro_switch]["delta_instability"].to_numpy(dtype=float)
    micro_df.to_csv(outdir / "micro_transitions.csv", index=False)

    working_cal_best = _working_with_delta(cal_frame, w=selected_w)
    micro_switch_cal_best = _micro_switch_idx(working_cal_best)
    blocks_cal_best = _build_macro_blocks(
        working_cal_best,
        micro_switch_idx=micro_switch_cal_best,
        k=selected_k,
        peak_min_gap_days=args.peak_min_gap_days,
        merge_gap_days=args.merge_gap_days,
        q_expand=selected_q,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )
    metrics_cal_best_live, align_cal_best = _metrics_for_blocks(working_cal_best, blocks_cal_best, events_cal)

    working_eval_best = _working_with_delta(eval_frame, w=selected_w)
    micro_switch_eval = _micro_switch_idx(working_eval_best)
    blocks_eval_best = _build_macro_blocks(
        working_eval_best,
        micro_switch_idx=micro_switch_eval,
        k=selected_k,
        peak_min_gap_days=args.peak_min_gap_days,
        merge_gap_days=args.merge_gap_days,
        q_expand=selected_q,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )
    metrics_eval_best, align_eval_best = _metrics_for_blocks(working_eval_best, blocks_eval_best, events_eval)
    eval_score = _score_candidate(metrics_eval_best)

    blocks_cal_best.to_csv(outdir / "macro_blocks_calibration.csv", index=False)
    align_cal_best.to_csv(outdir / "alignment_calibration.csv", index=False)
    blocks_eval_best.to_csv(outdir / "macro_blocks.csv", index=False)
    align_eval_best.to_csv(outdir / "alignment.csv", index=False)

    metrics_by_k: dict[str, Any] = {}
    for k in sorted(set(k_grid)):
        b = best_by_k.get(int(k))
        if b is not None:
            mk = b.get("metrics_cal", {})
            metrics_by_k[f"K{k}"] = {
                "best_w": int(b["w"]),
                "best_q_expand": float(b["q_expand"]),
                "score_calibration": float(b["score"]),
                "hit_rate_macro": float(mk.get("hit_rate_macro", 0.0)),
                "density_ratio_macro": float(mk.get("density_ratio_macro", 0.0)),
                "lead_rate_macro": float(mk.get("lead_rate_macro")) if mk.get("lead_rate_macro") is not None else None,
                "n_macro_events": int(mk.get("n_macro_events", 0)),
            }
    _json_dump(outdir / "metrics_by_K.json", metrics_by_k)

    micro_years = max(
        1e-9,
        (
            pd.Timestamp(working_eval_best["date"].max())
            - pd.Timestamp(working_eval_best["date"].min())
        ).days
        / 365.25,
    )
    micro_switch_rate = float(len(micro_switch_eval) / micro_years) if micro_years > 0 else 0.0

    event_windows = [(e.date - pd.Timedelta(days=90), e.date + pd.Timedelta(days=90)) for e in events_eval]
    micro_hits = 0
    for idx in micro_switch_eval.tolist():
        dt = pd.Timestamp(working_eval_best.iloc[idx]["date"])
        if any((dt >= w0) and (dt <= w1) for w0, w1 in event_windows):
            micro_hits += 1
    total_micro = max(1, len(micro_switch_eval))
    density_ratio_micro = float((micro_hits / total_micro) / max(1e-9, (1 - micro_hits / total_micro))) if micro_hits < total_micro else float("inf")

    metrics = {
        "asset": dataset.stem,
        "dataset": str(dataset),
        "n_points": int(n),
        "n_points_calibration": int(working_cal_best.shape[0]),
        "n_points_evaluation": int(working_eval_best.shape[0]),
        "n_regimes_total": int(np.unique(labels).shape[0]),
        "n_micro_transitions": int(len(micro_switch_eval)),
        "micro_switch_rate": micro_switch_rate,
        "density_ratio_micro": density_ratio_micro,
        "hit_rate_micro": float(min(1.0, micro_hits / max(1, len(events_eval)))),
        **metrics_eval_best,
        "selected_w": selected_w,
        "selected_k": selected_k,
        "selected_q_expand": selected_q,
        "score_calibration": float(chosen["score"]),
        "score_evaluation": float(eval_score),
        "calibration_split_date": split_date.strftime("%Y-%m-%d") if split_date is not None else "",
        "metrics_calibration_selected": metrics_cal_best_live,
    }

    # Pseudo-bifurcation: instability spikes without consistent structural regime change.
    ds = working_eval_best["delta_instability"].astype(float)
    high_delta = ds >= float(ds.quantile(0.90))
    high_ratio = float(high_delta.mean())
    switch_ratio = float(len(micro_switch_eval) / max(1, len(working_eval_best)))
    macro_event_count = int(metrics_eval_best.get("n_macro_events", 0) or 0)
    pseudo_flag = bool(high_ratio >= 0.20 and switch_ratio <= 0.04 and macro_event_count <= 2)
    pseudo_payload = {
        "status": "ok",
        "high_delta_ratio": high_ratio,
        "micro_switch_ratio": switch_ratio,
        "macro_event_count": macro_event_count,
        "pseudo_bifurcation_flag": pseudo_flag,
        "rule": "high_delta_ratio>=0.20 and micro_switch_ratio<=0.04 and macro_event_count<=2",
    }
    _json_dump(outdir / "pseudo_bifurcation.json", pseudo_payload)
    metrics["pseudo_bifurcation_flag"] = pseudo_flag
    metrics["high_delta_ratio"] = high_ratio
    metrics["micro_switch_ratio"] = switch_ratio

    _json_dump(outdir / "metrics.json", metrics)

    verdict_source = dict(metrics_eval_best)
    verdict_source["pseudo_bifurcation_flag"] = pseudo_flag
    verdict = _verdict(verdict_source, used_k=selected_k)
    verdict.update(
        {
            "notes": [
                "Macro eventos sao extraidos por picos de delta_instability por ano.",
                "Blocos >365 dias sao classificados como macro_long_regime e excluidos do score.",
                "Hit usa janela historica de +-90 dias por evento.",
                "Density ratio compara concentracao de macro_event em janelas historicas vs fundo.",
                "Quando --calibrate esta ativo, a grade e escolhida no bloco de calibracao e avaliada em holdout temporal.",
            ]
        }
    )
    _write_status(outdir, verdict)

    profile = {
        "asset": dataset.stem,
        "dataset": str(dataset),
        "value_col": value_col,
        "date_col": date_col,
        "date_min": pd.Timestamp(regimes["date"].min()).strftime("%Y-%m-%d"),
        "date_max": pd.Timestamp(regimes["date"].max()).strftime("%Y-%m-%d"),
        "n_points": int(n),
        "warning_short_history": bool(n < MIN_LONG_POINTS),
        "motor_params": {"m": int(m), "tau": int(tau), "seed": int(args.seed)},
        "macro_params": {
            "W_selected": selected_w,
            "K_selected": selected_k,
            "q_expand_selected": selected_q,
            "peak_min_gap_days": int(args.peak_min_gap_days),
            "merge_gap_days": int(args.merge_gap_days),
            "min_duration": int(args.min_duration),
            "max_duration": int(args.max_duration),
            "calibrate_mode": bool(args.calibrate),
            "calibration_ratio": float(args.calibration_ratio),
            "calibration_split_date": split_date.strftime("%Y-%m-%d") if split_date is not None else "",
        },
    }
    _json_dump(outdir / "data_profile.json", profile)

    print(
        "historical_shifts ok | "
        f"asset={dataset.stem} n={n} "
        f"hit_rate_macro={metrics['hit_rate_macro']:.3f} "
        f"density_ratio_macro={metrics['density_ratio_macro']:.3f} "
        f"lead_rate_macro={(metrics['lead_rate_macro'] if metrics['lead_rate_macro'] is not None else float('nan')):.3f} "
        f"status={verdict['status']}"
    )


if __name__ == "__main__":
    main()
