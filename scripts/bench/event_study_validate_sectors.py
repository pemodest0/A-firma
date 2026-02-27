#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


@dataclass
class EvalResult:
    recall: float
    precision: float
    false_alarm_per_year: float
    mean_lead_days: float
    coincident_rate: float
    n_events: int
    n_alert_days: int
    n_false_alert_days: int
    n_alert_episodes: int
    n_false_alert_episodes: int


def _ts_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_tickers(path: Path) -> list[str]:
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _load_returns_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, OSError):
        return None
    if "date" not in df.columns:
        return None
    out = df[["date"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if "r" in df.columns:
        out["ret"] = pd.to_numeric(df["r"], errors="coerce")
    elif "price" in df.columns:
        p = pd.to_numeric(df["price"], errors="coerce")
        out["ret"] = np.log(p / p.shift(1))
    else:
        return None
    out = out.dropna(subset=["date", "ret"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return out


def _load_sector_map(paths: list[Path]) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in paths:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, OSError):
            continue
        cols = {str(c).lower(): str(c) for c in df.columns}
        a_col = cols.get("asset") or cols.get("ticker") or cols.get("symbol")
        s_col = cols.get("group") or cols.get("sector") or cols.get("category")
        if not a_col or not s_col:
            continue
        for _, r in df.iterrows():
            a = str(r[a_col]).strip()
            s = str(r[s_col]).strip()
            if a and s:
                out[a] = s
    return out


def _dedupe_events(event_dates: list[pd.Timestamp], event_pos: list[int], cooldown_days: int = 20) -> list[pd.Timestamp]:
    out: list[pd.Timestamp] = []
    last_pos: int | None = None
    pairs = sorted(zip(event_dates, event_pos), key=lambda x: int(x[1]))
    for d, p in pairs:
        if last_pos is None or (int(p) - int(last_pos)) > int(cooldown_days):
            out.append(pd.Timestamp(d))
            last_pos = int(p)
    return out


def build_reference_series(tickers: list[str], prices_dir: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for t in tickers:
        df = _load_returns_csv(prices_dir / f"{t}.csv")
        if df is None or df.empty:
            continue
        s = df.set_index("date")["ret"].rename(t)
        parts.append(s.to_frame())
    if not parts:
        raise RuntimeError("No return series available to build reference market.")
    m = pd.concat(parts, axis=1, join="outer", sort=False).sort_index()
    ref = pd.DataFrame(index=m.index)
    ref["ret"] = m.mean(axis=1, skipna=True)
    ref = ref.dropna(subset=["ret"])
    ref["price"] = 100.0 * np.exp(ref["ret"].cumsum())
    ref["vol20"] = ref["ret"].rolling(20, min_periods=20).std() * np.sqrt(252.0)
    ref["dd20"] = ref["price"] / ref["price"].rolling(20, min_periods=20).max() - 1.0
    ref = ref.reset_index().rename(columns={"index": "date"})
    return ref


def build_sector_daily_series(
    tickers: list[str],
    sector_map: dict[str, str],
    assets_dir: Path,
    prices_dir: Path,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for t in tickers:
        rg_path = assets_dir / f"{t}_daily_regimes.csv"
        px_path = prices_dir / f"{t}.csv"
        if not rg_path.exists() or not px_path.exists():
            continue
        try:
            rg = pd.read_csv(rg_path)
        except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, OSError):
            continue
        if rg.empty or "regime" not in rg.columns or "confidence" not in rg.columns:
            continue
        px = _load_returns_csv(px_path)
        if px is None or px.empty:
            continue
        dates = px["date"].to_list()
        n_rg = int(len(rg))
        if n_rg <= 0 or len(dates) < n_rg:
            continue
        date_aligned = dates[-n_rg:]
        d = pd.DataFrame(
            {
                "date": date_aligned,
                "asset": t,
                "sector": str(sector_map.get(t, "unknown")),
                "regime": rg["regime"].astype(str).to_numpy(),
                "confidence": pd.to_numeric(rg["confidence"], errors="coerce").to_numpy(),
            }
        ).dropna(subset=["date", "confidence"])
        rows.append(d)
    if not rows:
        raise RuntimeError("No regime history found in assets folder.")

    panel = pd.concat(rows, axis=0, ignore_index=True)
    panel["is_transition"] = (panel["regime"] == "TRANSITION").astype(float)
    panel["is_unstable"] = panel["regime"].isin(["UNSTABLE", "NOISY"]).astype(float)
    panel["is_alert_regime"] = panel["regime"].isin(["TRANSITION", "UNSTABLE", "NOISY"]).astype(float)
    panel["confidence"] = panel["confidence"].clip(0.0, 1.0)

    agg = (
        panel.groupby(["date", "sector"], as_index=False)
        .agg(
            n_assets=("asset", "count"),
            share_transition=("is_transition", "mean"),
            share_unstable=("is_unstable", "mean"),
            share_alert_regime=("is_alert_regime", "mean"),
            mean_confidence=("confidence", "mean"),
        )
        .sort_values(["sector", "date"])
        .reset_index(drop=True)
    )
    agg["sector_score"] = (
        0.60 * agg["share_transition"] + 1.00 * agg["share_unstable"] + 0.40 * (1.0 - agg["mean_confidence"])
    )
    return agg


def build_event_dates(
    ref: pd.DataFrame,
    test_start: pd.Timestamp,
    ret_q01: float,
    vol_q95: float,
) -> dict[str, list[pd.Timestamp]]:
    x = ref.copy().sort_values("date")
    x["event_ret_tail"] = x["ret"] <= float(ret_q01)
    x["event_dd20"] = x["dd20"] <= -0.08
    x["event_vol_spike"] = x["vol20"] >= float(vol_q95)
    x["event_stress_combo"] = (x["dd20"] <= -0.06) & (x["vol20"] >= float(vol_q95))
    out: dict[str, list[pd.Timestamp]] = {}
    for name, col in [
        ("ret_tail", "event_ret_tail"),
        ("drawdown20", "event_dd20"),
        ("vol_spike20", "event_vol_spike"),
        ("stress_combo", "event_stress_combo"),
    ]:
        mask = (x["date"] >= test_start) & (x[col].fillna(False))
        dates = x.loc[mask, "date"].tolist()
        pos = x.index[mask].tolist()
        out[name] = _dedupe_events(dates, pos, cooldown_days=20)
    return out


def evaluate_alerts(
    dates: pd.Series,
    alert: pd.Series,
    event_dates: list[pd.Timestamp],
    lookback_days: int,
    assoc_horizon_days: int = 20,
) -> EvalResult:
    dts = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    s_alert = pd.Series(alert.to_numpy(dtype=bool)).reset_index(drop=True)
    date_to_idx = {d: i for i, d in enumerate(dts)}
    event_idx = sorted([date_to_idx[d] for d in pd.to_datetime(pd.Series(event_dates)).tolist() if d in date_to_idx])

    detected = 0
    coincident = 0
    lead_days: list[int] = []
    for e in event_idx:
        lo = max(0, e - int(lookback_days))
        hi = e - 1
        if hi >= lo:
            w_pre = s_alert.iloc[lo : hi + 1]
            if bool(w_pre.any()):
                detected += 1
                first_rel = int(np.argmax(w_pre.to_numpy(dtype=bool)))
                first_idx = lo + first_rel
                lead_days.append(int(e - first_idx))
        co_lo = e
        co_hi = min(len(s_alert) - 1, e + 2)
        if co_hi >= co_lo and bool(s_alert.iloc[co_lo : co_hi + 1].any()):
            coincident += 1

    alert_days_idx = np.where(s_alert.to_numpy(dtype=bool))[0]
    # Precision/false alarm are episode-based (entry alerts), avoiding distortion from long alert streaks.
    starts_mask = s_alert.to_numpy(dtype=bool) & (~s_alert.shift(1, fill_value=False).to_numpy(dtype=bool))
    alert_episode_idx = np.where(starts_mask)[0]
    good_alerts = 0
    for a in alert_episode_idx:
        has_future_event = any((ev >= a + 1) and (ev <= a + int(assoc_horizon_days)) for ev in event_idx)
        if has_future_event:
            good_alerts += 1
    n_alert_episodes = int(len(alert_episode_idx))
    n_false_episodes = int(max(0, n_alert_episodes - good_alerts))
    good_alert_days = 0
    for a in alert_days_idx:
        has_future_event = any((ev >= a + 1) and (ev <= a + int(assoc_horizon_days)) for ev in event_idx)
        if has_future_event:
            good_alert_days += 1
    n_false_alert_days = int(max(0, len(alert_days_idx) - good_alert_days))
    years = max(1e-9, len(dts) / 252.0)

    return EvalResult(
        recall=float(detected / len(event_idx)) if event_idx else float("nan"),
        precision=float(good_alerts / n_alert_episodes) if n_alert_episodes > 0 else float("nan"),
        false_alarm_per_year=float(n_false_episodes / years),
        mean_lead_days=float(np.mean(lead_days)) if lead_days else float("nan"),
        coincident_rate=float(coincident / len(event_idx)) if event_idx else float("nan"),
        n_events=int(len(event_idx)),
        n_alert_days=int(len(alert_days_idx)),
        n_false_alert_days=n_false_alert_days,
        n_alert_episodes=n_alert_episodes,
        n_false_alert_episodes=n_false_episodes,
    )


def random_baseline_distribution(
    dates: pd.Series,
    n_alert_days: int,
    event_dates: list[pd.Timestamp],
    lookback_days: int,
    n_boot: int = 300,
    seed: int = 7,
    method: str = "iid",
    block_size: int = 10,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_total = int(len(dates))
    idx = np.arange(n_total)
    target = int(max(0, min(int(n_alert_days), n_total)))
    method = str(method).strip().lower()
    bsz = int(max(2, min(int(block_size), max(2, n_total))))
    block_starts = np.arange(0, max(1, n_total - bsz + 1))
    out: list[dict[str, float]] = []
    for _ in range(int(n_boot)):
        a = np.zeros(n_total, dtype=bool)
        if target > 0:
            if method == "block":
                selected: list[int] = []
                max_iters = max(10, n_total * 4)
                iters = 0
                while len(selected) < target and iters < max_iters:
                    iters += 1
                    s = int(rng.choice(block_starts))
                    e = min(n_total, s + bsz)
                    selected.extend(range(s, e))
                if selected:
                    pick = np.unique(np.array(selected, dtype=int))
                    if len(pick) > target:
                        pick = rng.choice(pick, size=target, replace=False)
                    a[pick] = True
            else:
                pick = rng.choice(idx, size=target, replace=False)
                a[pick] = True
        ev = evaluate_alerts(dates=dates, alert=pd.Series(a), event_dates=event_dates, lookback_days=lookback_days)
        out.append(
            {
                "recall": ev.recall,
                "precision": ev.precision,
                "false_alarm_per_year": ev.false_alarm_per_year,
                "mean_lead_days": ev.mean_lead_days,
            }
        )
    return pd.DataFrame(out)


def _entry_alert(signal: pd.Series) -> pd.Series:
    s = pd.Series(signal).fillna(False).astype(bool)
    return s & (~s.shift(1, fill_value=False))


def _confirm_n_of_m(signal: pd.Series, n: int = 2, m: int = 3) -> pd.Series:
    s = pd.Series(signal).fillna(False).astype(int)
    return (s.rolling(m, min_periods=m).sum() >= int(n)).astype(bool)


def _apply_min_gap(alert: pd.Series, min_gap_days: int = 0) -> pd.Series:
    s = pd.Series(alert).fillna(False).astype(bool).copy()
    gap = max(0, int(min_gap_days))
    if gap <= 0 or s.empty:
        return s
    idx = np.where(s.to_numpy(dtype=bool))[0]
    if len(idx) <= 1:
        return s
    keep = np.zeros(len(s), dtype=bool)
    last_kept = -10**9
    for i in idx:
        if int(i - last_kept) > gap:
            keep[int(i)] = True
            last_kept = int(i)
    return pd.Series(keep, index=s.index)


def _safe_float(x: float | int | np.floating | None) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return v if np.isfinite(v) else 0.0


def _filter_events_between(
    events: dict[str, list[pd.Timestamp]],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, list[pd.Timestamp]]:
    out: dict[str, list[pd.Timestamp]] = {}
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    for k, vals in events.items():
        out[k] = [pd.Timestamp(v) for v in vals if (pd.Timestamp(v) >= s and pd.Timestamp(v) <= e)]
    return out


def build_alert_series_for_sector(
    cal: pd.DataFrame,
    test: pd.DataFrame,
    policy: str,
    params: dict[str, float] | None = None,
) -> tuple[pd.Series, dict[str, float]]:
    policy = str(policy).strip().lower()
    meta: dict[str, float] = {}
    p = dict(params or {})
    q_unstable = float(p.get("q_unstable", 0.80))
    q_transition = float(p.get("q_transition", 0.80))
    q_conf = float(p.get("q_confidence", 0.50))
    q_conf_guarded = float(p.get("q_confidence_guarded", 0.60))
    q_score_balanced = float(p.get("q_score_balanced", 0.70))
    q_score_guarded = float(p.get("q_score_guarded", 0.80))
    confirm_n = int(p.get("confirm_n", 2))
    confirm_m = int(p.get("confirm_m", 3))
    min_alert_gap_days = int(p.get("min_alert_gap_days", 2))
    confirm_m = max(1, confirm_m)
    confirm_n = max(1, min(confirm_n, confirm_m))

    if policy == "score_q80":
        thr = float(cal["sector_score"].quantile(0.80))
        meta["sector_score_q80"] = thr
        return (test["sector_score"] >= thr).astype(bool), meta

    if policy == "score_q90":
        thr = float(cal["sector_score"].quantile(0.90))
        meta["sector_score_q90"] = thr
        return (test["sector_score"] >= thr).astype(bool), meta

    if policy == "regime_entry":
        thr_unstable = float(cal["share_unstable"].quantile(q_unstable))
        thr_transition = float(cal["share_transition"].quantile(q_transition))
        thr_conf = float(cal["mean_confidence"].quantile(q_conf))
        raw = (test["share_unstable"] >= thr_unstable) | (
            (test["share_transition"] >= thr_transition) & (test["mean_confidence"] >= thr_conf)
        )
        meta["share_unstable_q"] = q_unstable
        meta["share_transition_q"] = q_transition
        meta["mean_confidence_q"] = q_conf
        meta["share_unstable_thr"] = thr_unstable
        meta["share_transition_thr"] = thr_transition
        meta["mean_confidence_thr"] = thr_conf
        return _apply_min_gap(_entry_alert(raw), min_gap_days=min_alert_gap_days), meta

    if policy == "regime_entry_confirm":
        thr_unstable = float(cal["share_unstable"].quantile(q_unstable))
        thr_transition = float(cal["share_transition"].quantile(q_transition))
        thr_conf = float(cal["mean_confidence"].quantile(q_conf))
        raw = (test["share_unstable"] >= thr_unstable) | (
            (test["share_transition"] >= thr_transition) & (test["mean_confidence"] >= thr_conf)
        )
        confirmed = _confirm_n_of_m(raw, n=confirm_n, m=confirm_m)
        meta["share_unstable_q"] = q_unstable
        meta["share_transition_q"] = q_transition
        meta["mean_confidence_q"] = q_conf
        meta["share_unstable_thr"] = thr_unstable
        meta["share_transition_thr"] = thr_transition
        meta["mean_confidence_thr"] = thr_conf
        meta["confirm_n"] = float(confirm_n)
        meta["confirm_m"] = float(confirm_m)
        return _apply_min_gap(_entry_alert(confirmed), min_gap_days=min_alert_gap_days), meta

    if policy == "regime_balanced":
        thr_unstable = float(cal["share_unstable"].quantile(q_unstable))
        thr_transition = float(cal["share_transition"].quantile(q_transition))
        thr_conf = float(cal["mean_confidence"].quantile(q_conf))
        thr_score = float(cal["sector_score"].quantile(q_score_balanced))
        raw = (
            ((test["share_unstable"] >= thr_unstable) & (test["sector_score"] >= thr_score))
            | (
                (test["share_transition"] >= thr_transition)
                & (test["mean_confidence"] >= thr_conf)
                & (test["sector_score"] >= thr_score)
            )
        )
        confirmed = _confirm_n_of_m(raw, n=confirm_n, m=confirm_m)
        meta["share_unstable_q"] = q_unstable
        meta["share_transition_q"] = q_transition
        meta["mean_confidence_q"] = q_conf
        meta["sector_score_q"] = q_score_balanced
        meta["share_unstable_thr"] = thr_unstable
        meta["share_transition_thr"] = thr_transition
        meta["mean_confidence_thr"] = thr_conf
        meta["sector_score_thr"] = thr_score
        meta["confirm_n"] = float(confirm_n)
        meta["confirm_m"] = float(confirm_m)
        return _apply_min_gap(_entry_alert(confirmed), min_gap_days=min_alert_gap_days), meta

    if policy == "regime_guarded":
        thr_unstable = float(cal["share_unstable"].quantile(q_unstable))
        thr_transition = float(cal["share_transition"].quantile(q_transition))
        thr_conf = float(cal["mean_confidence"].quantile(q_conf_guarded))
        thr_score = float(cal["sector_score"].quantile(q_score_guarded))
        raw = (
            ((test["share_unstable"] >= thr_unstable) & (test["sector_score"] >= thr_score))
            | (
                (test["share_transition"] >= thr_transition)
                & (test["mean_confidence"] >= thr_conf)
                & (test["sector_score"] >= thr_score)
            )
        )
        confirmed = _confirm_n_of_m(raw, n=confirm_n, m=confirm_m)
        meta["share_unstable_q"] = q_unstable
        meta["share_transition_q"] = q_transition
        meta["mean_confidence_q"] = q_conf_guarded
        meta["sector_score_q"] = q_score_guarded
        meta["share_unstable_thr"] = thr_unstable
        meta["share_transition_thr"] = thr_transition
        meta["mean_confidence_thr"] = thr_conf
        meta["sector_score_thr"] = thr_score
        meta["confirm_n"] = float(confirm_n)
        meta["confirm_m"] = float(confirm_m)
        return _apply_min_gap(_entry_alert(confirmed), min_gap_days=min_alert_gap_days), meta

    raise ValueError(f"Unknown alert policy: {policy}")


def build_layered_alerts_for_sector(
    cal: pd.DataFrame,
    test: pd.DataFrame,
    policy: str,
    params: dict[str, float] | None = None,
) -> tuple[dict[str, pd.Series], dict[str, float]]:
    policy = str(policy).strip().lower()
    meta: dict[str, float] = {}
    p = dict(params or {})
    q_unstable = float(p.get("q_unstable", 0.80))
    q_transition = float(p.get("q_transition", 0.80))
    q_conf = float(p.get("q_confidence", 0.50))
    q_conf_guarded = float(p.get("q_confidence_guarded", 0.60))
    q_score_balanced = float(p.get("q_score_balanced", 0.70))
    q_score_guarded = float(p.get("q_score_guarded", 0.80))
    confirm_n = int(p.get("confirm_n", 2))
    confirm_m = int(p.get("confirm_m", 3))
    min_alert_gap_days = int(p.get("min_alert_gap_days", 2))
    confirm_m = max(1, confirm_m)
    confirm_n = max(1, min(confirm_n, confirm_m))

    raw: pd.Series
    if policy == "score_q80":
        thr = float(cal["sector_score"].quantile(0.80))
        raw = (test["sector_score"] >= thr).astype(bool)
        meta["sector_score_q80"] = thr
    elif policy == "score_q90":
        thr = float(cal["sector_score"].quantile(0.90))
        raw = (test["sector_score"] >= thr).astype(bool)
        meta["sector_score_q90"] = thr
    elif policy in {"regime_entry", "regime_entry_confirm"}:
        thr_unstable = float(cal["share_unstable"].quantile(q_unstable))
        thr_transition = float(cal["share_transition"].quantile(q_transition))
        thr_conf = float(cal["mean_confidence"].quantile(q_conf))
        raw = ((test["share_unstable"] >= thr_unstable) | ((test["share_transition"] >= thr_transition) & (test["mean_confidence"] >= thr_conf))).astype(bool)
        meta["share_unstable_q"] = q_unstable
        meta["share_transition_q"] = q_transition
        meta["mean_confidence_q"] = q_conf
        meta["share_unstable_thr"] = thr_unstable
        meta["share_transition_thr"] = thr_transition
        meta["mean_confidence_thr"] = thr_conf
    elif policy == "regime_balanced":
        thr_unstable = float(cal["share_unstable"].quantile(q_unstable))
        thr_transition = float(cal["share_transition"].quantile(q_transition))
        thr_conf = float(cal["mean_confidence"].quantile(q_conf))
        thr_score = float(cal["sector_score"].quantile(q_score_balanced))
        raw = (
            ((test["share_unstable"] >= thr_unstable) & (test["sector_score"] >= thr_score))
            | (
                (test["share_transition"] >= thr_transition)
                & (test["mean_confidence"] >= thr_conf)
                & (test["sector_score"] >= thr_score)
            )
        ).astype(bool)
        meta["share_unstable_q"] = q_unstable
        meta["share_transition_q"] = q_transition
        meta["mean_confidence_q"] = q_conf
        meta["sector_score_q"] = q_score_balanced
        meta["share_unstable_thr"] = thr_unstable
        meta["share_transition_thr"] = thr_transition
        meta["mean_confidence_thr"] = thr_conf
        meta["sector_score_thr"] = thr_score
    elif policy == "regime_guarded":
        thr_unstable = float(cal["share_unstable"].quantile(q_unstable))
        thr_transition = float(cal["share_transition"].quantile(q_transition))
        thr_conf = float(cal["mean_confidence"].quantile(q_conf_guarded))
        thr_score = float(cal["sector_score"].quantile(q_score_guarded))
        raw = (
            ((test["share_unstable"] >= thr_unstable) & (test["sector_score"] >= thr_score))
            | (
                (test["share_transition"] >= thr_transition)
                & (test["mean_confidence"] >= thr_conf)
                & (test["sector_score"] >= thr_score)
            )
        ).astype(bool)
        meta["share_unstable_q"] = q_unstable
        meta["share_transition_q"] = q_transition
        meta["mean_confidence_q"] = q_conf_guarded
        meta["sector_score_q"] = q_score_guarded
        meta["share_unstable_thr"] = thr_unstable
        meta["share_transition_thr"] = thr_transition
        meta["mean_confidence_thr"] = thr_conf
        meta["sector_score_thr"] = thr_score
    else:
        raise ValueError(f"Unknown alert policy for layered mode: {policy}")

    confirmed_state = _confirm_n_of_m(raw, n=confirm_n, m=confirm_m).astype(bool)
    fast_alert = _apply_min_gap(_entry_alert(raw).astype(bool), min_gap_days=min_alert_gap_days)
    confirmed_alert = _apply_min_gap(_entry_alert(confirmed_state).astype(bool), min_gap_days=min_alert_gap_days)
    meta["confirm_n"] = float(confirm_n)
    meta["confirm_m"] = float(confirm_m)
    meta["min_alert_gap_days"] = float(min_alert_gap_days)
    return {
        "fast_state": raw,
        "confirmed_state": confirmed_state,
        "fast_alert": fast_alert,
        "confirmed_alert": confirmed_alert,
    }, meta


def choose_auto_policy_for_sector(
    cal: pd.DataFrame,
    events_cal: dict[str, list[pd.Timestamp]],
    params: dict[str, float],
    candidates: list[str],
) -> tuple[str, dict[str, float]]:
    default_policy = "regime_entry_confirm"
    candidates = [str(x).strip() for x in candidates if str(x).strip()]
    if not candidates:
        candidates = [default_policy]
    if default_policy not in candidates:
        candidates = [default_policy] + candidates
    n = int(len(cal))
    if n < 320:
        return default_policy, {"auto_reason": 0.0}

    split_idx = int(n * 0.70)
    split_idx = max(252, split_idx)
    split_idx = min(split_idx, n - 60)
    if split_idx < 252 or split_idx >= n - 30:
        return default_policy, {"auto_reason": 1.0}

    cal_thr = cal.iloc[:split_idx].copy()
    cal_eval = cal.iloc[split_idx:].copy().reset_index(drop=True)
    if cal_eval.empty:
        return default_policy, {"auto_reason": 2.0}

    ev_local = _filter_events_between(
        events=events_cal,
        start=pd.Timestamp(cal_eval["date"].min()),
        end=pd.Timestamp(cal_eval["date"].max()),
    )

    best_policy = default_policy
    best_score = -1e9
    best_meta: dict[str, float] = {}
    for cand in candidates:
        try:
            layered, _ = build_layered_alerts_for_sector(cal=cal_thr, test=cal_eval, policy=cand, params=params)
            alert_eval = layered["confirmed_alert"].astype(bool)
        except (TypeError, ValueError, KeyError, RuntimeError):
            continue
        m_draw5 = evaluate_alerts(cal_eval["date"], alert_eval, ev_local.get("drawdown20", []), lookback_days=5)
        m_draw10 = evaluate_alerts(cal_eval["date"], alert_eval, ev_local.get("drawdown20", []), lookback_days=10)
        m_draw20 = evaluate_alerts(cal_eval["date"], alert_eval, ev_local.get("drawdown20", []), lookback_days=20)
        m_tail10 = evaluate_alerts(cal_eval["date"], alert_eval, ev_local.get("ret_tail", []), lookback_days=10)
        m_stress10 = evaluate_alerts(cal_eval["date"], alert_eval, ev_local.get("stress_combo", []), lookback_days=10)
        yrs = max(1e-9, len(cal_eval) / 252.0)
        toggle_rate = float((alert_eval.astype(int).diff().abs().fillna(0.0).sum()) / yrs)
        event_factor = float(np.clip(_safe_float(m_draw10.n_events) / 8.0, 0.55, 1.0))
        score = (
            0.36 * _safe_float(m_draw10.recall)
            + 0.16 * _safe_float(m_draw20.recall)
            + 0.14 * _safe_float(m_draw5.recall)
            + 0.08 * _safe_float(m_tail10.recall)
            + 0.08 * _safe_float(m_stress10.recall)
            + 0.16 * _safe_float(m_draw10.precision)
            - 0.025 * _safe_float(m_draw10.false_alarm_per_year)
            - 0.01 * float(np.clip(toggle_rate, 0.0, 30.0))
        )
        score = float(score * event_factor)
        if _safe_float(m_draw10.false_alarm_per_year) > 12.0:
            score -= 0.15
        if _safe_float(m_draw10.precision) < 0.05:
            score -= 0.10
        if score > best_score:
            best_score = score
            best_policy = cand
            best_meta = {
                "auto_score": float(score),
                "auto_drawdown_recall_l10": _safe_float(m_draw10.recall),
                "auto_drawdown_recall_l20": _safe_float(m_draw20.recall),
                "auto_drawdown_recall_l5": _safe_float(m_draw5.recall),
                "auto_ret_tail_recall_l10": _safe_float(m_tail10.recall),
                "auto_stress_recall_l10": _safe_float(m_stress10.recall),
                "auto_drawdown_precision_l10": _safe_float(m_draw10.precision),
                "auto_drawdown_false_alarm_l10": _safe_float(m_draw10.false_alarm_per_year),
                "auto_toggle_rate": float(toggle_rate),
                "auto_event_factor": float(event_factor),
                "auto_chosen_policy": float(candidates.index(cand) + 1),
            }

    return best_policy, best_meta


def build_level_series_for_sector(cal: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.Series, dict[str, float]]:
    q70 = float(cal["sector_score"].quantile(0.70))
    q85 = float(cal["sector_score"].quantile(0.85))
    score = test["sector_score"].to_numpy(dtype=float)
    lvl = np.where(score >= q85, "vermelho", np.where(score >= q70, "amarelo", "verde"))
    meta = {"sector_score_q70": q70, "sector_score_q85": q85}
    return pd.Series(lvl, index=test.index), meta


def build_action_plan_row(
    *,
    level: str,
    score: float,
    score_delta_5d: float,
    mean_confidence: float,
    level_changes_30d: int,
    share_unstable: float,
    share_transition: float,
) -> dict[str, object]:
    lvl = str(level).strip().lower()
    conf = float(mean_confidence)
    delta = float(score_delta_5d)
    churn = int(level_changes_30d)
    unstable = float(share_unstable)
    transition = float(share_transition)

    if lvl == "vermelho":
        risk_min, risk_max = 0.30, 0.60
        tier = "defesa_forte"
    elif lvl == "amarelo":
        risk_min, risk_max = 0.65, 0.85
        tier = "cautela"
    else:
        risk_min, risk_max = 0.90, 1.00
        tier = "normal"

    worsened = (delta > 0.03) or (churn >= 6) or (unstable >= 0.30) or (transition >= 0.50)
    if worsened:
        risk_min = max(0.10, risk_min - 0.10)
        risk_max = max(risk_min + 0.05, risk_max - 0.10)
    if conf < 0.46:
        risk_min = max(0.10, risk_min - 0.05)
        risk_max = max(risk_min + 0.05, risk_max - 0.05)

    hedge_min = float(max(0.0, 1.0 - risk_max))
    hedge_max = float(max(hedge_min, 1.0 - risk_min))
    action_reason = "normal"
    if worsened and conf < 0.46:
        action_reason = "piora recente com confianca baixa"
    elif worsened:
        action_reason = "piora recente no setor"
    elif conf < 0.46:
        action_reason = "confianca baixa, operar com margem"
    elif lvl == "vermelho":
        action_reason = "risco estrutural alto no setor"
    elif lvl == "amarelo":
        action_reason = "setor em fase de cautela"

    priority = float(
        0.55 * (2.0 if lvl == "vermelho" else 1.0 if lvl == "amarelo" else 0.0)
        + 0.20 * max(0.0, unstable)
        + 0.15 * max(0.0, transition)
        + 0.10 * max(0.0, delta)
    )
    return {
        "action_tier": tier,
        "risk_budget_min": float(risk_min),
        "risk_budget_max": float(risk_max),
        "hedge_min": hedge_min,
        "hedge_max": hedge_max,
        "action_priority": priority,
        "action_reason": action_reason,
    }


def _format_metric(x: float) -> str:
    if pd.isna(x):
        return "nan"
    return f"{x:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Sector-level event study validation for the 470-asset motor.")
    ap.add_argument("--tickers-file", type=str, default="results/universe_470/tickers_470.txt")
    ap.add_argument("--assets-dir", type=str, default="results/latest_graph_universe470_batch/assets")
    ap.add_argument("--prices-dir", type=str, default="data/raw/finance/yfinance_daily")
    ap.add_argument(
        "--sector-map-files",
        nargs="*",
        default=[
            "data/asset_groups_470_enriched.csv",
            "data/asset_groups.csv",
            "results/finance_download/local_pack_20260218T060240Z/universe_fixed.csv",
        ],
    )
    ap.add_argument("--calibration-end", type=str, default="2019-12-31")
    ap.add_argument("--test-start", type=str, default="2020-01-01")
    ap.add_argument("--test-end", type=str, default="", help="Optional inclusive test end date (YYYY-MM-DD).")
    ap.add_argument("--lookbacks", type=str, default="1,5,10,20")
    ap.add_argument("--n-random", type=int, default=300)
    ap.add_argument(
        "--random-baseline-method",
        type=str,
        default="both",
        choices=["iid", "block", "both"],
        help="iid: dias aleatorios; block: blocos temporais; both: calcula os dois.",
    )
    ap.add_argument("--random-block-size", type=int, default=10)
    ap.add_argument("--min-sector-assets", type=int, default=10)
    ap.add_argument("--min-cal-days", type=int, default=252)
    ap.add_argument("--min-test-days", type=int, default=252)
    ap.add_argument("--q-unstable", type=float, default=0.80)
    ap.add_argument("--q-transition", type=float, default=0.80)
    ap.add_argument("--q-confidence", type=float, default=0.50)
    ap.add_argument("--q-confidence-guarded", type=float, default=0.60)
    ap.add_argument("--q-score-balanced", type=float, default=0.70)
    ap.add_argument("--q-score-guarded", type=float, default=0.80)
    ap.add_argument("--confirm-n", type=int, default=2)
    ap.add_argument("--confirm-m", type=int, default=3)
    ap.add_argument("--min-alert-gap-days", type=int, default=2, help="Gap minimo entre alertas de entrada.")
    ap.add_argument(
        "--two-layer-mode",
        type=str,
        default="on",
        choices=["on", "off"],
        help="on: gera sinal rapido e confirmado; off: usa apenas sinal confirmado.",
    )
    ap.add_argument(
        "--auto-candidates",
        type=str,
        default="regime_entry_confirm,regime_balanced,regime_guarded",
        help="Policies considered by regime_auto, comma separated.",
    )
    ap.add_argument(
        "--alert-policy",
        type=str,
        default="regime_entry_confirm",
        choices=[
            "regime_entry",
            "regime_entry_confirm",
            "regime_balanced",
            "regime_guarded",
            "regime_auto",
            "score_q80",
            "score_q90",
        ],
    )
    ap.add_argument("--out-root", type=str, default="results/event_study_sectors")
    ap.add_argument("--run-id", type=str, default="", help="Optional fixed run id for deterministic outdir.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    def log(msg: str) -> None:
        if args.verbose:
            print(msg, file=sys.stderr, flush=True)

    tickers = _read_tickers(ROOT / args.tickers_file)
    assets_dir = ROOT / args.assets_dir
    prices_dir = ROOT / args.prices_dir
    sector_map_paths = [ROOT / p for p in args.sector_map_files]
    sector_map = _load_sector_map(sector_map_paths)

    calibration_end = pd.to_datetime(args.calibration_end)
    test_start = pd.to_datetime(args.test_start)
    test_end = pd.to_datetime(args.test_end) if str(args.test_end).strip() else None
    lookbacks = [int(x.strip()) for x in str(args.lookbacks).split(",") if x.strip()]
    param_cfg = {
        "q_unstable": float(max(0.01, min(0.99, float(args.q_unstable)))),
        "q_transition": float(max(0.01, min(0.99, float(args.q_transition)))),
        "q_confidence": float(max(0.01, min(0.99, float(args.q_confidence)))),
        "q_confidence_guarded": float(max(0.01, min(0.99, float(args.q_confidence_guarded)))),
        "q_score_balanced": float(max(0.01, min(0.99, float(args.q_score_balanced)))),
        "q_score_guarded": float(max(0.01, min(0.99, float(args.q_score_guarded)))),
        "confirm_n": float(max(1, int(args.confirm_n))),
        "confirm_m": float(max(1, int(args.confirm_m))),
        "min_alert_gap_days": float(max(0, int(args.min_alert_gap_days))),
    }
    auto_candidates = [x.strip() for x in str(args.auto_candidates).split(",") if x.strip()]

    run_id = str(args.run_id).strip() or _ts_id()
    outdir = ROOT / args.out_root / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    log("step: build_reference")
    ref = build_reference_series(tickers=tickers, prices_dir=prices_dir)
    log("step: build_sector_daily")
    sector_daily = build_sector_daily_series(
        tickers=tickers,
        sector_map=sector_map,
        assets_dir=assets_dir,
        prices_dir=prices_dir,
    )
    df = pd.merge(
        sector_daily,
        ref[["date", "ret", "vol20", "dd20"]],
        on="date",
        how="inner",
    ).sort_values(["sector", "date"])
    df = df[df["date"] >= pd.to_datetime("2018-01-01")].reset_index(drop=True)

    ref_cal = ref[ref["date"] <= calibration_end].copy()
    if ref_cal.empty:
        raise RuntimeError("Reference calibration sample is empty.")
    ret_q01 = float(ref_cal["ret"].quantile(0.01))
    vol_q95 = float(ref_cal["vol20"].dropna().quantile(0.95))
    all_events = build_event_dates(
        ref=ref[["date", "ret", "vol20", "dd20"]],
        test_start=pd.to_datetime("2018-01-01"),
        ret_q01=ret_q01,
        vol_q95=vol_q95,
    )
    events = _filter_events_between(
        events=all_events,
        start=test_start,
        end=(test_end if test_end is not None else pd.Timestamp(ref["date"].max())),
    )
    events_cal = _filter_events_between(
        events=all_events,
        start=pd.to_datetime("2018-01-01"),
        end=calibration_end,
    )

    metrics_rows: list[dict[str, object]] = []
    events_rows: list[dict[str, object]] = []
    ts_rows: list[pd.DataFrame] = []
    elig_rows: list[dict[str, object]] = []
    latest_level_rows: list[dict[str, object]] = []

    log("step: sector_loop")
    for sector in sorted(df["sector"].dropna().astype(str).unique()):
        sdf = df[df["sector"] == sector].sort_values("date").reset_index(drop=True)
        cal = sdf[sdf["date"] <= calibration_end].copy()
        test = sdf[sdf["date"] >= test_start].copy()
        if test_end is not None:
            test = test[test["date"] <= test_end].copy()
        test = test.reset_index(drop=True)
        n_assets_med = float(test["n_assets"].median()) if not test.empty else 0.0
        is_eligible = bool(
            (n_assets_med >= float(args.min_sector_assets))
            and (len(cal) >= int(args.min_cal_days))
            and (len(test) >= int(args.min_test_days))
        )
        reason = "ok" if is_eligible else "insufficient_assets_or_history"
        elig_rows.append(
            {
                "sector": sector,
                "eligible": is_eligible,
                "reason": reason,
                "n_days_cal": int(len(cal)),
                "n_days_test": int(len(test)),
                "n_assets_median_test": n_assets_med,
            }
        )
        if not is_eligible:
            continue

        policy_used = str(args.alert_policy)
        policy_meta: dict[str, float] = {}
        if policy_used == "regime_auto":
            policy_used, policy_meta = choose_auto_policy_for_sector(
                cal=cal,
                events_cal=events_cal,
                params=param_cfg,
                candidates=auto_candidates,
            )
        layered_signals, layered_meta = build_layered_alerts_for_sector(
            cal=cal,
            test=test,
            policy=policy_used,
            params=param_cfg,
        )
        alert_meta = dict(layered_meta)
        alert_meta = {
            **alert_meta,
            **policy_meta,
            "policy_used": policy_used,
        }
        two_layer_on = str(args.two_layer_mode).lower() == "on"
        test["alert_fast"] = layered_signals["fast_alert"].astype(bool)
        test["alert_confirmed"] = layered_signals["confirmed_alert"].astype(bool)
        test["alert_state_fast"] = layered_signals["fast_state"].astype(bool)
        test["alert_state_confirmed"] = layered_signals["confirmed_state"].astype(bool)
        test["alert_motor"] = test["alert_confirmed"] if two_layer_on else test["alert_fast"]
        test["alert_level"], level_meta = build_level_series_for_sector(cal=cal, test=test)
        test["alert_vol95"] = test["vol20"] >= vol_q95
        test["alert_ret1"] = test["ret"] <= ret_q01
        test["eligible"] = True
        test["policy_used"] = policy_used
        ts_rows.append(test)
        if not test.empty:
            last = test.iloc[-1]
            i_last = int(len(test) - 1)
            i_ref = int(max(0, i_last - 5))
            score_delta_5d = float(test["sector_score"].iloc[i_last] - test["sector_score"].iloc[i_ref])
            tail30 = test["alert_level"].astype(str).str.lower().tail(30).tolist()
            level_changes_30d = int(sum(1 for i in range(1, len(tail30)) if tail30[i] != tail30[i - 1]))
            action_row = build_action_plan_row(
                level=str(last["alert_level"]),
                score=float(last["sector_score"]),
                score_delta_5d=score_delta_5d,
                mean_confidence=float(last["mean_confidence"]),
                level_changes_30d=level_changes_30d,
                share_unstable=float(last["share_unstable"]),
                share_transition=float(last["share_transition"]),
            )
            latest_level_rows.append(
                {
                    "sector": sector,
                    "date": pd.Timestamp(last["date"]).date().isoformat(),
                    "n_assets": int(last["n_assets"]),
                    "alert_level": str(last["alert_level"]),
                    "sector_score": float(last["sector_score"]),
                    "share_transition": float(last["share_transition"]),
                    "share_unstable": float(last["share_unstable"]),
                    "mean_confidence": float(last["mean_confidence"]),
                    "score_delta_5d": score_delta_5d,
                    "level_changes_30d": level_changes_30d,
                    "level_q70": float(level_meta["sector_score_q70"]),
                    "level_q85": float(level_meta["sector_score_q85"]),
                    "policy_used": str(policy_used),
                    **action_row,
                }
            )

        for ev_name, ev_dates in events.items():
            for L in lookbacks:
                motor_ev = evaluate_alerts(test["date"], test["alert_motor"], ev_dates, lookback_days=L)
                fast_ev = evaluate_alerts(test["date"], test["alert_fast"], ev_dates, lookback_days=L)
                confirmed_ev = evaluate_alerts(test["date"], test["alert_confirmed"], ev_dates, lookback_days=L)
                b1_ev = evaluate_alerts(test["date"], test["alert_vol95"], ev_dates, lookback_days=L)
                b2_ev = evaluate_alerts(test["date"], test["alert_ret1"], ev_dates, lookback_days=L)
                motor_alert_episodes = int(_entry_alert(test["alert_motor"]).sum())
                rnd_iid = None
                rnd_block = None
                if args.random_baseline_method in {"iid", "both"}:
                    rnd_iid = random_baseline_distribution(
                        dates=test["date"],
                        n_alert_days=motor_alert_episodes,
                        event_dates=ev_dates,
                        lookback_days=L,
                        n_boot=int(args.n_random),
                        seed=17 + L,
                        method="iid",
                        block_size=int(args.random_block_size),
                    )
                if args.random_baseline_method in {"block", "both"}:
                    rnd_block = random_baseline_distribution(
                        dates=test["date"],
                        n_alert_days=motor_alert_episodes,
                        event_dates=ev_dates,
                        lookback_days=L,
                        n_boot=int(args.n_random),
                        seed=170 + L,
                        method="block",
                        block_size=int(args.random_block_size),
                    )

                p_vs_random_iid = (
                    float((rnd_iid["recall"] >= motor_ev.recall).mean())
                    if (rnd_iid is not None and np.isfinite(motor_ev.recall))
                    else float("nan")
                )
                p_vs_random_block = (
                    float((rnd_block["recall"] >= motor_ev.recall).mean())
                    if (rnd_block is not None and np.isfinite(motor_ev.recall))
                    else float("nan")
                )
                if args.random_baseline_method == "iid":
                    p_vs_random = p_vs_random_iid
                    rnd_ref = rnd_iid
                else:
                    p_vs_random = p_vs_random_block
                    rnd_ref = rnd_block if rnd_block is not None else rnd_iid

                rnd_recall_mean = float(rnd_ref["recall"].mean()) if rnd_ref is not None else float("nan")
                rnd_recall_p95 = float(rnd_ref["recall"].quantile(0.95)) if rnd_ref is not None else float("nan")
                rnd_recall_ci_low = float(rnd_ref["recall"].quantile(0.025)) if rnd_ref is not None else float("nan")
                rnd_recall_ci_high = float(rnd_ref["recall"].quantile(0.975)) if rnd_ref is not None else float("nan")
                for model_name, evm in [
                    ("motor", motor_ev),
                    ("motor_fast", fast_ev),
                    ("motor_confirmed", confirmed_ev),
                    ("baseline_vol95", b1_ev),
                    ("baseline_ret1", b2_ev),
                ]:
                    metrics_rows.append(
                        {
                            "sector": sector,
                            "n_assets_median_test": n_assets_med,
                            "event_def": ev_name,
                            "lookback_days": L,
                            "model": model_name,
                            "recall": evm.recall,
                            "precision": evm.precision,
                            "false_alarm_per_year": evm.false_alarm_per_year,
                            "mean_lead_days": evm.mean_lead_days,
                            "coincident_rate": evm.coincident_rate,
                            "n_events": evm.n_events,
                            "n_alert_days": evm.n_alert_days,
                            "n_false_alert_days": evm.n_false_alert_days,
                            "n_alert_episodes": evm.n_alert_episodes,
                            "n_false_alert_episodes": evm.n_false_alert_episodes,
                            "p_vs_random_recall": p_vs_random if model_name in {"motor", "motor_confirmed"} else float("nan"),
                            "p_vs_random_recall_iid": p_vs_random_iid if model_name in {"motor", "motor_confirmed"} else float("nan"),
                            "p_vs_random_recall_block": p_vs_random_block if model_name in {"motor", "motor_confirmed"} else float("nan"),
                            "random_recall_mean": rnd_recall_mean,
                            "random_recall_p95": rnd_recall_p95,
                            "random_recall_ci_low": rnd_recall_ci_low,
                            "random_recall_ci_high": rnd_recall_ci_high,
                        }
                    )

            # event-level records for 1d / 5d
            d = test[["date", "alert_motor"]].copy().reset_index(drop=True)
            d_idx = {pd.Timestamp(x): i for i, x in enumerate(pd.to_datetime(d["date"]))}
            for ev in ev_dates:
                ev_ts = pd.Timestamp(ev)
                if ev_ts not in d_idx:
                    continue
                i = d_idx[ev_ts]
                i1 = i - 1
                lo5 = max(0, i - 5)
                hi5 = i - 1
                alert_1d = bool(i1 >= 0 and bool(d.loc[i1, "alert_motor"]))
                w5 = d.loc[lo5:hi5].copy() if hi5 >= lo5 else d.iloc[0:0].copy()
                first5 = w5[w5["alert_motor"]]
                first5_date = pd.to_datetime(first5["date"]).iloc[0] if not first5.empty else pd.NaT
                events_rows.append(
                    {
                        "sector": sector,
                        "event_def": ev_name,
                        "event_date": ev_ts.date().isoformat(),
                        "alert_1d_before": alert_1d,
                        "alert_week_before": bool(w5["alert_motor"].any()) if not w5.empty else False,
                        "first_alert_in_week": first5_date.date().isoformat() if pd.notna(first5_date) else "",
                        "lead_days_week": int(i - d_idx[pd.Timestamp(first5_date)]) if pd.notna(first5_date) else "",
                        "alert_policy": args.alert_policy,
                        "meta": json.dumps(alert_meta, ensure_ascii=True),
                    }
                )

    if not metrics_rows:
        raise RuntimeError("No eligible sectors produced metrics. Lower --min-sector-assets or check data.")

    metrics_df = pd.DataFrame(metrics_rows)
    events_df = pd.DataFrame(events_rows)
    elig_df = pd.DataFrame(elig_rows)
    ts_df = pd.concat(ts_rows, axis=0, ignore_index=True) if ts_rows else pd.DataFrame()
    latest_levels_df = pd.DataFrame(latest_level_rows)
    if not latest_levels_df.empty:
        latest_levels_df = latest_levels_df.sort_values(["alert_level", "sector_score"], ascending=[True, False])

    metrics_df.to_csv(outdir / "sector_metrics_summary.csv", index=False)
    events_df.to_csv(outdir / "sector_events_with_first_alert.csv", index=False)
    elig_df.to_csv(outdir / "sector_eligibility.csv", index=False)
    ts_df.to_csv(outdir / "sector_daily_signals.csv", index=False)
    latest_levels_df.to_csv(outdir / "sector_alert_levels_latest.csv", index=False)

    # Ranking by anticipation quality. Prefer L=5; fallback to smallest requested lookback.
    rank_lookback = 5 if 5 in lookbacks else int(min(lookbacks))
    draw5 = metrics_df[
        (metrics_df["model"] == "motor")
        & (metrics_df["event_def"] == "drawdown20")
        & (metrics_df["lookback_days"] == rank_lookback)
    ][
        [
            "sector",
            "recall",
            "precision",
            "false_alarm_per_year",
            "p_vs_random_recall",
            "p_vs_random_recall_iid",
            "p_vs_random_recall_block",
        ]
    ].rename(
        columns={
            "recall": "drawdown_recall_l5",
            "precision": "drawdown_precision_l5",
            "false_alarm_per_year": "drawdown_false_alarm_l5",
            "p_vs_random_recall": "drawdown_p_vs_random_l5",
            "p_vs_random_recall_iid": "drawdown_p_vs_random_iid_l5",
            "p_vs_random_recall_block": "drawdown_p_vs_random_block_l5",
        }
    )
    ret5 = metrics_df[
        (metrics_df["model"] == "motor")
        & (metrics_df["event_def"] == "ret_tail")
        & (metrics_df["lookback_days"] == rank_lookback)
    ][["sector", "recall", "precision"]].rename(
        columns={
            "recall": "ret_tail_recall_l5",
            "precision": "ret_tail_precision_l5",
        }
    )
    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date][
        ["sector", "n_assets", "share_transition", "share_unstable", "mean_confidence", "sector_score"]
    ].copy()
    rank = draw5.merge(ret5, on="sector", how="left").merge(latest, on="sector", how="left")
    rank = rank.merge(
        elig_df[["sector", "eligible", "n_assets_median_test"]],
        on="sector",
        how="left",
    )
    rank["composite_score"] = (
        0.55 * rank["drawdown_recall_l5"]
        + 0.15 * rank["ret_tail_recall_l5"].fillna(0.0)
        + 0.20 * rank["drawdown_precision_l5"]
        + 0.10 * rank["ret_tail_precision_l5"].fillna(0.0)
        - 0.01 * rank["drawdown_false_alarm_l5"]
    )
    rank["rank_lookback_days"] = int(rank_lookback)
    rank = rank.sort_values(
        ["eligible", "composite_score", "drawdown_recall_l5"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    rank.to_csv(outdir / "sector_rank_l5.csv", index=False)
    if rank_lookback != 5:
        rank.to_csv(outdir / f"sector_rank_l{rank_lookback}.csv", index=False)

    # Report
    lines: list[str] = []
    lines.append("Sector Event Study Validation - Motor de Regimes")
    lines.append(f"generated_at_utc: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"calibration_end: {calibration_end.date().isoformat()}")
    lines.append(f"test_start: {test_start.date().isoformat()}")
    lines.append(f"test_end: {test_end.date().isoformat() if test_end is not None else 'latest'}")
    lines.append(f"alert_policy: {args.alert_policy}")
    lines.append(f"n_random: {int(args.n_random)}")
    lines.append(f"random_baseline_method: {args.random_baseline_method}")
    lines.append(f"random_block_size: {int(args.random_block_size)}")
    lines.append(f"q_unstable: {param_cfg['q_unstable']:.4f}")
    lines.append(f"q_transition: {param_cfg['q_transition']:.4f}")
    lines.append(f"q_confidence: {param_cfg['q_confidence']:.4f}")
    lines.append(f"q_confidence_guarded: {param_cfg['q_confidence_guarded']:.4f}")
    lines.append(f"q_score_balanced: {param_cfg['q_score_balanced']:.4f}")
    lines.append(f"q_score_guarded: {param_cfg['q_score_guarded']:.4f}")
    lines.append(f"confirm_n: {int(param_cfg['confirm_n'])}")
    lines.append(f"confirm_m: {int(param_cfg['confirm_m'])}")
    lines.append(f"min_alert_gap_days: {int(param_cfg['min_alert_gap_days'])}")
    lines.append(f"two_layer_mode: {str(args.two_layer_mode)}")
    lines.append(f"auto_candidates: {','.join(auto_candidates)}")
    lines.append(f"min_sector_assets: {int(args.min_sector_assets)}")
    lines.append(f"min_cal_days: {int(args.min_cal_days)}")
    lines.append(f"min_test_days: {int(args.min_test_days)}")
    lines.append("")
    lines.append("Global event thresholds:")
    lines.append(f"- ret_q01: {ret_q01:.6f}")
    lines.append(f"- vol20_q95: {vol_q95:.6f}")
    lines.append("")
    lines.append("Sector eligibility:")
    lines.append(f"- eligible: {int(elig_df['eligible'].sum())}")
    lines.append(f"- total: {int(elig_df.shape[0])}")
    lines.append("Eventos avaliados:")
    for ev_name in sorted(events.keys()):
        lines.append(f"- {ev_name}: {int(len(events.get(ev_name, [])))}")
    lines.append("")

    lines.append(f"Top sectors (drawdown recall L={rank_lookback}, motor):")
    top = rank[rank["eligible"]].head(8)
    for _, r in top.iterrows():
        lines.append(
            f"- {r['sector']}: recall={_format_metric(float(r['drawdown_recall_l5']))}, "
            f"false_alarm/ano={_format_metric(float(r['drawdown_false_alarm_l5']))}, "
            f"precision={_format_metric(float(r['drawdown_precision_l5']))}, "
            f"p_vs_random={_format_metric(float(r['drawdown_p_vs_random_l5']))}, "
            f"p_iid={_format_metric(float(r.get('drawdown_p_vs_random_iid_l5', float('nan'))))}, "
            f"p_block={_format_metric(float(r.get('drawdown_p_vs_random_block_l5', float('nan'))))}, "
            f"assets_med={_format_metric(float(r['n_assets_median_test']))}"
        )
    lines.append("")

    if not latest_levels_df.empty:
        lines.append("Niveis atuais por setor (ultimo dia):")
        level_counts = latest_levels_df["alert_level"].value_counts().to_dict()
        for k in ["vermelho", "amarelo", "verde"]:
            lines.append(f"- {k}: {int(level_counts.get(k, 0))}")
        top_red = latest_levels_df[latest_levels_df["alert_level"] == "vermelho"].head(6)
        if not top_red.empty:
            lines.append("Setores em vermelho:")
            for _, rr in top_red.iterrows():
                lines.append(
                    f"- {rr['sector']}: score={_format_metric(float(rr['sector_score']))}, "
                    f"unstable={_format_metric(float(rr['share_unstable']))}, "
                    f"transition={_format_metric(float(rr['share_transition']))}, "
                    f"conf={_format_metric(float(rr['mean_confidence']))}"
                )
        if "policy_used" in latest_levels_df.columns:
            lines.append("Politica usada por setor (ultimo dia):")
            p_counts = latest_levels_df["policy_used"].astype(str).value_counts().to_dict()
            for k, v in sorted(p_counts.items(), key=lambda x: x[0]):
                lines.append(f"- {k}: {int(v)}")
        lines.append("")

    lines.append("Resumo por setor (L=1 e L=5, motor):")
    for sec in sorted(metrics_df["sector"].unique()):
        m = metrics_df[
            (metrics_df["sector"] == sec)
            & (metrics_df["model"] == "motor")
            & (metrics_df["event_def"] == "drawdown20")
            & (metrics_df["lookback_days"].isin([1, 5]))
        ].sort_values("lookback_days")
        if m.empty:
            continue
        lines.append(f"Setor: {sec}")
        for _, rr in m.iterrows():
            lines.append(
                f"- L={int(rr['lookback_days'])}: recall={_format_metric(float(rr['recall']))}, "
                f"precision={_format_metric(float(rr['precision']))}, "
                f"false_alarm/ano={_format_metric(float(rr['false_alarm_per_year']))}, "
                f"p_vs_random={_format_metric(float(rr['p_vs_random_recall']))}, "
                f"p_iid={_format_metric(float(rr.get('p_vs_random_recall_iid', float('nan'))))}, "
                f"p_block={_format_metric(float(rr.get('p_vs_random_recall_block', float('nan'))))}"
            )
    (outdir / "report_sector_event_study.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    cfg = {
        "tickers_file": str(ROOT / args.tickers_file),
        "assets_dir": str(assets_dir),
        "prices_dir": str(prices_dir),
        "sector_map_files": [str(p) for p in sector_map_paths],
        "calibration_end": calibration_end.date().isoformat(),
        "test_start": test_start.date().isoformat(),
        "test_end": (test_end.date().isoformat() if test_end is not None else ""),
        "lookbacks": lookbacks,
        "n_random": int(args.n_random),
        "random_baseline_method": str(args.random_baseline_method),
        "random_block_size": int(args.random_block_size),
        "q_unstable": float(param_cfg["q_unstable"]),
        "q_transition": float(param_cfg["q_transition"]),
        "q_confidence": float(param_cfg["q_confidence"]),
        "q_confidence_guarded": float(param_cfg["q_confidence_guarded"]),
        "q_score_balanced": float(param_cfg["q_score_balanced"]),
        "q_score_guarded": float(param_cfg["q_score_guarded"]),
        "confirm_n": int(param_cfg["confirm_n"]),
        "confirm_m": int(param_cfg["confirm_m"]),
        "min_alert_gap_days": int(param_cfg["min_alert_gap_days"]),
        "two_layer_mode": str(args.two_layer_mode),
        "auto_candidates": auto_candidates,
        "min_sector_assets": int(args.min_sector_assets),
        "min_cal_days": int(args.min_cal_days),
        "min_test_days": int(args.min_test_days),
        "alert_policy": str(args.alert_policy),
    }
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "outdir": str(outdir),
                "sectors_total": int(elig_df.shape[0]),
                "sectors_eligible": int(elig_df["eligible"].sum()),
                "events_ret_tail": int(len(events.get("ret_tail", []))),
                "events_drawdown20": int(len(events.get("drawdown20", []))),
                "events_stress_combo": int(len(events.get("stress_combo", []))),
                "events_vol_spike20": int(len(events.get("vol_spike20", []))),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
