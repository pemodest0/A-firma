#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bench.validation.run_profit_alpha_improvement_suite import _safe_float, _write_json  # noqa: E402
from scripts.bench.validation.run_profit_marketmode_criticality_suite import (  # noqa: E402
    _build_structure_layers,
    _classify_official_structural_regime,
    build_official_mode_allocations,
)


def _run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_return_series(prices_dir: Path, ticker: str) -> pd.Series:
    path = prices_dir / f"{ticker}.csv"
    if not path.exists():
        return pd.Series(dtype=float)
    frame = pd.read_csv(path)
    if frame.empty or "date" not in frame.columns:
        return pd.Series(dtype=float)
    date_index = pd.to_datetime(frame["date"], errors="coerce").dt.tz_localize(None)
    if "r" in frame.columns:
        values = pd.to_numeric(frame["r"], errors="coerce")
    elif "price" in frame.columns:
        prices = pd.to_numeric(frame["price"], errors="coerce")
        values = prices.pct_change()
    else:
        return pd.Series(dtype=float)
    series = pd.Series(values.to_numpy(dtype=float), index=date_index, dtype=float)
    series = series[~series.index.isna()].sort_index()
    return series.dropna()


def _future_window_metrics(returns: pd.Series, horizon: int) -> pd.DataFrame:
    values = pd.to_numeric(returns, errors="coerce").astype(float)
    idx = values.index
    future_ret = pd.Series(index=idx, dtype=float)
    future_dd = pd.Series(index=idx, dtype=float)
    future_vol = pd.Series(index=idx, dtype=float)
    arr = values.to_numpy(dtype=float)
    for pos, _dt in enumerate(idx):
        window = arr[pos + 1 : pos + 1 + int(horizon)]
        window = window[np.isfinite(window)]
        if window.size < max(5, horizon // 4):
            continue
        wealth = np.cumprod(1.0 + window)
        peak = np.maximum.accumulate(np.r_[1.0, wealth[:-1]])
        dd = wealth / peak - 1.0
        future_ret.iloc[pos] = float(wealth[-1] - 1.0)
        future_dd.iloc[pos] = float(np.nanmin(dd)) if dd.size else 0.0
        future_vol.iloc[pos] = float(np.nanstd(window, ddof=0) * np.sqrt(252.0))
    return pd.DataFrame(
        {
            "future_return": future_ret,
            "future_max_drawdown": future_dd,
            "future_vol": future_vol,
        },
        index=idx,
    )


def _change_points(series: pd.Series) -> pd.DatetimeIndex:
    if series.empty:
        return pd.DatetimeIndex([])
    clean = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(int)
    return clean.index[clean != clean.shift(1)]


def _smooth_binary(series: pd.Series, min_run: int) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(int)
    if int(min_run) <= 1 or clean.empty:
        return clean
    values = clean.to_numpy(copy=True)
    n = int(values.shape[0])
    i = 0
    while i < n:
        j = i + 1
        while j < n and values[j] == values[i]:
            j += 1
        run_len = j - i
        if run_len < int(min_run):
            left = values[i - 1] if i > 0 else None
            right = values[j] if j < n else None
            if left is not None:
                values[i:j] = left
            elif right is not None:
                values[i:j] = right
        i = j
    return pd.Series(values, index=clean.index, dtype=int)


def _apply_cooldown(series: pd.Series, cooldown: int) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(int)
    if int(cooldown) <= 0 or clean.empty:
        return clean
    values = clean.to_numpy(copy=True)
    last_on = -int(cooldown) - 1
    for i, val in enumerate(values):
        if int(val) != 1:
            continue
        if i - last_on <= int(cooldown):
            values[i] = 0
        else:
            last_on = i
    return pd.Series(values, index=clean.index, dtype=int)


def _turn_hit_rate(proxy_changes: pd.DatetimeIndex, engine_changes: pd.DatetimeIndex, window_days: int) -> dict[str, Any]:
    if len(proxy_changes) == 0:
        return {"hits": 0, "total": 0, "hit_rate": 0.0, "false_alarms": 0, "false_alarm_rate": 0.0}
    hits = 0
    for dt in proxy_changes:
        start = dt - pd.Timedelta(int(window_days), unit="D")
        end = dt + pd.Timedelta(int(window_days), unit="D")
        if ((engine_changes >= start) & (engine_changes <= end)).any():
            hits += 1
    false_alarms = 0
    for dt in engine_changes:
        start = dt - pd.Timedelta(int(window_days), unit="D")
        end = dt + pd.Timedelta(int(window_days), unit="D")
        if not ((proxy_changes >= start) & (proxy_changes <= end)).any():
            false_alarms += 1
    total = int(len(proxy_changes))
    return {
        "hits": int(hits),
        "total": total,
        "hit_rate": float(hits / total) if total else 0.0,
        "false_alarms": int(false_alarms),
        "false_alarm_rate": float(false_alarms / max(len(engine_changes), 1)),
    }


def _binary_tail_event(series: pd.Series, *, tail: str, quantile: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if values.empty:
        return pd.Series(dtype=int)
    threshold = float(values.quantile(float(quantile)))
    aligned = pd.to_numeric(series, errors="coerce").astype(float)
    if tail == "upper":
        return (aligned >= threshold).astype(int)
    return (aligned <= threshold).astype(int)


def _roc_auc(y_true: pd.Series, scores: pd.Series) -> float | None:
    frame = pd.DataFrame({"y": y_true, "score": scores}).dropna()
    if frame.empty:
        return None
    y = pd.to_numeric(frame["y"], errors="coerce").astype(int)
    s = pd.to_numeric(frame["score"], errors="coerce").astype(float)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        return None
    ranks = s.rank(method="average")
    rank_sum = float(ranks[y == 1].sum())
    auc = (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def _balanced_accuracy(y_true: pd.Series, scores: pd.Series, threshold: float) -> float | None:
    frame = pd.DataFrame({"y": y_true, "score": scores}).dropna()
    if frame.empty:
        return None
    y = pd.to_numeric(frame["y"], errors="coerce").astype(int)
    pred = (pd.to_numeric(frame["score"], errors="coerce").astype(float) >= float(threshold)).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tpr = tp / (tp + fn) if (tp + fn) else np.nan
    tnr = tn / (tn + fp) if (tn + fp) else np.nan
    if not np.isfinite(tpr) or not np.isfinite(tnr):
        return None
    return float((tpr + tnr) / 2.0)


def _classify_regime_series(structure_daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dt, row in structure_daily.iterrows():
        regime = _classify_official_structural_regime(
            as_of_date=str(pd.Timestamp(dt).date()),
            criticality_value=_safe_float(row.get("criticality"), 0.5),
            structural_stress_value=_safe_float(row.get("structural_stress"), 0.5),
            market_mode_share_pct_value=_safe_float(row.get("market_mode_share_pct"), 0.5),
        )
        regime["date"] = pd.Timestamp(dt).tz_localize(None)
        rows.append(regime)
    return pd.DataFrame(rows).set_index("date").sort_index()


def _regime_risk_score(regime_series: pd.Series) -> pd.Series:
    mapping = {
        "dispersion": 0.15,
        "stable": 0.30,
        "transition": 0.70,
        "stress": 1.00,
    }
    return regime_series.map(mapping).fillna(0.5).astype(float)


def _build_signal_frame(structure_daily: pd.DataFrame) -> pd.DataFrame:
    regimes = _classify_regime_series(structure_daily)
    frame = pd.DataFrame(index=structure_daily.index)
    frame["criticality"] = pd.to_numeric(structure_daily.get("criticality"), errors="coerce").astype(float)
    frame["structural_stress"] = pd.to_numeric(structure_daily.get("structural_stress"), errors="coerce").astype(float)
    frame["market_mode_share_pct"] = pd.to_numeric(structure_daily.get("market_mode_share_pct"), errors="coerce").astype(float)
    frame["composite_structural"] = (
        0.45 * frame["criticality"] + 0.35 * frame["structural_stress"] + 0.20 * frame["market_mode_share_pct"]
    )
    frame["regime_risk"] = _regime_risk_score(regimes["regime"])
    frame["regime_label"] = regimes["regime"].astype(str)
    return frame.sort_index()


def _score_signal_frame(signal_frame: pd.DataFrame, event_labels: pd.Series, *, lags: tuple[int, ...], target_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for signal_name in ["criticality", "structural_stress", "market_mode_share_pct", "composite_structural", "regime_risk"]:
        signal = pd.to_numeric(signal_frame.get(signal_name), errors="coerce").astype(float)
        for lag in lags:
            shifted = signal.shift(int(lag))
            rows.append(
                {
                    "target": target_name,
                    "signal": signal_name,
                    "lag": int(lag),
                    "roc_auc": _roc_auc(event_labels, shifted),
                    "balanced_accuracy": _balanced_accuracy(event_labels, shifted, threshold=0.5),
                    "mean_signal": float(shifted.dropna().mean()) if shifted.notna().any() else None,
                }
            )
    return rows


def _joint_signal_permutation(signal_frame: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    numeric_cols = ["criticality", "structural_stress", "market_mode_share_pct", "composite_structural", "regime_risk"]
    rng = np.random.default_rng(int(seed))
    values = signal_frame[numeric_cols].to_numpy(dtype=float)
    perm = rng.permutation(values.shape[0])
    out = pd.DataFrame(values[perm], index=signal_frame.index, columns=numeric_cols)
    out["regime_label"] = signal_frame["regime_label"].iloc[perm].to_numpy()
    return out


def _target_catalog(prices_dir: Path) -> dict[str, pd.Series]:
    raw = {
        "BTC-USD": ["BTC-USD"],
        "ETH-USD": ["ETH-USD"],
        "SOL-USD": ["SOL-USD"],
        "SPY": ["SPY"],
        "QQQ": ["QQQ"],
        "PETR4.SA": ["PETR4.SA"],
        "crypto_blend": ["BTC-USD", "ETH-USD", "SOL-USD"],
        "global_blend": ["SPY", "QQQ", "BTC-USD"],
        "brazil_crypto_blend": ["PETR4.SA", "BTC-USD", "ETH-USD"],
    }
    out: dict[str, pd.Series] = {}
    for name, members in raw.items():
        series_list = []
        for ticker in members:
            series = _read_return_series(prices_dir, ticker)
            if not series.empty:
                series_list.append(series.rename(ticker))
        if not series_list:
            continue
        if len(series_list) == 1:
            out[name] = series_list[0].astype(float)
            continue
        joined = pd.concat(series_list, axis=1).dropna(how="all")
        out[name] = joined.mean(axis=1, skipna=True).dropna().astype(float)
    return out


def _target_future_catalog(targets: dict[str, pd.Series], *, horizon: int) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for name, returns in targets.items():
        frame = _future_window_metrics(returns, horizon=int(horizon))
        if frame.empty:
            continue
        out[name] = frame
    return out


def _build_target_event_matrix(targets: dict[str, pd.Series], *, horizon: int) -> tuple[dict[str, pd.Series], list[dict[str, Any]]]:
    events: dict[str, pd.Series] = {}
    rows: list[dict[str, Any]] = []
    for name, returns in targets.items():
        future = _future_window_metrics(returns, horizon=int(horizon))
        drawdown_event = _binary_tail_event(future["future_max_drawdown"], tail="lower", quantile=0.20)
        vol_event = _binary_tail_event(future["future_vol"], tail="upper", quantile=0.80)
        combined = (drawdown_event | vol_event).astype(int)
        combined.name = name
        events[name] = combined
        rows.append(
            {
                "target": name,
                "rows": int(combined.dropna().shape[0]),
                "positive_rate": float(pd.to_numeric(combined, errors="coerce").mean()) if combined.notna().any() else None,
            }
        )
    return events, rows


def _load_cached_signal_frame(cache_root: Path) -> pd.DataFrame | None:
    path = cache_root / "signal_frame.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if frame.empty or "date" not in frame.columns:
        return None
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.tz_localize(None)
    frame = frame.dropna(subset=["date"]).set_index("date").sort_index()
    return frame


def _save_signal_frame(cache_root: Path, signal_frame: pd.DataFrame) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    frame = signal_frame.reset_index().rename(columns={"index": "date"})
    frame.to_csv(cache_root / "signal_frame.csv", index=False)


def _load_cached_target_events(cache_root: Path) -> tuple[dict[str, pd.Series], list[dict[str, Any]]] | None:
    matrix_path = cache_root / "target_events.csv"
    summary_path = cache_root / "target_event_summary.json"
    if not matrix_path.exists() or not summary_path.exists():
        return None
    frame = pd.read_csv(matrix_path)
    if frame.empty or "date" not in frame.columns:
        return None
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.tz_localize(None)
    frame = frame.dropna(subset=["date"]).set_index("date").sort_index()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    events = {col: pd.to_numeric(frame[col], errors="coerce").astype(float) for col in frame.columns}
    return events, summary if isinstance(summary, list) else []


def _save_target_events(cache_root: Path, events: dict[str, pd.Series], summary_rows: list[dict[str, Any]]) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(events).sort_index()
    frame = frame.reset_index().rename(columns={"index": "date"})
    frame.to_csv(cache_root / "target_events.csv", index=False)
    (cache_root / "target_event_summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")


def _load_cached_null_rows(cache_root: Path, *, permutation_id: int) -> list[dict[str, Any]] | None:
    path = cache_root / "null_permutations" / f"perm_{int(permutation_id):03d}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else None


def _save_cached_null_rows(cache_root: Path, *, permutation_id: int, rows: list[dict[str, Any]]) -> None:
    path = cache_root / "null_permutations" / f"perm_{int(permutation_id):03d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _load_cached_operational_null_rows(cache_root: Path, *, permutation_id: int) -> list[dict[str, Any]] | None:
    path = cache_root / "operational_null_permutations" / f"perm_{int(permutation_id):03d}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else None


def _save_cached_operational_null_rows(cache_root: Path, *, permutation_id: int, rows: list[dict[str, Any]]) -> None:
    path = cache_root / "operational_null_permutations" / f"perm_{int(permutation_id):03d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _signal_rollup(real_rows: list[dict[str, Any]], null_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    signals = sorted({str(row.get("signal") or "") for row in real_rows if int(row.get("lag") or 0) == 1})
    rows: list[dict[str, Any]] = []
    real_frame = pd.DataFrame(real_rows)
    null_frame = pd.DataFrame(null_rows)
    for signal in signals:
        real_sub = real_frame[(real_frame["signal"] == signal) & (real_frame["lag"] == 1)].copy()
        lag_sub = real_frame[real_frame["signal"] == signal].copy()
        null_sub = null_frame[(null_frame["signal"] == signal) & (null_frame["lag"] == 1)].copy()
        if real_sub.empty:
            continue
        real_mean_auc = float(pd.to_numeric(real_sub["roc_auc"], errors="coerce").dropna().mean())
        real_best_target = (
            real_sub.sort_values("roc_auc", ascending=False).iloc[0]["target"]
            if pd.to_numeric(real_sub["roc_auc"], errors="coerce").notna().any()
            else ""
        )
        lag_span = float(
            pd.to_numeric(lag_sub["roc_auc"], errors="coerce").dropna().groupby(lag_sub["target"]).agg(lambda x: float(x.max() - x.min())).mean()
        )
        if null_sub.empty:
            null_mean_auc = None
            null_p95 = None
            exceed_rate = None
        else:
            perm_scores = (
                null_sub.groupby("permutation_id")["roc_auc"]
                .apply(lambda s: float(pd.to_numeric(s, errors="coerce").dropna().mean()))
                .to_list()
            )
            null_mean_auc = float(np.mean(perm_scores)) if perm_scores else None
            null_p95 = float(np.quantile(perm_scores, 0.95)) if perm_scores else None
            exceed_rate = float(np.mean([score >= real_mean_auc for score in perm_scores])) if perm_scores else None
        decision = "cut"
        if null_p95 is not None and real_mean_auc > null_p95 and lag_span <= 0.08:
            decision = "keep"
        elif null_mean_auc is not None and real_mean_auc > null_mean_auc and lag_span <= 0.12:
            decision = "recalibrate"
        rows.append(
            {
                "signal": signal,
                "real_mean_auc_lag1": real_mean_auc,
                "real_mean_balanced_accuracy_lag1": float(pd.to_numeric(real_sub["balanced_accuracy"], errors="coerce").dropna().mean()),
                "best_target": str(real_best_target),
                "lag_auc_span_mean": lag_span,
                "null_mean_auc_lag1": null_mean_auc,
                "null_p95_auc_lag1": null_p95,
                "null_exceedance_rate": exceed_rate,
                "decision": decision,
            }
        )
    return rows


def _target_rollup(real_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    frame = pd.DataFrame(real_rows)
    targets = sorted({str(row.get("target") or "") for row in real_rows if int(row.get("lag") or 0) == 1})
    for target in targets:
        sub = frame[(frame["target"] == target) & (frame["lag"] == 1)].copy()
        if sub.empty:
            continue
        best_row = sub.sort_values("roc_auc", ascending=False).iloc[0]
        rows.append(
            {
                "target": target,
                "best_signal": str(best_row["signal"]),
                "best_auc_lag1": _safe_float(best_row.get("roc_auc"), None),
                "best_balanced_accuracy_lag1": _safe_float(best_row.get("balanced_accuracy"), None),
                "mean_auc_lag1": float(pd.to_numeric(sub["roc_auc"], errors="coerce").dropna().mean()),
            }
        )
    return rows


def _sensitivity_rows(structure_daily: pd.DataFrame, *, epsilons: list[float]) -> list[dict[str, Any]]:
    base = _classify_regime_series(structure_daily)["regime"]
    thresholds = [0.42, 0.52, 0.55, 0.65, 0.70, 0.72, 0.82]
    near_threshold_mask = pd.Series(False, index=structure_daily.index)
    for column in ("criticality", "structural_stress", "market_mode_share_pct"):
        values = pd.to_numeric(structure_daily.get(column), errors="coerce").astype(float)
        for threshold in thresholds:
            near_threshold_mask = near_threshold_mask | ((values - threshold).abs() <= 0.05)
    rows: list[dict[str, Any]] = []
    shocks = [
        ("crit_up", {"criticality": 1.0}),
        ("crit_down", {"criticality": -1.0}),
        ("stress_up", {"structural_stress": 1.0}),
        ("stress_down", {"structural_stress": -1.0}),
        ("market_up", {"market_mode_share_pct": 1.0}),
        ("market_down", {"market_mode_share_pct": -1.0}),
        ("all_up", {"criticality": 1.0, "structural_stress": 1.0, "market_mode_share_pct": 1.0}),
        ("all_down", {"criticality": -1.0, "structural_stress": -1.0, "market_mode_share_pct": -1.0}),
    ]
    for epsilon in epsilons:
        for shock_name, spec in shocks:
            perturbed = structure_daily.copy()
            for column, direction in spec.items():
                perturbed[column] = pd.to_numeric(perturbed.get(column), errors="coerce").astype(float) + float(direction) * float(epsilon)
                perturbed[column] = pd.to_numeric(perturbed.get(column), errors="coerce").clip(0.0, 1.0)
            labels = _classify_regime_series(perturbed)["regime"]
            flips = (labels != base).astype(int)
            rows.append(
                {
                    "epsilon": float(epsilon),
                    "shock": shock_name,
                    "flip_rate": float(flips.mean()),
                    "flip_count": int(flips.sum()),
                    "flip_rate_near_threshold": float(flips[near_threshold_mask].mean()) if near_threshold_mask.any() else 0.0,
                    "flip_rate_far_from_threshold": float(flips[~near_threshold_mask].mean()) if (~near_threshold_mask).any() else 0.0,
                }
            )
    return rows


def _build_operational_risk(signal: pd.Series, *, threshold: float, min_run: int, cooldown: int) -> pd.Series:
    binary = (pd.to_numeric(signal, errors="coerce").astype(float).shift(1) >= float(threshold)).astype(float)
    binary = pd.to_numeric(binary, errors="coerce").fillna(0.0).astype(int)
    binary = _smooth_binary(binary, min_run=int(min_run))
    binary = _apply_cooldown(binary, cooldown=int(cooldown))
    return binary.astype(int)


def _operational_target_rows(
    signal_frame: pd.DataFrame,
    target_events: dict[str, pd.Series],
    target_future: dict[str, pd.DataFrame],
    *,
    thresholds: tuple[float, ...],
    min_run: int,
    cooldown: int,
    turn_window: int,
    date_mask: pd.Series | None = None,
    subset_name: str | None = None,
    permutation_id: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    signal_names = ["criticality", "structural_stress", "market_mode_share_pct", "composite_structural", "regime_risk"]
    for signal_name in signal_names:
        signal = pd.to_numeric(signal_frame.get(signal_name), errors="coerce").astype(float)
        for threshold in thresholds:
            risk_days = _build_operational_risk(signal, threshold=float(threshold), min_run=int(min_run), cooldown=int(cooldown))
            for target_name, event_labels in target_events.items():
                future = target_future.get(target_name)
                if future is None or future.empty:
                    continue
                event_aligned = pd.to_numeric(event_labels, errors="coerce").reindex(signal_frame.index)
                joined = pd.DataFrame(
                    {
                        "risk": risk_days,
                        "event": event_aligned,
                        "future_return": pd.to_numeric(future["future_return"], errors="coerce").reindex(signal_frame.index),
                        "future_max_drawdown": pd.to_numeric(future["future_max_drawdown"], errors="coerce").reindex(signal_frame.index),
                    }
                ).dropna()
                if date_mask is not None:
                    mask = pd.to_numeric(date_mask.reindex(joined.index), errors="coerce").fillna(0.0).astype(int) == 1
                    joined = joined.loc[mask]
                if joined.empty:
                    continue
                event_binary = _smooth_binary(joined["event"].fillna(0.0).astype(int), min_run=max(2, int(min_run)))
                proxy_changes = _change_points(event_binary)
                engine_changes = _change_points(joined["risk"])
                risk_mask = joined["risk"] == 1
                safe_mask = joined["risk"] == 0
                risk_count = int(risk_mask.sum())
                safe_count = int(safe_mask.sum())
                event_rate_risk = float(joined.loc[risk_mask, "event"].mean()) if risk_count else None
                event_rate_safe = float(joined.loc[safe_mask, "event"].mean()) if safe_count else None
                future_return_risk = float(joined.loc[risk_mask, "future_return"].mean()) if risk_count else None
                future_return_safe = float(joined.loc[safe_mask, "future_return"].mean()) if safe_count else None
                future_dd_risk = float(joined.loc[risk_mask, "future_max_drawdown"].mean()) if risk_count else None
                future_dd_safe = float(joined.loc[safe_mask, "future_max_drawdown"].mean()) if safe_count else None
                turn = _turn_hit_rate(proxy_changes, engine_changes, window_days=int(turn_window))
                rows.append(
                    {
                        "target": target_name,
                        "signal": signal_name,
                        "threshold": float(threshold),
                        "risk_days": risk_count,
                        "safe_days": safe_count,
                        "risk_coverage": float(risk_count / max(risk_count + safe_count, 1)),
                        "event_rate_risk": event_rate_risk,
                        "event_rate_safe": event_rate_safe,
                        "event_rate_spread": (
                            float(event_rate_risk - event_rate_safe)
                            if event_rate_risk is not None and event_rate_safe is not None
                            else None
                        ),
                        "future_return_risk_mean": future_return_risk,
                        "future_return_safe_mean": future_return_safe,
                        "future_return_spread": (
                            float(future_return_safe - future_return_risk)
                            if future_return_risk is not None and future_return_safe is not None
                            else None
                        ),
                        "future_max_drawdown_risk_mean": future_dd_risk,
                        "future_max_drawdown_safe_mean": future_dd_safe,
                        "drawdown_severity_gain": (
                            float(future_dd_safe - future_dd_risk)
                            if future_dd_risk is not None and future_dd_safe is not None
                            else None
                        ),
                        "subset": str(subset_name or "all"),
                        "turn_hits": int(turn["hits"]),
                        "turn_total": int(turn["total"]),
                        "turn_hit_rate": float(turn["hit_rate"]),
                        "false_alarms": int(turn["false_alarms"]),
                        "false_alarm_rate": float(turn["false_alarm_rate"]),
                        "permutation_id": permutation_id,
                    }
                )
    return rows


def _operational_rollup(real_rows: list[dict[str, Any]], null_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    real_frame = pd.DataFrame(real_rows)
    null_frame = pd.DataFrame(null_rows)
    if real_frame.empty:
        return rows
    if null_frame.empty:
        null_frame = pd.DataFrame(columns=["signal", "threshold", "permutation_id"])
    combos = (
        real_frame[["signal", "threshold"]]
        .drop_duplicates()
        .sort_values(["signal", "threshold"])
        .itertuples(index=False, name=None)
    )
    metric_cols = [
        "turn_hit_rate",
        "event_rate_spread",
        "future_return_spread",
        "drawdown_severity_gain",
        "false_alarm_rate",
    ]
    for signal_name, threshold in combos:
        real_sub = real_frame[(real_frame["signal"] == signal_name) & (real_frame["threshold"] == threshold)].copy()
        null_sub = null_frame[(null_frame["signal"] == signal_name) & (null_frame["threshold"] == threshold)].copy()
        if real_sub.empty:
            continue
        row: dict[str, Any] = {
            "signal": str(signal_name),
            "threshold": float(threshold),
            "targets": int(real_sub["target"].nunique()),
        }
        score = 0
        for metric in metric_cols:
            real_value = float(pd.to_numeric(real_sub[metric], errors="coerce").dropna().mean())
            row[f"real_mean_{metric}"] = real_value
            if null_sub.empty:
                row[f"null_mean_{metric}"] = None
                row[f"null_p95_{metric}"] = None
                row[f"null_p05_{metric}"] = None
                continue
            perm_scores = (
                null_sub.groupby("permutation_id")[metric]
                .apply(lambda s: float(pd.to_numeric(s, errors="coerce").dropna().mean()))
                .to_list()
            )
            if not perm_scores:
                row[f"null_mean_{metric}"] = None
                row[f"null_p95_{metric}"] = None
                row[f"null_p05_{metric}"] = None
                continue
            row[f"null_mean_{metric}"] = float(np.mean(perm_scores))
            row[f"null_p95_{metric}"] = float(np.quantile(perm_scores, 0.95))
            row[f"null_p05_{metric}"] = float(np.quantile(perm_scores, 0.05))
            if metric == "false_alarm_rate":
                if real_value < row[f"null_p05_{metric}"]:
                    score += 2
                elif real_value < row[f"null_mean_{metric}"]:
                    score += 1
            else:
                if real_value > row[f"null_p95_{metric}"]:
                    score += 2
                elif real_value > row[f"null_mean_{metric}"]:
                    score += 1
        decision = "cut"
        if score >= 6 and row.get("real_mean_future_return_spread", 0.0) > 0 and row.get("real_mean_drawdown_severity_gain", 0.0) > 0:
            decision = "keep"
        elif score >= 3 and row.get("real_mean_drawdown_severity_gain", 0.0) > 0:
            decision = "recalibrate"
        row["operational_score"] = int(score)
        row["operational_decision"] = decision
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Audita a extração estrutural do motor com múltiplos alvos futuros, nulo e sensibilidade.")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--crypto-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-meta", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-meta", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--observer-mode", choices=["major8", "all22"], default="all22")
    ap.add_argument("--horizon-days", type=int, default=21)
    ap.add_argument("--null-permutations", type=int, default=16)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--turn-window", type=int, default=5)
    ap.add_argument("--operational-thresholds", default="0.6,0.7,0.8")
    ap.add_argument("--operational-min-run", type=int, default=2)
    ap.add_argument("--operational-cooldown", type=int, default=3)
    ap.add_argument("--outdir-root", default="results/validation/profit_structural_extraction_audit")
    ap.add_argument("--cache-root", default="results/cache/profit_structural_extraction_audit")
    args = ap.parse_args()

    outdir = (ROOT / args.outdir_root / _run_id()).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    prices_dir = (ROOT / args.prices_dir).resolve()
    cache_root = (
        ROOT
        / args.cache_root
        / f"observer_{str(args.observer_mode)}__h{int(args.horizon_days)}__v2"
    ).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    signal_frame = _load_cached_signal_frame(cache_root)
    if signal_frame is None:
        official = build_official_mode_allocations(
            prices_dir=prices_dir,
            crypto_groups=(ROOT / args.crypto_groups).resolve(),
            crypto_meta=(ROOT / args.crypto_meta).resolve(),
            equity_groups=(ROOT / args.equity_groups).resolve(),
            equity_meta=(ROOT / args.equity_meta).resolve(),
            benchmark_crypto=str(args.benchmark_crypto),
            benchmark_equity=str(args.benchmark_equity),
        )
        context = dict(official["context"])
        observer_crypto_cols = (
            list(context["crypto_returns"].columns)
            if str(args.observer_mode) == "all22"
            else list(context["crypto_tiers"]["crypto_major8"])
        )
        structure_daily, _, _, _ = _build_structure_layers(
            context,
            observer_context=context,
            observer_crypto_cols=observer_crypto_cols,
        )
        signal_frame = _build_signal_frame(structure_daily)
        _save_signal_frame(cache_root, signal_frame)
    else:
        structure_daily = signal_frame[["criticality", "structural_stress", "market_mode_share_pct"]].copy()
        crypto_meta_frame = pd.read_csv((ROOT / args.crypto_meta).resolve())
        cached_count = int(crypto_meta_frame.shape[0]) if not crypto_meta_frame.empty else (21 if str(args.observer_mode) == "all22" else 8)
        if str(args.observer_mode) != "all22":
            cached_count = min(cached_count, 8)
        observer_crypto_cols = [f"cached_{i}" for i in range(cached_count)]

    cached_target_bundle = _load_cached_target_events(cache_root)
    if cached_target_bundle is None:
        targets = _target_catalog(prices_dir)
        target_events, target_event_summary = _build_target_event_matrix(targets, horizon=int(args.horizon_days))
        _save_target_events(cache_root, target_events, target_event_summary)
    else:
        target_events, target_event_summary = cached_target_bundle
        targets = _target_catalog(prices_dir)

    target_future = _target_future_catalog(targets, horizon=int(args.horizon_days))
    operational_thresholds = tuple(float(x.strip()) for x in str(args.operational_thresholds).split(",") if x.strip())

    real_rows: list[dict[str, Any]] = []
    for target_name, event_labels in target_events.items():
        aligned = pd.to_numeric(event_labels, errors="coerce").reindex(signal_frame.index)
        real_rows.extend(_score_signal_frame(signal_frame, aligned, lags=(1, 2, 5), target_name=target_name))
    _write_csv(outdir / "real_target_metrics.csv", real_rows)
    _write_csv(outdir / "target_event_summary.csv", target_event_summary)

    operational_real_rows = _operational_target_rows(
        signal_frame,
        target_events,
        target_future,
        thresholds=operational_thresholds,
        min_run=int(args.operational_min_run),
        cooldown=int(args.operational_cooldown),
        turn_window=int(args.turn_window),
        permutation_id=None,
    )
    _write_csv(outdir / "operational_target_metrics.csv", operational_real_rows)

    null_rows: list[dict[str, Any]] = []
    operational_null_rows: list[dict[str, Any]] = []
    for permutation_id in range(int(args.null_permutations)):
        cached_rows = _load_cached_null_rows(cache_root, permutation_id=permutation_id)
        if cached_rows is None:
            permuted = _joint_signal_permutation(signal_frame, seed=int(args.seed) + permutation_id)
            cached_rows = []
            for target_name, event_labels in target_events.items():
                aligned = pd.to_numeric(event_labels, errors="coerce").reindex(permuted.index)
                cached_rows.extend(_score_signal_frame(permuted, aligned, lags=(1,), target_name=target_name))
            for row in cached_rows:
                row["permutation_id"] = int(permutation_id)
            _save_cached_null_rows(cache_root, permutation_id=permutation_id, rows=cached_rows)
        cached_operational_rows = _load_cached_operational_null_rows(cache_root, permutation_id=permutation_id)
        if cached_operational_rows is None:
            permuted = _joint_signal_permutation(signal_frame, seed=int(args.seed) + permutation_id)
            cached_operational_rows = _operational_target_rows(
                permuted,
                target_events,
                target_future,
                thresholds=operational_thresholds,
                min_run=int(args.operational_min_run),
                cooldown=int(args.operational_cooldown),
                turn_window=int(args.turn_window),
                permutation_id=int(permutation_id),
            )
            _save_cached_operational_null_rows(cache_root, permutation_id=permutation_id, rows=cached_operational_rows)
        operational_null_rows.extend(cached_operational_rows)
        null_rows.extend(cached_rows)
    _write_csv(outdir / "null_signal_distribution.csv", null_rows)
    _write_csv(outdir / "operational_null_distribution.csv", operational_null_rows)

    signal_rollup = _signal_rollup(real_rows, null_rows)
    target_rollup = _target_rollup(real_rows)
    _write_csv(outdir / "signal_rollup.csv", signal_rollup)
    _write_csv(outdir / "target_rollup.csv", target_rollup)
    operational_rollup = _operational_rollup(operational_real_rows, operational_null_rows)
    _write_csv(outdir / "operational_rollup.csv", operational_rollup)

    sensitivity_rows = _sensitivity_rows(structure_daily, epsilons=[0.02, 0.05])
    _write_csv(outdir / "label_sensitivity.csv", sensitivity_rows)

    signal_decisions = {str(row["signal"]): str(row["decision"]) for row in signal_rollup}
    best_signal_row = max(signal_rollup, key=lambda row: _safe_float(row.get("real_mean_auc_lag1"), -1.0), default={})
    best_target_row = max(target_rollup, key=lambda row: _safe_float(row.get("best_auc_lag1"), -1.0), default={})
    best_operational_row = max(
        operational_rollup,
        key=lambda row: _safe_float(row.get("operational_score"), -1.0),
        default={},
    )
    sensitivity_worst = max(sensitivity_rows, key=lambda row: _safe_float(row.get("flip_rate"), 0.0), default={})

    summary = {
        "suite": "profit_structural_extraction_audit",
        "generated_at": datetime.now(UTC).isoformat(),
        "observer_mode": str(args.observer_mode),
        "observer_crypto_count": int(len(observer_crypto_cols)),
        "horizon_days": int(args.horizon_days),
        "null_mode": "joint_signal_permutation",
        "cache_root": str(cache_root),
        "targets": target_event_summary,
        "best_signal": best_signal_row,
        "best_target": best_target_row,
        "best_operational_signal": best_operational_row,
        "signal_decisions": signal_decisions,
        "operational_signal_decisions": {
            f"{str(row['signal'])}@{float(row['threshold']):.2f}": str(row["operational_decision"])
            for row in operational_rollup
        },
        "label_sensitivity": {
            "worst_case": sensitivity_worst,
            "median_flip_rate": float(pd.DataFrame(sensitivity_rows)["flip_rate"].median()) if sensitivity_rows else None,
            "median_flip_rate_far_from_threshold": float(pd.DataFrame(sensitivity_rows)["flip_rate_far_from_threshold"].median()) if sensitivity_rows else None,
        },
        "verdict": {
            "extracts_structure": any(str(row.get("decision")) == "keep" for row in signal_rollup),
            "operational_value": any(str(row.get("operational_decision")) == "keep" for row in operational_rollup),
            "notes": [
                "O nulo agora embaralha conjuntamente o vetor de sinais estruturais e preserva a distribuição marginal e a dependência instantânea entre sinais.",
                "Os alvos futuros são múltiplos: cripto, equities, Brasil e blends, em vez de um único mix BTC+SPY.",
                "A decisão por sinal separa keep, recalibrate e cut; isso evita confundir criticidade com market mode share.",
                "A auditoria operacional mede acerto de viradas, falso alarme e se dias sinalizados realmente antecedem pior retorno/drawdown futuro.",
            ],
        },
        "artifacts": {
            "real_target_metrics_csv": str(outdir / "real_target_metrics.csv"),
            "null_signal_distribution_csv": str(outdir / "null_signal_distribution.csv"),
            "signal_rollup_csv": str(outdir / "signal_rollup.csv"),
            "target_rollup_csv": str(outdir / "target_rollup.csv"),
            "operational_target_metrics_csv": str(outdir / "operational_target_metrics.csv"),
            "operational_null_distribution_csv": str(outdir / "operational_null_distribution.csv"),
            "operational_rollup_csv": str(outdir / "operational_rollup.csv"),
            "label_sensitivity_csv": str(outdir / "label_sensitivity.csv"),
            "target_event_summary_csv": str(outdir / "target_event_summary.csv"),
        },
    }
    _write_json(outdir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
