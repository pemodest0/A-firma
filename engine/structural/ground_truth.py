from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def forward_max_drawdown_from_equity(
    equity: pd.Series | np.ndarray | list[float],
    horizon_days: int,
) -> pd.Series:
    e = pd.to_numeric(pd.Series(equity), errors="coerce")
    h = int(max(1, horizon_days))
    out = np.full(len(e), np.nan, dtype=float)
    arr = e.to_numpy(dtype=float)
    for i in range(len(arr)):
        base = arr[i]
        if not np.isfinite(base) or base <= 0.0:
            continue
        j = min(len(arr), i + h + 1)
        fwd = arr[i:j]
        if fwd.size < 2:
            continue
        dd = np.nanmin((fwd / base) - 1.0)
        out[i] = float(dd)
    return pd.Series(out, index=e.index)


def build_event_label(
    *,
    equity: pd.Series | np.ndarray | list[float],
    horizon_days: int,
    dd_threshold: float,
) -> pd.Series:
    dd = forward_max_drawdown_from_equity(equity=equity, horizon_days=int(horizon_days))
    thr = -abs(float(dd_threshold))
    y = (pd.to_numeric(dd, errors="coerce") <= thr).astype("Int64")
    y[pd.to_numeric(dd, errors="coerce").isna()] = pd.NA
    return y


def build_regime_future_event_label(
    regime: pd.Series | np.ndarray | list[Any],
    horizon_days: int,
    target_regimes: set[str] | None = None,
) -> pd.Series:
    r = pd.Series(regime).astype(str).str.lower()
    h = int(max(1, horizon_days))
    targets = {str(x).lower() for x in (target_regimes or {"stress", "transition"})}
    arr = r.to_numpy(dtype=object)
    out = np.zeros(len(arr), dtype=int)
    for i in range(len(arr)):
        j = min(len(arr), i + h + 1)
        fwd = arr[i + 1 : j]
        if len(fwd) == 0:
            out[i] = 0
            continue
        out[i] = 1 if any(str(x).lower() in targets for x in fwd) else 0
    return pd.Series(out, index=r.index, dtype="Int64")


def classification_report_binary(
    y_true: pd.Series | np.ndarray | list[Any],
    y_pred: pd.Series | np.ndarray | list[Any],
) -> dict[str, float]:
    yt = pd.Series(y_true).astype("float")
    yp = pd.Series(y_pred).astype("float")
    m = yt.notna() & yp.notna()
    yt = yt[m].astype(int)
    yp = yp[m].astype(int)
    if yt.empty:
        return {
            "n": 0.0,
            "tp": 0.0,
            "fp": 0.0,
            "tn": 0.0,
            "fn": 0.0,
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "accuracy": float("nan"),
            "event_rate": float("nan"),
            "alert_rate": float("nan"),
        }

    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    f1 = float((2.0 * precision * recall) / (precision + recall)) if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0 else float("nan")
    acc = float((tp + tn) / len(yt)) if len(yt) > 0 else float("nan")

    return {
        "n": float(len(yt)),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
        "event_rate": float(np.mean(yt == 1)),
        "alert_rate": float(np.mean(yp == 1)),
    }


def threshold_from_train(
    score: pd.Series | np.ndarray | list[float],
    train_mask: pd.Series | np.ndarray | list[bool],
    q: float,
) -> float:
    s = pd.to_numeric(pd.Series(score), errors="coerce")
    m = pd.Series(train_mask).astype(bool)
    x = s[m].dropna()
    if x.empty:
        x = s.dropna()
    if x.empty:
        return float("nan")
    qq = float(min(0.999, max(0.001, q)))
    return float(x.quantile(qq))
