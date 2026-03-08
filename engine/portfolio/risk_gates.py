from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


def _to_returns(values: Any) -> np.ndarray:
    s = pd.Series(values, dtype="float64")
    s = pd.to_numeric(s, errors="coerce").dropna().astype(float)
    arr = s.to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def total_return(values: Any) -> float:
    r = _to_returns(values)
    if r.size <= 0:
        return float("nan")
    return float(np.prod(1.0 + r) - 1.0)


def annualized_return(values: Any, periods_per_year: int = 12) -> float:
    r = _to_returns(values)
    if r.size <= 0:
        return float("nan")
    tot = total_return(r)
    return float((1.0 + tot) ** (float(periods_per_year) / float(r.size)) - 1.0)


def max_drawdown(values: Any) -> float:
    r = _to_returns(values)
    if r.size <= 0:
        return float("nan")
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    with np.errstate(invalid="ignore", divide="ignore"):
        dd = eq / peak - 1.0
    dd = dd[np.isfinite(dd)]
    return float(np.min(dd)) if dd.size > 0 else float("nan")


def cvar(values: Any, alpha: float = 0.95) -> float:
    r = _to_returns(values)
    if r.size <= 0:
        return float("nan")
    q = float(np.quantile(r, 1.0 - float(alpha)))
    tail = r[r <= q]
    if tail.size <= 0:
        return float("nan")
    return float(np.mean(tail))


def block_bootstrap_paths(
    values: Any,
    *,
    n_paths: int = 10000,
    block_len: int = 3,
    seed: int = 23,
) -> np.ndarray:
    r = _to_returns(values)
    if r.size <= 0:
        return np.empty((0, 0), dtype=float)
    n = int(r.size)
    lb = int(max(1, block_len))
    rng = np.random.default_rng(int(seed))
    paths = np.zeros((int(n_paths), n), dtype=float)
    starts_max = max(1, n - lb + 1)
    for i in range(int(n_paths)):
        idxs: list[int] = []
        while len(idxs) < n:
            st = int(rng.integers(0, starts_max))
            idxs.extend(range(st, min(n, st + lb)))
        idx = np.asarray(idxs[:n], dtype=int)
        paths[i, :] = r[idx]
    return paths


def _path_total(path: np.ndarray) -> float:
    if path.size <= 0:
        return float("nan")
    return float(np.prod(1.0 + path) - 1.0)


def _path_mdd(path: np.ndarray) -> float:
    if path.size <= 0:
        return float("nan")
    eq = np.cumprod(1.0 + path)
    peak = np.maximum.accumulate(eq)
    with np.errstate(invalid="ignore", divide="ignore"):
        dd = eq / peak - 1.0
    dd = dd[np.isfinite(dd)]
    return float(np.min(dd)) if dd.size > 0 else float("nan")


def summarize_distribution_bounds(
    paths: np.ndarray,
    *,
    ruin_threshold: float = -0.5,
) -> dict[str, float]:
    if paths.size <= 0:
        return {
            "n_paths": 0,
            "total_return_p05": float("nan"),
            "total_return_p50": float("nan"),
            "total_return_p95": float("nan"),
            "max_drawdown_p05_worst": float("nan"),
            "max_drawdown_p50": float("nan"),
            "prob_total_positive": float("nan"),
            "prob_total_below_minus20": float("nan"),
            "prob_total_below_ruin_threshold": float("nan"),
            "prob_dd_worse_30": float("nan"),
            "prob_dd_worse_50": float("nan"),
        }
    totals = np.asarray([_path_total(p) for p in paths], dtype=float)
    mdds = np.asarray([_path_mdd(p) for p in paths], dtype=float)
    totals = totals[np.isfinite(totals)]
    mdds = mdds[np.isfinite(mdds)]
    return {
        "n_paths": int(paths.shape[0]),
        "total_return_p05": float(np.quantile(totals, 0.05)) if totals.size else float("nan"),
        "total_return_p50": float(np.quantile(totals, 0.50)) if totals.size else float("nan"),
        "total_return_p95": float(np.quantile(totals, 0.95)) if totals.size else float("nan"),
        "max_drawdown_p05_worst": float(np.quantile(mdds, 0.05)) if mdds.size else float("nan"),
        "max_drawdown_p50": float(np.quantile(mdds, 0.50)) if mdds.size else float("nan"),
        "prob_total_positive": float(np.mean(totals > 0.0)) if totals.size else float("nan"),
        "prob_total_below_minus20": float(np.mean(totals <= -0.20)) if totals.size else float("nan"),
        "prob_total_below_ruin_threshold": float(np.mean(totals <= float(ruin_threshold))) if totals.size else float("nan"),
        "prob_dd_worse_30": float(np.mean(mdds <= -0.30)) if mdds.size else float("nan"),
        "prob_dd_worse_50": float(np.mean(mdds <= -0.50)) if mdds.size else float("nan"),
    }


def evaluate_tail_risk(
    values: Any,
    *,
    n_paths: int = 10000,
    block_len: int = 3,
    seed: int = 23,
    alpha: float = 0.95,
    ruin_threshold: float = -0.50,
) -> dict[str, Any]:
    r = _to_returns(values)
    base = {
        "months": int(r.size),
        "total_return": total_return(r),
        "annualized_return": annualized_return(r),
        "max_drawdown": max_drawdown(r),
        "cvar_95": cvar(r, alpha=float(alpha)),
    }
    paths = block_bootstrap_paths(r, n_paths=int(n_paths), block_len=int(block_len), seed=int(seed))
    dist = summarize_distribution_bounds(paths, ruin_threshold=float(ruin_threshold))
    return {
        "status": "ok" if int(r.size) > 0 else "empty",
        "base": base,
        "bootstrap": dist,
        "params": {
            "n_paths": int(n_paths),
            "block_len": int(block_len),
            "seed": int(seed),
            "cvar_alpha": float(alpha),
            "ruin_threshold": float(ruin_threshold),
        },
    }


@dataclass(frozen=True)
class TailRiskThresholds:
    max_drawdown_floor: float = -0.35
    cvar95_floor: float = -0.12
    max_prob_total_below_ruin: float = 0.05
    max_prob_dd_worse_50: float = 0.05
    min_prob_total_positive: float = 0.60


def apply_tail_gate(metrics: dict[str, Any], thresholds: TailRiskThresholds) -> dict[str, Any]:
    base = metrics.get("base", {}) if isinstance(metrics, dict) else {}
    boot = metrics.get("bootstrap", {}) if isinstance(metrics, dict) else {}
    reasons: list[str] = []

    mdd = float(base.get("max_drawdown", np.nan))
    if np.isfinite(mdd) and mdd < float(thresholds.max_drawdown_floor):
        reasons.append("max_drawdown_floor_failed")

    cvar95 = float(base.get("cvar_95", np.nan))
    if np.isfinite(cvar95) and cvar95 < float(thresholds.cvar95_floor):
        reasons.append("cvar95_floor_failed")

    p_ruin = float(boot.get("prob_total_below_ruin_threshold", np.nan))
    if np.isfinite(p_ruin) and p_ruin > float(thresholds.max_prob_total_below_ruin):
        reasons.append("prob_total_below_ruin_failed")

    p_dd50 = float(boot.get("prob_dd_worse_50", np.nan))
    if np.isfinite(p_dd50) and p_dd50 > float(thresholds.max_prob_dd_worse_50):
        reasons.append("prob_dd_worse_50_failed")

    p_pos = float(boot.get("prob_total_positive", np.nan))
    if np.isfinite(p_pos) and p_pos < float(thresholds.min_prob_total_positive):
        reasons.append("prob_total_positive_failed")

    passed = len(reasons) == 0
    return {
        "passed": bool(passed),
        "reasons": reasons,
        "thresholds": asdict(thresholds),
        "checked": {
            "max_drawdown": mdd,
            "cvar_95": cvar95,
            "prob_total_below_ruin_threshold": p_ruin,
            "prob_dd_worse_50": p_dd50,
            "prob_total_positive": p_pos,
        },
    }
