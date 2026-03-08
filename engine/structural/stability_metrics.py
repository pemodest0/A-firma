from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


def _principal_mode(corr: np.ndarray) -> tuple[float, np.ndarray]:
    vals, vecs = np.linalg.eigh(corr)
    if vals.size <= 0:
        return float("nan"), np.asarray([], dtype=float)
    idx = int(np.argmax(vals))
    v = np.asarray(vecs[:, idx], dtype=float)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= 0.0:
        return float("nan"), np.asarray([], dtype=float)
    v = v / n
    anchor = int(np.argmax(np.abs(v)))
    if v[anchor] < 0.0:
        v = -v
    return float(vals[idx]), v


def dominant_mode_series(
    monthly_returns: pd.DataFrame,
    *,
    window_months: int = 12,
    min_assets: int = 20,
    min_obs_asset: int = 8,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    if monthly_returns.empty:
        return pd.DataFrame(), {}
    mret = monthly_returns.copy().sort_index()
    out_rows: list[dict[str, Any]] = []
    vectors: dict[str, pd.Series] = {}

    prev_vec: pd.Series | None = None
    prev_label: str | None = None
    win = int(max(3, window_months))

    for end in range(win - 1, len(mret.index)):
        label = str(mret.index[end])
        block = mret.iloc[end - win + 1 : end + 1, :]
        obs = block.notna().sum(axis=0)
        cols = obs[obs >= int(max(2, min_obs_asset))].index.tolist()
        if len(cols) < int(max(2, min_assets)):
            continue
        x = block[cols].apply(pd.to_numeric, errors="coerce")
        sd = x.std(axis=0, ddof=0)
        cols2 = sd[sd > 1e-12].index.tolist()
        if len(cols2) < int(max(2, min_assets)):
            continue
        x = x[cols2].fillna(0.0).astype(float)
        c = np.corrcoef(x.to_numpy(dtype=float), rowvar=False)
        c = np.asarray(c, dtype=float)
        c[~np.isfinite(c)] = 0.0
        np.fill_diagonal(c, 1.0)
        eig1, vec = _principal_mode(c)
        if vec.size <= 0:
            continue

        cur = pd.Series(vec, index=cols2, dtype=float)
        vectors[label] = cur.copy()

        overlap = float("nan")
        drift = float("nan")
        overlap_assets = 0
        if prev_vec is not None and prev_label is not None:
            common = prev_vec.index.intersection(cur.index)
            overlap_assets = int(len(common))
            if overlap_assets >= 2:
                a = prev_vec.loc[common].to_numpy(dtype=float)
                b = cur.loc[common].to_numpy(dtype=float)
                na = float(np.linalg.norm(a))
                nb = float(np.linalg.norm(b))
                if np.isfinite(na) and np.isfinite(nb) and na > 0.0 and nb > 0.0:
                    overlap = float(np.abs(np.dot(a / na, b / nb)))
                    drift = float(1.0 - overlap)
        out_rows.append(
            {
                "ym": label,
                "window_months": int(win),
                "n_assets_mode": int(len(cols2)),
                "eig1": float(eig1),
                "overlap_prev": overlap,
                "drift_prev": drift,
                "overlap_assets_prev": int(overlap_assets),
            }
        )
        prev_vec = cur
        prev_label = label

    out = pd.DataFrame(out_rows)
    return out, vectors


def summarize_mode_stability(stability_df: pd.DataFrame) -> dict[str, Any]:
    if stability_df.empty:
        return {"status": "empty"}
    d = stability_df.copy()
    o = pd.to_numeric(d["overlap_prev"], errors="coerce")
    dr = pd.to_numeric(d["drift_prev"], errors="coerce")
    n = int(o.notna().sum())
    if n <= 0:
        return {"status": "insufficient", "n_valid_overlap": 0}
    return {
        "status": "ok",
        "n_points": int(len(d)),
        "n_valid_overlap": n,
        "median_overlap": float(o.median()),
        "p10_overlap": float(o.quantile(0.10)),
        "mean_overlap": float(o.mean()),
        "share_overlap_lt_05": float((o < 0.50).mean()),
        "share_overlap_lt_03": float((o < 0.30).mean()),
        "max_drift": float(dr.max()) if dr.notna().any() else float("nan"),
    }


@dataclass(frozen=True)
class ModeStabilityThresholds:
    min_median_overlap: float = 0.55
    min_p10_overlap: float = 0.30
    max_share_overlap_lt_05: float = 0.35
    max_max_drift: float = 0.90


def apply_mode_stability_gate(
    summary: dict[str, Any],
    thresholds: ModeStabilityThresholds,
) -> dict[str, Any]:
    reasons: list[str] = []
    if str(summary.get("status")) != "ok":
        reasons.append("stability_summary_not_ok")
    med = float(summary.get("median_overlap", np.nan))
    p10 = float(summary.get("p10_overlap", np.nan))
    share_low = float(summary.get("share_overlap_lt_05", np.nan))
    max_drift = float(summary.get("max_drift", np.nan))

    if np.isfinite(med) and med < float(thresholds.min_median_overlap):
        reasons.append("median_overlap_failed")
    if np.isfinite(p10) and p10 < float(thresholds.min_p10_overlap):
        reasons.append("p10_overlap_failed")
    if np.isfinite(share_low) and share_low > float(thresholds.max_share_overlap_lt_05):
        reasons.append("share_overlap_lt_05_failed")
    if np.isfinite(max_drift) and max_drift > float(thresholds.max_max_drift):
        reasons.append("max_drift_failed")

    return {
        "passed": bool(len(reasons) == 0),
        "reasons": reasons,
        "thresholds": asdict(thresholds),
        "checked": {
            "median_overlap": med,
            "p10_overlap": p10,
            "share_overlap_lt_05": share_low,
            "max_drift": max_drift,
        },
    }

