from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_outdir(outdir: str | None = None) -> Path:
    if outdir and str(outdir).strip():
        p = ROOT / str(outdir).strip()
    else:
        p = ROOT / "results" / run_id()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _latest_lab_run_dir() -> Path | None:
    base = ROOT / "results" / "lab_corr_macro"
    if not base.exists():
        return None
    runs = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
    for d in runs:
        if (d / "returns_wide_core.csv").exists():
            return d
    return None


def load_corr_matrix_or_synthetic(
    *,
    window: int = 120,
    min_assets: int = 20,
    seed: int = 23,
) -> tuple[np.ndarray, dict[str, Any]]:
    run_dir = _latest_lab_run_dir()
    if run_dir is not None:
        path = run_dir / "returns_wide_core.csv"
        try:
            df = pd.read_csv(path)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"]).set_index("date")
            else:
                first = str(df.columns[0])
                df[first] = pd.to_datetime(df[first], errors="coerce")
                df = df.dropna(subset=[first]).set_index(first)
            for c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            if int(df.shape[0]) >= int(window):
                block = df.iloc[-int(window) :].copy()
                cov = block.notna().mean(axis=0)
                keep = cov[cov >= 0.98].index.to_list()
                if len(keep) >= int(min_assets):
                    block = block[keep].dropna(how="any")
                    if block.shape[0] >= int(max(30, window // 2)) and block.shape[1] >= int(min_assets):
                        corr = np.corrcoef(block.to_numpy(dtype=float), rowvar=False)
                        if np.all(np.isfinite(corr)):
                            return corr, {
                                "source": "latest_lab_corr_macro",
                                "run_dir": str(run_dir),
                                "window": int(window),
                                "n_assets": int(block.shape[1]),
                                "n_rows": int(block.shape[0]),
                            }
        except Exception:
            pass

    rng = np.random.default_rng(int(seed))
    n = int(max(min_assets, 30))
    t = int(max(window, 200))
    x = rng.normal(0.0, 1.0, size=(t, n))
    common = rng.normal(0.0, 0.8, size=(t, 1))
    x = 0.6 * x + 0.4 * common
    corr = np.corrcoef(x, rowvar=False)
    return corr, {
        "source": "synthetic",
        "seed": int(seed),
        "window": int(t),
        "n_assets": int(n),
    }


def load_phi_series_from_latest() -> tuple[pd.Series, dict[str, Any]]:
    base = ROOT / "results" / "lab_corr_macro"
    if base.exists():
        runs = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
        for d in runs:
            path = d / "macro_timeseries_T120.csv"
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path)
                if "date" not in df.columns or "p1" not in df.columns:
                    continue
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df["p1"] = pd.to_numeric(df["p1"], errors="coerce")
                df = df.dropna(subset=["date", "p1"]).sort_values("date")
                if df.empty:
                    continue
                s = pd.Series(df["p1"].to_numpy(dtype=float), index=pd.DatetimeIndex(df["date"]), name="phi")
                return s, {"source": "latest_lab_corr_macro", "run_dir": str(d)}
            except Exception:
                continue

    rng = np.random.default_rng(23)
    n = 600
    eps = rng.normal(0.0, 0.05, size=n)
    phi = np.zeros(n, dtype=float)
    rho = 0.85
    for i in range(1, n):
        phi[i] = rho * phi[i - 1] + eps[i]
    phi = 0.45 + 0.08 * (phi - np.mean(phi))
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    s = pd.Series(phi, index=idx, name="phi")
    return s, {"source": "synthetic", "seed": 23}
