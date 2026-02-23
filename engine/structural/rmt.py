from __future__ import annotations

from typing import Any

import numpy as np


def mp_bounds(T: int, N: int, sigma: float = 1.0) -> tuple[float, float]:
    t = int(T)
    n = int(N)
    s = float(sigma)
    if t <= 0 or n <= 0:
        raise ValueError("T and N must be positive")
    if s <= 0:
        raise ValueError("sigma must be positive")
    q = float(t) / float(n)
    root = float(np.sqrt(1.0 / q))
    lmin = (s**2) * ((1.0 - root) ** 2)
    lmax = (s**2) * ((1.0 + root) ** 2)
    return float(lmin), float(lmax)


def significant_eigs(eigs: np.ndarray | list[float], lambda_max: float) -> tuple[np.ndarray, np.ndarray]:
    vals = np.asarray(eigs, dtype=float)
    if vals.ndim != 1:
        vals = vals.ravel()
    vals = np.where(np.isfinite(vals), vals, np.nan)
    mask = np.isfinite(vals) & (vals > float(lambda_max))
    idx = np.flatnonzero(mask)
    return mask.astype(bool), idx.astype(int)


def rmt_report(eigs: np.ndarray | list[float], T: int, N: int, sigma: float = 1.0) -> dict[str, Any]:
    vals = np.asarray(eigs, dtype=float)
    if vals.ndim != 1:
        vals = vals.ravel()
    vals = vals[np.isfinite(vals)]
    vals = np.sort(vals)[::-1]
    if vals.size == 0:
        raise ValueError("empty eigs")
    _, lmax = mp_bounds(T=T, N=N, sigma=float(sigma))
    mask, idx = significant_eigs(vals, lambda_max=lmax)
    return {
        "q": float(float(T) / float(N)),
        "lambda_max": float(lmax),
        "lambda1": float(vals[0]),
        "count_sig": int(mask.sum()),
        "frac_sig": float(mask.mean()) if mask.size > 0 else 0.0,
        "indices_sig": idx.astype(int).tolist(),
        "n_eigs": int(vals.size),
    }
