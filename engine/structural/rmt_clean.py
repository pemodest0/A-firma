from __future__ import annotations

from typing import Any

import numpy as np

from .covariance_estimators import ensure_psd


def mp_bounds_from_q(q: float, sigma2: float = 1.0) -> tuple[float, float]:
    qq = float(q)
    s2 = float(sigma2)
    if (not np.isfinite(qq)) or qq <= 0.0:
        raise ValueError("q must be positive")
    if (not np.isfinite(s2)) or s2 <= 0.0:
        raise ValueError("sigma2 must be positive")
    root = float(np.sqrt(qq))
    lmin = s2 * ((1.0 - root) ** 2)
    lmax = s2 * ((1.0 + root) ** 2)
    return float(lmin), float(lmax)


def mp_bounds(T: int, N: int, sigma2: float = 1.0) -> tuple[float, float]:
    t = int(T)
    n = int(N)
    if t <= 0 or n <= 0:
        raise ValueError("T and N must be positive")
    q = float(n) / float(t)
    return mp_bounds_from_q(q=q, sigma2=float(sigma2))


def clean_correlation_mp_clip(
    corr: np.ndarray,
    T: int,
    *,
    mode: str = "clip",
    keep_top_k: int | None = None,
    psd_floor: float = 1e-8,
) -> tuple[np.ndarray, dict[str, Any]]:
    c = np.asarray(corr, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("corr must be square")
    n = int(c.shape[0])
    if n < 2:
        raise ValueError("corr size must be >= 2")
    t = int(T)
    if t <= 1:
        raise ValueError("T must be > 1")

    c = ensure_psd(0.5 * (c + c.T), floor=float(psd_floor))
    np.fill_diagonal(c, 1.0)
    c = np.clip(c, -1.0, 1.0)

    vals, vecs = np.linalg.eigh(c)
    order = np.argsort(vals)[::-1]
    lam = np.asarray(vals[order], dtype=float)
    v = np.asarray(vecs[:, order], dtype=float)

    _, lam_max = mp_bounds(T=t, N=n, sigma2=1.0)
    k_keep = int(max(0, keep_top_k or 0))
    k_keep = int(min(k_keep, n))
    noise_mask = lam <= float(lam_max)
    if k_keep > 0:
        keep_mask = np.zeros(n, dtype=bool)
        keep_mask[:k_keep] = True
        noise_mask = noise_mask & (~keep_mask)

    if np.any(noise_mask):
        noise_vals = lam[noise_mask]
        if str(mode).strip().lower() == "threshold":
            fill = float(lam_max)
        else:
            fill = float(np.mean(noise_vals))
        lam_clean = lam.copy()
        lam_clean[noise_mask] = fill
    else:
        lam_clean = lam.copy()

    clean = v @ np.diag(lam_clean) @ v.T
    clean = ensure_psd(clean, floor=float(psd_floor))
    d = np.sqrt(np.clip(np.diag(clean), float(psd_floor), None))
    clean = clean / np.outer(d, d)
    clean = ensure_psd(clean, floor=float(psd_floor))
    np.fill_diagonal(clean, 1.0)
    clean = np.clip(clean, -1.0, 1.0)

    info = {
        "T": int(t),
        "N": int(n),
        "q": float(n / t),
        "lambda_max": float(lam_max),
        "mode": str(mode),
        "keep_top_k": int(k_keep),
        "n_noise_eigs": int(np.sum(noise_mask)),
        "noise_replacement": float(np.mean(lam[noise_mask])) if np.any(noise_mask) else float("nan"),
        "lambda1_raw": float(lam[0]) if lam.size else float("nan"),
        "lambda1_clean": float(np.linalg.eigvalsh(clean)[-1]) if clean.size else float("nan"),
    }
    return np.asarray(clean, dtype=float), info
