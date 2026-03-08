from __future__ import annotations

from typing import Literal

import numpy as np

try:
    from sklearn.covariance import LedoitWolf, OAS

    SKLEARN_OK = True
except Exception:  # pragma: no cover - handled by runtime guard
    SKLEARN_OK = False


CovEstimator = Literal["sample", "ewma", "ledoit_wolf", "oas"]


def _as_returns_matrix(returns: np.ndarray | list[list[float]]) -> np.ndarray:
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2:
        raise ValueError("returns must be a 2D array with shape (T, N)")
    if x.shape[0] < 2 or x.shape[1] < 2:
        raise ValueError("returns matrix must have at least 2 rows and 2 columns")
    if not np.isfinite(x).all():
        raise ValueError("returns matrix contains non-finite values")
    return x


def ensure_psd(matrix: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("matrix must be square")
    m = 0.5 * (m + m.T)
    vals, vecs = np.linalg.eigh(m)
    vals = np.asarray(vals, dtype=float)
    vals = np.maximum(vals, float(max(0.0, floor)))
    out = vecs @ np.diag(vals) @ vecs.T
    out = 0.5 * (out + out.T)
    return np.asarray(out, dtype=float)


def cov_to_corr(cov: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    c = np.asarray(cov, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("cov must be square")
    sd = np.sqrt(np.clip(np.diag(c), float(eps), None))
    denom = np.outer(sd, sd)
    corr = np.divide(c, denom, out=np.zeros_like(c), where=denom > float(eps))
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)
    return corr


def _sample_cov(x: np.ndarray) -> np.ndarray:
    return np.asarray(np.cov(x, rowvar=False, ddof=1), dtype=float)


def _ewma_cov(x: np.ndarray, ewma_lambda: float = 0.94) -> np.ndarray:
    lam = float(ewma_lambda)
    if (not np.isfinite(lam)) or lam <= 0.0 or lam >= 1.0:
        raise ValueError("ewma_lambda must be in (0, 1)")
    xc = x - np.nanmean(x, axis=0, keepdims=True)
    t, n = xc.shape
    cov = np.outer(xc[0], xc[0]).astype(float)
    for i in range(1, t):
        r = xc[i]
        cov = lam * cov + (1.0 - lam) * np.outer(r, r)
    return np.asarray(cov, dtype=float)


def _lw_cov(x: np.ndarray) -> np.ndarray:
    if not SKLEARN_OK:
        raise RuntimeError("scikit-learn not available for ledoit_wolf estimator")
    model = LedoitWolf(store_precision=False, assume_centered=False)
    model.fit(x)
    return np.asarray(model.covariance_, dtype=float)


def _oas_cov(x: np.ndarray) -> np.ndarray:
    if not SKLEARN_OK:
        raise RuntimeError("scikit-learn not available for oas estimator")
    model = OAS(store_precision=False, assume_centered=False)
    model.fit(x)
    return np.asarray(model.covariance_, dtype=float)


def estimate_cov(
    returns: np.ndarray | list[list[float]],
    *,
    method: CovEstimator = "sample",
    ewma_lambda: float = 0.94,
    psd_floor: float = 1e-8,
) -> np.ndarray:
    x = _as_returns_matrix(returns)
    m = str(method).strip().lower()
    if m == "sample":
        cov = _sample_cov(x)
    elif m == "ewma":
        cov = _ewma_cov(x, ewma_lambda=float(ewma_lambda))
    elif m == "ledoit_wolf":
        cov = _lw_cov(x)
    elif m == "oas":
        cov = _oas_cov(x)
    else:
        raise ValueError(f"unknown covariance estimator method: {method}")
    return ensure_psd(cov, floor=float(psd_floor))


def estimate_corr(
    returns: np.ndarray | list[list[float]],
    *,
    method: CovEstimator = "sample",
    ewma_lambda: float = 0.94,
    psd_floor: float = 1e-8,
) -> np.ndarray:
    cov = estimate_cov(
        returns=returns,
        method=method,
        ewma_lambda=float(ewma_lambda),
        psd_floor=float(psd_floor),
    )
    corr = cov_to_corr(cov)
    return ensure_psd(corr, floor=float(psd_floor))
