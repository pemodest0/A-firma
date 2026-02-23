from __future__ import annotations

from typing import Any

import numpy as np


def normalize_eigs(eigs: np.ndarray | list[float], eps: float = 1e-12) -> np.ndarray:
    vals = np.asarray(eigs, dtype=float)
    if vals.ndim != 1:
        vals = vals.ravel()
    vals = np.where(np.isfinite(vals), vals, 0.0)
    vals = np.clip(vals, 0.0, None)
    s = float(vals.sum())
    if s <= float(eps):
        if vals.size == 0:
            return vals
        return np.ones(vals.size, dtype=float) / float(vals.size)
    return vals / s


def spectral_entropy(eigs: np.ndarray | list[float], eps: float = 1e-12) -> float:
    p = normalize_eigs(eigs, eps=eps)
    if p.size == 0:
        return float("nan")
    h = -np.sum(p * np.log(np.clip(p, float(eps), None)))
    return float(h)


def effective_dimension(eigs: np.ndarray | list[float], eps: float = 1e-12) -> float:
    h = spectral_entropy(eigs, eps=eps)
    if not np.isfinite(h):
        return float("nan")
    return float(np.exp(h))


def order_param_phi(eigs: np.ndarray | list[float], eps: float = 1e-12) -> float:
    p = normalize_eigs(eigs, eps=eps)
    if p.size == 0:
        return float("nan")
    return float(p[0])


def spectral_pack(eigs: np.ndarray | list[float], topk: int = 5) -> dict[str, Any]:
    vals = np.asarray(eigs, dtype=float)
    if vals.ndim != 1:
        vals = vals.ravel()
    vals = vals[np.isfinite(vals)]
    vals = np.sort(np.clip(vals, 0.0, None))[::-1]
    if vals.size == 0:
        return {
            "phi": float("nan"),
            "H": float("nan"),
            "deff": float("nan"),
            "lambda1": float("nan"),
            "topk": float("nan"),
            "n_eigs": 0,
        }
    p = normalize_eigs(vals)
    k = int(max(1, topk))
    return {
        "phi": float(order_param_phi(vals)),
        "H": float(spectral_entropy(vals)),
        "deff": float(effective_dimension(vals)),
        "lambda1": float(vals[0]),
        "topk": float(np.sum(p[: min(k, p.size)])),
        "n_eigs": int(vals.size),
    }
