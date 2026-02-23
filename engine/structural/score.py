from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def fit_normalizer(train_series_dict: dict[str, Any]) -> dict[str, Any]:
    stats: dict[str, dict[str, float]] = {}
    for key, values in train_series_dict.items():
        x = pd.to_numeric(pd.Series(values), errors="coerce")
        mu = float(x.mean(skipna=True))
        sd = float(x.std(ddof=0, skipna=True))
        if (not np.isfinite(sd)) or sd <= 1e-12:
            sd = 1.0
        stats[str(key)] = {"mean": mu, "std": sd}
    return {"version": "normalizer.v1", "stats": stats}


def transform(series: Any, params: dict[str, Any], key: str | None = None) -> pd.Series:
    x = pd.to_numeric(pd.Series(series), errors="coerce")
    stats = (params or {}).get("stats") or {}
    k = str(key) if key is not None else None
    row = stats.get(k) if k is not None else None
    if not row:
        mu = float(x.mean(skipna=True))
        sd = float(x.std(ddof=0, skipna=True))
        if (not np.isfinite(sd)) or sd <= 1e-12:
            sd = 1.0
    else:
        mu = float(row.get("mean", 0.0))
        sd = float(row.get("std", 1.0))
        if (not np.isfinite(sd)) or sd <= 1e-12:
            sd = 1.0
    z = (x - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan)


def structural_score(
    features_z: dict[str, Any],
    weights: dict[str, float] | None = None,
) -> pd.Series:
    w = {
        "phi": 1.0,
        "deff": 1.0,
        "ac1_phi": 1.0,
        "neg_kappa_mean": 1.0,
    }
    if weights:
        for k, v in weights.items():
            w[str(k)] = float(v)

    phi = pd.to_numeric(pd.Series(features_z.get("phi")), errors="coerce")
    deff = pd.to_numeric(pd.Series(features_z.get("deff")), errors="coerce")
    ac1 = pd.to_numeric(pd.Series(features_z.get("ac1_phi")), errors="coerce")

    if "neg_kappa_mean" in features_z:
        nk = pd.to_numeric(pd.Series(features_z.get("neg_kappa_mean")), errors="coerce")
    else:
        km = pd.to_numeric(pd.Series(features_z.get("forman_mean")), errors="coerce")
        nk = -km

    score = (w["phi"] * phi) - (w["deff"] * deff) + (w["ac1_phi"] * ac1) + (w["neg_kappa_mean"] * nk)
    return score.replace([np.inf, -np.inf], np.nan)
