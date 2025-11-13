"""Autocorrelation and diffusion metrics for walk histories."""
from __future__ import annotations

import numpy as np

from gaq.analysis.hypercube_walks import vertices


def autocorrelation(series: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    series = np.asarray(series, dtype=float)
    series -= series.mean()
    n = len(series)
    max_lag = n - 1 if max_lag is None else min(max_lag, n - 1)
    result = np.zeros(max_lag + 1)
    denom = np.dot(series, series)
    if denom == 0:
        return result
    result[0] = 1.0
    for lag in range(1, max_lag + 1):
        result[lag] = np.dot(series[:-lag], series[lag:]) / denom
    return result


def hamming_diffusion(history: list[np.ndarray]) -> np.ndarray:
    count = history[0].shape[0]
    n = int(np.log2(count))
    bits = vertices(n).astype(float)
    baseline = bits[np.argmax(history[0])]
    diffusion = []
    for prob in history:
        expected = prob @ bits
        diffusion.append(np.mean(np.abs(expected - baseline)))
    return np.array(diffusion)


def probability_gap(history: list[np.ndarray]) -> np.ndarray:
    """Difference between max and median probability over time."""
    gaps = []
    for prob in history:
        gaps.append(float(prob.max() - np.median(prob)))
    return np.array(gaps)
