from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

__all__ = [
    "crps_from_samples",
    "evaluate_finance_distribution",
    "evaluate_finance_point_forecast",
    "evaluate_finance_events",
]


def crps_from_samples(samples: Sequence[float], observation: float) -> float:
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        raise ValueError("CRPS requires at least one sample.")
    diff_obs = np.abs(arr - observation).mean()
    diff_pair = np.abs(arr[:, None] - arr[None, :]).mean()
    return float(diff_obs - 0.5 * diff_pair)


def evaluate_finance_distribution(
    probabilities: Sequence[float],
    support: Sequence[float],
    observation: float,
) -> Dict[str, float]:
    probs = np.asarray(probabilities, dtype=float)
    support_arr = np.asarray(support, dtype=float)
    if probs.shape != support_arr.shape:
        raise ValueError("probabilities and support must share the same shape.")
    probs = probs / probs.sum()
    expectation = float(np.dot(probs, support_arr))
    variance = float(np.dot(probs, (support_arr - expectation) ** 2))
    crps = crps_from_samples(np.repeat(support_arr, np.maximum(1, (probs * len(probs)).astype(int))), observation)
    return {
        "expectation": expectation,
        "variance": variance,
        "crps": crps,
    }


def evaluate_finance_point_forecast(prediction: float, observation: float) -> Dict[str, float]:
    mae = abs(prediction - observation)
    mape = abs(mae / observation) if observation != 0 else float("inf")
    return {
        "mae": float(mae),
        "mape": float(mape),
    }


def evaluate_finance_events(
    probabilities: Sequence[float],
    support: Sequence[float],
    threshold: float,
) -> Dict[str, float]:
    probs = np.asarray(probabilities, dtype=float)
    support_arr = np.asarray(support, dtype=float)
    probs = probs / probs.sum()
    mask_upside = support_arr >= threshold
    mask_downside = support_arr <= -threshold
    return {
        "event_prob_upside": float(probs[mask_upside].sum()),
        "event_prob_downside": float(probs[mask_downside].sum()),
    }
