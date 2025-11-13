from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np

__all__ = [
    "evaluate_logistics_cost",
    "evaluate_logistics_schedule",
    "evaluate_logistics_robustness",
]


def evaluate_logistics_cost(
    predicted_costs: Sequence[float],
    actual_cost: float,
) -> Dict[str, float]:
    preds = np.asarray(predicted_costs, dtype=float)
    expectation = float(preds.mean())
    deviation = float(np.std(preds))
    delta = expectation - actual_cost
    return {
        "cost_expectation": expectation,
        "cost_std": deviation,
        "cost_bias": float(delta),
    }


def evaluate_logistics_schedule(
    on_time_probabilities: Sequence[float],
    actual_on_time: bool,
) -> Dict[str, float]:
    probs = np.asarray(on_time_probabilities, dtype=float)
    probs = np.clip(probs, 0.0, 1.0)
    confidence = float(probs.mean())
    brier = float(np.mean((probs - (1.0 if actual_on_time else 0.0)) ** 2))
    return {
        "on_time_confidence": confidence,
        "brier_score": brier,
    }


def evaluate_logistics_robustness(
    scenarios: Iterable[Dict[str, float]],
    baseline_metric: str,
) -> Dict[str, float]:
    values = [float(scenario.get(baseline_metric, 0.0)) for scenario in scenarios]
    if not values:
        raise ValueError("No scenarios provided for robustness evaluation.")
    arr = np.asarray(values, dtype=float)
    return {
        "robustness_mean": float(arr.mean()),
        "robustness_worst": float(arr.max()),
        "robustness_best": float(arr.min()),
        "robustness_range": float(arr.max() - arr.min()),
    }
