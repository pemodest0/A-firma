from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from classical_walk import ClassicalWalkResult
from quantum_walk import QuantumWalkResult
from quantum_walk_noise import NoisyQuantumWalkResult
from continuous_time_quantum_walk import ContinuousTimeQuantumWalkResult

__all__ = ["extract_walk_features"]


def _estimate_mixing_time(distributions: np.ndarray, tolerance: float = 1e-3) -> float:
    if distributions.shape[0] <= 1:
        return 0.0
    target = distributions[-1]
    for idx, dist in enumerate(distributions):
        if np.linalg.norm(dist - target, ord=1) <= tolerance:
            return float(idx)
    return float(distributions.shape[0] - 1)


def _estimate_entropy_stats(entropies: np.ndarray) -> Dict[str, float]:
    return {
        "entropy_mean": float(np.mean(entropies)),
        "entropy_std": float(np.std(entropies)),
        "entropy_final": float(entropies[-1]),
    }


def _estimate_hitting_time(
    distributions: np.ndarray,
    target_node: Optional[int],
    threshold: float,
) -> float:
    if target_node is None:
        return float("nan")
    hits = np.where(distributions[:, target_node] >= threshold)[0]
    if hits.size == 0:
        return float("inf")
    return float(hits[0])


def extract_walk_features(
    result: ClassicalWalkResult
    | QuantumWalkResult
    | NoisyQuantumWalkResult
    | ContinuousTimeQuantumWalkResult,
    *,
    target_node: Optional[int] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    distributions = np.asarray(result.distributions, dtype=float)
    entropies = np.asarray(result.entropies, dtype=float)
    features = {
        "steps": float(distributions.shape[0] - 1),
        "mixing_time": _estimate_mixing_time(distributions),
        "hitting_time": _estimate_hitting_time(distributions, target_node, threshold),
        "prob_peak": float(distributions.max()),
    }
    features.update(_estimate_entropy_stats(entropies))
    if hasattr(result, "coin_label"):
        features["coin_label_hash"] = float(hash(getattr(result, "coin_label")) % 10_000)
    if hasattr(result, "noise_profile"):
        features["noise_intensity"] = float(
            sum(item.get("strength", item.get("gamma", 0.0)) for item in result.noise_profile)  # type: ignore[attr-defined]
        )
    if hasattr(result, "gamma"):
        features["gamma"] = float(getattr(result, "gamma"))
    return features
