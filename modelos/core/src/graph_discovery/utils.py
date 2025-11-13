from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np

from classical_walk import ClassicalWalkResult

__all__ = ["summarize_walk"]


def _aggregate_hitting_times(
    distributions: np.ndarray,
    target_nodes: Tuple[int, ...],
    thresholds: Iterable[float],
) -> Tuple[np.ndarray, Dict[str, object]]:
    cumulative = distributions[:, list(target_nodes)].sum(axis=1)
    result: Dict[str, object] = {}
    for threshold in thresholds:
        hits = np.where(cumulative[1:] >= threshold)[0]
        result[f"hitting_time_ge_{threshold}"] = int(hits[0] + 1) if hits.size else None
    result["prob_final"] = float(cumulative[-1])
    return cumulative, result


def summarize_walk(
    walk_result: ClassicalWalkResult,
    thresholds: Iterable[float] = (0.5, 0.3, 0.1),
    aggregate_top_k: int = 5,
    target_nodes: Iterable[int] | None = None,
) -> Dict[str, object]:
    distributions = np.asarray(walk_result.distributions, dtype=float)
    entropies = np.asarray(walk_result.entropies, dtype=float)
    final_distribution = distributions[-1]
    target_node = int(final_distribution.argmax())
    summary = {
        "target_node": target_node,
        "prob_at_target": float(final_distribution[target_node]),
        "entropy_final": float(entropies[-1]),
    }
    for threshold in thresholds:
        hits = np.where(distributions[1:, target_node] >= threshold)[0]
        summary[f"hitting_time_ge_{threshold}"] = int(hits[0] + 1) if hits.size else None
    if target_nodes:
        custom_nodes = tuple(sorted({int(node) for node in target_nodes}))
        summary["custom_target_nodes"] = list(custom_nodes)
        _, custom_metrics = _aggregate_hitting_times(distributions, custom_nodes, thresholds)
        summary.update({f"custom_{k}": v for k, v in custom_metrics.items()})
    else:
        summary["custom_target_nodes"] = None
    if aggregate_top_k > 1:
        top_indices = tuple(int(idx) for idx in final_distribution.argsort()[-aggregate_top_k:])
        summary["top_k_nodes"] = list(top_indices)
        _, top_metrics = _aggregate_hitting_times(distributions, top_indices, thresholds)
        summary.update({f"topk_{aggregate_top_k}_{k}": v for k, v in top_metrics.items()})
    return summary
