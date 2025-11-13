from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from graph_utils import Graph, line_graph

__all__ = [
    "ClassicalWalkResult",
    "simulate_classical_walk",
    "classical_random_walk_time_series",
    "classical_random_walk",
    "compute_shannon_entropy",
    "compute_hitting_time",
]


@dataclass
class ClassicalWalkResult:
    positions: np.ndarray
    distributions: np.ndarray
    entropies: np.ndarray
    transition_matrix: np.ndarray
    hitting_time: Optional[int]


def compute_shannon_entropy(probabilities: Sequence[float]) -> float:
    probs = np.asarray(probabilities, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(probs > 0, np.log2(probs), 0.0)
    return float(-np.sum(probs * logp))


def compute_hitting_time(
    distributions: np.ndarray,
    target_node: Optional[int],
    threshold: float,
) -> Optional[int]:
    if target_node is None:
        return None
    if target_node < 0 or target_node >= distributions.shape[1]:
        raise ValueError("target_node is out of range.")
    hits = np.where(distributions[:, target_node] >= threshold)[0]
    return int(hits[0]) if hits.size else None


def simulate_classical_walk(
    graph: Graph,
    n_steps: int,
    start_node: int = 0,
    target_node: Optional[int] = None,
    threshold: float = 0.5,
    initial_distribution: Optional[np.ndarray] = None,
) -> ClassicalWalkResult:
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative.")
    num_nodes = graph.num_nodes
    if start_node < 0 or start_node >= num_nodes:
        raise ValueError("start_node is out of range.")

    positions = np.arange(num_nodes, dtype=int)
    transition = graph.transition_matrix()

    distributions = np.zeros((n_steps + 1, num_nodes), dtype=float)
    if initial_distribution is None:
        distribution = np.zeros(num_nodes, dtype=float)
        distribution[start_node] = 1.0
    else:
        distribution = np.asarray(initial_distribution, dtype=float)
        if distribution.shape != (num_nodes,):
            raise ValueError("initial_distribution must have shape (num_nodes,).")
        total = float(distribution.sum())
        if total <= 0:
            raise ValueError("initial_distribution must have positive mass.")
        distribution = distribution / total

    distributions[0] = distribution
    for step in range(1, n_steps + 1):
        distribution = distribution @ transition
        distributions[step] = distribution

    entropies = np.apply_along_axis(compute_shannon_entropy, 1, distributions)
    hitting_time = compute_hitting_time(distributions, target_node, threshold)
    return ClassicalWalkResult(
        positions=positions,
        distributions=distributions,
        entropies=entropies,
        transition_matrix=transition,
        hitting_time=hitting_time,
    )


def classical_random_walk_time_series(n_steps: int) -> np.ndarray:
    size = 2 * n_steps + 1
    graph = line_graph(size)
    result = simulate_classical_walk(
        graph,
        n_steps,
        start_node=n_steps,
        target_node=None,
    )
    return result.distributions


def classical_random_walk(n_steps: int) -> np.ndarray:
    return classical_random_walk_time_series(n_steps)[-1]
