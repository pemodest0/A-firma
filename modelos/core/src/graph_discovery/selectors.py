from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np

from graph_utils import Graph
from .builders import GraphCandidate

ScoreFn = Callable[[GraphCandidate], float]


@dataclass
class GraphPenalties:
    density: float = 0.0
    short_cycles: float = 0.0
    diameter: float = 0.0


def _graph_density(graph: Graph) -> float:
    edges = float(np.sum(graph.adjacency)) / 2.0
    max_edges = graph.num_nodes * (graph.num_nodes - 1) / 2.0
    if max_edges == 0:
        return 0.0
    return edges / max_edges


def _triangle_ratio(graph: Graph) -> float:
    if graph.num_nodes < 3:
        return 0.0
    adjacency = graph.adjacency
    tri_count = np.trace(np.linalg.matrix_power(adjacency, 3)) / 6.0
    return float(tri_count) / graph.num_nodes


def _approximate_diameter(graph: Graph) -> float:
    n = graph.num_nodes
    if n <= 1:
        return 0.0
    adjacency = graph.adjacency
    visited = np.zeros(n, dtype=bool)
    queue = [0]
    visited[0] = True
    dist = np.full(n, np.inf, dtype=float)
    dist[0] = 0.0
    while queue:
        node = queue.pop(0)
        neighbors = np.where(adjacency[node] > 0)[0]
        for nb in neighbors:
            if not visited[nb]:
                visited[nb] = True
                dist[nb] = dist[node] + 1.0
                queue.append(nb)
    if not np.all(visited):
        return float("inf")
    return float(np.max(dist))


def penalized_score(base_score: float, graph: Graph, penalties: GraphPenalties) -> float:
    density = _graph_density(graph)
    cycle = _triangle_ratio(graph)
    diameter = _approximate_diameter(graph)
    penalty = (
        penalties.density * density
        + penalties.short_cycles * cycle
        + penalties.diameter * (0.0 if np.isinf(diameter) else diameter)
    )
    return base_score + penalty


def evaluate_candidates(
    candidates: Iterable[GraphCandidate],
    score_fn: ScoreFn,
    penalties: GraphPenalties | None = None,
) -> List[Tuple[GraphCandidate, float]]:
    penalties = penalties or GraphPenalties()
    results: List[Tuple[GraphCandidate, float]] = []
    for candidate in candidates:
        base = score_fn(candidate)
        score = penalized_score(base, candidate.graph, penalties)
        results.append((candidate, score))
    return results


def select_best_graph(
    candidates: Sequence[GraphCandidate],
    score_fn: ScoreFn,
    penalties: GraphPenalties | None = None,
) -> Tuple[GraphCandidate, float]:
    scores = evaluate_candidates(candidates, score_fn, penalties)
    best_candidate, best_score = min(scores, key=lambda item: item[1])
    return best_candidate, best_score
