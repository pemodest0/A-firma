from __future__ import annotations

from typing import Optional

import numpy as np

from graph_utils import Graph
from classical_walk import ClassicalWalkResult, simulate_classical_walk

__all__ = ["run_classical_walk"]


def run_classical_walk(
    graph: Graph,
    steps: int,
    start_node: int = 0,
    target_node: Optional[int] = None,
    threshold: float = 0.5,
    initial_distribution: Optional[np.ndarray] = None,
) -> ClassicalWalkResult:
    return simulate_classical_walk(
        graph=graph,
        n_steps=steps,
        start_node=start_node,
        target_node=target_node,
        threshold=threshold,
        initial_distribution=initial_distribution,
    )
