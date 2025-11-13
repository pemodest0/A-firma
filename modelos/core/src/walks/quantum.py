from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from graph_utils import Graph
from quantum_walk import QuantumWalkResult, simulate_quantum_walk
from continuous_time_quantum_walk import (
    ContinuousTimeQuantumWalkResult,
    simulate_continuous_time_quantum_walk,
)

CoinSpec = Union[str, np.ndarray]

__all__ = [
    "run_discrete_quantum_walk",
    "run_continuous_quantum_walk",
]


def run_discrete_quantum_walk(
    graph: Graph,
    steps: int,
    coin: CoinSpec = "hadamard",
    coin_state: Optional[Tuple[complex, complex]] = None,
    start_node: int = 0,
    initial_distribution: Optional[np.ndarray] = None,
    target_node: Optional[int] = None,
    threshold: float = 0.5,
    measurement: str = "none",
    shots: int = 1024,
    seed: Optional[int] = None,
) -> QuantumWalkResult:
    return simulate_quantum_walk(
        graph=graph,
        n_steps=steps,
        coin=coin,
        coin_state=coin_state,
        start_node=start_node,
        initial_distribution=initial_distribution,
        target_node=target_node,
        threshold=threshold,
        measurement=measurement,  # type: ignore[arg-type]
        shots=shots,
        seed=seed,
    )


def run_continuous_quantum_walk(
    graph: Graph,
    steps: int,
    gamma: float = 1.0,
    time_step: float = 1.0,
    start_node: int = 0,
    initial_state: Optional[np.ndarray] = None,
    target_node: Optional[int] = None,
    threshold: float = 0.5,
) -> ContinuousTimeQuantumWalkResult:
    return simulate_continuous_time_quantum_walk(
        graph=graph,
        n_steps=steps,
        gamma=gamma,
        time_step=time_step,
        start_node=start_node,
        initial_state=initial_state,
        target_node=target_node,
        threshold=threshold,
    )
