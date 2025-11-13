from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from graph_utils import Graph
from quantum_walk_noise import (
    NoiseSpec,
    NoisyQuantumWalkResult,
    QISKIT_AVAILABLE,
    simulate_noisy_quantum_walk,
)

CoinSpec = Union[str, np.ndarray]

__all__ = ["run_noisy_quantum_walk", "QISKIT_AVAILABLE"]


def run_noisy_quantum_walk(
    graph: Graph,
    steps: int,
    coin: CoinSpec = "hadamard",
    coin_state: Optional[Tuple[complex, complex]] = None,
    start_node: int = 0,
    initial_distribution: Optional[np.ndarray] = None,
    target_node: Optional[int] = None,
    threshold: float = 0.5,
    noise_profile: Optional[Sequence[NoiseSpec]] = None,
) -> NoisyQuantumWalkResult:
    return simulate_noisy_quantum_walk(
        graph=graph,
        n_steps=steps,
        coin=coin,
        coin_state=coin_state,
        start_node=start_node,
        initial_distribution=initial_distribution,
        target_node=target_node,
        threshold=threshold,
        noise_profile=noise_profile,
    )
