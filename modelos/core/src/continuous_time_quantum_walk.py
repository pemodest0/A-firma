from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from graph_utils import Graph
from classical_walk import compute_shannon_entropy, compute_hitting_time

__all__ = [
    "ContinuousTimeQuantumWalkResult",
    "simulate_continuous_time_quantum_walk",
]


@dataclass
class ContinuousTimeQuantumWalkResult:
    graph: Graph
    positions: np.ndarray
    distributions: np.ndarray
    entropies: np.ndarray
    gamma: float
    time_step: float
    hitting_time: Optional[int]


def _build_hamiltonian(graph: Graph, gamma: float) -> np.ndarray:
    laplacian = np.diag(graph.degrees) - graph.adjacency
    return gamma * laplacian.astype(np.complex128)


def _normalize_initial_state(state: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(state)
    if norm == 0:
        raise ValueError("initial_state must have non-zero norm.")
    return state / norm


def simulate_continuous_time_quantum_walk(
    graph: Graph,
    n_steps: int,
    gamma: float = 1.0,
    time_step: float = 1.0,
    start_node: int = 0,
    initial_state: Optional[np.ndarray] = None,
    target_node: Optional[int] = None,
    threshold: float = 0.5,
) -> ContinuousTimeQuantumWalkResult:
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative.")
    if time_step <= 0:
        raise ValueError("time_step must be positive.")
    if gamma <= 0:
        raise ValueError("gamma must be positive.")

    num_nodes = graph.num_nodes
    if initial_state is None:
        if start_node < 0 or start_node >= num_nodes:
            raise ValueError("start_node is out of range.")
        state = np.zeros(num_nodes, dtype=np.complex128)
        state[start_node] = 1.0
    else:
        state = np.asarray(initial_state, dtype=np.complex128)
        if state.shape != (num_nodes,):
            raise ValueError("initial_state must have shape (num_nodes,).")
        state = _normalize_initial_state(state)

    hamiltonian = _build_hamiltonian(graph, gamma)
    eigvals, eigvecs = np.linalg.eigh(hamiltonian)
    phases = np.exp(-1j * eigvals * time_step)
    coefficients = eigvecs.conj().T @ state

    positions = np.arange(num_nodes, dtype=int)
    distributions = np.zeros((n_steps + 1, num_nodes), dtype=float)

    coeffs_t = coefficients.copy()
    for step in range(n_steps + 1):
        state_t = eigvecs @ coeffs_t
        probs = np.abs(state_t) ** 2
        distributions[step] = np.real_if_close(probs)
        if step != n_steps:
            coeffs_t = coeffs_t * phases

    entropies = np.apply_along_axis(compute_shannon_entropy, 1, distributions)
    hitting_time = compute_hitting_time(distributions, target_node, threshold)

    return ContinuousTimeQuantumWalkResult(
        graph=graph,
        positions=positions,
        distributions=distributions,
        entropies=entropies,
        gamma=gamma,
        time_step=time_step,
        hitting_time=hitting_time,
    )
