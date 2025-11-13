from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

try:  # Python < 3.8 compatibility
    from typing import Literal
except ImportError:  # pragma: no cover - fallback for older interpreters
    from typing_extensions import Literal  # type: ignore

import numpy as np

from graph_utils import Graph, cycle_graph, line_graph

CoinLabel = Literal["hadamard", "grover"]
MeasurementMode = Literal["none", "projective"]

try:  # pragma: no cover - runtime dependency
    from qiskit.quantum_info import Operator, Statevector  # type: ignore

    QISKIT_AVAILABLE = True
except Exception:  # pragma: no cover - qiskit might be missing
    Operator = None  # type: ignore
    Statevector = None  # type: ignore
    QISKIT_AVAILABLE = False

__all__ = [
    "QuantumWalkResult",
    "hadamard_coin",
    "grover_coin",
    "simulate_quantum_walk",
    "quantum_random_walk_time_series",
    "quantum_random_walk",
    "QISKIT_AVAILABLE",
]


@dataclass
class QuantumWalkResult:
    graph: Graph
    positions: np.ndarray
    distributions: np.ndarray
    entropies: np.ndarray
    hitting_time: Optional[int]
    coin_label: str
    measurement_mode: MeasurementMode
    measurement_distributions: Optional[np.ndarray]
    measurement_entropies: Optional[np.ndarray]
    measurement_hitting_time: Optional[int]


def hadamard_coin() -> np.ndarray:
    factor = 1.0 / math.sqrt(2.0)
    return factor * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)


def grover_coin() -> np.ndarray:
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)


def _coin_matrix(coin: Union[CoinLabel, np.ndarray]) -> Tuple[str, np.ndarray]:
    if isinstance(coin, str):
        label = coin.lower()
        if label == "hadamard":
            return "hadamard", hadamard_coin()
        if label == "grover":
            return "grover", grover_coin()
        raise ValueError("Unknown coin label. Use 'hadamard', 'grover', or provide a 2x2 matrix.")
    matrix = np.asarray(coin, dtype=np.complex128)
    if matrix.shape != (2, 2):
        raise ValueError("Custom coin matrix must be 2x2.")
    return "custom", matrix


def _normalize_coin_state(coin_state: Optional[Tuple[complex, complex]]) -> Tuple[complex, complex]:
    if coin_state is None:
        return complex(1.0), complex(0.0)
    alpha, beta = complex(coin_state[0]), complex(coin_state[1])
    norm = math.sqrt(abs(alpha) ** 2 + abs(beta) ** 2)
    if norm == 0:
        raise ValueError("coin_state cannot be the zero vector.")
    return alpha / norm, beta / norm


def _basis_index(coin: int, position: int, stride: int) -> int:
    return coin * stride + position


def _build_shift_operator(graph: Graph, stride: int) -> np.ndarray:
    num_nodes = graph.num_nodes
    dimension = 2 * stride
    shift = np.eye(dimension, dtype=np.complex128)

    for position in range(num_nodes):
        for coin in (0, 1):
            source = _basis_index(coin, position, stride)

            if graph.is_cycle:
                if coin == 0:
                    target_pos = (position - 1) % num_nodes
                    target_coin = 0
                else:
                    target_pos = (position + 1) % num_nodes
                    target_coin = 1
            else:
                is_left_edge = position == 0
                is_right_edge = position == num_nodes - 1
                if coin == 0:
                    if is_left_edge:
                        target_coin, target_pos = 1, position
                    else:
                        target_coin, target_pos = 0, position - 1
                else:
                    if is_right_edge:
                        target_coin, target_pos = 0, position
                    else:
                        target_coin, target_pos = 1, position + 1

            target = _basis_index(target_coin, target_pos, stride)
            shift[:, source] = 0.0
            shift[target, source] = 1.0
    return shift


def _position_distribution(state: Statevector, num_nodes: int, stride: int) -> np.ndarray:
    data = state.data.reshape(2, stride)
    probs = np.abs(data[0]) ** 2 + np.abs(data[1]) ** 2
    trimmed = np.real_if_close(probs[:num_nodes])
    total = trimmed.sum()
    if total == 0:
        return np.zeros(num_nodes, dtype=float)
    return np.asarray(trimmed / total, dtype=float)


def _shannon_entropy(distribution: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(distribution > 0, np.log2(distribution), 0.0)
    return float(-np.sum(distribution * logp))


def _collapse_to_position(
    state: Statevector,
    position: int,
    stride: int,
) -> Statevector:
    data = state.data.copy()
    idx0 = _basis_index(0, position, stride)
    idx1 = _basis_index(1, position, stride)
    prob = abs(data[idx0]) ** 2 + abs(data[idx1]) ** 2
    if prob == 0:
        return state
    new_state = np.zeros_like(data)
    new_state[idx0] = data[idx0]
    new_state[idx1] = data[idx1]
    new_state /= math.sqrt(prob)
    return Statevector(new_state)


def simulate_quantum_walk(
    graph: Graph,
    n_steps: int,
    coin: Union[CoinLabel, np.ndarray] = "hadamard",
    coin_state: Optional[Tuple[complex, complex]] = None,
    start_node: int = 0,
    initial_distribution: Optional[np.ndarray] = None,
    target_node: Optional[int] = None,
    threshold: float = 0.5,
    measurement: MeasurementMode = "none",
    shots: int = 1024,
    seed: Optional[int] = None,
) -> QuantumWalkResult:
    if measurement not in ("none", "projective"):
        raise ValueError("measurement must be 'none' or 'projective'.")
    if not QISKIT_AVAILABLE:
        raise RuntimeError("Qiskit is required to run the quantum walk simulation.")
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative.")
    num_nodes = graph.num_nodes
    if initial_distribution is None:
        if start_node < 0 or start_node >= num_nodes:
            raise ValueError("start_node is out of range.")
    else:
        distribution = np.asarray(initial_distribution, dtype=float)
        if distribution.shape != (num_nodes,):
            raise ValueError("initial_distribution must have shape (num_nodes,).")
        if np.any(distribution < 0):
            raise ValueError("initial_distribution must be non-negative.")
        total_mass = float(distribution.sum())
        if total_mass <= 0:
            raise ValueError("initial_distribution must have positive mass.")
        distribution = distribution / total_mass

    coin_label, coin_matrix = _coin_matrix(coin)
    alpha, beta = _normalize_coin_state(coin_state)

    pos_qubits = max(1, math.ceil(math.log2(num_nodes)))
    stride = 1 << pos_qubits
    dimension = 2 * stride

    position_eye = np.eye(stride, dtype=np.complex128)
    coin_operator = np.kron(coin_matrix, position_eye)
    shift_operator = _build_shift_operator(graph, stride)
    unitary_matrix = shift_operator @ coin_operator
    unitary = Operator(unitary_matrix)

    initial_state = np.zeros(dimension, dtype=np.complex128)
    if initial_distribution is None:
        initial_state[_basis_index(0, start_node, stride)] = alpha
        initial_state[_basis_index(1, start_node, stride)] = beta
    else:
        amplitudes = np.sqrt(distribution.astype(float))
        for pos in range(num_nodes):
            mag = amplitudes[pos]
            if mag == 0.0:
                continue
            initial_state[_basis_index(0, pos, stride)] = alpha * mag
            initial_state[_basis_index(1, pos, stride)] = beta * mag

    state = Statevector(initial_state)

    positions = np.arange(num_nodes, dtype=int)
    distributions = np.zeros((n_steps + 1, num_nodes), dtype=float)
    entropies = np.zeros(n_steps + 1, dtype=float)

    distributions[0] = _position_distribution(state, num_nodes, stride)
    entropies[0] = _shannon_entropy(distributions[0])

    for step in range(1, n_steps + 1):
        state = state.evolve(unitary)
        dist = _position_distribution(state, num_nodes, stride)
        distributions[step] = dist
        entropies[step] = _shannon_entropy(dist)

    hitting_time = None
    if target_node is not None:
        hits = np.where(distributions[:, target_node] >= threshold)[0]
        hitting_time = int(hits[0]) if hits.size else None

    measurement_distributions = None
    measurement_entropies = None
    measurement_hitting_time = None

    if measurement == "projective" and shots > 0:
        rng = np.random.default_rng(seed)
        counts = np.zeros_like(distributions)
        for shot in range(shots):
            meas_state = Statevector(initial_state)
            dist = _position_distribution(meas_state, num_nodes, stride)
            outcome0 = rng.choice(num_nodes, p=dist)
            counts[0, outcome0] += 1
            meas_state = _collapse_to_position(meas_state, outcome0, stride)
            for step in range(1, n_steps + 1):
                meas_state = meas_state.evolve(unitary)
                dist = _position_distribution(meas_state, num_nodes, stride)
                outcome = rng.choice(num_nodes, p=dist)
                counts[step, outcome] += 1
                meas_state = _collapse_to_position(meas_state, outcome, stride)
        measurement_distributions = counts / shots
        measurement_entropies = np.apply_along_axis(_shannon_entropy, 1, measurement_distributions)
        if target_node is not None:
            hits = np.where(measurement_distributions[:, target_node] >= threshold)[0]
            measurement_hitting_time = int(hits[0]) if hits.size else None

    return QuantumWalkResult(
        graph=graph,
        positions=positions,
        distributions=distributions,
        entropies=entropies,
        hitting_time=hitting_time,
        coin_label=coin_label,
        measurement_mode=measurement,
        measurement_distributions=measurement_distributions,
        measurement_entropies=measurement_entropies,
        measurement_hitting_time=measurement_hitting_time,
    )


def quantum_random_walk_time_series(
    n_steps: int,
    coin: Union[CoinLabel, np.ndarray] = "hadamard",
    coin_state: Optional[Tuple[complex, complex]] = None,
) -> np.ndarray:
    size = 2 * n_steps + 1
    graph = line_graph(size)
    result = simulate_quantum_walk(
        graph,
        n_steps,
        coin=coin,
        coin_state=coin_state,
        start_node=n_steps,
        measurement="none",
    )
    return result.distributions


def quantum_random_walk(
    n_steps: int,
    coin: Union[CoinLabel, np.ndarray] = "hadamard",
    coin_state: Optional[Tuple[complex, complex]] = None,
) -> np.ndarray:
    return quantum_random_walk_time_series(n_steps, coin=coin, coin_state=coin_state)[-1]
