from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from graph_utils import Graph
from classical_walk import compute_shannon_entropy, compute_hitting_time
from quantum_walk import hadamard_coin, grover_coin  # reuse canonical coins

try:  # pragma: no cover - optional dependency
    from qiskit.quantum_info import DensityMatrix, Operator  # type: ignore
    from qiskit.quantum_info.operators.channel import Kraus  # type: ignore

    QISKIT_AVAILABLE = True
except Exception:  # pragma: no cover - safe fallback if qiskit missing
    DensityMatrix = None  # type: ignore
    Operator = None  # type: ignore
    Kraus = None  # type: ignore
    QISKIT_AVAILABLE = False

CoinLabel = Union[str, np.ndarray]
NoiseSpec = Dict[str, Any]

__all__ = [
    "NoisyQuantumWalkResult",
    "simulate_noisy_quantum_walk",
    "QISKIT_AVAILABLE",
]


@dataclass
class NoisyQuantumWalkResult:
    graph: Graph
    positions: np.ndarray
    distributions: np.ndarray
    entropies: np.ndarray
    coin_label: str
    noise_profile: Sequence[NoiseSpec]
    hitting_time: Optional[int]


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
                    target_coin, target_pos = 0, (position - 1) % num_nodes
                else:
                    target_coin, target_pos = 1, (position + 1) % num_nodes
            else:
                left_edge = position == 0
                right_edge = position == num_nodes - 1
                if coin == 0:
                    if left_edge:
                        target_coin, target_pos = 1, position
                    else:
                        target_coin, target_pos = 0, position - 1
                else:
                    if right_edge:
                        target_coin, target_pos = 0, position
                    else:
                        target_coin, target_pos = 1, position + 1
            target = _basis_index(target_coin, target_pos, stride)
            shift[:, source] = 0.0
            shift[target, source] = 1.0
    return shift


def _coin_matrix(coin: CoinLabel) -> Tuple[str, np.ndarray]:
    if isinstance(coin, str):
        label = coin.lower()
        if label == "hadamard":
            return "hadamard", hadamard_coin()
        if label == "grover":
            return "grover", grover_coin()
        raise ValueError("Unknown coin label. Use 'hadamard', 'grover', or supply a 2x2 matrix.")
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


def _position_distribution(rho: "DensityMatrix", num_nodes: int, stride: int) -> np.ndarray:
    probs = np.asarray(rho.probabilities(), dtype=float)
    dist = np.zeros(num_nodes, dtype=float)
    for pos in range(num_nodes):
        idx0 = _basis_index(0, pos, stride)
        idx1 = _basis_index(1, pos, stride)
        dist[pos] = probs[idx0] + probs[idx1]
    total = dist.sum()
    if total == 0:
        return dist
    return dist / total


def _clip_strength(value: float) -> float:
    if value < 0:
        raise ValueError("Noise strength must be non-negative.")
    return min(value, 1.0)


def _phase_damping_channel(strength: float) -> "Kraus":
    gamma = _clip_strength(strength)
    k0 = np.array([[1.0, 0.0], [0.0, math.sqrt(1.0 - gamma)]], dtype=np.complex128)
    k1 = np.array([[0.0, 0.0], [0.0, math.sqrt(gamma)]], dtype=np.complex128)
    return Kraus([k0, k1])


def _phase_flip_channel(strength: float) -> "Kraus":
    p = _clip_strength(strength)
    k0 = math.sqrt(1.0 - p) * np.eye(2, dtype=np.complex128)
    k1 = math.sqrt(p) * np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    return Kraus([k0, k1])


def _bit_flip_channel(strength: float) -> "Kraus":
    p = _clip_strength(strength)
    k0 = math.sqrt(1.0 - p) * np.eye(2, dtype=np.complex128)
    k1 = math.sqrt(p) * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    return Kraus([k0, k1])


def _amplitude_damping_channel(strength: float) -> "Kraus":
    gamma = _clip_strength(strength)
    k0 = np.array([[1.0, 0.0], [0.0, math.sqrt(1.0 - gamma)]], dtype=np.complex128)
    k1 = np.array([[0.0, math.sqrt(gamma)], [0.0, 0.0]], dtype=np.complex128)
    return Kraus([k0, k1])


def _depolarizing_channel(strength: float) -> "Kraus":
    p = _clip_strength(strength)
    k0 = math.sqrt(1.0 - p) * np.eye(2, dtype=np.complex128)
    factor = math.sqrt(p / 3.0) if p > 0 else 0.0
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    pauli_y = np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
    pauli_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    return Kraus([k0, factor * pauli_x, factor * pauli_y, factor * pauli_z])


def _single_qubit_channel(label: str, strength: float) -> "Kraus":
    if label == "phase":
        return _phase_damping_channel(strength)
    if label == "phase_flip":
        return _phase_flip_channel(strength)
    if label == "bit_flip":
        return _bit_flip_channel(strength)
    if label == "amplitude":
        return _amplitude_damping_channel(strength)
    if label == "depolarizing":
        return _depolarizing_channel(strength)
    raise ValueError(f"Unsupported single-qubit channel '{label}'.")


def _prepare_noise_operations(
    specs: Iterable[NoiseSpec],
    pos_qubits: int,
) -> List[Tuple[Any, Optional[Tuple[int, ...]]]]:
    operations: List[Tuple[Any, Optional[Tuple[int, ...]]]] = []
    for spec in specs:
        label = spec.get("type", "").lower()
        if not label:
            raise ValueError("Noise spec must include a 'type' field.")
        target = spec.get("target", "coin").lower()
        strength = float(spec.get("strength", spec.get("gamma", 0.0)))
        channel = _single_qubit_channel(label, strength)
        indices = _resolve_target_indices(target, pos_qubits)
        for idx in indices:
            operations.append((channel, (idx,)))
    return operations


def _resolve_target_indices(target: str, pos_qubits: int) -> Tuple[int, ...]:
    if target == "coin":
        return (pos_qubits,)
    if target in ("positions", "position", "pos"):
        return tuple(range(pos_qubits))
    if target in ("all", "global"):
        return tuple(range(pos_qubits + 1))
    if target.startswith("q"):
        index = int(target[1:])
        if index < 0 or index > pos_qubits:
            raise ValueError("Target qubit index out of range.")
        return (index,)
    raise ValueError(f"Unknown noise target '{target}'.")


def simulate_noisy_quantum_walk(
    graph: Graph,
    n_steps: int,
    coin: CoinLabel = "hadamard",
    coin_state: Optional[Tuple[complex, complex]] = None,
    start_node: int = 0,
    initial_distribution: Optional[np.ndarray] = None,
    target_node: Optional[int] = None,
    threshold: float = 0.5,
    noise_profile: Optional[Sequence[NoiseSpec]] = None,
) -> NoisyQuantumWalkResult:
    if not QISKIT_AVAILABLE:
        raise RuntimeError("Qiskit is required to run the noisy quantum walk simulation.")
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative.")
    num_nodes = graph.num_nodes
    if initial_distribution is None:
        if start_node < 0 or start_node >= num_nodes:
            raise ValueError("start_node is out of range.")
    coin_label, coin_matrix = _coin_matrix(coin)
    alpha, beta = _normalize_coin_state(coin_state)

    pos_qubits = max(1, math.ceil(math.log2(num_nodes)))
    stride = 1 << pos_qubits
    dimension = 2 * stride

    if initial_distribution is None:
        initial_state = np.zeros(dimension, dtype=np.complex128)
        initial_state[_basis_index(0, start_node, stride)] = alpha
        initial_state[_basis_index(1, start_node, stride)] = beta
    else:
        distribution = np.asarray(initial_distribution, dtype=float)
        if distribution.shape != (num_nodes,):
            raise ValueError("initial_distribution must have shape (num_nodes,).")
        if np.any(distribution < 0):
            raise ValueError("initial_distribution must be non-negative.")
        total = float(distribution.sum())
        if total <= 0:
            raise ValueError("initial_distribution must have positive mass.")
        distribution = distribution / total
        amplitudes = np.sqrt(distribution.astype(float))
        initial_state = np.zeros(dimension, dtype=np.complex128)
        for pos in range(num_nodes):
            mag = amplitudes[pos]
            if mag == 0.0:
                continue
            initial_state[_basis_index(0, pos, stride)] = alpha * mag
            initial_state[_basis_index(1, pos, stride)] = beta * mag

    pos_eye = np.eye(stride, dtype=np.complex128)
    coin_operator = np.kron(coin_matrix, pos_eye)
    shift_operator = _build_shift_operator(graph, stride)
    unitary_matrix = shift_operator @ coin_operator
    unitary = Operator(unitary_matrix)

    rho = DensityMatrix(initial_state)

    operations = _prepare_noise_operations(noise_profile or (), pos_qubits=pos_qubits)

    positions = np.arange(num_nodes, dtype=int)
    distributions = np.zeros((n_steps + 1, num_nodes), dtype=float)

    distributions[0] = _position_distribution(rho, num_nodes, stride)
    for step in range(1, n_steps + 1):
        rho = rho.evolve(unitary)
        for channel, qargs in operations:
            rho = rho.evolve(channel, qargs=qargs)
        distributions[step] = _position_distribution(rho, num_nodes, stride)

    entropies = np.apply_along_axis(compute_shannon_entropy, 1, distributions)
    hitting_time = compute_hitting_time(distributions, target_node, threshold)

    return NoisyQuantumWalkResult(
        graph=graph,
        positions=positions,
        distributions=distributions,
        entropies=entropies,
        coin_label=coin_label,
        noise_profile=list(noise_profile or []),
        hitting_time=hitting_time,
    )
