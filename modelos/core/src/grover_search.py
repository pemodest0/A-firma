from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple

import numpy as np

try:
    from qiskit import Aer, QuantumCircuit, execute
    from qiskit.circuit.library import MCXGate

    QISKIT_AVAILABLE = True
except Exception:  # pragma: no cover - qiskit might not be installed
    QISKIT_AVAILABLE = False
    QuantumCircuit = None  # type: ignore
    MCXGate = None  # type: ignore


def grover_success_probabilities(num_items: int, iterations: int) -> np.ndarray:
    """Analytical Grover success probability for k = 0..iterations."""
    theta = np.arcsin(1 / math.sqrt(num_items))
    k = np.arange(iterations + 1)
    return np.sin((2 * k + 1) * theta) ** 2


@dataclass
class GroverResult:
    iterations: int
    probabilities: List[float]
    counts: Optional[dict] = None
    circuit: Optional["QuantumCircuit"] = None


def build_grover_circuit(num_qubits: int, target_index: int, iterations: int) -> "QuantumCircuit":
    if not QISKIT_AVAILABLE:
        raise RuntimeError("Qiskit is required to build the Grover circuit.")

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))

    target_bits = [(target_index >> i) & 1 for i in range(num_qubits)]

    oracle = QuantumCircuit(num_qubits)
    for qubit, bit in enumerate(target_bits):
        if bit == 0:
            oracle.x(qubit)
    oracle.append(MCXGate(num_qubits - 1), list(range(num_qubits)))
    for qubit, bit in enumerate(target_bits):
        if bit == 0:
            oracle.x(qubit)
    oracle = oracle.to_gate(label="Oracle")

    diffuser = QuantumCircuit(num_qubits)
    diffuser.h(range(num_qubits))
    diffuser.x(range(num_qubits))
    diffuser.append(MCXGate(num_qubits - 1), list(range(num_qubits)))
    diffuser.x(range(num_qubits))
    diffuser.h(range(num_qubits))
    diffuser = diffuser.to_gate(label="Diffuser")

    for _ in range(iterations):
        qc.append(oracle, range(num_qubits))
        qc.append(diffuser, range(num_qubits))

    qc.measure(range(num_qubits), range(num_qubits))
    return qc


def run_grover_simulation(
    num_qubits: int,
    target_index: int,
    iterations: int,
    shots: int = 4096,
) -> GroverResult:
    num_items = 1 << num_qubits
    analytic = grover_success_probabilities(num_items, iterations)

    if not QISKIT_AVAILABLE:
        return GroverResult(iterations, analytic.tolist(), counts=None, circuit=None)

    qc = build_grover_circuit(num_qubits, target_index, iterations)
    backend = Aer.get_backend("qasm_simulator")
    job = execute(qc, backend, shots=shots)
    counts = job.result().get_counts(qc)

    success_prob = counts.get(format(target_index, f"0{num_qubits}b"), 0) / shots
    analytic_list = analytic.tolist()
    analytic_list[-1] = success_prob

    return GroverResult(iterations, analytic_list, counts=counts, circuit=qc)
