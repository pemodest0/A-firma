"""Randomized compiling stub for noisy GA walk coins."""
from __future__ import annotations

import numpy as np

from gaq.walks.coin_ga import coin_rotor

PAULIS = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}


def _random_pauli(rng: np.random.Generator) -> tuple[str, np.ndarray]:
    label = rng.choice(tuple(PAULIS.keys()))
    return label, PAULIS[label]


def main(num_layers: int = 5, theta: float = np.pi / 4) -> None:
    """Demonstrate randomized compiling by sandwiching the GA coin with random Paulis."""
    rng = np.random.default_rng(2024)
    base_coin = coin_rotor(theta, axis=(0.0, 1.0, 0.0))
    print(f"Base coin (θ={theta:.3f}):\n{base_coin}")
    for layer in range(num_layers):
        pre_label, pre = _random_pauli(rng)
        post_label, post = _random_pauli(rng)
        dressed_coin = post @ base_coin @ pre
        print(f"Layer {layer:02d}: pre={pre_label}, post={post_label}, coin=\n{dressed_coin}")


if __name__ == "__main__":  # pragma: no cover
    main()
