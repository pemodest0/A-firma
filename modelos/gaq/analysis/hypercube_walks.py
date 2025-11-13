"""Reference results for random walks on hypercubes (Staples 2005)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class WalkResult:
    """Container for comparing analytic and simulated hitting statistics."""

    expected_steps: float
    simulated_mean: float
    simulated_std: float
    trials: int


def vertices(n: int) -> np.ndarray:
    """Return all vertices (bitstrings) of the n-dimensional hypercube."""
    return np.array(list(product([0, 1], repeat=n)), dtype=int)


@lru_cache(maxsize=None)
def adjacency(n: int) -> np.ndarray:
    """Return adjacency matrix of the simple hypercube graph."""
    verts = vertices(n)
    m = len(verts)
    idx_map = {_state_index(v): i for i, v in enumerate(verts)}
    adj = np.zeros((m, m), dtype=int)
    for i, v in enumerate(verts):
        for bit in range(n):
            flipped = v.copy()
            flipped[bit] ^= 1
            j = idx_map[_state_index(flipped)]
            adj[i, j] = 1
    return adj


def transition_matrix(n: int) -> np.ndarray:
    """Row-stochastic transition matrix for unbiased walk (flip random bit)."""
    adj = adjacency(n)
    return adj / adj.sum(axis=1, keepdims=True)


def _state_index(state: Iterable[int]) -> int:
    bits = np.array(list(state), dtype=int)
    powers = 1 << np.arange(bits.size - 1, -1, -1)
    return int(np.dot(bits, powers))


def hitting_time_exact(n: int, source: Iterable[int], target: Iterable[int]) -> float:
    """Return expected steps for unbiased walk to hit ``target`` from ``source``."""
    P = transition_matrix(n)
    m = P.shape[0]
    tgt_idx = _state_index(target)
    keep = [i for i in range(m) if i != tgt_idx]
    P_sub = P[np.ix_(keep, keep)]
    ones = np.ones(len(keep))
    h = np.linalg.solve(np.eye(len(keep)) - P_sub, ones)
    src_idx = _state_index(source)
    if src_idx == tgt_idx:
        return 0.0
    src_pos = keep.index(src_idx)
    return float(h[src_pos])


def simulate_hitting_time(
    n: int,
    source: Iterable[int],
    target: Iterable[int],
    trials: int = 10_000,
    rng: np.random.Generator | None = None,
) -> WalkResult:
    """Monte Carlo estimator for hitting times under unbiased walk."""
    rng = rng or np.random.default_rng()
    target_state = np.array(list(target), dtype=int)
    source_state = np.array(list(source), dtype=int)
    target_idx = _state_index(target_state)
    steps = []
    for _ in range(trials):
        state = source_state.copy()
        step = 0
        while _state_index(state) != target_idx:
            bit = rng.integers(0, n)
            state[bit] ^= 1
            step += 1
        steps.append(step)
    expected = hitting_time_exact(n, source_state, target_state)
    sims = np.array(steps, dtype=float)
    return WalkResult(
        expected_steps=expected,
        simulated_mean=float(sims.mean()),
        simulated_std=float(sims.std(ddof=1)),
        trials=trials,
    )
