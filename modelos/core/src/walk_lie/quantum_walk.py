#!/usr/bin/env python3
"""Caminhada quântica adaptativa no hipercubo."""
from __future__ import annotations

import math
from typing import Callable, Dict

import numpy as np

from .encoding import HypercubeEncoder


def grover_coin(d: int) -> np.ndarray:
    ones = np.ones((d, d), dtype=np.complex128)
    return (2.0 / d) * ones - np.eye(d, dtype=np.complex128)


def apply_coin(psi: np.ndarray, coin: np.ndarray) -> np.ndarray:
    return psi @ coin.T


def apply_shift(psi: np.ndarray, encoder: HypercubeEncoder) -> np.ndarray:
    d = encoder.total_bits
    out = np.zeros_like(psi)
    for vertex in range(encoder.vertex_count()):
        for k in range(d):
            nbr = vertex ^ (1 << k)
            out[nbr, k] += psi[vertex, k]
    return out


def normalize_state(psi: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(psi)
    if norm == 0:
        raise ValueError("Estado quântico degenerado")
    return psi / norm


def simulate_quantum_walk(
    encoder: HypercubeEncoder,
    steps: int,
    start_vertex: int,
    coin_fn: Callable[[Dict[str, float]], np.ndarray] | None = None,
) -> np.ndarray:
    d = encoder.total_bits
    psi = np.zeros((encoder.vertex_count(), d), dtype=np.complex128)
    psi[start_vertex, :] = 1.0 / math.sqrt(d)
    frames = []
    for step in range(steps + 1):
        probs = (np.abs(psi) ** 2).sum(axis=1).real
        frames.append(probs)
        if step == steps:
            break
        state = encoder.decode(start_vertex)
        coin = coin_fn(state) if coin_fn is not None else grover_coin(d)
        psi = normalize_state(apply_shift(apply_coin(psi, coin), encoder))
    return np.stack(frames, axis=0)
