#!/usr/bin/env python3
"""Caminhada clássica adaptativa no hipercubo."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List

import numpy as np

from .encoding import HypercubeEncoder


@dataclass
class ClassicalWalkConfig:
    encoder: HypercubeEncoder
    base_transition: Dict[int, Dict[int, float]]


def neighbors(vertex: int, encoder: HypercubeEncoder) -> Iterable[int]:
    for bit in range(encoder.total_bits):
        yield vertex ^ (1 << bit)


def normalize_transition(weights: Dict[int, float]) -> Dict[int, float]:
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Somatório das probabilidades <= 0")
    return {nbr: w / total for nbr, w in weights.items()}


def simulate_walk(
    config: ClassicalWalkConfig,
    start_vertex: int,
    steps: int,
    weight_func: Callable[[Dict[str, float]], Dict[int, float]] | None = None,
) -> np.ndarray:
    encoder = config.encoder
    probs = np.zeros(encoder.vertex_count(), dtype=float)
    probs[start_vertex] = 1.0
    frames = [probs.copy()]
    for _ in range(steps):
        next_probs = np.zeros_like(probs)
        for vertex, mass in enumerate(probs):
            if mass == 0.0:
                continue
            neigh_weights = config.base_transition.get(vertex)
            if neigh_weights is None:
                weights = {nbr: 1.0 for nbr in neighbors(vertex, encoder)}
            else:
                weights = dict(neigh_weights)
            if weight_func is not None:
                state = encoder.decode(vertex)
                adapt = weight_func(state)
                for nbr, factor in adapt.items():
                    weights[nbr] = weights.get(nbr, 0.0) * factor
            weights = normalize_transition(weights)
            for nbr, prob in weights.items():
                next_probs[nbr] += mass * prob
        frames.append(next_probs.copy())
        probs = next_probs
    return np.stack(frames, axis=0)
