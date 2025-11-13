from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class Graph:
    """Minimal undirected graph container for walk simulations."""

    adjacency: np.ndarray
    name: str
    is_cycle: bool

    @property
    def num_nodes(self) -> int:
        return int(self.adjacency.shape[0])

    @property
    def degrees(self) -> np.ndarray:
        return self.adjacency.sum(axis=1)

    def transition_matrix(self) -> np.ndarray:
        """Return the classical random-walk transition matrix P = D^{-1} A."""
        degrees = self.degrees
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_degrees = np.divide(
                1.0,
                degrees,
                out=np.zeros_like(degrees, dtype=float),
                where=degrees > 0,
            )
        return self.adjacency * inv_degrees[:, None]


def _validate_num_nodes(num_nodes: int) -> None:
    if num_nodes < 2:
        raise ValueError("num_nodes must be at least 2.")


def line_graph(num_nodes: int) -> Graph:
    """Return an open 1D chain with reflective boundaries."""
    _validate_num_nodes(num_nodes)
    adjacency = np.zeros((num_nodes, num_nodes), dtype=int)
    for node in range(num_nodes - 1):
        adjacency[node, node + 1] = 1
        adjacency[node + 1, node] = 1
    return Graph(adjacency=adjacency, name=f"Line({num_nodes})", is_cycle=False)


def cycle_graph(num_nodes: int) -> Graph:
    """Return a cycle graph where node i connects to (i±1) mod N."""
    _validate_num_nodes(num_nodes)
    adjacency = np.zeros((num_nodes, num_nodes), dtype=int)
    for node in range(num_nodes):
        adjacency[node, (node + 1) % num_nodes] = 1
        adjacency[(node + 1) % num_nodes, node] = 1
    return Graph(adjacency=adjacency, name=f"Cycle({num_nodes})", is_cycle=True)


def line_and_cycle(num_nodes: int) -> Tuple[Graph, Graph]:
    """Helper returning both line and cycle graphs with shared size."""
    return line_graph(num_nodes), cycle_graph(num_nodes)
