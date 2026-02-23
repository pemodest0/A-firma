from __future__ import annotations

import numpy as np

from engine.structural.graph import corr_to_graph


def test_corr_to_graph_topk_returns_edges() -> None:
    c = np.array(
        [
            [1.0, 0.9, 0.2, 0.1],
            [0.9, 1.0, 0.3, 0.0],
            [0.2, 0.3, 1.0, 0.8],
            [0.1, 0.0, 0.8, 1.0],
        ],
        dtype=float,
    )
    edges = corr_to_graph(c, method="topk", k=1, abs_weights=True)
    assert len(edges) >= 2
    assert all(len(e) == 3 for e in edges)
    assert all(e[2] > 0 for e in edges)


def test_corr_to_graph_dense_has_more_or_equal_edges_than_topk() -> None:
    rng = np.random.default_rng(23)
    x = rng.normal(size=(200, 12))
    c = np.corrcoef(x, rowvar=False)
    e_topk = corr_to_graph(c, method="topk", k=2, abs_weights=True)
    e_dense = corr_to_graph(c, method="dense", k=2, abs_weights=True)
    assert len(e_dense) >= len(e_topk)
