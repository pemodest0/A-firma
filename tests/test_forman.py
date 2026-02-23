from __future__ import annotations

import numpy as np

from engine.structural.forman_ricci import forman_edge_curvature, forman_summary


def _line_edges(n: int) -> list[tuple[int, int, float]]:
    return [(i, i + 1, 1.0) for i in range(n - 1)]


def _complete_edges(n: int) -> list[tuple[int, int, float]]:
    out: list[tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            out.append((i, j, 1.0))
    return out


def test_forman_line_vs_complete_sanity() -> None:
    n = 8
    curv_line = forman_edge_curvature(_line_edges(n), n_nodes=n)
    curv_full = forman_edge_curvature(_complete_edges(n), n_nodes=n)

    s_line = forman_summary(curv_line)
    s_full = forman_summary(curv_full)

    assert np.isfinite(s_line["mean"])
    assert np.isfinite(s_full["mean"])
    assert not np.isclose(s_line["mean"], s_full["mean"])
    assert s_line["n_edges"] != s_full["n_edges"]
