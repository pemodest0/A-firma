from __future__ import annotations

from collections import defaultdict

import numpy as np


def forman_edge_curvature(edges: list[tuple[int, int, float]], n_nodes: int) -> np.ndarray:
    n = int(max(0, n_nodes))
    if n <= 0 or len(edges) == 0:
        return np.asarray([], dtype=float)

    adj: dict[int, list[tuple[int, float]]] = defaultdict(list)
    node_w = np.zeros(n, dtype=float)
    for u, v, w in edges:
        uu = int(u)
        vv = int(v)
        ww = float(max(1e-12, w))
        if not (0 <= uu < n and 0 <= vv < n):
            continue
        adj[uu].append((vv, ww))
        adj[vv].append((uu, ww))
        node_w[uu] += ww
        node_w[vv] += ww
    node_w = np.where(node_w > 1e-12, node_w, 1.0)

    curv = np.full(len(edges), np.nan, dtype=float)
    for idx, (u, v, w) in enumerate(edges):
        uu = int(u)
        vv = int(v)
        ww = float(max(1e-12, w))
        if not (0 <= uu < n and 0 <= vv < n):
            continue
        wu = float(node_w[uu])
        wv = float(node_w[vv])

        su = 0.0
        for nei, w2 in adj[uu]:
            if int(nei) == vv:
                continue
            su += wu / float(np.sqrt(max(1e-12, ww * w2)))

        sv = 0.0
        for nei, w2 in adj[vv]:
            if int(nei) == uu:
                continue
            sv += wv / float(np.sqrt(max(1e-12, ww * w2)))

        curv[idx] = float(ww * (((wu + wv) / ww) - su - sv))
    return curv


def forman_summary(curvatures: np.ndarray | list[float]) -> dict[str, float]:
    c = np.asarray(curvatures, dtype=float)
    c = c[np.isfinite(c)]
    if c.size == 0:
        return {
            "n_edges": 0.0,
            "mean": float("nan"),
            "p5": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "share_negative": float("nan"),
        }
    return {
        "n_edges": float(c.size),
        "mean": float(np.mean(c)),
        "p5": float(np.percentile(c, 5)),
        "p50": float(np.percentile(c, 50)),
        "p95": float(np.percentile(c, 95)),
        "share_negative": float(np.mean(c < 0.0)),
    }
