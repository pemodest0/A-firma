from __future__ import annotations

import numpy as np


def corr_to_graph(
    C: np.ndarray,
    method: str = "topk",
    k: int = 10,
    abs_weights: bool = True,
) -> list[tuple[int, int, float]]:
    corr = np.asarray(C, dtype=float)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError("C must be a square matrix")
    n = int(corr.shape[0])
    if n < 2:
        return []

    w = np.abs(corr) if bool(abs_weights) else corr.copy()
    np.fill_diagonal(w, 0.0)
    w = np.where(np.isfinite(w), w, 0.0)

    edges: dict[tuple[int, int], float] = {}
    m = str(method).strip().lower()
    if m == "dense":
        for i in range(n):
            for j in range(i + 1, n):
                wij = float(w[i, j])
                if wij > 0.0:
                    edges[(i, j)] = wij
    elif m == "topk":
        kk = int(max(1, min(int(k), n - 1)))
        for i in range(n):
            row = w[i].copy()
            row[i] = 0.0
            idx = np.argpartition(row, -kk)[-kk:]
            idx = idx[np.argsort(row[idx])[::-1]]
            for j in idx.tolist():
                if i == j:
                    continue
                wij = float(row[j])
                if wij <= 0.0:
                    continue
                a, b = (i, int(j)) if i < int(j) else (int(j), i)
                old = edges.get((a, b))
                if old is None or wij > float(old):
                    edges[(a, b)] = wij
    else:
        raise ValueError(f"unknown method: {method}")

    out = [(int(i), int(j), float(wij)) for (i, j), wij in sorted(edges.items())]
    return out
