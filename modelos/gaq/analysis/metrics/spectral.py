"""Spectral diagnostics for hypercube walks."""
from __future__ import annotations

import numpy as np

from gaq.analysis.hypercube_walks import transition_matrix


def spectral_gap_unbiased(n: int) -> tuple[np.ndarray, float]:
    """Return eigenvalues and spectral gap (1-λ2) for unbiased walk on Q_n."""
    P = transition_matrix(n)
    evals = np.linalg.eigvals(P)
    evals = np.sort(np.real_if_close(evals))[::-1]
    gap = float(1 - evals[1])
    return evals, gap
