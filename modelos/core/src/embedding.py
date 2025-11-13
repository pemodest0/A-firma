import numpy as np


def takens_embedding(x: np.ndarray, delay: int, dim: int) -> np.ndarray:
    """Reconstrói o espaço de fase via embedding de Takens.
    Retorna matriz shape (N - delay*(dim-1), dim).
    """
    x = np.asarray(x)
    n = x.shape[0]
    L = n - delay * (dim - 1)
    if L <= 0:
        raise ValueError('Time series too short for given delay and dim')
    M = np.empty((L, dim), dtype=float)
    for i in range(dim):
        M[:, i] = x[i * delay:i * delay + L]
    return M


def auto_delay(x: np.ndarray, max_lag: int = 100) -> int:
    """Estimativa simples de atraso via autocorrelação: escolhe primeiro lag onde autocorr cruza 1/e.
    """
    x = np.asarray(x)
    x = x - x.mean()
    n = len(x)
    max_lag = min(max_lag, n - 1)
    acf = np.array([np.corrcoef(x[:-lag], x[lag:])[0, 1] if lag > 0 else 1.0 for lag in range(0, max_lag + 1)])
    # find first lag where acf < 1/e
    thresh = 1.0 / np.e
    for lag in range(1, len(acf)):
        if abs(acf[lag]) < thresh:
            return lag
    return max(1, int(max_lag // 10))


def false_nearest_neighbors(x: np.ndarray, max_dim: int = 10, delay: int = 1, rtol: float = 10.0) -> int:
    """Estimativa simples do número mínimo de dimensões via FNN (versão reduzida).
    Retorna dimensão estimada <= max_dim.
    """
    # naive: increase dim until embedding no longer reduces nearest-neighbor distances significantly
    x = np.asarray(x)
    n = len(x)
    for dim in range(1, max_dim + 1):
        try:
            M = takens_embedding(x, delay, dim)
        except ValueError:
            break
        if M.shape[0] < 10:
            return dim
        # compute nearest neighbor distances
        from sklearn.neighbors import NearestNeighbors
        # fit on M but ensure n_neighbors <= samples
        n_samples = M.shape[0]
        k_nn = 2 if n_samples >= 2 else 1
        nbrs = NearestNeighbors(n_neighbors=k_nn).fit(M)
        dists, idxs = nbrs.kneighbors(M)
        if k_nn == 1:
            nn_dist = np.zeros(n_samples)
        else:
            nn_dist = dists[:, 1]
        if dim > 1:
            # compute expansion when going to next dim
            try:
                M_next = takens_embedding(x, delay, dim + 1)
            except ValueError:
                break
            n2 = M_next.shape[0]
            # align lengths by truncating to min length
            L = min(n_samples, n2)
            if L < 5:
                return dim
            M_cut = M[:L]
            Mnext_cut = M_next[:L]
            k_nn2 = 2 if L >= 2 else 1
            nbrs2 = NearestNeighbors(n_neighbors=k_nn2).fit(Mnext_cut)
            d2, _ = nbrs2.kneighbors(Mnext_cut)
            if k_nn2 == 1:
                nn_dist2 = np.zeros(L)
            else:
                nn_dist2 = d2[:, 1]
            nn_dist_cut = nn_dist[:L]
            ratio = np.median(nn_dist2 / (nn_dist_cut + 1e-12))
            if ratio < rtol:
                return dim
    return max_dim
