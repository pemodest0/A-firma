import numpy as np


def compute_entropy(P: np.ndarray) -> float:
    p = np.asarray(P).astype(float)
    p = p / p.sum()
    p = p[p>0]
    return float(-(p * np.log(p)).sum())


def compute_alpha(P_t: np.ndarray, t_grid: np.ndarray) -> float:
    """Estimate alpha from variance scaling Var ~ t^alpha using linear fit in log-log."""
    P_t = np.asarray(P_t)
    var_t = np.var(P_t if P_t.ndim > 1 else P_t.reshape(-1,1), axis=1)
    mask = (var_t>0) & (t_grid>0)
    if mask.sum() < 2:
        return 0.0
    logt = np.log(t_grid[mask])
    logv = np.log(var_t[mask])
    a, b = np.polyfit(logt, logv, 1)
    return float(a)


def compute_coherence(rho: np.ndarray) -> float:
    """Soma dos módulos fora da diagonal de rho (quantidade de coerência)."""
    rho = np.asarray(rho)
    assert rho.ndim == 2
    off = rho.copy()
    np.fill_diagonal(off, 0)
    return float(np.sum(np.abs(off)))


def detect_regime(alpha: float, entropy: float) -> str:
    # heuristic thresholds (tunáveis)
    if alpha > 0.1 and entropy > 2.5:
        return 'Interferencia'
    if alpha < -0.02 and entropy > 2.5:
        return 'Caotico'
    return 'Difusivo'
