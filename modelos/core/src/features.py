import numpy as np
from scipy.signal import periodogram


def permutation_entropy(x, order=3, delay=1):
    x = np.asarray(x)
    n = len(x)
    if n < order * delay:
        return 0.0
    perms = {}
    for i in range(n - (order - 1) * delay):
        window = x[i:i + order * delay:delay]
        ranks = tuple(np.argsort(window))
        perms[ranks] = perms.get(ranks, 0) + 1
    ps = np.array(list(perms.values()), dtype=float)
    ps = ps / ps.sum()
    H = -(ps * np.log(ps)).sum()
    return float(H)


def psd_features(x, fs=1.0):
    f, Pxx = periodogram(x, fs=fs)
    if Pxx.sum() == 0:
        return {'p_tot': 0.0, 'p_peak': 0.0, 'f_peak': 0.0}
    idx = np.argmax(Pxx)
    return {'p_tot': float(Pxx.sum()), 'p_peak': float(Pxx[idx]), 'f_peak': float(f[idx])}
