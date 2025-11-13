import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def kmeans_partition(states: np.ndarray, n_clusters: int = 8):
    scaler = StandardScaler()
    X = scaler.fit_transform(states)
    k = min(n_clusters, len(states))
    km = KMeans(n_clusters=k, random_state=0).fit(X)
    centers = km.cluster_centers_
    labels = km.labels_
    return centers, labels


def entropy_partition(states: np.ndarray, max_bins: int = 20):
    """Greedy partition that tries different k and picks one maximizing symbolic entropy of labels."""
    best = None
    bestH = -np.inf
    for k in range(2, min(max_bins, len(states)) + 1):
        _, labels = kmeans_partition(states, n_clusters=k)
        # compute entropy
        ps = np.bincount(labels) / labels.size
        H = -(ps[ps>0] * np.log(ps[ps>0])).sum()
        if H > bestH:
            bestH = H
            best = (k, labels)
    return best


def symbolic_approximation(x: np.ndarray, alphabet_size: int = 5):
    """Simple SAX-like implementation: PAA + gaussian breakpoints via quantiles."""
    x = np.asarray(x)
    n = len(x)
    if alphabet_size < 2:
        raise ValueError('alphabet_size must be >=2')
    # normalize
    mu, sigma = x.mean(), x.std()
    if sigma == 0:
        sigma = 1.0
    z = (x - mu) / sigma
    # breakpoints via quantiles
    qs = np.linspace(0, 1, alphabet_size + 1)[1:-1]
    cuts = np.quantile(z, qs)
    symbols = np.digitize(z, cuts)
    return symbols


def dbscan_partition(states: np.ndarray, eps: float = 0.5, min_samples: int = 5):
    from sklearn.cluster import DBSCAN
    X = states.reshape(len(states), -1) if states.ndim == 1 else states
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_
    # relabel noise as new cluster id if present
    if np.any(labels == -1):
        labels = labels.copy()
        noise_id = labels.max() + 1
        labels[labels == -1] = noise_id
    # compute centers as means per label
    centers = []
    for lab in np.unique(labels):
        centers.append(X[labels == lab].mean(axis=0))
    centers = np.array(centers)
    return centers, labels
