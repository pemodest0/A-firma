import numpy as np
from .embedding import takens_embedding, auto_delay, false_nearest_neighbors
from .partitioning import entropy_partition, symbolic_approximation, kmeans_partition, dbscan_partition
from .features import permutation_entropy, psd_features
from .graph_builder import build_transition_graph, normalize_graph, validate_graph
from .regime_analysis import compute_entropy


def discretize_system(x, method='entropy', delay=None, dim=None, max_bins=20, alphabet_size=5, partition_method='entropy'):
    x = np.asarray(x)
    if delay is None:
        delay = auto_delay(x)
    if dim is None:
        dim = false_nearest_neighbors(x, max_dim=8, delay=delay)
    emb = takens_embedding(x, delay, dim)
    if method == 'entropy' or partition_method == 'entropy':
        k, labels = entropy_partition(emb, max_bins)
        centers, _ = kmeans_partition(emb, n_clusters=k)
    elif partition_method == 'sax' or method == 'sax':
        labels = symbolic_approximation(x, alphabet_size=alphabet_size)
        centers = None
    elif partition_method == 'dbscan':
        centers, labels = dbscan_partition(emb)
    else:
        centers, labels = kmeans_partition(emb, n_clusters=min(max_bins, 8))
    G = build_transition_graph(labels, directed=True)
    Gn = normalize_graph(G)
    stats = validate_graph(Gn, x)
    stats['entropy_symbolic'] = compute_entropy(np.bincount(labels) / labels.size)
    # additional features
    stats['perm_entropy'] = permutation_entropy(x)
    psd = psd_features(x)
    stats.update(psd)
    stats['n_clusters'] = int(np.unique(labels).size)
    return Gn, labels, centers, stats
