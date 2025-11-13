import networkx as nx
import numpy as np


def build_transition_graph(symbols: np.ndarray, directed: bool = True) -> nx.DiGraph:
    G = nx.DiGraph() if directed else nx.Graph()
    # unique symbols
    unique = np.unique(symbols)
    for u in unique:
        G.add_node(int(u))
    for a, b in zip(symbols[:-1], symbols[1:]):
        if G.has_edge(int(a), int(b)):
            G[int(a)][int(b)]['weight'] += 1
        else:
            G.add_edge(int(a), int(b), weight=1)
    return G


def normalize_graph(G: nx.DiGraph) -> nx.DiGraph:
    G = G.copy()
    for n in G.nodes():
        total = sum(G[n][nbr]['weight'] for nbr in G[n]) if len(list(G[n]))>0 else 0
        if total > 0:
            for nbr in G[n]:
                G[n][nbr]['weight'] /= total
    return G


def validate_graph(G: nx.DiGraph, original_series):
    # basic checks: connected, stochastic rows
    stats = {}
    stats['n_nodes'] = G.number_of_nodes()
    stats['n_edges'] = G.number_of_edges()
    stats['connected'] = nx.is_weakly_connected(G) if G.is_directed() else nx.is_connected(G)
    # row sums
    rows = []
    for n in G.nodes():
        s = sum(G[n][nbr]['weight'] for nbr in G[n]) if len(list(G[n]))>0 else 0.0
        rows.append(s)
    stats['row_sum_min'] = float(np.min(rows)) if rows else 0.0
    stats['row_sum_max'] = float(np.max(rows)) if rows else 0.0
    return stats
