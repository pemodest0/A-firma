from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from graph_utils import Graph
from walk_lie.encoding import FeatureBin, HypercubeEncoder

BuilderFn = Callable[..., "GraphCandidate"]


@dataclass
class GraphCandidate:
    family: str
    graph: Graph
    params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


def _ensure_dataframe(table: Any) -> pd.DataFrame:
    if isinstance(table, pd.DataFrame):
        return table
    raise TypeError("Expected a pandas.DataFrame as input table.")


def _build_hypercube_encoder(
    table: pd.DataFrame,
    feature_columns: Sequence[str],
    bins: Optional[Dict[str, Sequence[float]]] = None,
    bin_count: int = 4,
) -> HypercubeEncoder:
    feature_bins: List[FeatureBin] = []
    for column in feature_columns:
        values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            raise ValueError(f"Column '{column}' has no finite values.")
        if bins and column in bins:
            thresholds = list(bins[column])
        else:
            quantiles = np.linspace(0, 1, num=max(2, bin_count + 1))[1:-1]
            thresholds = [float(np.quantile(finite_values, q)) for q in quantiles]
        lower = float(np.nanmin(finite_values))
        upper = float(np.nanmax(finite_values))
        feature_bins.append(
            FeatureBin(
                name=column,
                bins=thresholds,
                lower_bound=lower,
                upper_bound=upper,
            )
        )
    return HypercubeEncoder(feature_bins)


def build_hypercube_graph(
    table: Any,
    *,
    feature_columns: Optional[Sequence[str]] = None,
    bins: Optional[Dict[str, Sequence[float]]] = None,
    bin_count: int = 4,
    max_vertices: int = 4096,
) -> GraphCandidate:
    df = _ensure_dataframe(table)
    if feature_columns is None:
        feature_columns = [
            col
            for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col]) and col not in {"return_t1", "label"}
        ]
    if not feature_columns:
        raise ValueError("No feature columns provided for hypercube graph.")
    encoder = _build_hypercube_encoder(df, feature_columns, bins=bins, bin_count=bin_count)
    total_vertices = encoder.vertex_count()
    if total_vertices > max_vertices:
        raise ValueError(
            f"Hypercube with {encoder.total_bits} bits would create {total_vertices} vertices. "
            f"Increase max_vertices explicitly if desejado."
        )
    adjacency = np.zeros((total_vertices, total_vertices), dtype=int)
    for vertex in range(total_vertices):
        for bit in range(encoder.total_bits):
            neighbor = vertex ^ (1 << bit)
            adjacency[vertex, neighbor] = 1
    graph = Graph(adjacency=adjacency, name=f"Hypercube({encoder.total_bits})", is_cycle=False)

    vertex_assignments = []
    for _, row in df[feature_columns].iterrows():
        values = {col: float(row[col]) for col in feature_columns}
        vertex_assignments.append(encoder.encode(values))

    metadata = {
        "encoder": encoder,
        "feature_columns": list(feature_columns),
        "vertex_assignments": np.asarray(vertex_assignments, dtype=int),
    }
    return GraphCandidate(
        family="hypercube",
        graph=graph,
        params={
            "bin_count": bin_count,
            "feature_columns": list(feature_columns),
            "max_vertices": max_vertices,
        },
        metadata=metadata,
    )


def build_knn_graph(
    table: Any,
    *,
    feature_columns: Optional[Sequence[str]] = None,
    k: int = 10,
    epsilon: Optional[float] = None,
) -> GraphCandidate:
    df = _ensure_dataframe(table)
    if feature_columns is None:
        feature_columns = [
            col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])
        ]
    if not feature_columns:
        raise ValueError("No feature columns provided for kNN graph.")

    coords = df[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if np.isnan(coords).any():
        raise ValueError("kNN graph received NaN values after coercion.")
    num_points = coords.shape[0]
    if num_points == 0:
        raise ValueError("Table is empty, cannot build kNN graph.")
    adjacency = np.zeros((num_points, num_points), dtype=int)
    for idx in range(num_points):
        diff = coords - coords[idx]
        distances = np.sqrt(np.sum(diff * diff, axis=1))
        distances[idx] = np.inf
        if epsilon is not None:
            neighbors = np.where(distances <= epsilon)[0]
        else:
            neighbors = np.argpartition(distances, min(k, num_points - 1))[:k]
        for nb in neighbors:
            if nb == idx:
                continue
            adjacency[idx, nb] = 1
            adjacency[nb, idx] = 1
    graph = Graph(adjacency=adjacency, name=f"kNN({k})", is_cycle=False)
    metadata = {
        "feature_columns": list(feature_columns),
        "coordinates": coords,
    }
    return GraphCandidate(
        family="knn",
        graph=graph,
        params={"k": k, "epsilon": epsilon, "feature_columns": list(feature_columns)},
        metadata=metadata,
    )


def build_multilayer_graph(
    table: Any,
    *,
    layer_col: str,
    entity_col: str,
    feature_columns: Optional[Sequence[str]] = None,
    intra_k: int = 5,
) -> GraphCandidate:
    df = _ensure_dataframe(table)
    if layer_col not in df.columns or entity_col not in df.columns:
        raise ValueError("layer_col and entity_col must be present in the dataframe.")
    if feature_columns is None:
        feature_columns = [
            col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])
        ]
    coords = df[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if np.isnan(coords).any():
        raise ValueError("multilayer graph received NaN values in features.")

    df = df.reset_index(drop=True)
    num_nodes = len(df)
    adjacency = np.zeros((num_nodes, num_nodes), dtype=int)

    groups = df.groupby(layer_col)
    for _, layer_df in groups:
        idxs = layer_df.index.to_numpy()
        layer_coords = coords[idxs]
        for pos, idx in enumerate(idxs):
            diff = layer_coords - layer_coords[pos]
            distances = np.sqrt(np.sum(diff * diff, axis=1))
            distances[pos] = np.inf
            neighbors = idxs[np.argpartition(distances, min(intra_k, len(distances) - 1))[:intra_k]]
            for nb in neighbors:
                if nb == idx:
                    continue
                adjacency[idx, nb] = 1
                adjacency[nb, idx] = 1

    entity_groups = df.groupby(entity_col)
    for _, entity_df in entity_groups:
        sorted_df = entity_df.sort_values(layer_col)
        idxs = sorted_df.index.to_list()
        for prev, cur in zip(idxs[:-1], idxs[1:]):
            adjacency[prev, cur] = 1
            adjacency[cur, prev] = 1

    graph = Graph(adjacency=adjacency, name="Multilayer", is_cycle=False)
    metadata = {
        "layer_col": layer_col,
        "entity_col": entity_col,
        "feature_columns": list(feature_columns),
    }
    return GraphCandidate(
        family="multilayer",
        graph=graph,
        params={"intra_k": intra_k},
        metadata=metadata,
    )


def build_bipartite_graph(
    table: Any,
    *,
    left_col: str,
    right_col: str,
) -> GraphCandidate:
    df = _ensure_dataframe(table)
    if left_col not in df.columns or right_col not in df.columns:
        raise ValueError("left_col and right_col must exist in the dataframe.")
    left_nodes = sorted(df[left_col].dropna().unique())
    right_nodes = sorted(df[right_col].dropna().unique())
    left_index = {node: idx for idx, node in enumerate(left_nodes)}
    right_index = {node: idx for idx, node in enumerate(right_nodes)}

    num_left = len(left_nodes)
    num_right = len(right_nodes)
    total_nodes = num_left + num_right
    adjacency = np.zeros((total_nodes, total_nodes), dtype=int)

    for _, row in df.iterrows():
        l_val = row[left_col]
        r_val = row[right_col]
        if pd.isna(l_val) or pd.isna(r_val):
            continue
        l_idx = left_index[l_val]
        r_idx = num_left + right_index[r_val]
        adjacency[l_idx, r_idx] = 1
        adjacency[r_idx, l_idx] = 1

    graph = Graph(adjacency=adjacency, name="Bipartite", is_cycle=False)
    metadata = {
        "left_nodes": left_nodes,
        "right_nodes": right_nodes,
        "left_col": left_col,
        "right_col": right_col,
    }
    return GraphCandidate(
        family="bipartite",
        graph=graph,
        params={},
        metadata=metadata,
    )


def _compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def build_mst_shortcuts_graph(
    table: Any,
    *,
    feature_columns: Sequence[str],
    shortcut_quantile: float = 0.9,
) -> GraphCandidate:
    df = _ensure_dataframe(table)
    coords = df[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if np.isnan(coords).any():
        raise ValueError("mst_shortcuts graph received NaN values in features.")
    num_nodes = coords.shape[0]
    if num_nodes == 0:
        raise ValueError("Table is empty, cannot build MST graph.")

    distances = _compute_distance_matrix(coords)
    adjacency = np.zeros((num_nodes, num_nodes), dtype=int)
    in_tree = np.zeros(num_nodes, dtype=bool)
    in_tree[0] = True
    edges: List[Tuple[int, int, float]] = []
    for _ in range(num_nodes - 1):
        mask = np.outer(in_tree, ~in_tree)
        if not np.any(mask):
            break
        masked = np.where(mask, distances, np.inf)
        idx = np.argmin(masked)
        i, j = divmod(idx, num_nodes)
        adjacency[i, j] = 1
        adjacency[j, i] = 1
        edges.append((i, j, distances[i, j]))
        in_tree[j] = True

    threshold = np.quantile(distances[np.isfinite(distances)], shortcut_quantile)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adjacency[i, j] == 1:
                continue
            if distances[i, j] <= threshold:
                adjacency[i, j] = 1
                adjacency[j, i] = 1

    graph = Graph(adjacency=adjacency, name="MST+Shortcuts", is_cycle=False)
    metadata = {
        "feature_columns": list(feature_columns),
        "shortcut_quantile": shortcut_quantile,
    }
    return GraphCandidate(
        family="mst_shortcuts",
        graph=graph,
        params={"shortcut_quantile": shortcut_quantile},
        metadata=metadata,
    )


def build_line_graph_from_base(
    table: Any = None,
    *,
    base_graph: Graph,
) -> GraphCandidate:
    adjacency = base_graph.adjacency
    edges: List[Tuple[int, int]] = []
    num_nodes = base_graph.num_nodes
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adjacency[i, j]:
                edges.append((i, j))
    edge_count = len(edges)
    line_adj = np.zeros((edge_count, edge_count), dtype=int)
    for idx, (u1, v1) in enumerate(edges):
        for jdx in range(idx + 1, edge_count):
            u2, v2 = edges[jdx]
            if len({u1, v1, u2, v2}) < 4:
                line_adj[idx, jdx] = 1
                line_adj[jdx, idx] = 1
    graph = Graph(adjacency=line_adj, name=f"LineGraph({base_graph.name})", is_cycle=False)
    metadata = {
        "base_edges": edges,
        "base_graph": base_graph,
    }
    return GraphCandidate(
        family="line_graph",
        graph=graph,
        params={},
        metadata=metadata,
    )


def register(name: str, builder: BuilderFn) -> None:
    registry[name] = builder


registry: Dict[str, BuilderFn] = {
    "hypercube": build_hypercube_graph,
    "knn": build_knn_graph,
    "multilayer": build_multilayer_graph,
    "bipartite": build_bipartite_graph,
    "mst_shortcuts": build_mst_shortcuts_graph,
    "line_graph": build_line_graph_from_base,
}
