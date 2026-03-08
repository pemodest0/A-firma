from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def _correl_dist(corr: pd.DataFrame) -> pd.DataFrame:
    clipped = corr.clip(-1.0, 1.0)
    return np.sqrt(0.5 * (1.0 - clipped))


def _quasi_diag(link: np.ndarray) -> list[int]:
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = int(link[-1, 3])
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix.loc[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df1]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.astype(int).tolist()


def _cluster_var(cov: pd.DataFrame, items: list[str]) -> float:
    sub = cov.loc[items, items]
    diag = np.clip(np.diag(sub.to_numpy(dtype=float)), 1e-8, None)
    inv_diag = 1.0 / diag
    weights = inv_diag / max(inv_diag.sum(), 1e-12)
    return float(weights @ sub.to_numpy(dtype=float) @ weights)


def hrp_weights(
    cov: pd.DataFrame,
    *,
    corr: pd.DataFrame | None = None,
    linkage_method: str = "single",
) -> pd.Series:
    cov_df = cov.copy().astype(float)
    labels = cov_df.index.astype(str).tolist()
    if len(labels) == 1:
        return pd.Series([1.0], index=labels, dtype=float)
    corr_df = corr.copy().astype(float) if corr is not None else cov_df.corr()
    dist = _correl_dist(corr_df)
    condensed = squareform(dist.to_numpy(dtype=float), checks=False)
    link = linkage(condensed, method=str(linkage_method))
    sort_ix = _quasi_diag(link)
    ordered = corr_df.index[sort_ix].astype(str).tolist()
    weights = pd.Series(1.0, index=ordered, dtype=float)
    clusters = [ordered]
    while clusters:
        cluster = clusters.pop(0)
        if len(cluster) <= 1:
            continue
        split = len(cluster) // 2
        left = cluster[:split]
        right = cluster[split:]
        left_var = _cluster_var(cov_df, left)
        right_var = _cluster_var(cov_df, right)
        alpha = 1.0 - left_var / max(left_var + right_var, 1e-12)
        weights[left] *= alpha
        weights[right] *= 1.0 - alpha
        clusters.extend([left, right])
    weights = weights.reindex(labels).fillna(0.0)
    total = float(weights.sum())
    if total <= 0.0:
        return pd.Series(np.full(len(labels), 1.0 / len(labels)), index=labels, dtype=float)
    return weights / total
