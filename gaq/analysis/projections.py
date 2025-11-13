"""Dimensionality-reduction utilities for GAQ walks (PCA, t-SNE, hyperbolic)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


@dataclass
class EmbeddingResult:
    method: str
    coordinates: np.ndarray  # shape (n_samples, 2)
    metadata: dict[str, float | np.ndarray]


def _distribution_to_samples(bits: np.ndarray, probs: np.ndarray, max_points: int = 2000) -> np.ndarray:
    probs = np.clip(probs, 0.0, None)
    probs = probs / probs.sum()
    counts = np.maximum(np.round(probs * max_points), 1).astype(int)
    return np.repeat(bits, counts, axis=0)


def pca_embedding(bits: np.ndarray, probs: np.ndarray) -> EmbeddingResult:
    samples = _distribution_to_samples(bits, probs)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(samples)
    return EmbeddingResult(
        method="PCA",
        coordinates=coords,
        metadata={"explained_variance": pca.explained_variance_ratio_},
    )


def tsne_embedding(bits: np.ndarray, probs: np.ndarray, seed: int = 0) -> EmbeddingResult:
    samples = _distribution_to_samples(bits, probs)
    tsne = TSNE(n_components=2, init="pca", random_state=seed, perplexity=40)
    coords = tsne.fit_transform(samples)
    return EmbeddingResult(method="t-SNE", coordinates=coords, metadata={})


def hypercube_hyperbolic_embedding(bits: np.ndarray) -> EmbeddingResult:
    """Project vertices to Poincaré disk preserving adjacency order."""
    n = bits.shape[1]
    total = bits.shape[0]
    angles = np.linspace(0, 2 * np.pi, total, endpoint=False)
    radius = 1 - 1 / (n + 1)
    coords = np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
    return EmbeddingResult(method="Hyperbolic", coordinates=coords, metadata={"radius": radius})
