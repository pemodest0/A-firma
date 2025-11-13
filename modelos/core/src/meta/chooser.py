from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

import numpy as np


@dataclass
class WalkChooser:
    feature_keys: List[str] = field(default_factory=list)
    centroids: Dict[str, np.ndarray] = field(default_factory=dict)

    def fit(
        self,
        feature_dicts: Sequence[Dict[str, float]],
        labels: Sequence[str],
    ) -> None:
        if len(feature_dicts) != len(labels):
            raise ValueError("feature_dicts and labels must have the same length.")
        keys: List[str] = sorted({key for feats in feature_dicts for key in feats.keys()})
        if not keys:
            raise ValueError("No features provided to WalkChooser.")
        self.feature_keys = keys
        accum: Dict[str, List[np.ndarray]] = {}
        for feats, label in zip(feature_dicts, labels):
            vector = self._dict_to_vector(feats)
            accum.setdefault(label, []).append(vector)
        self.centroids = {
            label: np.mean(np.stack(vectors, axis=0), axis=0) for label, vectors in accum.items()
        }

    def _dict_to_vector(self, feats: Dict[str, float]) -> np.ndarray:
        if not self.feature_keys:
            raise RuntimeError("WalkChooser has not been fitted yet.")
        return np.array([float(feats.get(key, 0.0)) for key in self.feature_keys], dtype=float)

    def predict(self, feature_dicts: Iterable[Dict[str, float]]) -> List[str]:
        if not self.centroids:
            raise RuntimeError("WalkChooser has not been fitted yet.")
        predictions: List[str] = []
        centroid_labels = list(self.centroids.keys())
        centroid_vectors = np.stack([self.centroids[label] for label in centroid_labels], axis=0)
        for feats in feature_dicts:
            vector = self._dict_to_vector(feats)
            distances = np.linalg.norm(centroid_vectors - vector[None, :], axis=1)
            best_idx = int(np.argmin(distances))
            predictions.append(centroid_labels[best_idx])
        return predictions

    def predict_one(self, features: Dict[str, float]) -> str:
        return self.predict([features])[0]
