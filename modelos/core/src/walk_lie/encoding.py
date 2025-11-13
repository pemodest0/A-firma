#!/usr/bin/env python3
"""
Conversões entre estados contínuos (features) e vértices do hipercubo.
Usa quantização por bins e Gray code para minimizar distância Hamming
quando features variam suavemente.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


def gray_encode(n: int) -> int:
    return n ^ (n >> 1)


def gray_decode(g: int) -> int:
    result = g
    while g > 0:
        g >>= 1
        result ^= g
    return result


@dataclass
class FeatureBin:
    name: str
    bins: Sequence[float]
    lower_bound: float | None = None
    upper_bound: float | None = None

    def digit_count(self) -> int:
        """Número de bits necessários para representar os bins."""
        levels = len(self.bins) + 1  # bins definem fronteiras
        return int(np.ceil(np.log2(max(2, levels))))

    def quantize(self, value: float) -> int:
        limits = [-np.inf] + list(self.bins) + [np.inf]
        return int(np.digitize(value, limits) - 1)

    def dequantize(self, index: int) -> float:
        limits = [-np.inf] + list(self.bins) + [np.inf]
        lower = limits[max(0, index)]
        upper = limits[min(len(limits) - 1, index + 1)]
        if not np.isfinite(lower):
            lower = self.lower_bound if self.lower_bound is not None else limits[index + 1] - 1.0
        if not np.isfinite(upper):
            upper = self.upper_bound if self.upper_bound is not None else limits[index] + 1.0
        return float((lower + upper) / 2.0)


class HypercubeEncoder:
    def __init__(self, feature_bins: Sequence[FeatureBin]) -> None:
        self.feature_bins = list(feature_bins)
        self.offsets: List[int] = []
        running = 0
        for feature in self.feature_bins:
            self.offsets.append(running)
            running += feature.digit_count()
        self.total_bits = running

    def encode(self, values: Dict[str, float]) -> int:
        if set(values.keys()) != {fb.name for fb in self.feature_bins}:
            raise ValueError("Valores fornecidos não correspondem aos features configurados.")
        vertex = 0
        for feature, offset in zip(self.feature_bins, self.offsets):
            raw_index = feature.quantize(values[feature.name])
            gray_index = gray_encode(raw_index)
            vertex |= gray_index << offset
        return vertex

    def decode(self, vertex: int) -> Dict[str, float]:
        values = {}
        for feature, offset in zip(self.feature_bins, self.offsets):
            mask = (1 << feature.digit_count()) - 1
            gray_index = (vertex >> offset) & mask
            raw_index = gray_decode(gray_index)
            values[feature.name] = feature.dequantize(raw_index)
        return values

    def vertex_count(self) -> int:
        return 1 << self.total_bits
