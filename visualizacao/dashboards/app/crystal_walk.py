#!/usr/bin/env python3
"""
Streamlit app: caminhada quântica em uma rede cristalina 2D com defeitos.

Mostra como a amplitude se propaga num material perfeito e como impurezas
alteram a propagação (espalhamento/localização). O modelo é tight-binding
com hopping t entre vizinhos e um potencial alto em sítios defeituosos.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.colors import ListedColormap
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import expm_multiply

ħ = 1.0  # usamos unidades naturais


@dataclass
class LatticeConfig:
    size: int
    hopping: float
    defect_mask: np.ndarray  # shape (size, size), True = defeito
    defect_potential: float


def build_hamiltonian(cfg: LatticeConfig) -> Tuple[lil_matrix, Dict[str, np.ndarray]]:
    n = cfg.size
    N = n * n
    H = lil_matrix((N, N), dtype=np.complex128)
    idx_map = np.arange(N).reshape(n, n)

    for i in range(n):
        for j in range(n):
            idx = idx_map[i, j]
            if cfg.defect_mask[i, j]:
                H[idx, idx] = cfg.defect_potential
                continue
            H[idx, idx] = 0.0
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < n and not cfg.defect_mask[ni, nj]:
                    nidx = idx_map[ni, nj]
                    H[idx, nidx] = -cfg.hopping
    metadata = {
        "idx_map": idx_map,
        "positions": idx_map,
    }
    return H.tocsr(), metadata


@st.cache_resource(show_spinner=False)
def compute_time_series(
    cfg: LatticeConfig,
    psi0: np.ndarray,
    total_time: float,
    frames: int,
) -> Dict[str, np.ndarray]:
    H, meta = build_hamiltonian(cfg)
    op = -1j * H / ħ
    times = np.linspace(0.0, total_time, frames)
    psi_t = expm_multiply(op, psi0, start=0.0, stop=total_time, num=frames, endpoint=True)
    probs = np.abs(psi_t) ** 2
    probs = probs / probs.sum(axis=1, keepdims=True)
    return {
        "times": times,
        "psi_t": psi_t,
        "probs": probs.reshape(frames, cfg.size, cfg.size),
        "idx_map": meta["idx_map"],
    }


def gaussian_wavepacket(cfg: LatticeConfig, center: Tuple[float, float], width: float, momentum: Tuple[float, float]) -> np.ndarray:
    n = cfg.size
    xs = np.arange(n)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    dx = X - center[0]
    dy = Y - center[1]
    envelope = np.exp(-(dx**2 + dy**2) / (2 * width**2))
    phase = np.exp(1j * (momentum[0] * X + momentum[1] * Y))
    psi = envelope * phase
    psi[cfg.defect_mask] = 0.0
    psi = psi.ravel()
    psi /= np.linalg.norm(psi)
    return psi


def random_defects(n: int, fraction: float, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = rng.random((n, n)) < fraction
    return mask


def plot_state(probs: np.ndarray, defects: np.ndarray, timestamp: float) -> plt.Figure:
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(probs, origin="lower", cmap=cmap, vmin=0, vmax=probs.max())
    ax.imshow(defects, origin="lower", cmap=ListedColormap([[1, 0, 0, 0.3]]))
    ax.set_title(f"|ψ|² no tempo t = {timestamp:.2f}")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


def compute_boundary_hitting(probs: np.ndarray, times: np.ndarray, threshold: float = 0.1) -> Tuple[float | None, np.ndarray]:
    boundary_mask = np.zeros_like(probs[0], dtype=bool)
    boundary_mask[0, :] = boundary_mask[-1, :] = True
    boundary_mask[:, 0] = boundary_mask[:, -1] = True
    cumulative = (probs * boundary_mask).sum(axis=(1, 2))
    hits = np.where(cumulative >= threshold)[0]
    return (times[hits[0]] if hits.size else None), cumulative


def main() -> None:
    st.set_page_config(page_title="Walk em Cristal 2D", layout="wide")
    st.title("Caminhada Quântica em um Cristal 2D")
    st.markdown(
        "Brinque com um cristal perfeito versus cristal com defeitos. "
        "Veja como o pacote de onda se espalha (ou fica preso) quando há impurezas."
    )

    with st.sidebar:
        st.header("Configuração do cristal")
        size = st.slider("Tamanho da rede (NxN)", min_value=8, max_value=20, value=14, step=2)
        hopping = st.slider("Hopping t", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
        defect_fraction = st.slider("Fraçao de defeitos", min_value=0.0, max_value=0.3, value=0.05, step=0.01)
        defect_potential = st.slider("Potencial dos defeitos", min_value=5.0, max_value=50.0, value=20.0, step=5.0)
        seed = st.number_input("Semente (random)", value=42, step=1)

        st.header("Estado inicial")
        center_x = st.slider("Centro (x)", 0.0, float(size - 1), float(size // 2 - 1), 0.5)
        center_y = st.slider("Centro (y)", 0.0, float(size - 1), float(size // 2), 0.5)
        width = st.slider("Largura do pacote", 0.5, 4.0, 1.5, 0.1)
        momentum_x = st.slider("Momento inicial px", -np.pi, np.pi, 0.0, 0.1)
        momentum_y = st.slider("Momento inicial py", -np.pi, np.pi, 1.5, 0.1)

        st.header("Tempo de evolução")
        total_time = st.slider("Tempo total", 1.0, 10.0, 5.0, 0.5)
        frames = st.slider("Quadros (frames)", 30, 150, 60, 10)

        if st.button("Simular", type="primary"):
            st.session_state["crystal_run"] = True

    if not st.session_state.get("crystal_run", False):
        st.info("Ajuste os parâmetros e clique em **Simular**.")
        return

    with st.spinner("Calculando evolução quântica..."):
        defect_mask = random_defects(size, defect_fraction, seed=int(seed))
        cfg = LatticeConfig(size=size, hopping=hopping, defect_mask=defect_mask, defect_potential=defect_potential)
        psi0 = gaussian_wavepacket(cfg, center=(center_x, center_y), width=width, momentum=(momentum_x, momentum_y))
        data = compute_time_series(cfg, psi0, total_time=total_time, frames=frames)

    probs = data["probs"]
    times = data["times"]

    frame_idx = st.slider("Escolha o instante", min_value=0, max_value=frames - 1, value=0)

    col1, col2 = st.columns((1.4, 1))
    with col1:
        fig = plot_state(probs[frame_idx], defect_mask, times[frame_idx])
        st.pyplot(fig, clear_figure=True)

    with col2:
        hit_time, cumulative = compute_boundary_hitting(probs, times)
        st.metric("Probabilidade total", f"{probs[frame_idx].sum():.3f}")
        st.metric("Prob. na fronteira", f"{cumulative[frame_idx]:.3f}")
        if hit_time is not None:
            st.success(f"Hit > 10% na borda em t ≈ {hit_time:.2f}")
        else:
            st.warning("Nunca passou de 10% na borda — a onda ficou presa ou lenta.")
        st.line_chart({"tempo": times, "probabilidade na borda": cumulative})

    st.markdown(
        """
### O que está rolando?
- **Rede perfeita (defeitos = 0)** → a onda se espalha rápido como Bloch waves.
- **Muitos defeitos** → a amplitude reflete e pode ficar presa (localização).
- **Hitting time** aqui é o tempo até que uma parte significativa da onda chegue na borda do cristal.
        """
    )


if __name__ == "__main__":
    main()
