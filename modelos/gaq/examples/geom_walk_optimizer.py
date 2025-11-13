"""Streamlit app for geometric hypercube optimization walks."""
from __future__ import annotations

from functools import lru_cache
from itertools import product
from typing import Callable, Sequence

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from clifford.g3 import e12, e23

from gaq.analysis.metrics import (
    autocorrelation as compute_autocorrelation,
    hamming_diffusion as compute_diffusion,
    probability_gap,
    spectral_gap_unbiased,
)

# Clifford exports vary by version; build e31 manual.
from clifford.g3 import e3, e1

e31 = e3 * e1

PALETTE = "Viridis"


# ---------------------------------------------------------------------------
# Hypercube + energy definitions
# ---------------------------------------------------------------------------


def generate_vertices(n: int) -> np.ndarray:
    """Return array of shape (2**n, n) with binary vertices."""
    verts = np.array(list(product([0, 1], repeat=n)), dtype=float)
    return verts


def generate_edges(verts: np.ndarray) -> list[tuple[int, int, int]]:
    """Return list of (i, j, dim) edges (dim = bit flipped)."""
    n = verts.shape[1]
    edges: list[tuple[int, int, int]] = []
    index_map = {"".join(map(str, map(int, v))): i for i, v in enumerate(verts)}
    for i, v in enumerate(verts):
        for dim in range(n):
            w = v.copy()
            w[dim] = 1 - w[dim]
            key = "".join(map(str, map(int, w)))
            j = index_map[key]
            if i < j:
                edges.append((i, j, dim))
    return edges


def cost_function(
    name: str,
    n: int,
    seed: int = 0,
    params: dict[str, float] | None = None,
) -> Callable[[np.ndarray], float]:
    params = params or {}
    rng = np.random.default_rng(seed)
    if name == "Quadrático (QUBO)":
        Q = rng.normal(scale=0.5, size=(n, n))
        Q = 0.5 * (Q + Q.T) + n * np.eye(n)
        bias = rng.normal(scale=0.2, size=n)

        def qubo(v: np.ndarray) -> float:
            return float(v @ Q @ v + bias @ v)

        return qubo
    if name == "Polinomial":
        coeff = rng.normal(size=3)

        def poly(v: np.ndarray) -> float:
            s = v.sum()
            return float(coeff[0] * s + coeff[1] * s**2 + coeff[2] * s**3)

        return poly
    if name == "Ising (J,h)":
        J = params.get("J", 1.0)
        h = params.get("h", 0.0)
        rows = int(np.ceil(np.sqrt(n)))
        cols = int(np.ceil(n / rows))

        def grid_neighbors(idx: int) -> list[int]:
            neighbors = []
            r, c = divmod(idx, cols)
            if c + 1 < cols and idx + 1 < n:
                neighbors.append(idx + 1)
            if c - 1 >= 0:
                neighbors.append(idx - 1)
            down = idx + cols
            up = idx - cols
            if down < n:
                neighbors.append(down)
            if up >= 0:
                neighbors.append(up)
            return neighbors

        neighbor_cache = [grid_neighbors(i) for i in range(n)]

        def ising_energy(v: np.ndarray) -> float:
            spins = 2 * v - 1  # map {0,1}→{-1,+1}
            energy = -h * np.sum(spins)
            for i in range(n):
                si = spins[i]
                for j in neighbor_cache[i]:
                    if j <= i:
                        continue
                    energy += -J * si * spins[j]
            return float(energy)

        return ising_energy

    def hamming(v: np.ndarray) -> float:
        return float(np.count_nonzero(v))

    return hamming


# ---------------------------------------------------------------------------
# Geometric walk dynamics
# ---------------------------------------------------------------------------

BIVECTORS = [e12, e23, e31]


def rotor_weights(angle: float, dim: int) -> tuple[float, float]:
    """Return (cos, sin)-like weights extracted from the Clifford rotor."""
    biv = BIVECTORS[dim % len(BIVECTORS)]
    rotor = (-angle * biv).exp()
    scalar = getattr(rotor, "value", None)
    if isinstance(scalar, dict):
        cos_val = float(scalar.get((), 0.0))
        # pick bivector coefficient if present, else sin fallback
        biv_key = next(iter({k for k in scalar.keys() if k}), None)
        if biv_key is not None:
            sin_val = float(scalar.get(biv_key, 0.0))
        else:
            sin_val = float(np.sin(angle))
    else:
        cos_val = float(np.cos(angle))
        sin_val = float(np.sin(angle))
    return cos_val, sin_val


def simulate_walk(
    energies: np.ndarray,
    edges: list[tuple[int, int, int]],
    steps: int,
    alpha: float,
    beta: float,
    noise: float,
    hitting_threshold: float,
) -> tuple[list[np.ndarray], list[float], int | None]:
    n_vertices = energies.shape[0]
    prob = np.ones(n_vertices, dtype=float) / n_vertices
    history = [prob.copy()]
    energy_mean = [float(prob @ energies)]
    target_idx = int(np.argmin(energies))
    # Adaptive threshold relative to uniform probability
    adaptive_thresh = max(hitting_threshold, 1.2 / n_vertices)
    hitting_step: int | None = None

    adjacency: dict[int, list[tuple[int, int]]] = {i: [] for i in range(n_vertices)}
    for i, j, dim in edges:
        adjacency[i].append((j, dim))
        adjacency[j].append((i, dim))

    for step in range(1, steps + 1):
        new_prob = prob.copy()
        for i in range(n_vertices):
            degree = max(len(adjacency[i]), 1)
            for j, dim in adjacency[i]:
                angle = alpha * (energies[i] - energies[j])
                cos_w, sin_w = rotor_weights(angle, dim)
                symmetric_flow = cos_w * (prob[i] - prob[j]) / degree
                orient_flow = sin_w * 0.5 * (prob[i] + prob[j])
                bias_flow = beta * (energies[j] - energies[i]) * prob[i]
                flow = symmetric_flow + orient_flow - bias_flow
                new_prob[i] -= flow
                new_prob[j] += flow
        if noise > 0:
            new_prob = (1 - noise) * new_prob + noise / n_vertices
        if beta > 0:
            boltz = np.exp(-beta * energies)
            new_prob *= boltz
        new_prob = np.clip(new_prob, 0, None)
        new_prob /= new_prob.sum()
        prob = new_prob
        history.append(prob.copy())
        energy_mean.append(float(prob @ energies))
        if hitting_step is None and prob[target_idx] >= adaptive_thresh:
            hitting_step = step
    return history, energy_mean, hitting_step


# ---------------------------------------------------------------------------
# Projections + plotting
# ---------------------------------------------------------------------------


def project_vertices(verts: np.ndarray) -> np.ndarray:
    rows, dims = verts.shape
    if dims >= 3:
        centered = verts - verts.mean(axis=0)
        u, s, vh = np.linalg.svd(centered, full_matrices=False)
        coords = centered @ vh[:3].T
    else:
        coords = np.zeros((rows, 3))
        coords[:, :dims] = verts
    return coords


def plot_hypercube(
    coords: np.ndarray,
    edges: list[tuple[int, int, int]],
    probs: np.ndarray,
    energies: np.ndarray,
    step: int,
    min_mask: np.ndarray,
) -> go.Figure:
    edge_traces = []
    for i, j, _ in edges:
        edge_traces.append(
            go.Scatter3d(
                x=[coords[i, 0], coords[j, 0], None],
                y=[coords[i, 1], coords[j, 1], None],
                z=[coords[i, 2], coords[j, 2], None],
                mode="lines",
                line=dict(color="rgba(150,150,150,0.2)", width=1),
                hoverinfo="skip",
            )
        )
    marker_sizes = 6 + 30 * probs / probs.max()
    node_trace = go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode="markers",
        marker=dict(
            size=marker_sizes,
            color=np.where(min_mask, energies.min() - 0.05, energies),
            colorscale=PALETTE,
            colorbar=dict(title="Energia"),
            line=dict(color="black", width=0.5),
        ),
        text=[f"v={i}<br>P={probs[i]:.3f}<br>E={energies[i]:.3f}" for i in range(len(probs))],
        hoverinfo="text",
    )
    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title=f"Walk geométrico – passo {step}",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def plot_energy_curve(energy_mean: Sequence[float]) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(y=energy_mean, mode="lines+markers", line=dict(color="orangered"))
    fig.update_layout(xaxis_title="Passo", yaxis_title="Energia média", title="Energia vs passo")
    return fig


def plot_series(series: Sequence[float], title: str, ylabel: str) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(y=series, mode="lines", line=dict(color="tab:green"))
    fig.update_layout(xaxis_title="Passo", yaxis_title=ylabel, title=title)
    return fig


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------


def sidebar_controls() -> dict:
    st.sidebar.header("Configuração do walk")
    n = st.sidebar.slider("Bits (dimensão do hipercubo)", 2, 8, 4)
    steps = st.sidebar.slider("Passos", 5, 150, 50)
    alpha = st.sidebar.slider("Força geométrica (α)", 0.1, 3.0, 1.0, 0.1)
    beta = st.sidebar.slider("Viés energético / β", 0.0, 5.0, 1.0, 0.1)
    noise = st.sidebar.slider("Ruído térmico", 0.0, 0.5, 0.05, 0.01)
    threshold = st.sidebar.slider("Threshold hitting", 0.05, 0.5, 0.2, 0.01)
    cost_name = st.sidebar.selectbox(
        "Função de energia",
        ["Quadrático (QUBO)", "Polinomial", "Hamming", "Ising (J,h)"],
        index=0,
    )
    cost_params: dict[str, float] = {}
    if cost_name == "Ising (J,h)":
        cost_params["J"] = st.sidebar.slider("J (acoplamento)", -3.0, 3.0, 1.0, 0.1)
        cost_params["h"] = st.sidebar.slider("Campo externo h", -3.0, 3.0, 0.0, 0.1)
    return dict(
        n=n,
        steps=steps,
        alpha=alpha,
        beta=beta,
        noise=noise,
        threshold=threshold,
        cost_name=cost_name,
        cost_params=cost_params,
    )


def main() -> None:
    st.set_page_config(page_title="Walk Geométrico no Hipercubo", layout="wide")
    st.title("Simulador físico/visual de otimização geométrica")
    st.write(
        "Hipercubo discretizado, energia atribuída por função de custo e caminhada geométrica com operadores Clifford."
    )

    cfg = sidebar_controls()
    verts = generate_vertices(cfg["n"])
    edges = generate_edges(verts)
    project = project_vertices(verts)
    cost_fn = cost_function(cfg["cost_name"], cfg["n"], seed=42, params=cfg.get("cost_params"))
    energies = np.array([cost_fn(v) for v in verts])
    energies = (energies - energies.min()) / (np.ptp(energies) + 1e-9)

    history, energy_curve, hitting = simulate_walk(
        energies,
        edges,
        steps=cfg["steps"],
        alpha=cfg["alpha"],
        beta=cfg["beta"],
        noise=cfg["noise"],
        hitting_threshold=cfg["threshold"],
    )
    diffusion_curve = compute_diffusion(history)
    prob_gap_curve = probability_gap(history)
    autocorr_curve = compute_autocorrelation(
        np.array(energy_curve), max_lag=min(20, len(energy_curve) - 1)
    )
    _, spec_gap = spectral_gap_unbiased(cfg["n"])

    st.subheader("Evolução 3D")
    slider = st.slider("Escolha o passo para visualizar", 0, cfg["steps"], 0)
    min_mask = energies <= energies.min() + 1e-9
    min_mask = energies <= energies.min() + 1e-9
    st.plotly_chart(
        plot_hypercube(project, edges, history[slider], energies, slider, min_mask),
        use_container_width=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_energy_curve(energy_curve), use_container_width=True)
        st.metric("Energia inicial", f"{energy_curve[0]:.3f}")
        st.metric("Energia final", f"{energy_curve[-1]:.3f}")
    with col2:
        best_idx = np.argmin(energies)
        prob_min = history[-1][min_mask].sum()
        st.metric("Prob. total nos mínimos", f"{prob_min:.3f}")
        if hitting is None:
            st.warning("Hitting threshold não atingido.")
        else:
            st.success(f"Hitting threshold alcançado no passo {hitting}")

    st.subheader("Métricas de calibração (gap, difusão, autocorrelação)")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Gap espectral Q_n", f"{spec_gap:.3f}")
        st.plotly_chart(
            plot_series(diffusion_curve, "Difusão de Hamming", "ΔH média"),
            use_container_width=True,
        )
    with col_b:
        st.metric("Gap prob. final", f"{prob_gap_curve[-1]:.3f}")
        st.plotly_chart(
            plot_series(autocorr_curve, "Autocorrelação da energia", "ρ(lag)"),
            use_container_width=True,
        )

    st.info(
        "Fluxo agora inclui termo anti-simétrico ~ β(Ej−Ei) e reweighting Boltzmann (β slider). "
        "A componente senoidal do rotor inverte o sentido quando o vizinho tem energia menor, "
        "favorecendo concentração nas bases com menor custo."
    )


if __name__ == "__main__":
    main()
