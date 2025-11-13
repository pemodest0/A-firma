#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explorador interativo da caminhada em hipercubo (Streamlit).

- Caminhada clássica (difusão) e quântica (moeda de Grover) com animação.
- Exibe distribuição de probabilidades por nó e destaca um "walker" amostrado.
- Limita a dimensão a d ≤ 5 (32 nós) para manter interação fluida.

Requisitos: pip install streamlit numpy plotly networkx scikit-learn
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA

# ---------------------------------------------------------------
# Utilidades do hipercubo
# ---------------------------------------------------------------

def int_to_bits(n: int, d: int) -> np.ndarray:
    return np.array([(n >> k) & 1 for k in range(d)], dtype=np.float64)


def neighbors(node: int, d: int) -> Iterable[int]:
    for k in range(d):
        yield node ^ (1 << k)


def all_nodes(d: int) -> np.ndarray:
    return np.arange(2**d, dtype=int)


@lru_cache(maxsize=16)
def hypercube_positions_2d(d: int, seed: int = 0) -> Dict[int, np.ndarray]:
    points = np.stack([int_to_bits(n, d) for n in all_nodes(d)], axis=0)
    pca = PCA(n_components=2, random_state=seed)
    xy = pca.fit_transform(points)
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0) + 1e-12)
    return {n: xy[i] for i, n in enumerate(all_nodes(d))}


@lru_cache(maxsize=16)
def hypercube_positions_3d(d: int, seed: int = 0) -> Dict[int, np.ndarray]:
    points = np.stack([int_to_bits(n, d) for n in all_nodes(d)], axis=0)
    pca = PCA(n_components=3, random_state=seed)
    xyz = pca.fit_transform(points)
    xyz = (xyz - xyz.min(0)) / (xyz.max(0) - xyz.min(0) + 1e-12)
    return {n: xyz[i] for i, n in enumerate(all_nodes(d))}


def _tuple_bits_to_int(bits: Tuple[int, ...]) -> int:
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


# ---------------------------------------------------------------
# Caminhada clássica e quântica
# ---------------------------------------------------------------

def classical_step(probs: np.ndarray, d: int) -> np.ndarray:
    out = np.zeros_like(probs)
    for node in range(probs.shape[0]):
        acc = 0.0
        for nb in neighbors(node, d):
            acc += probs[nb]
        out[node] = acc / d
    return out


def grover_coin(d: int) -> np.ndarray:
    ones = np.ones((d, d), dtype=np.complex128)
    return (2.0 / d) * ones - np.eye(d, dtype=np.complex128)


def quantum_step(psi: np.ndarray, d: int, coin: np.ndarray) -> np.ndarray:
    psi_coin = psi @ coin.T
    out = np.zeros_like(psi, dtype=np.complex128)
    for node in range(psi.shape[0]):
        for k in range(d):
            nb = node ^ (1 << k)
            out[nb, k] += psi_coin[node, k]
    return out


def quantum_prob(psi: np.ndarray) -> np.ndarray:
    return (np.abs(psi) ** 2).sum(axis=1).real


def init_quantum_state(d: int, start: int, bias: float = 0.0) -> np.ndarray:
    state = np.zeros((2**d, d), dtype=np.complex128)
    base = np.ones(d, dtype=np.complex128) / math.sqrt(d)
    if abs(bias) > 0:
        weights = np.linspace(1.0, 0.0, d)
        vec = (1.0 + bias * (weights - weights.mean()))
        vec = vec / np.linalg.norm(vec)
        base = vec.astype(np.complex128)
    state[start, :] = base
    return state


def simulate_classical(d: int, steps: int, start: int) -> np.ndarray:
    probs = np.zeros(2**d, dtype=np.float64)
    probs[start] = 1.0
    frames = [probs.copy()]
    for _ in range(steps):
        probs = classical_step(probs, d)
        frames.append(probs.copy())
    return np.stack(frames, axis=0)


def simulate_quantum(d: int, steps: int, start: int, bias: float = 0.0) -> np.ndarray:
    coin = grover_coin(d)
    psi = init_quantum_state(d, start, bias=bias)
    frames = [quantum_prob(psi)]
    for _ in range(steps):
        psi = quantum_step(psi, d, coin)
        frames.append(quantum_prob(psi))
    return np.stack(frames, axis=0)


def sample_path(frames: np.ndarray, start: int, rng: np.random.Generator) -> List[int]:
    path = [int(start)]
    for step in range(1, frames.shape[0]):
        probs = frames[step]
        total = probs.sum()
        if total <= 0 or not np.isfinite(total):
            node = path[-1]
        else:
            distribution = probs / total
            node = int(rng.choice(len(distribution), p=distribution))
        path.append(node)
    return path


# ---------------------------------------------------------------
# Plotly helpers
# ---------------------------------------------------------------

MODE_META = {
    "classical": {"name": "Clássica", "color": "#1f77b4"},
    "quantum": {"name": "Quântica (Grover)", "color": "#d62728"},
}


@dataclass
class GraphInfo:
    name: str
    description: str
    adjacency: nx.Graph


def build_hypercube_graph(d: int) -> nx.Graph:
    G = nx.hypercube_graph(d)
    G.graph["type"] = "Hypercube"
    G.graph["description"] = f"Grafo hipercubo Q_{d} com {2**d} vértices, arestas entre vértices com Hamming=1"
    G.graph["connections"] = "Cada aresta inverte um bit (Hamming distance 1)"
    return G


def build_mesh_graph(d: int) -> nx.Graph:
    dims = (2,) * d
    G = nx.grid_graph(dims)
    mapping = {node: idx for idx, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    G.graph["type"] = "Grid"
    G.graph["description"] = f"Malha cartesiana {dims}, conexões ortogonais"
    G.graph["connections"] = "Arestas entre vizinhos ortogonais em grade cartesiana"
    return G


GRAPH_FACTORIES = {
    "Hypercubo": build_hypercube_graph,
    "Malha cartesiana": build_mesh_graph,
}


@dataclass
class Scenario:
    name: str
    description: str
    features: List[Tuple[str, str]]
    signals: List[str]
    notes: str
    recommended_dim: int


SCENARIOS: Dict[str, Scenario] = {
    "Finanças (regimes de mercado)": Scenario(
        name="Finanças",
        description=(
            "Bits representam regimes binarizados: momentum↑/↓, volatilidade alta/baixa, drawdown em risco/ok. "
            "O walker modela a difusão entre estados de mercado; a moeda quântica captura interferência entre sinais."
        ),
        features=[
            ("Momentum", "positivo vs. negativo"),
            ("Volatilidade", "> limiar vs. < limiar"),
            ("Drawdown", "em risco vs. confortável"),
        ],
        signals=[
            "coin_risk_flag ativa coin mais difusiva quando vol_ratio é alto",
            "Λ elevado indica conflito entre momentum e volatilidade",
        ],
        notes="Bias > 0 favorece rally; hitting time até estado neutro indica duração dos regimes.",
        recommended_dim=3,
    ),
    "Cadeia logística": Scenario(
        name="Logística",
        description=(
            "Bits codificam estoque, transporte e demanda (ok vs. alerta). O walk mede como gargalos propagam entre nós."
        ),
        features=[
            ("Estoque", ">= alvo vs. < alvo"),
            ("Transporte", "lead time normal vs. atraso"),
            ("Demanda", "alta vs. baixa"),
        ],
        signals=[
            "Coin adaptada reduz amplitude quando transporte atrasado",
            "Λ monitora conflito estoque × demanda",
        ],
        notes="Use hitting time até estado 'todos ok' como KPI; coin bias pode priorizar correções de estoque.",
        recommended_dim=3,
    ),
    "Epidemiologia": Scenario(
        name="Epidemiologia",
        description=(
            "Bits para prevalência, mobilidade, vacinação e hospitalização (4 dimensões). "
            "Difusão captura alternância de medidas não farmacológicas."
        ),
        features=[
            ("Prevalência", "> R_limite vs. <= R_limite"),
            ("Mobilidade", "restrita vs. normal"),
            ("Vacinação", "alta vs. baixa"),
            ("Hospitalização", "alerta vs. normal"),
        ],
        signals=[
            "Coin difusiva em lockdown (volta para estados seguros)",
            "Λ alto indica conflito mobilidade × prevalência",
        ],
        notes="Meta: tempo até estado seguro (todos bits 0). Use viés para simular campanhas de vacinação.",
        recommended_dim=4,
    ),
}


def build_traces(
    positions: Dict[int, np.ndarray],
    probs: np.ndarray,
    path: List[int],
    mode_key: str,
    global_max: float,
    show_colorbar: bool,
    subplot: int,
    use_3d: bool,
) -> Tuple[go.Scatter, go.Scatter]:
    nodes = len(probs)
    coords = np.array([positions[i] for i in range(nodes)])
    with np.errstate(divide="ignore", invalid="ignore"):
        sizes = 18 + 32 * np.sqrt(np.clip(probs / (global_max + 1e-12), 0.0, 1.0))
    if use_3d:
        base = go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="markers",
            marker=dict(
                size=sizes,
                color=probs,
                colorscale="Viridis",
                cmin=0,
                cmax=global_max,
                showscale=show_colorbar,
                colorbar=dict(title="P", x=1.02) if show_colorbar else None,
            ),
            text=[f"n={i}<br>P={probs[i]:.4f}" for i in range(nodes)],
            hoverinfo="text",
            name=f"Prob. {MODE_META[mode_key]['name']}",
            legendgroup=mode_key,
            showlegend=False,
        )
        walker_node = path[0]
        walker = go.Scatter3d(
            x=[coords[walker_node, 0]],
            y=[coords[walker_node, 1]],
            z=[coords[walker_node, 2]],
            mode="markers",
            marker=dict(size=9, color=MODE_META[mode_key]["color"], symbol="x"),
            name=f"Walker {MODE_META[mode_key]['name']}",
            legendgroup=mode_key,
            showlegend=True,
        )
    else:
        base = go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="markers",
            marker=dict(
                size=sizes,
                color=probs,
                colorscale="Viridis",
                cmin=0,
                cmax=global_max,
                showscale=show_colorbar,
                colorbar=dict(title="P", x=1.03) if show_colorbar else None,
            ),
            text=[f"n={i}<br>P={probs[i]:.4f}" for i in range(nodes)],
            hoverinfo="text",
            name=f"Prob. {MODE_META[mode_key]['name']}",
            legendgroup=mode_key,
            showlegend=False,
        )
        walker_node = path[0]
        walker = go.Scatter(
            x=[coords[walker_node, 0]],
            y=[coords[walker_node, 1]],
            mode="markers",
            marker=dict(size=26, color=MODE_META[mode_key]["color"], symbol="x"),
            name=f"Walker {MODE_META[mode_key]['name']}",
            legendgroup=mode_key,
            showlegend=True,
        )
    return base, walker


def build_frames(
    positions: Dict[int, np.ndarray],
    frames_dict: Dict[str, np.ndarray],
    paths: Dict[str, List[int]],
    modes: List[str],
    global_max: Dict[str, float],
    use_3d: bool,
) -> Tuple[List[go.Frame], List[go.Scatter]]:
    num_steps = next(iter(frames_dict.values())).shape[0]
    subplot_traces = []
    frames = []
    for col, mode_key in enumerate(modes, start=1):
        base, walker = build_traces(
            positions,
            frames_dict[mode_key][0],
            paths[mode_key],
            mode_key,
            global_max[mode_key],
            show_colorbar=(col == len(modes)),
            subplot=col,
            use_3d=use_3d,
        )
        if not use_3d:
            base.update(xaxis=f"x{col if len(modes) > 1 else ''}", yaxis=f"y{col if len(modes) > 1 else ''}")
            walker.update(xaxis=f"x{col if len(modes) > 1 else ''}", yaxis=f"y{col if len(modes) > 1 else ''}")
        else:
            base.update(scene=f"scene{col if len(modes) > 1 else ''}")
            walker.update(scene=f"scene{col if len(modes) > 1 else ''}")
        subplot_traces.append((base, walker))
    for step in range(num_steps):
        data = []
        for col, mode_key in enumerate(modes, start=1):
            probs = frames_dict[mode_key][step]
            coords = np.array([positions[i] for i in range(len(probs))])
            size_vals = 18 + 32 * np.sqrt(np.clip(probs / (global_max[mode_key] + 1e-12), 0.0, 1.0))
            if use_3d:
                base = go.Scatter3d(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    z=coords[:, 2],
                    mode="markers",
                    marker=dict(size=size_vals, color=probs, colorscale="Viridis", cmin=0, cmax=global_max[mode_key], showscale=False),
                    text=[f"n={i}<br>P={probs[i]:.4f}" for i in range(len(probs))],
                    hoverinfo="text",
                    legendgroup=mode_key,
                    showlegend=False,
                )
                base.update(scene=f"scene{col if len(modes) > 1 else ''}")
                walker_node = paths[mode_key][step]
                walker = go.Scatter3d(
                    x=[coords[walker_node, 0]],
                    y=[coords[walker_node, 1]],
                    z=[coords[walker_node, 2]],
                    mode="markers",
                    marker=dict(size=9, color=MODE_META[mode_key]["color"], symbol="x"),
                    legendgroup=mode_key,
                    showlegend=False,
                )
                walker.update(scene=f"scene{col if len(modes) > 1 else ''}")
            else:
                base = go.Scatter(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    mode="markers",
                    marker=dict(size=size_vals, color=probs, colorscale="Viridis", cmin=0, cmax=global_max[mode_key], showscale=False),
                    text=[f"n={i}<br>P={probs[i]:.4f}" for i in range(len(probs))],
                    hoverinfo="text",
                    legendgroup=mode_key,
                    showlegend=False,
                )
                base.update(xaxis=f"x{col if len(modes) > 1 else ''}", yaxis=f"y{col if len(modes) > 1 else ''}")
                walker_node = paths[mode_key][step]
                walker = go.Scatter(
                    x=[coords[walker_node, 0]],
                    y=[coords[walker_node, 1]],
                    mode="markers",
                    marker=dict(size=26, color=MODE_META[mode_key]["color"], symbol="x"),
                    legendgroup=mode_key,
                    showlegend=False,
                )
                walker.update(xaxis=f"x{col if len(modes) > 1 else ''}", yaxis=f"y{col if len(modes) > 1 else ''}")
            data.extend([base, walker])
        frames.append(go.Frame(data=data, name=str(step)))
    initial_data = [trace for pair in subplot_traces for trace in pair]
    return frames, initial_data


# ---------------------------------------------------------------
# Interface Streamlit
# ---------------------------------------------------------------
st.set_page_config(page_title="Hypercube Walk Explorer", layout="wide")
st.title("🔭 Walker no Hipercubo")
st.markdown(
    "Compare a difusão clássica com a caminhada quântica (moeda de Grover). "
    "Os círculos maiores indicam maior probabilidade de encontrar o walker; o 'X' marca um caminho amostrado."\
)

scenario_name = st.selectbox("Cenário", list(SCENARIOS.keys()), index=0)
scenario = SCENARIOS[scenario_name]
st.markdown(f"**Descrição do cenário:** {scenario.description}")
signals_md = "<br>".join(f"• {text}" for text in scenario.signals)
st.markdown("**Sinais em destaque:**<br>" + signals_md, unsafe_allow_html=True)
feature_df = pd.DataFrame(scenario.features, columns=["Feature", "Significado"])
st.table(feature_df)

col_a, col_b, col_c = st.columns(3)
with col_a:
    graph_type = st.selectbox("Tipo de grafo", list(GRAPH_FACTORIES.keys()), index=0)
with col_b:
    default_dim = min(max(2, scenario.recommended_dim), 5)
    dim = st.number_input("Dimensão (d)", min_value=2, max_value=5, value=default_dim, step=1)
with col_c:
    max_steps = st.slider("Passos", min_value=1, max_value=60, value=25)

use_3d = st.checkbox("Visualização 3D", value=True)

start_node = st.number_input("Nó inicial", min_value=0, max_value=(2**dim) - 1, value=0, step=1)

mode_choice = st.selectbox("Modo", ["Clássica", "Quântica", "Ambas"], index=2)
bias = 0.0
if mode_choice != "Clássica":
    bias = st.slider("Viés da moeda quântica", min_value=-0.9, max_value=0.9, value=0.0, step=0.05)
seed = st.number_input("Seed do walker", min_value=0, max_value=10_000, value=42, step=1)

graph = GRAPH_FACTORIES[graph_type](dim)
st.markdown(f"**Grafo selecionado:** {graph.graph['description']}")
st.markdown(f"**Tipo de conexão:** {graph.graph.get('connections', 'N/A')}")
sample_edges = list(graph.edges())[: min(10, graph.number_of_edges())]
st.table(pd.DataFrame(sample_edges, columns=["v1", "v2"]))
if graph_type != "Hypercubo":
    st.warning("Simulação atual utiliza hipercubo; tipos adicionais exibem apenas topologia para referência.")

positions = hypercube_positions_3d(dim) if use_3d else hypercube_positions_2d(dim)
frames_dict: Dict[str, np.ndarray] = {}
paths: Dict[str, List[int]] = {}
rng = np.random.default_rng(seed)

if mode_choice in ("Clássica", "Ambas"):
    classical_frames = simulate_classical(dim, max_steps, start_node)
    frames_dict["classical"] = classical_frames
    paths["classical"] = sample_path(classical_frames, start_node, rng)
if mode_choice in ("Quântica", "Ambas"):
    quantum_frames = simulate_quantum(dim, max_steps, start_node, bias=bias)
    frames_dict["quantum"] = quantum_frames
    paths["quantum"] = sample_path(quantum_frames, start_node, rng)

if not frames_dict:
    st.warning("Selecione pelo menos um modo para visualizar.")
    st.stop()

modes = list(frames_dict.keys())
num_steps = next(iter(frames_dict.values())).shape[0]
global_max = {key: frames.max() for key, frames in frames_dict.items()}
frames, initial_data = build_frames(positions, frames_dict, paths, modes, global_max, use_3d)

subplot_titles = [MODE_META[m]["name"] for m in modes]
specs = [[{"type": "scene"} for _ in modes]] if use_3d else [[{"type": "xy"} for _ in modes]]
fig = make_subplots(rows=1, cols=len(modes), subplot_titles=subplot_titles, horizontal_spacing=0.08, specs=specs)
for idx, trace in enumerate(initial_data):
    fig.add_trace(trace, row=1, col=idx // 2 + 1)

if use_3d:
    scene_template = dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False))
    if len(modes) == 1:
        fig.update_layout(scene=scene_template)
    else:
        for i in range(len(modes)):
            fig.update_layout(**{f"scene{i + 1}": scene_template})
else:
    axis_template = dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.1, 1.1])
    if len(modes) == 1:
        fig.update_xaxes(axis_template)
        fig.update_yaxes(axis_template, scaleanchor="x", scaleratio=1)
    else:
        for i in range(len(modes)):
            fig.update_xaxes(axis_template, col=i + 1)
            fig.update_yaxes(axis_template, col=i + 1, scaleanchor=f"x{i + 1}", scaleratio=1)

fig.frames = frames
fig.update_layout(
    height=520,
    margin=dict(l=40, r=40, t=60, b=40),
    updatemenus=[
        {
            "type": "buttons",
            "showactive": False,
            "x": 0.05,
            "y": 1.15,
            "buttons": [
                {
                    "label": "▶ Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 400, "redraw": True}, "fromcurrent": True}],
                },
                {
                    "label": "⏸ Pause",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                },
            ],
        }
    ],
    sliders=[
        {
            "pad": {"b": 10, "t": 40},
            "currentvalue": {"prefix": "Passo: ", "font": {"size": 16}},
            "steps": [
                {
                    "label": str(step),
                    "method": "animate",
                    "args": [[str(step)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                }
                for step in range(num_steps)
            ],
        }
    ],
)

st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)
for idx, mode_key in enumerate(modes):
    with (col1 if idx % 2 == 0 else col2):
        path = paths[mode_key]
        preview = ", ".join(map(str, path[: min(12, len(path))]))
        st.markdown(
            f"**Caminho {MODE_META[mode_key]['name']}:** {preview}{'…' if len(path) > 12 else ''}"
        )

st.caption(
    "Dica: ajuste a seed para gerar trajetórias diferentes do walker. "
    "A moeda Grover está normalizada; o bias opcional altera a superposição inicial das direções."
    f"\nNotas do cenário **{scenario.name}**: {scenario.notes}"
)
