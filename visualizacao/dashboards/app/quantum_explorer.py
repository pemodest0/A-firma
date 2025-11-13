#!/usr/bin/env python3
from __future__ import annotations

import cmath
import math
import sys
from pathlib import Path
from typing import Any, Dict, Tuple, TYPE_CHECKING

import numpy as np
import streamlit as st
try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - optional dependency
    go = None

if TYPE_CHECKING:  # pragma: no cover - type checking helper
    from plotly.graph_objects import Figure
else:
    Figure = Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "modelos" / "core" / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(candidate)
    if path_str not in sys.path:
        sys.path.append(path_str)

from src.classical_walk import simulate_classical_walk
from src.graph_utils import line_graph
from src.quantum_walk import QISKIT_AVAILABLE, simulate_quantum_walk


def _format_coin_state(alpha: complex, beta: complex) -> str:
    amp0 = f"{alpha.real:+.3f}{alpha.imag:+.3f}i"
    amp1 = f"{beta.real:+.3f}{beta.imag:+.3f}i"
    return f"|ψ₀⟩ = {amp0} |0⟩ + {amp1} |1⟩"


@st.cache_data(show_spinner=False)
def _simulate_classical(num_nodes: int, steps: int) -> Dict[str, np.ndarray]:
    center = num_nodes // 2
    graph = line_graph(num_nodes)
    result = simulate_classical_walk(graph, steps, start_node=center)
    return {
        "positions": result.positions - center,
        "distributions": result.distributions,
        "entropies": result.entropies,
    }


@st.cache_data(show_spinner=False)
def _simulate_quantum(
    num_nodes: int,
    steps: int,
    coin: str,
    alpha_components: Tuple[float, float],
    beta_components: Tuple[float, float],
) -> Dict[str, np.ndarray]:
    center = num_nodes // 2
    graph = line_graph(num_nodes)
    alpha = complex(alpha_components[0], alpha_components[1])
    beta = complex(beta_components[0], beta_components[1])
    result = simulate_quantum_walk(
        graph,
        steps,
        coin=coin,
        start_node=center,
        coin_state=(alpha, beta),
    )
    return {
        "positions": result.positions - center,
        "distributions": result.distributions,
        "entropies": result.entropies,
        "coin_label": result.coin_label,
    }


def _build_distribution_animation(
    positions: np.ndarray,
    classical: np.ndarray,
    quantum: np.ndarray,
) -> Figure:
    steps = classical.shape[0]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=positions,
            y=classical[0],
            name="Clássico",
            marker_color="#1f77b4",
        )
    )
    fig.add_trace(
        go.Bar(
            x=positions,
            y=quantum[0],
            name="Quântico",
            marker_color="#d62728",
        )
    )

    frames = []
    for step in range(steps):
        frames.append(
            go.Frame(
                data=[
                    go.Bar(x=positions, y=classical[step]),
                    go.Bar(x=positions, y=quantum[step]),
                ],
                name=str(step),
            )
        )

    slider_steps = [
        {"args": [[str(k)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}], "label": str(k), "method": "animate"}
        for k in range(steps)
    ]

    fig.update_layout(
        title="Distribuição de Probabilidade (|ψ|²)",
        barmode="overlay",
        bargap=0.15,
        bargroupgap=0.05,
        xaxis_title="Posição (deslocamento)",
        yaxis_title="Probabilidade",
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "▶️ Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 150, "redraw": True}, "fromcurrent": True}],
                    },
                    {
                        "label": "⏸ Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"},
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "currentvalue": {"prefix": "Passo: "},
                "steps": slider_steps,
            }
        ],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.frames = frames
    return fig


def _build_entropy_chart(classical: np.ndarray, quantum: np.ndarray) -> Figure:
    steps = np.arange(classical.shape[0])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=classical,
            mode="lines",
            name="Clássico",
            line=dict(color="#1f77b4"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=quantum,
            mode="lines",
            name="Quântico",
            line=dict(color="#d62728"),
        )
    )
    fig.update_layout(
        title="Entropia de Shannon ao longo do tempo",
        xaxis_title="Passo",
        yaxis_title="Entropia (bits)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _build_quantum_heatmap(positions: np.ndarray, quantum: np.ndarray) -> Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=quantum,
            x=positions,
            y=np.arange(quantum.shape[0]),
            colorscale="Viridis",
        )
    )
    fig.update_layout(
        title="Mapa de calor da distribuição quântica",
        xaxis_title="Posição (deslocamento)",
        yaxis_title="Passo",
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="Explorador Quântico", page_icon="🔭", layout="wide")
    st.title("🔭 Explorador Quântico")
    st.markdown(
        """
        Compare caminhadas aleatórias clássicas e quânticas em 1D. Ajuste a moeda quântica, observe
        a interferência na distribuição (|ψ|²) e acompanhe a entropia ao longo dos passos.
        Todas as simulações rodam localmente com Qiskit (Statevector).
        """
    )

    if go is None:
        st.error("Plotly não está instalado. Execute `pip install plotly` para habilitar os gráficos interativos.")
        st.stop()

    if not QISKIT_AVAILABLE:
        st.error(
            "Qiskit não está instalado/encontrado. Execute `pip install qiskit` (e qiskit-aer) antes de rodar este explorador."
        )
        st.stop()

    with st.sidebar:
        st.header("🎛️ Controles")
        max_steps = st.slider("Número de passos", min_value=5, max_value=120, value=40, step=5)
        coin_choice = st.selectbox("Moeda quântica", options=["hadamard", "grover"], index=0)
        theta = st.slider(
            "Ângulo θ (define |α| e |β|)",
            min_value=0.0,
            max_value=math.pi / 2,
            value=math.pi / 4,
            step=math.pi / 100,
        )
        phi = st.slider(
            "Fase φ (rad) aplicada em |1⟩",
            min_value=0.0,
            max_value=2 * math.pi,
            value=0.0,
            step=math.pi / 50,
        )
    st.caption("α = cos θ · 1, β = sin θ · e^{iφ}")

    alpha = math.cos(theta)
    beta = math.sin(theta) * cmath.exp(1j * phi)
    alpha_components = (float(alpha.real), float(alpha.imag))
    beta_components = (float(beta.real), float(beta.imag))
    num_nodes = 2 * max_steps + 1
    center = num_nodes // 2

    classical = _simulate_classical(num_nodes, max_steps)
    quantum = _simulate_quantum(
        num_nodes,
        max_steps,
        coin_choice,
        alpha_components,
        beta_components,
    )

    st.subheader("Estado inicial da moeda")
    st.write(_format_coin_state(alpha, beta))

    dist_fig = _build_distribution_animation(
        classical["positions"],
        classical["distributions"],
        quantum["distributions"],
    )
    entropy_fig = _build_entropy_chart(classical["entropies"], quantum["entropies"])
    heatmap_fig = _build_quantum_heatmap(classical["positions"], quantum["distributions"])

    st.plotly_chart(dist_fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(entropy_fig, use_container_width=True)
    with col2:
        st.plotly_chart(heatmap_fig, use_container_width=True)

    st.markdown(
        """
        ### Por que a distribuição é diferente?
        - A caminhada **clássica** espalha probabilidade de forma difusiva (Gaussiana).
        - A caminhada **quântica** sofre interferência: amplitudes positivas/negativas se cancelam ou reforçam,
          criando lóbulos assimétricos.
        - A entropia quântica oscila porque a interferência pode concentrar probabilidade em poucos nós.
        """
    )


if __name__ == "__main__":
    main()
