#!/usr/bin/env python3
"""
Streamlit App — Schrödinger Deformado interativo.

Executa uma simulação 1D com termo de deformação ε·F(x,t) e amortecimento γ.
O solver usa Crank–Nicolson para garantir estabilidade em passos pequenos.
"""
from __future__ import annotations

import functools
from typing import Dict, Tuple

import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

ħ = 1.0
m = 1.0


def _build_laplacian(n_points: int, dx: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    main = np.full(n_points, -2.0)
    off = np.ones(n_points - 1)
    diagonals = [main, off, off]
    laplacian = diags(diagonals, [0, -1, 1], format="csc")
    return laplacian / dx**2


@functools.lru_cache(maxsize=16)
def _precompute_system(
    potential_key: str,
    epsilon: float,
    gamma: float,
    driving_freq: float,
    n_points: int,
    x_min: float,
    x_max: float,
    dt: float,
) -> Dict[str, np.ndarray]:
    x = np.linspace(x_min, x_max, n_points)
    dx = x[1] - x[0]

    laplacian = _build_laplacian(n_points, dx)

    if potential_key == "harmônico":
        omega = 1.0
        V = 0.5 * m * omega**2 * x**2
    elif potential_key == "barreira":
        V = np.where(np.abs(x) < 0.5, 5.0, 0.0)
    elif potential_key == "duplo poço":
        V = 0.5 * (x**4 - x**2)
    else:
        V = np.zeros_like(x)

    H0 = (-ħ**2 / (2 * m)) * laplacian + diags(V, 0)

    H_diag = H0.diagonal()
    H_off_lower = H0.diagonal(-1)
    H_off_upper = H0.diagonal(1)

    A_diag = 1.0 + 0.5j * dt * H_diag / ħ + 0.5 * dt * gamma
    B_diag = 1.0 - 0.5j * dt * H_diag / ħ - 0.5 * dt * gamma

    A_lower = 0.5j * dt * H_off_lower / ħ
    A_upper = 0.5j * dt * H_off_upper / ħ
    B_lower = -0.5j * dt * H_off_lower / ħ
    B_upper = -0.5j * dt * H_off_upper / ħ

    A = diags([A_lower, A_diag, A_upper], [-1, 0, 1], format="csc")
    B = diags([B_lower, B_diag, B_upper], [-1, 0, 1], format="csc")

    return {
        "x": x,
        "A": A,
        "B": B,
        "potential": V,
        "epsilon": epsilon,
        "gamma": gamma,
        "drive_freq": driving_freq,
    }


def _drive_field(x: np.ndarray, t: float, epsilon: float, drive_freq: float) -> np.ndarray:
    return epsilon * np.sin(np.pi * x) * np.cos(drive_freq * t)


def simulate(
    potential_key: str,
    epsilon: float,
    gamma: float,
    driving_freq: float,
    n_points: int,
    x_bounds: Tuple[float, float],
    dt: float,
    n_steps: int,
    wave_center: float,
    wave_width: float,
    momentum: float,
) -> Dict[str, np.ndarray]:
    system = _precompute_system(
        potential_key,
        epsilon,
        gamma,
        driving_freq,
        n_points,
        x_bounds[0],
        x_bounds[1],
        dt,
    )
    x = system["x"]
    A = system["A"]
    B = system["B"]
    V = system["potential"]

    psi0 = (
        (1.0 / (np.pi * wave_width**2) ** 0.25)
        * np.exp(-((x - wave_center) ** 2) / (2 * wave_width**2))
        * np.exp(1j * momentum * x / ħ)
    )
    psi = psi0.copy()

    times = np.linspace(0, n_steps * dt, n_steps + 1)
    probs = np.empty((n_steps + 1, n_points), dtype=float)
    expectation_x = np.empty(n_steps + 1, dtype=float)
    expectation_p = np.empty(n_steps + 1, dtype=float)
    probs[0] = np.abs(psi0) ** 2
    expectation_x[0] = np.trapz(probs[0] * x, x)
    expectation_p[0] = np.trapz((np.conjugate(psi0) * (-1j * ħ * np.gradient(psi0, x))).real, x)

    for idx in range(1, n_steps + 1):
        t_mid = (idx - 0.5) * dt
        deform = _drive_field(x, t_mid, epsilon, driving_freq)
        rhs = B.dot(psi) + 1j * dt * deform * psi / ħ
        psi = spsolve(A, rhs)

        probs[idx] = np.abs(psi) ** 2
        expectation_x[idx] = np.trapz(probs[idx] * x, x)
        expectation_p[idx] = np.trapz((np.conjugate(psi) * (-1j * ħ * np.gradient(psi, x))).real, x)

        norm = np.sqrt(np.trapz(probs[idx], x))
        if norm > 0:
            psi /= norm
            probs[idx] /= norm**2

    target_mask = x >= 0
    cumulative_prob = probs[:, target_mask].sum(axis=1)
    threshold = 0.1
    hits = np.where(cumulative_prob >= threshold)[0]
    hitting_time = times[hits[0]] if hits.size else None

    return {
        "x": x,
        "times": times,
        "probs": probs,
        "potential": V,
        "expectation_x": expectation_x,
        "expectation_p": expectation_p,
        "cumulative_prob_right": cumulative_prob,
        "hitting_time": hitting_time,
        "threshold": threshold,
    }


def plot_probability(x: np.ndarray, probs: np.ndarray, times: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    pcm = ax.pcolormesh(times, x, probs.T, shading="auto", cmap="magma")
    ax.set_xlabel("tempo")
    ax.set_ylabel("posição x")
    fig.colorbar(pcm, ax=ax, label="|ψ|²")
    return fig


def plot_expectations(times: np.ndarray, expectation_x: np.ndarray, expectation_p: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    ax[0].plot(times, expectation_x, color="tab:blue")
    ax[0].set_ylabel("⟨x⟩")
    ax[0].grid(True, alpha=0.3)
    ax[1].plot(times, expectation_p, color="tab:orange")
    ax[1].set_ylabel("⟨p⟩")
    ax[1].set_xlabel("tempo")
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title="Schrödinger Deformado", layout="wide")
    st.title("Simulador — Equação de Schrödinger Deformada (1D)")

    with st.sidebar:
        st.header("Parâmetros físicos")
        potential = st.selectbox("Potencial", ["livre", "harmônico", "barreira", "duplo poço"])
        epsilon = st.slider("Força de deformação ε", 0.0, 2.0, 0.5, 0.05)
        gamma = st.slider("Amortecimento γ", 0.0, 0.5, 0.05, 0.01)
        drive_freq = st.slider("Frequência de drive", 0.0, 5.0, 1.0, 0.1)

        st.header("Configuração numérica")
        n_points = st.slider("Pontos na malha", 200, 600, 300, 50)
        n_steps = st.slider("Passos de tempo", 50, 400, 150, 25)
        dt = st.slider("Δt", 0.001, 0.02, 0.005, 0.001)
        x_min, x_max = -5.0, 5.0

        st.header("Estado inicial")
        wave_center = st.slider("Centro inicial", -2.0, 2.0, -1.5, 0.1)
        wave_width = st.slider("Largura inicial", 0.2, 1.5, 0.5, 0.05)
        momentum = st.slider("Momento inicial", -5.0, 5.0, 2.0, 0.1)

        if st.button("Executar simulação", type="primary"):
            st.session_state["run_sim"] = True

    if st.session_state.get("run_sim", False):
        with st.spinner("Simulando..."):
            results = simulate(
                potential_key=potential,
                epsilon=epsilon,
                gamma=gamma,
                driving_freq=drive_freq,
                n_points=n_points,
                x_bounds=(x_min, x_max),
                dt=dt,
                n_steps=n_steps,
                wave_center=wave_center,
                wave_width=wave_width,
                momentum=momentum,
            )

        col_left, col_right = st.columns((1.4, 1))
        with col_left:
            st.subheader("Distribuição de probabilidade |ψ(x,t)|²")
            fig_prob = plot_probability(results["x"], results["probs"], results["times"])
            st.pyplot(fig_prob, clear_figure=True)

            st.subheader("Evolução de ⟨x⟩ e ⟨p⟩")
            fig_exp = plot_expectations(results["times"], results["expectation_x"], results["expectation_p"])
            st.pyplot(fig_exp, clear_figure=True)

        with col_right:
            st.subheader("Probabilidade acumulada (x ≥ 0)")
            st.line_chart(
                {
                    "tempo": results["times"],
                    "prob_acumulada": results["cumulative_prob_right"],
                },
                x="tempo",
                y="prob_acumulada",
            )
            hit_time = results["hitting_time"]
            if hit_time is not None:
                st.success(f"Hitting time (prob ≥ {results['threshold']:.2f}) ≈ {hit_time:.3f}")
            else:
                st.warning("Probabilidade nunca atingiu o limiar escolhido.")

            st.subheader("Parâmetros utilizados")
            st.json(
                {
                    "potencial": potential,
                    "ε": epsilon,
                    "γ": gamma,
                    "freq_drive": drive_freq,
                    "passos": n_steps,
                    "Δt": dt,
                    "n_pontos": n_points,
                }
            )
    else:
        st.info("Configure os parâmetros na barra lateral e clique em 'Executar simulação'.")


if __name__ == "__main__":
    main()
