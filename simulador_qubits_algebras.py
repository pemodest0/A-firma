"""Streamlit app comparing complex, quaternion, and GA qubit evolutions."""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterable, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from clifford.g3 import e1, e2, e3, e12, e23

# e31 nem sempre é exportado; construímos manualmente para evitar ImportError.
e31 = e3 * e1

from gaq.core import channels
from gaq.core.ga import bloch_from_rho, expm as ga_expm, rotor_to_unitary

Algebra = Literal["Complex", "Quaternion", "Geométrica"]
PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
PAULI_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
PAULIS = np.stack([PAULI_X, PAULI_Y, PAULI_Z])
STATE_MAP = {
    "|0⟩": np.array([1.0, 0.0], dtype=complex),
    "|1⟩": np.array([0.0, 1.0], dtype=complex),
    "|+⟩": (1 / math.sqrt(2)) * np.array([1.0, 1.0], dtype=complex),
    "|-⟩": (1 / math.sqrt(2)) * np.array([1.0, -1.0], dtype=complex),
}


@dataclass
class Trajectory:
    times: np.ndarray
    bloch_points: np.ndarray
    density_history: list[np.ndarray]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalize(vector: Iterable[float]) -> np.ndarray:
    vec = np.array(vector, dtype=float)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def state_to_density(psi: np.ndarray) -> np.ndarray:
    psi = psi / np.linalg.norm(psi)
    return np.outer(psi, psi.conj())


def bloch_to_density(bloch: np.ndarray) -> np.ndarray:
    return 0.5 * (np.eye(2) + bloch[0] * PAULI_X + bloch[1] * PAULI_Y + bloch[2] * PAULI_Z)


def apply_noise(rho: np.ndarray, noise_kind: str, strength: float) -> np.ndarray:
    if noise_kind == "Nenhum" or strength == 0.0:
        return rho
    match noise_kind:
        case "Dephasing":
            return channels.apply_kraus(rho, channels.dephasing(strength))
        case "Depolarizing":
            return channels.apply_kraus(rho, channels.depolarizing(strength))
        case "Amplitude damping":
            return channels.apply_kraus(rho, channels.amplitude_damping(strength))
        case _:
            raise ValueError(noise_kind)


def quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def rotate_quaternion(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = normalize(axis)
    w = math.cos(angle / 2.0)
    xyz = axis * math.sin(angle / 2.0)
    q = np.concatenate(([w], xyz))
    q_conj = np.concatenate(([w], -xyz))
    v = np.concatenate(([0.0], vector))
    rotated = quaternion_multiply(quaternion_multiply(q, v), q_conj)
    return rotated[1:]


def fidelity_pure(rho: np.ndarray, psi: np.ndarray) -> float:
    psi = psi.reshape(-1, 1)
    return float(np.real((psi.conj().T @ rho @ psi))[0, 0])


def hitting_time(traj: Trajectory, target: np.ndarray, eps: float = 0.05) -> float | None:
    distances = np.linalg.norm(traj.bloch_points - target, axis=1)
    idx = np.where(distances <= eps)[0]
    if idx.size:
        return float(traj.times[idx[0]])
    return None


# ---------------------------------------------------------------------------
# Evolutions
# ---------------------------------------------------------------------------


def timeline(total_time: float, steps: int) -> np.ndarray:
    return np.linspace(0.0, total_time, steps)


def complex_evolution(
    psi0: np.ndarray,
    axis: np.ndarray,
    angle: float,
    times: np.ndarray,
    noise_kind: str,
    noise_strength: float,
) -> Trajectory:
    axis = normalize(axis)
    hamiltonian = angle * (axis[0] * PAULI_X + axis[1] * PAULI_Y + axis[2] * PAULI_Z)
    history: list[np.ndarray] = []
    bloch_pts: list[np.ndarray] = []
    for t in times:
        unitary = ga_expm(-1j * hamiltonian * t)
        rho = state_to_density(unitary @ psi0)
        rho = apply_noise(rho, noise_kind, noise_strength)
        history.append(rho)
        bloch_pts.append(bloch_from_rho(rho))
    return Trajectory(times, np.array(bloch_pts), history)


def quaternion_evolution(
    psi0: np.ndarray,
    axis: np.ndarray,
    angle: float,
    times: np.ndarray,
    noise_kind: str,
    noise_strength: float,
) -> Trajectory:
    init_bloch = bloch_from_rho(state_to_density(psi0))
    bloch_pts: list[np.ndarray] = []
    history: list[np.ndarray] = []
    for t in times:
        bloch_vec = rotate_quaternion(init_bloch, axis, angle * t)
        rho = bloch_to_density(bloch_vec)
        rho = apply_noise(rho, noise_kind, noise_strength)
        history.append(rho)
        bloch_pts.append(bloch_from_rho(rho))
    return Trajectory(times, np.array(bloch_pts), history)


def geometric_evolution(
    psi0: np.ndarray,
    axis: np.ndarray,
    angle: float,
    times: np.ndarray,
    noise_kind: str,
    noise_strength: float,
) -> Trajectory:
    rho0 = state_to_density(psi0)
    bloch_pts: list[np.ndarray] = []
    history: list[np.ndarray] = []
    for t in times:
        unitary = rotor_to_unitary(axis, angle * t)
        rho = unitary @ rho0 @ unitary.conj().T
        rho = apply_noise(rho, noise_kind, noise_strength)
        history.append(rho)
        bloch_pts.append(bloch_from_rho(rho))
    return Trajectory(times, np.array(bloch_pts), history)


SIMULATORS: dict[Algebra, Callable[[np.ndarray, np.ndarray, float, np.ndarray, str, float], Trajectory]] = {
    "Complex": complex_evolution,
    "Quaternion": quaternion_evolution,
    "Geométrica": geometric_evolution,
}


# ---------------------------------------------------------------------------
# Visualizações
# ---------------------------------------------------------------------------


def bloch_figure(traj: Trajectory, title: str) -> plt.Figure:
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(traj.bloch_points[:, 0], traj.bloch_points[:, 1], traj.bloch_points[:, 2], color="tab:blue")
    ax.scatter([0], [0], [1], color="green", label="|0⟩")
    ax.scatter([0], [0], [-1], color="red", label="|1⟩")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    return fig


def hitting_plot(traj: Trajectory, target: np.ndarray, eps: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 2.5))
    distances = np.linalg.norm(traj.bloch_points - target, axis=1)
    ax.plot(traj.times, distances, color="tab:purple")
    hit = hitting_time(traj, target, eps)
    if hit is not None:
        ax.axvline(hit, color="red", linestyle="--", label=f"hit @ {hit:.2f}")
    ax.set_xlabel("tempo")
    ax.set_ylabel("‖Bloch - alvo‖")
    ax.legend()
    return fig


def heatmap_hitting(
    algebra: Algebra,
    psi0: np.ndarray,
    axis: np.ndarray,
    angle_range: Sequence[float],
    time_range: Sequence[float],
    steps: int,
    target: np.ndarray,
    eps: float,
    noise_kind: str,
    noise_strength: float,
) -> tuple[plt.Figure, np.ndarray]:
    data = np.full((len(time_range), len(angle_range)), np.nan)
    for i, total_time in enumerate(time_range):
        times = timeline(total_time, steps)
        for j, angle_deg in enumerate(angle_range):
            angle = math.radians(angle_deg)
            traj = SIMULATORS[algebra](psi0, axis, angle, times, noise_kind, noise_strength)
            ht = hitting_time(traj, target, eps)
            data[i, j] = ht if ht is not None else total_time
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    c = ax.imshow(
        data,
        extent=[angle_range[0], angle_range[-1], time_range[0], time_range[-1]],
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    ax.set_xlabel("Ângulo (graus)")
    ax.set_ylabel("Tempo total")
    ax.set_title(f"Heatmap hitting ({algebra})")
    fig.colorbar(c, ax=ax, label="Tempo até alvo")
    return fig, data


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def sidebar_controls() -> dict:
    st.sidebar.header("Configuração")
    algebras = st.sidebar.multiselect(
        "Álgebras", ["Complex", "Quaternion", "Geométrica"], default=["Complex", "Quaternion", "Geométrica"]
    )
    init_state_label = st.sidebar.selectbox("Estado inicial", list(STATE_MAP.keys()), index=0)
    target_label = st.sidebar.selectbox("Estado alvo", list(STATE_MAP.keys()), index=1)
    axis_x = st.sidebar.slider("Eixo X", -1.0, 1.0, 0.0)
    axis_y = st.sidebar.slider("Eixo Y", -1.0, 1.0, 1.0)
    axis_z = st.sidebar.slider("Eixo Z", -1.0, 1.0, 0.0)
    angle_deg = st.sidebar.slider("Ângulo base (graus)", 0.0, 360.0, 90.0)
    total_time = st.sidebar.slider("Tempo total", 0.1, 10.0, 2.0)
    steps = st.sidebar.slider("Passos de simulação", 10, 200, 80)
    noise_kind = st.sidebar.selectbox("Ruído", ["Nenhum", "Dephasing", "Depolarizing", "Amplitude damping"], index=0)
    noise_strength = st.sidebar.slider("Força do ruído", 0.0, 1.0, 0.0)
    eps = st.sidebar.slider("Tolerância hitting", 0.01, 0.3, 0.05)
    axis = np.array([axis_x, axis_y, axis_z], dtype=float)
    if np.linalg.norm(axis) < 1e-9:
        axis = np.array([0.0, 1.0, 0.0])
    return {
        "algebras": algebras,
        "init_state": STATE_MAP[init_state_label],
        "target_state": STATE_MAP[target_label],
        "axis": axis,
        "angle": math.radians(angle_deg),
        "total_time": total_time,
        "steps": steps,
        "noise_kind": noise_kind,
        "noise_strength": noise_strength,
        "eps": eps,
    }


def main() -> None:
    st.set_page_config(page_title="Simulador de Dinâmicas Quânticas", layout="wide")
    st.title("Simulação comparativa: Complexos × Quaternions × Álgebra Geométrica")
    st.write(
        "Explore a evolução temporal de um qubit usando representações distintas. Ajuste o eixo, o ângulo, "
        "o tempo e o ruído; compare hitting time, fidelidade e trajetórias 3D."
    )

    cfg = sidebar_controls()
    if not cfg["algebras"]:
        st.warning("Selecione ao menos uma álgebra.")
        return

    times = timeline(cfg["total_time"], cfg["steps"])
    target_rho = state_to_density(cfg["target_state"])
    target_bloch = bloch_from_rho(target_rho)

    tabs = st.tabs(cfg["algebras"])
    for tab, algebra in zip(tabs, cfg["algebras"]):
        with tab:
            traj = SIMULATORS[algebra](
                cfg["init_state"],
                cfg["axis"],
                cfg["angle"],
                times,
                cfg["noise_kind"],
                cfg["noise_strength"],
            )
            st.pyplot(bloch_figure(traj, f"Trajetória Bloch ({algebra})"))
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(hitting_plot(traj, target_bloch, cfg["eps"]))
                hit = hitting_time(traj, target_bloch, cfg["eps"])
                st.metric("Hitting time", f"{hit:.3f}" if hit is not None else "não alcançado")
            with col2:
                fidelity = fidelity_pure(traj.density_history[-1], cfg["target_state"])
                st.metric("Fidelidade final", f"{fidelity:.4f}")
                heatmap_fig, _ = heatmap_hitting(
                    algebra,
                    cfg["init_state"],
                    cfg["axis"],
                    angle_range=np.linspace(0.0, 360.0, 20),
                    time_range=np.linspace(cfg["total_time"] / 4, cfg["total_time"], 10),
                    steps=max(20, cfg["steps"] // 2),
                    target=target_bloch,
                    eps=cfg["eps"],
                    noise_kind=cfg["noise_kind"],
                    noise_strength=cfg["noise_strength"],
                )
                st.pyplot(heatmap_fig)
                with st.expander("Rotor na álgebra geométrica (clifford.g3)"):
                    biv = cfg["axis"][0] * e23 + cfg["axis"][1] * e31 + cfg["axis"][2] * e12
                    rotor = (-0.5 * cfg["angle"] * biv).exp()
                    st.write(rotor)

    st.info(
        "As trajetórias da álgebra geométrica usam rotores SU(2); já os quaternions giram o vetor de Bloch e em seguida "
        "reconstroem a densidade, mantendo números reais. Ambos podem receber ruído mapeado em canais de Kraus."
    )


if __name__ == "__main__":
    main()
