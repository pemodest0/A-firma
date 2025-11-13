#!/usr/bin/env python3
"""
Dashboard interativo para o modelo de Ising 2D focado em uma lâmina de ferro.

Objetivo: mostrar, de maneira visual e simples, como os spins se organizam
quando mudamos temperatura e parâmetros da rede cristalina.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

SpinArray = np.ndarray

TC_REDUCED = 2.269  # Ising 2D adimensional
TC_REAL = 1043.0  # Ferro (Curie ~ 1043 K)


@dataclass
class IsingConfig:
    size: int
    temperature: float
    coupling: float
    mcs_per_frame: int
    lattice_spacing: float
    spring_constant: float


def plot_bcc_structure(lattice_spacing: float, magnetization: float) -> plt.Figure:
    """Desenha uma célula cúbica de corpo centrado (BCC) simples."""
    a = lattice_spacing
    base_points = np.array(
        [
            [0, 0, 0],
            [a, 0, 0],
            [0, a, 0],
            [0, 0, a],
            [a, a, 0],
            [a, 0, a],
            [0, a, a],
            [a, a, a],
            [a / 2, a / 2, a / 2],
        ]
    )
    major_color = (0.85, 0.15, 0.15) if magnetization >= 0 else (0.15, 0.2, 0.85)

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        base_points[:, 0],
        base_points[:, 1],
        base_points[:, 2],
        s=120,
        c=[major_color] * len(base_points),
        edgecolors="k",
    )
    cube = np.array(
        [
            [0, 0, 0],
            [a, 0, 0],
            [a, a, 0],
            [0, a, 0],
            [0, 0, 0],
            [0, 0, a],
            [a, 0, a],
            [a, a, a],
            [0, a, a],
            [0, 0, a],
        ]
    )
    ax.plot3D(cube[:5, 0], cube[:5, 1], cube[:5, 2], color="dimgray", linewidth=0.8)
    ax.plot3D(cube[5:, 0], cube[5:, 1], cube[5:, 2], color="dimgray", linewidth=0.8)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title("Célula BCC do ferro (cores = direção magnética)")
    return fig


def initial_lattice(size: int, mode: str, rng: np.random.Generator) -> SpinArray:
    if mode == "aleatório":
        spins = rng.choice([-1, 1], size=(size, size))
    elif mode == "tudo para cima":
        spins = np.ones((size, size), dtype=int)
    elif mode == "faixa metade/metade":
        spins = np.ones((size, size), dtype=int)
        spins[:, size // 2 :] = -1
    elif mode == "xadrez":
        grid = np.indices((size, size))
        spins = 1 - 2 * ((grid[0] + grid[1]) % 2)
    else:
        spins = rng.choice([-1, 1], size=(size, size))
    return spins.astype(int)


def metropolis_step(spins: SpinArray, beta: float, J: float, rng: np.random.Generator) -> None:
    n = spins.shape[0]
    for _ in range(n * n):
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        spin = spins[i, j]
        nn_sum = (
            spins[(i + 1) % n, j]
            + spins[(i - 1) % n, j]
            + spins[i, (j + 1) % n]
            + spins[i, (j - 1) % n]
        )
        dE = 2.0 * J * spin * nn_sum
        if dE <= 0 or rng.random() < np.exp(-beta * dE):
            spins[i, j] = -spin


def energy_per_spin(spins: SpinArray, J: float) -> float:
    horizontal = spins * np.roll(spins, -1, axis=1)
    vertical = spins * np.roll(spins, -1, axis=0)
    e_total = -J * (horizontal.sum() + vertical.sum())
    return e_total / spins.size


def domain_boundary_mask(spins: SpinArray) -> SpinArray:
    mask = np.zeros_like(spins, dtype=bool)
    mask[:-1, :] |= spins[:-1, :] != spins[1:, :]
    mask[:, :-1] |= spins[:, :-1] != spins[:, 1:]
    mask[-1, :] |= spins[-1, :] != spins[0, :]
    mask[:, -1] |= spins[:, -1] != spins[:, 0]
    return mask.astype(float)


def phase_description(reduced_temp: float) -> str:
    if reduced_temp < 0.8:
        return "Fase ordenada (ferromagnética). Domínios grandes predominam."
    if reduced_temp < 1.1:
        return "Região crítica: domínios crescem/encolhem; flutuações fortes."
    return "Fase desordenada (paramagnética). Spins se comportam de forma aleatória."


def scenario_notes(
    temperature_ratio: float, spring_constant: float, lattice_spacing: float
) -> str:
    notes = []
    if temperature_ratio < 0.3:
        notes.append(
            "Temperatura superfria: spins quase travados — pense em um cenário estilo supercondutividade."
        )
    elif temperature_ratio < 0.9:
        notes.append(
            "Regime ferromagnético estável: os domínios crescem e quase todo mundo aponta na mesma direção."
        )
    else:
        notes.append("Perto do ponto crítico: domínios nascem e morrem rápido; espere muita agitação.")

    if spring_constant > 60:
        notes.append("Molas internas rígidas mantêm a rede quase congelada, paredes de domínio movem devagar.")
    elif spring_constant < 5:
        notes.append("Molas frouxas deixam a rede vibrar bastante, então as fronteiras entre domínios são móveis.")
    else:
        notes.append("Rigidez intermediária: dá para ver fronteiras andando sem derreter o cristal.")

    notes.append(f"Distância média entre átomos usada no desenho: {lattice_spacing:.2f} Å.")
    return " ".join(notes)


def convert_temperature_input(temp_slider: float) -> Dict[str, float]:
    """
    Mapeia o controle 0.1–100 K para um valor reduzido.

    Convenção didática: 100 K no controle equivale ao ponto de Curie (~1043 K).
    Assim mantemos todas as fases acessíveis mesmo com um range pequeno.
    """
    ratio = temp_slider / 100.0
    ratio = max(ratio, 1e-3)
    reduced = ratio * TC_REDUCED
    real_equivalent = ratio * TC_REAL
    return {
        "reduced": reduced,
        "ratio": ratio,
        "real_equivalent": real_equivalent,
    }


def ensure_state_initialized(size: int, init_mode: str, rng_seed: int, cfg: IsingConfig) -> None:
    """Inicializa o estado no session_state se necessário."""
    if "ising_state" not in st.session_state:
        st.session_state.ising_state = {}
    state = st.session_state.ising_state
    if (
        state.get("spins") is None
        or state.get("size") != size
        or state.get("init_mode") != init_mode
        or state.get("seed") != rng_seed
        or st.session_state.get("reset_lattice", False)
    ):
        rng = np.random.default_rng(rng_seed)
        spins = initial_lattice(size, init_mode, rng)
        st.session_state.ising_state = {
            "spins": spins,
            "rng": rng,
            "size": size,
            "cfg": cfg,
            "seed": rng_seed,
            "init_mode": init_mode,
            "snapshots": [spins.copy()],
            "mag_history": [spins.mean()],
            "energy_history": [energy_per_spin(spins, cfg.coupling)],
            "frame_offset": 0,
            "max_snapshots": 400,
            "steps_done": 0,
        }
    st.session_state["reset_lattice"] = False


def record_snapshot(state: Dict, cfg: IsingConfig) -> None:
    spins = state["spins"]
    state["snapshots"].append(spins.copy())
    state["mag_history"].append(spins.mean())
    state["energy_history"].append(energy_per_spin(spins, cfg.coupling))
    if len(state["snapshots"]) > state["max_snapshots"]:
        state["snapshots"].pop(0)
        state["mag_history"].pop(0)
        state["energy_history"].pop(0)
        state["frame_offset"] += 1


def perform_iterations(state: Dict, cfg: IsingConfig, frames_to_add: int) -> None:
    spins = state["spins"]
    rng = state["rng"]
    beta = 1.0 / cfg.temperature if cfg.temperature > 0 else np.inf
    for _ in range(frames_to_add):
        for _ in range(cfg.mcs_per_frame):
            metropolis_step(spins, beta=beta, J=cfg.coupling, rng=rng)
        state["steps_done"] += cfg.size * cfg.size * cfg.mcs_per_frame
        record_snapshot(state, cfg)


def plot_spins(spins: SpinArray, boundaries: SpinArray, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.get_cmap("coolwarm")
    im = ax.imshow(spins, cmap=cmap, vmin=-1, vmax=1, origin="lower", interpolation="nearest")
    ax.imshow(boundaries, cmap="gray", alpha=0.25, origin="lower", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spin (+1 vermelho / -1 azul)")
    return fig


def render_dashboard(
    state: Dict,
    cfg: IsingConfig,
    temperature_ratio: float,
    temperature_display: float,
    temperature_real: float,
) -> None:
    snapshots: List[SpinArray] = state["snapshots"]
    frame_labels = list(range(state["frame_offset"], state["frame_offset"] + len(snapshots)))
    if len(frame_labels) == 1:
        frame_choice = frame_labels[0]
        st.info("Somente um quadro disponível até agora. Rode mais iterações para ver a evolução.")
    else:
        frame_choice = st.slider(
            "Escolha o quadro para visualizar",
            min_value=frame_labels[0],
            max_value=frame_labels[-1],
            value=frame_labels[-1],
            step=1,
        )
    idx = frame_choice - state["frame_offset"]
    current_spins = snapshots[idx]
    boundaries = domain_boundary_mask(current_spins)

    col_lattice, col_info = st.columns((1.3, 1))
    with col_lattice:
        title = f"Rede {cfg.size}×{cfg.size} — snapshot #{frame_choice}"
        fig = plot_spins(current_spins, boundaries, title)
        st.pyplot(fig, clear_figure=True)

    with col_info:
        phase_text = phase_description(cfg.temperature)
        up_fraction = (current_spins == 1).mean()
        down_fraction = 1.0 - up_fraction

        st.subheader("Painel rápido")
        st.metric("Magnetização atual", f"{state['mag_history'][idx]:+.3f}")
        st.metric("Energia por spin", f"{state['energy_history'][idx]:+.3f}")
        st.metric("Spins +1", f"{up_fraction * 100:.1f}%")
        st.metric("Spins -1", f"{down_fraction * 100:.1f}%")
        st.metric("Temperatura indicada", f"{temperature_display:.1f} K*")
        st.metric("Equivalente real", f"{temperature_real:.1f} K ≈ {temperature_ratio * 100:.0f}% T_c")
        st.caption(phase_text.replace("ferromagnética", "ferro alinhado").replace("paramagnética", "desalinhado"))

        st.subheader("Evolução da magnetização")
        df_hist = pd.DataFrame(
            {
                "Magnetização": state["mag_history"],
                "Energia": state["energy_history"],
            },
            index=frame_labels,
        )
        st.line_chart(df_hist)
        st.caption("Linhas mostram como o cristal foi se organizando em cada cena.")

        st.subheader("Célula BCC de ferro")
        structure_fig = plot_bcc_structure(cfg.lattice_spacing, state["mag_history"][idx])
        st.pyplot(structure_fig, clear_figure=True)
        st.caption("Pontinhos representam átomos de ferro na célula cúbica de corpo centrado.")

        st.subheader("Resumo físico")
        st.write(
            scenario_notes(
                temperature_ratio=temperature_ratio,
                spring_constant=cfg.spring_constant,
                lattice_spacing=cfg.lattice_spacing,
            )
        )


def main() -> None:
    st.set_page_config(page_title="Ising 2D interativo", layout="wide")
    st.title("Brinque com o cristal de ferro (Modelo de Ising 2D)")
    st.markdown(
        """
O tabuleiro abaixo representa uma lâmina quadrada de ferro. Cada quadradinho é um spin que pode
apontar para cima (+1, vermelho) ou para baixo (−1, azul). Ao mudar a temperatura e os parâmetros
de rede você vê, quase como um desenho animado, como os átomos conversam entre si.

Para deixar tudo acessível, usamos um controle de temperatura de 0,1 K a 100 K onde
100 K corresponde ao ponto de Curie real do ferro (~1043 K). Isso permite passear por todas as fases
sem precisar digitar números gigantes.
        """
    )

    with st.sidebar:
        st.header("Ajustes do cristal")
        temperature_display = st.slider("Temperatura alvo (K*)", 0.1, 100.0, 20.0, 0.1)
        temp_info = convert_temperature_input(temperature_display)

        coupling = st.slider("Força de acoplamento magnético J", 0.1, 2.0, 1.0, 0.05)
        lattice_spacing = st.slider("Distância entre átomos (Å)", 0.5, 3.0, 2.5, 0.1)
        spring_constant = st.slider("Rigidez das ligações (N/m)", 0.1, 100.0, 10.0, 0.1)

        st.header("Como a simulação anda")
        size = st.slider("Número de átomos por lado", 1, 100, 32, 1)
        mcs_per_frame = st.slider("Tentativas de flip por cena", 1, 200, 40, 1)
        frames_to_add = st.slider("Cenas por rodada", 1, 300, 80, 1)

        init_mode = st.selectbox(
            "Comece assim",
            ["aleatório", "tudo para cima", "faixa metade/metade", "xadrez"],
        )
        seed = st.number_input("Semente (para repetir o experimento)", value=2024, step=1)

        col_buttons = st.columns(2)
        if col_buttons[0].button("Reiniciar cristal", use_container_width=True):
            st.session_state["reset_lattice"] = True
            st.session_state["run_flag"] = False
        if col_buttons[1].button("Rodar cenas", use_container_width=True):
            st.session_state["run_flag"] = True

    cfg = IsingConfig(
        size=size,
        temperature=temp_info["reduced"],
        coupling=coupling,
        mcs_per_frame=mcs_per_frame,
        lattice_spacing=lattice_spacing,
        spring_constant=spring_constant,
    )
    ensure_state_initialized(size=size, init_mode=init_mode, rng_seed=int(seed), cfg=cfg)

    state = st.session_state.ising_state
    state["cfg"] = cfg  # atualiza config guardada

    if st.session_state.get("run_flag", False):
        perform_iterations(state, cfg, frames_to_add=frames_to_add)
        st.session_state["run_flag"] = False

    render_dashboard(
        state,
        cfg,
        temperature_ratio=temp_info["ratio"],
        temperature_display=temperature_display,
        temperature_real=temp_info["real_equivalent"],
    )

    st.markdown("---")
    st.markdown(
        """
**Dicas didáticas**:
- Use `faixa metade/metade` para acompanhar o passeio de uma parede de domínio.
- Para cidades superfrias (temperatura < 10 K*) os spins quase não mudam — ótimo para discutir “supercondutividade”.
- Temperaturas próximas de 100 K* mostram flutuações grandes, boas para falar de ponto crítico.
        """
    )


if __name__ == "__main__":
    main()
