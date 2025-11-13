"""Interactive portfolio describing GA-rotor coins for quantum walks."""
from __future__ import annotations

import math
from textwrap import dedent

import numpy as np

from gaq.core.ga import bloch_from_rho, rho_from_bloch
from gaq.walks.coin_ga import coin_rotor

PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
HADAMARD = (1 / np.sqrt(2)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex)
BASIS_STATES = {
    "|0⟩": np.array([[1.0], [0.0]], dtype=complex),
    "|1⟩": np.array([[0.0], [1.0]], dtype=complex),
}


def _format_complex(value: complex) -> str:
    real = value.real
    imag = value.imag
    if abs(imag) < 1e-12:
        return f"{real:+.6f}"
    if abs(real) < 1e-12:
        return f"{imag:+.6f}i"
    sign = "+" if imag >= 0 else "-"
    return f"{real:+.6f} {sign} {abs(imag):.6f}i"


def _print_matrix(name: str, matrix: np.ndarray) -> None:
    print(f"{name}:")
    for row in matrix:
        formatted = "  ".join(_format_complex(val) for val in row)
        print(f"  [{formatted}]")
    print()


def _intro() -> None:
    print(
        dedent(
            """\
            🔹 Bem-vindo ao portfólio interativo da moeda GAQ.
            A meta aqui é explicar, com matemática explícita, como uma rotação geométrica
            (rotor) gera a porta de moeda usada nos quantum walks.
            """
        )
    )


def _section_rotor_math() -> None:
    text = dedent(
        """\
        📐 Matemática do rotor
        ----------------------
        A moeda é construída como

            U(n, θ) = exp(-i * θ/2 * (n · σ)),

        onde:
          • n = (nx, ny, nz) é um vetor unitário.
          • σ = (σx, σy, σz) são as matrizes de Pauli.
          • θ é o ângulo de rotação.

        A exponencial é avaliada via série (scaling & squaring), garantindo precisão ~1e-12.
        """
    )
    print(text)


def _section_hadamard_equivalence() -> None:
    theta = math.pi / 2
    unitary = coin_rotor(theta, axis=(0.0, 1.0, 0.0))
    corrected = unitary @ PAULI_Z
    diff = np.linalg.norm(corrected - HADAMARD)
    print("🎯 Rotor vs Hadamard (θ = π/2, eixo = Y)")
    _print_matrix("U(θ)", unitary)
    _print_matrix("U(θ) · Z", corrected)
    _print_matrix("Hadamard", HADAMARD)
    print(f"‖U·Z - H‖₂ = {diff:.3e}\n")


def _section_custom_angle() -> None:
    try:
        theta_deg = float(input("Informe o ângulo em graus (ex.: 45): ").strip() or "45")
    except ValueError:
        print("Valor inválido, usando 45°.\n")
        theta_deg = 45.0
    theta = math.radians(theta_deg)
    axis = (0.0, 1.0, 0.0)
    unitary = coin_rotor(theta, axis=axis)
    print(f"🔄 Rotor para θ = {theta_deg:.2f}° em torno de Y:")
    _print_matrix("U(θ)", unitary)
    for label, state in BASIS_STATES.items():
        out = unitary @ state
        print(f"U |{label}> = [{_format_complex(out[0,0])}, {_format_complex(out[1,0])}]ᵀ")
    print()


def _section_bloch_rotation() -> None:
    axis = np.array([0.0, 1.0, 0.0])
    theta = math.pi / 2
    initial_bloch = np.array([0.0, 0.0, 1.0])
    rho = rho_from_bloch(initial_bloch)
    unitary = coin_rotor(theta, axis=axis)
    rotated = unitary @ rho @ unitary.conj().T
    final_bloch = bloch_from_rho(rotated)
    print("🌐 Ação no vetor de Bloch |0⟩ (0,0,1):")
    print(f"Axis = {axis}, θ = π/2")
    print(f"Inicial: {initial_bloch}")
    print(f"Final:   {final_bloch}\n")


def _section_exit() -> None:
    print("Encerrando o portfólio. Até a próxima! 👋")


SECTIONS = {
    "1": ("Matemática do rotor", _section_rotor_math),
    "2": ("Comparação com Hadamard", _section_hadamard_equivalence),
    "3": ("Explorar ângulo personalizado", _section_custom_angle),
    "4": ("Rotação no Bloch", _section_bloch_rotation),
    "q": ("Sair", _section_exit),
}


def main() -> None:
    """Interactive walkthrough covering the GAQ rotor coin."""
    _intro()
    while True:
        print("Menu principal:")
        for key, (title, _) in SECTIONS.items():
            label = "Quit" if key == "q" else title
            print(f"  [{key}] {label}")
        choice = input("Selecione uma opção: ").strip().lower()
        if choice not in SECTIONS:
            print("Opção inválida.\n")
            continue
        title, action = SECTIONS[choice]
        print(f"\n=== {title} ===")
        action()
        if choice == "q":
            break


if __name__ == "__main__":  # pragma: no cover
    main()
