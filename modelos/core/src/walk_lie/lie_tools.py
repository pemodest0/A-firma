#!/usr/bin/env python3
"""Ferramentas de álgebra de Lie para coins SU(2)."""
from __future__ import annotations

import numpy as np

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
PAULI = [SIGMA_X, SIGMA_Y, SIGMA_Z]


def su2_generator(theta: np.ndarray) -> np.ndarray:
    if theta.shape != (3,):
        raise ValueError("theta deve ter 3 componentes")
    G = sum(t * sigma for t, sigma in zip(theta, PAULI))
    return 1j * G  # geradores anti-hermitianos


def su2_unitary(theta: np.ndarray) -> np.ndarray:
    G = su2_generator(theta)
    return np.linalg.expm(G)


def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B - B @ A


def lie_penalty(thetas: np.ndarray) -> float:
    gens = [su2_generator(theta) for theta in thetas]
    penalty = 0.0
    for i in range(len(gens)):
        for j in range(i + 1, len(gens)):
            C = commutator(gens[i], gens[j])
            penalty += np.linalg.norm(C, ord="fro") ** 2
    return float(penalty)
