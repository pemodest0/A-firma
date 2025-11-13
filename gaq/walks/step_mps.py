"""Single-step walker interface for an MPS backend (stub)."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

__all__ = ["StepResult", "step"]


class StepResult(Protocol):
    """Minimal protocol for objects returning after a walk step."""

    def state(self) -> object:  # pragma: no cover - structural typing only
        """Return the backend-specific state."""

    def diagnostics(self) -> dict[str, float]:  # pragma: no cover
        """Return optional diagnostics about the step."""


@dataclass(frozen=True)
class _SimpleStepResult:
    _state: object
    _diagnostics: dict[str, float]

    def state(self) -> object:
        return self._state

    def diagnostics(self) -> dict[str, float]:
        return dict(self._diagnostics)


def _require_callable(obj, name: str) -> Callable:
    fn = getattr(obj, name, None)
    if fn is None:
        raise AttributeError(f"Backend missing required method '{name}'.")
    return fn


def step(
    state_mps: object | None,
    coin_unitary: np.ndarray,
    backend,
    *,
    coin_sites: tuple[int, ...] = (0,),
    initial_state: np.ndarray | None = None,
    shift_fn: Callable[[object], object] | None = None,
    max_bond: int | None = None,
    truncation_tol: float = 1e-9,
) -> StepResult:
    """Apply a coined walk step using the provided backend hooks."""
    apply_gate = _require_callable(backend, "apply_gate_mps")
    truncate = _require_callable(backend, "truncate")
    state = state_mps
    if state is None:
        if initial_state is None:
            raise ValueError("Provide `initial_state` when state_mps is None.")
        to_mps = _require_callable(backend, "to_mps")
        state = to_mps(initial_state)
    matrix = np.asarray(coin_unitary, dtype=complex)
    if matrix.shape != (2, 2):
        raise ValueError("coin_unitary must be 2x2.")
    state = apply_gate(state, matrix, coin_sites)
    if shift_fn is not None:
        state = shift_fn(state)
    diagnostics: dict[str, float] = {}
    if max_bond is not None:
        state = truncate(state, max_bond, tol=truncation_tol)
        diagnostics["max_bond"] = float(max_bond)
        diagnostics["truncation_tol"] = truncation_tol
    return _SimpleStepResult(state, diagnostics)
