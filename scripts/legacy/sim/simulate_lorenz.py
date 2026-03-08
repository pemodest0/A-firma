"""Simula o sistema de Lorenz e retorna um DataFrame."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def generate_lorenz_series(
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    dt: float = 0.01,
    steps: int = 5000,
    x0: float = 1.0,
    y0: float = 1.0,
    z0: float = 1.0,
) -> pd.DataFrame:
    try:
        from scipy.integrate import solve_ivp
    except Exception as exc:
        raise RuntimeError("scipy é necessário para simular Lorenz.") from exc

    def lorenz(t, state):
        x, y, z = state
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    T = dt * (steps - 1)
    t_eval = np.linspace(0.0, T, steps)
    sol = solve_ivp(lorenz, (0.0, T), [x0, y0, z0], t_eval=t_eval, method="RK45")
    if not sol.success:
        raise RuntimeError("Falha na integração numérica do Lorenz.")

    return pd.DataFrame({"t": sol.t, "x": sol.y[0], "y": sol.y[1], "z": sol.y[2]})


def main() -> None:
    df = generate_lorenz_series()
    out = Path("results/lorenz_series.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Serie salva em {out}")


if __name__ == "__main__":
    main()
