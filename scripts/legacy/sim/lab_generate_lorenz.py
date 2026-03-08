import json
from pathlib import Path

import numpy as np


def lorenz_step(state, sigma, rho, beta):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz], dtype=float)


def integrate_lorenz(n_points, dt, sigma, rho, beta, initial_state):
    states = np.zeros((n_points, 3), dtype=float)
    state = np.array(initial_state, dtype=float)
    for i in range(n_points):
        states[i] = state
        k1 = lorenz_step(state, sigma, rho, beta)
        k2 = lorenz_step(state + 0.5 * dt * k1, sigma, rho, beta)
        k3 = lorenz_step(state + 0.5 * dt * k2, sigma, rho, beta)
        k4 = lorenz_step(state + dt * k3, sigma, rho, beta)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return states


def main():
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    dt = 0.01
    n_points = 55000
    discard = 5000

    states = integrate_lorenz(
        n_points=n_points,
        dt=dt,
        sigma=sigma,
        rho=rho,
        beta=beta,
        initial_state=(1.0, 1.0, 1.0),
    )

    states = states[discard:]
    n_final = states.shape[0]
    t = (np.arange(n_final) * dt).tolist()

    base_dir = Path(__file__).resolve().parents[1]
    out_dir = base_dir / "website" / "assets" / "lab_lorenz"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "dt": dt,
        "sigma": sigma,
        "rho": rho,
        "beta": beta,
        "n_points": int(n_points),
        "discard": int(discard),
    }

    lorenz_true = {
        "meta": meta,
        "t": t,
        "x": states[:, 0].round(6).tolist(),
        "y": states[:, 1].round(6).tolist(),
        "z": states[:, 2].round(6).tolist(),
    }
    lorenz_obs = {
        "meta": meta,
        "t": t,
        "obs": states[:, 0].round(6).tolist(),
    }

    (out_dir / "lorenz_true.json").write_text(
        json.dumps(lorenz_true, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "lorenz_observed.json").write_text(
        json.dumps(lorenz_obs, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
