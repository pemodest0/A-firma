"""Gera sÃ©ries sintÃ©ticas pesadas para balancear classes raras."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.diagnostics.regime_labels import RegimeClassifier


def write_summary_for_label(series: np.ndarray, label: str, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rc = RegimeClassifier(tau=1, m=2, clustering_method="hdbscan")
    embedded = rc.embed(series)
    velocity = rc.compute_velocity(series)
    energy = rc.compute_energy(embedded[:, 0], velocity)
    local = rc.compute_local_features(embedded[:, 0])
    labels = np.full(embedded.shape[0], label, dtype=object)
    summary = rc._build_summary(embedded, velocity, energy, labels, extra_features=local)
    pd.DataFrame(summary).to_csv(out_dir / f"summary_{name}.csv", index=False)
    pd.DataFrame({"x": series}).to_csv(out_dir / f"{name}.csv", index=False)


def gen_periodic(n: int, freq: float, phase: float, noise: float) -> np.ndarray:
    t = np.linspace(0, 10, n)
    x = np.sin(2 * np.pi * freq * t + phase)
    if noise > 0:
        x = x + np.random.default_rng(42).normal(0, noise, size=n)
    return x


def gen_damped(n: int, freq: float, decay: float, noise: float) -> np.ndarray:
    t = np.linspace(0, 10, n)
    x = np.exp(-decay * t) * np.sin(2 * np.pi * freq * t)
    if noise > 0:
        x = x + np.random.default_rng(42).normal(0, noise, size=n)
    return x


def gen_rotation(n: int, speed: float, noise: float) -> np.ndarray:
    t = np.linspace(0, 10, n)
    phase = speed * t
    x = np.sin(phase)
    if noise > 0:
        x = x + np.random.default_rng(42).normal(0, noise, size=n)
    return x


def gen_quasi(n: int, f1: float, f2: float, noise: float) -> np.ndarray:
    t = np.linspace(0, 10, n)
    x = 0.6 * np.sin(2 * np.pi * f1 * t) + 0.4 * np.sin(2 * np.pi * f2 * t)
    if noise > 0:
        x = x + np.random.default_rng(42).normal(0, noise, size=n)
    return x


def gen_chirp(n: int, f0: float, f1: float, noise: float) -> np.ndarray:
    t = np.linspace(0, 10, n)
    k = (f1 - f0) / (t[-1] - t[0])
    phase = 2 * np.pi * (f0 * t + 0.5 * k * t**2)
    x = np.sin(phase)
    if noise > 0:
        x = x + np.random.default_rng(42).normal(0, noise, size=n)
    return x


def logistic_map(r: float, n: int, x0: float = 0.2, discard: int = 500) -> np.ndarray:
    x = np.zeros(n + discard)
    x[0] = x0
    for i in range(1, n + discard):
        x[i] = r * x[i - 1] * (1 - x[i - 1])
    return x[discard:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Gerar sÃ©ries sintÃ©ticas pesadas.")
    parser.add_argument("--outdir", type=str, default="results/phase1_heavy_synthetic")
    parser.add_argument("--n", type=int, default=5000, help="Tamanho de cada sÃ©rie.")
    parser.add_argument("--per-class", type=int, default=20, help="Qtd de sÃ©ries por classe.")
    parser.add_argument("--noise", type=float, default=0.01)
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    n = args.n
    per_class = args.per_class
    noise = args.noise
    total = per_class * 6
    iterator = tqdm(total=total, desc="Heavy synthetic", unit="series") if tqdm else None

    for i in range(per_class):
        freq = rng.uniform(0.5, 2.0)
        phase = rng.uniform(0, 2 * np.pi)
        series = gen_periodic(n, freq, phase, noise)
        write_summary_for_label(series, "periodico", out_dir, f"periodico_{i}")
        if iterator:
            iterator.update(1)

    for i in range(per_class):
        freq = rng.uniform(0.5, 2.0)
        decay = rng.uniform(0.05, 0.3)
        series = gen_damped(n, freq, decay, noise)
        write_summary_for_label(series, "oscilatorio", out_dir, f"oscilatorio_{i}")
        if iterator:
            iterator.update(1)

    for i in range(per_class):
        speed = rng.uniform(4.0, 10.0)
        series = gen_rotation(n, speed, noise)
        write_summary_for_label(series, "rotation", out_dir, f"rotation_{i}")
        if iterator:
            iterator.update(1)

    for i in range(per_class):
        f1 = rng.uniform(0.5, 1.5)
        f2 = rng.uniform(1.6, 3.0)
        series = gen_quasi(n, f1, f2, noise)
        write_summary_for_label(series, "libration", out_dir, f"libration_{i}")
        if iterator:
            iterator.update(1)

    for i in range(per_class):
        series = gen_chirp(n, 0.5, 3.0, noise)
        write_summary_for_label(series, "transicao", out_dir, f"transicao_{i}")
        if iterator:
            iterator.update(1)

    for i in range(per_class):
        r = rng.uniform(3.7, 3.99)
        series = logistic_map(r, n)
        write_summary_for_label(series, "caotico", out_dir, f"caotico_{i}")
        if iterator:
            iterator.update(1)

    if iterator:
        iterator.close()


if __name__ == "__main__":
    main()

