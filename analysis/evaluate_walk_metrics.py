"""Run geometric walk and report calibration metrics (spectral gap, diffusion, autocorr)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gaq.analysis.metrics import autocorrelation, hamming_diffusion, probability_gap, spectral_gap_unbiased
from gaq.examples.geom_walk_optimizer import (
    cost_function,
    generate_edges,
    generate_vertices,
    simulate_walk,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    verts = generate_vertices(args.n)
    edges = generate_edges(verts)
    cost_fn = cost_function("Ising (J,h)", args.n, seed=args.seed, params={"J": 1.0, "h": 0.1})
    energies = np.array([cost_fn(v) for v in verts])
    energies = (energies - energies.min()) / (np.ptp(energies) + 1e-9)
    history, energy_curve, hitting = simulate_walk(
        energies,
        edges,
        steps=args.steps,
        alpha=args.alpha,
        beta=args.beta,
        noise=args.noise,
        hitting_threshold=0.1,
    )

    evals, gap = spectral_gap_unbiased(args.n)
    print(f"Spectral gap (unbiased walk Q_{args.n}): {gap:.6f}")

    diffusion = hamming_diffusion(history)
    autocorr = autocorrelation(np.array(energy_curve))
    probs_gap = probability_gap(history)

    outdir = Path("analysis/metrics_output")
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "diffusion.npy", diffusion)
    np.save(outdir / "autocorr.npy", autocorr)
    np.save(outdir / "prob_gap.npy", probs_gap)
    print(f"Metrics saved to {outdir}")
    print(f"Hitting step: {hitting}")


if __name__ == "__main__":
    main()
