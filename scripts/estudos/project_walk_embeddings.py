"""Generate PCA/t-SNE/Hyperbolic projections for GAQ geometric walk snapshots."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gaq.analysis.projections import (
    hypercube_hyperbolic_embedding,
    pca_embedding,
    tsne_embedding,
)
from gaq.examples.geom_walk_optimizer import (
    cost_function,
    generate_edges,
    generate_vertices,
    simulate_walk,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
EMBED_DIR = REPO_ROOT / "dados" / "benchmarks" / "embed_output"


def snapshot_probs(history, step: int) -> np.ndarray:
    step = min(max(step, 0), len(history) - 1)
    return history[step]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=4, help="Bits of the hypercube")
    parser.add_argument("--step", type=int, default=0, help="Snapshot step index")
    parser.add_argument("--cost", default="Ising (J,h)", help="Energy function")
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    verts = generate_vertices(args.n)
    params = {"J": 1.0, "h": 0.0} if "Ising" in args.cost else {}
    cost_fn = cost_function(args.cost, args.n, seed=args.seed, params=params)
    energies = np.array([cost_fn(v) for v in verts])
    energies = (energies - energies.min()) / (np.ptp(energies) + 1e-9)
    edges = generate_edges(verts)
    history, _, _ = simulate_walk(
        energies,
        edges=edges,
        steps=args.steps,
        alpha=args.alpha,
        beta=args.beta,
        noise=args.noise,
        hitting_threshold=0.1,
    )
    probs = snapshot_probs(history, args.step)

    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    for embedding_fn in (pca_embedding, tsne_embedding):
        emb = embedding_fn(verts, probs)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(emb.coordinates[:, 0], emb.coordinates[:, 1], s=5, alpha=0.4)
        ax.set_title(f"{emb.method} snapshot step {args.step}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.tight_layout()
        fig.savefig(EMBED_DIR / f"{emb.method.lower()}_step{args.step}.png", dpi=200)
        plt.close(fig)

    hyp = hypercube_hyperbolic_embedding(verts)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(hyp.coordinates[:, 0], hyp.coordinates[:, 1], s=10, alpha=0.7)
    ax.set_title("Hyperbolic embedding (deterministic)")
    ax.set_aspect("equal")
    fig.savefig(EMBED_DIR / "hyperbolic.png", dpi=200)
    plt.close(fig)

    print(f"Embeddings saved under {EMBED_DIR}")


if __name__ == "__main__":
    main()
