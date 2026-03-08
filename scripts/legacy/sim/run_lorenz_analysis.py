"""Pipeline de analise do sistema de Lorenz."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import os

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except Exception:
    plt = None
    PdfPages = None

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.diagnostics.regime_labels import RegimeClassifier
from scripts.legacy.sim.simulate_lorenz import generate_lorenz_series


def plot_entropy_vs_tau(metrics: list[dict[str, float]], out_path: Path) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    grouped: dict[int, list[dict[str, float]]] = {}
    for row in metrics:
        grouped.setdefault(int(row["m"]), []).append(row)
    for m, rows in grouped.items():
        rows_sorted = sorted(rows, key=lambda r: r["tau"])
        taus = [r["tau"] for r in rows_sorted]
        ent = [r["entropy"] for r in rows_sorted]
        ax.plot(taus, ent, marker="o", label=f"m={m}")
    ax.set_xlabel("τ")
    ax.set_ylabel("Entropia")
    ax.set_title("Entropia vs τ")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_recurrence(embedded: np.ndarray, out_path: Path) -> None:
    if plt is None:
        return
    sample = embedded
    if sample.shape[0] > 600:
        idx = np.random.default_rng(42).choice(sample.shape[0], size=600, replace=False)
        sample = sample[idx]
    diffs = sample[:, None, :] - sample[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    eps = np.percentile(dists[np.triu_indices(dists.shape[0], k=1)], 10.0)
    rec = (dists <= eps).astype(float)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(rec, cmap="Greys", origin="lower")
    ax.set_title("Recurrence plot")
    ax.set_xlabel("t")
    ax.set_ylabel("t")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_report_md(out_dir: Path, config: dict) -> None:
    md_path = out_dir / "report_lorenz.md"
    lines = [
        "# Relatório – Sistema de Lorenz",
        "",
        "## Configuração",
        f"- σ={config['sigma']}, ρ={config['rho']}, β={config['beta']}",
        f"- dt={config['dt']}, steps={config['steps']}",
        f"- método: {config['method']}",
        f"- embedding: m={config['best_m']}, τ={config['best_tau']}",
        "",
        "## Gráficos",
        "![xv_regime](xv_regime_lorenz.png)",
        "![labels_over_time](labels_over_time_lorenz.png)",
        "![entropy_vs_tau](entropy_vs_tau_lorenz.png)",
        "![recurrence_plot](recurrence_plot_lorenz.png)",
        "![regime_map](regime_map_lorenz.png)",
        "![regime_3d](regime_3d_lorenz.png)",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def generate_report_pdf(out_dir: Path) -> None:
    if plt is None or PdfPages is None:
        return
    pdf_path = out_dir / "report_lorenz.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.05, 0.95, "Relatório – Sistema de Lorenz", va="top", fontsize=14)
        pdf.savefig(fig)
        plt.close(fig)

        for name in [
            "xv_regime_lorenz.png",
            "labels_over_time_lorenz.png",
            "entropy_vs_tau_lorenz.png",
            "recurrence_plot_lorenz.png",
            "regime_map_lorenz.png",
            "regime_3d_lorenz.png",
        ]:
            path = out_dir / name
            if not path.exists():
                continue
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.axis("off")
            ax.imshow(plt.imread(path))
            ax.set_title(name.replace("_", " ").replace(".png", ""))
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analise do sistema de Lorenz.")
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--rho", type=float, default=28.0)
    parser.add_argument("--beta", type=float, default=8.0 / 3.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--signal", type=str, default="x", choices=("x", "y", "z"))
    parser.add_argument("--method", type=str, default="hdbscan", choices=("hdbscan", "kmeans", "auto"))
    parser.add_argument("--system-type", type=str, default="lorenz")
    parser.add_argument("--outdir", type=str, default="results/lorenz")
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = generate_lorenz_series(args.sigma, args.rho, args.beta, args.dt, args.steps)
    df.to_csv(out_dir / "lorenz_series.csv", index=False)

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    z = df["z"].to_numpy()
    energy = x**2 + y**2 + z**2

    series = df[args.signal].to_numpy()
    classifier = RegimeClassifier(clustering_method=args.method)
    metrics = classifier.scan_embeddings(series, tau_range=range(3, 11), m_range=range(2, 4))
    best_m, best_tau = classifier.select_embedding(metrics, criterion="min_entropy")
    classifier.m = best_m
    classifier.tau = best_tau

    embedded = classifier.embed(series)
    velocity = classifier.compute_velocity(series)
    energy_aligned = energy[(best_m - 1) * best_tau : (best_m - 1) * best_tau + embedded.shape[0]]
    xyz_aligned = np.column_stack(
        [
            x[(best_m - 1) * best_tau : (best_m - 1) * best_tau + embedded.shape[0]],
            y[(best_m - 1) * best_tau : (best_m - 1) * best_tau + embedded.shape[0]],
            z[(best_m - 1) * best_tau : (best_m - 1) * best_tau + embedded.shape[0]],
        ]
    )
    features = {"velocity": velocity, "energy": energy_aligned, "xyz": xyz_aligned}
    cluster_labels = classifier.cluster_states(embedded, features)
    label_features = dict(features)
    system_type = args.system_type.strip().lower()
    if system_type in {"", "none", "generico"}:
        system_type = None
    if system_type:
        label_features["system_type"] = np.array([system_type], dtype=object)
    regimes = classifier.label_sequence(series, cluster_labels, embedded, label_features)

    label_names = np.unique(regimes)
    label_to_idx = {label: i for i, label in enumerate(label_names)}
    regime_codes = np.array([label_to_idx[label] for label in regimes], dtype=int)

    plot_entropy_vs_tau(metrics, out_dir / "entropy_vs_tau_lorenz.png")
    plot_recurrence(embedded, out_dir / "recurrence_plot_lorenz.png")

    if plt is not None:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(embedded[:, 0], velocity, c=regime_codes, s=6, cmap="tab10")
        ax.set_xlabel("x(t)")
        ax.set_ylabel("v(t)")
        ax.set_title("x(t) vs v(t) por regime")
        fig.tight_layout()
        fig.savefig(out_dir / "xv_regime_lorenz.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 4))
        idx_start = (best_m - 1) * best_tau
        time = df["t"].to_numpy()[idx_start : idx_start + embedded.shape[0]]
        ax.plot(time, series[idx_start : idx_start + embedded.shape[0]], color="#64748b")
        ax.scatter(time, series[idx_start : idx_start + embedded.shape[0]], c=regime_codes, s=8, cmap="tab10")
        ax.set_title("Labels ao longo do tempo")
        ax.set_xlabel("t")
        fig.tight_layout()
        fig.savefig(out_dir / "labels_over_time_lorenz.png", dpi=150)
        plt.close(fig)

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(xyz_aligned[:, 0], xyz_aligned[:, 1], xyz_aligned[:, 2], c=regime_codes, s=6, cmap="tab10")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("Lorenz 3D por regime")
        fig.tight_layout()
        fig.savefig(out_dir / "regime_3d_lorenz.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(embedded[:, 0], embedded[:, 1], c=regime_codes, s=6, cmap="tab10")
        ax.set_xlabel("x(t)")
        ax.set_ylabel(f"x(t-{best_tau})")
        ax.set_title("Regime map (embedding)")
        fig.tight_layout()
        fig.savefig(out_dir / "regime_map_lorenz.png", dpi=150)
        plt.close(fig)

    summary_rows = []
    transitions = int(np.sum(regimes[1:] != regimes[:-1])) if regimes.size > 1 else 0
    for label in label_names:
        mask = regimes == label
        summary_rows.append(
            {
                "regime": str(label),
                "count": int(np.sum(mask)),
                "percent": float(np.mean(mask) * 100.0),
                "x_mean": float(np.mean(xyz_aligned[:, 0][mask])),
                "y_mean": float(np.mean(xyz_aligned[:, 1][mask])),
                "z_mean": float(np.mean(xyz_aligned[:, 2][mask])),
                "energy_mean": float(np.mean(energy_aligned[mask])),
                "transitions": transitions,
            }
        )
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary_lorenz.csv", index=False)

    config = {
        "sigma": args.sigma,
        "rho": args.rho,
        "beta": args.beta,
        "dt": args.dt,
        "steps": args.steps,
        "method": args.method,
        "best_m": best_m,
        "best_tau": best_tau,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    write_report_md(out_dir, config)
    generate_report_pdf(out_dir)


if __name__ == "__main__":
    main()
