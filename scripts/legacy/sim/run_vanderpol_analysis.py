"""Pipeline completo para simulação e análise do oscilador de Van der Pol."""

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
from sklearn.decomposition import PCA


def simulate_vanderpol(mu: float, dt: float, steps: int, x0: float, y0: float) -> pd.DataFrame:
    try:
        from scipy.integrate import solve_ivp
    except Exception as exc:
        raise RuntimeError("scipy é necessário para simular o Van der Pol.") from exc

    def system(t, y):
        x, v = y
        dx = v
        dv = mu * (1 - x**2) * v - x
        return [dx, dv]

    T = dt * (steps - 1)
    t_eval = np.linspace(0.0, T, steps)
    sol = solve_ivp(system, (0.0, T), [x0, y0], t_eval=t_eval, method="RK45")
    if not sol.success:
        raise RuntimeError("Falha na integração numérica do Van der Pol.")

    return pd.DataFrame({"t": sol.t, "x": sol.y[0], "y": sol.y[1]})


def compute_energy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kinetic = 0.5 * y**2
    potential = 0.5 * x**2
    return kinetic, potential


def label_regimes(cluster_labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(cluster_labels)
    unique, counts = np.unique(labels, return_counts=True)
    total = counts.sum() if counts.size else 0
    mapping = {}
    if total == 0:
        return np.array([], dtype=object)
    order = sorted(zip(unique, counts), key=lambda item: item[1])
    mapping[order[0][0]] = "transicao"
    mapping[order[-1][0]] = "periodico"
    for label, _ in order[1:-1]:
        mapping[label] = "caotico"
    return np.array([mapping.get(lbl, "caotico") for lbl in labels], dtype=object)


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
    md_path = out_dir / "report_vanderpol.md"
    lines = [
        "# Relatório – Oscilador de Van der Pol",
        "",
        "## Configuração",
        f"- μ={config['mu']}",
        f"- dt={config['dt']}, steps={config['steps']}",
        f"- x(0)={config['x0']}, y(0)={config['y0']}",
        f"- método: {config['method']}",
        f"- embedding: m={config['best_m']}, τ={config['best_tau']}",
        "",
        "## Gráficos",
        "![xv_regime](xv_regime_vanderpol.png)",
        "![labels_over_time](labels_over_time_vanderpol.png)",
        "![entropy_vs_tau](entropy_vs_tau_vanderpol.png)",
        "![recurrence_plot](recurrence_plot_vanderpol.png)",
        "![regime_map](regime_map_vanderpol.png)",
        "![regime_map_pca](regime_map_pca_vanderpol.png)",
        "",
        "## Análises complementares",
        "![xv_colored_by_time](xv_colored_by_time_vanderpol.png)",
        "![energy_by_regime](energy_by_regime_vanderpol.png)",
        "![potential_energy_by_regime](potential_energy_by_regime_vanderpol.png)",
        "![regime_recurrence_plot](regime_recurrence_plot_vanderpol.png)",
        "![takens_3d_colored](takens_3d_colored_vanderpol.png)",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def generate_report_pdf(out_dir: Path) -> None:
    if plt is None or PdfPages is None:
        return
    pdf_path = out_dir / "report_vanderpol.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.05, 0.95, "Relatório – Oscilador de Van der Pol", va="top", fontsize=14)
        pdf.savefig(fig)
        plt.close(fig)

        for name in [
            "xv_regime_vanderpol.png",
            "labels_over_time_vanderpol.png",
            "entropy_vs_tau_vanderpol.png",
            "recurrence_plot_vanderpol.png",
            "regime_map_vanderpol.png",
            "regime_map_pca_vanderpol.png",
            "xv_colored_by_time_vanderpol.png",
            "energy_by_regime_vanderpol.png",
            "potential_energy_by_regime_vanderpol.png",
            "regime_recurrence_plot_vanderpol.png",
            "takens_3d_colored_vanderpol.png",
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
    parser = argparse.ArgumentParser(description="Analise do oscilador de Van der Pol.")
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--x0", type=float, default=0.1)
    parser.add_argument("--y0", type=float, default=0.1)
    parser.add_argument("--outdir", type=str, default="results/vanderpol_hdbscan")
    parser.add_argument("--method", type=str, default="hdbscan", choices=("hdbscan", "auto"))
    parser.add_argument("--label-mode", type=str, default="heuristic", choices=("heuristic", "auto"))
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = simulate_vanderpol(args.mu, args.dt, args.steps, args.x0, args.y0)
    df.to_csv(out_dir / "vanderpol_series.csv", index=False)

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    kinetic, potential = compute_energy(x, y)

    classifier = RegimeClassifier(tau=1, m=2, clustering_method=args.method)
    metrics = classifier.scan_embeddings(x, tau_range=range(1, 11), m_range=range(2, 6))
    best_m, best_tau = 2, 1

    embedded = classifier.embed(x)
    velocity = classifier.compute_velocity(x)
    kinetic_aligned = kinetic[(best_m - 1) * best_tau : (best_m - 1) * best_tau + embedded.shape[0]]
    potential_aligned = potential[(best_m - 1) * best_tau : (best_m - 1) * best_tau + embedded.shape[0]]
    energy_aligned = kinetic_aligned + potential_aligned
    features = {"velocity": velocity, "energy": energy_aligned}
    cluster_labels = classifier.cluster_states(embedded, features)

    entropy = float(classifier.shannon_entropy(embedded))
    rr = float(classifier.recurrence_rate(embedded))
    if args.label_mode == "auto":
        features["system_type"] = np.array(["auto"], dtype=object)
        regimes = classifier.label_sequence(x, cluster_labels, embedded, features)
    else:
        regimes = label_regimes(cluster_labels)

    label_names = np.unique(regimes)
    label_to_idx = {label: i for i, label in enumerate(label_names)}
    regime_codes = np.array([label_to_idx[label] for label in regimes], dtype=int)

    plot_entropy_vs_tau(metrics, out_dir / "entropy_vs_tau_vanderpol.png")
    plot_recurrence(embedded, out_dir / "recurrence_plot_vanderpol.png")

    if plt is not None and embedded.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(embedded[:, 0], embedded[:, 1], c=regime_codes, s=6, cmap="tab10")
        ax.set_xlabel("x(t)")
        ax.set_ylabel(f"x(t-{best_tau})")
        ax.set_title("Regime map (embedding)")
        fig.tight_layout()
        fig.savefig(out_dir / "regime_map_vanderpol.png", dpi=150)
        plt.close(fig)

    if plt is not None:
        fig, ax = plt.subplots(figsize=(9, 4))
        idx_start = (best_m - 1) * best_tau
        time = df["t"].to_numpy()[idx_start : idx_start + embedded.shape[0]]
        ax.plot(time, x[idx_start : idx_start + embedded.shape[0]], color="#64748b")
        scatter = ax.scatter(
            time,
            x[idx_start : idx_start + embedded.shape[0]],
            c=regime_codes,
            s=8,
            cmap="tab10",
        )
        ax.set_title("Labels ao longo do tempo")
        ax.set_xlabel("t")
        ax.legend(
            handles=scatter.legend_elements()[0],
            labels=[str(name) for name in label_names],
            title="Regimes",
            fontsize=8,
        )
        fig.tight_layout()
        fig.savefig(out_dir / "labels_over_time_vanderpol.png", dpi=150)
        plt.close(fig)

    if plt is not None:
        fig, ax = plt.subplots(figsize=(7, 5))
        scatter = ax.scatter(embedded[:, 0], velocity, c=regime_codes, s=6, cmap="tab10")
        ax.set_xlabel("x(t)")
        ax.set_ylabel("v(t)")
        ax.set_title("x(t) vs v(t) por regime")
        ax.legend(
            handles=scatter.legend_elements()[0],
            labels=[str(name) for name in label_names],
            title="Regimes",
            fontsize=8,
        )
        fig.tight_layout()
        fig.savefig(out_dir / "xv_regime_vanderpol.png", dpi=150)
        plt.close(fig)

    if plt is not None:
        fig, ax = plt.subplots(figsize=(7, 5))
        idx_start = (best_m - 1) * best_tau
        time = df["t"].to_numpy()[idx_start : idx_start + embedded.shape[0]]
        scatter = ax.scatter(
            embedded[:, 0],
            velocity,
            c=time,
            s=6,
            cmap="viridis",
        )
        ax.set_xlabel("x(t)")
        ax.set_ylabel("v(t)")
        ax.set_title("Espaco de fase colorido pelo tempo")
        fig.colorbar(scatter, ax=ax, label="t")
        fig.tight_layout()
        fig.savefig(out_dir / "xv_colored_by_time_vanderpol.png", dpi=150)
        plt.close(fig)

    if plt is not None:
        fig, ax = plt.subplots(figsize=(9, 4))
        idx_start = (best_m - 1) * best_tau
        time = df["t"].to_numpy()[idx_start : idx_start + embedded.shape[0]]
        for label in label_names:
            mask = regimes == label
            series_energy = np.where(mask, kinetic_aligned, np.nan)
            ax.plot(time, series_energy, label=str(label))
        ax.set_xlabel("t")
        ax.set_ylabel("Energia cinetica")
        ax.set_title("Energia cinetica por regime")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "energy_by_regime_vanderpol.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 4))
        for label in label_names:
            mask = regimes == label
            series_energy = np.where(mask, potential_aligned, np.nan)
            ax.plot(time, series_energy, label=str(label))
        ax.set_xlabel("t")
        ax.set_ylabel("Energia potencial")
        ax.set_title("Energia potencial por regime")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "potential_energy_by_regime_vanderpol.png", dpi=150)
        plt.close(fig)

    if plt is not None:
        reg_codes = regime_codes
        max_points = 600
        if reg_codes.size > max_points:
            idx = np.linspace(0, reg_codes.size - 1, max_points).astype(int)
            reg_codes = reg_codes[idx]
        rec = (reg_codes[:, None] == reg_codes[None, :]).astype(float)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(rec, cmap="binary", origin="lower")
        ax.set_title("Recorrencia de regimes")
        ax.set_xlabel("t")
        ax.set_ylabel("t")
        fig.tight_layout()
        fig.savefig(out_dir / "regime_recurrence_plot_vanderpol.png", dpi=150)
        plt.close(fig)

    if plt is not None and embedded.shape[1] >= 3:
        try:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(
                embedded[:, 0],
                embedded[:, 1],
                embedded[:, 2],
                c=regime_codes,
                cmap="tab10",
                s=6,
                alpha=0.7,
            )
            ax.set_xlabel("x(t)")
            ax.set_ylabel(f"x(t-{best_tau})")
            ax.set_zlabel(f"x(t-{2 * best_tau})")
            ax.set_title("Takens 3D colorido por regime")
            fig.tight_layout()
            fig.savefig(out_dir / "takens_3d_colored_vanderpol.png", dpi=150)
            plt.close(fig)
        except Exception:
            pass

    if plt is not None and embedded.shape[0] >= 5:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(embedded)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(coords[:, 0], coords[:, 1], c=regime_codes, s=6, cmap="tab10")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Regime map (PCA 2D)")
        fig.tight_layout()
        fig.savefig(out_dir / "regime_map_pca_vanderpol.png", dpi=150)
        plt.close(fig)

    if plt is not None:
        dE = np.gradient(energy_aligned, args.dt)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(energy_aligned, dE, c=regime_codes, s=6, cmap="tab10")
        ax.set_xlabel("E")
        ax.set_ylabel("dE/dt")
        ax.set_title("Mapa de regimes (E vs dE/dt)")
        fig.tight_layout()
        fig.savefig(out_dir / "regime_map_vanderpol.png", dpi=150)
        plt.close(fig)

    summary_rows = []
    transitions = int(np.sum(regimes[1:] != regimes[:-1])) if regimes.size > 1 else 0
    x_aligned = x[(best_m - 1) * best_tau : (best_m - 1) * best_tau + embedded.shape[0]]
    y_aligned = y[(best_m - 1) * best_tau : (best_m - 1) * best_tau + embedded.shape[0]]
    for label in label_names:
        mask = regimes == label
        summary_rows.append(
            {
                "regime": str(label),
                "count": int(np.sum(mask)),
                "percent": float(np.mean(mask) * 100.0),
                "x_mean": float(np.mean(x_aligned[mask])),
                "y_mean": float(np.mean(y_aligned[mask])),
                "energy_mean": float(np.mean(energy_aligned[mask])),
                "kinetic_mean": float(np.mean(kinetic_aligned[mask])),
                "potential_mean": float(np.mean(potential_aligned[mask])),
                "transitions": transitions,
            }
        )
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary_vanderpol.csv", index=False)

    config = {
        "mu": args.mu,
        "dt": args.dt,
        "steps": args.steps,
        "x0": args.x0,
        "y0": args.y0,
        "method": args.method,
        "best_m": best_m,
        "best_tau": best_tau,
        "entropy": entropy,
        "recurrence_rate": rr,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    write_report_md(out_dir, config)
    generate_report_pdf(out_dir)


if __name__ == "__main__":
    main()

