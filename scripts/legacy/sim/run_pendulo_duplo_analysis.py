#!/usr/bin/env python3
"""Pipeline completo para simulaÃ§Ã£o e anÃ¡lise do pÃªndulo duplo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import os

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


def simulate_double_pendulum(
    m1: float,
    m2: float,
    L1: float,
    L2: float,
    g: float,
    theta1: float,
    theta2: float,
    omega1: float,
    omega2: float,
    dt: float,
    steps: int,
) -> pd.DataFrame:
    try:
        from scipy.integrate import solve_ivp
    except Exception as exc:
        raise RuntimeError("scipy Ã© necessÃ¡rio para simulaÃ§Ã£o do pÃªndulo duplo.") from exc

    def pendulum_system(t, y):
        th1, w1, th2, w2 = y
        c = np.cos(th1 - th2)
        s = np.sin(th1 - th2)
        delta = m1 + m2 * s**2
        w1_dot = (
            m2 * g * np.sin(th2) * c
            - m2 * s * (L1 * w1**2 * c + L2 * w2**2)
            - (m1 + m2) * g * np.sin(th1)
        ) / (L1 * delta)
        w2_dot = (
            (m1 + m2) * (L1 * w1**2 * s - g * np.sin(th2) + g * np.sin(th1) * c)
            + m2 * L2 * w2**2 * s * c
        ) / (L2 * delta)
        return [w1, w1_dot, w2, w2_dot]

    T = dt * (steps - 1)
    t_eval = np.linspace(0.0, T, steps)
    y0 = [theta1, omega1, theta2, omega2]
    sol = solve_ivp(pendulum_system, (0.0, T), y0, t_eval=t_eval, method="RK45")
    if not sol.success:
        raise RuntimeError("Falha na integraÃ§Ã£o numÃ©rica do pÃªndulo duplo.")

    df = pd.DataFrame(
        {
            "t": sol.t,
            "theta1": sol.y[0],
            "omega1": sol.y[1],
            "theta2": sol.y[2],
            "omega2": sol.y[3],
        }
    )
    return df


def compute_energy(df: pd.DataFrame, m1: float, m2: float, L1: float, L2: float, g: float) -> np.ndarray:
    v1 = L1 * df["omega1"].to_numpy()
    v2 = L2 * df["omega2"].to_numpy()
    th1 = df["theta1"].to_numpy()
    th2 = df["theta2"].to_numpy()
    kinetic = 0.5 * m1 * v1**2 + 0.5 * m2 * v2**2
    potential = m1 * g * (1 - np.cos(th1)) + m2 * g * (1 - np.cos(th2))
    return kinetic + potential


def compute_kinetic(df: pd.DataFrame, m1: float, m2: float, L1: float, L2: float) -> np.ndarray:
    v1 = L1 * df["omega1"].to_numpy()
    v2 = L2 * df["omega2"].to_numpy()
    kinetic = 0.5 * m1 * v1**2 + 0.5 * m2 * v2**2
    return kinetic


def label_regimes(entropy: float, rr: float, energy: np.ndarray) -> np.ndarray:
    e = np.asarray(energy)
    if e.size == 0:
        return np.array([], dtype=object)
    low_thr = np.nanpercentile(e, 30)
    high_thr = np.nanpercentile(e, 70)
    labels = np.empty(e.shape[0], dtype=object)
    for i, val in enumerate(e):
        if val <= low_thr and entropy < 1.5 and rr > 0.2:
            labels[i] = "coerente"
        elif val >= high_thr and entropy > 1.5 and rr < 0.2:
            labels[i] = "caotico"
        else:
            labels[i] = "transicao"
    return labels


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
    ax.set_xlabel("Ï„")
    ax.set_ylabel("Entropia")
    ax.set_title("Entropia vs Ï„")
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


def generate_report_pdf(out_dir: Path, config: dict, plots: dict) -> None:
    if plt is None or PdfPages is None:
        return
    pdf_path = out_dir / "report_pendulo_duplo.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        text = (
            "RelatÃ³rio - PÃªndulo Duplo\n\n"
            "Resumo: simulaÃ§Ã£o numÃ©rica com integraÃ§Ã£o RK45 e anÃ¡lise de regimes.\n"
            f"ParÃ¢metros: m1={config['m1']}, m2={config['m2']}, "
            f"L1={config['L1']}, L2={config['L2']}, g={config['g']}\n"
            f"dt={config['dt']}, steps={config['steps']}\n"
        )
        ax.text(0.05, 0.95, text, va="top", fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)

        for name in [
            "labels_over_time.png",
            "xv_regime.png",
            "entropy_vs_tau.png",
            "recurrence_plot.png",
            "regime_map.png",
            "xv_colored_by_time.png",
            "energy_by_regime.png",
            "regime_recurrence_plot.png",
            "takens_3d_colored.png",
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


def write_report_md(out_dir: Path, config: dict) -> None:
    md_path = out_dir / "report_pendulo_duplo.md"
    lines = [
        "# RelatÃ³rio - PÃªndulo Duplo",
        "",
        "## ConfiguraÃ§Ã£o",
        f"- m1={config['m1']}, m2={config['m2']}, L1={config['L1']}, L2={config['L2']}, g={config['g']}",
        f"- theta1={config['theta1']}, theta2={config['theta2']}, omega1={config['omega1']}, omega2={config['omega2']}",
        f"- dt={config['dt']}, steps={config['steps']}",
        f"- sÃ©rie principal: {config['series']}",
        f"- mÃ©todo de clustering: {config['method']}",
        f"- melhor embedding: m={config['best_m']} tau={config['best_tau']}",
        f"- entropia global: {config['entropy']:.4f}",
        f"- recurrence rate global: {config['recurrence_rate']:.4f}",
        "",
        "## GrÃ¡ficos principais",
        "![labels_over_time](labels_over_time.png)",
        "![xv_regime](xv_regime.png)",
        "![entropy_vs_tau](entropy_vs_tau.png)",
        "![recurrence_plot](recurrence_plot.png)",
        "![regime_map](regime_map.png)",
        "",
        "## AnÃ¡lises complementares",
        "",
        "### EspaÃ§o de fase colorido por tempo",
        "![xv_colored_by_time](xv_colored_by_time.png)",
        "Mostra a progressÃ£o temporal dos estados no espaÃ§o de fase.",
        "",
        "### Energia cinÃ©tica por regime",
        "![energy_by_regime](energy_by_regime.png)",
        "Curvas de energia cinÃ©tica separadas por regime (coerente/caÃ³tico/transiÃ§Ã£o).",
        "",
        "### RecorrÃªncia de regimes",
        "![regime_recurrence_plot](regime_recurrence_plot.png)",
        "Matriz binÃ¡ria indicando quando o regime atual coincide com o passado.",
        "",
        "### Embedding Takens 3D colorido",
        "![takens_3d_colored](takens_3d_colored.png)",
        "VisualizaÃ§Ã£o da estrutura atratora no espaÃ§o reconstruÃ­do.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analise do pendulo duplo.")
    parser.add_argument("--m1", type=float, default=1.0)
    parser.add_argument("--m2", type=float, default=1.0)
    parser.add_argument("--L1", type=float, default=1.0)
    parser.add_argument("--L2", type=float, default=1.0)
    parser.add_argument("--g", type=float, default=9.81)
    parser.add_argument("--theta1", type=float, default=0.5)
    parser.add_argument("--theta2", type=float, default=0.1)
    parser.add_argument("--omega1", type=float, default=0.0)
    parser.add_argument("--omega2", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--series", type=str, default="x2", choices=("x1", "x2", "xsum"))
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--outdir", type=str, default="results/pendulo_duplo")
    parser.add_argument("--method", type=str, default="hdbscan", choices=("hdbscan", "kmeans", "dbscan"))
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = simulate_double_pendulum(
        m1=args.m1,
        m2=args.m2,
        L1=args.L1,
        L2=args.L2,
        g=args.g,
        theta1=args.theta1,
        theta2=args.theta2,
        omega1=args.omega1,
        omega2=args.omega2,
        dt=args.dt,
        steps=args.steps,
    )

    df["x1"] = np.sin(df["theta1"])
    df["x2"] = np.sin(df["theta2"])
    if args.series == "x1":
        series = df["x1"].to_numpy()
    elif args.series == "xsum":
        series = (df["x1"] + df["x2"]).to_numpy()
    else:
        series = df["x2"].to_numpy()

    if args.normalize:
        series = (series - np.mean(series)) / (np.std(series) + 1e-9)

    energy = compute_energy(df, args.m1, args.m2, args.L1, args.L2, args.g)
    kinetic = compute_kinetic(df, args.m1, args.m2, args.L1, args.L2)

    df.to_csv(out_dir / "pendulo_duplo_series.csv", index=False)

    classifier = RegimeClassifier(clustering_method=args.method)
    metrics = classifier.scan_embeddings(series, tau_range=range(1, 11), m_range=range(2, 6))
    best_m, best_tau = classifier.select_embedding(metrics, criterion="min_entropy")
    classifier.m = best_m
    classifier.tau = best_tau

    embedded = classifier.embed(series)
    velocity = classifier.compute_velocity(series)
    energy_aligned = energy[(best_m - 1) * best_tau : (best_m - 1) * best_tau + embedded.shape[0]]
    features = {"velocity": velocity, "energy": energy_aligned}
    cluster_labels = classifier.cluster_states(embedded, features)

    entropy = float(classifier.shannon_entropy(embedded))
    rr = float(classifier.recurrence_rate(embedded))
    regimes = label_regimes(entropy, rr, energy_aligned)

    # plots
    plot_entropy_vs_tau(metrics, out_dir / "entropy_vs_tau.png")
    plot_recurrence(embedded, out_dir / "recurrence_plot.png")

    # regime map
    if plt is not None and embedded.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(embedded[:, 0], embedded[:, 1], c=cluster_labels, s=6, cmap="tab10")
        ax.set_xlabel("x(t)")
        ax.set_ylabel(f"x(t-{best_tau})")
        ax.set_title("Regime map (embedding)")
        fig.tight_layout()
        fig.savefig(out_dir / "regime_map.png", dpi=150)
        plt.close(fig)

    label_names = np.unique(regimes)
    label_to_idx = {label: i for i, label in enumerate(label_names)}
    regime_codes = np.array([label_to_idx[label] for label in regimes], dtype=int)

    # labels over time
    if plt is not None:
        fig, ax = plt.subplots(figsize=(9, 4))
        idx_start = (best_m - 1) * best_tau
        time = df["t"].to_numpy()[idx_start : idx_start + embedded.shape[0]]
        ax.plot(time, series[idx_start : idx_start + embedded.shape[0]], color="#64748b")
        scatter = ax.scatter(
            time,
            series[idx_start : idx_start + embedded.shape[0]],
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
        fig.savefig(out_dir / "labels_over_time.png", dpi=150)
        plt.close(fig)

    # x vs v colored by time
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
        fig.savefig(out_dir / "xv_colored_by_time.png", dpi=150)
        plt.close(fig)

    # x vs v plot
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
        fig.savefig(out_dir / "xv_regime.png", dpi=150)
        plt.close(fig)

    # kinetic energy by regime
    if plt is not None:
        fig, ax = plt.subplots(figsize=(9, 4))
        idx_start = (best_m - 1) * best_tau
        time = df["t"].to_numpy()[idx_start : idx_start + embedded.shape[0]]
        kinetic_aligned = kinetic[idx_start : idx_start + embedded.shape[0]]
        for label in label_names:
            mask = regimes == label
            series_energy = np.where(mask, kinetic_aligned, np.nan)
            ax.plot(time, series_energy, label=str(label))
        ax.set_xlabel("t")
        ax.set_ylabel("Energia cinetica")
        ax.set_title("Energia cinetica por regime")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "energy_by_regime.png", dpi=150)
        plt.close(fig)

    # regime recurrence plot
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
        fig.savefig(out_dir / "regime_recurrence_plot.png", dpi=150)
        plt.close(fig)

    # Takens 3D colored by regime
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
            fig.savefig(out_dir / "takens_3d_colored.png", dpi=150)
            plt.close(fig)
        except Exception:
            pass

    # summary
    summary_rows = []
    for label in np.unique(regimes):
        mask = regimes == label
        rr_local = classifier.recurrence_rate(embedded[mask], max_points=800) if np.sum(mask) > 5 else float("nan")
        entropy_local = classifier.shannon_entropy(embedded[mask]) if np.sum(mask) > 5 else float("nan")
        summary_rows.append(
            {
                "regime": str(label),
                "count": int(np.sum(mask)),
                "percent": float(np.mean(mask) * 100.0),
                "entropy_mean": float(entropy_local),
                "recurrence_rate": float(rr_local),
                "energy_mean": float(np.mean(energy_aligned[mask])),
                "energy_std": float(np.std(energy_aligned[mask])),
            }
        )
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)

    config = {
        "m1": args.m1,
        "m2": args.m2,
        "L1": args.L1,
        "L2": args.L2,
        "g": args.g,
        "theta1": args.theta1,
        "theta2": args.theta2,
        "omega1": args.omega1,
        "omega2": args.omega2,
        "dt": args.dt,
        "steps": args.steps,
        "series": args.series,
        "method": args.method,
        "best_m": best_m,
        "best_tau": best_tau,
        "entropy": entropy,
        "recurrence_rate": rr,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    write_report_md(out_dir, config)
    generate_report_pdf(out_dir, config, {})


if __name__ == "__main__":
    main()

