"""Generate synthetic pendulum data and run regime analysis."""

from pathlib import Path
import sys

import numpy as np
import csv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.diagnostics.regime_labels import RegimeClassifier


def simulate_pendulum(
    steps: int = 3000,
    dt: float = 0.02,
    damping: float = 0.05,
    forcing: float = 1.2,
    freq: float = 0.5,
    noise_std: float = 0.01,
) -> np.ndarray:
    """Simula um pêndulo forçado e amortecido com integração de Euler.

    Returns:
        Série 1-D com a posição angular x(t).
    """
    theta = np.zeros(steps)
    omega = np.zeros(steps)
    theta[0] = 0.5
    omega[0] = 0.0

    for i in range(1, steps):
        t = i * dt
        domega = -damping * omega[i - 1] - np.sin(theta[i - 1]) + forcing * np.cos(freq * t)
        omega[i] = omega[i - 1] + dt * domega
        theta[i] = theta[i - 1] + dt * omega[i]

    if noise_std > 0:
        rng = np.random.default_rng(42)
        theta = theta + rng.normal(0.0, noise_std, size=theta.shape)

    return theta


def run_method(method: str, output_dir: Path) -> None:
    classifier = RegimeClassifier(
        clustering_method=method,
        cluster_params={"min_cluster_size": 25} if method == "hdbscan" else {"n_clusters": 3},
    )
    series = simulate_pendulum()
    classifier.run_full_analysis(
        series=series,
        output_dir=output_dir,
        system_type="pendulo",
        filename_suffix=f"_{method}",
    )


def load_summary(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_comparison(output_dir: Path, methods: tuple[str, ...]) -> None:
    summaries = {}
    for method in methods:
        summary_path = output_dir / f"summary_{method}.csv"
        if summary_path.exists():
            summaries[method] = load_summary(summary_path)

    lines = ["# Comparação entre métodos", ""]
    if not summaries:
        lines.append("Nenhum summary encontrado para comparação.")
        (output_dir / "report_comparison.md").write_text("\n".join(lines), encoding="utf-8")
        return

    lines.append("## Regimes detectados")
    for method, rows in summaries.items():
        regimes = ", ".join(sorted({row["regime"] for row in rows}))
        lines.append(f"- {method}: {regimes}")

    lines.append("")
    lines.append("## Observações")
    if "hdbscan" in summaries and "kmeans" in summaries:
        lines.append(
            "- HDBSCAN tende a separar regiões de transição com mais flexibilidade; "
            "KMeans força partições esféricas e pode suavizar transições."
        )
    else:
        lines.append("- Apenas um método disponível para comparação.")

    (output_dir / "report_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    output_dir = Path("outputs") / "regime" / "pendulum"
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = ("hdbscan", "kmeans")
    for method in methods:
        run_method(method, output_dir)

    write_comparison(output_dir, methods)


if __name__ == "__main__":
    main()

