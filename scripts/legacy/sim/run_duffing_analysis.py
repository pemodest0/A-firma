"""Simulate Duffing oscillator and run regime analysis."""

from pathlib import Path
import sys
import csv
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.diagnostics.regime_labels import RegimeClassifier

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def simulate_duffing(
    steps: int = 3000,
    dt: float = 0.01,
    damping: float = 0.2,
    forcing: float = 0.3,
    freq: float = 1.2,
    noise_std: float = 0.01,
) -> np.ndarray:
    """Simula o oscilador de Duffing forçado e amortecido.

    Equação: x¨ + δ x˙ − x + x^3 = γ cos(ω t)
    """
    x = np.zeros(steps)
    v = np.zeros(steps)
    x[0] = 0.5
    v[0] = 0.0

    for i in range(1, steps):
        t = i * dt
        accel = -damping * v[i - 1] + x[i - 1] - x[i - 1] ** 3 + forcing * np.cos(freq * t)
        v[i] = v[i - 1] + dt * accel
        x[i] = x[i - 1] + dt * v[i]

    if noise_std > 0:
        rng = np.random.default_rng(42)
        x = x + rng.normal(0.0, noise_std, size=x.shape)

    return x


def run_method(method: str, output_dir: Path, series: np.ndarray) -> None:
    base_params = {
        "merge_small_clusters": True,
        "merge_min_pct": 0.02,
        "merge_max_distance": 2.0,
        "score_small_cluster_penalty": 0.1,
        "score_max_clusters": 6,
        "score_too_many_penalty": 0.05,
    }
    classifier = RegimeClassifier(
        clustering_method=method,
        cluster_params=(
            {
                **base_params,
                "min_cluster_size": 60,
                "min_samples": 10,
            }
            if method == "hdbscan"
            else {**base_params, "n_clusters": 3}
        ),
    )
    classifier.run_full_analysis(
        series=series,
        output_dir=output_dir,
        system_type="duffing",
        filename_suffix="",
    )


def load_summary(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_report_md(output_root: Path, methods: tuple[str, ...]) -> Path:
    report_path = output_root / "report_duffing.md"
    def rel(path: Path) -> str:
        return path.as_posix()
    lines = [
        "# Análise dos Regimes do Sistema de Duffing Forçado e Amortecido",
        "",
        "## Introdução",
        "Este relatório apresenta os resultados de uma análise automatizada de regimes",
        "dinâmicos no oscilador de Duffing. O objetivo é identificar padrões de regime",
        "a partir de embedding atrasado, entropia, recorrência e clusterização não supervisionada.",
        "",
        "Parâmetros da simulação:",
        "- Equação: x¨ + δ x˙ − x + x³ = γ cos(ωt)",
        "- δ = 0.2, γ = 0.3, ω = 1.2",
        "- 3000 passos com Δt = 0.01",
        "- Ruído leve gaussiano (σ = 0.01)",
        "- Integração numérica por Euler explícito",
        "",
    ]

    def add_method_section(method: str) -> None:
        method_dir = output_root / f"duffing_{method}"
        lines.extend(
            [
                f"## Resultados ({method.upper()})",
                "",
                f"### Labels no tempo ({method})",
                f"![labels_over_time]({rel(method_dir / 'labels_over_time.png')})",
                "Mostra como os regimes se alternam ao longo do tempo.",
                "",
                f"### Espaço de fases x(t) vs v(t) ({method})",
                f"![xv_regime]({rel(method_dir / 'xv_regime.png')})",
                "Representa a separação de regimes no espaço de fases.",
                "",
                f"### Entropia vs τ ({method})",
                f"![entropy_vs_tau]({rel(method_dir / 'entropy_vs_tau.png')})",
                "Entropia crescente com o atraso sugere maior complexidade e mistura de estados.",
                "",
                f"### Recurrence plot ({method})",
                f"![recurrence_plot]({rel(method_dir / 'recurrence_plot.png')})",
                "Diagonais indicam comportamento determinístico; padrões fragmentados sugerem caos.",
                "",
            ]
        )

        regime_3d = method_dir / "regime_3d.png"
        if regime_3d.exists():
            lines.extend(
                [
                    f"### Embedding 3D ({method})",
                    f"![regime_3d]({rel(regime_3d)})",
                    "Visualização tridimensional dos regimes no embedding.",
                    "",
                ]
            )

        summary_path = method_dir / "summary.csv"
        if summary_path.exists():
            rows = load_summary(summary_path)
            lines.append(f"### Estatísticas dos regimes ({method})")
            if rows:
                columns = list(rows[0].keys())
                header = "| " + " | ".join(columns) + " |"
                separator = "| " + " | ".join(["---"] * len(columns)) + " |"
                lines.append(header)
                lines.append(separator)
                for row in rows:
                    lines.append("| " + " | ".join(row[col] for col in columns) + " |")
            lines.append("")

    for method in methods:
        add_method_section(method)

    # Comparison summary table
    comparison_rows = []
    for method in methods:
        summary_path = output_root / f"duffing_{method}" / "summary.csv"
        if summary_path.exists():
            rows = load_summary(summary_path)
            if rows:
                regimes = ", ".join(sorted({row["regime"] for row in rows}))
                dominant = max(rows, key=lambda r: float(r["percent"]))
                comparison_rows.append(
                    {
                        "metodo": method,
                        "n_regimes": str(len(rows)),
                        "regimes": regimes,
                        "dominante": dominant["regime"],
                        "percent_dominante": dominant["percent"],
                    }
                )

    lines.extend(
        [
            "## Comparação entre métodos",
            "",
            "### Resumo comparativo",
        ]
    )
    if comparison_rows:
        columns = list(comparison_rows[0].keys())
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
        for row in comparison_rows:
            lines.append("| " + " | ".join(row[col] for col in columns) + " |")
        lines.append("")

    lines.extend(
        [
            "| KMeans | HDBSCAN |",
            "| --- | --- |",
            f"| ![regime_map_kmeans]({rel(output_root / 'duffing_kmeans' / 'regime_map.png')}) | "
            f"![regime_map_hdbscan]({rel(output_root / 'duffing_hdbscan' / 'regime_map.png')}) |",
            f"| ![labels_kmeans]({rel(output_root / 'duffing_kmeans' / 'labels_over_time.png')}) | "
            f"![labels_hdbscan]({rel(output_root / 'duffing_hdbscan' / 'labels_over_time.png')}) |",
            "",
            "KMeans tende a impor partições rígidas, enquanto HDBSCAN pode isolar regiões de transição.",
            "Diferenças nos resumos de energia e nas porcentagens indicam regimes fundidos ou separados.",
            "",
            "## Discussão final",
            "Os regimes encontrados refletem a dinâmica em torno dos poços de potencial e transições.",
            "A clusterização não supervisionada consegue separar estados estáveis e regiões de troca",
            "de poço, mas pode confundir regimes em regiões com energia intermediária.",
            "",
            "Limitações:",
            "- Ruído leve pode borrar fronteiras entre regimes.",
            "- O embedding ótimo depende do critério de entropia escolhido.",
            "",
            "Próximos passos sugeridos:",
            "- Incluir GMM para avaliar regimes sobrepostos.",
            "- Explorar Ulam/partições Markovianas para transições mais ricas.",
            "- Usar seleção de τ baseada em entropia mínima ou contraste máximo.",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def render_pdf(report_md: Path) -> None:
    try:
        subprocess.run(
            ["pandoc", str(report_md), "-o", str(report_md.with_suffix(".pdf"))],
            check=True,
        )
    except Exception:
        fallback_pdf(report_md.parent)


def fallback_pdf(output_root: Path) -> None:
    if plt is None:
        return

    pdf_path = output_root / "report_duffing.pdf"
    methods = ("kmeans", "hdbscan")
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(pdf_path) as pdf:
        # Title + intro
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        intro = (
            "Análise dos Regimes do Sistema de Duffing Forçado e Amortecido\n\n"
            "Equação: x¨ + δ x˙ − x + x³ = γ cos(ωt)\n"
            "Parâmetros: δ = 0.2, γ = 0.3, ω = 1.2\n"
            "Simulação: 3000 passos, Δt = 0.01, ruído gaussiano leve (σ = 0.01)\n\n"
            "Objetivo: detectar regimes dinâmicos via embedding atrasado, entropia,\n"
            "recorrência e clusterização não supervisionada."
        )
        ax.text(0.05, 0.95, intro, va="top", fontsize=13)
        pdf.savefig(fig)
        plt.close(fig)

        # Method pages
        for method in methods:
            method_dir = output_root / f"duffing_{method}"
            if not method_dir.exists():
                continue

            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            fig.suptitle(f"Duffing - {method.upper()}", fontsize=14)
            plot_map = {
                "labels_over_time.png": (0, 0, "Regimes ao longo do tempo"),
                "xv_regime.png": (0, 1, "Separação no espaço x(t) vs v(t)"),
                "entropy_vs_tau.png": (1, 0, "Entropia vs atraso τ"),
                "recurrence_plot.png": (1, 1, "Recurrence plot"),
            }
            for filename, (r, c, title) in plot_map.items():
                axes[r, c].axis("off")
                img_path = method_dir / filename
                if img_path.exists():
                    axes[r, c].imshow(plt.imread(img_path))
                    axes[r, c].set_title(title, fontsize=10)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # 3D plot if available
            regime_3d = method_dir / "regime_3d.png"
            if regime_3d.exists():
                fig, ax = plt.subplots(figsize=(7, 6))
                ax.axis("off")
                ax.imshow(plt.imread(regime_3d))
                ax.set_title("Embedding 3D dos regimes", fontsize=12)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

            # Summary table
            summary_path = method_dir / "summary.csv"
            rows = load_summary(summary_path) if summary_path.exists() else []
            fig, ax = plt.subplots(figsize=(11, 5))
            ax.axis("off")
            if rows:
                columns = list(rows[0].keys())
                table = ax.table(
                    cellText=[[row[col] for col in columns] for row in rows],
                    colLabels=columns,
                    loc="center",
                )
                table.auto_set_font_size(False)
                table.set_fontsize(8)
            ax.set_title(f"Resumo quantitativo - {method.upper()}", fontsize=12)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # Comparison page
        fig, axes = plt.subplots(1, 2, figsize=(11, 6))
        fig.suptitle("Comparação KMeans vs HDBSCAN", fontsize=14)
        for ax, method in zip(axes, methods):
            ax.axis("off")
            img_path = output_root / f"duffing_{method}" / "regime_map.png"
            if img_path.exists():
                ax.imshow(plt.imread(img_path))
                ax.set_title(method.upper(), fontsize=12)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Discussion
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        discussion = (
            "Discussão final\n\n"
            "- Regimes refletem a dinâmica nos poços e regiões de transição.\n"
            "- KMeans impõe partições rígidas; HDBSCAN pode separar transições.\n"
            "- Ruído leve pode borrar fronteiras entre regimes.\n\n"
            "Próximos passos:\n"
            "- Incluir GMM para regimes sobrepostos.\n"
            "- Explorar Ulam/partições Markovianas.\n"
            "- Ajustar τ com base em entropia mínima."
        )
        ax.text(0.05, 0.95, discussion, va="top", fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)


def main() -> None:
    output_root = Path("results")
    output_root.mkdir(parents=True, exist_ok=True)

    series = simulate_duffing()
    methods = ("kmeans", "hdbscan")

    for method in methods:
        method_dir = output_root / f"duffing_{method}"
        method_dir.mkdir(parents=True, exist_ok=True)
        run_method(method, method_dir, series)

    for method in methods:
        method_dir = output_root / f"duffing_{method}"
        summary_path = method_dir / "summary.csv"
        if summary_path.exists():
            (output_root / f"summary_{method}.csv").write_text(
                summary_path.read_text(encoding="utf-8"), encoding="utf-8"
            )

    report_md = write_report_md(output_root, methods)
    render_pdf(report_md)


if __name__ == "__main__":
    main()

