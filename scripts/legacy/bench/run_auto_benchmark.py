"""Benchmark completo do motor automatico com progresso/ETA por simulacao."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import subprocess
import csv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.run_with_progress import run_with_progress


TASKS = [
    {
        "name": "Duffing (kmeans+hdbscan)",
        "command": ["python", "scripts/legacy/sim/run_duffing_analysis.py"],
        "expected_seconds": 120,
        "out_dir": "results/duffing_kmeans",
    },
    {
        "name": "Lorenz (hdbscan)",
        "command": [
            "python",
            "scripts/legacy/sim/run_lorenz_analysis.py",
            "--method",
            "auto",
            "--outdir",
            "results/lorenz_benchmark_hdbscan",
            "--signal",
            "x",
            "--system-type",
            "lorenz",
        ],
        "expected_seconds": 120,
        "out_dir": "results/lorenz_benchmark_hdbscan",
        "summary_file": "summary_lorenz.csv",
    },
    {
        "name": "Lorenz (kmeans)",
        "command": [
            "python",
            "scripts/legacy/sim/run_lorenz_analysis.py",
            "--method",
            "auto",
            "--outdir",
            "results/lorenz_benchmark_kmeans",
            "--signal",
            "x",
            "--system-type",
            "lorenz",
        ],
        "expected_seconds": 120,
        "out_dir": "results/lorenz_benchmark_kmeans",
        "summary_file": "summary_lorenz.csv",
    },
    {
        "name": "Van der Pol (heuristico)",
        "command": [
            "python",
            "scripts/legacy/sim/run_vanderpol_analysis.py",
            "--outdir",
            "results/vanderpol_benchmark",
            "--label-mode",
            "heuristic",
            "--method",
            "auto",
        ],
        "expected_seconds": 120,
        "out_dir": "results/vanderpol_benchmark",
        "summary_file": "summary_vanderpol.csv",
    },
    {
        "name": "Regimes sintéticos",
        "command": [
            "python",
            "scripts/legacy/sim/run_synthetic_regimes.py",
            "--outdir",
            "results/synthetic_benchmark",
            "--system-type",
            "generico",
        ],
        "expected_seconds": 40,
        "out_dir": "results/synthetic_benchmark",
    },
    {
        "name": "Pendulo duplo (pesado)",
        "command": [
            "python",
            "scripts/legacy/sim/run_pendulo_duplo_analysis.py",
            "--outdir",
            "results/pendulo_duplo_benchmark",
        ],
        "expected_seconds": 240,
        "out_dir": "results/pendulo_duplo_benchmark",
    },
]


def summarize_summary_csv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    regimes = sorted({row.get("regime", "") for row in rows if row.get("regime", "")})
    dominant = max(rows, key=lambda row: float(row.get("percent", 0.0)))
    total = sum(float(row.get("count", 0.0)) for row in rows)
    return {
        "regimes": ", ".join(regimes),
        "n_regimes": str(len(regimes)),
        "dominant": dominant.get("regime", ""),
        "dominant_percent": dominant.get("percent", ""),
        "total_points": f"{total:.0f}",
    }


def build_benchmark_report(output_dir: Path, summaries: list[dict[str, str]]) -> None:
    report_md = output_dir / "benchmark_report.md"
    lines = [
        "# Benchmark do motor automatico",
        "",
        "## Resumo por serie",
    ]
    if summaries:
        columns = list(summaries[0].keys())
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
        for row in summaries:
            lines.append("| " + " | ".join(row.get(col, "") for col in columns) + " |")
        lines.append("")

    eval_dir = output_dir / "auto_regime_eval"
    report_json = eval_dir / "classification_report_holdout.json"
    cv_json = eval_dir / "cv_scores.json"
    if report_json.exists():
        lines.extend(
            [
                "## Avaliacao automatica (holdout)",
                "```json",
                report_json.read_text(encoding="utf-8"),
                "```",
                "",
            ]
        )
    if cv_json.exists():
        lines.extend(
            [
                "## Cross-validation",
                "```json",
                cv_json.read_text(encoding="utf-8"),
                "```",
                "",
            ]
        )

    report_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark completo do motor automatico.")
    parser.add_argument("--outdir", type=str, default="results/benchmark_auto")
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--kfold", type=int, default=5)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Permite seguir com treino/avaliacao mesmo com falhas nas series.",
    )
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.logdir) if args.logdir else (out_dir / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    summaries: list[dict[str, str]] = []

    for task in TASKS:
        safe_name = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_"
            for ch in task["name"].lower()
        )
        log_path = log_dir / f"{safe_name}.log"
        code = run_with_progress(
            task["command"],
            task["name"],
            task["expected_seconds"],
            log_path=str(log_path),
        )
        if code != 0:
            failures += 1
            summaries.append(
                {
                    "serie": task["name"],
                    "summary_csv": "",
                    "regimes": "",
                    "n_regimes": "",
                    "dominant": "",
                    "dominant_percent": "",
                    "total_points": "",
                    "log": str(log_path),
                    "status": "FAILED",
                }
            )
            continue
        summary_file = task.get("summary_file", "summary.csv")
        summary_path = Path(task["out_dir"]) / summary_file
        summary_info = summarize_summary_csv(summary_path)
        summary_info["serie"] = task["name"]
        summary_info["summary_csv"] = str(summary_path)
        summary_info["log"] = str(log_path)
        summary_info["status"] = "OK"
        summaries.append(summary_info)

    if failures and not args.allow_partial:
        meta = {
            "generated_at": time.time(),
            "failures": failures,
            "tasks": [task["name"] for task in TASKS],
        }
        (out_dir / "benchmark_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        build_benchmark_report(out_dir, summaries)
        print(
            f"[ERRO] {failures} tarefa(s) falharam. "
            "Use --allow-partial para seguir com treino/avaliacao."
        )
        sys.exit(1)

    # Treino automatico
    train_cmd = [
        "python",
        "scripts/bench/train_auto_regime_model.py",
        "--results",
        "results",
        "--model-dir",
        "models",
    ]
    run_with_progress(
        train_cmd,
        "Treino automatico",
        40,
        log_path=str(log_dir / "treino_automatico.log"),
    )

    # Avaliacao automatica
    eval_dir = out_dir / "auto_regime_eval"
    eval_cmd = [
        "python",
        "scripts/bench/evaluate_auto_regime_model.py",
        "--results",
        "results",
        "--model-dir",
        "models",
        "--outdir",
        str(eval_dir),
        "--min-count",
        str(args.min_count),
        "--kfold",
        str(args.kfold),
        "--group-holdout",
        "--group-kfold",
    ]
    run_with_progress(
        eval_cmd,
        "Avaliacao automatica",
        30,
        log_path=str(log_dir / "avaliacao_automatica.log"),
    )

    summary_csv = out_dir / "benchmark_summary.csv"
    if summaries:
        with summary_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)

    build_benchmark_report(out_dir, summaries)

    meta = {
        "generated_at": time.time(),
        "failures": failures,
        "tasks": [task["name"] for task in TASKS],
    }
    (out_dir / "benchmark_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
