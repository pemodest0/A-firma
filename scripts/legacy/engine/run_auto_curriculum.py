"""Executa um curriculo de simulacoes para treinar/validar o motor."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


TASKS = [
    {
        "name": "Duffing (kmeans+hdbscan)",
        "command": ["python", "scripts/legacy/sim/run_duffing_analysis.py"],
        "expected_seconds": 120,
    },
    {
        "name": "Lorenz (hdbscan)",
        "command": [
            "python",
            "scripts/legacy/sim/run_lorenz_analysis.py",
            "--method",
            "hdbscan",
            "--outdir",
            "results/lorenz_hdbscan_curriculum",
        ],
        "expected_seconds": 90,
    },
    {
        "name": "Van der Pol (hdbscan)",
        "command": [
            "python",
            "scripts/legacy/sim/run_vanderpol_analysis.py",
            "--outdir",
            "results/vanderpol_hdbscan_curriculum",
        ],
        "expected_seconds": 90,
    },
    {
        "name": "Regimes sintéticos",
        "command": ["python", "scripts/legacy/sim/run_synthetic_regimes.py"],
        "expected_seconds": 40,
    },
]

HEAVY_TASKS = [
    {
        "name": "Pendulo duplo (pesado)",
        "command": [
            "python",
            "scripts/legacy/sim/run_pendulo_duplo_analysis.py",
            "--outdir",
            "results/pendulo_duplo_curriculum",
        ],
        "expected_seconds": 240,
    },
]


def run_with_progress(command: list[str], label: str, expected_seconds: int) -> int:
    if tqdm is None:
        print(f"[start] {label}: {' '.join(command)}")
        result = subprocess.run(command)
        print(f"[done] {label} (exit={result.returncode})")
        return result.returncode

    bar = tqdm(total=1.0, desc=label, unit="task")
    start = time.time()
    process = subprocess.Popen(command)
    try:
        while process.poll() is None:
            elapsed = time.time() - start
            progress = min(elapsed / max(expected_seconds, 1), 0.98)
            bar.n = progress
            bar.refresh()
            time.sleep(1.0)
        bar.n = 1.0
        bar.refresh()
    finally:
        bar.close()
    return process.returncode or 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Curriculo automatico de simulacoes.")
    parser.add_argument("--heavy", action="store_true", help="Inclui simulacoes pesadas.")
    args = parser.parse_args()

    tasks = list(TASKS)
    if args.heavy:
        tasks.extend(HEAVY_TASKS)

    failures = 0
    for task in tasks:
        code = run_with_progress(
            task["command"], task["name"], task["expected_seconds"]
        )
        if code != 0:
            failures += 1

    if failures:
        print(f"Falhas: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
