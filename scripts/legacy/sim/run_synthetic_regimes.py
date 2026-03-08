"""Experimento de validação com regimes sintéticos conhecidos."""

from __future__ import annotations

from pathlib import Path
import sys
import argparse

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.diagnostics.regime_labels import RegimeClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Regimes sintéticos com rotulagem.")
    parser.add_argument("--outdir", type=str, default="results/synthetic_test")
    parser.add_argument("--system-type", type=str, default="generico")
    args = parser.parse_args()

    rng = np.random.default_rng(42)

    n = 1500
    t = np.linspace(0, 10, n)

    # Regime A: oscilador harmônico
    x1 = np.sin(2 * np.pi * 1 * t)

    # Regime B: crescimento exponencial
    x2 = np.exp(0.01 * np.arange(n))

    # Regime C: ruído branco
    x3 = rng.normal(0, 1, n)

    x = np.concatenate([x1, x2, x3])
    series = pd.DataFrame({"t": np.arange(len(x)), "x": x})

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    series.to_csv(out_dir / "synthetic_regimes.csv", index=False)

    rc = RegimeClassifier(
        clustering_method="kmeans",
        cluster_params={
            "n_clusters": 3,
            "merge_small_clusters": False,
        },
    )
    system_type = args.system_type.strip().lower()
    if system_type in {"", "none", "generico"}:
        system_type = None
    rc.run_full_analysis(
        series=series["x"].values,
        output_dir=out_dir,
        system_type=system_type,
        filename_suffix="",
    )


if __name__ == "__main__":
    main()

