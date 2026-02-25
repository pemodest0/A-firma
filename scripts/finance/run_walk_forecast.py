#!/usr/bin/env python3
"""Script demonstrativo que gera forecast usando caminhada no hipercubo."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast com caminhada no hipercubo.")
    parser.add_argument("data", type=str, help="CSV com colunas: date, momentum, vol_ratio, drawdown, price")
    parser.add_argument("output", type=str, help="Diretório de saída")
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    try:
        from src.walk_lie.encoding import FeatureBin, HypercubeEncoder
        from src.walk_lie.classical_walk import ClassicalWalkConfig, simulate_walk
        from src.walk_lie.quantum_walk import simulate_quantum_walk
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Modulo 'src.walk_lie' indisponivel. Reinstale o pacote legado walk_lie ou atualize este script."
        ) from exc

    df = pd.read_csv(args.data, parse_dates=["date"])  # columns: date, momentum, vol_ratio, drawdown, price
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    bins = [FeatureBin("momentum", [-0.5, 0.5]), FeatureBin("vol_ratio", [1.0]), FeatureBin("drawdown", [-0.1])]
    encoder = HypercubeEncoder(bins)

    def weight_func(state: dict[str, float]) -> dict[int, float]:
        weights = {}
        return weights

    config = ClassicalWalkConfig(encoder=encoder, base_transition={})
    probs = simulate_walk(config, start_vertex=0, steps=args.steps)
    q_probs = simulate_quantum_walk(encoder, steps=args.steps, start_vertex=0)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "classical_probs.npy", probs)
    np.save(out_dir / "quantum_probs.npy", q_probs)
    print(f"Salvos forecasts em {out_dir}")


if __name__ == "__main__":
    main()
