#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

def main() -> None:
    parser = argparse.ArgumentParser(description="Calcula métrica Λ de comutadores.")
    parser.add_argument("theta_json", type=str, help="JSON com lista de vetores theta")
    args = parser.parse_args()

    try:
        from src.walk_lie.lie_tools import lie_penalty
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Modulo 'src.walk_lie' indisponivel. Reinstale o pacote legado walk_lie ou atualize este script."
        ) from exc

    data = json.loads(Path(args.theta_json).read_text())
    thetas = [np.array(theta, dtype=float) for theta in data]
    penalty = lie_penalty(np.stack(thetas, axis=0))
    print(f"Λ = {penalty:.4f}")


if __name__ == "__main__":
    main()
