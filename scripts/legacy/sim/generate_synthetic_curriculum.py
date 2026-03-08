"""Gera séries sintéticas curriculadas e salva summaries para treino."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.diagnostics.regime_labels import RegimeClassifier
from engine.diagnostics.auto_regime_model import train_auto_regime_model


def _safe_tqdm(total: int):
    try:
        from tqdm import tqdm

        return tqdm(total=total, desc="Sintéticos", unit="series")
    except Exception:
        return None


def logistic_map(r: float, x0: float, n: int) -> np.ndarray:
    x = np.zeros(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = r * x[i - 1] * (1.0 - x[i - 1])
    return x


def synth_series(kind: str, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0, 10, n)
    if kind == "periodico":
        x = np.sin(2 * np.pi * 1.0 * t)
        labels = np.array(["periodico"] * n, dtype=object)
    elif kind == "oscilatorio":
        x = np.sin(2 * np.pi * 1.0 * t) * np.exp(-0.15 * t)
        labels = np.array(["oscilatorio"] * n, dtype=object)
    elif kind == "rotation":
        x = 0.05 * np.arange(n) + 0.2 * np.sin(2 * np.pi * 0.5 * t)
        labels = np.array(["rotation"] * n, dtype=object)
    elif kind == "caotico":
        x = logistic_map(3.9, rng.uniform(0.1, 0.9), n)
        labels = np.array(["caotico"] * n, dtype=object)
    elif kind == "libration":
        x = 0.6 * np.sin(2 * np.pi * 0.8 * t)
        labels = np.array(["libration"] * n, dtype=object)
    elif kind == "transicao":
        n1 = n // 3
        n2 = n // 3
        n3 = n - n1 - n2
        x1 = np.sin(2 * np.pi * 1.0 * t[:n1])
        x2 = logistic_map(3.9, rng.uniform(0.1, 0.9), n2)
        x3 = 0.3 * rng.normal(size=n3)
        x = np.concatenate([x1, x2, x3])
        labels = np.array(["periodico"] * n1 + ["transicao"] * n2 + ["caotico"] * n3, dtype=object)
    else:
        x = rng.normal(size=n)
        labels = np.array(["caotico"] * n, dtype=object)
    return x.astype(float), labels


def build_summary(
    rc: RegimeClassifier,
    series: np.ndarray,
    labels: np.ndarray,
) -> list[dict[str, float | str]]:
    embedded = rc.embed(series)
    velocity = rc.compute_velocity(series)
    energy = rc.compute_energy(embedded[:, 0], velocity)
    local_features = rc.compute_local_features(embedded[:, 0], window=50)
    start = (rc.m - 1) * rc.tau
    labels_aligned = labels[start : start + embedded.shape[0]]
    return rc._build_summary(embedded, velocity, energy, labels_aligned, extra_features=local_features)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera séries sintéticas curriculadas.")
    parser.add_argument("--outdir", type=str, default="results/synthetic_curriculum")
    parser.add_argument("--per-family", type=int, default=25)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--train-results", type=str, default="results")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--model-prefix", type=str, default="auto_regime_model_synth")
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--balance-mode", type=str, default="oversample", choices=("oversample", "downsample", "none"))
    parser.add_argument("--max-per-class", type=int, default=500)
    parser.add_argument("--no-train", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    families = [
        "periodico",
        "oscilatorio",
        "rotation",
        "libration",
        "caotico",
        "transicao",
    ]

    total = args.per_family * len(families)
    progress = _safe_tqdm(total)

    rc = RegimeClassifier(tau=1, m=3)
    generated = 0

    for family in families:
        for idx in range(args.per_family):
            series, labels = synth_series(family, args.steps, rng)
            case_dir = outdir / f"{family}_{idx:03d}"
            case_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"x": series}).to_csv(case_dir / "series.csv", index=False)

            summary_rows = build_summary(rc, series, labels)
            if summary_rows:
                rc._write_csv(case_dir / "summary.csv", summary_rows)

            generated += 1
            if progress is not None:
                progress.update(1)

            if args.no_train:
                continue
            if args.checkpoint_every > 0 and generated % args.checkpoint_every == 0:
                model_path = Path(args.model_dir) / f"{args.model_prefix}_{generated:04d}.joblib"
                meta_path = Path(args.model_dir) / f"{args.model_prefix}_{generated:04d}_meta.json"
                train_auto_regime_model(
                    results_root=Path(args.train_results),
                    model_path=model_path,
                    meta_path=meta_path,
                    balance=args.balance_mode != "none",
                    balance_mode=args.balance_mode,
                    min_count=args.min_count,
                    max_per_class=args.max_per_class,
                )

    if progress is not None:
        progress.close()

    if not args.no_train:
        model_path = Path(args.model_dir) / f"{args.model_prefix}_final.joblib"
        meta_path = Path(args.model_dir) / f"{args.model_prefix}_final_meta.json"
        train_auto_regime_model(
            results_root=Path(args.train_results),
            model_path=model_path,
            meta_path=meta_path,
            balance=args.balance_mode != "none",
            balance_mode=args.balance_mode,
            min_count=args.min_count,
            max_per_class=args.max_per_class,
        )


if __name__ == "__main__":
    main()

