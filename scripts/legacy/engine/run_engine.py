#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class EnginePreset:
    name: str
    assets: Tuple[str, ...]
    frequencies: Tuple[str, ...]
    targets: Tuple[str, ...]
    horizons: Tuple[int, ...]
    models: Tuple[str, ...]
    test_years: Tuple[int, ...]
    train_window_years: int
    allow_downloads: bool = False
    early_stop: bool = False
    checkpoint_every: int = 50
    volatility_assets: Tuple[str, ...] | None = None
    max_jobs: int | None = None
    parallel_workers: int = 1
    cache_embeddings: bool = True
    skip_long_horizon_on_fail: bool = False


TERMINAL_EXHAUSTIVE = EnginePreset(
    name="terminal_exhaustive",
    assets=("SPY", "QQQ", "IWM", "TLT", "GLD", "XLE", "XLK", "EEM", "BTC-USD", "^VIX"),
    frequencies=("daily", "weekly"),
    targets=("return", "volatility"),
    horizons=(1, 5, 20, 60),
    models=("persist", "ma5", "knn_phase", "markov_phase"),
    test_years=(2018, 2019, 2020, 2021, 2022, 2023, 2024),
    train_window_years=5,
    allow_downloads=False,
    early_stop=False,
    checkpoint_every=50,
)


def iter_jobs(preset: EnginePreset) -> Iterable[dict]:
    volatility_assets = set(preset.volatility_assets or preset.assets)
    for asset in preset.assets:
        for freq in preset.frequencies:
            for target in preset.targets:
                if target == "volatility" and asset not in volatility_assets:
                    continue
                for horizon in preset.horizons:
                    for model in preset.models:
                        for year in preset.test_years:
                            train_start = year - preset.train_window_years
                            train_end = year - 1
                            yield {
                                "asset": asset,
                                "frequency": freq,
                                "target": target,
                                "horizon": horizon,
                                "model": model,
                                "train_start_year": train_start,
                                "train_end_year": train_end,
                                "test_year": year,
                            }


def write_plan(preset: EnginePreset, output: Path) -> None:
    jobs = list(iter_jobs(preset))
    if preset.max_jobs is not None:
        jobs = jobs[: preset.max_jobs]
    payload = {
        "preset": asdict(preset),
        "total_jobs": len(jobs),
        "jobs": jobs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def init_state(output: Path, preset: EnginePreset, total_jobs: int) -> None:
    state_path = output.with_suffix(".state.json")
    if state_path.exists():
        return
    state = {
        "preset": asdict(preset),
        "total_jobs": total_jobs,
        "completed_jobs": 0,
        "skipped_jobs": 0,
        "last_checkpoint": None,
        "resume_cursor": 0,
        "early_stop_triggered": False,
        "notes": [],
    }
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def should_early_stop(stats: dict) -> bool:
    """Heurística simples para early stopping (placeholder).

    Esta função não executa modelos; apenas define o ponto de extensão para
    parar execuções futuras quando não houver ganho em horizontes longos.
    """
    if not stats:
        return False
    return bool(stats.get("all_failed_long_horizon"))


def write_checkpoint(output_dir: Path, state: dict, partial_rows: List[dict]) -> None:
    """Salva progresso parcial para permitir resume."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "overview.partial.csv"
    if partial_rows:
        fieldnames = list(partial_rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(partial_rows)
    state_path = output_dir / "engine_state.json"
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch planner for motor de analise de serie temporal tests.")
    parser.add_argument("--preset", default="terminal_exhaustive")
    parser.add_argument("--out", default="results/engine_plan.json")
    parser.add_argument("--resume", action="store_true", help="Retoma execução a partir do checkpoint.")
    parser.add_argument("--early-stop", action="store_true", help="Ativa heurística de early stopping.")
    parser.add_argument("--checkpoint-every", type=int, default=None, help="Checkpoint a cada N jobs.")
    parser.add_argument("--volatility-assets", type=str, default=None, help="Lista de ativos para volatilidade.")
    parser.add_argument("--max-jobs", type=int, default=None, help="Limita número total de jobs.")
    parser.add_argument("--workers", type=int, default=None, help="Números de workers para execução paralela.")
    parser.add_argument("--no-cache-embeddings", action="store_true", help="Desativa cache de embeddings.")
    parser.add_argument(
        "--skip-long-horizon",
        action="store_true",
        help="Sugere pular horizontes longos se falhar no curto prazo.",
    )
    args = parser.parse_args()

    if args.preset == "terminal_exhaustive":
        preset = TERMINAL_EXHAUSTIVE
    else:
        raise ValueError(f"Unknown preset: {args.preset}")

    if args.early_stop:
        preset = EnginePreset(**{**asdict(preset), "early_stop": True})
    if args.checkpoint_every is not None:
        preset = EnginePreset(**{**asdict(preset), "checkpoint_every": int(args.checkpoint_every)})
    if args.volatility_assets:
        volatility_assets = tuple(a.strip() for a in args.volatility_assets.split(",") if a.strip())
        preset = EnginePreset(**{**asdict(preset), "volatility_assets": volatility_assets})
    if args.max_jobs is not None:
        preset = EnginePreset(**{**asdict(preset), "max_jobs": int(args.max_jobs)})
    if args.workers is not None:
        preset = EnginePreset(**{**asdict(preset), "parallel_workers": int(args.workers)})
    if args.no_cache_embeddings:
        preset = EnginePreset(**{**asdict(preset), "cache_embeddings": False})
    if args.skip_long_horizon:
        preset = EnginePreset(**{**asdict(preset), "skip_long_horizon_on_fail": True})

    write_plan(preset, Path(args.out))
    init_state(Path(args.out), preset, total_jobs=len(list(iter_jobs(preset))))
    print(f"Plano gerado em {args.out}")


if __name__ == "__main__":
    main()
