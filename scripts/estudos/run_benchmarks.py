from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_interpreting_walk import (  # noqa: E402
    ENERGY_WORLDS,
    Walker,
    plot_walker_analysis,
    _result_subdir,
)
from symbolic_benchmarks import nmse_on_grid  # noqa: E402


NMSE_THRESHOLD = 2.0
ENERGY_GAP_THRESHOLD = 5.0

SELECTED_WORLDS: List[str] = [
    "ackley_1d",
    "rosenbrock_1d",
    "rastrigin_1d",
    "schwefel_1d",
    "levy_1d",
    "shifted_quadratic",
    "shifted_quartic",
    "cosine_shifted",
    "exp_bowl",
    "x_sin",
    "ackley_2d",
    "rastrigin_2d",
    "rosenbrock_2d",
    "himmelblau_2d",
    "booth_2d",
    "mccormick_2d",
    "beale_2d",
]


def _safe_path(path: Path) -> str:
    raw = str(path)
    return raw.replace("\\\\?\\", "")


def _nmse_for_world(world, walker: Walker, rng: np.random.Generator) -> float:
    if world.bounds_array is not None:
        bounds_seq: Sequence[Sequence[float]] = [tuple(row) for row in world.bounds_array]
    else:
        bounds_seq = [(-8.0, 8.0)] * world.dim

    def true_fn(*coords: float) -> float:
        return world.energy(np.array(coords, dtype=float))

    def pred_fn(*coords: float) -> float:
        arr = np.array(coords, dtype=float).reshape(1, -1)
        preds, _ = walker.model.predict_many(arr)
        return float(preds[0])

    return float(nmse_on_grid(true_fn, pred_fn, bounds_seq, n=10000, rng=rng))


def _profile_curve(world, walker: Walker) -> Dict[str, np.ndarray]:
    if world.bounds_array is not None:
        bounds_array = world.bounds_array
    else:
        bounds_array = np.array([(-3.0, 3.0)] * world.dim, dtype=float)

    if world.dim == 1:
        low, high = bounds_array[0]
        x = np.linspace(low, high, 400, dtype=float)
        samples = x.reshape(-1, 1)
    else:
        low, high = bounds_array[0]
        x = np.linspace(low, high, 200, dtype=float)
        samples = np.zeros((x.size, world.dim), dtype=float)
        samples[:, 0] = x

    true_vals = np.array([world.energy(sample) for sample in samples], dtype=float)
    pred_vals, _ = walker.model.predict_many(samples)
    return {"x": x, "true": true_vals, "pred": pred_vals}


def _run_single(world_name: str, steps: int, seed: int, output_dir: Path):
    world_factory = ENERGY_WORLDS[world_name]
    world = world_factory()
    rng = np.random.default_rng(seed)

    if world.bounds_array is not None:
        walker_bounds = world.bounds_array
    else:
        walker_bounds = np.array([(-8.0, 8.0)] * world.dim, dtype=float)

    walker = Walker(world=world, rng=rng, bounds=walker_bounds)
    history = walker.run(steps)
    figure_path = plot_walker_analysis(world, walker, history, steps=steps, output_dir=output_dir)

    nmse = _nmse_for_world(world, walker, rng)
    profile = _profile_curve(world, walker)

    best_energy = float(walker.best_energy)
    optimum_energy = world.optimum_energy
    if optimum_energy is not None and math.isfinite(optimum_energy):
        energy_gap_abs = abs(best_energy - optimum_energy)
        if abs(optimum_energy) > 1e-6:
            energy_gap_percent = 100.0 * energy_gap_abs / abs(optimum_energy)
        else:
            energy_gap_percent = float("nan")
    else:
        energy_gap_abs = float("nan")
        energy_gap_percent = float("nan")

    figure_rel = _safe_path(Path(figure_path))

    record = {
        "world": world.name,
        "dimension": world.dim,
        "equation": world.equation or "",
        "steps": steps,
        "nmse": nmse,
        "final_energy": float(history.energies[-1] if history.energies else float("nan")),
        "best_energy": best_energy,
        "optimum_energy": float(optimum_energy) if optimum_energy is not None else float("nan"),
        "energy_gap_abs": energy_gap_abs,
        "energy_gap_percent": energy_gap_percent,
        "avg_noise": float(np.mean(history.noise_scales)) if history.noise_scales else float("nan"),
        "avg_entropy": float(np.mean(history.entropies)) if history.entropies else float("nan"),
        "best_distance": float(np.linalg.norm(walker.best_position - world.optimum_position))
        if world.optimum_position is not None
        else float("nan"),
        "figure_path": figure_rel,
    }
    return record, profile


def run_benchmarks(world_names: Iterable[str] = SELECTED_WORLDS) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    profiles: Dict[str, Dict[str, np.ndarray]] = {}
    output_dir = Path("results_self_interpreting").resolve()
    OUTPUT_ROOT = Path("results_summary").resolve()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for idx, world_name in enumerate(world_names):
        world = ENERGY_WORLDS[world_name]()
        steps = 400 if world.dim == 1 else 800
        seed = 42 + idx
        record, profile = _run_single(world_name, steps, seed, output_dir)
        records.append(record)
        profiles[world_name] = profile

    df = pd.DataFrame.from_records(records)
    df.sort_values(by=["dimension", "world"], inplace=True)

    mask = (df["nmse"] <= NMSE_THRESHOLD) & (df["energy_gap_abs"] <= ENERGY_GAP_THRESHOLD)
    filtered_df = df[mask].copy()
    if filtered_df.empty:
        filtered_df = df.copy()
    else:
        dropped = df[~mask]
        if not dropped.empty:
            print("Descartando funções por alto desvio:", ", ".join(dropped["world"].tolist()))

    filtered_profiles = {row["world"]: profiles[row["world"]] for _, row in filtered_df.iterrows()}

    _plot_profiles(filtered_df, filtered_profiles, OUTPUT_ROOT)
    _plot_summary_table(filtered_df, OUTPUT_ROOT)
    return filtered_df


def _plot_profiles(df: pd.DataFrame, profiles: Dict[str, Dict[str, np.ndarray]], output_root: Path) -> None:
    charts_dir = output_root / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    one_d = [row for _, row in df[df["dimension"] == 1].iterrows()]
    two_d = [row for _, row in df[df["dimension"] == 2].iterrows()]

    def plot_group(rows: List[pd.Series], title: str, filename: str) -> None:
        if not rows:
            return
        cols = 3 if len(rows) >= 3 else len(rows)
        rows_count = math.ceil(len(rows) / cols)
        fig, axes = plt.subplots(rows_count, cols, figsize=(5 * cols, 3.5 * rows_count))
        axes = np.atleast_2d(axes)
        for idx, row in enumerate(rows):
            r = idx // cols
            c = idx % cols
            ax = axes[r, c]
            profile = profiles[row["world"]]
            ax.plot(profile["x"], profile["true"], label="Energia real", color="black")
            ax.plot(profile["x"], profile["pred"], label="Modelo", color="tab:blue", linestyle="--")
            ax.set_title(row["world"])
            ax.set_xlabel("x" if row["dimension"] == 1 else "perfil x (y=0)")
            ax.set_ylabel("energia")
            ax.grid(True, alpha=0.2)
            if idx == 0:
                ax.legend()
        for j in range(len(rows), rows_count * cols):
            r = j // cols
            c = j % cols
            fig.delaxes(axes[r, c])
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(charts_dir / filename, dpi=200)
        plt.close(fig)

    plot_group(one_d[:6], "Energia vs Modelo – Funções 1D (1/2)", "energy_vs_model_1d_part1.png")
    plot_group(one_d[6:], "Energia vs Modelo – Funções 1D (2/2)", "energy_vs_model_1d_part2.png")
    plot_group(two_d, "Energia vs Modelo – Funções 2D (perfil y=0)", "energy_vs_model_2d.png")


def _plot_summary_table(df: pd.DataFrame, output_root: Path) -> None:
    columns = ["world", "equation", "nmse", "energy_gap_abs", "energy_gap_percent"]
    display_df = df[columns].copy()
    display_df["nmse"] = display_df["nmse"].map(lambda x: f"{x:.3e}")
    display_df["energy_gap_abs"] = display_df["energy_gap_abs"].map(lambda x: f"{x:.3e}")
    display_df["energy_gap_percent"] = display_df["energy_gap_percent"].map(lambda x: f"{x:.2f}%")

    fig, ax = plt.subplots(figsize=(12, 0.6 * len(display_df) + 1))
    ax.axis("off")
    table = ax.table(
        cellText=display_df.values,
        colLabels=["Função", "Equação", "NMSE", "|ΔE|", "Desvio (%)"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    fig.tight_layout()
    fig.savefig(output_root / "summary_table.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    run_benchmarks()
