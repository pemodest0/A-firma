from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _resolve_output_path(output_path: Optional[Path]) -> Optional[Path]:
    if output_path is None:
        return None
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_distribution_heatmap(
    distributions: np.ndarray,
    positions: Sequence[int],
    title: str,
    output_path: Optional[Path] = None,
    cmap: str = "viridis",
) -> plt.Figure:
    """Plot a probability heatmap over time (rows) and positions (columns)."""
    positions = np.asarray(positions, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5))
    extent = (positions[0], positions[-1], 0, distributions.shape[0] - 1)
    im = ax.imshow(
        distributions,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        extent=extent,
    )
    ax.set_xlabel("Position")
    ax.set_ylabel("Step")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Probability")
    fig.tight_layout()
    path = _resolve_output_path(output_path)
    if path is not None:
        fig.savefig(path, dpi=300)
    return fig


def plot_entropy_curves(
    entropy_series: Dict[str, Sequence[float]],
    output_path: Optional[Path] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, entropies in entropy_series.items():
        steps = np.arange(len(entropies))
        ax.plot(steps, entropies, linewidth=2.0, label=label)
    ax.set_xlabel("Step")
    ax.set_ylabel("Shannon entropy (bits)")
    ax.set_title("Entropy evolution")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = _resolve_output_path(output_path)
    if path is not None:
        fig.savefig(path, dpi=300)
    return fig


def plot_hitting_time_bars(
    hitting_times: Iterable[Tuple[str, Optional[int]]],
    output_path: Optional[Path] = None,
) -> plt.Figure:
    labels: List[str] = []
    values: List[float] = []
    annotations: List[str] = []
    for label, value in hitting_times:
        labels.append(label)
        if value is None:
            values.append(np.nan)
            annotations.append("N/R")
        else:
            values.append(float(value))
            annotations.append("")

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Hitting time (steps)")
    ax.set_title("Hitting time to target node")

    ylim = ax.get_ylim()
    top = np.nanmax(values) if np.isfinite(np.nanmax(values)) else 1.0
    if not np.isfinite(top):
        top = 1.0
    ax.set_ylim(0, max(top * 1.1, 1.0))

    for idx, bar in enumerate(bars):
        annotation = annotations[idx]
        if annotation:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                ax.get_ylim()[1] * 0.05,
                annotation,
                ha="center",
                va="bottom",
                color="red",
                fontweight="bold",
            )
        elif values[idx] == values[idx]:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{values[idx]:.0f}",
                ha="center",
                va="bottom",
            )

    fig.tight_layout()
    path = _resolve_output_path(output_path)
    if path is not None:
        fig.savefig(path, dpi=300)
    return fig
