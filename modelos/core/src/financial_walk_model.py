from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from classical_walk import simulate_classical_walk
from graph_utils import Graph, line_graph
from quantum_walk import QISKIT_AVAILABLE, QuantumWalkResult, simulate_quantum_walk

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from dados.brutos.financial_loader import ReturnWindow
DEFAULT_STEPS = 40
DISPERSION_FIT_MIN = 5
NOISE_LEVEL = 0.1

MODE_CLASSICAL = "classical"
MODE_HADAMARD = "quantum_hadamard"
MODE_GROVER = "quantum_grover"

PHASE_LOW = 0.6
PHASE_HIGH = 0.85

__all__ = [
    "MODE_CLASSICAL",
    "MODE_HADAMARD",
    "MODE_GROVER",
    "PHASE_LOW",
    "PHASE_HIGH",
    "ModeMetrics",
    "WindowAnalysis",
    "ForecastCurve",
    "ForecastSummary",
    "classify_phase",
    "describe_phase",
    "analyze_financial_windows",
    "forecast_from_distribution",
]


@dataclass(frozen=True)
class ModeMetrics:
    """Summary statistics for a single walk mode within a window."""

    mode: str
    entropy_initial: float
    entropy_final: float
    entropy_change: float
    dispersion_alpha: float
    dispersion_alpha_noisy: float
    dispersion_sensitivity: float


@dataclass(frozen=True)
class WindowAnalysis:
    """Aggregated walk metrics for a single time window."""

    start: np.datetime64
    end: np.datetime64
    distribution: np.ndarray
    metrics: Dict[str, ModeMetrics]


def _standard_deviation_series(distributions: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Return σ(t) for every step."""
    stds = np.zeros(distributions.shape[0], dtype=float)
    for idx, probs in enumerate(distributions):
        mean = float(np.dot(values, probs))
        variance = float(np.dot((values - mean) ** 2, probs))
        stds[idx] = math.sqrt(max(variance, 0.0))
    return stds


def _fit_dispersion_alpha(stds: np.ndarray, fit_min: int, fit_max: Optional[int]) -> float:
    """Fit σ(t) ∝ t^α via log-log regression."""
    steps = np.arange(stds.size, dtype=float)
    mask = steps >= max(fit_min, 1)
    if fit_max is not None:
        mask &= steps <= fit_max
    mask &= np.isfinite(stds)
    mask &= stds > 0.0
    if mask.sum() < 2:
        return float("nan")
    x = np.log(steps[mask])
    y = np.log(stds[mask])
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def _mix_with_uniform(distributions: np.ndarray, noise_level: float) -> np.ndarray:
    if noise_level <= 0:
        return distributions.copy()
    uniform = np.full(distributions.shape[1], 1.0 / distributions.shape[1], dtype=float)
    mix = (1.0 - noise_level) * distributions + noise_level * uniform
    mix_sum = mix.sum(axis=1, keepdims=True)
    mix_sum[mix_sum == 0.0] = 1.0
    return mix / mix_sum


def _build_metrics(
    mode: str,
    entropies: np.ndarray,
    distributions: np.ndarray,
    values: np.ndarray,
    fit_min: int,
    fit_max: Optional[int],
    noise_level: float,
) -> ModeMetrics:
    stds = _standard_deviation_series(distributions, values)
    alpha = _fit_dispersion_alpha(stds, fit_min, fit_max)

    noisy_distributions = _mix_with_uniform(distributions, noise_level)
    noisy_stds = _standard_deviation_series(noisy_distributions, values)
    alpha_noisy = _fit_dispersion_alpha(noisy_stds, fit_min, fit_max)

    entropy_initial = float(entropies[0])
    entropy_final = float(entropies[-1])
    entropy_change = entropy_final - entropy_initial
    sensitivity = alpha - alpha_noisy

    return ModeMetrics(
        mode=mode,
        entropy_initial=entropy_initial,
        entropy_final=entropy_final,
        entropy_change=entropy_change,
        dispersion_alpha=alpha,
        dispersion_alpha_noisy=alpha_noisy,
        dispersion_sensitivity=sensitivity,
    )


def _simulate_quantum_mode(
    graph: Graph,
    steps: int,
    distribution: np.ndarray,
    coin: str,
) -> Optional[QuantumWalkResult]:
    if not QISKIT_AVAILABLE:
        return None
    start_node = graph.num_nodes // 2
    return simulate_quantum_walk(
        graph,
        steps,
        coin=coin,
        start_node=start_node,
        initial_distribution=distribution,
        measurement="none",
    )


@dataclass(frozen=True)
class ForecastCurve:
    """Predicted evolution of the walk for a given mode."""

    mode: str
    expected_returns: np.ndarray
    cumulative_returns: np.ndarray
    predicted_prices: np.ndarray
    std_per_step: np.ndarray


@dataclass(frozen=True)
class ForecastSummary:
    """Aggregate indicators derived from a forecast curve."""

    mode: str
    total_return: float
    avg_return: float
    volatility: float
    direction: str
    phase: str
    return_mode: str


def classify_phase(alpha: float, low: float = PHASE_LOW, high: float = PHASE_HIGH) -> str:
    if not np.isfinite(alpha):
        return "indefinido"
    if alpha < low:
        return "difusiva"
    if alpha < high:
        return "transicao"
    return "coerente"


def describe_phase(alpha: float, entropy: float, low: float = PHASE_LOW, high: float = PHASE_HIGH) -> str:
    phase = classify_phase(alpha, low, high)
    entropy_state = "alta" if entropy > 2.0 else "moderada" if entropy > 1.0 else "baixa"
    return f"{phase} (alpha={alpha:.2f}, entropia {entropy_state})"


def analyze_financial_windows(
    windows: List["ReturnWindow"],
    bin_centers: np.ndarray,
    steps: int = DEFAULT_STEPS,
    fit_min: int = DISPERSION_FIT_MIN,
    fit_max: Optional[int] = None,
    noise_level: float = NOISE_LEVEL,
) -> List[WindowAnalysis]:
    """
    Run classical and quantum walks for each window and compute summary metrics.

    Parameters
    ----------
    windows:
        Output from dados.brutos.financial_loader.generate_return_windows.
    bin_centers:
        Centers returned by discretize_returns; used as physical positions.
    steps:
        Number of walk steps to simulate.
    fit_min, fit_max:
        Step range for dispersion exponent fitting. fit_max=None uses all steps.
    noise_level:
        Mixing level with the uniform distribution when computing sensitivity.
    """
    graph = line_graph(len(bin_centers))
    analyses: List[WindowAnalysis] = []

    for window in windows:
        classical = simulate_classical_walk(
            graph,
            steps,
            initial_distribution=window.distribution,
            target_node=None,
        )

        metrics: Dict[str, ModeMetrics] = {}
        metrics[MODE_CLASSICAL] = _build_metrics(
            MODE_CLASSICAL,
            classical.entropies,
            classical.distributions,
            bin_centers,
            fit_min,
            fit_max,
            noise_level,
        )

        if QISKIT_AVAILABLE:
            hadamard = _simulate_quantum_mode(graph, steps, window.distribution, "hadamard")
            grover = _simulate_quantum_mode(graph, steps, window.distribution, "grover")

            for mode_label, result in (
                (MODE_HADAMARD, hadamard),
                (MODE_GROVER, grover),
            ):
                if result is None:
                    continue
                metrics[mode_label] = _build_metrics(
                    mode_label,
                    result.entropies,
                    result.distributions,
                    bin_centers,
                    fit_min,
                    fit_max,
                    noise_level,
                )
        else:
            metrics[MODE_HADAMARD] = ModeMetrics(
                mode=MODE_HADAMARD,
                entropy_initial=float("nan"),
                entropy_final=float("nan"),
                entropy_change=float("nan"),
                dispersion_alpha=float("nan"),
                dispersion_alpha_noisy=float("nan"),
                dispersion_sensitivity=float("nan"),
            )
            metrics[MODE_GROVER] = ModeMetrics(
                mode=MODE_GROVER,
                entropy_initial=float("nan"),
                entropy_final=float("nan"),
                entropy_change=float("nan"),
                dispersion_alpha=float("nan"),
                dispersion_alpha_noisy=float("nan"),
                dispersion_sensitivity=float("nan"),
            )

        analyses.append(
            WindowAnalysis(
                start=np.datetime64(window.start),
                end=np.datetime64(window.end),
                distribution=window.distribution,
                metrics=metrics,
            )
        )

    return analyses


def _forecast_from_result(
    mode: str,
    distributions: np.ndarray,
    bin_centers: np.ndarray,
    last_price: float,
    return_mode: str = "log",
) -> ForecastCurve:
    # Skip the initial state when computing forward evolution
    future = distributions[1:]
    expected_returns = future @ bin_centers
    if return_mode == "log":
        cumulative_returns = np.cumsum(expected_returns)
        predicted_prices = last_price * np.exp(cumulative_returns)
    elif return_mode == "simple":
        cumulative_returns = np.cumprod(1.0 + expected_returns) - 1.0
        predicted_prices = last_price * (1.0 + cumulative_returns)
    elif return_mode == "diff":
        mad = float(np.median(np.abs(bin_centers - np.median(bin_centers))))
        diff_scale = 1.4826 * mad  # consistent MAD estimator
        if not np.isfinite(diff_scale) or diff_scale == 0.0:
            diff_scale = float(np.std(bin_centers))
        if not np.isfinite(diff_scale) or diff_scale == 0.0:
            diff_scale = float(np.max(np.abs(bin_centers)))
        if not np.isfinite(diff_scale) or diff_scale == 0.0:
            diff_scale = 1.0
        expected_returns = diff_scale * np.tanh(expected_returns / diff_scale)
        if expected_returns.size:
            decay_tau = max(float(expected_returns.size) / 2.0, 1.0)
            decay = np.exp(-np.arange(1, expected_returns.size + 1, dtype=float) / decay_tau)
            expected_returns = expected_returns * decay
        cumulative_returns = np.cumsum(expected_returns)
        predicted_prices = last_price + cumulative_returns
        price_cap = float(3.5 * diff_scale)
        if np.isfinite(price_cap) and price_cap > 0.0:
            lower = last_price - price_cap
            upper = last_price + price_cap
            predicted_prices = np.clip(predicted_prices, lower, upper)
    else:
        raise ValueError("return_mode must be 'log', 'simple', or 'diff'")
    stds = _standard_deviation_series(distributions, bin_centers)[1:]
    return ForecastCurve(
        mode=mode,
        expected_returns=expected_returns,
        cumulative_returns=cumulative_returns,
        predicted_prices=predicted_prices,
        std_per_step=stds,
    )


def _summarize_curve(curve: ForecastCurve, current_phase: str, return_mode: str) -> ForecastSummary:
    total_return = float(curve.cumulative_returns[-1]) if curve.cumulative_returns.size else 0.0
    avg_return = float(curve.expected_returns.mean()) if curve.expected_returns.size else 0.0
    volatility = float(np.sqrt(np.mean(curve.std_per_step ** 2))) if curve.std_per_step.size else 0.0
    threshold = 0.002 if return_mode in ("log", "simple") else 0.05
    if total_return > threshold:
        direction = "alta"
    elif total_return < -threshold:
        direction = "queda"
    else:
        direction = "neutro"
    return ForecastSummary(
        mode=curve.mode,
        total_return=total_return,
        avg_return=avg_return,
        volatility=volatility,
        direction=direction,
        phase=current_phase,
        return_mode=return_mode,
    )


def forecast_from_distribution(
    distribution: np.ndarray,
    bin_centers: np.ndarray,
    last_price: float,
    steps: int = 15,
    phase_by_mode: Optional[Dict[str, str]] = None,
    return_mode: str = "log",
) -> Tuple[List[ForecastCurve], List[ForecastSummary]]:
    """
    Generate forward projections for each walk mode starting from the given distribution.

    Returns lists of detailed curves and aggregated summaries.
    """
    graph = line_graph(len(bin_centers))
    curves: List[ForecastCurve] = []
    summaries: List[ForecastSummary] = []

    classical = simulate_classical_walk(
        graph,
        steps,
        initial_distribution=distribution,
        target_node=None,
    )
    classical_curve = _forecast_from_result(
        MODE_CLASSICAL,
        classical.distributions,
        bin_centers,
        last_price,
        return_mode=return_mode,
    )
    curves.append(classical_curve)
    summaries.append(
        _summarize_curve(
            classical_curve,
            current_phase=(phase_by_mode or {}).get(MODE_CLASSICAL, "indefinido"),
            return_mode=return_mode,
        )
    )

    if QISKIT_AVAILABLE:
        for coin_label, mode in (("hadamard", MODE_HADAMARD), ("grover", MODE_GROVER)):
            result = _simulate_quantum_mode(graph, steps, distribution, coin_label)
            if result is None:
                continue
            curve = _forecast_from_result(
                mode,
                result.distributions,
                bin_centers,
                last_price,
                return_mode=return_mode,
            )
            curves.append(curve)
            summaries.append(
                _summarize_curve(
                    curve,
                    current_phase=(phase_by_mode or {}).get(mode, "indefinido"),
                    return_mode=return_mode,
                )
            )

    return curves, summaries
