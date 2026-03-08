from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeMoments:
    label: str
    mean: np.ndarray
    cov: np.ndarray
    n_obs: int


def _as_frame(returns: pd.DataFrame | np.ndarray | Iterable[Iterable[float]]) -> pd.DataFrame:
    if isinstance(returns, pd.DataFrame):
        frame = returns.copy()
    else:
        frame = pd.DataFrame(returns)
    if frame.empty:
        raise ValueError("returns frame is empty")
    return frame.apply(pd.to_numeric, errors="coerce")


def _ensure_psd(mat: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    sym = 0.5 * (mat + mat.T)
    vals, vecs = np.linalg.eigh(sym)
    vals = np.clip(vals, float(floor), None)
    return (vecs * vals) @ vecs.T


def covariance_cholesky(cov: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    cov_psd = _ensure_psd(np.asarray(cov, dtype=float), floor=float(floor))
    jitter = 0.0
    eye = np.eye(cov_psd.shape[0], dtype=float)
    for _ in range(6):
        try:
            return np.linalg.cholesky(cov_psd + jitter * eye)
        except np.linalg.LinAlgError:
            jitter = max(float(floor), 10.0 * (jitter if jitter > 0.0 else float(floor)))
    raise np.linalg.LinAlgError("failed to compute Cholesky factor for covariance matrix")


def estimate_regime_moments(
    returns: pd.DataFrame | np.ndarray | Iterable[Iterable[float]],
    regime: pd.Series | Iterable[str],
    *,
    min_obs: int = 20,
    floor: float = 1e-8,
) -> dict[str, RegimeMoments]:
    frame = _as_frame(returns)
    reg = pd.Series(regime, index=frame.index).astype(str).str.lower()
    out: dict[str, RegimeMoments] = {}
    global_mean = frame.mean(skipna=True).fillna(0.0).to_numpy(dtype=float)
    global_cov = _ensure_psd(frame.cov(min_periods=2).fillna(0.0).to_numpy(dtype=float), floor=float(floor))
    for label, sub in frame.groupby(reg, sort=True):
        clean = sub.dropna(how="all")
        if len(clean) < int(min_obs):
            out[str(label)] = RegimeMoments(label=str(label), mean=global_mean, cov=global_cov, n_obs=int(len(clean)))
            continue
        mean = clean.mean(skipna=True).fillna(0.0).to_numpy(dtype=float)
        cov = clean.cov(min_periods=2).fillna(0.0).to_numpy(dtype=float)
        out[str(label)] = RegimeMoments(
            label=str(label),
            mean=mean,
            cov=_ensure_psd(cov, floor=float(floor)),
            n_obs=int(len(clean)),
        )
    if not out:
        out["default"] = RegimeMoments(label="default", mean=global_mean, cov=global_cov, n_obs=int(len(frame)))
    return out


def estimate_transition_matrix(
    regime: pd.Series | Iterable[str],
    *,
    state_order: list[str] | None = None,
    smoothing: float = 1.0,
) -> tuple[list[str], np.ndarray]:
    reg = pd.Series(regime).astype(str).str.lower().dropna()
    states = list(state_order or sorted(pd.Index(reg.unique()).astype(str).tolist()))
    if not states:
        raise ValueError("no regime states available")
    idx = {state: pos for pos, state in enumerate(states)}
    counts = np.full((len(states), len(states)), float(smoothing), dtype=float)
    arr = reg.to_numpy(dtype=object)
    for prev, curr in zip(arr[:-1], arr[1:]):
        if prev not in idx or curr not in idx:
            continue
        counts[idx[str(prev)], idx[str(curr)]] += 1.0
    row_sum = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        transition = np.divide(counts, row_sum, out=np.full_like(counts, 1.0 / len(states)), where=row_sum > 0)
    return states, transition


def simulate_correlated_paths(
    *,
    mean: np.ndarray,
    cov: np.ndarray,
    horizon: int,
    n_paths: int,
    random_state: int = 7,
) -> np.ndarray:
    mu = np.asarray(mean, dtype=float)
    chol = covariance_cholesky(np.asarray(cov, dtype=float))
    rng = np.random.default_rng(int(random_state))
    z = rng.standard_normal((int(n_paths), int(horizon), mu.shape[0]))
    return mu[None, None, :] + np.einsum("ij,thj->thi", chol, z, optimize=True)


def simulate_regime_conditioned_paths(
    *,
    regime_moments: dict[str, RegimeMoments],
    transition_matrix: np.ndarray,
    states: list[str],
    start_state: str,
    horizon: int,
    n_paths: int,
    random_state: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    if str(start_state) not in regime_moments:
        raise KeyError(f"unknown start_state={start_state}")
    rng = np.random.default_rng(int(random_state))
    state_to_idx = {state: pos for pos, state in enumerate(states)}
    dims = regime_moments[str(start_state)].mean.shape[0]
    paths = np.zeros((int(n_paths), int(horizon), dims), dtype=float)
    state_paths = np.empty((int(n_paths), int(horizon)), dtype=object)
    transition = np.asarray(transition_matrix, dtype=float)
    means = {state: np.asarray(mom.mean, dtype=float) for state, mom in regime_moments.items()}
    factors = {state: covariance_cholesky(mom.cov) for state, mom in regime_moments.items()}
    current = np.full(int(n_paths), int(state_to_idx[str(start_state)]), dtype=int)
    for t in range(int(horizon)):
        noise = rng.standard_normal((int(n_paths), dims))
        for state_idx in np.unique(current):
            mask = current == int(state_idx)
            if not np.any(mask):
                continue
            state = states[int(state_idx)]
            state_paths[mask, t] = state
            paths[mask, t, :] = noise[mask] @ factors[state].T + means[state]
        if t + 1 < int(horizon):
            next_state = np.zeros_like(current)
            draws = rng.random(int(n_paths))
            for state_idx in np.unique(current):
                mask = current == int(state_idx)
                probs = transition[int(state_idx)]
                cdf = np.cumsum(probs)
                next_state[mask] = np.searchsorted(cdf, draws[mask], side="right")
            current = next_state
    return paths, state_paths


def summarize_portfolio_distribution(
    simulated_returns: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    rets = np.asarray(simulated_returns, dtype=float)
    w = np.asarray(weights, dtype=float)
    if rets.ndim != 3:
        raise ValueError("simulated_returns must have shape (paths, horizon, assets)")
    port_daily = np.einsum("tha,a->th", rets, w, optimize=True)
    cumulative = np.prod(1.0 + port_daily, axis=1) - 1.0
    path_wealth = np.cumprod(1.0 + port_daily, axis=1)
    path_peaks = np.maximum.accumulate(path_wealth, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdowns = path_wealth / path_peaks - 1.0
    max_dd = np.nanmin(drawdowns, axis=1)
    q05 = float(np.nanquantile(cumulative, 0.05))
    tail = cumulative[cumulative <= q05]
    return {
        "terminal_p05": q05,
        "terminal_p50": float(np.nanquantile(cumulative, 0.50)),
        "terminal_p95": float(np.nanquantile(cumulative, 0.95)),
        "expected_shortfall_p05": float(np.nanmean(tail)) if tail.size else q05,
        "ruin_prob_m10": float(np.mean(cumulative <= -0.10)),
        "ruin_prob_m20": float(np.mean(cumulative <= -0.20)),
        "max_drawdown_p50": float(np.nanquantile(max_dd, 0.50)),
        "max_drawdown_p95": float(np.nanquantile(max_dd, 0.95)),
    }


def rolling_regime_conditioned_summary(
    returns: pd.DataFrame,
    regime: pd.Series,
    weights: pd.DataFrame,
    *,
    states: list[str] | None = None,
    lookback: int = 252,
    horizon: int = 21,
    n_paths: int = 1000,
    step: int = 21,
    min_obs: int = 20,
    random_state: int = 7,
) -> pd.DataFrame:
    frame = _as_frame(returns)
    reg = pd.Series(regime, index=frame.index).astype(str).str.lower()
    w = weights.reindex(frame.index).fillna(0.0).astype(float)
    rows: list[dict[str, float | str | pd.Timestamp]] = []
    lookback = int(max(60, lookback))
    step = int(max(1, step))
    points = list(range(lookback, len(frame), step))
    for pos in points:
        end_idx = frame.index[pos]
        hist = frame.iloc[max(0, pos - lookback) : pos]
        hist_reg = reg.iloc[max(0, pos - lookback) : pos]
        if hist.dropna(how="all").empty:
            continue
        moments = estimate_regime_moments(hist, hist_reg, min_obs=int(min_obs))
        state_names = list(states or sorted(moments.keys()))
        state_names, transition = estimate_transition_matrix(hist_reg, state_order=state_names)
        current_state = str(reg.iloc[pos])
        if current_state not in moments:
            current_state = state_names[0]
        sim, state_path = simulate_regime_conditioned_paths(
            regime_moments=moments,
            transition_matrix=transition,
            states=state_names,
            start_state=current_state,
            horizon=int(horizon),
            n_paths=int(n_paths),
            random_state=int(random_state + pos),
        )
        stats = summarize_portfolio_distribution(sim, w.loc[end_idx].to_numpy(dtype=float))
        rows.append(
            {
                "date": pd.Timestamp(end_idx),
                "state": current_state,
                "dominant_next_state": str(pd.Series(state_path[:, -1]).mode().iloc[0]) if state_path.size else current_state,
                **stats,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.set_index("date").sort_index()
    return out.reindex(frame.index).ffill()
