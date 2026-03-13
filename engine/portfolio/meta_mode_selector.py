from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetaModeSelectorConfig:
    training_months: int = 36
    min_training_months: int = 24
    neighbor_months: int = 12
    downside_penalty: float = 1.15
    underperformance_penalty: float = 0.55
    tail_penalty: float = 0.35
    switch_penalty: float = 0.01
    min_neighbors: int = 6
    fallback_mode: str = "best_recent"


def monthly_last(panel: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    if isinstance(panel, pd.Series):
        return panel.resample("ME").last()
    return panel.resample("ME").last()


def monthly_total_return(ret: pd.Series) -> pd.Series:
    values = pd.to_numeric(ret, errors="coerce").fillna(0.0).astype(float)
    return (1.0 + values).resample("ME").prod() - 1.0


def _candidate_score(
    returns: pd.Series,
    benchmark: pd.Series,
    *,
    switch_penalty: float,
    switched: bool,
    downside_penalty: float,
    underperformance_penalty: float,
    tail_penalty: float,
) -> float:
    ret = pd.to_numeric(returns, errors="coerce").dropna().astype(float)
    bench = pd.to_numeric(benchmark.reindex(ret.index), errors="coerce").fillna(0.0).astype(float)
    if ret.empty:
        return float("-inf")
    mean_ret = float(ret.mean())
    losses = ret[ret < 0.0]
    downside = abs(float(losses.mean())) if not losses.empty else 0.0
    underperf = float((bench - ret).clip(lower=0.0).mean())
    tail = abs(float(ret.min())) if not ret.empty else 0.0
    score = (
        mean_ret
        - float(downside_penalty) * downside
        - float(underperformance_penalty) * underperf
        - float(tail_penalty) * tail
    )
    if switched:
        score -= float(switch_penalty)
    return float(score)


def _select_history(
    train_features: pd.DataFrame,
    current_features: pd.Series,
    *,
    neighbor_months: int,
    min_neighbors: int,
) -> pd.Index:
    features = train_features.apply(pd.to_numeric, errors="coerce").astype(float)
    current = pd.to_numeric(current_features, errors="coerce").astype(float)
    usable_cols = [
        col
        for col in features.columns
        if pd.notna(current.get(col))
        and pd.notna(features[col]).sum() >= max(6, min_neighbors)
    ]
    if not usable_cols:
        return features.index
    sub = features[usable_cols]
    cur = current[usable_cols]
    mu = sub.mean(axis=0)
    sigma = sub.std(axis=0, ddof=0).replace(0.0, np.nan)
    norm = (sub - mu).divide(sigma)
    cur_norm = (cur - mu).divide(sigma)
    valid = norm.notna().all(axis=1) & cur_norm.notna()
    if not bool(valid.any()):
        return features.index
    distances = ((norm.loc[valid] - cur_norm) ** 2).sum(axis=1).pow(0.5).sort_values()
    take = max(int(min_neighbors), min(int(neighbor_months), int(len(distances))))
    return distances.index[:take]


def run_causal_meta_mode_selector(
    *,
    feature_frame: pd.DataFrame,
    candidate_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    config: MetaModeSelectorConfig | None = None,
) -> pd.DataFrame:
    cfg = config or MetaModeSelectorConfig()
    idx = (
        candidate_returns.index.intersection(feature_frame.index)
        .intersection(benchmark_returns.index)
        .sort_values()
    )
    if idx.empty:
        return pd.DataFrame(columns=["selected_mode", "selected_score", "runner_up", "runner_up_score", "score_gap", "neighbors_used", "training_months_used", "selection_confidence"])
    features = feature_frame.reindex(idx)
    returns = candidate_returns.reindex(idx).apply(pd.to_numeric, errors="coerce").astype(float)
    benchmark = pd.to_numeric(benchmark_returns.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    rows: list[dict[str, Any]] = []
    previous_mode: str | None = None
    for pos, dt in enumerate(idx):
        train_end = pos
        train_start = max(0, train_end - int(cfg.training_months))
        train_idx = idx[train_start:train_end]
        if len(train_idx) < int(cfg.min_training_months):
            rows.append(
                {
                    "date": dt,
                    "selected_mode": None,
                    "selected_score": float("nan"),
                    "runner_up": None,
                    "runner_up_score": float("nan"),
                    "score_gap": float("nan"),
                    "neighbors_used": 0,
                    "training_months_used": int(len(train_idx)),
                    "selection_confidence": float("nan"),
                }
            )
            continue
        hist_idx = _select_history(
            features.loc[train_idx],
            features.loc[dt],
            neighbor_months=int(cfg.neighbor_months),
            min_neighbors=int(cfg.min_neighbors),
        )
        if len(hist_idx) < int(cfg.min_neighbors):
            hist_idx = train_idx
        scores: dict[str, float] = {}
        for candidate in returns.columns:
            scores[str(candidate)] = _candidate_score(
                returns.loc[hist_idx, candidate],
                benchmark.loc[hist_idx],
                switch_penalty=float(cfg.switch_penalty),
                switched=previous_mode is not None and str(candidate) != previous_mode,
                downside_penalty=float(cfg.downside_penalty),
                underperformance_penalty=float(cfg.underperformance_penalty),
                tail_penalty=float(cfg.tail_penalty),
            )
        ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        chosen_mode, chosen_score = ranking[0]
        runner_up, runner_score = ranking[1] if len(ranking) > 1 else ("", float("nan"))
        gap = float(chosen_score - runner_score) if pd.notna(runner_score) else float("nan")
        confidence = float(1.0 / (1.0 + np.exp(-12.0 * gap))) if pd.notna(gap) else float("nan")
        previous_mode = str(chosen_mode)
        rows.append(
            {
                "date": dt,
                "selected_mode": str(chosen_mode),
                "selected_score": float(chosen_score),
                "runner_up": str(runner_up),
                "runner_up_score": float(runner_score) if pd.notna(runner_score) else float("nan"),
                "score_gap": gap,
                "neighbors_used": int(len(hist_idx)),
                "training_months_used": int(len(train_idx)),
                "selection_confidence": confidence,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.set_index("date").sort_index()
