from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


@dataclass(frozen=True)
class HMMChallengerResult:
    states: pd.Series
    regime_label: pd.Series
    risk_on_probability: pd.Series
    state_probabilities: pd.DataFrame
    state_summary: pd.DataFrame


def build_hmm_feature_frame(
    *,
    primary_ret: pd.Series,
    secondary_ret: pd.Series | None = None,
    volatility_window: int = 21,
) -> pd.DataFrame:
    primary = pd.to_numeric(primary_ret, errors="coerce").astype(float)
    frame = pd.DataFrame({"primary_ret": primary})
    if secondary_ret is not None:
        secondary = pd.to_numeric(secondary_ret, errors="coerce").reindex(primary.index).astype(float)
        frame["secondary_ret"] = secondary
        frame["spread"] = primary - secondary
    frame["primary_vol"] = primary.rolling(int(volatility_window), min_periods=max(5, int(volatility_window) // 2)).std(ddof=0)
    frame["primary_trend"] = (1.0 + primary.fillna(0.0)).rolling(63, min_periods=21).apply(np.prod, raw=True) - 1.0
    if secondary_ret is not None:
        frame["secondary_vol"] = frame["secondary_ret"].rolling(int(volatility_window), min_periods=max(5, int(volatility_window) // 2)).std(ddof=0)
    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def fit_hmm_challenger(
    features: pd.DataFrame,
    *,
    n_states: int = 3,
    train_end: pd.Timestamp | None = None,
    random_state: int = 7,
    n_iter: int = 300,
) -> HMMChallengerResult:
    feat = features.copy().astype(float)
    feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
    if feat.empty:
        raise ValueError("empty features for HMM")
    train = feat.loc[:train_end] if train_end is not None else feat
    if len(train) < int(n_states) * 15:
        raise ValueError("not enough observations to fit HMM challenger")
    model = GaussianHMM(
        n_components=int(n_states),
        covariance_type="full",
        random_state=int(random_state),
        n_iter=int(n_iter),
    )
    model.fit(train.to_numpy(dtype=float))
    probs = model.predict_proba(feat.to_numpy(dtype=float))
    states = pd.Series(model.predict(feat.to_numpy(dtype=float)), index=feat.index, dtype=int)

    summary_rows: list[dict[str, float | int | str]] = []
    state_labels: dict[int, str] = {}
    for state in range(int(n_states)):
        mask = states == state
        sub = feat.loc[mask]
        ann_mean = float(sub["primary_ret"].mean() * 252.0) if not sub.empty else float("nan")
        vol = float(sub["primary_ret"].std(ddof=0) * np.sqrt(252.0)) if not sub.empty else float("nan")
        trend = float(sub["primary_trend"].mean()) if "primary_trend" in sub else float("nan")
        score = ann_mean - 0.75 * vol + 0.5 * trend
        summary_rows.append(
            {
                "state": int(state),
                "ann_mean": ann_mean,
                "ann_vol": vol,
                "trend": trend,
                "score": score,
                "n_obs": int(mask.sum()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("score", ascending=False).reset_index(drop=True)
    if len(summary) >= 1:
        state_labels[int(summary.iloc[0]["state"])] = "risk_on"
    if len(summary) >= 2:
        state_labels[int(summary.iloc[-1]["state"])] = "risk_off"
    for state in range(int(n_states)):
        state_labels.setdefault(state, "neutral")

    regime_label = states.map(state_labels).astype(str)
    risk_on_cols = [state for state, label in state_labels.items() if label == "risk_on"]
    if risk_on_cols:
        risk_on_probability = pd.Series(probs[:, risk_on_cols].sum(axis=1), index=feat.index, dtype=float)
    else:
        risk_on_probability = pd.Series(np.zeros(len(feat), dtype=float), index=feat.index, dtype=float)
    prob_cols = [f"state_{i}" for i in range(probs.shape[1])]
    state_probabilities = pd.DataFrame(probs, index=feat.index, columns=prob_cols)
    state_summary = summary.assign(regime_label=summary["state"].map(state_labels).fillna("neutral"))
    return HMMChallengerResult(
        states=states,
        regime_label=regime_label,
        risk_on_probability=risk_on_probability,
        state_probabilities=state_probabilities,
        state_summary=state_summary,
    )
