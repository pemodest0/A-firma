from __future__ import annotations

import pandas as pd

from scripts.bench.validation.run_profit_alpha_hardening_suite import (
    _build_promoted_attack_confidence_score,
)


def test_attack_confidence_score_extends_past_regime_tail() -> None:
    idx = pd.date_range("2025-07-01", periods=260, freq="D")
    attack_returns = pd.DataFrame(
        {
            "crypto": [0.01 if i % 7 else -0.005 for i in range(len(idx))],
            "equity": [0.004 if i % 5 else -0.002 for i in range(len(idx))],
        },
        index=idx,
        dtype=float,
    )
    context = {
        "btc_prices": pd.Series([100 + i for i in range(len(idx))], index=idx, dtype=float),
        "spy_prices": pd.Series([200 + i for i in range(len(idx))], index=idx, dtype=float),
        "regime_series": pd.Series(
            ["stable", "transition", "stable", "stable"],
            index=idx[:4],
            dtype=object,
        ),
    }

    score = _build_promoted_attack_confidence_score(context, attack_returns)

    assert list(score.index) == list(idx)
    assert pd.notna(score.iloc[-1])


def test_attack_confidence_score_defaults_when_regime_is_empty() -> None:
    idx = pd.date_range("2025-07-01", periods=260, freq="D")
    attack_returns = pd.DataFrame(
        {
            "crypto": [0.008 if i % 6 else -0.004 for i in range(len(idx))],
            "equity": [0.003 if i % 4 else -0.001 for i in range(len(idx))],
        },
        index=idx,
        dtype=float,
    )
    context = {
        "btc_prices": pd.Series([100 + i for i in range(len(idx))], index=idx, dtype=float),
        "spy_prices": pd.Series([200 + i for i in range(len(idx))], index=idx, dtype=float),
        "regime_series": pd.Series(dtype=object),
    }

    score = _build_promoted_attack_confidence_score(context, attack_returns)

    assert list(score.index) == list(idx)
    assert pd.notna(score.iloc[-1])


def test_attack_confidence_score_does_not_backfill_future_regime() -> None:
    idx = pd.date_range("2025-07-01", periods=260, freq="D")
    attack_returns = pd.DataFrame(
        {
            "crypto": [0.006 if i % 9 else -0.003 for i in range(len(idx))],
            "equity": [0.002 if i % 5 else -0.001 for i in range(len(idx))],
        },
        index=idx,
        dtype=float,
    )
    base_context = {
        "btc_prices": pd.Series([100 + i for i in range(len(idx))], index=idx, dtype=float),
        "spy_prices": pd.Series([200 + i for i in range(len(idx))], index=idx, dtype=float),
    }
    score_empty = _build_promoted_attack_confidence_score(
        {**base_context, "regime_series": pd.Series(dtype=object)},
        attack_returns,
    )
    future_stress = pd.Series(["stress"] * 30, index=idx[-30:], dtype=object)
    score_future = _build_promoted_attack_confidence_score(
        {**base_context, "regime_series": future_stress},
        attack_returns,
    )

    early_date = idx[150]
    assert float(score_empty.loc[early_date]) == float(score_future.loc[early_date])
