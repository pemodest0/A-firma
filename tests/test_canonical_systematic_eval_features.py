from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import math

import pandas as pd
import pytest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ops" / "run_canonical_systematic_eval.py"
_SPEC = importlib.util.spec_from_file_location("run_canonical_systematic_eval", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_continuous_risk_budget_recognizes_low_stress_as_dispersion() -> None:
    months = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06", "2025-07"]
    state = pd.DataFrame(
        {
            "global_score": [0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.20],
            "regime": ["stress", "transition", "transition", "stable", "stable", "stable", "dispersion"],
        },
        index=months,
    )
    corr_signal = pd.Series([0.70, 0.68, 0.65, 0.55, 0.45, 0.40, 0.20], index=months, dtype=float)
    vol_signal = pd.Series([0.30, 0.28, 0.27, 0.25, 0.22, 0.20, 0.10], index=months, dtype=float)
    cfg = _MODULE.ContinuousRiskConfig(
        enabled=True,
        weight_global_score=0.5,
        weight_corr=0.3,
        weight_vol=0.2,
        regime_bias=0.8,
        score_power=1.0,
    )

    rb, bucket, score, *_ = _MODULE._risk_budget(
        mode="score",
        prev_ym="2025-07",
        prev_idx=len(months),
        months=months,
        state=state,
        rb_stress=0.2,
        rb_transition=0.5,
        rb_stable=0.8,
        rb_dispersion=1.1,
        corr_signal=corr_signal,
        vol_signal=vol_signal,
        continuous_cfg=cfg,
    )

    assert bucket == "dispersion"
    assert score < 0.25
    assert rb > 0.95


def test_load_external_benchmark_monthly_compounds_daily_returns(tmp_path: Path) -> None:
    prices_dir = tmp_path / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "date": ["2026-01-02", "2026-01-05", "2026-02-03"],
            "r": [0.01, -0.02, 0.03],
        }
    ).to_csv(prices_dir / "SPY.csv", index=False)

    monthly = _MODULE._load_external_benchmark_monthly(prices_dir, "SPY", pd.Index(["2026-01", "2026-02"]))

    assert monthly.index.tolist() == ["2026-01", "2026-02"]
    assert monthly.round(6).tolist() == [round(math.expm1(-0.01), 6), round(math.expm1(0.03), 6)]


def test_load_external_benchmark_monthly_raises_on_missing_coverage(tmp_path: Path) -> None:
    prices_dir = tmp_path / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": ["2026-01-02"], "r": [0.01]}).to_csv(prices_dir / "SPY.csv", index=False)

    with pytest.raises(ValueError, match="missing monthly coverage"):
        _MODULE._load_external_benchmark_monthly(prices_dir, "SPY", pd.Index(["2026-01", "2026-02"]))
