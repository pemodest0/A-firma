from __future__ import annotations

import pandas as pd

from scripts.bench.validation.run_profit_pbo_suite import _cscv_splits, _pbo_verdict, _rank_to_omega


def test_rank_to_omega_orders_best_to_worst() -> None:
    assert _rank_to_omega(1, 5) > _rank_to_omega(3, 5) > _rank_to_omega(5, 5)


def test_cscv_splits_avoid_duplicate_complements() -> None:
    idx = pd.Index([f"m{i}" for i in range(16)])
    splits = _cscv_splits(idx, n_slices=8)
    assert len(splits) == 35


def test_pbo_verdict_bands() -> None:
    assert _pbo_verdict(0.05) == "robusto"
    assert _pbo_verdict(0.15) == "aceitavel"
    assert _pbo_verdict(0.30) == "fragil"
    assert _pbo_verdict(0.60) == "provavel_overfit"
