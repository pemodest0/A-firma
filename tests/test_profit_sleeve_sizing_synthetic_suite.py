from __future__ import annotations

from scripts.bench.validation.run_profit_sleeve_sizing_synthetic_suite import (
    _human_label,
    _simulate_synthetic_shift,
    _synthetic_structural_series,
)


def test_human_label_maps_known_candidates() -> None:
    assert _human_label("pure_crypto_attack") == "Cripto puro agressivo"
    assert _human_label("unknown_candidate") == "unknown_candidate"


def test_synthetic_shift_builds_structural_series() -> None:
    returns, states, sector_map = _simulate_synthetic_shift(seed=11, n_sectors=3, assets_per_sector=4)
    assert not returns.empty
    assert not states.empty
    assert not sector_map.empty

    series = _synthetic_structural_series(returns, window=60)
    assert not series.empty
    assert {"date", "p1", "deff", "lambda1"}.issubset(series.columns)
