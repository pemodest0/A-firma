from __future__ import annotations

import pandas as pd

from scripts.bench.validation.run_profit_execution_resilience_suite import (
    _capacity_profile_for_candidate,
    _extra_bps_from_utilization,
)


def test_extra_bps_grows_with_utilization() -> None:
    low = _extra_bps_from_utilization(0.10, sleeve="crypto")
    mid = _extra_bps_from_utilization(0.75, sleeve="crypto")
    high = _extra_bps_from_utilization(1.50, sleeve="crypto")
    assert low == 0.0
    assert mid > low
    assert high > mid


def test_attack_profile_is_harsher_than_main() -> None:
    attack = _capacity_profile_for_candidate("alpha_attack_major8_equity25")
    main = _capacity_profile_for_candidate("meta_major8_eq_a2r1")
    assert attack.delay_crypto_days >= main.delay_crypto_days
    assert attack.crypto_capacity_brl < main.crypto_capacity_brl


def test_profile_factory_returns_finite_capacities() -> None:
    for candidate_id in [
        "meta_major8_eq_a2r1",
        "alpha_attack_major8_equity25",
        "meta_major8_eq_a2r1_mc_guard",
        "pure_crypto_attack",
    ]:
        profile = _capacity_profile_for_candidate(candidate_id)
        assert pd.notna(profile.crypto_capacity_brl)
        assert pd.notna(profile.equity_capacity_brl)
        assert profile.crypto_capacity_brl > 0.0
        assert profile.equity_capacity_brl > 0.0
