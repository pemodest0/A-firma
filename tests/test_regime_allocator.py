from engine.portfolio.regime_allocator import map_risk_state_to_exposure


def test_allocator_returns_cash_biased_when_year_is_bad():
    profile = map_risk_state_to_exposure(
        {"confidence_score": 0.9, "structural_stress": 0.1},
        {"period_action": "NORMAL", "year_bad_state": True},
    )
    assert profile.state == "CASH_BIASED"
    assert profile.cash_fraction > profile.attack_fraction


def test_allocator_returns_attack_full_for_clean_high_confidence():
    profile = map_risk_state_to_exposure(
        {"confidence_score": 0.9, "structural_stress": 0.2},
        {"period_action": "NORMAL", "year_bad_state": False},
    )
    assert profile.state == "ATTACK_FULL"
    assert profile.attack_fraction == 1.0


def test_allocator_returns_protected_when_stress_is_high():
    profile = map_risk_state_to_exposure(
        {"confidence_score": 0.7, "structural_stress": 0.9},
        {"period_action": "NORMAL", "year_bad_state": False},
    )
    assert profile.state == "PROTECTED"
    assert profile.protection_fraction > profile.attack_fraction
