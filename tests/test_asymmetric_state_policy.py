from engine.portfolio.asymmetric_state_policy import AsymmetricPolicyConfig, next_mode_state


def test_attack_state_exits_when_defense_rises():
    cfg = AsymmetricPolicyConfig()
    state = next_mode_state(
        current_state="ATTACK",
        attack_signal=0.7,
        defense_signal=0.7,
        config=cfg,
    )
    assert state == "PROTECT"


def test_protect_state_requires_stronger_reentry_signal():
    cfg = AsymmetricPolicyConfig()
    assert (
        next_mode_state(
            current_state="PROTECT",
            attack_signal=0.65,
            defense_signal=0.1,
            config=cfg,
        )
        == "PROTECT"
    )
    assert (
        next_mode_state(
            current_state="PROTECT",
            attack_signal=0.8,
            defense_signal=0.1,
            config=cfg,
        )
        == "ATTACK"
    )
