from __future__ import annotations

from engine.portfolio.mode_confidence import decide_attack_vs_protection


def test_decision_prefers_attack_when_structure_is_clean() -> None:
    decision = decide_attack_vs_protection(
        structural_risk_level="dispersion",
        structural_confidence_score=0.72,
        vigilance_status="ok",
        vigilance_alert_count=0,
        pbo_verdict="robusto",
        attack_underperform_prob_63=0.48,
        attack_top3_retention=0.52,
        attack_drawdown=-0.58,
        protection_drawdown=-0.51,
        execution_winner="Modo ataque",
    )
    assert decision.recommended_mode == "ataque"
    assert decision.confidence_level in {"média", "alta"}


def test_decision_prefers_protection_when_crypto_fragility_is_high() -> None:
    decision = decide_attack_vs_protection(
        structural_risk_level="transition",
        structural_confidence_score=0.41,
        vigilance_status="warn",
        vigilance_alert_count=3,
        pbo_verdict="robusto",
        attack_underperform_prob_63=0.63,
        attack_top3_retention=0.28,
        attack_drawdown=-0.81,
        protection_drawdown=-0.57,
        execution_winner="Modo principal com guarda",
    )
    assert decision.recommended_mode == "proteção"
    assert decision.confidence_score > 0.0
    assert any("cripto" in reason.lower() for reason in decision.reasons)
