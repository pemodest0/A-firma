from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ExposureProfile:
    state: str
    attack_fraction: float
    protection_fraction: float
    cash_fraction: float
    crypto_cap: float
    turnover_cap: float

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


def map_risk_state_to_exposure(signal_bundle: dict[str, Any], guards: dict[str, Any], config: dict[str, Any] | None = None) -> ExposureProfile:
    cfg = dict(config or {})
    confidence = float(signal_bundle.get("confidence_score", 0.5) or 0.5)
    stress = float(signal_bundle.get("structural_stress", 0.0) or 0.0)
    yearly_bad = bool(guards.get("year_bad_state", False))
    guard_action = str(guards.get("period_action", "NORMAL")).upper()

    if yearly_bad or guard_action == "CASH_HEAVY":
        return ExposureProfile(
            state="CASH_BIASED",
            attack_fraction=0.10,
            protection_fraction=0.35,
            cash_fraction=0.55,
            crypto_cap=float(cfg.get("cash_biased_crypto_cap", 0.20)),
            turnover_cap=float(cfg.get("cash_biased_turnover_cap", 0.10)),
        )
    if guard_action == "PROTECTED" or stress >= float(cfg.get("protected_stress_threshold", 0.72)):
        return ExposureProfile(
            state="PROTECTED",
            attack_fraction=0.25,
            protection_fraction=0.65,
            cash_fraction=0.10,
            crypto_cap=float(cfg.get("protected_crypto_cap", 0.35)),
            turnover_cap=float(cfg.get("protected_turnover_cap", 0.16)),
        )
    if guard_action == "REDUCED_ATTACK" or confidence < float(cfg.get("neutral_confidence_threshold", 0.58)):
        return ExposureProfile(
            state="NEUTRAL",
            attack_fraction=0.52,
            protection_fraction=0.38,
            cash_fraction=0.10,
            crypto_cap=float(cfg.get("neutral_crypto_cap", 0.55)),
            turnover_cap=float(cfg.get("neutral_turnover_cap", 0.22)),
        )
    if confidence >= float(cfg.get("attack_full_confidence_threshold", 0.76)) and stress <= float(cfg.get("attack_full_stress_threshold", 0.45)):
        return ExposureProfile(
            state="ATTACK_FULL",
            attack_fraction=1.00,
            protection_fraction=0.00,
            cash_fraction=0.00,
            crypto_cap=float(cfg.get("attack_full_crypto_cap", 0.92)),
            turnover_cap=float(cfg.get("attack_full_turnover_cap", 0.50)),
        )
    return ExposureProfile(
        state="ATTACK_PARTIAL",
        attack_fraction=0.78,
        protection_fraction=0.22,
        cash_fraction=0.00,
        crypto_cap=float(cfg.get("attack_partial_crypto_cap", 0.78)),
        turnover_cap=float(cfg.get("attack_partial_turnover_cap", 0.34)),
    )
