from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

ModeState = Literal["ATTACK", "PROTECT"]


@dataclass(frozen=True)
class AsymmetricPolicyConfig:
    enter_attack_threshold: float = 0.74
    stay_attack_threshold: float = 0.60
    defense_threshold: float = 0.55
    release_threshold: float = 0.66

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def next_mode_state(
    *,
    current_state: ModeState,
    attack_signal: float,
    defense_signal: float,
    config: AsymmetricPolicyConfig,
) -> ModeState:
    attack = float(attack_signal)
    defense = float(defense_signal)
    if str(current_state).upper() == "ATTACK":
        if defense >= float(config.defense_threshold) or attack < float(config.stay_attack_threshold):
            return "PROTECT"
        return "ATTACK"
    if attack >= float(config.enter_attack_threshold) and defense < float(config.release_threshold):
        return "ATTACK"
    return "PROTECT"
