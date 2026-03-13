from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

RiskAction = Literal["NORMAL", "REDUCED_ATTACK", "PROTECTED", "CASH_HEAVY"]


@dataclass(frozen=True)
class PeriodLossGuardConfig:
    monthly_reduce_threshold: float = -0.03
    monthly_protect_threshold: float = -0.06
    monthly_cash_threshold: float = -0.10
    quarterly_reduce_threshold: float = -0.05
    quarterly_protect_threshold: float = -0.09
    quarterly_cash_threshold: float = -0.14

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _resolve_action(value: float, *, reduce_threshold: float, protect_threshold: float, cash_threshold: float) -> RiskAction:
    x = float(value)
    if x <= float(cash_threshold):
        return "CASH_HEAVY"
    if x <= float(protect_threshold):
        return "PROTECTED"
    if x <= float(reduce_threshold):
        return "REDUCED_ATTACK"
    return "NORMAL"


def monthly_loss_guard(month_return: float, config: PeriodLossGuardConfig) -> RiskAction:
    return _resolve_action(
        float(month_return),
        reduce_threshold=float(config.monthly_reduce_threshold),
        protect_threshold=float(config.monthly_protect_threshold),
        cash_threshold=float(config.monthly_cash_threshold),
    )


def quarterly_loss_guard(quarter_return: float, config: PeriodLossGuardConfig) -> RiskAction:
    return _resolve_action(
        float(quarter_return),
        reduce_threshold=float(config.quarterly_reduce_threshold),
        protect_threshold=float(config.quarterly_protect_threshold),
        cash_threshold=float(config.quarterly_cash_threshold),
    )


def combine_guard_actions(*actions: RiskAction) -> RiskAction:
    rank = {
        "NORMAL": 0,
        "REDUCED_ATTACK": 1,
        "PROTECTED": 2,
        "CASH_HEAVY": 3,
    }
    best = "NORMAL"
    best_rank = -1
    for action in actions:
        current_rank = rank.get(str(action), -1)
        if current_rank > best_rank:
            best = str(action)
            best_rank = current_rank
    return best  # type: ignore[return-value]
