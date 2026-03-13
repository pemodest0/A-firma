from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class YearDefenseConfig:
    ytd_return_floor: float = -0.10
    ytd_drawdown_floor: float = -0.14
    stress_trigger: float = 0.62
    min_bad_days: int = 25

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def compute_ytd_stress(ytd_returns: pd.Series, drawdown_path: pd.Series, signal_bundle: dict[str, Any]) -> float:
    ytd_ret = float(pd.to_numeric(ytd_returns, errors="coerce").fillna(0.0).iloc[-1]) if not ytd_returns.empty else 0.0
    ytd_dd = float(pd.to_numeric(drawdown_path, errors="coerce").fillna(0.0).min()) if not drawdown_path.empty else 0.0
    structural = float(signal_bundle.get("structural_stress", 0.0) or 0.0)
    liquidation = float(signal_bundle.get("liquidation", 0.0) or 0.0)
    breadth = float(signal_bundle.get("breadth", 0.5) or 0.5)
    stress = (
        0.36 * np.clip(-ytd_ret / 0.20, 0.0, 1.0)
        + 0.28 * np.clip(-ytd_dd / 0.25, 0.0, 1.0)
        + 0.18 * np.clip(structural, 0.0, 1.0)
        + 0.10 * np.clip(liquidation, 0.0, 1.0)
        + 0.08 * np.clip(1.0 - breadth, 0.0, 1.0)
    )
    return float(np.clip(stress, 0.0, 1.0))


def year_bad_state_trigger(metrics: dict[str, Any], config: YearDefenseConfig) -> bool:
    ytd_return = float(metrics.get("ytd_return", 0.0) or 0.0)
    ytd_drawdown = float(metrics.get("ytd_drawdown", 0.0) or 0.0)
    year_stress = float(metrics.get("year_stress", 0.0) or 0.0)
    bad_days = int(metrics.get("bad_days", 0) or 0)
    return bool(
        (
            ytd_return <= float(config.ytd_return_floor)
            or ytd_drawdown <= float(config.ytd_drawdown_floor)
            or year_stress >= float(config.stress_trigger)
        )
        and bad_days >= int(config.min_bad_days)
    )
