from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketCostProfile:
    name: str
    brokerage_bps: float
    spread_bps: float
    slippage_bps: float
    tax_rate: float = 0.0
    tax_mode: str = "monthly_positive"

    @property
    def transaction_cost_bps(self) -> float:
        return float(self.brokerage_bps) + float(self.spread_bps) + float(self.slippage_bps)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["transaction_cost_bps"] = float(self.transaction_cost_bps)
        return d


def _to_series(values: Any) -> pd.Series:
    s = pd.Series(values, dtype="float64")
    s = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
    return s


def _total_return(ret: pd.Series) -> float:
    if ret.empty:
        return float("nan")
    return float(np.prod(1.0 + ret.to_numpy(dtype=float)) - 1.0)


def _annualized_return(ret: pd.Series, periods_per_year: int = 12) -> float:
    if ret.empty:
        return float("nan")
    tot = _total_return(ret)
    return float((1.0 + tot) ** (float(periods_per_year) / float(len(ret))) - 1.0)


def _max_drawdown(ret: pd.Series) -> float:
    if ret.empty:
        return float("nan")
    eq = np.cumprod(1.0 + ret.to_numpy(dtype=float))
    peak = np.maximum.accumulate(eq)
    with np.errstate(invalid="ignore", divide="ignore"):
        dd = eq / peak - 1.0
    dd = dd[np.isfinite(dd)]
    return float(np.min(dd)) if dd.size else float("nan")


def _sharpe(ret: pd.Series, periods_per_year: int = 12) -> float:
    if ret.empty:
        return float("nan")
    mu = float(ret.mean())
    sd = float(ret.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0.0:
        return float("nan")
    return float((mu / sd) * np.sqrt(float(periods_per_year)))


def _sortino(ret: pd.Series, periods_per_year: int = 12) -> float:
    if ret.empty:
        return float("nan")
    mu = float(ret.mean())
    downside = ret[ret < 0.0]
    if downside.empty:
        return float("inf")
    ds = float(downside.std(ddof=1))
    if not np.isfinite(ds) or ds <= 0.0:
        return float("nan")
    return float((mu / ds) * np.sqrt(float(periods_per_year)))


def _profit_factor(ret: pd.Series) -> float:
    if ret.empty:
        return float("nan")
    gains = float(ret[ret > 0.0].sum())
    losses = float(-ret[ret < 0.0].sum())
    if losses <= 0.0:
        return float("inf")
    return float(gains / losses)


def summarize_return_series(ret: Any, *, periods_per_year: int = 12) -> dict[str, float]:
    s = _to_series(ret)
    return {
        "months": int(len(s)),
        "total_return": _total_return(s),
        "annualized_return": _annualized_return(s, periods_per_year=periods_per_year),
        "max_drawdown": _max_drawdown(s),
        "sharpe": _sharpe(s, periods_per_year=periods_per_year),
        "sortino": _sortino(s, periods_per_year=periods_per_year),
        "win_rate": float((s > 0.0).mean()) if not s.empty else float("nan"),
        "expectancy_per_period": float(s.mean()) if not s.empty else float("nan"),
        "profit_factor": _profit_factor(s),
    }


def apply_cost_model(
    gross_ret: Any,
    turnover: Any,
    *,
    profile: MarketCostProfile,
    extra_slippage_bps: float = 0.0,
) -> pd.DataFrame:
    g = _to_series(gross_ret)
    t = _to_series(turnover)
    if len(t) != len(g):
        t = t.reindex(g.index).fillna(0.0)

    transaction_bps = float(profile.transaction_cost_bps) + float(max(0.0, extra_slippage_bps))
    cost_rate = transaction_bps / 10000.0
    trans_cost = t * cost_rate
    after_cost = g - trans_cost

    tax = pd.Series(np.zeros(len(after_cost), dtype=float), index=after_cost.index, dtype=float)
    if float(profile.tax_rate) > 0.0 and str(profile.tax_mode).strip().lower() == "monthly_positive":
        taxable = after_cost.clip(lower=0.0)
        tax = taxable * float(profile.tax_rate)

    net = after_cost - tax
    out = pd.DataFrame(
        {
            "gross_ret": g.astype(float),
            "turnover": t.astype(float),
            "transaction_cost_ret": trans_cost.astype(float),
            "ret_after_cost": after_cost.astype(float),
            "tax_ret": tax.astype(float),
            "net_ret": net.astype(float),
        }
    )
    return out


def default_market_profiles(*, tax_rate: float = 0.15) -> dict[str, MarketCostProfile]:
    return {
        "BR": MarketCostProfile(
            name="BR",
            brokerage_bps=1.0,
            spread_bps=6.0,
            slippage_bps=8.0,
            tax_rate=float(max(0.0, tax_rate)),
            tax_mode="monthly_positive",
        ),
        "US": MarketCostProfile(
            name="US",
            brokerage_bps=0.5,
            spread_bps=2.0,
            slippage_bps=3.0,
            tax_rate=float(max(0.0, tax_rate)),
            tax_mode="monthly_positive",
        ),
    }

