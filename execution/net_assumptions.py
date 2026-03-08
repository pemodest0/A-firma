from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cost_model import summarize_return_series


@dataclass(frozen=True)
class NetAssumptionProfile:
    profile_id: str
    label: str
    jurisdiction: str
    transaction_cost_bps_assumed: float
    fx_spread_bps_assumed: float
    capital_gains_tax_rate: float
    tax_timing: str
    dividend_withholding_mode: str
    monthly_sales_exemption_modeled: bool = False
    notes: tuple[str, ...] = ()

    @property
    def total_cost_bps_assumed(self) -> float:
        return float(self.transaction_cost_bps_assumed) + float(self.fx_spread_bps_assumed)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["notes"] = list(self.notes)
        out["total_cost_bps_assumed"] = float(self.total_cost_bps_assumed)
        return out


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _to_series(values: Any) -> pd.Series:
    return pd.to_numeric(pd.Series(values, copy=True), errors="coerce").fillna(0.0).astype(float)


def load_net_assumption_profiles(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    payload = _read_json(path)
    raw_profiles = payload.get("profiles", {})
    if not isinstance(raw_profiles, dict):
        raise ValueError(f"invalid profiles payload: {path}")
    profiles: dict[str, NetAssumptionProfile] = {}
    for profile_id, raw in raw_profiles.items():
        if not isinstance(raw, dict):
            continue
        profiles[str(profile_id)] = NetAssumptionProfile(
            profile_id=str(profile_id),
            label=str(raw.get("label", profile_id)),
            jurisdiction=str(raw.get("jurisdiction", "")),
            transaction_cost_bps_assumed=float(raw.get("transaction_cost_bps_assumed", 0.0)),
            fx_spread_bps_assumed=float(raw.get("fx_spread_bps_assumed", 0.0)),
            capital_gains_tax_rate=float(raw.get("capital_gains_tax_rate", 0.0)),
            tax_timing=str(raw.get("tax_timing", "monthly_positive_proxy")),
            dividend_withholding_mode=str(raw.get("dividend_withholding_mode", "not_modeled")),
            monthly_sales_exemption_modeled=bool(raw.get("monthly_sales_exemption_modeled", False)),
            notes=tuple(str(x) for x in raw.get("notes", []) if str(x).strip()),
        )
    return {
        "config_path": str(path),
        "version": str(payload.get("version", "")),
        "statement": str(payload.get("statement", "")),
        "official_sources": payload.get("official_sources", {}) if isinstance(payload.get("official_sources", {}), dict) else {},
        "profiles": profiles,
    }


def blend_profiles(
    foreign_share: float,
    *,
    foreign_profile: NetAssumptionProfile,
    br_profile: NetAssumptionProfile,
) -> NetAssumptionProfile:
    share = float(np.clip(float(foreign_share), 0.0, 1.0))
    local = 1.0 - share
    tax_timing = foreign_profile.tax_timing if share >= 0.5 else br_profile.tax_timing
    notes = (
        f"foreign_share_proxy={share:.4f}",
        "Perfil blended usa media ponderada de custos e aliquota entre exterior e Brasil.",
    )
    return NetAssumptionProfile(
        profile_id="blended_proxy",
        label="Blended proxy exterior/Brasil",
        jurisdiction="blended",
        transaction_cost_bps_assumed=share * foreign_profile.transaction_cost_bps_assumed + local * br_profile.transaction_cost_bps_assumed,
        fx_spread_bps_assumed=share * foreign_profile.fx_spread_bps_assumed + local * br_profile.fx_spread_bps_assumed,
        capital_gains_tax_rate=share * foreign_profile.capital_gains_tax_rate + local * br_profile.capital_gains_tax_rate,
        tax_timing=tax_timing,
        dividend_withholding_mode="not_modeled_adjusted_close",
        monthly_sales_exemption_modeled=False,
        notes=notes,
    )


def apply_net_assumptions(
    gross_ret: Any,
    turnover: Any,
    *,
    profile: NetAssumptionProfile,
    periods_index: pd.Index | None = None,
) -> pd.DataFrame:
    g = _to_series(gross_ret)
    t = _to_series(turnover)
    if len(t) != len(g):
        t = t.reindex(g.index).fillna(0.0)
    if periods_index is not None:
        idx = pd.Index(periods_index)
        g.index = idx
        t.index = idx

    cost_rate = float(profile.total_cost_bps_assumed) / 10000.0
    transaction_cost = t * cost_rate
    after_cost = g - transaction_cost

    tax = pd.Series(np.zeros(len(after_cost), dtype=float), index=after_cost.index, dtype=float)
    timing = str(profile.tax_timing).strip().lower()
    tax_rate = float(max(0.0, profile.capital_gains_tax_rate))
    if tax_rate > 0.0:
        if timing == "monthly_positive_proxy":
            tax = after_cost.clip(lower=0.0) * tax_rate
        elif timing == "annual_positive_proxy":
            years = pd.Index(after_cost.index.astype(str)).str.slice(0, 4)
            yearly_positive = after_cost.groupby(years).transform(lambda s: float(max(0.0, float(s.sum()))))
            yearly_tax = yearly_positive.groupby(years).transform("first") * tax_rate
            counts = pd.Series(1.0, index=after_cost.index).groupby(years).transform("sum")
            tax = pd.to_numeric(yearly_tax / counts.replace(0.0, np.nan), errors="coerce").fillna(0.0).astype(float)
        elif timing == "terminal_positive_proxy":
            total_positive = max(0.0, float(after_cost.sum()))
            tax.iloc[-1] = total_positive * tax_rate
        else:
            raise ValueError(f"unsupported tax_timing: {profile.tax_timing}")

    net = after_cost - tax
    out = pd.DataFrame(
        {
            "gross_ret": g.astype(float),
            "turnover": t.astype(float),
            "transaction_cost_ret": transaction_cost.astype(float),
            "ret_after_cost": after_cost.astype(float),
            "tax_ret": tax.astype(float),
            "net_ret": net.astype(float),
        }
    )
    return out


def summarize_net_series(net_frame: pd.DataFrame, *, periods_per_year: int = 12) -> dict[str, Any]:
    gross = summarize_return_series(net_frame["gross_ret"], periods_per_year=periods_per_year)
    net = summarize_return_series(net_frame["net_ret"], periods_per_year=periods_per_year)
    return {
        "gross": gross,
        "net": net,
        "avg_turnover": float(pd.to_numeric(net_frame["turnover"], errors="coerce").fillna(0.0).mean()),
        "avg_transaction_cost_ret": float(pd.to_numeric(net_frame["transaction_cost_ret"], errors="coerce").fillna(0.0).mean()),
        "avg_tax_ret": float(pd.to_numeric(net_frame["tax_ret"], errors="coerce").fillna(0.0).mean()),
    }
