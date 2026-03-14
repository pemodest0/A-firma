from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
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
    monthly_sales_exemption_brl: float = 0.0
    capital_gains_brackets: tuple[tuple[float, float], ...] = ()
    loss_compensation_enabled: bool = False
    withholding_bps_on_sales: float = 0.0
    withholding_compensates_tax: bool = False
    assumed_portfolio_base_brl: float = 10000.0
    sell_turnover_fraction_proxy: float = 0.5
    cash_yield_enabled: bool = False
    cash_rate_source_path: str = ""
    cash_rate_annual_fallback: float = 0.0
    notes: tuple[str, ...] = ()

    @property
    def total_cost_bps_assumed(self) -> float:
        return float(self.transaction_cost_bps_assumed) + float(self.fx_spread_bps_assumed)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["capital_gains_brackets"] = [
            {"up_to_brl": float(up_to_brl), "rate": float(rate)}
            for up_to_brl, rate in self.capital_gains_brackets
        ]
        out["notes"] = list(self.notes)
        out["total_cost_bps_assumed"] = float(self.total_cost_bps_assumed)
        return out


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _to_series(values: Any) -> pd.Series:
    return pd.to_numeric(pd.Series(values, copy=True), errors="coerce").fillna(0.0).astype(float)


def _coerce_datetime_index(index: pd.Index) -> pd.DatetimeIndex:
    if isinstance(index, pd.DatetimeIndex):
        return index
    if isinstance(index, pd.PeriodIndex):
        return index.to_timestamp(how="end")
    return pd.to_datetime(pd.Index(index.astype(str)), errors="coerce")


def _infer_periods_per_year(index: pd.Index) -> int:
    if isinstance(index, pd.PeriodIndex):
        freq = str(index.freqstr or "").upper()
        if "M" in freq:
            return 12
        if "W" in freq:
            return 52
        return 252
    as_dt = _coerce_datetime_index(index)
    if as_dt.notna().sum() <= 2:
        return 252
    diffs = pd.Series(as_dt).dropna().sort_values().diff().dropna().dt.days
    if diffs.empty:
        return 252
    median_days = float(diffs.median())
    if median_days >= 24.0:
        return 12
    if median_days >= 5.0:
        return 52
    return 252


@lru_cache(maxsize=16)
def _load_cash_rate_source(path_str: str) -> pd.Series:
    path = Path(path_str)
    if not path.exists():
        return pd.Series(dtype=float)
    frame = pd.read_csv(path)
    date_col = "date" if "date" in frame.columns else "data" if "data" in frame.columns else ""
    value_col = "value" if "value" in frame.columns else "valor" if "valor" in frame.columns else ""
    if not date_col or not value_col:
        return pd.Series(dtype=float)
    date = pd.to_datetime(frame[date_col], errors="coerce", dayfirst=True)
    value = pd.to_numeric(frame[value_col], errors="coerce")
    clean = pd.DataFrame({"date": date, "value": value}).dropna(subset=["date", "value"]).sort_values("date")
    if clean.empty:
        return pd.Series(dtype=float)
    scale_probe = float(clean["value"].abs().median())
    if scale_probe > 1000.0:
        clean["value"] = clean["value"] / 1_000_000.0
    elif scale_probe > 1.0:
        clean["value"] = clean["value"] / 100.0
    clean["value"] = clean["value"].clip(lower=-0.50, upper=2.00)
    return clean.drop_duplicates("date", keep="last").set_index("date")["value"].astype(float)


def _resolve_progressive_rate(gain_brl: float, profile: NetAssumptionProfile) -> float:
    gain = float(max(0.0, gain_brl))
    if not profile.capital_gains_brackets:
        return float(max(0.0, profile.capital_gains_tax_rate))
    for up_to_brl, rate in profile.capital_gains_brackets:
        if gain <= float(up_to_brl):
            return float(max(0.0, rate))
    return float(max(0.0, profile.capital_gains_brackets[-1][1]))


def _cash_return_series(index: pd.Index, profile: NetAssumptionProfile) -> pd.Series:
    if not bool(profile.cash_yield_enabled):
        return pd.Series(np.zeros(len(index), dtype=float), index=index, dtype=float)
    annual_fallback = float(max(-0.99, profile.cash_rate_annual_fallback))
    cash_rate = pd.Series(dtype=float)
    if str(profile.cash_rate_source_path).strip():
        cash_rate = _load_cash_rate_source(str(profile.cash_rate_source_path))
    periods_per_year = max(1, int(_infer_periods_per_year(index)))
    idx_dt = _coerce_datetime_index(index)
    if cash_rate.empty:
        annual = pd.Series(annual_fallback, index=index, dtype=float)
    else:
        annual = (
            pd.to_numeric(cash_rate.reindex(idx_dt, method="ffill"), errors="coerce")
            .fillna(annual_fallback)
            .astype(float)
        )
        annual.index = index
    return ((1.0 + annual).clip(lower=1e-9) ** (1.0 / float(periods_per_year)) - 1.0).astype(float)


def _group_labels(index: pd.Index, *, mode: str) -> pd.Series:
    as_dt = _coerce_datetime_index(index)
    if str(mode).strip().lower().startswith("annual"):
        values = as_dt.strftime("%Y")
    elif str(mode).strip().lower().startswith("terminal"):
        values = np.repeat("terminal", len(index))
    else:
        values = as_dt.strftime("%Y-%m")
    out = pd.Series(values, index=index, dtype=object)
    return out.fillna("unknown")


def _nav_before_tax(after_cost: pd.Series, initial_capital_brl: float) -> pd.Series:
    base = float(max(1.0, initial_capital_brl))
    equity = (1.0 + pd.to_numeric(after_cost, errors="coerce").fillna(0.0).astype(float)).cumprod()
    return (equity.shift(1).fillna(1.0) * base).astype(float)


def _apply_monthly_realistic_tax_proxy(
    *,
    after_cost: pd.Series,
    turnover: pd.Series,
    profile: NetAssumptionProfile,
    initial_capital_brl: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    idx = after_cost.index
    nav_prev = _nav_before_tax(after_cost, initial_capital_brl)
    sell_fraction = float(np.clip(float(profile.sell_turnover_fraction_proxy), 0.0, 1.0))
    sales_notional_brl = (nav_prev * pd.to_numeric(turnover, errors="coerce").reindex(idx).fillna(0.0).clip(lower=0.0) * sell_fraction).astype(float)
    withholding_ret = (
        pd.to_numeric(turnover, errors="coerce").reindex(idx).fillna(0.0).clip(lower=0.0)
        * sell_fraction
        * (float(max(0.0, profile.withholding_bps_on_sales)) / 10000.0)
    ).astype(float)
    after_withholding = (after_cost - withholding_ret).astype(float)
    labels = _group_labels(idx, mode="monthly_realistic_proxy")
    tax_ret = pd.Series(np.zeros(len(idx), dtype=float), index=idx, dtype=float)
    carry_loss_brl = 0.0

    for label in labels.drop_duplicates().tolist():
        mask = labels.eq(label)
        group_idx = labels.index[mask]
        if len(group_idx) <= 0:
            continue
        nav_group = nav_prev.loc[group_idx].clip(lower=1.0)
        profit_brl = float((after_withholding.loc[group_idx] * nav_group).sum())
        sales_brl = float(sales_notional_brl.loc[group_idx].sum())
        withholding_brl = float((withholding_ret.loc[group_idx] * nav_group).sum())

        if profile.loss_compensation_enabled and profit_brl < 0.0:
            carry_loss_brl += profit_brl

        taxable_brl = 0.0
        exempt_limit = float(max(0.0, profile.monthly_sales_exemption_brl))
        if not (exempt_limit > 0.0 and sales_brl <= exempt_limit):
            if profile.loss_compensation_enabled:
                effective_brl = profit_brl + carry_loss_brl
                taxable_brl = float(max(0.0, effective_brl))
                carry_loss_brl = float(min(0.0, effective_brl))
            else:
                taxable_brl = float(max(0.0, profit_brl))

        if taxable_brl <= 0.0:
            continue

        tax_rate = _resolve_progressive_rate(taxable_brl, profile)
        tax_brl = taxable_brl * tax_rate
        if profile.withholding_compensates_tax:
            tax_brl = max(0.0, tax_brl - withholding_brl)
        last_idx = group_idx[-1]
        tax_ret.loc[last_idx] = float(tax_brl / max(1.0, float(nav_prev.loc[last_idx])))

    return after_withholding, tax_ret, withholding_ret


def _apply_monthly_inventory_tax_proxy(
    *,
    after_cost: pd.Series,
    turnover: pd.Series,
    profile: NetAssumptionProfile,
    initial_capital_brl: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    idx = after_cost.index
    turnover_s = pd.to_numeric(turnover, errors="coerce").reindex(idx).fillna(0.0).clip(lower=0.0).astype(float)
    sell_fraction = float(np.clip(float(profile.sell_turnover_fraction_proxy), 0.0, 1.0))
    withholding_ret = (
        turnover_s
        * sell_fraction
        * (float(max(0.0, profile.withholding_bps_on_sales)) / 10000.0)
    ).astype(float)
    after_withholding = (after_cost - withholding_ret).astype(float)
    labels = _group_labels(idx, mode="monthly_inventory_proxy")
    tax_ret = pd.Series(np.zeros(len(idx), dtype=float), index=idx, dtype=float)

    market_value_brl = float(max(1.0, initial_capital_brl))
    book_basis_brl = float(max(1.0, initial_capital_brl))
    carry_loss_brl = 0.0
    month_sales_brl = 0.0
    month_realized_gain_brl = 0.0
    month_withholding_brl = 0.0
    month_last_idx: Any | None = None
    current_label: str | None = None

    def finalize_month(last_idx: Any, ending_nav_brl: float) -> None:
        nonlocal carry_loss_brl, month_sales_brl, month_realized_gain_brl, month_withholding_brl
        taxable_brl = 0.0
        exempt_limit = float(max(0.0, profile.monthly_sales_exemption_brl))
        if not (exempt_limit > 0.0 and month_sales_brl <= exempt_limit):
            if profile.loss_compensation_enabled:
                effective_brl = month_realized_gain_brl + carry_loss_brl
                taxable_brl = float(max(0.0, effective_brl))
                carry_loss_brl = float(min(0.0, effective_brl))
            else:
                taxable_brl = float(max(0.0, month_realized_gain_brl))
        elif profile.loss_compensation_enabled and month_realized_gain_brl < 0.0:
            carry_loss_brl += month_realized_gain_brl

        if taxable_brl > 0.0:
            tax_rate = _resolve_progressive_rate(taxable_brl, profile)
            tax_brl = taxable_brl * tax_rate
            if profile.withholding_compensates_tax:
                tax_brl = max(0.0, tax_brl - month_withholding_brl)
            tax_ret.loc[last_idx] = float(tax_brl / max(1.0, ending_nav_brl))

        month_sales_brl = 0.0
        month_realized_gain_brl = 0.0
        month_withholding_brl = 0.0

    for stamp in idx:
        label = str(labels.loc[stamp])
        if current_label is None:
            current_label = label
        elif label != current_label and month_last_idx is not None:
            finalize_month(month_last_idx, market_value_brl)
            current_label = label

        period_ret = float(after_withholding.loc[stamp])
        period_turnover = float(turnover_s.loc[stamp])
        market_value_brl = float(max(0.0, market_value_brl * (1.0 + period_ret)))

        sale_notional_brl = float(
            min(
                market_value_brl,
                max(0.0, market_value_brl * period_turnover * sell_fraction),
            )
        )
        month_sales_brl += sale_notional_brl
        withholding_brl = float(max(0.0, sale_notional_brl * (float(max(0.0, profile.withholding_bps_on_sales)) / 10000.0)))
        month_withholding_brl += withholding_brl

        if sale_notional_brl > 0.0 and market_value_brl > 0.0 and book_basis_brl > 0.0:
            sale_fraction_of_book = float(np.clip(sale_notional_brl / market_value_brl, 0.0, 1.0))
            cost_basis_sold_brl = float(book_basis_brl * sale_fraction_of_book)
            realized_gain_brl = float(sale_notional_brl - cost_basis_sold_brl)
            month_realized_gain_brl += realized_gain_brl
            remaining_book_brl = float(max(0.0, book_basis_brl - cost_basis_sold_brl))
            buy_notional_brl = sale_notional_brl
            book_basis_brl = remaining_book_brl + buy_notional_brl

        month_last_idx = stamp

    if month_last_idx is not None:
        finalize_month(month_last_idx, market_value_brl)

    return after_withholding, tax_ret, withholding_ret


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
        raw_brackets = raw.get("capital_gains_brackets", [])
        brackets: list[tuple[float, float]] = []
        if isinstance(raw_brackets, list):
            for item in raw_brackets:
                if isinstance(item, dict):
                    up_to_brl = item.get("up_to_brl", float("inf"))
                    rate = item.get("rate", raw.get("capital_gains_tax_rate", 0.0))
                    try:
                        brackets.append((float(up_to_brl), float(rate)))
                    except (TypeError, ValueError):
                        continue
        cash_path = str(raw.get("cash_rate_source_path", "") or "").strip()
        if cash_path and not Path(cash_path).is_absolute():
            cash_path = str((path.parent.parent / cash_path).resolve())
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
            monthly_sales_exemption_brl=float(raw.get("monthly_sales_exemption_brl", 0.0)),
            capital_gains_brackets=tuple(brackets),
            loss_compensation_enabled=bool(raw.get("loss_compensation_enabled", False)),
            withholding_bps_on_sales=float(raw.get("withholding_bps_on_sales", 0.0)),
            withholding_compensates_tax=bool(raw.get("withholding_compensates_tax", False)),
            assumed_portfolio_base_brl=float(raw.get("assumed_portfolio_base_brl", 10000.0)),
            sell_turnover_fraction_proxy=float(raw.get("sell_turnover_fraction_proxy", 0.5)),
            cash_yield_enabled=bool(raw.get("cash_yield_enabled", False)),
            cash_rate_source_path=cash_path,
            cash_rate_annual_fallback=float(raw.get("cash_rate_annual_fallback", 0.0)),
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
        monthly_sales_exemption_modeled=bool(foreign_profile.monthly_sales_exemption_modeled or br_profile.monthly_sales_exemption_modeled),
        monthly_sales_exemption_brl=share * foreign_profile.monthly_sales_exemption_brl + local * br_profile.monthly_sales_exemption_brl,
        capital_gains_brackets=tuple(
            foreign_profile.capital_gains_brackets if share >= 0.5 and foreign_profile.capital_gains_brackets else br_profile.capital_gains_brackets
        ),
        loss_compensation_enabled=bool(foreign_profile.loss_compensation_enabled or br_profile.loss_compensation_enabled),
        withholding_bps_on_sales=share * foreign_profile.withholding_bps_on_sales + local * br_profile.withholding_bps_on_sales,
        withholding_compensates_tax=bool(foreign_profile.withholding_compensates_tax or br_profile.withholding_compensates_tax),
        assumed_portfolio_base_brl=share * foreign_profile.assumed_portfolio_base_brl + local * br_profile.assumed_portfolio_base_brl,
        sell_turnover_fraction_proxy=share * foreign_profile.sell_turnover_fraction_proxy + local * br_profile.sell_turnover_fraction_proxy,
        cash_yield_enabled=bool(foreign_profile.cash_yield_enabled or br_profile.cash_yield_enabled),
        cash_rate_source_path=str(foreign_profile.cash_rate_source_path or br_profile.cash_rate_source_path),
        cash_rate_annual_fallback=share * foreign_profile.cash_rate_annual_fallback + local * br_profile.cash_rate_annual_fallback,
        notes=notes,
    )


def apply_net_assumptions(
    gross_ret: Any,
    turnover: Any,
    *,
    profile: NetAssumptionProfile,
    periods_index: pd.Index | None = None,
    cash_weight: Any | None = None,
    initial_capital_brl: float | None = None,
) -> pd.DataFrame:
    g = _to_series(gross_ret)
    t = _to_series(turnover)
    if len(t) != len(g):
        t = t.reindex(g.index).fillna(0.0)
    if periods_index is not None:
        idx = pd.Index(periods_index)
        g.index = idx
        t.index = idx

    c = pd.Series(np.zeros(len(g), dtype=float), index=g.index, dtype=float)
    if cash_weight is not None:
        c = pd.to_numeric(pd.Series(cash_weight, copy=True), errors="coerce").reindex(g.index).fillna(0.0).clip(lower=0.0, upper=1.0).astype(float)
    cash_ret = (c * _cash_return_series(g.index, profile)).astype(float)
    gross_with_cash = (g + cash_ret).astype(float)

    cost_rate = float(profile.total_cost_bps_assumed) / 10000.0
    transaction_cost = t * cost_rate
    after_cost = gross_with_cash - transaction_cost

    tax = pd.Series(np.zeros(len(after_cost), dtype=float), index=after_cost.index, dtype=float)
    withholding_ret = pd.Series(np.zeros(len(after_cost), dtype=float), index=after_cost.index, dtype=float)
    timing = str(profile.tax_timing).strip().lower()
    tax_rate = float(max(0.0, profile.capital_gains_tax_rate))
    tax_capital_brl = float(initial_capital_brl if initial_capital_brl is not None else profile.assumed_portfolio_base_brl)
    if tax_rate > 0.0:
        if timing == "monthly_positive_proxy":
            tax = after_cost.clip(lower=0.0) * tax_rate
        elif timing == "monthly_realistic_proxy":
            after_cost, tax, withholding_ret = _apply_monthly_realistic_tax_proxy(
                after_cost=after_cost,
                turnover=t,
                profile=profile,
                initial_capital_brl=tax_capital_brl,
            )
        elif timing == "monthly_inventory_proxy":
            after_cost, tax, withholding_ret = _apply_monthly_inventory_tax_proxy(
                after_cost=after_cost,
                turnover=t,
                profile=profile,
                initial_capital_brl=tax_capital_brl,
            )
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
            "cash_weight": c.astype(float),
            "cash_ret": cash_ret.astype(float),
            "gross_ret_with_cash": gross_with_cash.astype(float),
            "turnover": t.astype(float),
            "transaction_cost_ret": transaction_cost.astype(float),
            "withholding_ret": withholding_ret.astype(float),
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
        "avg_cash_ret": float(pd.to_numeric(net_frame.get("cash_ret"), errors="coerce").fillna(0.0).mean()) if "cash_ret" in net_frame else 0.0,
        "avg_turnover": float(pd.to_numeric(net_frame["turnover"], errors="coerce").fillna(0.0).mean()),
        "avg_transaction_cost_ret": float(pd.to_numeric(net_frame["transaction_cost_ret"], errors="coerce").fillna(0.0).mean()),
        "avg_withholding_ret": float(pd.to_numeric(net_frame.get("withholding_ret"), errors="coerce").fillna(0.0).mean()) if "withholding_ret" in net_frame else 0.0,
        "avg_tax_ret": float(pd.to_numeric(net_frame["tax_ret"], errors="coerce").fillna(0.0).mean()),
    }
