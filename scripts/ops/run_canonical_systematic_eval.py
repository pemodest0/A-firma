#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution.returns import daily_simple_to_monthly, load_return_series_csv  # noqa: E402
from engine.structural.run_manifest import write_run_manifest  # noqa: E402


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_int_list(s: str) -> list[int]:
    out = [int(x.strip()) for x in str(s).split(",") if str(x).strip()]
    if not out:
        raise ValueError("empty integer list")
    return out


def _parse_float_list(s: str) -> list[float]:
    out = [float(x.strip()) for x in str(s).split(",") if str(x).strip()]
    if not out:
        raise ValueError("empty float list")
    return out


def _thin_grid_items[T](items: list[T], max_items: int) -> list[T]:
    if int(max_items) <= 0 or len(items) <= int(max_items):
        return items
    idx = np.linspace(0, len(items) - 1, num=int(max_items), dtype=int)
    keep = sorted({int(i) for i in idx.tolist() if 0 <= int(i) < len(items)})
    return [items[i] for i in keep]


def _latest_impact_dir() -> tuple[Path, Path]:
    base = ROOT / "results" / "lab_corr_macro"
    if not base.exists():
        raise FileNotFoundError(f"missing base dir: {base}")
    runs = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
    for run in runs:
        hier = run / "hierarchical"
        if not hier.exists():
            continue
        returns_csv = run / "returns_wide_core.csv"
        if not returns_csv.exists():
            continue
        for cand in sorted(hier.glob("impact_learning*"), key=lambda p: p.name, reverse=True):
            if not cand.is_dir():
                continue
            if (cand / "impact_training_dataset.csv").exists():
                return cand, returns_csv
    raise FileNotFoundError("no impact dir + returns_wide_core.csv found")


def _load_price_returns(prices_dir: Path, ticker: str) -> pd.Series:
    path = prices_dir / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing benchmark price returns: {path}")
    out = load_return_series_csv(path, source_kind="log", target_kind="simple", series_name=ticker)
    if out.empty:
        raise ValueError(f"empty benchmark price returns after cleaning: {path}")
    return out.astype(float).sort_index()


def _load_external_benchmark_monthly(prices_dir: Path, ticker: str, month_index: pd.Index) -> pd.Series:
    daily = _load_price_returns(prices_dir, ticker)
    monthly = daily_simple_to_monthly(daily)
    aligned = pd.to_numeric(monthly, errors="coerce").reindex(pd.Index(month_index, dtype=str))
    missing = aligned[aligned.isna()]
    if not missing.empty:
        sample = ",".join(missing.index.astype(str).tolist()[:5])
        raise ValueError(f"benchmark {ticker} missing monthly coverage for {missing.shape[0]} months: {sample}")
    return aligned.astype(float)


def _series_value_or_raise(series: pd.Series, key: str, label: str) -> float:
    if key not in series.index:
        raise KeyError(f"{label} missing key {key}")
    value = float(pd.to_numeric(pd.Series([series.loc[key]]), errors="coerce").iloc[0])
    if not np.isfinite(value):
        raise ValueError(f"{label} has non-finite value for {key}")
    return value


def _cap_weights(base: np.ndarray, w_max: float) -> np.ndarray:
    w = np.asarray(base, dtype=float)
    if w.size <= 0:
        return w
    if not np.isfinite(w).any() or float(np.nansum(w)) <= 0.0:
        w = np.ones_like(w, dtype=float)
    w = w / float(np.sum(w))
    cap = float(max(1e-6, w_max))
    for _ in range(8):
        over = w > cap
        if not np.any(over):
            break
        extra = float(np.sum(w[over] - cap))
        w[over] = cap
        under = ~over
        if not np.any(under):
            break
        s_under = float(np.sum(w[under]))
        if s_under <= 0:
            break
        w[under] += extra * (w[under] / s_under)
    total = float(np.sum(w))
    if total <= 0:
        return np.ones_like(w, dtype=float) / float(w.size)
    return w / total


def _ann(ret: pd.Series) -> float:
    s = pd.to_numeric(ret, errors="coerce").dropna().astype(float)
    if s.empty:
        return float("nan")
    total = float(np.prod(1.0 + s.to_numpy(dtype=float)) - 1.0)
    return float((1.0 + total) ** (12.0 / float(len(s))) - 1.0)


def _total(ret: pd.Series) -> float:
    s = pd.to_numeric(ret, errors="coerce").dropna().astype(float)
    if s.empty:
        return float("nan")
    return float(np.prod(1.0 + s.to_numpy(dtype=float)) - 1.0)


def _mdd(ret: pd.Series) -> float:
    s = pd.to_numeric(ret, errors="coerce").fillna(0.0).astype(float)
    if s.empty:
        return float("nan")
    eq = np.cumprod(1.0 + s.to_numpy(dtype=float))
    peak = np.maximum.accumulate(eq)
    dd = eq / np.where(peak == 0.0, np.nan, peak) - 1.0
    dd = dd[np.isfinite(dd)]
    return float(np.min(dd)) if dd.size > 0 else float("nan")


@dataclass(frozen=True)
class StrategyParams:
    mode: str
    top_k: int
    impact_power: float
    w_max: float
    mom_lookback: int
    mom_threshold: float
    rb_stress: float
    rb_transition: float
    rb_stable: float
    rb_dispersion: float


@dataclass(frozen=True)
class DefenseConfig:
    enabled: bool
    multiplier: float
    corr_quantile: float
    vol_quantile: float
    min_history_months: int
    require_both: bool


@dataclass(frozen=True)
class DecelerationConfig:
    enabled: bool
    lookback_months: int
    alpha_threshold: float
    min_streak: int
    multiplier: float
    topk_multiplier: float


@dataclass(frozen=True)
class AttackConfig:
    enabled: bool
    multiplier: float
    corr_quantile: float
    vol_quantile: float
    min_history_months: int
    require_both: bool
    require_positive_alpha: bool
    alpha_lookback_months: int


@dataclass(frozen=True)
class DrawdownGuardConfig:
    enabled: bool
    soft_threshold: float
    hard_threshold: float
    soft_multiplier: float
    hard_multiplier: float


@dataclass(frozen=True)
class HedgeConfig:
    enabled: bool
    multiplier: float
    mom_lookback: int
    mom_threshold: float
    activate_on_defense: bool


@dataclass(frozen=True)
class RegimeTopKConfig:
    enabled: bool
    stable_multiplier: float
    transition_multiplier: float
    stress_multiplier: float


@dataclass(frozen=True)
class WeeklyStressConfig:
    enabled: bool
    weekly_quantile: float
    min_history_months: int
    multiplier: float


@dataclass(frozen=True)
class TailAdaptConfig:
    enabled: bool
    lookback_months: int
    alpha_threshold: float
    down_multiplier: float
    up_multiplier: float
    start_ym: str


@dataclass(frozen=True)
class HybridRankingConfig:
    enabled: bool
    lookback_months: int
    weight_impact: float
    weight_momentum: float
    weight_liquidity: float
    weight_persistence: float
    weight_sector_strength: float
    weight_low_corr: float
    weight_low_vol: float
    weight_low_concentration: float
    liquidity_csv: str
    persistence_lookback_months: int
    volatility_lookback_months: int


@dataclass(frozen=True)
class ContinuousRiskConfig:
    enabled: bool
    weight_global_score: float
    weight_corr: float
    weight_vol: float
    regime_bias: float
    score_power: float


@dataclass(frozen=True)
class RegimeBasketConfig:
    enabled: bool
    sector_bonus: float
    global_sleeve_bonus: float
    vote_threshold: float


@dataclass(frozen=True)
class LayeredRotationConfig:
    enabled: bool
    min_sectors: int
    max_sectors: int
    target_assets_per_sector: int
    sector_score_power: float
    min_assets_per_sector: int


@dataclass(frozen=True)
class AutoAggressiveConfig:
    enabled: bool
    multiplier: float
    topk_multiplier: float
    score_quantile: float
    corr_quantile: float
    vol_quantile: float
    min_history_months: int
    require_positive_alpha: bool
    alpha_lookback_months: int
    confirm_months: int


@dataclass(frozen=True)
class RebalanceControlConfig:
    enabled: bool
    deadband_l1: float
    force_l1: float
    cooldown_months: int


DEFAULT_BUCKET_SECTOR_PREFERENCES: dict[str, set[str]] = {
    "stress": {"consumer_staples", "utilities", "health_care", "equities_us_broad", "equities_ex_us"},
    "transition": {"equities_us_broad", "financials", "health_care", "industrials", "technology"},
    "stable": {"equities_us_broad", "financials", "health_care", "industrials", "technology"},
    "dispersion": {"equities_us_broad", "equities_ex_us", "energy", "financials", "industrials", "materials", "technology"},
}

GLOBAL_SLEEVE_SECTORS: set[str] = {"equities_us_broad", "equities_ex_us"}
GLOBAL_SLEEVE_ASSETS: set[str] = {
    "DIA",
    "EEM",
    "EFA",
    "EWJ",
    "EWZ",
    "IWM",
    "QQQ",
    "RSP",
    "SPY",
    "VT",
    "VTI",
    "XLB",
    "XLE",
    "XLF",
    "XLI",
    "XLP",
    "XLK",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
}


def _build_monthly_matrices(
    returns_csv: Path,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    df = pd.read_csv(returns_csv)
    if "date" not in df.columns:
        raise ValueError(f"missing date column in {returns_csv}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    assets = [c for c in df.columns if c != "date"]
    if not assets:
        raise ValueError(f"no asset columns in {returns_csv}")
    for c in assets:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    ym = df["date"].dt.to_period("M").astype(str)

    month_labels: list[str] = []
    month_rows: list[np.ndarray] = []
    eqw_monthly: dict[str, float] = {}
    corr_monthly: dict[str, float] = {}
    vol_monthly: dict[str, float] = {}
    weekly_stress_monthly: dict[str, float] = {}
    for label, g in df.groupby(ym):
        arr = g[assets].to_numpy(dtype=float)
        month_rows.append(np.prod(1.0 + arr, axis=0) - 1.0)
        month_labels.append(str(label))
        eqw_daily = np.mean(arr, axis=1)
        eqw_monthly[str(label)] = float(np.prod(1.0 + eqw_daily) - 1.0)
        if eqw_daily.size >= 5:
            r5 = (
                pd.Series(eqw_daily, dtype=float)
                .rolling(5, min_periods=5)
                .apply(lambda x: float(np.prod(1.0 + x) - 1.0), raw=True)
            )
            v = pd.to_numeric(r5, errors="coerce")
            weekly_stress_monthly[str(label)] = float(v.min()) if v.notna().any() else float("nan")
        else:
            weekly_stress_monthly[str(label)] = float("nan")
        vol_vals = np.nanstd(arr, axis=0, ddof=0)
        vol_monthly[str(label)] = float(np.nanmean(vol_vals)) if np.isfinite(vol_vals).any() else float("nan")
        if arr.shape[0] >= 2 and arr.shape[1] >= 2:
            with np.errstate(invalid="ignore", divide="ignore"):
                c = np.corrcoef(arr, rowvar=False)
            tri = c[np.triu_indices_from(c, k=1)]
            tri = tri[np.isfinite(tri)]
            corr_monthly[str(label)] = float(np.mean(np.abs(tri))) if tri.size > 0 else float("nan")
        else:
            corr_monthly[str(label)] = float("nan")
    mret = pd.DataFrame(np.vstack(month_rows), index=month_labels, columns=assets).sort_index()
    eqw = pd.Series(eqw_monthly).sort_index()
    corr = pd.Series(corr_monthly).sort_index()
    vol = pd.Series(vol_monthly).sort_index()
    weekly_stress = pd.Series(weekly_stress_monthly).sort_index()
    market = mret["SPY"].copy() if "SPY" in mret.columns else pd.Series(index=mret.index, dtype=float)
    return mret, eqw, market, corr, vol, weekly_stress


def _load_allowed_assets(path_value: str) -> set[str]:
    text = str(path_value).strip()
    if not text:
        return set()
    path = Path(text)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.exists():
        return set()
    try:
        head = pd.read_csv(path, nrows=0)
        cols = set(head.columns)
        key = next((c for c in ["asset_id", "ticker", "asset"] if c in cols), "")
        if not key:
            return set()
        d = pd.read_csv(path, usecols=[key])
    except Exception:
        return set()
    return {str(x).strip() for x in d[key].dropna().astype(str).tolist() if str(x).strip()}


def _build_snapshots(
    impact_csv: Path,
    *,
    max_assets_per_month: int,
    allowed_assets: set[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    cols = list(pd.read_csv(impact_csv, nrows=0).columns)
    req = ["date", "asset_id", "impact_global", "global_score", "regime"]
    opt = ["sector", "sector_kind", "impact_sector", "sector_loading", "overlap_sector_global", "global_share_within_sector"]
    usecols = [c for c in req + opt if c in cols]
    d = pd.read_csv(impact_csv, usecols=usecols)
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date", "asset_id"]).sort_values("date").reset_index(drop=True)
    d["impact_global"] = pd.to_numeric(d["impact_global"], errors="coerce").fillna(0.0)
    d["global_score"] = pd.to_numeric(d.get("global_score"), errors="coerce")
    d["regime"] = d.get("regime", "stable").astype(str).str.lower()
    if "sector" in d.columns:
        d["sector"] = d["sector"].astype(str).str.strip().replace({"": "unknown"}).fillna("unknown")
    if "impact_sector" in d.columns:
        d["impact_sector"] = pd.to_numeric(d.get("impact_sector"), errors="coerce").fillna(0.0)
    if "sector_loading" in d.columns:
        d["sector_loading"] = pd.to_numeric(d.get("sector_loading"), errors="coerce")
    if "overlap_sector_global" in d.columns:
        d["overlap_sector_global"] = pd.to_numeric(d.get("overlap_sector_global"), errors="coerce")
    if "global_share_within_sector" in d.columns:
        d["global_share_within_sector"] = pd.to_numeric(d.get("global_share_within_sector"), errors="coerce")
    d["ym"] = d["date"].dt.to_period("M").astype(str)
    if allowed_assets:
        d = d[d["asset_id"].astype(str).isin(set(allowed_assets))].copy()
    snap = d.groupby(["ym", "asset_id"], as_index=False).tail(1).copy()
    snap = (
        snap.sort_values(["ym", "impact_global"], ascending=[True, False])
        .groupby("ym", as_index=False)
        .head(int(max_assets_per_month))
        .reset_index(drop=True)
    )
    by_month: dict[str, pd.DataFrame] = {}
    state_rows: list[dict[str, Any]] = []
    for ym, g in snap.groupby("ym"):
        g2 = g.sort_values("impact_global", ascending=False).reset_index(drop=True)
        keep_cols = [c for c in ["asset_id", "impact_global", "sector", "sector_kind", "impact_sector", "sector_loading", "overlap_sector_global", "global_share_within_sector"] if c in g2.columns]
        by_month[str(ym)] = g2[keep_cols].copy()
        state_rows.append(
            {
                "ym": str(ym),
                "global_score": float(g2["global_score"].iloc[0]) if "global_score" in g2.columns and pd.notna(g2["global_score"].iloc[0]) else float("nan"),
                "regime": str(g2["regime"].iloc[0]).lower() if "regime" in g2.columns else "stable",
            }
        )
    state = pd.DataFrame(state_rows).drop_duplicates(subset=["ym"]).set_index("ym").sort_index()
    return by_month, state


def _resolve_liquidity_csv(path_value: str) -> Path | None:
    p = Path(str(path_value).strip()) if str(path_value).strip() else None
    candidates: list[Path] = []
    if p is not None:
        candidates.append(p if p.is_absolute() else (ROOT / p))
    candidates.extend(
        [
            ROOT / "data" / "asset_metadata_global_plus.csv",
            ROOT / "data" / "asset_metadata.csv",
        ]
    )
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    return None


def _load_liquidity_map(path_value: str) -> dict[str, float]:
    csv_path = _resolve_liquidity_csv(path_value)
    if csv_path is None:
        return {}
    try:
        d = pd.read_csv(csv_path, usecols=[c for c in ["asset_id", "liquidity_proxy"] if c in pd.read_csv(csv_path, nrows=0).columns])
    except Exception:
        return {}
    if "asset_id" not in d.columns or "liquidity_proxy" not in d.columns:
        return {}
    d["asset_id"] = d["asset_id"].astype(str)
    d["liquidity_proxy"] = pd.to_numeric(d["liquidity_proxy"], errors="coerce")
    d = d.dropna(subset=["asset_id", "liquidity_proxy"]).sort_values("liquidity_proxy", ascending=False).drop_duplicates("asset_id")
    return {str(r.asset_id): float(r.liquidity_proxy) for r in d.itertuples(index=False)}


def _pct_rank(s: pd.Series, *, ascending: bool = True, fill: float = 0.5) -> pd.Series:
    z = pd.to_numeric(s, errors="coerce")
    if z.isna().all():
        return pd.Series(np.full(len(z), float(fill)), index=s.index, dtype=float)
    r = z.rank(method="average", pct=True, ascending=ascending)
    return r.fillna(float(fill)).astype(float)


def _quantile_position(history: pd.Series, current: float) -> float:
    hist = pd.to_numeric(history, errors="coerce").dropna().astype(float)
    if hist.empty or not np.isfinite(current):
        return float("nan")
    arr = np.sort(hist.to_numpy(dtype=float))
    if arr.size <= 0:
        return float("nan")
    pos = np.searchsorted(arr, float(current), side="right")
    return float(np.clip(pos / float(arr.size), 0.0, 1.0))


def _build_persistence_maps(
    *,
    snap_by_month: dict[str, pd.DataFrame],
    months: list[str],
    lookback_months: int,
) -> dict[str, dict[str, float]]:
    lb = int(max(1, lookback_months))
    out: dict[str, dict[str, float]] = {}
    for i, ym in enumerate(months):
        hist = months[max(0, i - lb) : i]
        if not hist:
            out[str(ym)] = {}
            continue
        rows: list[pd.DataFrame] = []
        for hm in hist:
            s = snap_by_month.get(hm)
            if s is None or s.empty:
                continue
            part = s[["asset_id", "impact_global"]].copy()
            part["asset_id"] = part["asset_id"].astype(str)
            rows.append(part)
        if not rows:
            out[str(ym)] = {}
            continue
        d = pd.concat(rows, ignore_index=True)
        d["impact_global"] = pd.to_numeric(d["impact_global"], errors="coerce").fillna(0.0)
        agg = d.groupby("asset_id", as_index=False).agg(
            impact_mean=("impact_global", "mean"),
            seen=("impact_global", "size"),
        )
        agg["persist_seen_norm"] = agg["seen"].astype(float) / float(len(hist))
        agg["persist_impact_norm"] = _pct_rank(agg["impact_mean"], ascending=True, fill=0.5)
        agg["persistence_score"] = 0.5 * agg["persist_seen_norm"].astype(float) + 0.5 * agg["persist_impact_norm"].astype(float)
        out[str(ym)] = {str(r.asset_id): float(r.persistence_score) for r in agg.itertuples(index=False)}
    return out


def _bucket_sector_bonus(
    *,
    asset_id: str,
    sector: str,
    bucket: str,
    cfg: RegimeBasketConfig,
) -> float:
    if not cfg.enabled:
        return 0.0
    b = str(bucket).strip().lower()
    s = str(sector).strip().lower()
    bonus = 0.0
    preferred = DEFAULT_BUCKET_SECTOR_PREFERENCES.get(b, set())
    if s in preferred:
        bonus += float(max(0.0, cfg.sector_bonus))
    if s in GLOBAL_SLEEVE_SECTORS or str(asset_id).strip().upper() in GLOBAL_SLEEVE_ASSETS:
        bonus += float(max(0.0, cfg.global_sleeve_bonus))
    return float(bonus)


def _apply_hybrid_ranking(
    *,
    s: pd.DataFrame,
    prev_ym: str,
    mom_df: pd.DataFrame | None,
    vol_df: pd.DataFrame | None,
    cfg: HybridRankingConfig,
    liquidity_map: dict[str, float],
    persistence_map: dict[str, float],
    bucket: str,
    basket_cfg: RegimeBasketConfig,
) -> pd.DataFrame:
    if s.empty:
        out = s.copy()
        out["rank_score"] = np.nan
        return out
    out = s.copy()
    if not cfg.enabled:
        out = out.sort_values("impact_global", ascending=False).reset_index(drop=True)
        out["rank_score"] = pd.to_numeric(out["impact_global"], errors="coerce").fillna(0.0).astype(float)
        return out

    out["impact_norm"] = _pct_rank(out["impact_global"], ascending=True, fill=0.5)
    if mom_df is not None and prev_ym in mom_df.index:
        mm = mom_df.loc[prev_ym]
        out["mom_raw"] = out["asset_id"].map(mm.to_dict())
    else:
        out["mom_raw"] = np.nan
    out["mom_norm"] = _pct_rank(out["mom_raw"], ascending=True, fill=0.5)
    out["liq_raw"] = out["asset_id"].map(liquidity_map).astype(float)
    out["liq_norm"] = _pct_rank(out["liq_raw"], ascending=True, fill=0.5)
    out["persistence_raw"] = out["asset_id"].map(persistence_map).astype(float)
    out["persistence_norm"] = _pct_rank(out["persistence_raw"], ascending=True, fill=0.5)
    out["sector_strength_raw"] = pd.to_numeric(out.get("impact_sector"), errors="coerce")
    if out["sector_strength_raw"].isna().all() and "sector" in out.columns:
        sec_mean = out.groupby("sector")["impact_global"].transform("mean")
        out["sector_strength_raw"] = pd.to_numeric(sec_mean, errors="coerce")
    out["sector_strength_norm"] = _pct_rank(out["sector_strength_raw"], ascending=True, fill=0.5)
    out["low_corr_raw"] = pd.to_numeric(out.get("overlap_sector_global"), errors="coerce")
    out["low_corr_norm"] = _pct_rank(out["low_corr_raw"], ascending=False, fill=0.5)
    out["low_concentration_raw"] = pd.to_numeric(out.get("global_share_within_sector"), errors="coerce")
    out["low_concentration_norm"] = _pct_rank(out["low_concentration_raw"], ascending=False, fill=0.5)
    if vol_df is not None and prev_ym in vol_df.index:
        vm = vol_df.loc[prev_ym]
        out["vol_raw"] = out["asset_id"].map(vm.to_dict())
    else:
        out["vol_raw"] = np.nan
    out["low_vol_norm"] = _pct_rank(out["vol_raw"], ascending=False, fill=0.5)

    weight_pairs = [
        (float(max(0.0, cfg.weight_impact)), "impact_norm"),
        (float(max(0.0, cfg.weight_momentum)), "mom_norm"),
        (float(max(0.0, cfg.weight_liquidity)), "liq_norm"),
        (float(max(0.0, cfg.weight_persistence)), "persistence_norm"),
        (float(max(0.0, cfg.weight_sector_strength)), "sector_strength_norm"),
        (float(max(0.0, cfg.weight_low_corr)), "low_corr_norm"),
        (float(max(0.0, cfg.weight_low_vol)), "low_vol_norm"),
        (float(max(0.0, cfg.weight_low_concentration)), "low_concentration_norm"),
    ]

    s_w = float(sum(w for w, _ in weight_pairs))
    if s_w <= 0:
        weight_pairs = [(1.0, "impact_norm")]
        s_w = 1.0
    numer = np.zeros(out.shape[0], dtype=float)
    for weight, col in weight_pairs:
        numer += float(weight) * pd.to_numeric(out[col], errors="coerce").fillna(0.5).to_numpy(dtype=float)
    out["basket_bonus"] = [
        _bucket_sector_bonus(
            asset_id=str(asset_id),
            sector=str(sector),
            bucket=str(bucket),
            cfg=basket_cfg,
        )
        for asset_id, sector in zip(
            out["asset_id"].astype(str).tolist(),
            out.get("sector", pd.Series([""] * out.shape[0], index=out.index)).astype(str).tolist(),
            strict=False,
        )
    ]
    out["rank_score"] = numer / float(s_w) + pd.to_numeric(out["basket_bonus"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    out = out.sort_values(["rank_score", "impact_global"], ascending=[False, False]).reset_index(drop=True)
    return out


def _select_layered_assets(
    *,
    ranked: pd.DataFrame,
    effective_top_k: int,
    cfg: LayeredRotationConfig,
) -> tuple[pd.DataFrame, int, str]:
    if ranked.empty:
        return ranked.copy(), 0, ""
    if not cfg.enabled or "sector" not in ranked.columns:
        top = ranked.head(int(max(1, effective_top_k))).copy().reset_index(drop=True)
        sec = top["sector"].astype(str).nunique() if "sector" in top.columns else 0
        sec_names = ",".join(sorted(top["sector"].astype(str).dropna().unique().tolist())) if "sector" in top.columns else ""
        return top, int(sec), sec_names

    d = ranked.copy()
    d["sector"] = d["sector"].astype(str).str.strip().replace({"": "unknown"}).fillna("unknown")
    n_total = int(max(1, effective_top_k))
    available = int(d["sector"].nunique())
    if available <= 1:
        top = d.head(n_total).copy().reset_index(drop=True)
        return top, int(available), ",".join(sorted(top["sector"].astype(str).dropna().unique().tolist()))

    target = int(round(n_total / float(max(1, cfg.target_assets_per_sector))))
    n_sector = int(max(1, min(available, max(int(cfg.min_sectors), min(int(cfg.max_sectors), target)))))

    if "impact_sector" in d.columns:
        sec_base = pd.to_numeric(d["impact_sector"], errors="coerce").fillna(0.0)
    else:
        sec_base = pd.to_numeric(d["impact_global"], errors="coerce").fillna(0.0)
    d["_sec_base"] = np.maximum(sec_base.to_numpy(dtype=float), 0.0)
    sec_scores = d.groupby("sector", as_index=True)["_sec_base"].mean().astype(float)
    if float(cfg.sector_score_power) != 1.0:
        sec_scores = np.power(np.maximum(sec_scores.to_numpy(dtype=float), 0.0), float(max(0.1, cfg.sector_score_power)))
        sec_scores = pd.Series(sec_scores, index=d.groupby("sector", as_index=True)["_sec_base"].mean().index, dtype=float)
    sec_scores = sec_scores.sort_values(ascending=False).head(n_sector)
    chosen_sectors = sec_scores.index.tolist()
    if not chosen_sectors:
        top = d.head(n_total).copy().reset_index(drop=True)
        return top, int(top["sector"].nunique()), ",".join(sorted(top["sector"].astype(str).dropna().unique().tolist()))

    w = sec_scores.to_numpy(dtype=float)
    if float(np.sum(w)) <= 0:
        w = np.ones_like(w, dtype=float)
    w = w / float(np.sum(w))
    quotas = np.full(len(chosen_sectors), int(max(1, cfg.min_assets_per_sector)), dtype=int)
    remaining = int(n_total - int(np.sum(quotas)))
    if remaining > 0:
        ideal = w * float(remaining)
        add = np.floor(ideal).astype(int)
        quotas += add
        rem = int(remaining - int(np.sum(add)))
        if rem > 0:
            frac_order = np.argsort(-(ideal - add))
            for idx in frac_order[:rem]:
                quotas[idx] += 1
    elif remaining < 0:
        # Scale down if floor allocation exceeds total.
        over = int(-remaining)
        idx_order = np.argsort(w)  # remove from weaker sectors first
        for idx in idx_order:
            can_take = max(0, quotas[idx] - 1)
            take = min(can_take, over)
            quotas[idx] -= take
            over -= take
            if over <= 0:
                break

    selected_parts: list[pd.DataFrame] = []
    selected_ids: set[str] = set()
    for sec, q in zip(chosen_sectors, quotas, strict=False):
        if q <= 0:
            continue
        g = d[d["sector"] == sec].head(int(q)).copy()
        if g.empty:
            continue
        selected_parts.append(g)
        selected_ids.update(g["asset_id"].astype(str).tolist())
    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame(columns=d.columns)
    if selected.shape[0] < n_total:
        fill = d[~d["asset_id"].astype(str).isin(selected_ids)].head(int(n_total - selected.shape[0])).copy()
        if not fill.empty:
            selected = pd.concat([selected, fill], ignore_index=True)
    selected = selected.head(n_total).copy().reset_index(drop=True)
    if "rank_score" in selected.columns:
        selected = selected.sort_values(["rank_score", "impact_global"], ascending=[False, False]).reset_index(drop=True)
    sec_names = ",".join(sorted(selected["sector"].astype(str).dropna().unique().tolist()))
    return selected, int(selected["sector"].nunique()), sec_names


def _risk_budget(
    *,
    mode: str,
    prev_ym: str,
    prev_idx: int,
    months: list[str],
    state: pd.DataFrame,
    rb_stress: float,
    rb_transition: float,
    rb_stable: float,
    rb_dispersion: float,
    corr_signal: pd.Series,
    vol_signal: pd.Series,
    continuous_cfg: ContinuousRiskConfig,
) -> tuple[float, str, float, float, float, float]:
    if mode == "const":
        return float(rb_stable), "stable", float("nan"), float("nan"), float("nan"), float("nan")
    if prev_ym not in state.index:
        return float(rb_stable), "stable", float("nan"), float("nan"), float("nan"), float("nan")
    regime = str(state.loc[prev_ym, "regime"]).strip().lower()
    hist_months = [m for m in months[:prev_idx] if m in state.index]
    gs = float(state.loc[prev_ym, "global_score"]) if pd.notna(state.loc[prev_ym, "global_score"]) else float("nan")
    hist_score = pd.to_numeric(state.loc[hist_months, "global_score"], errors="coerce").dropna().astype(float)
    corr_hist_months = [m for m in months[:prev_idx] if m in corr_signal.index]
    vol_hist_months = [m for m in months[:prev_idx] if m in vol_signal.index]
    curr_corr = float(pd.to_numeric(pd.Series([corr_signal.get(prev_ym, np.nan)]), errors="coerce").iloc[0]) if prev_ym in corr_signal.index else float("nan")
    curr_vol = float(pd.to_numeric(pd.Series([vol_signal.get(prev_ym, np.nan)]), errors="coerce").iloc[0]) if prev_ym in vol_signal.index else float("nan")
    corr_hist = pd.to_numeric(corr_signal.loc[corr_hist_months], errors="coerce").dropna().astype(float) if corr_hist_months else pd.Series(dtype=float)
    vol_hist = pd.to_numeric(vol_signal.loc[vol_hist_months], errors="coerce").dropna().astype(float) if vol_hist_months else pd.Series(dtype=float)

    if mode == "regime" and not continuous_cfg.enabled:
        if regime == "stress":
            return float(rb_stress), "stress", float("nan"), float("nan"), float("nan"), float("nan")
        if regime == "transition":
            return float(rb_transition), "transition", float("nan"), float("nan"), float("nan"), float("nan")
        if regime == "dispersion":
            return float(rb_dispersion), "dispersion", float("nan"), float("nan"), float("nan"), float("nan")
        return float(rb_stable), "stable", float("nan"), float("nan"), float("nan"), float("nan")
    if mode == "score" and not continuous_cfg.enabled:
        if len(hist_score) < 12 or not np.isfinite(gs):
            return float(rb_stable), "stable", float("nan"), float("nan"), float("nan"), float("nan")
        q30 = float(np.quantile(hist_score.to_numpy(dtype=float), 0.30))
        q70 = float(np.quantile(hist_score.to_numpy(dtype=float), 0.70))
        q90 = float(np.quantile(hist_score.to_numpy(dtype=float), 0.90))
        if gs <= q30:
            return float(rb_dispersion), "dispersion", float("nan"), float("nan"), float("nan"), float("nan")
        if gs >= q90:
            return float(rb_stress), "stress", float("nan"), float("nan"), float("nan"), float("nan")
        if gs >= q70:
            return float(rb_transition), "transition", float("nan"), float("nan"), float("nan"), float("nan")
        return float(rb_stable), "stable", float("nan"), float("nan"), float("nan"), float("nan")

    gs_pos = _quantile_position(hist_score, gs) if len(hist_score) >= 6 else float("nan")
    corr_pos = _quantile_position(corr_hist, curr_corr) if len(corr_hist) >= 6 else float("nan")
    vol_pos = _quantile_position(vol_hist, curr_vol) if len(vol_hist) >= 6 else float("nan")
    score_parts: list[tuple[float, float]] = []
    for weight, value in [
        (float(max(0.0, continuous_cfg.weight_global_score)), gs_pos),
        (float(max(0.0, continuous_cfg.weight_corr)), corr_pos),
        (float(max(0.0, continuous_cfg.weight_vol)), vol_pos),
    ]:
        if weight > 0.0 and np.isfinite(value):
            score_parts.append((weight, value))
    if score_parts:
        raw_badness = float(sum(w * v for w, v in score_parts) / sum(w for w, _ in score_parts))
    elif np.isfinite(gs_pos):
        raw_badness = float(gs_pos)
    else:
        raw_badness = 0.5
    regime_adjust = {
        "stress": 0.18,
        "transition": 0.06,
        "stable": -0.04,
        "dispersion": -0.16,
    }.get(regime, 0.0)
    badness = float(np.clip(raw_badness + float(continuous_cfg.regime_bias) * regime_adjust, 0.0, 1.0))
    if float(max(0.1, continuous_cfg.score_power)) != 1.0:
        badness = float(np.clip(np.power(badness, float(max(0.1, continuous_cfg.score_power))), 0.0, 1.0))

    if badness <= 0.20:
        bucket = "dispersion"
    elif badness <= 0.50:
        bucket = "stable"
    elif badness <= 0.78:
        bucket = "transition"
    else:
        bucket = "stress"
    anchors_x = np.array([0.0, 0.33, 0.66, 1.0], dtype=float)
    anchors_y = np.array([float(rb_dispersion), float(rb_stable), float(rb_transition), float(rb_stress)], dtype=float)
    rb = float(np.interp(badness, anchors_x, anchors_y))
    rb = float(max(0.0, min(max(float(rb_dispersion), float(rb_stable), float(rb_transition), float(rb_stress)), rb)))
    return rb, bucket, badness, gs_pos, corr_pos, vol_pos


def _defense_factor(
    *,
    prev_ym: str,
    prev_idx: int,
    months: list[str],
    corr_signal: pd.Series,
    vol_signal: pd.Series,
    cfg: DefenseConfig,
) -> tuple[float, bool]:
    if not cfg.enabled:
        return 1.0, False
    if prev_ym not in corr_signal.index or prev_ym not in vol_signal.index:
        return 1.0, False
    hist_months = [m for m in months[: max(0, prev_idx - 1)] if m in corr_signal.index and m in vol_signal.index]
    if len(hist_months) < int(cfg.min_history_months):
        return 1.0, False
    hist_corr = pd.to_numeric(corr_signal.loc[hist_months], errors="coerce").dropna().astype(float)
    hist_vol = pd.to_numeric(vol_signal.loc[hist_months], errors="coerce").dropna().astype(float)
    curr_corr = float(pd.to_numeric(pd.Series([corr_signal.loc[prev_ym]]), errors="coerce").iloc[0])
    curr_vol = float(pd.to_numeric(pd.Series([vol_signal.loc[prev_ym]]), errors="coerce").iloc[0])
    if not (np.isfinite(curr_corr) and np.isfinite(curr_vol)) or hist_corr.empty or hist_vol.empty:
        return 1.0, False
    q_corr = float(np.quantile(hist_corr.to_numpy(dtype=float), float(cfg.corr_quantile)))
    q_vol = float(np.quantile(hist_vol.to_numpy(dtype=float), float(cfg.vol_quantile)))
    hi_corr = bool(curr_corr >= q_corr)
    hi_vol = bool(curr_vol >= q_vol)
    active = bool(hi_corr and hi_vol) if cfg.require_both else bool(hi_corr or hi_vol)
    if not active:
        return 1.0, False
    factor = float(max(0.0, min(1.0, cfg.multiplier)))
    return factor, True


def _deceleration_state(
    *,
    alpha_history: list[float],
    cfg: DecelerationConfig,
) -> tuple[float, bool, float, int, float]:
    if not cfg.enabled:
        return 1.0, False, float("nan"), 0, 1.0
    lb = int(max(1, cfg.lookback_months))
    streak_needed = int(max(1, cfg.min_streak))
    if len(alpha_history) < lb:
        return 1.0, False, float("nan"), 0, 1.0

    arr = np.asarray(alpha_history, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < lb:
        return 1.0, False, float("nan"), 0, 1.0

    recent_alpha = float(np.mean(arr[-lb:]))
    streak = 0
    # Check consecutive rolling windows ending in the past only.
    for off in range(streak_needed):
        end = arr.size - off
        start = end - lb
        if start < 0:
            break
        val = float(np.mean(arr[start:end]))
        if val < float(cfg.alpha_threshold):
            streak += 1
        else:
            break
    active = bool(streak >= streak_needed)
    if not active:
        return 1.0, False, recent_alpha, streak, 1.0
    factor = float(max(0.0, min(1.0, cfg.multiplier)))
    topk_factor = float(max(0.1, min(1.0, cfg.topk_multiplier)))
    return factor, True, recent_alpha, streak, topk_factor


def _attack_factor(
    *,
    prev_ym: str,
    prev_idx: int,
    months: list[str],
    corr_signal: pd.Series,
    vol_signal: pd.Series,
    alpha_history: list[float],
    cfg: AttackConfig,
) -> tuple[float, bool]:
    if not cfg.enabled:
        return 1.0, False
    if prev_ym not in corr_signal.index or prev_ym not in vol_signal.index:
        return 1.0, False
    hist_months = [m for m in months[: max(0, prev_idx - 1)] if m in corr_signal.index and m in vol_signal.index]
    if len(hist_months) < int(cfg.min_history_months):
        return 1.0, False
    hist_corr = pd.to_numeric(corr_signal.loc[hist_months], errors="coerce").dropna().astype(float)
    hist_vol = pd.to_numeric(vol_signal.loc[hist_months], errors="coerce").dropna().astype(float)
    curr_corr = float(pd.to_numeric(pd.Series([corr_signal.loc[prev_ym]]), errors="coerce").iloc[0])
    curr_vol = float(pd.to_numeric(pd.Series([vol_signal.loc[prev_ym]]), errors="coerce").iloc[0])
    if not (np.isfinite(curr_corr) and np.isfinite(curr_vol)) or hist_corr.empty or hist_vol.empty:
        return 1.0, False
    q_corr = float(np.quantile(hist_corr.to_numpy(dtype=float), float(cfg.corr_quantile)))
    q_vol = float(np.quantile(hist_vol.to_numpy(dtype=float), float(cfg.vol_quantile)))
    lo_corr = bool(curr_corr <= q_corr)
    lo_vol = bool(curr_vol <= q_vol)
    active = bool(lo_corr and lo_vol) if cfg.require_both else bool(lo_corr or lo_vol)
    if not active:
        return 1.0, False
    if cfg.require_positive_alpha:
        lb = int(max(1, cfg.alpha_lookback_months))
        if len(alpha_history) < lb:
            return 1.0, False
        recent = np.asarray(alpha_history[-lb:], dtype=float)
        recent = recent[np.isfinite(recent)]
        if recent.size < lb or float(np.mean(recent)) <= 0.0:
            return 1.0, False
    factor = float(max(1.0, cfg.multiplier))
    return factor, True


def _drawdown_guard_factor(
    *,
    equity: float,
    peak: float,
    cfg: DrawdownGuardConfig,
) -> tuple[float, bool, float]:
    if not cfg.enabled:
        return 1.0, False, 0.0
    if not (np.isfinite(equity) and np.isfinite(peak)) or peak <= 0.0:
        return 1.0, False, 0.0
    dd = float(equity / peak - 1.0)
    soft = float(min(cfg.soft_threshold, 0.0))
    hard = float(min(cfg.hard_threshold, soft))
    if dd <= hard:
        return float(max(0.0, min(1.0, cfg.hard_multiplier))), True, dd
    if dd <= soft:
        return float(max(0.0, min(1.0, cfg.soft_multiplier))), True, dd
    return 1.0, False, dd


def _regime_topk_factor(*, prev_ym: str, state: pd.DataFrame, cfg: RegimeTopKConfig) -> tuple[float, str]:
    if not cfg.enabled or prev_ym not in state.index:
        return 1.0, "stable"
    regime = str(state.loc[prev_ym, "regime"]).strip().lower()
    if regime == "stress":
        return float(max(0.1, cfg.stress_multiplier)), "stress"
    if regime == "transition":
        return float(max(0.1, cfg.transition_multiplier)), "transition"
    return float(max(0.1, cfg.stable_multiplier)), "stable"


def _weekly_stress_factor(
    *,
    prev_ym: str,
    prev_idx: int,
    months: list[str],
    weekly_stress: pd.Series,
    cfg: WeeklyStressConfig,
) -> tuple[float, bool, float, float]:
    if not cfg.enabled or prev_ym not in weekly_stress.index:
        return 1.0, False, float("nan"), float("nan")
    hist_months = [m for m in months[: max(0, prev_idx - 1)] if m in weekly_stress.index]
    if len(hist_months) < int(cfg.min_history_months):
        return 1.0, False, float("nan"), float("nan")
    hist = pd.to_numeric(weekly_stress.loc[hist_months], errors="coerce").dropna().astype(float)
    curr = float(pd.to_numeric(pd.Series([weekly_stress.loc[prev_ym]]), errors="coerce").iloc[0])
    if hist.empty or not np.isfinite(curr):
        return 1.0, False, curr, float("nan")
    # Lower rolling-5d return means stronger stress.
    q = float(np.quantile(hist.to_numpy(dtype=float), float(cfg.weekly_quantile)))
    active = bool(curr <= q)
    if not active:
        return 1.0, False, curr, q
    return float(max(0.0, min(1.0, cfg.multiplier))), True, curr, q


def _tail_adapt_factor(*, prev_ym: str, alpha_history: list[float], cfg: TailAdaptConfig) -> tuple[float, bool, float]:
    if not cfg.enabled:
        return 1.0, False, float("nan")
    if cfg.start_ym and str(prev_ym) < str(cfg.start_ym):
        return 1.0, False, float("nan")
    lb = int(max(1, cfg.lookback_months))
    if len(alpha_history) < lb:
        return 1.0, False, float("nan")
    recent = np.asarray(alpha_history[-lb:], dtype=float)
    recent = recent[np.isfinite(recent)]
    if recent.size < lb:
        return 1.0, False, float("nan")
    a = float(np.mean(recent))
    thr = float(cfg.alpha_threshold)
    if a < thr:
        return float(max(0.0, min(1.0, cfg.down_multiplier))), True, a
    if a > thr:
        return float(max(1.0, cfg.up_multiplier)), True, a
    return 1.0, False, a


def _auto_aggressive_factor(
    *,
    prev_ym: str,
    prev_idx: int,
    months: list[str],
    state: pd.DataFrame,
    corr_signal: pd.Series,
    vol_signal: pd.Series,
    alpha_history: list[float],
    cfg: AutoAggressiveConfig,
    blocked: bool,
) -> tuple[float, float, bool, float, float, float]:
    if not cfg.enabled or blocked:
        return 1.0, 1.0, False, float("nan"), float("nan"), float("nan")
    if prev_ym not in corr_signal.index or prev_ym not in vol_signal.index:
        return 1.0, 1.0, False, float("nan"), float("nan"), float("nan")
    hist_months = [m for m in months[: max(0, prev_idx - 1)] if m in corr_signal.index and m in vol_signal.index]
    if len(hist_months) < int(cfg.min_history_months):
        return 1.0, 1.0, False, float("nan"), float("nan"), float("nan")

    hist_corr = pd.to_numeric(corr_signal.loc[hist_months], errors="coerce").dropna().astype(float)
    hist_vol = pd.to_numeric(vol_signal.loc[hist_months], errors="coerce").dropna().astype(float)
    curr_corr = float(pd.to_numeric(pd.Series([corr_signal.loc[prev_ym]]), errors="coerce").iloc[0])
    curr_vol = float(pd.to_numeric(pd.Series([vol_signal.loc[prev_ym]]), errors="coerce").iloc[0])
    if hist_corr.empty or hist_vol.empty or not (np.isfinite(curr_corr) and np.isfinite(curr_vol)):
        return 1.0, 1.0, False, float("nan"), curr_corr, curr_vol

    q_corr = float(np.quantile(hist_corr.to_numpy(dtype=float), float(cfg.corr_quantile)))
    q_vol = float(np.quantile(hist_vol.to_numpy(dtype=float), float(cfg.vol_quantile)))
    low_corr = bool(curr_corr <= q_corr)
    low_vol = bool(curr_vol <= q_vol)

    low_score = True
    q_score = float("nan")
    if prev_ym in state.index and "global_score" in state.columns:
        hist_score_months = [m for m in months[: max(0, prev_idx - 1)] if m in state.index]
        if len(hist_score_months) >= int(cfg.min_history_months):
            hist_score = pd.to_numeric(state.loc[hist_score_months, "global_score"], errors="coerce").dropna().astype(float)
            curr_score = float(pd.to_numeric(pd.Series([state.loc[prev_ym, "global_score"]]), errors="coerce").iloc[0])
            if not hist_score.empty and np.isfinite(curr_score):
                q_score = float(np.quantile(hist_score.to_numpy(dtype=float), float(cfg.score_quantile)))
                low_score = bool(curr_score <= q_score)

    if cfg.require_positive_alpha:
        lb = int(max(1, cfg.alpha_lookback_months))
        if len(alpha_history) < lb:
            return 1.0, 1.0, False, q_score, curr_corr, curr_vol
        recent = np.asarray(alpha_history[-lb:], dtype=float)
        recent = recent[np.isfinite(recent)]
        if recent.size < lb or float(np.mean(recent)) <= 0.0:
            return 1.0, 1.0, False, q_score, curr_corr, curr_vol

    active = bool(low_corr and low_vol and low_score)
    if not active:
        return 1.0, 1.0, False, q_score, curr_corr, curr_vol
    return float(max(1.0, cfg.multiplier)), float(max(1.0, cfg.topk_multiplier)), True, q_score, curr_corr, curr_vol


def _l1_turnover(
    *,
    pre_w: dict[str, float],
    pre_cash: float,
    tgt_w: dict[str, float],
    tgt_cash: float,
) -> float:
    keys = set(pre_w.keys()) | set(tgt_w.keys())
    l1 = sum(abs(float(tgt_w.get(k, 0.0)) - float(pre_w.get(k, 0.0))) for k in keys) + abs(float(tgt_cash) - float(pre_cash))
    return float(0.5 * l1)


def _post_trade_weights(
    *,
    exec_w: dict[str, float],
    exec_cash: float,
    ym: str,
    mret: pd.DataFrame,
) -> tuple[dict[str, float], float]:
    if not exec_w:
        return {}, 1.0
    numer: dict[str, float] = {}
    total = 0.0
    for a, w in exec_w.items():
        if ym in mret.index and a in mret.columns:
            r = float(pd.to_numeric(pd.Series([mret.at[ym, a]]), errors="coerce").fillna(0.0).iloc[0])
        else:
            r = 0.0
        v = float(w) * float(1.0 + r)
        if abs(v) > 1e-14:
            numer[str(a)] = v
            total += v
    cash_val = float(exec_cash)
    total += cash_val
    if not np.isfinite(total) or total <= 0.0:
        return {}, 1.0
    post_w = {a: float(v / total) for a, v in numer.items()}
    post_cash = float(max(0.0, min(1.0, cash_val / total)))
    # Re-normalize to avoid drift after clipping.
    s = float(sum(post_w.values()) + post_cash)
    if s > 0:
        post_w = {a: float(v / s) for a, v in post_w.items()}
        post_cash = float(post_cash / s)
    return post_w, post_cash


def _weights_json(weights: dict[str, float]) -> str:
    if not weights:
        return "{}"
    clean = {str(k): float(v) for k, v in sorted(weights.items()) if np.isfinite(float(v)) and abs(float(v)) > 1e-14}
    return json.dumps(clean, ensure_ascii=False, sort_keys=True)


def _evaluate(
    *,
    params: StrategyParams,
    months: list[str],
    snap_by_month: dict[str, pd.DataFrame],
    state: pd.DataFrame,
    mret: pd.DataFrame,
    eqw: pd.Series,
    market: pd.Series,
    mom_cache: dict[int, pd.DataFrame],
    corr_signal: pd.Series,
    vol_signal: pd.Series,
    weekly_stress: pd.Series,
    defense_cfg: DefenseConfig,
    decel_cfg: DecelerationConfig,
    attack_cfg: AttackConfig,
    dd_guard_cfg: DrawdownGuardConfig,
    hedge_cfg: HedgeConfig,
    regime_topk_cfg: RegimeTopKConfig,
    weekly_stress_cfg: WeeklyStressConfig,
    tail_adapt_cfg: TailAdaptConfig,
    continuous_risk_cfg: ContinuousRiskConfig,
    hybrid_cfg: HybridRankingConfig,
    basket_cfg: RegimeBasketConfig,
    layered_cfg: LayeredRotationConfig,
    auto_aggr_cfg: AutoAggressiveConfig,
    rebalance_cfg: RebalanceControlConfig,
    liquidity_map: dict[str, float],
    persistence_by_month: dict[str, dict[str, float]],
    volatility_cache: dict[int, pd.DataFrame],
    market_mom: pd.Series,
    rb_cap: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    alpha_history: list[float] = []
    strat_equity = 1.0
    strat_peak = 1.0
    pre_w: dict[str, float] = {}
    pre_cash: float = 1.0
    reb_cooldown: int = 0
    auto_signal_streak: int = 0
    for i in range(1, len(months)):
        ym = months[i]
        prev = months[i - 1]
        s = snap_by_month.get(prev)
        if s is None or ym not in mret.index:
            continue
        rb, bucket, risk_signal_score, risk_signal_gs_q, risk_signal_corr_q, risk_signal_vol_q = _risk_budget(
            mode=params.mode,
            prev_ym=prev,
            prev_idx=i,
            months=months,
            state=state,
            rb_stress=params.rb_stress,
            rb_transition=params.rb_transition,
            rb_stable=params.rb_stable,
            rb_dispersion=params.rb_dispersion,
            corr_signal=corr_signal,
            vol_signal=vol_signal,
            continuous_cfg=continuous_risk_cfg,
        )
        defense_factor, defense_active = _defense_factor(
            prev_ym=prev,
            prev_idx=i,
            months=months,
            corr_signal=corr_signal,
            vol_signal=vol_signal,
            cfg=defense_cfg,
        )
        attack_factor, attack_active = _attack_factor(
            prev_ym=prev,
            prev_idx=i,
            months=months,
            corr_signal=corr_signal,
            vol_signal=vol_signal,
            alpha_history=alpha_history,
            cfg=attack_cfg,
        )
        dd_guard_factor, dd_guard_active, dd_pre_trade = _drawdown_guard_factor(
            equity=strat_equity,
            peak=strat_peak,
            cfg=dd_guard_cfg,
        )
        weekly_factor, weekly_active, weekly_signal_prev, weekly_threshold = _weekly_stress_factor(
            prev_ym=prev,
            prev_idx=i,
            months=months,
            weekly_stress=weekly_stress,
            cfg=weekly_stress_cfg,
        )
        tail_factor, tail_active, tail_recent_alpha = _tail_adapt_factor(
            prev_ym=prev,
            alpha_history=alpha_history,
            cfg=tail_adapt_cfg,
        )
        decel_factor, decel_active, decel_recent_alpha, decel_streak, topk_factor = _deceleration_state(
            alpha_history=alpha_history,
            cfg=decel_cfg,
        )
        auto_aggr_factor_raw, auto_topk_factor_raw, auto_aggr_raw_active, auto_aggr_score_threshold, auto_aggr_corr_prev, auto_aggr_vol_prev = _auto_aggressive_factor(
            prev_ym=prev,
            prev_idx=i,
            months=months,
            state=state,
            corr_signal=corr_signal,
            vol_signal=vol_signal,
            alpha_history=alpha_history,
            cfg=auto_aggr_cfg,
            blocked=bool(defense_active or dd_guard_active or weekly_active),
        )
        if auto_aggr_raw_active:
            auto_signal_streak += 1
        else:
            auto_signal_streak = 0
        auto_aggr_active = bool(auto_aggr_cfg.enabled and auto_signal_streak >= int(max(1, auto_aggr_cfg.confirm_months)))
        auto_aggr_factor = float(auto_aggr_factor_raw) if auto_aggr_active else 1.0
        auto_topk_factor = float(auto_topk_factor_raw) if auto_aggr_active else 1.0
        regime_topk_factor, regime_topk_bucket = _regime_topk_factor(prev_ym=prev, state=state, cfg=regime_topk_cfg)
        effective_top_k = int(max(1, round(float(params.top_k) * float(topk_factor) * float(regime_topk_factor) * float(auto_topk_factor))))
        s_ranked = s.copy()
        if params.mom_lookback > 0:
            mom_df = mom_cache.get(int(params.mom_lookback))
            if mom_df is not None and prev in mom_df.index:
                mm = mom_df.loc[prev].to_dict()
                s_ranked["mom"] = s_ranked["asset_id"].map(mm)
                s_ranked = s_ranked[(s_ranked["mom"].isna()) | (s_ranked["mom"] >= float(params.mom_threshold))]

        hybrid_mom_df = mom_cache.get(int(max(0, hybrid_cfg.lookback_months)))
        hybrid_vol_df = volatility_cache.get(int(max(0, hybrid_cfg.volatility_lookback_months)))
        s_ranked = _apply_hybrid_ranking(
            s=s_ranked,
            prev_ym=prev,
            mom_df=hybrid_mom_df,
            vol_df=hybrid_vol_df,
            cfg=hybrid_cfg,
            liquidity_map=liquidity_map,
            persistence_map=persistence_by_month.get(prev, {}),
            bucket=bucket,
            basket_cfg=basket_cfg,
        )
        s2, sector_count_selected, sector_names_selected = _select_layered_assets(
            ranked=s_ranked,
            effective_top_k=effective_top_k,
            cfg=layered_cfg,
        )
        n_selected = int(s2.shape[0])
        if n_selected <= 0:
            rb_eff = (
                float(rb)
                * float(defense_factor)
                * float(decel_factor)
                * float(attack_factor)
                * float(dd_guard_factor)
                * float(weekly_factor)
                * float(tail_factor)
                * float(auto_aggr_factor)
            )
            rb_trade = float(max(0.0, min(float(rb_cap), rb_eff)))
            target_w: dict[str, float] = {}
            target_cash = 1.0
            turnover_target = _l1_turnover(pre_w=pre_w, pre_cash=pre_cash, tgt_w=target_w, tgt_cash=target_cash)
            cooldown_before = int(reb_cooldown)
            do_rebalance = True
            reb_reason = "initial_or_forced"
            if rebalance_cfg.enabled and (pre_w or pre_cash < 0.999999):
                if turnover_target < float(max(0.0, rebalance_cfg.deadband_l1)):
                    do_rebalance = False
                    reb_reason = "deadband_hold"
                elif reb_cooldown > 0 and turnover_target < float(max(float(rebalance_cfg.deadband_l1), float(rebalance_cfg.force_l1))):
                    do_rebalance = False
                    reb_reason = "cooldown_hold"
            if do_rebalance:
                exec_w = target_w
                exec_cash = target_cash
                turnover_exec = float(turnover_target)
                reb_cooldown = int(max(0, rebalance_cfg.cooldown_months)) if rebalance_cfg.enabled else 0
            else:
                exec_w = dict(pre_w)
                exec_cash = float(pre_cash)
                turnover_exec = 0.0
                reb_cooldown = int(max(0, reb_cooldown - 1))
            cooldown_after = int(reb_cooldown)
            core_ret = 0.0
            if exec_w:
                vals_core = []
                w_core = []
                for a, wv in exec_w.items():
                    if ym in mret.index and a in mret.columns:
                        rv = float(pd.to_numeric(pd.Series([mret.at[ym, a]]), errors="coerce").fillna(0.0).iloc[0])
                    else:
                        rv = 0.0
                    vals_core.append(rv)
                    w_core.append(float(wv))
                if w_core:
                    core_ret = float(np.dot(np.asarray(w_core, dtype=float), np.asarray(vals_core, dtype=float)))
            eqw_ret = float(eqw.get(ym, 0.0))
            hedge_active = False
            hedge_ret = 0.0
            hedge_weight = 0.0
            if hedge_cfg.enabled and prev in market_mom.index:
                prev_mom = float(pd.to_numeric(pd.Series([market_mom.loc[prev]]), errors="coerce").iloc[0])
                cond = bool(np.isfinite(prev_mom) and prev_mom <= float(hedge_cfg.mom_threshold))
                if hedge_cfg.activate_on_defense:
                    cond = bool(cond and (defense_active or dd_guard_active))
                if cond:
                    hedge_active = True
                    hedge_weight = float(-abs(hedge_cfg.multiplier))
                    hedge_ret = float(hedge_weight * _series_value_or_raise(market, ym, "market"))
            strategy_ret = float(core_ret + hedge_ret)
            alpha_now = float(strategy_ret - eqw_ret)
            alpha_history.append(alpha_now)
            strat_equity = float(strat_equity * (1.0 + strategy_ret))
            strat_peak = float(max(strat_peak, strat_equity))
            pre_w, pre_cash = _post_trade_weights(exec_w=exec_w, exec_cash=exec_cash, ym=ym, mret=mret)
            rows.append(
                {
                    "ym": ym,
                    "ret": strategy_ret,
                    "eqw_ret": eqw_ret,
                    "mkt_ret": _series_value_or_raise(market, ym, "market"),
                    "motor_ret": float(rb_trade) * eqw_ret,
                    "risk_bucket": bucket,
                    "risk_budget": float(rb_eff),
                    "risk_budget_trade": float(rb_trade),
                    "risk_signal_score": float(risk_signal_score) if np.isfinite(risk_signal_score) else float("nan"),
                    "risk_signal_gs_q": float(risk_signal_gs_q) if np.isfinite(risk_signal_gs_q) else float("nan"),
                    "risk_signal_corr_q": float(risk_signal_corr_q) if np.isfinite(risk_signal_corr_q) else float("nan"),
                    "risk_signal_vol_q": float(risk_signal_vol_q) if np.isfinite(risk_signal_vol_q) else float("nan"),
                    "defense_active": bool(defense_active),
                    "defense_factor": float(defense_factor),
                    "decel_active": bool(decel_active),
                    "decel_factor": float(decel_factor),
                    "decel_recent_alpha": float(decel_recent_alpha) if np.isfinite(decel_recent_alpha) else float("nan"),
                    "decel_streak": int(decel_streak),
                    "attack_active": bool(attack_active),
                    "attack_factor": float(attack_factor),
                    "dd_guard_active": bool(dd_guard_active),
                    "dd_guard_factor": float(dd_guard_factor),
                    "dd_pre_trade": float(dd_pre_trade),
                    "weekly_stress_active": bool(weekly_active),
                    "weekly_stress_factor": float(weekly_factor),
                    "weekly_stress_prev": float(weekly_signal_prev) if np.isfinite(weekly_signal_prev) else float("nan"),
                    "weekly_stress_threshold": float(weekly_threshold) if np.isfinite(weekly_threshold) else float("nan"),
                    "tail_adapt_active": bool(tail_active),
                    "tail_adapt_factor": float(tail_factor),
                    "tail_recent_alpha": float(tail_recent_alpha) if np.isfinite(tail_recent_alpha) else float("nan"),
                    "auto_aggressive_active": bool(auto_aggr_active),
                    "auto_aggressive_raw_active": bool(auto_aggr_raw_active),
                    "auto_aggressive_streak": int(auto_signal_streak),
                    "auto_aggressive_factor": float(auto_aggr_factor),
                    "auto_aggressive_topk_factor": float(auto_topk_factor),
                    "auto_aggressive_score_threshold": float(auto_aggr_score_threshold) if np.isfinite(auto_aggr_score_threshold) else float("nan"),
                    "auto_aggressive_corr_prev": float(auto_aggr_corr_prev) if np.isfinite(auto_aggr_corr_prev) else float("nan"),
                    "auto_aggressive_vol_prev": float(auto_aggr_vol_prev) if np.isfinite(auto_aggr_vol_prev) else float("nan"),
                    "regime_topk_bucket": str(regime_topk_bucket),
                    "regime_topk_factor": float(regime_topk_factor),
                    "hedge_active": bool(hedge_active),
                    "hedge_ret": float(hedge_ret),
                    "effective_top_k": int(effective_top_k),
                    "sector_count_selected": int(sector_count_selected),
                    "sector_names_selected": str(sector_names_selected),
                    "regime_basket_enabled": bool(basket_cfg.enabled),
                    "hybrid_ranking_enabled": bool(hybrid_cfg.enabled),
                    "layered_rotation_enabled": bool(layered_cfg.enabled),
                    "selected_assets": "",
                    "executed_assets": ",".join(sorted(exec_w.keys())),
                    "executed_weights_json": _weights_json(exec_w),
                    "cash_weight": float(exec_cash),
                    "hedge_weight": float(hedge_weight),
                    "core_gross_exposure": float(sum(abs(float(v)) for v in exec_w.values())),
                    "net_exposure": float(sum(float(v) for v in exec_w.values()) + float(hedge_weight)),
                    "turnover_target": float(turnover_target),
                    "turnover": float(turnover_exec),
                    "rebalance_executed": bool(do_rebalance),
                    "rebalance_reason": str(reb_reason),
                    "rebalance_cooldown_before": int(cooldown_before),
                    "rebalance_cooldown_after": int(cooldown_after),
                    "n_selected": 0,
                }
            )
            continue
        base = np.ones(n_selected, dtype=float)
        if float(params.impact_power) > 0.0:
            base = np.power(np.maximum(s2["impact_global"].to_numpy(dtype=float), 0.0), float(params.impact_power))
            if float(np.sum(base)) <= 0.0:
                base = np.ones(n_selected, dtype=float)
        w = _cap_weights(base, w_max=float(params.w_max))
        rb_eff = (
            float(rb)
            * float(defense_factor)
            * float(decel_factor)
            * float(attack_factor)
            * float(dd_guard_factor)
            * float(weekly_factor)
            * float(tail_factor)
            * float(auto_aggr_factor)
        )
        rb_trade = float(max(0.0, min(float(rb_cap), rb_eff)))
        target_w = {str(a): float(rb_trade * ww) for a, ww in zip(s2["asset_id"].astype(str).tolist(), w.tolist(), strict=False)}
        target_cash = float(max(0.0, 1.0 - float(sum(target_w.values()))))
        turnover_target = _l1_turnover(pre_w=pre_w, pre_cash=pre_cash, tgt_w=target_w, tgt_cash=target_cash)
        cooldown_before = int(reb_cooldown)
        do_rebalance = True
        reb_reason = "initial_or_forced"
        if rebalance_cfg.enabled and (pre_w or pre_cash < 0.999999):
            if turnover_target < float(max(0.0, rebalance_cfg.deadband_l1)):
                do_rebalance = False
                reb_reason = "deadband_hold"
            elif reb_cooldown > 0 and turnover_target < float(max(float(rebalance_cfg.deadband_l1), float(rebalance_cfg.force_l1))):
                do_rebalance = False
                reb_reason = "cooldown_hold"
        if do_rebalance:
            exec_w = target_w
            exec_cash = target_cash
            turnover_exec = float(turnover_target)
            reb_cooldown = int(max(0, rebalance_cfg.cooldown_months)) if rebalance_cfg.enabled else 0
        else:
            exec_w = dict(pre_w)
            exec_cash = float(pre_cash)
            turnover_exec = 0.0
            reb_cooldown = int(max(0, reb_cooldown - 1))
        cooldown_after = int(reb_cooldown)
        core_ret = 0.0
        if exec_w:
            vals_core = []
            w_core = []
            for a, wv in exec_w.items():
                if ym in mret.index and a in mret.columns:
                    rv = float(pd.to_numeric(pd.Series([mret.at[ym, a]]), errors="coerce").fillna(0.0).iloc[0])
                else:
                    rv = 0.0
                vals_core.append(rv)
                w_core.append(float(wv))
            core_ret = float(np.dot(np.asarray(w_core, dtype=float), np.asarray(vals_core, dtype=float)))
        eqw_ret = float(eqw.get(ym, 0.0))
        hedge_active = False
        hedge_ret = 0.0
        hedge_weight = 0.0
        if hedge_cfg.enabled and prev in market_mom.index:
            prev_mom = float(pd.to_numeric(pd.Series([market_mom.loc[prev]]), errors="coerce").iloc[0])
            cond = bool(np.isfinite(prev_mom) and prev_mom <= float(hedge_cfg.mom_threshold))
            if hedge_cfg.activate_on_defense:
                cond = bool(cond and (defense_active or dd_guard_active))
            if cond:
                hedge_active = True
                hedge_weight = float(-abs(hedge_cfg.multiplier))
                hedge_ret = float(hedge_weight * _series_value_or_raise(market, ym, "market"))
        strategy_ret = float(core_ret + hedge_ret)
        alpha_now = float(strategy_ret - eqw_ret)
        alpha_history.append(alpha_now)
        strat_equity = float(strat_equity * (1.0 + strategy_ret))
        strat_peak = float(max(strat_peak, strat_equity))
        pre_w, pre_cash = _post_trade_weights(exec_w=exec_w, exec_cash=exec_cash, ym=ym, mret=mret)
        rows.append(
            {
                "ym": ym,
                "ret": strategy_ret,
                "eqw_ret": eqw_ret,
                "mkt_ret": _series_value_or_raise(market, ym, "market"),
                "motor_ret": float(rb_trade) * eqw_ret,
                "risk_bucket": bucket,
                "risk_budget": float(rb_eff),
                "risk_budget_trade": float(rb_trade),
                "risk_signal_score": float(risk_signal_score) if np.isfinite(risk_signal_score) else float("nan"),
                "risk_signal_gs_q": float(risk_signal_gs_q) if np.isfinite(risk_signal_gs_q) else float("nan"),
                "risk_signal_corr_q": float(risk_signal_corr_q) if np.isfinite(risk_signal_corr_q) else float("nan"),
                "risk_signal_vol_q": float(risk_signal_vol_q) if np.isfinite(risk_signal_vol_q) else float("nan"),
                "defense_active": bool(defense_active),
                "defense_factor": float(defense_factor),
                "decel_active": bool(decel_active),
                "decel_factor": float(decel_factor),
                "decel_recent_alpha": float(decel_recent_alpha) if np.isfinite(decel_recent_alpha) else float("nan"),
                "decel_streak": int(decel_streak),
                "attack_active": bool(attack_active),
                "attack_factor": float(attack_factor),
                "dd_guard_active": bool(dd_guard_active),
                "dd_guard_factor": float(dd_guard_factor),
                "dd_pre_trade": float(dd_pre_trade),
                "weekly_stress_active": bool(weekly_active),
                "weekly_stress_factor": float(weekly_factor),
                "weekly_stress_prev": float(weekly_signal_prev) if np.isfinite(weekly_signal_prev) else float("nan"),
                "weekly_stress_threshold": float(weekly_threshold) if np.isfinite(weekly_threshold) else float("nan"),
                "tail_adapt_active": bool(tail_active),
                "tail_adapt_factor": float(tail_factor),
                "tail_recent_alpha": float(tail_recent_alpha) if np.isfinite(tail_recent_alpha) else float("nan"),
                "auto_aggressive_active": bool(auto_aggr_active),
                "auto_aggressive_raw_active": bool(auto_aggr_raw_active),
                "auto_aggressive_streak": int(auto_signal_streak),
                "auto_aggressive_factor": float(auto_aggr_factor),
                "auto_aggressive_topk_factor": float(auto_topk_factor),
                "auto_aggressive_score_threshold": float(auto_aggr_score_threshold) if np.isfinite(auto_aggr_score_threshold) else float("nan"),
                "auto_aggressive_corr_prev": float(auto_aggr_corr_prev) if np.isfinite(auto_aggr_corr_prev) else float("nan"),
                "auto_aggressive_vol_prev": float(auto_aggr_vol_prev) if np.isfinite(auto_aggr_vol_prev) else float("nan"),
                "regime_topk_bucket": str(regime_topk_bucket),
                "regime_topk_factor": float(regime_topk_factor),
                "hedge_active": bool(hedge_active),
                "hedge_ret": float(hedge_ret),
                "effective_top_k": int(effective_top_k),
                "sector_count_selected": int(sector_count_selected),
                "sector_names_selected": str(sector_names_selected),
                "regime_basket_enabled": bool(basket_cfg.enabled),
                "hybrid_ranking_enabled": bool(hybrid_cfg.enabled),
                "layered_rotation_enabled": bool(layered_cfg.enabled),
                "selected_assets": ",".join(s2["asset_id"].astype(str).tolist()),
                "executed_assets": ",".join(sorted(exec_w.keys())),
                "executed_weights_json": _weights_json(exec_w),
                "cash_weight": float(exec_cash),
                "hedge_weight": float(hedge_weight),
                "core_gross_exposure": float(sum(abs(float(v)) for v in exec_w.values())),
                "net_exposure": float(sum(float(v) for v in exec_w.values()) + float(hedge_weight)),
                "turnover_target": float(turnover_target),
                "turnover": float(turnover_exec),
                "rebalance_executed": bool(do_rebalance),
                "rebalance_reason": str(reb_reason),
                "rebalance_cooldown_before": int(cooldown_before),
                "rebalance_cooldown_after": int(cooldown_after),
                "n_selected": n_selected,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["year"] = out["ym"].str[:4].astype(int)
    return out


def _split_metrics(monthly: pd.DataFrame, train_end_ym: str, *, opt_cost_bps: float) -> dict[str, float]:
    d = monthly.copy()
    if "turnover" not in d.columns:
        d["turnover"] = 0.0
    d["turnover"] = pd.to_numeric(d["turnover"], errors="coerce").fillna(0.0).astype(float)
    cost_rate = float(max(0.0, opt_cost_bps) / 10000.0)
    d["ret_net"] = pd.to_numeric(d["ret"], errors="coerce").fillna(0.0).astype(float) - d["turnover"] * cost_rate
    tr = d[d["ym"] <= train_end_ym].copy()
    te = d[d["ym"] > train_end_ym].copy()

    def _pack(z: pd.DataFrame, pref: str) -> dict[str, float]:
        if z.empty:
            return {
                f"{pref}_ann": float("nan"),
                f"{pref}_eqw_ann": float("nan"),
                f"{pref}_mdd": float("nan"),
                f"{pref}_prob_positive": float("nan"),
                f"{pref}_worth_rate": float("nan"),
                f"{pref}_min_year_return": float("nan"),
            }
        alpha = pd.to_numeric(z["ret"], errors="coerce") - pd.to_numeric(z["eqw_ret"], errors="coerce")
        yearly = (
            z.groupby("year", as_index=False)
            .agg(
                strategy_total=("ret", _total),
                eqw_total=("eqw_ret", _total),
            )
            .copy()
        )
        worth = float((yearly["strategy_total"] > yearly["eqw_total"]).mean()) if not yearly.empty else float("nan")
        return {
            f"{pref}_ann": _ann(z["ret"]),
            f"{pref}_ann_net": _ann(z["ret_net"]),
            f"{pref}_eqw_ann": _ann(z["eqw_ret"]),
            f"{pref}_mdd": _mdd(z["ret"]),
            f"{pref}_mdd_net": _mdd(z["ret_net"]),
            f"{pref}_prob_positive": float((alpha > 0.0).mean()) if not alpha.empty else float("nan"),
            f"{pref}_worth_rate": worth,
            f"{pref}_min_year_return": float(pd.to_numeric(yearly["strategy_total"], errors="coerce").min()) if not yearly.empty else float("nan"),
        }

    m = {}
    m.update(_pack(tr, "train"))
    m.update(_pack(te, "test"))
    alpha_full = pd.to_numeric(d["ret"], errors="coerce") - pd.to_numeric(d["eqw_ret"], errors="coerce")
    m["full_ann"] = _ann(d["ret"])
    m["full_ann_net"] = _ann(d["ret_net"])
    m["full_eqw_ann"] = _ann(d["eqw_ret"])
    m["full_mdd"] = _mdd(d["ret"])
    m["full_mdd_net"] = _mdd(d["ret_net"])
    m["full_prob_positive"] = float((alpha_full > 0.0).mean()) if not alpha_full.empty else float("nan")
    m["full_alpha_recent6"] = float(alpha_full.tail(6).mean()) if len(alpha_full) >= 1 else float("nan")
    alpha_full_net = pd.to_numeric(d["ret_net"], errors="coerce") - pd.to_numeric(d["eqw_ret"], errors="coerce")
    m["full_alpha_recent6_net"] = float(alpha_full_net.tail(6).mean()) if len(alpha_full_net) >= 1 else float("nan")
    m["avg_turnover"] = float(pd.to_numeric(d["turnover"], errors="coerce").mean()) if not d.empty else float("nan")
    yearly_full = (
        d.groupby("year", as_index=False)
        .agg(strategy_total=("ret", _total), eqw_total=("eqw_ret", _total))
        .copy()
    )
    m["full_worth_rate"] = float((yearly_full["strategy_total"] > yearly_full["eqw_total"]).mean()) if not yearly_full.empty else float("nan")
    m["full_min_year_return"] = float(pd.to_numeric(yearly_full["strategy_total"], errors="coerce").min()) if not yearly_full.empty else float("nan")
    return m


def _score_candidate(
    m: dict[str, float],
    *,
    objective_mode: str = "balanced",
    use_net_ann: bool = True,
    turnover_penalty: float = 0.0,
) -> float:
    train_alpha = float(m.get("train_ann_net" if use_net_ann else "train_ann", np.nan)) - float(m.get("train_eqw_ann", np.nan))
    test_alpha = float(m.get("test_ann_net" if use_net_ann else "test_ann", np.nan)) - float(m.get("test_eqw_ann", np.nan))
    full_alpha6 = float(m.get("full_alpha_recent6_net" if use_net_ann else "full_alpha_recent6", np.nan))
    full_worth = float(m.get("full_worth_rate", np.nan))
    full_prob = float(m.get("full_prob_positive", np.nan))
    full_mdd = float(m.get("full_mdd_net" if use_net_ann else "full_mdd", np.nan))
    test_min_year = float(m.get("test_min_year_return", np.nan))
    full_min_year = float(m.get("full_min_year_return", np.nan))
    avg_turnover = float(m.get("avg_turnover", np.nan))
    mode = str(objective_mode).strip().lower()
    if mode == "maximin":
        score = 0.0
        score += 6.0 * (test_min_year if np.isfinite(test_min_year) else -1.0)
        score += 2.0 * (test_alpha if np.isfinite(test_alpha) else -1.0)
        score += 1.0 * (full_worth if np.isfinite(full_worth) else 0.0)
        score += 0.8 * (full_prob if np.isfinite(full_prob) else 0.0)
        score += 0.5 * (full_alpha6 if np.isfinite(full_alpha6) else -1.0)
        if np.isfinite(test_min_year) and test_min_year < 0.0:
            score -= 8.0 * abs(test_min_year)
        if np.isfinite(full_min_year) and full_min_year < 0.0:
            score -= 4.0 * abs(full_min_year)
        if np.isfinite(full_mdd) and full_mdd < -0.35:
            score -= 6.0 * abs(full_mdd + 0.35)
        if np.isfinite(avg_turnover):
            score -= float(max(0.0, turnover_penalty)) * float(avg_turnover)
        return float(score)
    score = 0.0
    score += 2.5 * (train_alpha if np.isfinite(train_alpha) else -1.0)
    score += 2.0 * (test_alpha if np.isfinite(test_alpha) else -1.0)
    score += 1.2 * (full_worth if np.isfinite(full_worth) else 0.0)
    score += 1.2 * (full_prob if np.isfinite(full_prob) else 0.0)
    score += 0.8 * (full_alpha6 if np.isfinite(full_alpha6) else -1.0)
    if np.isfinite(full_mdd) and full_mdd < -0.35:
        score -= 6.0 * abs(full_mdd + 0.35)
    if np.isfinite(full_alpha6) and full_alpha6 < -0.003:
        score -= 12.0 * abs(full_alpha6 + 0.003)
    if np.isfinite(avg_turnover):
        score -= float(max(0.0, turnover_penalty)) * float(avg_turnover)
    return float(score)


def _yearly_eval(monthly: pd.DataFrame) -> pd.DataFrame:
    y = (
        monthly.groupby("year", as_index=False)
        .agg(
            strategy_total=("ret", _total),
            eqw_total=("eqw_ret", _total),
            market_total=("mkt_ret", _total),
            motor_total=("motor_ret", _total),
        )
        .copy()
    )
    y["alpha_total_vs_eqw"] = y["strategy_total"] - y["eqw_total"]
    y["alpha_total_vs_market"] = y["strategy_total"] - y["market_total"]
    y["worth_it_vs_eqw"] = y["strategy_total"] > y["eqw_total"]
    y["worth_it_vs_market"] = y["strategy_total"] > y["market_total"]
    return y


def main() -> None:
    ap = argparse.ArgumentParser(description="Canonical causal systematic evaluation + grid optimization.")
    ap.add_argument("--impact-dir", default="", help="Path to impact_learning* directory.")
    ap.add_argument("--returns-csv", default="", help="Path to returns_wide_core.csv.")
    ap.add_argument("--outdir", default="", help="Output dir (default: results/portfolio_sim/<runid>_systematic_yearly).")
    ap.add_argument("--train-end", default="2023-12-31")
    ap.add_argument("--start-ym", default="2019-01")
    ap.add_argument("--max-assets-per-month", type=int, default=80)
    ap.add_argument("--top-k-options", default="20,24,28,32,36,40,48")
    ap.add_argument("--impact-power-options", default="0,1")
    ap.add_argument("--wmax-options", default="0.1,0.12,0.15")
    ap.add_argument("--mom-lookback-options", default="0,1,3")
    ap.add_argument("--mom-threshold-options", default="-0.02,0.0,0.02")
    ap.add_argument("--modes", default="const,regime,score")
    ap.add_argument("--rb-stress-options", default="", help="Optional comma-separated risk budget values for stress.")
    ap.add_argument("--rb-transition-options", default="", help="Optional comma-separated risk budget values for transition.")
    ap.add_argument("--rb-stable-options", default="", help="Optional comma-separated risk budget values for stable.")
    ap.add_argument("--rb-dispersion-options", default="", help="Optional comma-separated risk budget values for dispersion.")
    ap.add_argument("--objective-mode", default="balanced", choices=["balanced", "maximin"])
    ap.add_argument("--benchmark-symbol", default="SPY")
    ap.add_argument("--prices-dir", default=str(ROOT / "data" / "raw" / "finance" / "yfinance_daily"))
    ap.add_argument(
        "--max-grid-combos",
        type=int,
        default=0,
        help="Maximum number of parameter combinations to evaluate (0 = evaluate full grid).",
    )
    ap.add_argument(
        "--min-months",
        type=int,
        default=24,
        help="Minimum overlapping months required after --start-ym (default: 24).",
    )
    ap.add_argument("--defense-enabled", type=int, default=1)
    ap.add_argument("--defense-multiplier", type=float, default=0.85)
    ap.add_argument("--defense-corr-quantile", type=float, default=0.8)
    ap.add_argument("--defense-vol-quantile", type=float, default=0.8)
    ap.add_argument("--defense-min-history-months", type=int, default=12)
    ap.add_argument("--defense-require-both", type=int, default=0)
    ap.add_argument("--decel-enabled", type=int, default=1)
    ap.add_argument("--decel-lookback-months", type=int, default=6)
    ap.add_argument("--decel-alpha-threshold", type=float, default=0.0)
    ap.add_argument("--decel-min-streak", type=int, default=2)
    ap.add_argument("--decel-multiplier", type=float, default=0.9)
    ap.add_argument("--decel-topk-multiplier", type=float, default=0.85)
    ap.add_argument("--attack-enabled", type=int, default=0)
    ap.add_argument("--attack-multiplier", type=float, default=1.15)
    ap.add_argument("--attack-corr-quantile", type=float, default=0.35)
    ap.add_argument("--attack-vol-quantile", type=float, default=0.35)
    ap.add_argument("--attack-min-history-months", type=int, default=12)
    ap.add_argument("--attack-require-both", type=int, default=1)
    ap.add_argument("--attack-require-positive-alpha", type=int, default=1)
    ap.add_argument("--attack-alpha-lookback-months", type=int, default=3)
    ap.add_argument("--dd-guard-enabled", type=int, default=0)
    ap.add_argument("--dd-soft-threshold", type=float, default=-0.08)
    ap.add_argument("--dd-hard-threshold", type=float, default=-0.12)
    ap.add_argument("--dd-soft-multiplier", type=float, default=0.5)
    ap.add_argument("--dd-hard-multiplier", type=float, default=0.2)
    ap.add_argument("--hedge-enabled", type=int, default=0)
    ap.add_argument("--hedge-multiplier", type=float, default=0.35)
    ap.add_argument("--hedge-mom-lookback", type=int, default=3)
    ap.add_argument("--hedge-mom-threshold", type=float, default=-0.01)
    ap.add_argument("--hedge-activate-on-defense", type=int, default=1)
    ap.add_argument("--regime-topk-enabled", type=int, default=0)
    ap.add_argument("--regime-topk-stable-multiplier", type=float, default=1.0)
    ap.add_argument("--regime-topk-transition-multiplier", type=float, default=0.9)
    ap.add_argument("--regime-topk-stress-multiplier", type=float, default=0.8)
    ap.add_argument("--weekly-stress-enabled", type=int, default=0)
    ap.add_argument("--weekly-stress-quantile", type=float, default=0.2)
    ap.add_argument("--weekly-stress-min-history-months", type=int, default=12)
    ap.add_argument("--weekly-stress-multiplier", type=float, default=0.9)
    ap.add_argument("--tail-adapt-enabled", type=int, default=0)
    ap.add_argument("--tail-adapt-lookback-months", type=int, default=3)
    ap.add_argument("--tail-adapt-alpha-threshold", type=float, default=0.0)
    ap.add_argument("--tail-adapt-down-multiplier", type=float, default=0.9)
    ap.add_argument("--tail-adapt-up-multiplier", type=float, default=1.05)
    ap.add_argument("--tail-adapt-start-ym", default="")
    ap.add_argument("--hybrid-enabled", type=int, default=0)
    ap.add_argument("--hybrid-lookback-months", type=int, default=3)
    ap.add_argument("--hybrid-weight-impact", type=float, default=0.6)
    ap.add_argument("--hybrid-weight-momentum", type=float, default=0.25)
    ap.add_argument("--hybrid-weight-liquidity", type=float, default=0.15)
    ap.add_argument("--hybrid-weight-persistence", type=float, default=0.15)
    ap.add_argument("--hybrid-weight-sector-strength", type=float, default=0.10)
    ap.add_argument("--hybrid-weight-low-corr", type=float, default=0.10)
    ap.add_argument("--hybrid-weight-low-vol", type=float, default=0.10)
    ap.add_argument("--hybrid-weight-low-concentration", type=float, default=0.05)
    ap.add_argument("--hybrid-liquidity-csv", default="")
    ap.add_argument("--hybrid-persistence-lookback-months", type=int, default=3)
    ap.add_argument("--hybrid-volatility-lookback-months", type=int, default=3)
    ap.add_argument("--layered-enabled", type=int, default=0)
    ap.add_argument("--layered-min-sectors", type=int, default=3)
    ap.add_argument("--layered-max-sectors", type=int, default=8)
    ap.add_argument("--layered-target-assets-per-sector", type=int, default=8)
    ap.add_argument("--layered-sector-score-power", type=float, default=1.0)
    ap.add_argument("--layered-min-assets-per-sector", type=int, default=1)
    ap.add_argument("--execution-universe-csv", type=str, default="")
    ap.add_argument("--continuous-risk-enabled", type=int, default=0)
    ap.add_argument("--continuous-risk-weight-global-score", type=float, default=0.5)
    ap.add_argument("--continuous-risk-weight-corr", type=float, default=0.3)
    ap.add_argument("--continuous-risk-weight-vol", type=float, default=0.2)
    ap.add_argument("--continuous-risk-regime-bias", type=float, default=0.75)
    ap.add_argument("--continuous-risk-score-power", type=float, default=1.0)
    ap.add_argument("--regime-basket-enabled", type=int, default=0)
    ap.add_argument("--regime-basket-sector-bonus", type=float, default=0.08)
    ap.add_argument("--regime-basket-global-sleeve-bonus", type=float, default=0.05)
    ap.add_argument("--regime-basket-vote-threshold", type=float, default=0.5)
    ap.add_argument("--auto-aggr-enabled", type=int, default=0)
    ap.add_argument("--auto-aggr-multiplier", type=float, default=1.1)
    ap.add_argument("--auto-aggr-topk-multiplier", type=float, default=1.1)
    ap.add_argument("--auto-aggr-score-quantile", type=float, default=0.35)
    ap.add_argument("--auto-aggr-corr-quantile", type=float, default=0.45)
    ap.add_argument("--auto-aggr-vol-quantile", type=float, default=0.45)
    ap.add_argument("--auto-aggr-min-history-months", type=int, default=12)
    ap.add_argument("--auto-aggr-require-positive-alpha", type=int, default=1)
    ap.add_argument("--auto-aggr-alpha-lookback-months", type=int, default=3)
    ap.add_argument("--auto-aggr-confirm-months", type=int, default=2)
    ap.add_argument("--rebalance-control-enabled", type=int, default=0)
    ap.add_argument("--rebalance-deadband-l1", type=float, default=0.08)
    ap.add_argument("--rebalance-force-l1", type=float, default=0.22)
    ap.add_argument("--rebalance-cooldown-months", type=int, default=1)
    ap.add_argument("--opt-cost-bps", type=float, default=10.0)
    ap.add_argument("--opt-turnover-penalty", type=float, default=0.0)
    ap.add_argument("--opt-use-net-ann", type=int, default=1)
    ap.add_argument("--shadow-tail-months", type=int, default=0)
    ap.add_argument("--rb-cap", type=float, default=1.5)
    args = ap.parse_args()

    if args.impact_dir.strip():
        impact_dir = Path(args.impact_dir).resolve()
        returns_csv = Path(args.returns_csv).resolve() if args.returns_csv.strip() else impact_dir.parents[1] / "returns_wide_core.csv"
    else:
        impact_dir, returns_csv = _latest_impact_dir()
    if not impact_dir.exists():
        raise FileNotFoundError(f"impact dir not found: {impact_dir}")
    impact_csv = impact_dir / "impact_training_dataset.csv"
    if not impact_csv.exists():
        raise FileNotFoundError(f"missing file: {impact_csv}")
    if not returns_csv.exists():
        raise FileNotFoundError(f"returns csv not found: {returns_csv}")

    run_id = _run_id()
    outdir = Path(args.outdir).resolve() if args.outdir.strip() else (ROOT / "results" / "portfolio_sim" / f"{run_id}_systematic_yearly")
    outdir.mkdir(parents=True, exist_ok=True)

    prices_dir = Path(str(args.prices_dir)).resolve()
    mret, eqw, market, corr_signal, vol_signal, weekly_stress_signal = _build_monthly_matrices(returns_csv)
    if args.benchmark_symbol in mret.columns:
        market = mret[args.benchmark_symbol].copy()
    else:
        market = _load_external_benchmark_monthly(prices_dir, args.benchmark_symbol, mret.index)
    defense_cfg = DefenseConfig(
        enabled=bool(int(args.defense_enabled)),
        multiplier=float(args.defense_multiplier),
        corr_quantile=float(args.defense_corr_quantile),
        vol_quantile=float(args.defense_vol_quantile),
        min_history_months=int(args.defense_min_history_months),
        require_both=bool(int(args.defense_require_both)),
    )
    decel_cfg = DecelerationConfig(
        enabled=bool(int(args.decel_enabled)),
        lookback_months=int(args.decel_lookback_months),
        alpha_threshold=float(args.decel_alpha_threshold),
        min_streak=int(args.decel_min_streak),
        multiplier=float(args.decel_multiplier),
        topk_multiplier=float(args.decel_topk_multiplier),
    )
    attack_cfg = AttackConfig(
        enabled=bool(int(args.attack_enabled)),
        multiplier=float(args.attack_multiplier),
        corr_quantile=float(args.attack_corr_quantile),
        vol_quantile=float(args.attack_vol_quantile),
        min_history_months=int(args.attack_min_history_months),
        require_both=bool(int(args.attack_require_both)),
        require_positive_alpha=bool(int(args.attack_require_positive_alpha)),
        alpha_lookback_months=int(args.attack_alpha_lookback_months),
    )
    dd_guard_cfg = DrawdownGuardConfig(
        enabled=bool(int(args.dd_guard_enabled)),
        soft_threshold=float(args.dd_soft_threshold),
        hard_threshold=float(args.dd_hard_threshold),
        soft_multiplier=float(args.dd_soft_multiplier),
        hard_multiplier=float(args.dd_hard_multiplier),
    )
    hedge_cfg = HedgeConfig(
        enabled=bool(int(args.hedge_enabled)),
        multiplier=float(args.hedge_multiplier),
        mom_lookback=int(args.hedge_mom_lookback),
        mom_threshold=float(args.hedge_mom_threshold),
        activate_on_defense=bool(int(args.hedge_activate_on_defense)),
    )
    regime_topk_cfg = RegimeTopKConfig(
        enabled=bool(int(args.regime_topk_enabled)),
        stable_multiplier=float(args.regime_topk_stable_multiplier),
        transition_multiplier=float(args.regime_topk_transition_multiplier),
        stress_multiplier=float(args.regime_topk_stress_multiplier),
    )
    weekly_stress_cfg = WeeklyStressConfig(
        enabled=bool(int(args.weekly_stress_enabled)),
        weekly_quantile=float(args.weekly_stress_quantile),
        min_history_months=int(args.weekly_stress_min_history_months),
        multiplier=float(args.weekly_stress_multiplier),
    )
    tail_adapt_cfg = TailAdaptConfig(
        enabled=bool(int(args.tail_adapt_enabled)),
        lookback_months=int(args.tail_adapt_lookback_months),
        alpha_threshold=float(args.tail_adapt_alpha_threshold),
        down_multiplier=float(args.tail_adapt_down_multiplier),
        up_multiplier=float(args.tail_adapt_up_multiplier),
        start_ym=str(args.tail_adapt_start_ym).strip(),
    )
    hybrid_cfg = HybridRankingConfig(
        enabled=bool(int(args.hybrid_enabled)),
        lookback_months=int(args.hybrid_lookback_months),
        weight_impact=float(args.hybrid_weight_impact),
        weight_momentum=float(args.hybrid_weight_momentum),
        weight_liquidity=float(args.hybrid_weight_liquidity),
        weight_persistence=float(args.hybrid_weight_persistence),
        weight_sector_strength=float(args.hybrid_weight_sector_strength),
        weight_low_corr=float(args.hybrid_weight_low_corr),
        weight_low_vol=float(args.hybrid_weight_low_vol),
        weight_low_concentration=float(args.hybrid_weight_low_concentration),
        liquidity_csv=str(args.hybrid_liquidity_csv).strip(),
        persistence_lookback_months=int(args.hybrid_persistence_lookback_months),
        volatility_lookback_months=int(args.hybrid_volatility_lookback_months),
    )
    continuous_risk_cfg = ContinuousRiskConfig(
        enabled=bool(int(args.continuous_risk_enabled)),
        weight_global_score=float(args.continuous_risk_weight_global_score),
        weight_corr=float(args.continuous_risk_weight_corr),
        weight_vol=float(args.continuous_risk_weight_vol),
        regime_bias=float(args.continuous_risk_regime_bias),
        score_power=float(args.continuous_risk_score_power),
    )
    basket_cfg = RegimeBasketConfig(
        enabled=bool(int(args.regime_basket_enabled)),
        sector_bonus=float(args.regime_basket_sector_bonus),
        global_sleeve_bonus=float(args.regime_basket_global_sleeve_bonus),
        vote_threshold=float(args.regime_basket_vote_threshold),
    )
    layered_cfg = LayeredRotationConfig(
        enabled=bool(int(args.layered_enabled)),
        min_sectors=int(args.layered_min_sectors),
        max_sectors=int(args.layered_max_sectors),
        target_assets_per_sector=int(args.layered_target_assets_per_sector),
        sector_score_power=float(args.layered_sector_score_power),
        min_assets_per_sector=int(args.layered_min_assets_per_sector),
    )
    auto_aggr_cfg = AutoAggressiveConfig(
        enabled=bool(int(args.auto_aggr_enabled)),
        multiplier=float(args.auto_aggr_multiplier),
        topk_multiplier=float(args.auto_aggr_topk_multiplier),
        score_quantile=float(args.auto_aggr_score_quantile),
        corr_quantile=float(args.auto_aggr_corr_quantile),
        vol_quantile=float(args.auto_aggr_vol_quantile),
        min_history_months=int(args.auto_aggr_min_history_months),
        require_positive_alpha=bool(int(args.auto_aggr_require_positive_alpha)),
        alpha_lookback_months=int(args.auto_aggr_alpha_lookback_months),
        confirm_months=int(args.auto_aggr_confirm_months),
    )
    rebalance_cfg = RebalanceControlConfig(
        enabled=bool(int(args.rebalance_control_enabled)),
        deadband_l1=float(args.rebalance_deadband_l1),
        force_l1=float(args.rebalance_force_l1),
        cooldown_months=int(args.rebalance_cooldown_months),
    )
    liquidity_map = _load_liquidity_map(hybrid_cfg.liquidity_csv)
    allowed_assets = _load_allowed_assets(str(args.execution_universe_csv).strip())
    snap_by_month, state = _build_snapshots(
        impact_csv,
        max_assets_per_month=int(args.max_assets_per_month),
        allowed_assets=allowed_assets or None,
    )
    months = sorted([m for m in snap_by_month.keys() if m in mret.index and m in eqw.index])
    months = [m for m in months if m >= str(args.start_ym)]
    min_months = max(1, int(args.min_months))
    if len(months) < min_months:
        raise RuntimeError(
            f"not enough months in common between snapshots and returns "
            f"(found={len(months)}, required={min_months})"
        )

    mom_cache: dict[int, pd.DataFrame] = {}
    all_mom_lookbacks = set(_parse_int_list(args.mom_lookback_options))
    all_mom_lookbacks.add(int(max(0, hybrid_cfg.lookback_months)))
    all_mom_lookbacks.add(int(max(0, hedge_cfg.mom_lookback)))
    all_vol_lookbacks = {int(max(0, hybrid_cfg.volatility_lookback_months))}
    for lb in sorted(all_mom_lookbacks):
        if int(lb) <= 0:
            mom_cache[int(lb)] = pd.DataFrame(index=mret.index, columns=mret.columns, dtype=float)
            continue
        mom_cache[int(lb)] = ((1.0 + mret).rolling(int(lb), min_periods=int(lb)).apply(np.prod, raw=True) - 1.0).shift(1)
    volatility_cache: dict[int, pd.DataFrame] = {}
    for lb in sorted(all_vol_lookbacks):
        if int(lb) <= 0:
            volatility_cache[int(lb)] = pd.DataFrame(index=mret.index, columns=mret.columns, dtype=float)
            continue
        min_periods = int(max(2, lb))
        volatility_cache[int(lb)] = mret.rolling(int(lb), min_periods=min_periods).std(ddof=0).shift(1)
    persistence_by_month = _build_persistence_maps(
        snap_by_month=snap_by_month,
        months=months,
        lookback_months=int(max(1, hybrid_cfg.persistence_lookback_months)),
    )
    market_mom = pd.Series(index=mret.index, dtype=float)
    if int(hedge_cfg.mom_lookback) > 0:
        lb = int(hedge_cfg.mom_lookback)
        market_mom = ((1.0 + market).rolling(lb, min_periods=lb).apply(np.prod, raw=True) - 1.0).shift(1)

    top_k_options = _parse_int_list(args.top_k_options)
    impact_power_options = _parse_float_list(args.impact_power_options)
    wmax_options = _parse_float_list(args.wmax_options)
    mom_lookback_options = _parse_int_list(args.mom_lookback_options)
    mom_threshold_options = _parse_float_list(args.mom_threshold_options)
    modes = [x.strip().lower() for x in str(args.modes).split(",") if x.strip()]
    risk_profiles = [
        (0.20, 0.45, 0.80, 1.05),
        (0.25, 0.55, 0.90, 1.15),
        (0.30, 0.65, 1.00, 1.25),
    ]
    rb_stress_options = _parse_float_list(args.rb_stress_options) if str(args.rb_stress_options).strip() else []
    rb_transition_options = _parse_float_list(args.rb_transition_options) if str(args.rb_transition_options).strip() else []
    rb_stable_options = _parse_float_list(args.rb_stable_options) if str(args.rb_stable_options).strip() else []
    rb_dispersion_options = _parse_float_list(args.rb_dispersion_options) if str(args.rb_dispersion_options).strip() else []
    custom_risk_profiles = (
        list(itertools.product(rb_stress_options, rb_transition_options, rb_stable_options, rb_dispersion_options))
        if rb_stress_options and rb_transition_options and rb_stable_options and rb_dispersion_options
        else []
    )

    train_end_ym = pd.Timestamp(str(args.train_end)).to_period("M").strftime("%Y-%m")
    grid_rows: list[dict[str, Any]] = []
    best_params: StrategyParams | None = None
    best_score = float("-inf")
    best_monthly = pd.DataFrame()
    best_metrics: dict[str, float] = {}
    grid_candidates: list[StrategyParams] = []
    for mode in modes:
        if mode == "const":
            stable_values = rb_stable_options if rb_stable_options else [0.8, 0.95, 1.1]
            profile_iter = [(x, x, x, x) for x in stable_values]
        else:
            profile_iter = custom_risk_profiles if custom_risk_profiles else risk_profiles
        for top_k, impact_power, w_max, mom_lb, mom_thr, (rb_s, rb_t, rb_st, rb_d) in itertools.product(
            top_k_options,
            impact_power_options,
            wmax_options,
            mom_lookback_options,
            mom_threshold_options,
            profile_iter,
        ):
            grid_candidates.append(
                StrategyParams(
                    mode=mode,
                    top_k=int(top_k),
                    impact_power=float(impact_power),
                    w_max=float(w_max),
                    mom_lookback=int(mom_lb),
                    mom_threshold=float(mom_thr),
                    rb_stress=float(rb_s),
                    rb_transition=float(rb_t),
                    rb_stable=float(rb_st),
                    rb_dispersion=float(rb_d),
                )
            )

    grid_candidates_total = int(len(grid_candidates))
    grid_candidates = _thin_grid_items(grid_candidates, int(args.max_grid_combos))
    grid_candidates_evaluated = int(len(grid_candidates))

    for p in grid_candidates:
        monthly = _evaluate(
            params=p,
            months=months,
            snap_by_month=snap_by_month,
            state=state,
            mret=mret,
            eqw=eqw,
            market=market,
            mom_cache=mom_cache,
            corr_signal=corr_signal,
            vol_signal=vol_signal,
            weekly_stress=weekly_stress_signal,
            defense_cfg=defense_cfg,
            decel_cfg=decel_cfg,
            attack_cfg=attack_cfg,
            dd_guard_cfg=dd_guard_cfg,
            hedge_cfg=hedge_cfg,
            regime_topk_cfg=regime_topk_cfg,
            weekly_stress_cfg=weekly_stress_cfg,
            tail_adapt_cfg=tail_adapt_cfg,
            continuous_risk_cfg=continuous_risk_cfg,
            hybrid_cfg=hybrid_cfg,
            basket_cfg=basket_cfg,
            layered_cfg=layered_cfg,
            auto_aggr_cfg=auto_aggr_cfg,
            rebalance_cfg=rebalance_cfg,
            liquidity_map=liquidity_map,
            persistence_by_month=persistence_by_month,
            volatility_cache=volatility_cache,
            market_mom=market_mom,
            rb_cap=float(args.rb_cap),
        )
        if monthly.empty:
            continue
        metrics = _split_metrics(monthly, train_end_ym=train_end_ym, opt_cost_bps=float(args.opt_cost_bps))
        score = _score_candidate(
            metrics,
            objective_mode=str(args.objective_mode),
            use_net_ann=bool(int(args.opt_use_net_ann)),
            turnover_penalty=float(args.opt_turnover_penalty),
        )
        row = {
            "mode": p.mode,
            "top_k": p.top_k,
            "impact_power": p.impact_power,
            "w_max": p.w_max,
            "mom_lookback": p.mom_lookback,
            "mom_threshold": p.mom_threshold,
            "rb_stress": p.rb_stress,
            "rb_transition": p.rb_transition,
            "rb_stable": p.rb_stable,
            "rb_dispersion": p.rb_dispersion,
            "score": score,
        } | metrics
        grid_rows.append(row)
        if score > best_score:
            best_score = float(score)
            best_params = p
            best_monthly = monthly.copy()
            best_metrics = metrics.copy()

    if best_params is None or best_monthly.empty:
        raise RuntimeError("no valid strategy found")

    grid_df = pd.DataFrame(grid_rows).sort_values("score", ascending=False).reset_index(drop=True)
    grid_path = outdir / "grid_results.csv"
    grid_df.to_csv(grid_path, index=False)

    monthly_path = outdir / "monthly_systematic_eval.csv"
    best_monthly.to_csv(monthly_path, index=False)

    latest_alloc_rows: list[dict[str, Any]] = []
    if not best_monthly.empty and "executed_weights_json" in best_monthly.columns:
        latest_row = best_monthly.iloc[-1]
        try:
            latest_weights = json.loads(str(latest_row.get("executed_weights_json", "{}")))
        except json.JSONDecodeError:
            latest_weights = {}
        if isinstance(latest_weights, dict):
            for asset_id, weight in sorted(latest_weights.items()):
                latest_alloc_rows.append(
                    {
                        "ym": str(latest_row.get("ym", "")),
                        "asset_id": str(asset_id),
                        "weight": float(weight),
                        "cash_weight": float(pd.to_numeric(pd.Series([latest_row.get("cash_weight", np.nan)]), errors="coerce").fillna(0.0).iloc[0]),
                        "hedge_weight": float(pd.to_numeric(pd.Series([latest_row.get("hedge_weight", np.nan)]), errors="coerce").fillna(0.0).iloc[0]),
                        "risk_bucket": str(latest_row.get("risk_bucket", "")),
                    }
                )
    latest_alloc_path = outdir / "latest_allocation_weights.csv"
    pd.DataFrame(latest_alloc_rows).to_csv(latest_alloc_path, index=False)

    yearly_df = _yearly_eval(best_monthly)
    yearly_path = outdir / "yearly_systematic_eval.csv"
    yearly_df.to_csv(yearly_path, index=False)

    # Rebuild canonical summary from monthly/yearly files.
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "ops" / "rebuild_systematic_summary.py"), "--yearly-dir", str(outdir)],
        check=True,
        cwd=ROOT,
    )
    summary_path = outdir / "systematic_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    sim_summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "impact_dir": str(impact_dir),
        "returns_csv": str(returns_csv),
        "train_end": str(args.train_end),
        "start_ym": str(args.start_ym),
        "months_evaluated": int(best_monthly.shape[0]),
        "best_params": {
            "mode": best_params.mode,
            "top_k": best_params.top_k,
            "impact_power": best_params.impact_power,
            "w_max": best_params.w_max,
            "mom_lookback": best_params.mom_lookback,
            "mom_threshold": best_params.mom_threshold,
            "rb_stress": best_params.rb_stress,
            "rb_transition": best_params.rb_transition,
            "rb_stable": best_params.rb_stable,
            "rb_dispersion": best_params.rb_dispersion,
        },
        "best_metrics": best_metrics,
        "defense_config": {
            "enabled": defense_cfg.enabled,
            "multiplier": defense_cfg.multiplier,
            "corr_quantile": defense_cfg.corr_quantile,
            "vol_quantile": defense_cfg.vol_quantile,
            "min_history_months": defense_cfg.min_history_months,
            "require_both": defense_cfg.require_both,
        },
        "deceleration_config": {
            "enabled": decel_cfg.enabled,
            "lookback_months": decel_cfg.lookback_months,
            "alpha_threshold": decel_cfg.alpha_threshold,
            "min_streak": decel_cfg.min_streak,
            "multiplier": decel_cfg.multiplier,
            "topk_multiplier": decel_cfg.topk_multiplier,
        },
        "hedge_config": {
            "enabled": hedge_cfg.enabled,
            "multiplier": hedge_cfg.multiplier,
            "mom_lookback": hedge_cfg.mom_lookback,
            "mom_threshold": hedge_cfg.mom_threshold,
            "activate_on_defense": hedge_cfg.activate_on_defense,
        },
        "regime_topk_config": {
            "enabled": regime_topk_cfg.enabled,
            "stable_multiplier": regime_topk_cfg.stable_multiplier,
            "transition_multiplier": regime_topk_cfg.transition_multiplier,
            "stress_multiplier": regime_topk_cfg.stress_multiplier,
        },
        "weekly_stress_config": {
            "enabled": weekly_stress_cfg.enabled,
            "weekly_quantile": weekly_stress_cfg.weekly_quantile,
            "min_history_months": weekly_stress_cfg.min_history_months,
            "multiplier": weekly_stress_cfg.multiplier,
        },
        "tail_adapt_config": {
            "enabled": tail_adapt_cfg.enabled,
            "lookback_months": tail_adapt_cfg.lookback_months,
            "alpha_threshold": tail_adapt_cfg.alpha_threshold,
            "down_multiplier": tail_adapt_cfg.down_multiplier,
            "up_multiplier": tail_adapt_cfg.up_multiplier,
            "start_ym": tail_adapt_cfg.start_ym,
        },
        "hybrid_ranking_config": {
            "enabled": hybrid_cfg.enabled,
            "lookback_months": hybrid_cfg.lookback_months,
            "weight_impact": hybrid_cfg.weight_impact,
            "weight_momentum": hybrid_cfg.weight_momentum,
            "weight_liquidity": hybrid_cfg.weight_liquidity,
            "weight_persistence": hybrid_cfg.weight_persistence,
            "weight_sector_strength": hybrid_cfg.weight_sector_strength,
            "weight_low_corr": hybrid_cfg.weight_low_corr,
            "weight_low_vol": hybrid_cfg.weight_low_vol,
            "weight_low_concentration": hybrid_cfg.weight_low_concentration,
            "liquidity_csv": hybrid_cfg.liquidity_csv,
            "persistence_lookback_months": hybrid_cfg.persistence_lookback_months,
            "volatility_lookback_months": hybrid_cfg.volatility_lookback_months,
            "liquidity_assets_loaded": int(len(liquidity_map)),
        },
        "continuous_risk_config": {
            "enabled": continuous_risk_cfg.enabled,
            "weight_global_score": continuous_risk_cfg.weight_global_score,
            "weight_corr": continuous_risk_cfg.weight_corr,
            "weight_vol": continuous_risk_cfg.weight_vol,
            "regime_bias": continuous_risk_cfg.regime_bias,
            "score_power": continuous_risk_cfg.score_power,
        },
        "regime_basket_config": {
            "enabled": basket_cfg.enabled,
            "sector_bonus": basket_cfg.sector_bonus,
            "global_sleeve_bonus": basket_cfg.global_sleeve_bonus,
            "vote_threshold": basket_cfg.vote_threshold,
        },
        "layered_rotation_config": {
            "enabled": layered_cfg.enabled,
            "min_sectors": layered_cfg.min_sectors,
            "max_sectors": layered_cfg.max_sectors,
            "target_assets_per_sector": layered_cfg.target_assets_per_sector,
            "sector_score_power": layered_cfg.sector_score_power,
            "min_assets_per_sector": layered_cfg.min_assets_per_sector,
        },
        "auto_aggressive_config": {
            "enabled": auto_aggr_cfg.enabled,
            "multiplier": auto_aggr_cfg.multiplier,
            "topk_multiplier": auto_aggr_cfg.topk_multiplier,
            "score_quantile": auto_aggr_cfg.score_quantile,
            "corr_quantile": auto_aggr_cfg.corr_quantile,
            "vol_quantile": auto_aggr_cfg.vol_quantile,
            "min_history_months": auto_aggr_cfg.min_history_months,
            "require_positive_alpha": auto_aggr_cfg.require_positive_alpha,
            "alpha_lookback_months": auto_aggr_cfg.alpha_lookback_months,
            "confirm_months": auto_aggr_cfg.confirm_months,
        },
        "rebalance_control_config": {
            "enabled": rebalance_cfg.enabled,
            "deadband_l1": rebalance_cfg.deadband_l1,
            "force_l1": rebalance_cfg.force_l1,
            "cooldown_months": rebalance_cfg.cooldown_months,
        },
        "attack_config": {
            "enabled": attack_cfg.enabled,
            "multiplier": attack_cfg.multiplier,
            "corr_quantile": attack_cfg.corr_quantile,
            "vol_quantile": attack_cfg.vol_quantile,
            "min_history_months": attack_cfg.min_history_months,
            "require_both": attack_cfg.require_both,
            "require_positive_alpha": attack_cfg.require_positive_alpha,
            "alpha_lookback_months": attack_cfg.alpha_lookback_months,
        },
        "drawdown_guard_config": {
            "enabled": dd_guard_cfg.enabled,
            "soft_threshold": dd_guard_cfg.soft_threshold,
            "hard_threshold": dd_guard_cfg.hard_threshold,
            "soft_multiplier": dd_guard_cfg.soft_multiplier,
            "hard_multiplier": dd_guard_cfg.hard_multiplier,
        },
        "optimization_config": {
            "opt_cost_bps": float(args.opt_cost_bps),
            "opt_turnover_penalty": float(args.opt_turnover_penalty),
            "opt_use_net_ann": bool(int(args.opt_use_net_ann)),
            "max_grid_combos": int(args.max_grid_combos),
            "grid_candidates_total": int(grid_candidates_total),
            "grid_candidates_evaluated": int(grid_candidates_evaluated),
        },
        "promotion_candidate_flags": {
            "worth_rate_ge_055": bool(float(summary.get("worth_it_rate_vs_eqw", np.nan)) >= 0.55),
            "prob_positive_ge_055": bool(float(summary.get("monthly_alpha_prob_positive_vs_eqw", np.nan)) >= 0.55),
            "max_drawdown_ge_floor": bool(float(summary.get("strategy_max_drop", np.nan)) >= -0.35),
            "recent_alpha6_ge_floor": bool(float(best_metrics.get("full_alpha_recent6", np.nan)) >= -0.003),
        },
    }
    sim_summary_path = outdir / "simulation_summary.json"
    sim_summary_path.write_text(json.dumps(sim_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    shadow_csv_path: Path | None = None
    shadow_json_path: Path | None = None
    shadow_tail = int(max(0, args.shadow_tail_months))
    if shadow_tail > 0 and not best_monthly.empty:
        keep_cols = [
            "ym",
            "risk_bucket",
            "risk_budget",
            "risk_budget_trade",
            "n_selected",
            "effective_top_k",
            "selected_assets",
            "turnover",
            "rebalance_executed",
            "rebalance_reason",
            "ret",
            "eqw_ret",
            "auto_aggressive_active",
            "auto_aggressive_streak",
            "defense_active",
            "weekly_stress_active",
            "dd_guard_active",
        ]
        keep_cols = [c for c in keep_cols if c in best_monthly.columns]
        shadow_df = best_monthly.tail(shadow_tail)[keep_cols].copy()
        shadow_csv_path = outdir / "shadow_mode_signals.csv"
        shadow_df.to_csv(shadow_csv_path, index=False)
        latest = shadow_df.tail(1).to_dict(orient="records")[0] if not shadow_df.empty else {}
        shadow_payload = {
            "status": "ok",
            "mode": "paper_trading_shadow",
            "tail_months": int(shadow_tail),
            "latest_signal": latest,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        shadow_json_path = outdir / "shadow_mode_signals_latest.json"
        shadow_json_path.write_text(json.dumps(shadow_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    write_run_manifest(
        outdir,
        script="scripts/ops/run_canonical_systematic_eval.py",
        params={
            "impact_dir": str(impact_dir),
            "returns_csv": str(returns_csv),
            "train_end": str(args.train_end),
            "start_ym": str(args.start_ym),
            "max_assets_per_month": int(args.max_assets_per_month),
            "top_k_options": top_k_options,
            "impact_power_options": impact_power_options,
            "wmax_options": wmax_options,
            "mom_lookback_options": mom_lookback_options,
            "mom_threshold_options": mom_threshold_options,
            "modes": modes,
            "rb_stress_options": rb_stress_options,
            "rb_transition_options": rb_transition_options,
            "rb_stable_options": rb_stable_options,
            "rb_dispersion_options": rb_dispersion_options,
            "benchmark_symbol": str(args.benchmark_symbol),
            "defense_enabled": int(args.defense_enabled),
            "defense_multiplier": float(args.defense_multiplier),
            "defense_corr_quantile": float(args.defense_corr_quantile),
            "defense_vol_quantile": float(args.defense_vol_quantile),
            "defense_min_history_months": int(args.defense_min_history_months),
            "defense_require_both": int(args.defense_require_both),
            "decel_enabled": int(args.decel_enabled),
            "decel_lookback_months": int(args.decel_lookback_months),
            "decel_alpha_threshold": float(args.decel_alpha_threshold),
            "decel_min_streak": int(args.decel_min_streak),
            "decel_multiplier": float(args.decel_multiplier),
            "decel_topk_multiplier": float(args.decel_topk_multiplier),
            "hedge_enabled": int(args.hedge_enabled),
            "hedge_multiplier": float(args.hedge_multiplier),
            "hedge_mom_lookback": int(args.hedge_mom_lookback),
            "hedge_mom_threshold": float(args.hedge_mom_threshold),
            "hedge_activate_on_defense": int(args.hedge_activate_on_defense),
            "regime_topk_enabled": int(args.regime_topk_enabled),
            "regime_topk_stable_multiplier": float(args.regime_topk_stable_multiplier),
            "regime_topk_transition_multiplier": float(args.regime_topk_transition_multiplier),
            "regime_topk_stress_multiplier": float(args.regime_topk_stress_multiplier),
            "weekly_stress_enabled": int(args.weekly_stress_enabled),
            "weekly_stress_quantile": float(args.weekly_stress_quantile),
            "weekly_stress_min_history_months": int(args.weekly_stress_min_history_months),
            "weekly_stress_multiplier": float(args.weekly_stress_multiplier),
            "tail_adapt_enabled": int(args.tail_adapt_enabled),
            "tail_adapt_lookback_months": int(args.tail_adapt_lookback_months),
            "tail_adapt_alpha_threshold": float(args.tail_adapt_alpha_threshold),
            "tail_adapt_down_multiplier": float(args.tail_adapt_down_multiplier),
            "tail_adapt_up_multiplier": float(args.tail_adapt_up_multiplier),
            "tail_adapt_start_ym": str(args.tail_adapt_start_ym),
            "hybrid_enabled": int(args.hybrid_enabled),
            "hybrid_lookback_months": int(args.hybrid_lookback_months),
            "hybrid_weight_impact": float(args.hybrid_weight_impact),
            "hybrid_weight_momentum": float(args.hybrid_weight_momentum),
            "hybrid_weight_liquidity": float(args.hybrid_weight_liquidity),
            "hybrid_weight_persistence": float(args.hybrid_weight_persistence),
            "hybrid_weight_sector_strength": float(args.hybrid_weight_sector_strength),
            "hybrid_weight_low_corr": float(args.hybrid_weight_low_corr),
            "hybrid_weight_low_vol": float(args.hybrid_weight_low_vol),
            "hybrid_weight_low_concentration": float(args.hybrid_weight_low_concentration),
            "hybrid_liquidity_csv": str(args.hybrid_liquidity_csv),
            "hybrid_persistence_lookback_months": int(args.hybrid_persistence_lookback_months),
            "hybrid_volatility_lookback_months": int(args.hybrid_volatility_lookback_months),
            "execution_universe_csv": str(args.execution_universe_csv),
            "continuous_risk_enabled": int(args.continuous_risk_enabled),
            "continuous_risk_weight_global_score": float(args.continuous_risk_weight_global_score),
            "continuous_risk_weight_corr": float(args.continuous_risk_weight_corr),
            "continuous_risk_weight_vol": float(args.continuous_risk_weight_vol),
            "continuous_risk_regime_bias": float(args.continuous_risk_regime_bias),
            "continuous_risk_score_power": float(args.continuous_risk_score_power),
            "regime_basket_enabled": int(args.regime_basket_enabled),
            "regime_basket_sector_bonus": float(args.regime_basket_sector_bonus),
            "regime_basket_global_sleeve_bonus": float(args.regime_basket_global_sleeve_bonus),
            "regime_basket_vote_threshold": float(args.regime_basket_vote_threshold),
            "layered_enabled": int(args.layered_enabled),
            "layered_min_sectors": int(args.layered_min_sectors),
            "layered_max_sectors": int(args.layered_max_sectors),
            "layered_target_assets_per_sector": int(args.layered_target_assets_per_sector),
            "layered_sector_score_power": float(args.layered_sector_score_power),
            "layered_min_assets_per_sector": int(args.layered_min_assets_per_sector),
            "auto_aggr_enabled": int(args.auto_aggr_enabled),
            "auto_aggr_multiplier": float(args.auto_aggr_multiplier),
            "auto_aggr_topk_multiplier": float(args.auto_aggr_topk_multiplier),
            "auto_aggr_score_quantile": float(args.auto_aggr_score_quantile),
            "auto_aggr_corr_quantile": float(args.auto_aggr_corr_quantile),
            "auto_aggr_vol_quantile": float(args.auto_aggr_vol_quantile),
            "auto_aggr_min_history_months": int(args.auto_aggr_min_history_months),
            "auto_aggr_require_positive_alpha": int(args.auto_aggr_require_positive_alpha),
            "auto_aggr_alpha_lookback_months": int(args.auto_aggr_alpha_lookback_months),
            "auto_aggr_confirm_months": int(args.auto_aggr_confirm_months),
            "rebalance_control_enabled": int(args.rebalance_control_enabled),
            "rebalance_deadband_l1": float(args.rebalance_deadband_l1),
            "rebalance_force_l1": float(args.rebalance_force_l1),
            "rebalance_cooldown_months": int(args.rebalance_cooldown_months),
            "opt_cost_bps": float(args.opt_cost_bps),
            "opt_turnover_penalty": float(args.opt_turnover_penalty),
            "opt_use_net_ann": int(args.opt_use_net_ann),
            "shadow_tail_months": int(args.shadow_tail_months),
            "attack_enabled": int(args.attack_enabled),
            "attack_multiplier": float(args.attack_multiplier),
            "attack_corr_quantile": float(args.attack_corr_quantile),
            "attack_vol_quantile": float(args.attack_vol_quantile),
            "attack_min_history_months": int(args.attack_min_history_months),
            "attack_require_both": int(args.attack_require_both),
            "attack_require_positive_alpha": int(args.attack_require_positive_alpha),
            "attack_alpha_lookback_months": int(args.attack_alpha_lookback_months),
            "dd_guard_enabled": int(args.dd_guard_enabled),
            "dd_soft_threshold": float(args.dd_soft_threshold),
            "dd_hard_threshold": float(args.dd_hard_threshold),
            "dd_soft_multiplier": float(args.dd_soft_multiplier),
            "dd_hard_multiplier": float(args.dd_hard_multiplier),
            "objective_mode": str(args.objective_mode),
            "max_grid_combos": int(args.max_grid_combos),
            "rb_cap": float(args.rb_cap),
        },
        paths={
            "grid_results_csv": str(grid_path),
            "monthly_eval_csv": str(monthly_path),
            "yearly_eval_csv": str(yearly_path),
            "summary_json": str(summary_path),
            "simulation_summary_json": str(sim_summary_path),
            "latest_allocation_weights_csv": str(latest_alloc_path),
            "shadow_mode_signals_csv": str(shadow_csv_path) if shadow_csv_path is not None else "",
            "shadow_mode_signals_latest_json": str(shadow_json_path) if shadow_json_path is not None else "",
        },
        gates={
            "promotion_worth_rate_ge_055": bool(float(summary.get("worth_it_rate_vs_eqw", np.nan)) >= 0.55),
            "promotion_prob_positive_ge_055": bool(float(summary.get("monthly_alpha_prob_positive_vs_eqw", np.nan)) >= 0.55),
            "monitoring_max_drawdown_floor": bool(float(summary.get("strategy_max_drop", np.nan)) >= -0.35),
            "monitoring_recent_alpha6_floor": bool(float(best_metrics.get("full_alpha_recent6", np.nan)) >= -0.003),
        },
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "outdir": str(outdir),
                "best_params": sim_summary["best_params"],
                "worth_it_rate_vs_eqw": summary.get("worth_it_rate_vs_eqw"),
                "monthly_alpha_prob_positive_vs_eqw": summary.get("monthly_alpha_prob_positive_vs_eqw"),
                "strategy_max_drop": summary.get("strategy_max_drop"),
                "full_alpha_recent6": best_metrics.get("full_alpha_recent6"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
