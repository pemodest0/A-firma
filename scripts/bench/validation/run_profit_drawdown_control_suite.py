#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.structural.run_manifest import write_run_manifest  # noqa: E402
from execution.cost_model import summarize_return_series  # noqa: E402
from execution.net_assumptions import NetAssumptionProfile, load_net_assumption_profiles  # noqa: E402
from scripts.bench.validation.run_profit_frontier_expansion_suite import (  # noqa: E402
    EQUITY_EXCLUDED,
    StrategyResult,
    _build_meta_switch,
    _ensure_benchmark_columns,
    _evaluate_net,
    _load_asset_table,
    _load_daily_universe,
    _result_row,
    _run_id,
    _safe_float,
    _select_crypto_tiers,
    _simulate_asset_rule,
)
from scripts.bench.validation.run_profit_layered_engine_suite import (  # noqa: E402
    StrategyBundle,
    _apply_breadth_overlay_to_bundle,
    _build_breadth_signal,
    _build_meta_switch_v2,
    _build_meta_switch_v3,
    _equity_v2_group_scores,
    _meta_blended_profile,
    _profile_scaled,
    _research_rows,
    _simulate_equity_group_sleeve_v2,
    _simulate_equity_trail_switch_bundle,
    _simulate_equity_group_sleeve_v3,
    _stress_bundle,
    _walkforward_rows,
)
from scripts.bench.validation.run_profit_shadow_discovery_measures import _drawdown_duration_stats  # noqa: E402


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _latest_json(root: Path, pattern: str) -> dict[str, Any]:
    matches = sorted(root.glob(pattern))
    if not matches:
        return {}
    payload = json.loads(matches[-1].read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _latest_path(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern))
    return matches[-1] if matches else None


def _apply_scale_overlay(
    *,
    candidate_id: str,
    base_bundle: StrategyBundle,
    scale: pd.Series,
    suite: str,
    family: str,
    notes: str,
) -> StrategyBundle:
    idx = base_bundle.result.gross_ret.index.intersection(scale.index)
    gross = pd.to_numeric(base_bundle.result.gross_ret.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    turnover = pd.to_numeric(base_bundle.result.turnover.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    scale = pd.to_numeric(scale.reindex(idx), errors="coerce").fillna(0.0).clip(0.0, 1.0).astype(float)
    gross = gross * scale
    turnover = turnover * scale + scale.diff().abs().fillna(scale.abs()) * 0.5
    perf = _evaluate_net(
        gross_ret=gross,
        turnover=turnover,
        profile=base_bundle.profile,
        benchmark_ret=base_bundle.benchmark_gross_ret.reindex(idx).fillna(0.0).astype(float),
        benchmark_profile=base_bundle.benchmark_profile,
    )
    result = StrategyResult(
        suite=suite,
        candidate_id=candidate_id,
        family=family,
        benchmark_ticker=base_bundle.result.benchmark_ticker,
        gross_ret=gross,
        turnover=turnover,
        net_ret=perf["net_ret"],
        benchmark_net_ret=perf["benchmark_net_ret"],
        net_ann_return=_safe_float(perf.get("net_ann_return")),
        net_total_return=_safe_float(perf.get("net_total_return")),
        net_sharpe=_safe_float(perf.get("net_sharpe")),
        net_max_drawdown=_safe_float(perf.get("net_max_drawdown")),
        edge_vs_benchmark=_safe_float(perf.get("edge_vs_benchmark")),
        avg_turnover_daily=_safe_float(perf.get("avg_turnover_daily")),
        hit_rate_10x_5y=float("nan"),
        years_to_10x_full=float("nan"),
        notes=notes,
    )
    return StrategyBundle(
        result=result,
        benchmark_gross_ret=base_bundle.benchmark_gross_ret.reindex(idx).fillna(0.0).astype(float),
        profile=base_bundle.profile,
        benchmark_profile=base_bundle.benchmark_profile,
    )


def _build_meta_context(
    *,
    base_bundle: StrategyBundle,
    crypto_bundle: StrategyBundle,
    equity_bundle: StrategyBundle,
    btc_prices: pd.Series,
    spy_prices: pd.Series,
    crypto_breadth: pd.Series,
) -> pd.DataFrame:
    idx = (
        base_bundle.result.gross_ret.index
        .intersection(crypto_bundle.result.gross_ret.index)
        .intersection(equity_bundle.result.gross_ret.index)
        .intersection(btc_prices.index)
        .intersection(spy_prices.index)
        .intersection(crypto_breadth.index)
    )
    crypto_ret = pd.to_numeric(crypto_bundle.result.gross_ret.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    equity_ret = pd.to_numeric(equity_bundle.result.gross_ret.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    btc_close = pd.to_numeric(btc_prices.reindex(idx), errors="coerce").astype(float)
    spy_close = pd.to_numeric(spy_prices.reindex(idx), errors="coerce").astype(float)
    breadth = pd.to_numeric(crypto_breadth.reindex(idx), errors="coerce").fillna(0.0).clip(0.0, 1.0).astype(float)
    btc_ok = (btc_close.shift(1) > btc_close.shift(1).rolling(200, min_periods=100).mean()).fillna(False)
    spy_ok = (spy_close.shift(1) > spy_close.shift(1).rolling(200, min_periods=100).mean()).fillna(False)
    crypto_trail21 = (1.0 + crypto_ret).rolling(21, min_periods=10).apply(np.prod, raw=True) - 1.0
    equity_trail21 = (1.0 + equity_ret).rolling(21, min_periods=10).apply(np.prod, raw=True) - 1.0
    crypto_trail = (1.0 + crypto_ret).rolling(63, min_periods=21).apply(np.prod, raw=True) - 1.0
    equity_trail = (1.0 + equity_ret).rolling(63, min_periods=21).apply(np.prod, raw=True) - 1.0
    crypto_eq = (1.0 + crypto_ret).clip(lower=1e-9, upper=10.0).cumprod()
    crypto_dd = (crypto_eq / crypto_eq.cummax() - 1.0).shift(1).fillna(0.0).astype(float)
    source = pd.Series("cash", index=idx, dtype=object)
    prefer_crypto = btc_ok & (pd.to_numeric(crypto_trail, errors="coerce") > pd.to_numeric(equity_trail, errors="coerce"))
    source = source.mask(spy_ok | btc_ok, "equity")
    source = source.mask(prefer_crypto, "crypto")
    return pd.DataFrame(
        {
            "source": source.astype(str),
            "btc_ok": btc_ok.astype(bool),
            "spy_ok": spy_ok.astype(bool),
            "crypto_ret": crypto_ret,
            "equity_ret": equity_ret,
            "crypto_trail21": pd.to_numeric(crypto_trail21, errors="coerce").fillna(0.0).astype(float),
            "equity_trail21": pd.to_numeric(equity_trail21, errors="coerce").fillna(0.0).astype(float),
            "crypto_trail63": pd.to_numeric(crypto_trail, errors="coerce").fillna(0.0).astype(float),
            "equity_trail63": pd.to_numeric(equity_trail, errors="coerce").fillna(0.0).astype(float),
            "crypto_drawdown_prev": crypto_dd,
            "crypto_breadth": breadth,
        },
        index=idx,
    )


def _build_conviction_scale(
    context: pd.DataFrame,
    *,
    min_active: float,
    max_active: float,
) -> pd.Series:
    idx = context.index
    out = pd.Series(np.zeros(len(idx), dtype=float), index=idx, dtype=float)
    lo = float(min_active)
    hi = float(max_active)
    for dt, row in context.iterrows():
        source = str(row.get("source", "cash"))
        if source == "cash":
            out.loc[dt] = 0.0
            continue
        c_trail = _safe_float(row.get("crypto_trail63"), 0.0)
        e_trail = _safe_float(row.get("equity_trail63"), 0.0)
        breadth = _safe_float(row.get("crypto_breadth"), 0.0)
        if source == "crypto":
            gap = np.clip((c_trail - e_trail) / 0.35, 0.0, 1.0)
            bscore = np.clip((breadth - 0.45) / 0.35, 0.0, 1.0)
            score = 0.45 + 0.35 * gap + 0.20 * bscore if bool(row.get("btc_ok", False)) else 0.0
        else:
            eq_score = np.clip(e_trail / 0.20, 0.0, 1.0)
            crypto_dom = np.clip((c_trail - e_trail) / 0.35, 0.0, 1.0)
            score = 0.55 * float(bool(row.get("spy_ok", False))) + 0.35 * eq_score + 0.10 * (1.0 - crypto_dom)
        out.loc[dt] = float(np.clip(lo + (hi - lo) * np.clip(score, 0.0, 1.0), 0.0, 1.0))
    return out.astype(float)


def _build_vol_target_scale(
    gross_ret: pd.Series,
    *,
    target_ann_vol: float,
    window: int,
    min_scale: float,
    max_scale: float,
) -> pd.Series:
    x = pd.to_numeric(gross_ret, errors="coerce").fillna(0.0).astype(float)
    vol = x.shift(1).rolling(int(window), min_periods=max(20, int(window) // 2)).std(ddof=0) * np.sqrt(252.0)
    scale = (float(target_ann_vol) / vol.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    scale = scale.fillna(float(max_scale)).clip(float(min_scale), float(max_scale))
    return scale.astype(float)


def _build_drawdown_guard_scale(
    gross_ret: pd.Series,
    *,
    trigger_dd: float,
    release_dd: float,
    reduced_scale: float,
    cooldown_days: int,
) -> pd.Series:
    x = pd.to_numeric(gross_ret, errors="coerce").fillna(0.0).astype(float)
    eq = (1.0 + x).clip(lower=1e-9, upper=10.0).cumprod()
    dd_prev = (eq / eq.cummax() - 1.0).shift(1).fillna(0.0).astype(float)
    out = pd.Series(np.ones(len(dd_prev), dtype=float), index=dd_prev.index, dtype=float)
    active = False
    cool = 0
    trig = -abs(float(trigger_dd))
    rel = -abs(float(release_dd))
    reduced = float(np.clip(reduced_scale, 0.0, 1.0))
    for dt in dd_prev.index:
        dd = float(dd_prev.loc[dt])
        if active:
            if cool > 0:
                cool -= 1
            if dd >= rel and cool <= 0:
                active = False
        if (not active) and dd <= trig:
            active = True
            cool = int(max(0, cooldown_days))
        out.loc[dt] = reduced if active else 1.0
    return out.astype(float)


def _build_source_specific_scale(
    context: pd.DataFrame,
    *,
    crypto_scale: pd.Series | float,
    equity_scale: pd.Series | float = 1.0,
    cash_scale: float = 0.0,
) -> pd.Series:
    idx = context.index
    if isinstance(crypto_scale, pd.Series):
        crypto_s = pd.to_numeric(crypto_scale.reindex(idx), errors="coerce").fillna(0.0).clip(0.0, 1.0).astype(float)
    else:
        crypto_s = pd.Series(float(np.clip(crypto_scale, 0.0, 1.0)), index=idx, dtype=float)
    if isinstance(equity_scale, pd.Series):
        equity_s = pd.to_numeric(equity_scale.reindex(idx), errors="coerce").fillna(0.0).clip(0.0, 1.0).astype(float)
    else:
        equity_s = pd.Series(float(np.clip(equity_scale, 0.0, 1.0)), index=idx, dtype=float)
    out = pd.Series(float(np.clip(cash_scale, 0.0, 1.0)), index=idx, dtype=float)
    source = context["source"].astype(str)
    out.loc[source.eq("equity")] = equity_s.loc[source.eq("equity")]
    out.loc[source.eq("crypto")] = crypto_s.loc[source.eq("crypto")]
    return out.astype(float)


def _build_crypto_guard_scale(
    context: pd.DataFrame,
    *,
    low_breadth: float,
    high_breadth: float,
    min_crypto_scale: float,
    edge_floor: float,
    edge_full: float,
) -> pd.Series:
    idx = context.index
    out = pd.Series(np.ones(len(idx), dtype=float), index=idx, dtype=float)
    lo = float(low_breadth)
    hi = float(max(high_breadth, low_breadth + 1e-6))
    e_floor = float(edge_floor)
    e_full = float(max(edge_full, edge_floor + 1e-6))
    min_scale = float(np.clip(min_crypto_scale, 0.0, 1.0))
    for dt, row in context.iterrows():
        source = str(row.get("source", "cash"))
        if source != "crypto":
            out.loc[dt] = 1.0 if source == "equity" else 0.0
            continue
        if not bool(row.get("btc_ok", False)):
            out.loc[dt] = 0.0
            continue
        breadth = _safe_float(row.get("crypto_breadth"), 0.0)
        edge = _safe_float(row.get("crypto_trail63"), 0.0) - _safe_float(row.get("equity_trail63"), 0.0)
        breadth_score = float(np.clip((breadth - lo) / (hi - lo), 0.0, 1.0))
        edge_score = float(np.clip((edge - e_floor) / (e_full - e_floor), 0.0, 1.0))
        scale = min_scale + (1.0 - min_scale) * (0.55 * breadth_score + 0.45 * edge_score)
        out.loc[dt] = float(np.clip(scale, 0.0, 1.0))
    return out.astype(float)


def _load_structural_regime_series(root: Path) -> pd.Series:
    candidates = sorted(root.glob("results/lab_corr_macro*/**/impact_training_dataset.csv"))
    if not candidates:
        return pd.Series(dtype=object)
    latest = candidates[-1]
    try:
        df = pd.read_csv(latest, usecols=["date", "regime"])
    except Exception:
        return pd.Series(dtype=object)
    if df.empty or "date" not in df.columns or "regime" not in df.columns:
        return pd.Series(dtype=object)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["regime"] = df["regime"].astype(str).str.strip().str.lower()
    df = df.dropna(subset=["date"])
    if df.empty:
        return pd.Series(dtype=object)
    agg = (
        df.groupby("date", sort=True)["regime"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
        .sort_index()
    )
    return agg.astype(str)


def _regime_forward_fill(index: pd.Index, regime_series: pd.Series) -> pd.Series:
    if regime_series.empty:
        return pd.Series("stable", index=index, dtype=object)
    s = regime_series.copy()
    s.index = pd.to_datetime(s.index)
    out = s.reindex(pd.to_datetime(index), method="ffill")
    out = out.fillna("stable").astype(str).str.lower()
    return pd.Series(out.to_numpy(dtype=object), index=index, dtype=object)


def _build_regime_scaled_series(index: pd.Index, regime_series: pd.Series, mapping: dict[str, float], *, default: float = 1.0) -> pd.Series:
    reg = _regime_forward_fill(index, regime_series)
    return reg.map({str(k).lower(): float(v) for k, v in mapping.items()}).fillna(float(default)).astype(float)


def _build_regime_aware_crypto_guard_scale(
    context: pd.DataFrame,
    regime_series: pd.Series,
    *,
    params_by_regime: dict[str, dict[str, float]],
    default_regime: str = "stable",
) -> pd.Series:
    idx = context.index
    regimes = _regime_forward_fill(idx, regime_series)
    out = pd.Series(np.ones(len(idx), dtype=float), index=idx, dtype=float)
    defaults = params_by_regime.get(default_regime, {})
    for dt, row in context.iterrows():
        source = str(row.get("source", "cash"))
        reg = str(regimes.loc[dt]).lower()
        params = {**defaults, **params_by_regime.get(reg, {})}
        if source != "crypto":
            out.loc[dt] = 1.0 if source == "equity" else 0.0
            continue
        if not bool(row.get("btc_ok", False)):
            out.loc[dt] = 0.0
            continue
        low_breadth = float(params.get("low_breadth", 0.45))
        high_breadth = float(max(params.get("high_breadth", 0.72), low_breadth + 1e-6))
        min_crypto_scale = float(np.clip(params.get("min_crypto_scale", 0.30), 0.0, 1.0))
        edge_floor = float(params.get("edge_floor", 0.0))
        edge_full = float(max(params.get("edge_full", 0.18), edge_floor + 1e-6))
        breadth = _safe_float(row.get("crypto_breadth"), 0.0)
        edge = _safe_float(row.get("crypto_trail63"), 0.0) - _safe_float(row.get("equity_trail63"), 0.0)
        breadth_score = float(np.clip((breadth - low_breadth) / (high_breadth - low_breadth), 0.0, 1.0))
        edge_score = float(np.clip((edge - edge_floor) / (edge_full - edge_floor), 0.0, 1.0))
        scale = min_crypto_scale + (1.0 - min_crypto_scale) * (0.55 * breadth_score + 0.45 * edge_score)
        out.loc[dt] = float(np.clip(scale, 0.0, 1.0))
    return out.astype(float)


def _build_meta_early_exit_bundle(
    *,
    candidate_id: str,
    context: pd.DataFrame,
    base_bundle: StrategyBundle,
    exit_breadth: float,
    reentry_breadth: float,
    exit_edge21: float,
    reentry_edge21: float,
    exit_drawdown: float,
    exit_mode: str,
    cooldown_days: int,
) -> StrategyBundle:
    idx = base_bundle.result.gross_ret.index.intersection(context.index)
    ctx = context.reindex(idx).copy()
    gross = pd.Series(np.zeros(len(idx), dtype=float), index=idx, dtype=float)
    turnover = pd.Series(np.zeros(len(idx), dtype=float), index=idx, dtype=float)
    prev_source = "cash"
    cooldown = 0
    for dt, row in ctx.iterrows():
        desired = str(row.get("source", "cash"))
        breadth = _safe_float(row.get("crypto_breadth"), 0.0)
        edge21 = _safe_float(row.get("crypto_trail21"), 0.0) - _safe_float(row.get("equity_trail21"), 0.0)
        crypto_dd = _safe_float(row.get("crypto_drawdown_prev"), 0.0)
        btc_ok = bool(row.get("btc_ok", False))
        spy_ok = bool(row.get("spy_ok", False))
        actual = desired
        if cooldown > 0:
            cooldown -= 1
        if desired == "crypto":
            exit_now = (
                (breadth < float(exit_breadth))
                or (edge21 <= float(exit_edge21))
                or (crypto_dd <= -abs(float(exit_drawdown)))
                or (not btc_ok)
            )
            reentry_ok = (
                btc_ok
                and (breadth >= float(reentry_breadth))
                and (edge21 >= float(reentry_edge21))
                and cooldown <= 0
            )
            if prev_source == "crypto":
                actual = "crypto" if not exit_now else ("equity" if exit_mode == "equity" and spy_ok else "cash")
                if actual != "crypto":
                    cooldown = int(max(0, cooldown_days))
            else:
                actual = "crypto" if reentry_ok else ("equity" if spy_ok else "cash")
        elif desired == "equity":
            actual = "equity" if spy_ok else "cash"
        else:
            actual = "cash"

        if actual != prev_source:
            turnover.loc[dt] = 1.0 if prev_source != "cash" and actual != "cash" else 0.5
        prev_source = actual

        if actual == "crypto":
            gross.loc[dt] = _safe_float(row.get("crypto_ret"), 0.0)
        elif actual == "equity":
            gross.loc[dt] = _safe_float(row.get("equity_ret"), 0.0)
        else:
            gross.loc[dt] = 0.0

    perf = _evaluate_net(
        gross_ret=gross,
        turnover=turnover,
        profile=base_bundle.profile,
        benchmark_ret=base_bundle.benchmark_gross_ret.reindex(idx).fillna(0.0).astype(float),
        benchmark_profile=base_bundle.benchmark_profile,
    )
    result = StrategyResult(
        suite="meta_early_exit",
        candidate_id=candidate_id,
        family="meta_early_exit",
        benchmark_ticker=base_bundle.result.benchmark_ticker,
        gross_ret=gross,
        turnover=turnover,
        net_ret=perf["net_ret"],
        benchmark_net_ret=perf["benchmark_net_ret"],
        net_ann_return=_safe_float(perf.get("net_ann_return")),
        net_total_return=_safe_float(perf.get("net_total_return")),
        net_sharpe=_safe_float(perf.get("net_sharpe")),
        net_max_drawdown=_safe_float(perf.get("net_max_drawdown")),
        edge_vs_benchmark=_safe_float(perf.get("edge_vs_benchmark")),
        avg_turnover_daily=_safe_float(perf.get("avg_turnover_daily")),
        hit_rate_10x_5y=float("nan"),
        years_to_10x_full=float("nan"),
        notes=(
            f"exit_mode={exit_mode};exit_breadth={exit_breadth:.2f};reentry_breadth={reentry_breadth:.2f};"
            f"exit_edge21={exit_edge21:.3f};reentry_edge21={reentry_edge21:.3f};exit_dd={exit_drawdown:.2f};"
            f"cooldown={cooldown_days}"
        ),
    )
    return StrategyBundle(
        result=result,
        benchmark_gross_ret=base_bundle.benchmark_gross_ret.reindex(idx).fillna(0.0).astype(float),
        profile=base_bundle.profile,
        benchmark_profile=base_bundle.benchmark_profile,
    )


def _candidate_diag(bundle: StrategyBundle) -> dict[str, float]:
    ret = pd.to_numeric(bundle.result.net_ret, errors="coerce").fillna(0.0).astype(float)
    dd_stats = _drawdown_duration_stats(ret)
    monthly = (1.0 + ret).resample("ME").prod() - 1.0 if not ret.empty else pd.Series(dtype=float)
    worst_month = float(monthly.min()) if not monthly.empty else float("nan")
    return {
        "ulcer_index": _safe_float(dd_stats.get("ulcer_index")),
        "max_drawdown_duration_days": _safe_float(dd_stats.get("max_drawdown_duration_days")),
        "worst_month_return": worst_month,
    }


def _stress_and_walkforward_rows(
    bundles: list[StrategyBundle],
    *,
    foreign_hard_profile: NetAssumptionProfile,
    crypto_hard_profile: NetAssumptionProfile,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stress_rows: list[dict[str, Any]] = []
    for bundle in bundles:
        if bundle.result.suite.startswith("meta"):
            hard_profile = _profile_scaled(
                bundle.profile,
                profile_id=f"{bundle.profile.profile_id}_hard",
                label=f"{bundle.profile.label} hard",
                transaction_cost_bps=bundle.profile.transaction_cost_bps_assumed + 20.0,
                fx_spread_bps=bundle.profile.fx_spread_bps_assumed + 15.0,
                capital_gains_tax_rate=bundle.profile.capital_gains_tax_rate,
                tax_timing=bundle.profile.tax_timing,
            )
        elif bundle.result.suite.startswith("equities"):
            hard_profile = foreign_hard_profile
        else:
            hard_profile = crypto_hard_profile
        stress_rows.append(_stress_bundle(bundle, delay_days=0, profile=bundle.profile, benchmark_profile=bundle.benchmark_profile, label="base"))
        stress_rows.append(_stress_bundle(bundle, delay_days=1, profile=bundle.profile, benchmark_profile=bundle.benchmark_profile, label="delay_d1"))
        stress_rows.append(_stress_bundle(bundle, delay_days=0, profile=hard_profile, benchmark_profile=hard_profile, label="hard_cost"))
    stress_df = pd.DataFrame(stress_rows)
    wf_blocks = [
        ("test_2022", "2022-01-01", "2022-12-31"),
        ("test_2023_2024", "2023-01-01", "2024-12-31"),
        ("test_2025_now", "2025-01-01", str(pd.Timestamp.now("UTC").date())),
    ]
    wf_rows: list[dict[str, Any]] = []
    for bundle in bundles:
        wf_rows.extend(_walkforward_rows(bundle, wf_blocks))
    return stress_df, pd.DataFrame(wf_rows)


def _worth_it_row(
    *,
    bundle: StrategyBundle,
    base_bundle: StrategyBundle,
    stress_df: pd.DataFrame,
    wf_df: pd.DataFrame,
) -> dict[str, Any]:
    diag = _candidate_diag(bundle)
    base_diag = _candidate_diag(base_bundle)
    stress_sub = stress_df[stress_df["candidate_id"].astype(str) == str(bundle.result.candidate_id)].copy()
    wf_sub = wf_df[wf_df["candidate_id"].astype(str) == str(bundle.result.candidate_id)].copy()
    hard_cost = stress_sub[stress_sub["stress_label"].astype(str) == "hard_cost"]
    delay = stress_sub[stress_sub["stress_label"].astype(str) == "delay_d1"]
    positive_test_share = float((pd.to_numeric(wf_sub.get("edge_vs_benchmark_net_total_return"), errors="coerce") > 0.0).mean()) if not wf_sub.empty else 0.0
    mean_test_edge = float(pd.to_numeric(wf_sub.get("edge_vs_benchmark_net_total_return"), errors="coerce").dropna().mean()) if not wf_sub.empty else float("nan")
    base_ann = max(float(base_bundle.result.net_ann_return), 1e-9)
    base_sharpe = max(float(base_bundle.result.net_sharpe), 1e-9)
    base_mdd_abs = max(abs(float(base_bundle.result.net_max_drawdown)), 1e-9)
    ann_retention = float(bundle.result.net_ann_return / base_ann)
    sharpe_ratio = float(bundle.result.net_sharpe / base_sharpe)
    dd_closure = float((base_mdd_abs - abs(float(bundle.result.net_max_drawdown))) / base_mdd_abs)
    dd_closure = float(np.clip(dd_closure, -1.0, 1.0))
    hard_cost_retention = float(
        _safe_float(hard_cost["net_ann_return"].iloc[0], 0.0) / base_ann if not hard_cost.empty else 0.0
    )
    ulcer_improvement = float(
        (_safe_float(base_diag.get("ulcer_index"), float("nan")) - _safe_float(diag.get("ulcer_index"), float("nan")))
        / max(_safe_float(base_diag.get("ulcer_index"), 1.0), 1e-9)
    ) if np.isfinite(_safe_float(diag.get("ulcer_index"), float("nan"))) and np.isfinite(_safe_float(base_diag.get("ulcer_index"), float("nan"))) else float("nan")
    duration_improvement = float(
        (_safe_float(base_diag.get("max_drawdown_duration_days"), float("nan")) - _safe_float(diag.get("max_drawdown_duration_days"), float("nan")))
        / max(_safe_float(base_diag.get("max_drawdown_duration_days"), 1.0), 1e-9)
    ) if np.isfinite(_safe_float(diag.get("max_drawdown_duration_days"), float("nan"))) and np.isfinite(_safe_float(base_diag.get("max_drawdown_duration_days"), float("nan"))) else float("nan")
    balanced_score = (
        0.40 * ann_retention
        + 0.25 * max(dd_closure, -0.5)
        + 0.15 * min(sharpe_ratio, 1.5)
        + 0.10 * positive_test_share
        + 0.10 * max(0.0, hard_cost_retention)
    )
    worth_it = bool(
        (dd_closure >= 0.08 and ann_retention >= 0.75 and positive_test_share >= (2.0 / 3.0))
        or (dd_closure >= 0.05 and ann_retention >= 0.85 and bundle.result.net_sharpe >= base_bundle.result.net_sharpe - 0.03)
    )
    defensive_trade = bool(dd_closure >= 0.12 and ann_retention >= 0.65)
    return {
        **_result_row(bundle.result),
        **diag,
        "ann_retention_vs_base": ann_retention,
        "sharpe_ratio_vs_base": sharpe_ratio,
        "drawdown_closure_vs_base": dd_closure,
        "ulcer_improvement_vs_base": ulcer_improvement,
        "duration_improvement_vs_base": duration_improvement,
        "mean_test_edge": mean_test_edge,
        "positive_test_share": positive_test_share,
        "hard_cost_ann_return": _safe_float(hard_cost["net_ann_return"].iloc[0], float("nan")) if not hard_cost.empty else float("nan"),
        "hard_cost_edge": _safe_float(hard_cost["edge_vs_benchmark_net_total_return"].iloc[0], float("nan")) if not hard_cost.empty else float("nan"),
        "delay_d1_ann_return": _safe_float(delay["net_ann_return"].iloc[0], float("nan")) if not delay.empty else float("nan"),
        "balanced_score": balanced_score,
        "worth_it": worth_it,
        "defensive_trade": defensive_trade,
    }


def _build_current_stacks(
    *,
    prices_dir: Path,
    crypto_asset_groups: Path,
    crypto_asset_metadata: Path,
    equity_asset_groups: Path,
    equity_asset_metadata: Path,
    benchmark_crypto: str,
    benchmark_equity: str,
    crypto_profile: NetAssumptionProfile,
    foreign_profile: NetAssumptionProfile,
    layered_summary: dict[str, Any],
) -> tuple[StrategyBundle, StrategyBundle, StrategyBundle, pd.Series, pd.Series, pd.Series]:
    crypto_assets = _load_asset_table(crypto_asset_groups, crypto_asset_metadata)
    crypto_returns, crypto_prices, crypto_viability = _load_daily_universe(
        prices_dir=prices_dir,
        asset_table=crypto_assets,
        min_history_days=600,
        max_abs_daily_return=1.5,
    )
    crypto_returns, crypto_prices = _ensure_benchmark_columns(crypto_returns, crypto_prices, prices_dir, [str(benchmark_crypto), "ETH-USD"])
    crypto_tiers = _select_crypto_tiers(crypto_assets, crypto_viability)
    crypto_breadth_all = _build_breadth_signal(
        returns=crypto_returns,
        prices=crypto_prices,
        tickers=crypto_tiers["crypto_all"],
        lookback_days=63,
        ma_days=200,
    )
    crypto_breadth_major = _build_breadth_signal(
        returns=crypto_returns,
        prices=crypto_prices,
        tickers=crypto_tiers["crypto_major8"],
        lookback_days=63,
        ma_days=200,
    )
    crypto_candidates: list[StrategyBundle] = []
    crypto_specs = [
        ("crypto_all__momvol21_base", crypto_tiers["crypto_all"], dict(score_mode="mom_vol_adj", lookback_days=21, rebalance_days=7, top_k=3, asset_ma_days=0, market_ma_days=200, relative_to_benchmark=False, skip_recent_days=0, trailing_stop_dd=None, hard_stop_loss=None)),
        ("crypto_all__momvol21_hard15", crypto_tiers["crypto_all"], dict(score_mode="mom_vol_adj", lookback_days=21, rebalance_days=7, top_k=3, asset_ma_days=0, market_ma_days=200, relative_to_benchmark=False, skip_recent_days=0, trailing_stop_dd=None, hard_stop_loss=0.15)),
        ("crypto_all__slowrel252", crypto_tiers["crypto_all"], dict(score_mode="mom_total", lookback_days=252, rebalance_days=7, top_k=2, asset_ma_days=200, market_ma_days=200, relative_to_benchmark=True, skip_recent_days=0, trailing_stop_dd=None, hard_stop_loss=None)),
        ("crypto_major8__momvol21", crypto_tiers["crypto_major8"], dict(score_mode="mom_vol_adj", lookback_days=21, rebalance_days=7, top_k=3, asset_ma_days=0, market_ma_days=200, relative_to_benchmark=False, skip_recent_days=0, trailing_stop_dd=None, hard_stop_loss=None)),
    ]
    for candidate_id, tickers, kwargs in crypto_specs:
        result = _simulate_asset_rule(
            candidate_id=candidate_id,
            family="crypto",
            allowed_tickers=tickers,
            returns=crypto_returns,
            prices=crypto_prices,
            asset_table=crypto_assets,
            benchmark_ticker=str(benchmark_crypto),
            fallback_ticker=str(benchmark_crypto),
            profile=crypto_profile,
            benchmark_profile=crypto_profile,
            stop_to_cash=True,
            **kwargs,
        )
        if result is not None:
            crypto_candidates.append(
                StrategyBundle(
                    result=result,
                    benchmark_gross_ret=pd.to_numeric(crypto_returns[str(benchmark_crypto)], errors="coerce").fillna(0.0).astype(float),
                    profile=crypto_profile,
                    benchmark_profile=crypto_profile,
                )
            )
    base_crypto_candidates = list(crypto_candidates)
    breadth_overlays = [
        ("crypto_breadth_all__55_70", base_crypto_candidates[0], crypto_breadth_all, 0.55, 0.70, "scale"),
        ("crypto_breadth_all__60_75", base_crypto_candidates[0], crypto_breadth_all, 0.60, 0.75, "scale"),
        ("crypto_breadth_major__55_70", base_crypto_candidates[0], crypto_breadth_major, 0.55, 0.70, "scale"),
    ]
    for candidate_id, base_bundle, breadth_signal, low, high, mode in breadth_overlays:
        crypto_candidates.append(
            _apply_breadth_overlay_to_bundle(
                candidate_id=candidate_id,
                bundle=base_bundle,
                breadth_signal=breadth_signal,
                low_threshold=low,
                high_threshold=high,
                mode=mode,
            )
        )
    best_crypto_id = str(((layered_summary.get("best_crypto") or {}).get("candidate_id", "crypto_all__momvol21_hard15"))).strip()
    crypto_map = {bundle.result.candidate_id: bundle for bundle in crypto_candidates}
    best_crypto_bundle = crypto_map.get(best_crypto_id, crypto_map["crypto_all__momvol21_hard15"])

    equity_assets = _load_asset_table(equity_asset_groups, equity_asset_metadata)
    equity_assets = equity_assets[~equity_assets["asset_group"].astype(str).isin(EQUITY_EXCLUDED)].copy()
    equity_returns, equity_prices, _ = _load_daily_universe(
        prices_dir=prices_dir,
        asset_table=equity_assets,
        min_history_days=1200,
        max_abs_daily_return=0.8,
    )
    equity_returns, equity_prices = _ensure_benchmark_columns(equity_returns, equity_prices, prices_dir, [str(benchmark_equity)])
    group_map = {}
    for group, sub in equity_assets.groupby("asset_group", sort=True):
        tickers = [ticker for ticker in sub["ticker"].astype(str).tolist() if ticker in equity_returns.columns]
        if len(tickers) >= 6:
            group_map[str(group)] = tickers

    equity_candidates: list[StrategyBundle] = []
    v2_specs = [
        ("equities_v2__slow126__g2__a2", 63, 126, 2, 2, 126, 200, 200),
        ("equities_v2__slow126__g2__a3", 63, 126, 2, 3, 126, 200, 200),
        ("equities_v2__slow126__g3__a2", 63, 126, 3, 2, 126, 200, 200),
        ("equities_v2__slow126__g3__a3", 63, 126, 3, 3, 126, 200, 200),
        ("equities_v2__slow189__g2__a2", 63, 189, 2, 2, 126, 200, 200),
        ("equities_v2__slow189__g2__a3", 63, 189, 2, 3, 126, 200, 200),
        ("equities_v2__slow189__g3__a2", 63, 189, 3, 2, 126, 200, 200),
        ("equities_v2__slow189__g3__a3", 63, 189, 3, 3, 126, 200, 200),
        ("equities_v2__slow189__g3__a1", 63, 189, 3, 1, 126, 200, 200),
        ("equities_v2__slow189__g4__a1", 63, 189, 4, 1, 126, 200, 200),
        ("equities_v2__slow252__g3__a1", 63, 252, 3, 1, 126, 200, 200),
        ("equities_v2__slow252__g3__a2_m150", 63, 252, 3, 2, 126, 150, 150),
    ]
    for cid, gf, gs, gk, apg, alb, ama, mma in v2_specs:
        bundle = _simulate_equity_group_sleeve_v2(
            candidate_id=cid,
            returns=equity_returns,
            prices=equity_prices,
            asset_table=equity_assets,
            equity_groups=group_map,
            benchmark_ticker=str(benchmark_equity),
            group_lookback_fast=int(gf),
            group_lookback_slow=int(gs),
            group_top_k=int(gk),
            assets_per_group=int(apg),
            asset_lookback=int(alb),
            asset_ma_days=int(ama),
            market_ma_days=int(mma),
            profile=foreign_profile,
            benchmark_profile=foreign_profile,
        )
        if bundle is not None:
            equity_candidates.append(bundle)
    v3_specs = [
        ("equities_v3__slow189__g2__a2__br30__cap45", 63, 189, 2, 2, 126, 200, 200, 0.30, 0.45),
        ("equities_v3__slow189__g2__a2__br35__cap40", 63, 189, 2, 2, 126, 200, 200, 0.35, 0.40),
        ("equities_v3__slow189__g3__a2__br30__cap45", 63, 189, 3, 2, 126, 200, 200, 0.30, 0.45),
        ("equities_v3__slow189__g3__a2__br35__cap40", 63, 189, 3, 2, 126, 200, 200, 0.35, 0.40),
        ("equities_v3__slow252__g3__a2__br30__cap45", 63, 252, 3, 2, 126, 200, 200, 0.30, 0.45),
        ("equities_v3__slow252__g2__a2__br35__cap40", 63, 252, 2, 2, 126, 200, 200, 0.35, 0.40),
    ]
    for cid, gf, gs, gk, apg, alb, ama, mma, br, cap in v3_specs:
        bundle = _simulate_equity_group_sleeve_v3(
            candidate_id=cid,
            returns=equity_returns,
            prices=equity_prices,
            asset_table=equity_assets,
            equity_groups=group_map,
            benchmark_ticker=str(benchmark_equity),
            group_lookback_fast=int(gf),
            group_lookback_slow=int(gs),
            group_top_k=int(gk),
            assets_per_group=int(apg),
            asset_lookback=int(alb),
            asset_ma_days=int(ama),
            market_ma_days=int(mma),
            min_group_breadth=float(br),
            max_group_weight=float(cap),
            profile=foreign_profile,
            benchmark_profile=foreign_profile,
        )
        if bundle is not None:
            equity_candidates.append(bundle)
    regime_series = _load_structural_regime_series(ROOT)
    spy_series = pd.to_numeric(equity_prices[str(benchmark_equity)], errors="coerce")
    equity_base_df = pd.DataFrame([_result_row(b.result) for b in equity_candidates]).sort_values(
        ["net_ann_return", "net_sharpe"], ascending=[False, False]
    )
    robust_base_df = equity_base_df.copy()
    robust_base_df["robust_score"] = (
        0.45 * pd.to_numeric(robust_base_df["net_ann_return"], errors="coerce").fillna(0.0)
        + 0.35 * pd.to_numeric(robust_base_df["net_sharpe"], errors="coerce").fillna(0.0)
        + 0.20 * (1.0 + pd.to_numeric(robust_base_df["net_max_drawdown"], errors="coerce").fillna(-1.0))
    )
    equity_map = {bundle.result.candidate_id: bundle for bundle in equity_candidates}
    ann_pool = [equity_map[str(cid)] for cid in equity_base_df.head(4)["candidate_id"].astype(str).tolist()]
    robust_pool = [equity_map[str(cid)] for cid in robust_base_df.sort_values(["robust_score", "net_sharpe"], ascending=[False, False]).head(4)["candidate_id"].astype(str).tolist()]
    for agg_rank, agg_bundle in enumerate(ann_pool, start=1):
        for rob_rank, rob_bundle in enumerate(robust_pool, start=1):
            if agg_bundle.result.candidate_id == rob_bundle.result.candidate_id:
                continue
            equity_candidates.append(
                _simulate_equity_trail_switch_bundle(
                    candidate_id=f"equities_meta__trail_switch__a{agg_rank}__r{rob_rank}",
                    aggressive_bundle=agg_bundle,
                    robust_bundle=rob_bundle,
                    regime_series=regime_series,
                    spy_prices=spy_series,
                )
            )
    best_equity_id = str(((layered_summary.get("best_equity") or {}).get("candidate_id", "equities_v2__slow189__g3__a2"))).strip()
    equity_map = {bundle.result.candidate_id: bundle for bundle in equity_candidates}
    best_equity_bundle = equity_map.get(best_equity_id, equity_map["equities_v2__slow189__g3__a2"])

    btc_prices = pd.to_numeric(crypto_prices[str(benchmark_crypto)], errors="coerce")
    spy_prices = pd.to_numeric(equity_prices[str(benchmark_equity)], errors="coerce")
    meta_candidates: list[StrategyBundle] = []
    meta_v1 = _build_meta_switch(
        candidate_id="meta_v1__btc63_vs_equity",
        crypto=best_crypto_bundle.result,
        equities=best_equity_bundle.result,
        btc_prices=btc_prices,
        spy_prices=spy_prices,
        crypto_profile=crypto_profile,
        equity_profile=foreign_profile,
    )
    blended_profile = _profile_scaled(
        crypto_profile,
        profile_id="meta_v1_profile",
        label="Meta v1 blended",
        transaction_cost_bps=0.5 * crypto_profile.transaction_cost_bps_assumed + 0.5 * foreign_profile.transaction_cost_bps_assumed,
        fx_spread_bps=0.5 * crypto_profile.fx_spread_bps_assumed + 0.5 * foreign_profile.fx_spread_bps_assumed,
        capital_gains_tax_rate=0.5 * crypto_profile.capital_gains_tax_rate + 0.5 * foreign_profile.capital_gains_tax_rate,
        tax_timing="monthly_positive_proxy",
    )
    meta_candidates.append(
        StrategyBundle(
            result=meta_v1,
            benchmark_gross_ret=(0.5 * pd.to_numeric(btc_prices.pct_change(), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0) + 0.5 * pd.to_numeric(spy_prices.pct_change(), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)).astype(float),
            profile=blended_profile,
            benchmark_profile=blended_profile,
        )
    )
    meta_candidates.append(
        _build_meta_switch_v2(
            candidate_id="meta_v2_disc__63_126",
            crypto_bundle=best_crypto_bundle,
            equity_bundle=best_equity_bundle,
            btc_prices=btc_prices,
            spy_prices=spy_prices,
            mode="discrete",
            fast_window=63,
            slow_window=126,
            vol_window=63,
            vol_quantile=0.80,
            max_crypto_weight=0.9,
        )
    )
    meta_candidates.append(
        _build_meta_switch_v2(
            candidate_id="meta_v2_cont__63_126_q80",
            crypto_bundle=best_crypto_bundle,
            equity_bundle=best_equity_bundle,
            btc_prices=btc_prices,
            spy_prices=spy_prices,
            mode="continuous",
            fast_window=63,
            slow_window=126,
            vol_window=63,
            vol_quantile=0.80,
            max_crypto_weight=0.85,
        )
    )
    meta_candidates.append(
        _build_meta_switch_v2(
            candidate_id="meta_v2_cont__126_189_q75",
            crypto_bundle=best_crypto_bundle,
            equity_bundle=best_equity_bundle,
            btc_prices=btc_prices,
            spy_prices=spy_prices,
            mode="continuous",
            fast_window=126,
            slow_window=189,
            vol_window=63,
            vol_quantile=0.75,
            max_crypto_weight=0.75,
        )
    )
    meta_candidates.append(
        _build_meta_switch_v3(
            candidate_id="meta_v3_asym__63_126__br55_45",
            crypto_bundle=best_crypto_bundle,
            equity_bundle=best_equity_bundle,
            btc_prices=btc_prices,
            spy_prices=spy_prices,
            breadth_signal=crypto_breadth_all,
            fast_window=63,
            slow_window=126,
            entry_breadth=0.55,
            exit_breadth=0.45,
            max_crypto_weight=0.80,
            cash_floor=0.10,
        )
    )
    meta_candidates.append(
        _build_meta_switch_v3(
            candidate_id="meta_v3_asym__63_126__br60_50",
            crypto_bundle=best_crypto_bundle,
            equity_bundle=best_equity_bundle,
            btc_prices=btc_prices,
            spy_prices=spy_prices,
            breadth_signal=crypto_breadth_all,
            fast_window=63,
            slow_window=126,
            entry_breadth=0.60,
            exit_breadth=0.50,
            max_crypto_weight=0.75,
            cash_floor=0.12,
        )
    )
    meta_candidates.append(
        _build_meta_switch_v3(
            candidate_id="meta_v3_major__63_126__br55_45",
            crypto_bundle=best_crypto_bundle,
            equity_bundle=best_equity_bundle,
            btc_prices=btc_prices,
            spy_prices=spy_prices,
            breadth_signal=crypto_breadth_major,
            fast_window=63,
            slow_window=126,
            entry_breadth=0.55,
            exit_breadth=0.45,
            max_crypto_weight=0.70,
            cash_floor=0.15,
        )
    )
    promoted = (
        layered_summary.get("promoted_candidate")
        or layered_summary.get("tournament_winner")
        or layered_summary.get("frozen_walkforward_winner")
        or {}
    )
    base_meta_id = str((promoted or {}).get("candidate_id", "meta_v1__btc63_vs_equity")).strip()
    meta_map = {bundle.result.candidate_id: bundle for bundle in meta_candidates}
    base_meta_bundle = meta_map.get(base_meta_id, meta_map["meta_v1__btc63_vs_equity"])
    return base_meta_bundle, best_crypto_bundle, best_equity_bundle, btc_prices, spy_prices, crypto_breadth_all


def main() -> None:
    ap = argparse.ArgumentParser(description="Suite de controles anti-drawdown sobre o meta-switch atual.")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--net-assumptions", default="config/profit_net_assumptions.json")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--outdir-root", default="results/validation/profit_drawdown_control_suite")
    args = ap.parse_args()

    outdir = (ROOT / args.outdir_root / _run_id()).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    prices_dir = (ROOT / args.prices_dir).resolve()

    profiles = load_net_assumption_profiles((ROOT / args.net_assumptions).resolve())
    foreign_profile = profiles["profiles"]["foreign_financial_brazil_resident"]
    crypto_profile = profiles["profiles"]["crypto_global_brazil_resident_conservative"]
    foreign_hard_profile = _profile_scaled(
        foreign_profile,
        profile_id="foreign_hard",
        label="Foreign hard frictions",
        transaction_cost_bps=20.0,
        fx_spread_bps=45.0,
        capital_gains_tax_rate=0.15,
        tax_timing="annual_positive_proxy",
    )
    crypto_hard_profile = _profile_scaled(
        crypto_profile,
        profile_id="crypto_hard",
        label="Crypto hard frictions",
        transaction_cost_bps=45.0,
        fx_spread_bps=45.0,
        capital_gains_tax_rate=0.15,
        tax_timing="monthly_positive_proxy",
    )
    layered_summary = _latest_json(ROOT, "results/validation/profit_layered_engine_suite/*/summary.json")
    base_bundle, crypto_bundle, equity_bundle, btc_prices, spy_prices, crypto_breadth = _build_current_stacks(
        prices_dir=prices_dir,
        crypto_asset_groups=(ROOT / args.crypto_asset_groups).resolve(),
        crypto_asset_metadata=(ROOT / args.crypto_asset_metadata).resolve(),
        equity_asset_groups=(ROOT / args.equity_asset_groups).resolve(),
        equity_asset_metadata=(ROOT / args.equity_asset_metadata).resolve(),
        benchmark_crypto=str(args.benchmark_crypto),
        benchmark_equity=str(args.benchmark_equity),
        crypto_profile=crypto_profile,
        foreign_profile=foreign_profile,
        layered_summary=layered_summary,
    )
    context = _build_meta_context(
        base_bundle=base_bundle,
        crypto_bundle=crypto_bundle,
        equity_bundle=equity_bundle,
        btc_prices=btc_prices,
        spy_prices=spy_prices,
        crypto_breadth=crypto_breadth,
    )
    structural_regime = _load_structural_regime_series(ROOT)

    base_scale = pd.Series(1.0, index=base_bundle.result.gross_ret.index, dtype=float)
    conviction_25_100 = _build_conviction_scale(context, min_active=0.25, max_active=1.0)
    conviction_40_100 = _build_conviction_scale(context, min_active=0.40, max_active=1.0)
    vol_20 = _build_vol_target_scale(base_bundle.result.gross_ret, target_ann_vol=0.20, window=63, min_scale=0.25, max_scale=1.0)
    vol_25 = _build_vol_target_scale(base_bundle.result.gross_ret, target_ann_vol=0.25, window=63, min_scale=0.30, max_scale=1.0)
    dd_reduce35 = _build_drawdown_guard_scale(base_bundle.result.gross_ret, trigger_dd=0.12, release_dd=0.06, reduced_scale=0.35, cooldown_days=10)
    kill_18 = _build_drawdown_guard_scale(base_bundle.result.gross_ret, trigger_dd=0.18, release_dd=0.08, reduced_scale=0.0, cooldown_days=21)
    crypto_guard_30 = _build_source_specific_scale(
        context,
        crypto_scale=_build_crypto_guard_scale(
            context,
            low_breadth=0.45,
            high_breadth=0.72,
            min_crypto_scale=0.30,
            edge_floor=0.00,
            edge_full=0.18,
        ),
        equity_scale=1.0,
        cash_scale=0.0,
    )
    crypto_guard_20 = _build_source_specific_scale(
        context,
        crypto_scale=_build_crypto_guard_scale(
            context,
            low_breadth=0.50,
            high_breadth=0.78,
            min_crypto_scale=0.20,
            edge_floor=0.02,
            edge_full=0.20,
        ),
        equity_scale=1.0,
        cash_scale=0.0,
    )
    crypto_vol_35 = _build_source_specific_scale(
        context,
        crypto_scale=_build_vol_target_scale(crypto_bundle.result.gross_ret.reindex(base_bundle.result.gross_ret.index).fillna(0.0), target_ann_vol=0.35, window=42, min_scale=0.25, max_scale=1.0),
        equity_scale=1.0,
        cash_scale=0.0,
    )
    regime_global_1 = _build_regime_scaled_series(
        base_bundle.result.gross_ret.index,
        structural_regime,
        {"stress": 0.45, "transition": 0.75, "stable": 1.0, "dispersion": 1.0},
        default=1.0,
    )
    regime_global_2 = _build_regime_scaled_series(
        base_bundle.result.gross_ret.index,
        structural_regime,
        {"stress": 0.35, "transition": 0.70, "stable": 0.95, "dispersion": 1.0},
        default=1.0,
    )
    regime_crypto_guard_1 = _build_source_specific_scale(
        context,
        crypto_scale=_build_regime_aware_crypto_guard_scale(
            context,
            structural_regime,
            params_by_regime={
                "stress": {"low_breadth": 0.62, "high_breadth": 0.82, "min_crypto_scale": 0.10, "edge_floor": 0.02, "edge_full": 0.22},
                "transition": {"low_breadth": 0.55, "high_breadth": 0.78, "min_crypto_scale": 0.20, "edge_floor": 0.01, "edge_full": 0.20},
                "stable": {"low_breadth": 0.45, "high_breadth": 0.72, "min_crypto_scale": 0.30, "edge_floor": 0.00, "edge_full": 0.18},
                "dispersion": {"low_breadth": 0.42, "high_breadth": 0.70, "min_crypto_scale": 0.35, "edge_floor": -0.01, "edge_full": 0.16},
            },
        ),
        equity_scale=1.0,
        cash_scale=0.0,
    )
    regime_crypto_guard_2 = _build_source_specific_scale(
        context,
        crypto_scale=_build_regime_aware_crypto_guard_scale(
            context,
            structural_regime,
            params_by_regime={
                "stress": {"low_breadth": 0.65, "high_breadth": 0.85, "min_crypto_scale": 0.00, "edge_floor": 0.03, "edge_full": 0.24},
                "transition": {"low_breadth": 0.58, "high_breadth": 0.80, "min_crypto_scale": 0.15, "edge_floor": 0.02, "edge_full": 0.22},
                "stable": {"low_breadth": 0.47, "high_breadth": 0.74, "min_crypto_scale": 0.25, "edge_floor": 0.00, "edge_full": 0.18},
                "dispersion": {"low_breadth": 0.44, "high_breadth": 0.72, "min_crypto_scale": 0.30, "edge_floor": -0.01, "edge_full": 0.16},
            },
        ),
        equity_scale=1.0,
        cash_scale=0.0,
    )

    combo_conv_vol = (conviction_25_100 * _build_vol_target_scale(base_bundle.result.gross_ret * conviction_25_100, target_ann_vol=0.25, window=63, min_scale=0.30, max_scale=1.0)).clip(0.0, 1.0)
    combo_full = (
        conviction_25_100
        * _build_vol_target_scale(base_bundle.result.gross_ret * conviction_25_100, target_ann_vol=0.22, window=63, min_scale=0.25, max_scale=1.0)
        * _build_drawdown_guard_scale(base_bundle.result.gross_ret * conviction_25_100, trigger_dd=0.14, release_dd=0.07, reduced_scale=0.35, cooldown_days=14)
    ).clip(0.0, 1.0)
    combo_kill = (
        conviction_40_100
        * _build_vol_target_scale(base_bundle.result.gross_ret * conviction_40_100, target_ann_vol=0.24, window=42, min_scale=0.35, max_scale=1.0)
        * _build_drawdown_guard_scale(base_bundle.result.gross_ret * conviction_40_100, trigger_dd=0.16, release_dd=0.08, reduced_scale=0.0, cooldown_days=21)
    ).clip(0.0, 1.0)
    combo_crypto_guard_dd = (crypto_guard_30 * dd_reduce35).clip(0.0, 1.0)
    combo_crypto_guard_conv = (crypto_guard_20 * conviction_25_100).clip(0.0, 1.0)
    combo_regime_guard_1 = (regime_global_1 * regime_crypto_guard_1).clip(0.0, 1.0)
    combo_regime_guard_2 = (regime_global_2 * regime_crypto_guard_2).clip(0.0, 1.0)
    combo_regime_guard_dd = (regime_global_1 * regime_crypto_guard_1 * dd_reduce35).clip(0.0, 1.0)
    exit_eq_fast = _build_meta_early_exit_bundle(
        candidate_id="meta_exit__eq_fast_br50_ed0_dd12",
        context=context,
        base_bundle=base_bundle,
        exit_breadth=0.50,
        reentry_breadth=0.62,
        exit_edge21=0.00,
        reentry_edge21=0.03,
        exit_drawdown=0.12,
        exit_mode="equity",
        cooldown_days=10,
    )
    exit_eq_strict = _build_meta_early_exit_bundle(
        candidate_id="meta_exit__eq_strict_br55_ed2_dd10",
        context=context,
        base_bundle=base_bundle,
        exit_breadth=0.55,
        reentry_breadth=0.68,
        exit_edge21=0.02,
        reentry_edge21=0.05,
        exit_drawdown=0.10,
        exit_mode="equity",
        cooldown_days=14,
    )
    exit_cash_fast = _build_meta_early_exit_bundle(
        candidate_id="meta_exit__cash_br48_ed0_dd12",
        context=context,
        base_bundle=base_bundle,
        exit_breadth=0.48,
        reentry_breadth=0.64,
        exit_edge21=0.00,
        reentry_edge21=0.04,
        exit_drawdown=0.12,
        exit_mode="cash",
        cooldown_days=10,
    )
    exit_eq_guard = _build_meta_early_exit_bundle(
        candidate_id="meta_exit__eq_guard_br52_ed1_dd14",
        context=context,
        base_bundle=base_bundle,
        exit_breadth=0.52,
        reentry_breadth=0.66,
        exit_edge21=0.01,
        reentry_edge21=0.04,
        exit_drawdown=0.14,
        exit_mode="equity",
        cooldown_days=7,
    )

    candidates = [
        base_bundle,
        _apply_scale_overlay(candidate_id="meta_dd_conviction__25_100", base_bundle=base_bundle, scale=conviction_25_100, suite="drawdown_control", family="conviction", notes="scale por conviccao; piso ativo=25%"),
        _apply_scale_overlay(candidate_id="meta_dd_conviction__40_100", base_bundle=base_bundle, scale=conviction_40_100, suite="drawdown_control", family="conviction", notes="scale por conviccao; piso ativo=40%"),
        _apply_scale_overlay(candidate_id="meta_dd_voltarget__20", base_bundle=base_bundle, scale=vol_20, suite="drawdown_control", family="vol_target", notes="vol target 20% anual; janela 63d"),
        _apply_scale_overlay(candidate_id="meta_dd_voltarget__25", base_bundle=base_bundle, scale=vol_25, suite="drawdown_control", family="vol_target", notes="vol target 25% anual; janela 63d"),
        _apply_scale_overlay(candidate_id="meta_dd_guard__12_06_reduce35", base_bundle=base_bundle, scale=dd_reduce35, suite="drawdown_control", family="drawdown_guard", notes="guard 12%/6%; reduz para 35%; cooldown 10d"),
        _apply_scale_overlay(candidate_id="meta_dd_killswitch__18_08_cash21", base_bundle=base_bundle, scale=kill_18, suite="drawdown_control", family="kill_switch", notes="kill-switch 18%/8%; caixa; cooldown 21d"),
        _apply_scale_overlay(candidate_id="meta_dd_crypto_guard__br45_72_min30", base_bundle=base_bundle, scale=crypto_guard_30, suite="drawdown_control", family="crypto_guard", notes="protege so a perna cripto; breadth 45-72; min 30%"),
        _apply_scale_overlay(candidate_id="meta_dd_crypto_guard__br50_78_min20", base_bundle=base_bundle, scale=crypto_guard_20, suite="drawdown_control", family="crypto_guard", notes="protege so a perna cripto; breadth 50-78; min 20%"),
        _apply_scale_overlay(candidate_id="meta_dd_crypto_vol__35", base_bundle=base_bundle, scale=crypto_vol_35, suite="drawdown_control", family="crypto_vol", notes="vol target so na perna cripto; alvo 35% a.a.; janela 42d"),
        _apply_scale_overlay(candidate_id="meta_dd_combo__conv25_vol25", base_bundle=base_bundle, scale=combo_conv_vol, suite="drawdown_control", family="combo", notes="conviccao 25-100 + vol target 25%"),
        _apply_scale_overlay(candidate_id="meta_dd_combo__conv25_vol22_guard14", base_bundle=base_bundle, scale=combo_full, suite="drawdown_control", family="combo", notes="conviccao 25-100 + vol target 22% + guard 14%/7%"),
        _apply_scale_overlay(candidate_id="meta_dd_combo__conv40_vol24_kill16", base_bundle=base_bundle, scale=combo_kill, suite="drawdown_control", family="combo", notes="conviccao 40-100 + vol target 24% + kill-switch 16%/8%"),
        _apply_scale_overlay(candidate_id="meta_dd_combo__crypto_guard30_dd35", base_bundle=base_bundle, scale=combo_crypto_guard_dd, suite="drawdown_control", family="combo", notes="crypto guard 30% + drawdown guard global 35%"),
        _apply_scale_overlay(candidate_id="meta_dd_combo__crypto_guard20_conv25", base_bundle=base_bundle, scale=combo_crypto_guard_conv, suite="drawdown_control", family="combo", notes="crypto guard 20% + conviccao 25-100"),
        _apply_scale_overlay(candidate_id="meta_dd_regime_guard__global45_crypto10", base_bundle=base_bundle, scale=combo_regime_guard_1, suite="drawdown_control", family="regime_guard", notes="thresholds dinamicos por regime estrutural + crypto guard leve"),
        _apply_scale_overlay(candidate_id="meta_dd_regime_guard__global35_crypto00", base_bundle=base_bundle, scale=combo_regime_guard_2, suite="drawdown_control", family="regime_guard", notes="thresholds dinamicos por regime estrutural mais apertados"),
        _apply_scale_overlay(candidate_id="meta_dd_regime_guard__global45_crypto10_dd35", base_bundle=base_bundle, scale=combo_regime_guard_dd, suite="drawdown_control", family="regime_guard", notes="guard global + crypto guard leve calibrados pelo regime estrutural"),
        exit_eq_fast,
        exit_eq_strict,
        exit_cash_fast,
        exit_eq_guard,
    ]

    stress_df, wf_df = _stress_and_walkforward_rows(
        candidates,
        foreign_hard_profile=foreign_hard_profile,
        crypto_hard_profile=crypto_hard_profile,
    )
    compare_rows = [_worth_it_row(bundle=b, base_bundle=base_bundle, stress_df=stress_df, wf_df=wf_df) for b in candidates]
    compare_df = pd.DataFrame(compare_rows).sort_values(
        ["worth_it", "balanced_score", "drawdown_closure_vs_base", "net_ann_return"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    compare_df.to_csv(outdir / "candidate_compare.csv", index=False)
    stress_df.to_csv(outdir / "stress_compare.csv", index=False)
    wf_df.to_csv(outdir / "walkforward_blocks.csv", index=False)

    worthwhile_df = compare_df[compare_df["worth_it"].fillna(False)].copy()
    defensive_df = compare_df[compare_df["defensive_trade"].fillna(False)].copy()
    best_balanced = compare_df.iloc[0].to_dict()
    best_drawdown = compare_df.sort_values(["net_max_drawdown", "balanced_score"], ascending=[False, False]).iloc[0].to_dict()
    worthwhile_ids = worthwhile_df["candidate_id"].astype(str).tolist()
    defensive_ids = [x for x in defensive_df["candidate_id"].astype(str).tolist() if x not in worthwhile_ids]
    winner_bundle = next(b for b in candidates if b.result.candidate_id == str(best_balanced["candidate_id"]))

    status_map: dict[str, str] = {bundle.result.candidate_id: "kill" for bundle in candidates}
    status_map[base_bundle.result.candidate_id] = "keep"
    for cid in worthwhile_ids:
        status_map[str(cid)] = "watch"
    for cid in defensive_ids:
        status_map[str(cid)] = "watch"
    research_rows = _research_rows(
        [base_bundle.result] + [b.result for b in candidates if b.result.candidate_id != base_bundle.result.candidate_id],
        outdir=outdir,
        summary_path=outdir / "summary.json",
        status_map=status_map,
    )

    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "outdir": str(outdir),
        "base_candidate": compare_rows[0],
        "best_balanced_candidate": best_balanced,
        "best_drawdown_candidate": best_drawdown,
        "worth_it_candidates": worthwhile_df.to_dict(orient="records"),
        "defensive_trade_candidates": defensive_df[~defensive_df["candidate_id"].isin(worthwhile_ids)].to_dict(orient="records"),
        "worth_it_count": int(worthwhile_df.shape[0]),
        "defensive_trade_count": int(defensive_df.shape[0]),
        "verdict": {
            "winner_candidate_id": str(best_balanced["candidate_id"]),
            "winner_is_base": bool(str(best_balanced["candidate_id"]) == str(base_bundle.result.candidate_id)),
            "worth_it_candidates": worthwhile_ids,
            "defensive_trade_candidates": defensive_ids,
        },
        "insights": [
            f"Base atual: {base_bundle.result.candidate_id} com ann={base_bundle.result.net_ann_return:.4f}, sharpe={base_bundle.result.net_sharpe:.4f}, mdd={base_bundle.result.net_max_drawdown:.4f}.",
            f"Melhor trade-off anti-drawdown: {best_balanced['candidate_id']} com fechamento relativo de drawdown={_safe_float(best_balanced.get('drawdown_closure_vs_base'), float('nan')):.4f} e retencao de retorno={_safe_float(best_balanced.get('ann_retention_vs_base'), float('nan')):.4f}.",
            ("Nenhum overlay compensou de forma clara; o campeao atual segue melhor." if str(best_balanced["candidate_id"]) == str(base_bundle.result.candidate_id) else f"O overlay {best_balanced['candidate_id']} compensou melhor o corte de drawdown."),
            (f"Candidatos que realmente valeram a pena: {', '.join(worthwhile_ids)}." if worthwhile_ids else "Nenhum overlay entrou como melhora clara; so houve trocas defensivas."),
        ],
        "artifacts": {
            "candidate_compare_csv": str(outdir / "candidate_compare.csv"),
            "stress_compare_csv": str(outdir / "stress_compare.csv"),
            "walkforward_blocks_csv": str(outdir / "walkforward_blocks.csv"),
        },
    }
    summary_path = outdir / "summary.json"
    _write_json(summary_path, summary)
    (outdir / "profit_research_rows.json").write_text(json.dumps(research_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_run_manifest(
        outdir,
        script="scripts/bench/validation/run_profit_drawdown_control_suite.py",
        params={
            "benchmark_crypto": str(args.benchmark_crypto),
            "benchmark_equity": str(args.benchmark_equity),
        },
        paths={
            "summary_json": str(summary_path),
            "candidate_compare_csv": str(outdir / "candidate_compare.csv"),
            "stress_compare_csv": str(outdir / "stress_compare.csv"),
            "walkforward_blocks_csv": str(outdir / "walkforward_blocks.csv"),
            "profit_research_rows_json": str(outdir / "profit_research_rows.json"),
        },
        extra={
            "notes": [
                "Suite compara overlays anti-drawdown causais sobre o campeao atual.",
            ]
        },
    )
    print(json.dumps({"status": "ok", "outdir": str(outdir), "winner": str(best_balanced["candidate_id"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
