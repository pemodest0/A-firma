#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
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

from scripts.ops.run_profit_shadow_suite import (  # noqa: E402
    _load_price_returns,
    _load_returns_wide,
    _resolve_benchmark_series,
)


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _sanitize_json_value(x: Any) -> Any:
    if isinstance(x, float):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): _sanitize_json_value(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_sanitize_json_value(v) for v in x]
    return x


def _json_weight_map(raw: Any) -> dict[str, float]:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in payload.items():
        w = _safe_float(value)
        if np.isfinite(w) and abs(w) > 1e-14:
            out[str(key)] = float(w)
    return out


def _ann_from_simple_returns(simple_returns: pd.Series, periods_per_year: float) -> float:
    x = pd.to_numeric(simple_returns, errors="coerce").dropna().astype(float)
    if x.empty:
        return float("nan")
    eq = np.prod(1.0 + x.to_numpy(dtype=float))
    if eq <= 0.0:
        return float("nan")
    return float(np.power(eq, periods_per_year / float(len(x))) - 1.0)


def _mdd_from_simple_returns(simple_returns: pd.Series) -> float:
    x = pd.to_numeric(simple_returns, errors="coerce").fillna(0.0).astype(float)
    if x.empty:
        return float("nan")
    eq = (1.0 + x).clip(lower=1e-9, upper=10.0).cumprod()
    dd = eq / eq.cummax() - 1.0
    return float(dd.min()) if not dd.empty else float("nan")


def _perf_from_simple_returns(simple_returns: pd.Series, *, periods_per_year: float = 252.0) -> dict[str, float]:
    x = pd.to_numeric(simple_returns, errors="coerce").dropna().astype(float)
    if x.empty:
        return {
            "total_return": float("nan"),
            "ann_return": float("nan"),
            "ann_vol": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "positive_share": float("nan"),
        }
    eq = (1.0 + x).clip(lower=1e-9, upper=10.0).cumprod()
    ann_return = _ann_from_simple_returns(x, periods_per_year=periods_per_year)
    ann_vol = float(x.std(ddof=0) * math.sqrt(periods_per_year))
    drawdown = float((eq / eq.cummax() - 1.0).min())
    return {
        "total_return": float(eq.iloc[-1] - 1.0),
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": float(ann_return / ann_vol) if ann_vol > 1e-12 and np.isfinite(ann_return) else float("nan"),
        "max_drawdown": drawdown,
        "positive_share": float((x > 0).mean()),
    }


def _first_trading_day_per_month(index: pd.DatetimeIndex) -> set[pd.Timestamp]:
    return set(pd.Series(index, index=index).groupby(index.to_period("M")).head(1).index.to_pydatetime())


def _first_trading_day_per_week(index: pd.DatetimeIndex) -> set[pd.Timestamp]:
    return set(pd.Series(index, index=index).groupby(index.to_period("W-MON")).head(1).index.to_pydatetime())


def _l1_turnover(current_core: dict[str, float], current_cash: float, target_core: dict[str, float], target_cash: float) -> float:
    assets = sorted(set(current_core.keys()) | set(target_core.keys()))
    total = abs(float(current_cash) - float(target_cash))
    for asset_id in assets:
        total += abs(float(current_core.get(asset_id, 0.0)) - float(target_core.get(asset_id, 0.0)))
    return float(total)


def _normalize_weights(core_values: dict[str, float], cash_value: float) -> tuple[dict[str, float], float]:
    total_value = float(cash_value + sum(float(v) for v in core_values.values()))
    if total_value <= 0.0:
        return {}, 1.0
    core_weights = {str(k): float(v) / total_value for k, v in core_values.items() if float(v) > 0.0}
    cash_weight = float(cash_value) / total_value
    return core_weights, cash_weight


def build_daily_replay_with_rebalance(
    *,
    monthly_eval: pd.DataFrame,
    returns_wide: pd.DataFrame,
    benchmark_symbol: str,
    benchmark_returns: pd.Series | None = None,
    initial_capital: float,
    cost_bps: float,
    rebalance_frequency: str,
) -> pd.DataFrame:
    if monthly_eval.empty or returns_wide.empty:
        return pd.DataFrame()
    freq = str(rebalance_frequency).strip().lower()
    if freq not in {"monthly", "weekly"}:
        raise ValueError(f"unsupported rebalance_frequency: {rebalance_frequency}")

    daily = returns_wide.copy().sort_index()
    benchmark = _resolve_benchmark_series(
        returns_wide=daily,
        benchmark_symbol=benchmark_symbol,
        benchmark_returns=benchmark_returns,
    )
    month_rows = monthly_eval.copy()
    month_rows["ym"] = month_rows["ym"].astype(str)
    month_rows = month_rows.drop_duplicates(subset=["ym"], keep="last").sort_values("ym").reset_index(drop=True)
    month_map = {str(row["ym"]): row for _, row in month_rows.iterrows()}

    if freq == "monthly":
        rebalance_days = _first_trading_day_per_month(daily.index)
    else:
        rebalance_days = _first_trading_day_per_week(daily.index) | _first_trading_day_per_month(daily.index)

    cost_rate = float(max(0.0, cost_bps) / 10000.0)
    core_weights: dict[str, float] = {}
    cash_weight = 1.0
    hedge_weight = 0.0
    capital = float(initial_capital)
    benchmark_capital = float(initial_capital)
    rows: list[dict[str, Any]] = []

    for dt, ret_row in daily.iterrows():
        ym = dt.to_period("M").strftime("%Y-%m")
        month_row = month_map.get(ym)
        if month_row is None:
            continue
        target_core = _json_weight_map(month_row.get("executed_weights_json", "{}"))
        target_cash = float(max(0.0, _safe_float(month_row.get("cash_weight"), 0.0)))
        target_hedge = _safe_float(month_row.get("hedge_weight"), 0.0)
        turnover = 0.0
        if pd.Timestamp(dt) in rebalance_days:
            turnover = _l1_turnover(core_weights, cash_weight, target_core, target_cash)
            core_weights = dict(target_core)
            cash_weight = float(target_cash)
            hedge_weight = float(target_hedge)
        bench_ret = float(benchmark.loc[dt])
        core_ret = 0.0
        for asset_id, weight in core_weights.items():
            if asset_id in ret_row.index:
                core_ret += float(weight) * float(ret_row[asset_id])
        gross_ret = float(core_ret + hedge_weight * bench_ret)
        net_ret = float(gross_ret - turnover * cost_rate)
        capital *= 1.0 + net_ret
        benchmark_capital *= 1.0 + bench_ret
        core_values = {}
        for asset_id, weight in core_weights.items():
            val = float(weight) * float(max(0.0, 1.0 + float(ret_row.get(asset_id, 0.0))))
            if val > 0.0:
                core_values[str(asset_id)] = val
        cash_value = float(cash_weight)
        core_weights, cash_weight = _normalize_weights(core_values, cash_value)
        rows.append(
            {
                "date": dt.date().isoformat(),
                "ym": ym,
                "rebalance_frequency": freq,
                "cost_bps": float(cost_bps),
                "turnover": float(turnover),
                "gross_return": float(gross_ret),
                "net_return": float(net_ret),
                "benchmark_return": float(bench_ret),
                "capital": float(capital),
                "benchmark_capital": float(benchmark_capital),
                "n_assets": int(len(core_weights)),
                "cash_weight_post": float(cash_weight),
                "hedge_weight": float(hedge_weight),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["capital_peak"] = pd.to_numeric(out["capital"], errors="coerce").cummax()
    out["drawdown"] = pd.to_numeric(out["capital"], errors="coerce") / out["capital_peak"] - 1.0
    return out


def summarize_replay(history: pd.DataFrame, *, return_col: str = "net_return") -> dict[str, Any]:
    if history.empty:
        return {"status": "empty"}
    perf = _perf_from_simple_returns(pd.to_numeric(history[return_col], errors="coerce"), periods_per_year=252.0)
    bench = _perf_from_simple_returns(pd.to_numeric(history["benchmark_return"], errors="coerce"), periods_per_year=252.0)
    last = history.iloc[-1]
    return _sanitize_json_value(
        {
            "status": "ok",
            "start_date": str(history.iloc[0]["date"]),
            "end_date": str(last["date"]),
            "n_days": int(history.shape[0]),
            "capital_end": float(last["capital"]),
            "benchmark_capital_end": float(last["benchmark_capital"]),
            "portfolio": perf,
            "benchmark": bench,
            "edge_vs_benchmark_total_return": _safe_float(perf.get("total_return")) - _safe_float(bench.get("total_return")),
            "avg_turnover_daily": float(pd.to_numeric(history["turnover"], errors="coerce").mean()),
            "total_turnover": float(pd.to_numeric(history["turnover"], errors="coerce").sum()),
        }
    )


def _block_metrics(monthly_eval: pd.DataFrame, start_ym: str, end_ym: str, cost_bps: float) -> dict[str, Any]:
    x = monthly_eval.copy()
    x = x[(x["ym"] >= str(start_ym)) & (x["ym"] <= str(end_ym))].copy()
    if x.empty:
        return {"status": "empty", "start_ym": start_ym, "end_ym": end_ym}
    x["ret"] = pd.to_numeric(x["ret"], errors="coerce").fillna(0.0)
    x["eqw_ret"] = pd.to_numeric(x["eqw_ret"], errors="coerce").fillna(0.0)
    x["turnover"] = pd.to_numeric(x["turnover"], errors="coerce").fillna(0.0)
    x["ret_net"] = x["ret"] - x["turnover"] * float(max(0.0, cost_bps) / 10000.0)
    x["alpha"] = x["ret"] - x["eqw_ret"]
    x["alpha_net"] = x["ret_net"] - x["eqw_ret"]
    return _sanitize_json_value(
        {
            "status": "ok",
            "start_ym": str(start_ym),
            "end_ym": str(end_ym),
            "n_months": int(x.shape[0]),
            "strategy_total_return": float(np.prod(1.0 + x["ret"].to_numpy(dtype=float)) - 1.0),
            "strategy_total_return_net": float(np.prod(1.0 + x["ret_net"].to_numpy(dtype=float)) - 1.0),
            "benchmark_total_return": float(np.prod(1.0 + x["eqw_ret"].to_numpy(dtype=float)) - 1.0),
            "strategy_ann_return": _ann_from_simple_returns(x["ret"], periods_per_year=12.0),
            "strategy_ann_return_net": _ann_from_simple_returns(x["ret_net"], periods_per_year=12.0),
            "benchmark_ann_return": _ann_from_simple_returns(x["eqw_ret"], periods_per_year=12.0),
            "strategy_mdd": _mdd_from_simple_returns(x["ret"]),
            "strategy_mdd_net": _mdd_from_simple_returns(x["ret_net"]),
            "positive_alpha_share": float((x["alpha"] > 0.0).mean()),
            "positive_alpha_share_net": float((x["alpha_net"] > 0.0).mean()),
            "avg_turnover_monthly": float(x["turnover"].mean()),
        }
    )


def _bootstrap_returns(
    *,
    returns: np.ndarray,
    benchmark: np.ndarray,
    periods_per_year: float,
    rng: np.random.Generator,
    n_iter: int,
    sample_len: int,
    block_size: int,
) -> dict[str, Any]:
    if returns.size <= 0 or benchmark.size <= 0:
        return {"status": "empty"}
    strat_total: list[float] = []
    alpha_total: list[float] = []
    strat_ann: list[float] = []
    strat_mdd: list[float] = []
    bench_total: list[float] = []
    n = int(sample_len)
    b = int(max(1, block_size))
    for _ in range(int(max(1, n_iter))):
        idxs: list[int] = []
        while len(idxs) < n:
            start = int(rng.integers(0, len(returns)))
            idxs.extend(((start + j) % len(returns)) for j in range(b))
        idx = np.asarray(idxs[:n], dtype=int)
        r = returns[idx]
        m = benchmark[idx]
        strat_total.append(float(np.prod(1.0 + r) - 1.0))
        bench_total.append(float(np.prod(1.0 + m) - 1.0))
        alpha_total.append(float(np.prod(1.0 + r) - np.prod(1.0 + m)))
        ann = float(np.power(max(1e-9, float(np.prod(1.0 + r))), periods_per_year / float(len(r))) - 1.0)
        strat_ann.append(ann)
        eq = np.cumprod(1.0 + r)
        dd = eq / np.maximum.accumulate(eq) - 1.0
        strat_mdd.append(float(np.min(dd)) if dd.size else float("nan"))
    arr_total = np.asarray(strat_total, dtype=float)
    arr_alpha = np.asarray(alpha_total, dtype=float)
    arr_ann = np.asarray(strat_ann, dtype=float)
    arr_mdd = np.asarray(strat_mdd, dtype=float)
    arr_bench = np.asarray(bench_total, dtype=float)
    return _sanitize_json_value(
        {
            "status": "ok",
            "iterations": int(n_iter),
            "sample_len": int(sample_len),
            "block_size": int(block_size),
            "strategy_total_return_p05": float(np.quantile(arr_total, 0.05)),
            "strategy_total_return_p50": float(np.quantile(arr_total, 0.50)),
            "strategy_total_return_p95": float(np.quantile(arr_total, 0.95)),
            "benchmark_total_return_p50": float(np.quantile(arr_bench, 0.50)),
            "strategy_ann_return_p05": float(np.quantile(arr_ann, 0.05)),
            "strategy_ann_return_p50": float(np.quantile(arr_ann, 0.50)),
            "strategy_ann_return_p95": float(np.quantile(arr_ann, 0.95)),
            "strategy_mdd_p05": float(np.quantile(arr_mdd, 0.05)),
            "strategy_mdd_p50": float(np.quantile(arr_mdd, 0.50)),
            "strategy_mdd_p95": float(np.quantile(arr_mdd, 0.95)),
            "prob_strategy_total_positive": float((arr_total > 0.0).mean()),
            "prob_strategy_beats_benchmark": float((arr_total > arr_bench).mean()),
            "prob_alpha_positive": float((arr_alpha > 0.0).mean()),
            "prob_mdd_worse_than_35pct": float((arr_mdd < -0.35).mean()),
        }
    )


@dataclass(frozen=True)
class FrozenProfile:
    profile_dir: Path
    manifest: dict[str, Any]
    simulation_summary: dict[str, Any]
    params: dict[str, Any]
    best_params: dict[str, Any]


def _load_frozen_profile(profile_dir: Path) -> FrozenProfile:
    manifest = _read_json(profile_dir / "RUN_MANIFEST.json")
    sim_summary = _read_json(profile_dir / "simulation_summary.json")
    params = manifest.get("params", {}) if isinstance(manifest.get("params"), dict) else {}
    best_params = sim_summary.get("best_params", {}) if isinstance(sim_summary.get("best_params"), dict) else {}
    if not params or not best_params:
        raise SystemExit(f"invalid profile dir: {profile_dir}")
    return FrozenProfile(
        profile_dir=profile_dir,
        manifest=manifest,
        simulation_summary=sim_summary,
        params=params,
        best_params=best_params,
    )


def _build_frozen_cmd(profile: FrozenProfile, *, impact_dir: Path, outdir: Path, start_ym: str, train_end: str, opt_cost_bps: float) -> list[str]:
    p = profile.params
    bp = profile.best_params
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "ops" / "run_canonical_systematic_eval.py"),
        "--impact-dir",
        str(impact_dir),
        "--returns-csv",
        str(p["returns_csv"]),
        "--prices-dir",
        str(p.get("prices_dir", ROOT / "data" / "raw" / "finance" / "yfinance_daily")),
        "--outdir",
        str(outdir),
        "--train-end",
        str(train_end),
        "--start-ym",
        str(start_ym),
        "--max-assets-per-month",
        str(_safe_int(p.get("max_assets_per_month"), 220)),
        "--top-k-options",
        str(_safe_int(bp.get("top_k"), 24)),
        "--max-grid-combos",
        "1",
        "--impact-power-options",
        str(_safe_float(bp.get("impact_power"), 0.0)),
        "--wmax-options",
        str(_safe_float(bp.get("w_max"), 0.12)),
        "--mom-lookback-options",
        str(_safe_int(bp.get("mom_lookback"), 0)),
        "--mom-threshold-options",
        str(_safe_float(bp.get("mom_threshold"), 0.0)),
        "--modes",
        str(bp.get("mode", "regime")),
        "--rb-stress-options",
        str(_safe_float(bp.get("rb_stress"), 0.4)),
        "--rb-transition-options",
        str(_safe_float(bp.get("rb_transition"), 0.7)),
        "--rb-stable-options",
        str(_safe_float(bp.get("rb_stable"), 1.0)),
        "--objective-mode",
        str(p.get("objective_mode", "balanced")),
        "--benchmark-symbol",
        str(p.get("benchmark_symbol", "SPY")),
        "--defense-enabled",
        str(_safe_int(p.get("defense_enabled"), 1)),
        "--defense-multiplier",
        str(_safe_float(p.get("defense_multiplier"), 0.9)),
        "--defense-corr-quantile",
        str(_safe_float(p.get("defense_corr_quantile"), 0.8)),
        "--defense-vol-quantile",
        str(_safe_float(p.get("defense_vol_quantile"), 0.8)),
        "--defense-min-history-months",
        str(_safe_int(p.get("defense_min_history_months"), 12)),
        "--defense-require-both",
        str(_safe_int(p.get("defense_require_both"), 0)),
        "--decel-enabled",
        str(_safe_int(p.get("decel_enabled"), 1)),
        "--decel-lookback-months",
        str(_safe_int(p.get("decel_lookback_months"), 6)),
        "--decel-alpha-threshold",
        str(_safe_float(p.get("decel_alpha_threshold"), 0.0)),
        "--decel-min-streak",
        str(_safe_int(p.get("decel_min_streak"), 2)),
        "--decel-multiplier",
        str(_safe_float(p.get("decel_multiplier"), 0.95)),
        "--decel-topk-multiplier",
        str(_safe_float(p.get("decel_topk_multiplier"), 0.9)),
        "--attack-enabled",
        str(_safe_int(p.get("attack_enabled"), 1)),
        "--attack-multiplier",
        str(_safe_float(p.get("attack_multiplier"), 1.2)),
        "--attack-corr-quantile",
        str(_safe_float(p.get("attack_corr_quantile"), 0.4)),
        "--attack-vol-quantile",
        str(_safe_float(p.get("attack_vol_quantile"), 0.4)),
        "--attack-min-history-months",
        str(_safe_int(p.get("attack_min_history_months"), 12)),
        "--attack-require-both",
        str(_safe_int(p.get("attack_require_both"), 1)),
        "--attack-require-positive-alpha",
        str(_safe_int(p.get("attack_require_positive_alpha"), 1)),
        "--attack-alpha-lookback-months",
        str(_safe_int(p.get("attack_alpha_lookback_months"), 3)),
        "--dd-guard-enabled",
        str(_safe_int(p.get("dd_guard_enabled"), 1)),
        "--dd-soft-threshold",
        str(_safe_float(p.get("dd_soft_threshold"), -0.15)),
        "--dd-hard-threshold",
        str(_safe_float(p.get("dd_hard_threshold"), -0.25)),
        "--dd-soft-multiplier",
        str(_safe_float(p.get("dd_soft_multiplier"), 0.85)),
        "--dd-hard-multiplier",
        str(_safe_float(p.get("dd_hard_multiplier"), 0.65)),
        "--regime-topk-enabled",
        str(_safe_int(p.get("regime_topk_enabled"), 1)),
        "--regime-topk-stable-multiplier",
        str(_safe_float(p.get("regime_topk_stable_multiplier"), 1.15)),
        "--regime-topk-transition-multiplier",
        str(_safe_float(p.get("regime_topk_transition_multiplier"), 1.0)),
        "--regime-topk-stress-multiplier",
        str(_safe_float(p.get("regime_topk_stress_multiplier"), 0.85)),
        "--weekly-stress-enabled",
        str(_safe_int(p.get("weekly_stress_enabled"), 1)),
        "--weekly-stress-quantile",
        str(_safe_float(p.get("weekly_stress_quantile"), 0.2)),
        "--weekly-stress-min-history-months",
        str(_safe_int(p.get("weekly_stress_min_history_months"), 12)),
        "--weekly-stress-multiplier",
        str(_safe_float(p.get("weekly_stress_multiplier"), 0.9)),
        "--hybrid-enabled",
        str(_safe_int(p.get("hybrid_enabled"), 1)),
        "--hybrid-lookback-months",
        str(_safe_int(p.get("hybrid_lookback_months"), 3)),
        "--hybrid-weight-impact",
        str(_safe_float(p.get("hybrid_weight_impact"), 0.5)),
        "--hybrid-weight-momentum",
        str(_safe_float(p.get("hybrid_weight_momentum"), 0.35)),
        "--hybrid-weight-liquidity",
        str(_safe_float(p.get("hybrid_weight_liquidity"), 0.15)),
        "--hybrid-liquidity-csv",
        str(p.get("hybrid_liquidity_csv", "")),
        "--layered-enabled",
        str(_safe_int(p.get("layered_enabled"), 1)),
        "--layered-min-sectors",
        str(_safe_int(p.get("layered_min_sectors"), 3)),
        "--layered-max-sectors",
        str(_safe_int(p.get("layered_max_sectors"), 7)),
        "--layered-target-assets-per-sector",
        str(_safe_int(p.get("layered_target_assets_per_sector"), 5)),
        "--layered-sector-score-power",
        str(_safe_float(p.get("layered_sector_score_power"), 1.0)),
        "--layered-min-assets-per-sector",
        str(_safe_int(p.get("layered_min_assets_per_sector"), 1)),
        "--auto-aggr-enabled",
        str(_safe_int(p.get("auto_aggr_enabled"), 1)),
        "--auto-aggr-multiplier",
        str(_safe_float(p.get("auto_aggr_multiplier"), 1.15)),
        "--auto-aggr-topk-multiplier",
        str(_safe_float(p.get("auto_aggr_topk_multiplier"), 1.15)),
        "--auto-aggr-score-quantile",
        str(_safe_float(p.get("auto_aggr_score_quantile"), 0.4)),
        "--auto-aggr-corr-quantile",
        str(_safe_float(p.get("auto_aggr_corr_quantile"), 0.45)),
        "--auto-aggr-vol-quantile",
        str(_safe_float(p.get("auto_aggr_vol_quantile"), 0.45)),
        "--auto-aggr-min-history-months",
        str(_safe_int(p.get("auto_aggr_min_history_months"), 12)),
        "--auto-aggr-require-positive-alpha",
        str(_safe_int(p.get("auto_aggr_require_positive_alpha"), 1)),
        "--auto-aggr-alpha-lookback-months",
        str(_safe_int(p.get("auto_aggr_alpha_lookback_months"), 3)),
        "--auto-aggr-confirm-months",
        str(_safe_int(p.get("auto_aggr_confirm_months"), 2)),
        "--rebalance-control-enabled",
        str(_safe_int(p.get("rebalance_control_enabled"), 1)),
        "--rebalance-deadband-l1",
        str(_safe_float(p.get("rebalance_deadband_l1"), 0.06)),
        "--rebalance-force-l1",
        str(_safe_float(p.get("rebalance_force_l1"), 0.2)),
        "--rebalance-cooldown-months",
        str(_safe_int(p.get("rebalance_cooldown_months"), 1)),
        "--opt-cost-bps",
        str(float(opt_cost_bps)),
        "--opt-turnover-penalty",
        str(_safe_float(p.get("opt_turnover_penalty"), 0.03)),
        "--opt-use-net-ann",
        str(_safe_int(p.get("opt_use_net_ann"), 1)),
        "--shadow-tail-months",
        str(_safe_int(p.get("shadow_tail_months"), 12)),
        "--rb-cap",
        str(_safe_float(p.get("rb_cap"), 1.35)),
    ]
    return cmd


def _run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return int(proc.returncode), proc.stdout or "", proc.stderr or ""


def _materialize_filtered_impact(src_csv: Path, outdir: Path, asset_ids: set[str]) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(src_csv)
    df["asset_id"] = df["asset_id"].astype(str)
    df = df[df["asset_id"].isin({str(x) for x in asset_ids})].copy()
    (outdir / "impact_training_dataset.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    return outdir


def _summarize_scenario(
    outdir: Path,
    returns_wide: pd.DataFrame,
    benchmark_symbol: str,
    benchmark_returns: pd.Series | None,
    initial_capital: float,
) -> dict[str, Any]:
    monthly_eval = pd.read_csv(outdir / "monthly_systematic_eval.csv")
    daily_history = build_daily_replay_with_rebalance(
        monthly_eval=monthly_eval,
        returns_wide=returns_wide,
        benchmark_symbol=benchmark_symbol,
        benchmark_returns=benchmark_returns,
        initial_capital=initial_capital,
        cost_bps=0.0,
        rebalance_frequency="monthly",
    )
    daily_path = outdir / "daily_replay_frozen.csv"
    daily_history.to_csv(daily_path, index=False)
    daily_summary = summarize_replay(daily_history, return_col="net_return")
    _write_json(outdir / "daily_replay_frozen_summary.json", daily_summary)
    sys_summary = _read_json(outdir / "systematic_summary.json")
    sim_summary = _read_json(outdir / "simulation_summary.json")
    return _sanitize_json_value(
        {
            "outdir": str(outdir),
            "daily_total_return": _safe_float(daily_summary.get("portfolio", {}).get("total_return")),
            "daily_ann_return": _safe_float(daily_summary.get("portfolio", {}).get("ann_return")),
            "daily_sharpe": _safe_float(daily_summary.get("portfolio", {}).get("sharpe")),
            "daily_max_drawdown": _safe_float(daily_summary.get("portfolio", {}).get("max_drawdown")),
            "daily_edge_vs_benchmark": _safe_float(daily_summary.get("edge_vs_benchmark_total_return")),
            "worth_it_rate_vs_eqw": _safe_float(sys_summary.get("worth_it_rate_vs_eqw")),
            "monthly_alpha_prob_positive_vs_eqw": _safe_float(sys_summary.get("monthly_alpha_prob_positive_vs_eqw")),
            "strategy_max_drop": _safe_float(sys_summary.get("strategy_max_drop")),
            "best_params": sim_summary.get("best_params", {}),
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Frozen validation suite for profit_attack robustness.")
    ap.add_argument(
        "--profile-dir",
        type=str,
        default="results/ops/profit_shadow/runs/20260306T053930Z/profiles_fast/profit_attack",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default="",
    )
    ap.add_argument("--bootstrap-iters", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()

    profile_dir = Path(str(args.profile_dir)).resolve()
    profile = _load_frozen_profile(profile_dir)
    outdir = Path(str(args.outdir)).resolve() if str(args.outdir).strip() else (ROOT / "results" / "validation" / "profit_attack_validation" / _run_id())
    outdir.mkdir(parents=True, exist_ok=True)

    impact_dir = Path(str(profile.params["impact_dir"])).resolve()
    impact_csv = impact_dir / "impact_training_dataset.csv"
    returns_csv = Path(str(profile.params["returns_csv"])).resolve()
    returns_wide = _load_returns_wide(returns_csv)
    benchmark_symbol = str(profile.params.get("benchmark_symbol", "SPY"))
    prices_dir_raw = str(profile.params.get("prices_dir", "")).strip()
    prices_dir = Path(prices_dir_raw).resolve() if prices_dir_raw else (ROOT / "data" / "raw" / "finance" / "yfinance_daily")
    benchmark_returns: pd.Series | None = None
    try:
        benchmark_returns = _load_price_returns(prices_dir, benchmark_symbol)
    except FileNotFoundError:
        if benchmark_symbol not in returns_wide.columns:
            raise
    initial_capital = 10000.0

    frozen_dir = outdir / "frozen_base"
    cmd = _build_frozen_cmd(
        profile,
        impact_dir=impact_dir,
        outdir=frozen_dir,
        start_ym=str(profile.params.get("start_ym", "2019-01")),
        train_end=str(profile.params.get("train_end", "2023-12-31")),
        opt_cost_bps=_safe_float(profile.params.get("opt_cost_bps"), 10.0),
    )
    code, stdout, stderr = _run_cmd(cmd)
    if code != 0:
        raise SystemExit(f"frozen base run failed: {stderr or stdout}")
    frozen_summary = _summarize_scenario(
        frozen_dir,
        returns_wide,
        benchmark_symbol,
        benchmark_returns,
        initial_capital,
    )

    monthly_eval = pd.read_csv(frozen_dir / "monthly_systematic_eval.csv")
    walkforward_blocks = [
        _block_metrics(monthly_eval, "2024-01", "2024-12", _safe_float(profile.params.get("opt_cost_bps"), 10.0)),
        _block_metrics(monthly_eval, "2025-01", "2025-12", _safe_float(profile.params.get("opt_cost_bps"), 10.0)),
        _block_metrics(monthly_eval, "2026-01", "2026-12", _safe_float(profile.params.get("opt_cost_bps"), 10.0)),
    ]
    _write_json(outdir / "walkforward_frozen.json", {"blocks": walkforward_blocks})

    cost_rows: list[dict[str, Any]] = []
    for freq in ["monthly", "weekly"]:
        for cost_bps in [10.0, 20.0, 30.0, 50.0]:
            history = build_daily_replay_with_rebalance(
                monthly_eval=monthly_eval,
                returns_wide=returns_wide,
                benchmark_symbol=benchmark_symbol,
                benchmark_returns=benchmark_returns,
                initial_capital=initial_capital,
                cost_bps=float(cost_bps),
                rebalance_frequency=freq,
            )
            history_path = outdir / "cost_stress" / f"{freq}_{int(cost_bps)}bps_daily.csv"
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history.to_csv(history_path, index=False)
            summary = summarize_replay(history, return_col="net_return")
            _write_json(history_path.with_suffix(".summary.json"), summary)
            cost_rows.append(
                _sanitize_json_value(
                    {
                        "rebalance_frequency": freq,
                        "cost_bps": float(cost_bps),
                        "daily_total_return": _safe_float(summary.get("portfolio", {}).get("total_return")),
                        "daily_ann_return": _safe_float(summary.get("portfolio", {}).get("ann_return")),
                        "daily_sharpe": _safe_float(summary.get("portfolio", {}).get("sharpe")),
                        "daily_max_drawdown": _safe_float(summary.get("portfolio", {}).get("max_drawdown")),
                        "edge_vs_benchmark_total_return": _safe_float(summary.get("edge_vs_benchmark_total_return")),
                        "avg_turnover_daily": _safe_float(summary.get("avg_turnover_daily")),
                        "total_turnover": _safe_float(summary.get("total_turnover")),
                        "capital_end": _safe_float(summary.get("capital_end")),
                    }
                )
            )
    cost_df = pd.DataFrame(cost_rows)
    cost_df.to_csv(outdir / "cost_stress.csv", index=False)
    _write_json(outdir / "cost_stress.json", {"rows": cost_rows})

    # Universe perturbations: sector drop is not informative here because current impact universe is 100% industrials.
    impact_df = pd.read_csv(impact_csv, usecols=["asset_id"])
    asset_ids = sorted(impact_df["asset_id"].astype(str).unique().tolist())
    selected_freq = (
        monthly_eval["executed_assets"]
        .fillna("")
        .astype(str)
        .str.split(",")
        .explode()
        .astype(str)
        .str.strip()
    )
    selected_freq = selected_freq[selected_freq != ""].value_counts()
    top5_assets = set(selected_freq.head(5).index.tolist())
    rng = random.Random(int(args.seed))
    keep_80 = set(rng.sample(asset_ids, max(1, int(round(len(asset_ids) * 0.80)))))
    rng = random.Random(int(args.seed) + 11)
    keep_65 = set(rng.sample(asset_ids, max(1, int(round(len(asset_ids) * 0.65)))))
    scenarios = [
        {"name": "start_2020", "start_ym": "2020-01", "asset_ids": set(asset_ids)},
        {"name": "start_2021", "start_ym": "2021-01", "asset_ids": set(asset_ids)},
        {"name": "start_2022", "start_ym": "2022-01", "asset_ids": set(asset_ids)},
        {"name": "drop_top5_selected", "start_ym": str(profile.params.get("start_ym", "2019-01")), "asset_ids": set(asset_ids) - top5_assets},
        {"name": "random_keep_80", "start_ym": str(profile.params.get("start_ym", "2019-01")), "asset_ids": keep_80},
        {"name": "random_keep_65", "start_ym": str(profile.params.get("start_ym", "2019-01")), "asset_ids": keep_65},
    ]
    perturb_rows: list[dict[str, Any]] = []
    perturb_root = outdir / "perturbations"
    for scenario in scenarios:
        name = str(scenario["name"])
        scenario_impact_dir = perturb_root / "impact_inputs" / name
        _materialize_filtered_impact(impact_csv, scenario_impact_dir, set(scenario["asset_ids"]))
        scenario_outdir = perturb_root / "runs" / name
        scenario_cmd = _build_frozen_cmd(
            profile,
            impact_dir=scenario_impact_dir,
            outdir=scenario_outdir,
            start_ym=str(scenario["start_ym"]),
            train_end=str(profile.params.get("train_end", "2023-12-31")),
            opt_cost_bps=_safe_float(profile.params.get("opt_cost_bps"), 10.0),
        )
        code, stdout, stderr = _run_cmd(scenario_cmd)
        if code != 0:
            perturb_rows.append({"scenario": name, "status": "failed", "stderr": (stderr or stdout)[:1000]})
            continue
        summary = _summarize_scenario(scenario_outdir, returns_wide, benchmark_symbol, initial_capital)
        perturb_rows.append(
            _sanitize_json_value(
                {
                    "scenario": name,
                    "status": "ok",
                    "start_ym": str(scenario["start_ym"]),
                    "n_assets_kept": int(len(set(scenario["asset_ids"]))),
                }
                | summary
            )
        )
    perturb_df = pd.DataFrame(perturb_rows)
    perturb_df.to_csv(outdir / "perturbations.csv", index=False)
    _write_json(
        outdir / "perturbations.json",
        {
            "note": "Sector-drop perturbation was skipped because the current impact dataset is entirely industrials.",
            "rows": perturb_rows,
        },
    )

    frozen_daily = pd.read_csv(frozen_dir / "daily_replay_frozen.csv")
    monthly_returns = pd.to_numeric(monthly_eval["ret"], errors="coerce").dropna().to_numpy(dtype=float)
    monthly_benchmark = pd.to_numeric(monthly_eval["eqw_ret"], errors="coerce").dropna().to_numpy(dtype=float)
    daily_returns = pd.to_numeric(frozen_daily["net_return"], errors="coerce").dropna().to_numpy(dtype=float)
    daily_benchmark = pd.to_numeric(frozen_daily["benchmark_return"], errors="coerce").dropna().to_numpy(dtype=float)
    rng_np = np.random.default_rng(int(args.seed))
    bootstrap_summary = {
        "monthly_block_bootstrap": _bootstrap_returns(
            returns=monthly_returns,
            benchmark=monthly_benchmark,
            periods_per_year=12.0,
            rng=rng_np,
            n_iter=int(args.bootstrap_iters),
            sample_len=int(len(monthly_returns)),
            block_size=3,
        ),
        "daily_block_bootstrap": _bootstrap_returns(
            returns=daily_returns,
            benchmark=daily_benchmark,
            periods_per_year=252.0,
            rng=np.random.default_rng(int(args.seed) + 101),
            n_iter=int(args.bootstrap_iters),
            sample_len=int(len(daily_returns)),
            block_size=5,
        ),
    }
    _write_json(outdir / "bootstrap.json", bootstrap_summary)

    invest_shadow_summary = _read_json(ROOT / "results" / "ops" / "invest_shadow" / "latest_summary.json")
    profit_shadow_latest = _read_json(ROOT / "results" / "ops" / "profit_shadow" / "latest_summary.json")

    summary = _sanitize_json_value(
        {
            "status": "ok",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "profile_dir": str(profile_dir),
            "frozen_base": frozen_summary,
            "walkforward_frozen": {"blocks": walkforward_blocks},
            "cost_stress": {"rows": cost_rows},
            "perturbations": {
                "note": "Sector-drop perturbation skipped because the current impact universe is entirely industrials.",
                "rows": perturb_rows,
            },
            "bootstrap": bootstrap_summary,
            "shadow_live": {
                "investment_shadow": {
                    "status": invest_shadow_summary.get("status"),
                    "latest_signal": invest_shadow_summary.get("latest_signal"),
                    "live_portfolio_value": ((invest_shadow_summary.get("live") or {}) if isinstance(invest_shadow_summary.get("live"), dict) else {}).get("portfolio_value_brl"),
                },
                "profit_shadow": {
                    "run_id": profit_shadow_latest.get("run_id"),
                    "best_by_profit": (profit_shadow_latest.get("best_by_profit") or {}) if isinstance(profit_shadow_latest.get("best_by_profit"), dict) else {},
                },
            },
            "artifacts": {
                "frozen_base_dir": str(frozen_dir),
                "walkforward_json": str(outdir / "walkforward_frozen.json"),
                "cost_stress_csv": str(outdir / "cost_stress.csv"),
                "perturbations_csv": str(outdir / "perturbations.csv"),
                "bootstrap_json": str(outdir / "bootstrap.json"),
            },
        }
    )
    _write_json(outdir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
