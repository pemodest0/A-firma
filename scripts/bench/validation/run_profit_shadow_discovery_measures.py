#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bench.validation.run_profit_attack_validation_suite import (  # noqa: E402
    _l1_turnover,
    _perf_from_simple_returns,
    build_daily_replay_with_rebalance,
)
from scripts.bench.validation.run_profit_shadow_realism_battery import (  # noqa: E402
    _build_candidate_context,
    _json_weight_map,
    _load_price_returns,
    _resolve_path,
    _safe_float,
    build_daily_replay_with_delay,
    classify_market_slices,
)


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


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sanitize_json_value(x: Any) -> Any:
    if isinstance(x, float):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): _sanitize_json_value(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_sanitize_json_value(v) for v in x]
    return x


def _load_monthly_eval(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ym" not in df.columns:
        raise SystemExit(f"monthly eval missing ym: {path}")
    df["ym"] = df["ym"].astype(str)
    return df.sort_values("ym").drop_duplicates(subset=["ym"], keep="last").reset_index(drop=True)


def _history_for_scenario(
    *,
    monthly_eval: pd.DataFrame,
    returns_wide: pd.DataFrame,
    benchmark_symbol: str,
    benchmark_series: pd.Series,
    scenario: str,
) -> pd.DataFrame:
    token = str(scenario).strip().lower()
    if token == "delay_d0":
        raw = build_daily_replay_with_delay(
            monthly_eval=monthly_eval,
            returns_wide=returns_wide,
            benchmark_returns=benchmark_series,
            initial_capital=10000.0,
            execution_delay_days=0,
        )
        if raw.empty:
            return pd.DataFrame()
        out = raw.copy()
        out["strategy_return"] = pd.to_numeric(out["portfolio_return"], errors="coerce").fillna(0.0).astype(float)
        out["benchmark_return"] = pd.to_numeric(out["benchmark_return"], errors="coerce").fillna(0.0).astype(float)
        return out
    if token == "monthly_10bps":
        raw = build_daily_replay_with_rebalance(
            monthly_eval=monthly_eval,
            returns_wide=returns_wide,
            benchmark_symbol=benchmark_symbol,
            benchmark_returns=benchmark_series,
            initial_capital=10000.0,
            cost_bps=10.0,
            rebalance_frequency="monthly",
        )
        if raw.empty:
            return pd.DataFrame()
        out = raw.copy()
        out["strategy_return"] = pd.to_numeric(out["net_return"], errors="coerce").fillna(0.0).astype(float)
        out["benchmark_return"] = pd.to_numeric(out["benchmark_return"], errors="coerce").fillna(0.0).astype(float)
        return out
    raise ValueError(f"unsupported scenario: {scenario}")


def _compound_return(simple_returns: pd.Series) -> float:
    x = pd.to_numeric(simple_returns, errors="coerce").dropna().astype(float)
    if x.empty:
        return float("nan")
    return float((1.0 + x).clip(lower=1e-9, upper=10.0).prod() - 1.0)


def _capture_ratio(strategy_returns: pd.Series, benchmark_returns: pd.Series, *, positive: bool) -> float:
    strat = pd.to_numeric(strategy_returns, errors="coerce").fillna(0.0).astype(float)
    bench = pd.to_numeric(benchmark_returns, errors="coerce").fillna(0.0).astype(float)
    mask = bench > 0.0 if positive else bench < 0.0
    if not bool(mask.any()):
        return float("nan")
    strat_total = _compound_return(strat.loc[mask])
    bench_total = _compound_return(bench.loc[mask])
    if not np.isfinite(bench_total) or abs(float(bench_total)) < 1e-12:
        return float("nan")
    return float(strat_total / bench_total)


def _drawdown_duration_stats(simple_returns: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(simple_returns, errors="coerce").fillna(0.0).astype(float)
    if x.empty:
        return {"max_drawdown_duration_days": float("nan"), "ulcer_index": float("nan")}
    eq = (1.0 + x).clip(lower=1e-9, upper=10.0).cumprod()
    dd = eq / eq.cummax() - 1.0
    ulcer = float(np.sqrt(np.mean(np.square(dd.to_numpy(dtype=float))))) if len(dd) else float("nan")
    max_len = 0
    cur_len = 0
    for value in dd.to_numpy(dtype=float):
        if value < -1e-12:
            cur_len += 1
            max_len = max(max_len, cur_len)
        else:
            cur_len = 0
    return {
        "max_drawdown_duration_days": float(max_len),
        "ulcer_index": ulcer,
    }


def _rolling_outperformance_share(strategy_returns: pd.Series, benchmark_returns: pd.Series, window: int) -> float:
    strat = pd.to_numeric(strategy_returns, errors="coerce").fillna(0.0).astype(float)
    bench = pd.to_numeric(benchmark_returns, errors="coerce").fillna(0.0).astype(float)
    if len(strat) < int(window) or len(bench) < int(window):
        return float("nan")
    strat_roll = (1.0 + strat).rolling(int(window)).apply(np.prod, raw=True) - 1.0
    bench_roll = (1.0 + bench).rolling(int(window)).apply(np.prod, raw=True) - 1.0
    mask = strat_roll.notna() & bench_roll.notna()
    if not bool(mask.any()):
        return float("nan")
    return float((strat_roll[mask] > bench_roll[mask]).mean())


def _beta_and_corr(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> dict[str, float]:
    strat = pd.to_numeric(strategy_returns, errors="coerce").fillna(0.0).astype(float)
    bench = pd.to_numeric(benchmark_returns, errors="coerce").fillna(0.0).astype(float)
    aligned = pd.concat([strat.rename("s"), bench.rename("b")], axis=1).dropna()
    if aligned.empty:
        return {"beta": float("nan"), "corr": float("nan")}
    var_b = float(aligned["b"].var(ddof=0))
    beta = float(aligned["s"].cov(aligned["b"], ddof=0) / var_b) if var_b > 1e-12 else float("nan")
    corr = float(aligned["s"].corr(aligned["b"])) if aligned.shape[0] >= 2 else float("nan")
    return {"beta": beta, "corr": corr}


def _tail_alpha(strategy_returns: pd.Series, benchmark_returns: pd.Series, q: float) -> dict[str, float]:
    strat = pd.to_numeric(strategy_returns, errors="coerce").fillna(0.0).astype(float)
    bench = pd.to_numeric(benchmark_returns, errors="coerce").fillna(0.0).astype(float)
    if strat.empty or bench.empty:
        return {
            "worst_tail_strategy_mean": float("nan"),
            "worst_tail_benchmark_mean": float("nan"),
            "worst_tail_alpha_mean": float("nan"),
            "best_tail_strategy_mean": float("nan"),
            "best_tail_benchmark_mean": float("nan"),
            "best_tail_alpha_mean": float("nan"),
        }
    lo = float(bench.quantile(float(q)))
    hi = float(bench.quantile(float(1.0 - q)))
    worst = bench <= lo
    best = bench >= hi
    return {
        "worst_tail_strategy_mean": float(strat.loc[worst].mean()) if bool(worst.any()) else float("nan"),
        "worst_tail_benchmark_mean": float(bench.loc[worst].mean()) if bool(worst.any()) else float("nan"),
        "worst_tail_alpha_mean": float((strat.loc[worst] - bench.loc[worst]).mean()) if bool(worst.any()) else float("nan"),
        "best_tail_strategy_mean": float(strat.loc[best].mean()) if bool(best.any()) else float("nan"),
        "best_tail_benchmark_mean": float(bench.loc[best].mean()) if bool(best.any()) else float("nan"),
        "best_tail_alpha_mean": float((strat.loc[best] - bench.loc[best]).mean()) if bool(best.any()) else float("nan"),
    }


def _yearly_returns(history: pd.DataFrame, label: str, scenario: str) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()
    out = history.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date")
    out["year"] = out["date"].dt.year.astype(int)
    rows: list[dict[str, Any]] = []
    for year, group in out.groupby("year"):
        strat_total = _compound_return(pd.to_numeric(group["strategy_return"], errors="coerce"))
        bench_total = _compound_return(pd.to_numeric(group["benchmark_return"], errors="coerce"))
        rows.append(
            {
                "profile": label,
                "scenario": scenario,
                "year": int(year),
                "strategy_total_return": _safe_float(strat_total),
                "benchmark_total_return": _safe_float(bench_total),
                "edge_total_return": _safe_float(strat_total) - _safe_float(bench_total),
            }
        )
    return pd.DataFrame(rows)


def _slice_summary(history: pd.DataFrame, label: str, scenario: str) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()
    idx = pd.to_datetime(history["date"], errors="coerce")
    strat = pd.Series(pd.to_numeric(history["strategy_return"], errors="coerce").fillna(0.0).astype(float).to_numpy(dtype=float), index=idx)
    bench = pd.Series(pd.to_numeric(history["benchmark_return"], errors="coerce").fillna(0.0).astype(float).to_numpy(dtype=float), index=idx)
    labels = classify_market_slices(bench)
    rows: list[dict[str, Any]] = []
    for slice_name in ["bull", "bear", "recovery", "sideways"]:
        mask = labels == slice_name
        if not bool(mask.any()):
            continue
        perf_s = _perf_from_simple_returns(strat.loc[mask], periods_per_year=252.0)
        perf_b = _perf_from_simple_returns(bench.loc[mask], periods_per_year=252.0)
        rows.append(
            {
                "profile": label,
                "scenario": scenario,
                "slice": slice_name,
                "n_days": int(mask.sum()),
                "share_days": float(mask.mean()),
                "strategy_total_return": _safe_float(perf_s.get("total_return")),
                "strategy_ann_return": _safe_float(perf_s.get("ann_return")),
                "strategy_sharpe": _safe_float(perf_s.get("sharpe")),
                "benchmark_total_return": _safe_float(perf_b.get("total_return")),
                "benchmark_ann_return": _safe_float(perf_b.get("ann_return")),
                "alpha_total_return": _safe_float(perf_s.get("total_return")) - _safe_float(perf_b.get("total_return")),
                "alpha_ann_return": _safe_float(perf_s.get("ann_return")) - _safe_float(perf_b.get("ann_return")),
            }
        )
    return pd.DataFrame(rows)


def _sector_weights(weights: dict[str, float], asset_to_sector: dict[str, str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for asset, weight in weights.items():
        sector = str(asset_to_sector.get(asset, "unknown")).strip() or "unknown"
        out[sector] = float(out.get(sector, 0.0)) + float(weight)
    return out


def _monthly_structure(monthly_eval: pd.DataFrame, label: str, asset_to_sector: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    prev_weights: dict[str, float] = {}
    prev_cash = 1.0
    for _, row in monthly_eval.iterrows():
        ym = str(row.get("ym", ""))
        weights = _json_weight_map(row.get("executed_weights_json", "{}"))
        cash_weight = float(max(0.0, _safe_float(row.get("cash_weight"), 0.0)))
        sector_map = _sector_weights(weights, asset_to_sector)
        positive_weights = np.array([max(0.0, float(v)) for v in weights.values()], dtype=float)
        positive_weights = positive_weights[positive_weights > 1e-14]
        max_asset = float(positive_weights.max()) if positive_weights.size else 0.0
        effective_n = float(1.0 / np.square(positive_weights).sum()) if positive_weights.size else 0.0
        sector_weights = np.array([max(0.0, float(v)) for v in sector_map.values()], dtype=float)
        top_sector = float(sector_weights.max()) if sector_weights.size else 0.0
        sector_hhi = float(np.square(sector_weights).sum()) if sector_weights.size else 0.0
        turnover = float(_l1_turnover(prev_weights, prev_cash, weights, cash_weight))
        rows.append(
            {
                "profile": label,
                "ym": ym,
                "n_selected": int(len(weights)),
                "cash_weight": cash_weight,
                "gross_core_weight": float(sum(abs(float(v)) for v in weights.values())),
                "max_asset_weight": max_asset,
                "effective_n": effective_n,
                "sector_count": int(len(sector_map)),
                "top_sector_weight": top_sector,
                "sector_hhi": sector_hhi,
                "turnover_l1": turnover,
            }
        )
        prev_weights, prev_cash = dict(weights), float(cash_weight)
    return pd.DataFrame(rows)


def _structure_summary(monthly_structure: pd.DataFrame) -> dict[str, float]:
    if monthly_structure.empty:
        return {}
    out: dict[str, float] = {}
    for col in [
        "n_selected",
        "cash_weight",
        "gross_core_weight",
        "max_asset_weight",
        "effective_n",
        "sector_count",
        "top_sector_weight",
        "sector_hhi",
        "turnover_l1",
    ]:
        series = pd.to_numeric(monthly_structure[col], errors="coerce").dropna().astype(float)
        out[f"avg_{col}"] = float(series.mean()) if not series.empty else float("nan")
        out[f"p90_{col}"] = float(series.quantile(0.9)) if not series.empty else float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build extra discovery measures for profit-shadow candidates.")
    ap.add_argument("--lock-path", required=True)
    ap.add_argument("--candidate-monthly", required=True)
    ap.add_argument(
        "--meta-monthly",
        default=str(ROOT / "results" / "validation" / "profit_shadow_combo_validation" / "20260306T211751Z" / "causal_meta_switch_monthly_eval.csv"),
    )
    ap.add_argument("--prices-dir", default=str(ROOT / "data" / "raw" / "finance" / "yfinance_daily"))
    ap.add_argument("--outdir", default="")
    args = ap.parse_args()

    lock_path = Path(args.lock_path).resolve()
    lock = _read_json(lock_path)
    if not lock:
        raise SystemExit(f"missing lock: {lock_path}")
    outdir = _resolve_path(args.outdir) or (ROOT / "results" / "validation" / "profit_shadow_discovery_measures" / _run_id())
    outdir.mkdir(parents=True, exist_ok=True)
    prices_dir = _resolve_path(args.prices_dir)
    if prices_dir is None or not prices_dir.exists():
        raise SystemExit(f"missing prices_dir: {args.prices_dir}")

    main_ctx = _build_candidate_context(lock.get("main", {}))
    challenger_ctx = _build_candidate_context(lock.get("challenger", {}))
    benchmark_symbol = str(main_ctx["benchmark_symbol"])
    returns_wide = main_ctx["returns_wide"]
    benchmark_series = _load_price_returns(prices_dir, benchmark_symbol)
    if benchmark_series.empty:
        benchmark_series = pd.Series(np.zeros(len(returns_wide), dtype=float), index=returns_wide.index, dtype=float)

    combined_asset_to_sector = dict(main_ctx.get("asset_to_sector", {}))
    combined_asset_to_sector.update(challenger_ctx.get("asset_to_sector", {}))

    profiles = {
        "new_combo": _load_monthly_eval(Path(args.candidate_monthly).resolve()),
        "causal_meta_switch": _load_monthly_eval(Path(args.meta_monthly).resolve()),
        "main": main_ctx["monthly"].copy(),
        "challenger": challenger_ctx["monthly"].copy(),
    }

    structure_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    slice_frames: list[pd.DataFrame] = []
    yearly_frames: list[pd.DataFrame] = []
    scenario_map: dict[str, dict[str, pd.DataFrame]] = {}

    for label, monthly_eval in profiles.items():
        structure = _monthly_structure(monthly_eval, label, combined_asset_to_sector)
        structure.to_csv(outdir / f"{label}_monthly_structure.csv", index=False)
        structure_rows.append({"profile": label, **_structure_summary(structure)})
        scenario_map[label] = {}
        for scenario in ["delay_d0", "monthly_10bps"]:
            history = _history_for_scenario(
                monthly_eval=monthly_eval,
                returns_wide=returns_wide,
                benchmark_symbol=benchmark_symbol,
                benchmark_series=benchmark_series,
                scenario=scenario,
            )
            scenario_map[label][scenario] = history
            if history.empty:
                continue
            history.to_csv(outdir / f"{label}_{scenario}_daily.csv", index=False)
            strat = pd.to_numeric(history["strategy_return"], errors="coerce").fillna(0.0).astype(float)
            bench = pd.to_numeric(history["benchmark_return"], errors="coerce").fillna(0.0).astype(float)
            perf = _perf_from_simple_returns(strat, periods_per_year=252.0)
            rel = _beta_and_corr(strat, bench)
            tail = _tail_alpha(strat, bench, q=0.05)
            dd = _drawdown_duration_stats(strat)
            profile_rows.append(
                {
                    "profile": label,
                    "scenario": scenario,
                    "ann_return": _safe_float(perf.get("ann_return")),
                    "sharpe": _safe_float(perf.get("sharpe")),
                    "max_drawdown": _safe_float(perf.get("max_drawdown")),
                    "total_return": _safe_float(perf.get("total_return")),
                    "positive_share": _safe_float(perf.get("positive_share")),
                    "beta": _safe_float(rel.get("beta")),
                    "corr": _safe_float(rel.get("corr")),
                    "upside_capture": _safe_float(_capture_ratio(strat, bench, positive=True)),
                    "downside_capture": _safe_float(_capture_ratio(strat, bench, positive=False)),
                    "rolling_outperf_63d_share": _safe_float(_rolling_outperformance_share(strat, bench, 63)),
                    "rolling_outperf_126d_share": _safe_float(_rolling_outperformance_share(strat, bench, 126)),
                    "rolling_outperf_252d_share": _safe_float(_rolling_outperformance_share(strat, bench, 252)),
                    **tail,
                    **dd,
                }
            )
            slice_frames.append(_slice_summary(history, label, scenario))
            yearly_frames.append(_yearly_returns(history, label, scenario))

    structure_df = pd.DataFrame(structure_rows).sort_values("profile").reset_index(drop=True)
    profile_df = pd.DataFrame(profile_rows).sort_values(["scenario", "ann_return"], ascending=[True, False]).reset_index(drop=True)
    slices_df = pd.concat(slice_frames, ignore_index=True) if slice_frames else pd.DataFrame()
    years_df = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()

    structure_df.to_csv(outdir / "structure_summary.csv", index=False)
    profile_df.to_csv(outdir / "scenario_profile_metrics.csv", index=False)
    slices_df.to_csv(outdir / "market_slice_compare.csv", index=False)
    years_df.to_csv(outdir / "calendar_year_compare.csv", index=False)

    findings: list[str] = []
    if not profile_df.empty:
        for scenario in ["delay_d0", "monthly_10bps"]:
            sub = profile_df[profile_df["scenario"] == scenario].sort_values(["ann_return", "sharpe"], ascending=False)
            if not sub.empty:
                best = sub.iloc[0]
                findings.append(
                    f"{scenario}: melhor ann_return = {best['profile']} ({best['ann_return']:.4f}, sharpe {best['sharpe']:.3f})"
                )
    if not profile_df.empty and set(["new_combo", "causal_meta_switch"]).issubset(set(profile_df["profile"].astype(str))):
        for scenario in ["delay_d0", "monthly_10bps"]:
            sub = profile_df[profile_df["scenario"] == scenario].set_index("profile")
            if {"new_combo", "causal_meta_switch"}.issubset(set(sub.index)):
                delta_ann = _safe_float(sub.loc["new_combo", "ann_return"]) - _safe_float(sub.loc["causal_meta_switch", "ann_return"])
                delta_beta = _safe_float(sub.loc["new_combo", "beta"]) - _safe_float(sub.loc["causal_meta_switch", "beta"])
                findings.append(f"{scenario}: new_combo vs meta ann_delta={delta_ann:.4f}, beta_delta={delta_beta:.4f}")
    if not slices_df.empty:
        for scenario in ["delay_d0", "monthly_10bps"]:
            sub = slices_df[(slices_df["scenario"] == scenario) & (slices_df["profile"] == "new_combo")]
            if sub.empty:
                continue
            bull = sub[sub["slice"] == "bull"]
            bear = sub[sub["slice"] == "bear"]
            if not bull.empty:
                findings.append(f"{scenario}: new_combo bull alpha_ann={float(bull.iloc[0]['alpha_ann_return']):.4f}")
            if not bear.empty:
                findings.append(f"{scenario}: new_combo bear alpha_ann={float(bear.iloc[0]['alpha_ann_return']):.4f}")

    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "lock_path": str(lock_path),
        "candidate_monthly": str(Path(args.candidate_monthly).resolve()),
        "meta_monthly": str(Path(args.meta_monthly).resolve()),
        "profiles": list(profiles.keys()),
        "best_by_scenario": {
            scenario: (
                profile_df[profile_df["scenario"] == scenario]
                .sort_values(["ann_return", "sharpe"], ascending=False)
                .head(1)
                .iloc[0]
                .to_dict()
                if not profile_df[profile_df["scenario"] == scenario].empty
                else {}
            )
            for scenario in ["delay_d0", "monthly_10bps"]
        },
        "findings": findings,
        "artifacts": {
            "structure_summary_csv": str(outdir / "structure_summary.csv"),
            "scenario_profile_metrics_csv": str(outdir / "scenario_profile_metrics.csv"),
            "market_slice_compare_csv": str(outdir / "market_slice_compare.csv"),
            "calendar_year_compare_csv": str(outdir / "calendar_year_compare.csv"),
        },
    }
    _write_json(outdir / "summary.json", _sanitize_json_value(summary))
    print(json.dumps(_sanitize_json_value(summary), ensure_ascii=False))


if __name__ == "__main__":
    main()
