#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.structural.run_manifest import write_run_manifest  # noqa: E402
from scripts.bench.validation.run_profit_alpha_hardening_suite import (  # noqa: E402
    AllocationBundle,
    _build_candidates,
)
from scripts.bench.validation.run_profit_frontier_expansion_suite import _write_json  # noqa: E402
from scripts.bench.validation.run_profit_sector_pressure_suite import _research_row  # noqa: E402
from scripts.bench.validation.run_profit_shadow_realism_battery import classify_market_slices  # noqa: E402


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _perf_from_simple_returns(simple_returns: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(simple_returns, errors="coerce").dropna().astype(float)
    if x.empty:
        return {
            "total_return": float("nan"),
            "ann_return": float("nan"),
            "ann_vol": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
        }
    eq = (1.0 + x).clip(lower=1e-9, upper=10.0).cumprod()
    ann_return = float(np.power(float(eq.iloc[-1]), 252.0 / max(int(x.shape[0]), 1)) - 1.0)
    ann_vol = float(x.std(ddof=0) * np.sqrt(252.0))
    return {
        "total_return": float(eq.iloc[-1] - 1.0),
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": float(ann_return / ann_vol) if ann_vol > 1e-12 else float("nan"),
        "max_drawdown": float((eq / eq.cummax() - 1.0).min()),
    }


def _first_of_period(index: pd.DatetimeIndex, periods: pd.Index) -> pd.Series:
    mask = np.zeros(len(index), dtype=bool)
    prev = None
    for idx, period in enumerate(periods):
        if period != prev:
            mask[idx] = True
            prev = period
    return pd.Series(mask, index=index, dtype=bool)


def build_rebalance_mask(index: pd.DatetimeIndex, frequency: str) -> pd.Series:
    freq = str(frequency).strip().lower()
    if freq == "daily":
        return pd.Series(True, index=index, dtype=bool)
    if freq == "weekly":
        return _first_of_period(index, index.to_period("W-MON"))
    if freq == "biweekly":
        weekly = _first_of_period(index, index.to_period("W-MON"))
        points = weekly[weekly].index
        out = pd.Series(False, index=index, dtype=bool)
        out.loc[points[::2]] = True
        return out
    if freq == "monthly":
        return _first_of_period(index, index.to_period("M"))
    raise ValueError(f"unsupported frequency: {frequency}")


def simulate_allocation_execution(
    *,
    allocation: AllocationBundle,
    sleeve_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    rebalance_frequency: str,
    delay_days: int,
    extra_cost_bps: float,
    extra_spread_bps: float,
    extra_slippage_bps: float,
    initial_capital: float,
) -> pd.DataFrame:
    idx = (
        allocation.weights.index.intersection(sleeve_returns.index)
        .intersection(benchmark_returns.index)
        .sort_values()
    )
    if idx.empty:
        return pd.DataFrame()

    weights = allocation.weights.reindex(idx).fillna(0.0).astype(float)
    sleeves = sleeve_returns.reindex(idx).fillna(0.0).astype(float)
    benchmark = pd.to_numeric(benchmark_returns.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    rebalance_mask = build_rebalance_mask(pd.DatetimeIndex(idx), rebalance_frequency)
    base_cost_bps = float(allocation.bundle.profile.total_cost_bps_assumed)
    total_extra_bps = float(max(0.0, extra_cost_bps)) + float(max(0.0, extra_spread_bps)) + float(max(0.0, extra_slippage_bps))
    total_cost_rate = (base_cost_bps + total_extra_bps) / 10000.0

    current = pd.Series({"crypto": 0.0, "equity": 0.0, "cash": 1.0}, dtype=float)
    capital = float(initial_capital)
    benchmark_capital = float(initial_capital)
    rows: list[dict[str, Any]] = []
    delay = int(max(0, delay_days))

    for pos, dt in enumerate(idx):
        turnover = 0.0
        signal_pos = pos - delay
        if bool(rebalance_mask.loc[dt]):
            if signal_pos >= 0:
                target = weights.iloc[signal_pos][["crypto", "equity", "cash"]].astype(float)
            else:
                target = pd.Series({"crypto": 0.0, "equity": 0.0, "cash": 1.0}, dtype=float)
            target = target.clip(lower=0.0)
            total_target = float(target.sum())
            if total_target > 1e-12:
                target = target / total_target
            else:
                target.loc["cash"] = 1.0
            turnover = 0.5 * float((current - target).abs().sum())
            current = target

        gross_ret = float(
            current.get("crypto", 0.0) * float(sleeves.loc[dt, "crypto"])
            + current.get("equity", 0.0) * float(sleeves.loc[dt, "equity"])
        )
        cost_ret = float(turnover * total_cost_rate)
        net_ret = float(gross_ret - cost_ret)
        bench_ret = float(benchmark.loc[dt])
        capital *= 1.0 + net_ret
        benchmark_capital *= 1.0 + bench_ret

        post = pd.Series(
            {
                "crypto": float(current.get("crypto", 0.0)) * (1.0 + float(sleeves.loc[dt, "crypto"])),
                "equity": float(current.get("equity", 0.0)) * (1.0 + float(sleeves.loc[dt, "equity"])),
                "cash": float(current.get("cash", 0.0)),
            },
            dtype=float,
        )
        denom = float(post.sum())
        if denom > 1e-12:
            current = post / denom
        else:
            current = pd.Series({"crypto": 0.0, "equity": 0.0, "cash": 1.0}, dtype=float)

        rows.append(
            {
                "date": pd.Timestamp(dt).date().isoformat(),
                "year": int(pd.Timestamp(dt).year),
                "rebalance_frequency": str(rebalance_frequency),
                "delay_days": int(delay),
                "extra_cost_bps": float(extra_cost_bps),
                "extra_spread_bps": float(extra_spread_bps),
                "extra_slippage_bps": float(extra_slippage_bps),
                "turnover": float(turnover),
                "gross_return": float(gross_ret),
                "net_return": float(net_ret),
                "benchmark_return": float(bench_ret),
                "capital": float(capital),
                "benchmark_capital": float(benchmark_capital),
                "operation_day": int(turnover > 1e-10),
                "crypto_weight_post": float(current.get("crypto", 0.0)),
                "equity_weight_post": float(current.get("equity", 0.0)),
                "cash_weight_post": float(current.get("cash", 0.0)),
                "source_signal": str(allocation.source.reindex(idx).fillna("cash").iloc[max(signal_pos, 0)]) if len(idx) else "cash",
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["capital_peak"] = pd.to_numeric(out["capital"], errors="coerce").cummax()
    out["drawdown"] = pd.to_numeric(out["capital"], errors="coerce") / out["capital_peak"] - 1.0
    return out


def summarize_execution_history(history: pd.DataFrame) -> dict[str, Any]:
    if history.empty:
        return {"status": "empty"}
    perf = _perf_from_simple_returns(pd.to_numeric(history["net_return"], errors="coerce"))
    bench = _perf_from_simple_returns(pd.to_numeric(history["benchmark_return"], errors="coerce"))
    return {
        "status": "ok",
        "n_days": int(history.shape[0]),
        "portfolio_total_return": _safe_float(perf.get("total_return")),
        "portfolio_ann_return": _safe_float(perf.get("ann_return")),
        "portfolio_sharpe": _safe_float(perf.get("sharpe")),
        "portfolio_max_drawdown": _safe_float(perf.get("max_drawdown")),
        "benchmark_total_return": _safe_float(bench.get("total_return")),
        "benchmark_ann_return": _safe_float(bench.get("ann_return")),
        "edge_total_return": _safe_float(perf.get("total_return")) - _safe_float(bench.get("total_return")),
        "edge_ann_return": _safe_float(perf.get("ann_return")) - _safe_float(bench.get("ann_return")),
        "avg_turnover_daily": float(pd.to_numeric(history["turnover"], errors="coerce").fillna(0.0).mean()),
        "operation_days": int(pd.to_numeric(history["operation_day"], errors="coerce").fillna(0.0).sum()),
        "final_capital_brl": _safe_float(history.iloc[-1]["capital"]),
        "benchmark_final_capital_brl": _safe_float(history.iloc[-1]["benchmark_capital"]),
    }


def summarize_market_phases(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()
    idx = pd.to_datetime(history["date"], errors="coerce")
    bench = pd.to_numeric(history["benchmark_return"], errors="coerce")
    labels = classify_market_slices(pd.Series(bench.to_numpy(dtype=float), index=idx))
    rows: list[dict[str, Any]] = []
    for phase in ["bull", "bear", "recovery", "sideways"]:
        mask = labels == phase
        if not bool(mask.any()):
            continue
        strat = pd.to_numeric(history.loc[mask.to_numpy(dtype=bool), "net_return"], errors="coerce").fillna(0.0)
        bench_phase = bench.loc[mask.to_numpy(dtype=bool)].fillna(0.0)
        perf_s = _perf_from_simple_returns(strat)
        perf_b = _perf_from_simple_returns(bench_phase)
        rows.append(
            {
                "phase": phase,
                "n_days": int(mask.sum()),
                "share_days": float(mask.mean()),
                "strategy_total_return": _safe_float(perf_s.get("total_return")),
                "strategy_ann_return": _safe_float(perf_s.get("ann_return")),
                "strategy_sharpe": _safe_float(perf_s.get("sharpe")),
                "strategy_max_drawdown": _safe_float(perf_s.get("max_drawdown")),
                "benchmark_total_return": _safe_float(perf_b.get("total_return")),
                "benchmark_ann_return": _safe_float(perf_b.get("ann_return")),
                "alpha_total_return": _safe_float(perf_s.get("total_return")) - _safe_float(perf_b.get("total_return")),
                "alpha_ann_return": _safe_float(perf_s.get("ann_return")) - _safe_float(perf_b.get("ann_return")),
            }
        )
    return pd.DataFrame(rows)


def calendar_year_rows(history: pd.DataFrame, *, candidate_id: str, candidate_label: str, scenario: str) -> list[dict[str, Any]]:
    if history.empty:
        return []
    out: list[dict[str, Any]] = []
    years = sorted({int(y) for y in pd.to_numeric(history["year"], errors="coerce").dropna().astype(int).tolist()})
    for year in years:
        part = history[pd.to_numeric(history["year"], errors="coerce").astype(int) == int(year)].copy()
        if part.empty:
            continue
        perf = _perf_from_simple_returns(pd.to_numeric(part["net_return"], errors="coerce"))
        bench = _perf_from_simple_returns(pd.to_numeric(part["benchmark_return"], errors="coerce"))
        start_cap = float(part.iloc[0]["capital"] / (1.0 + float(part.iloc[0]["net_return"])))
        end_cap = float(part.iloc[-1]["capital"])
        out.append(
            {
                "candidate_id": candidate_id,
                "candidate_label": candidate_label,
                "scenario": scenario,
                "year": int(year),
                "start_capital_brl": start_cap,
                "end_capital_brl": end_cap,
                "profit_brl": end_cap - start_cap,
                "strategy_total_return": _safe_float(perf.get("total_return")),
                "strategy_ann_return": _safe_float(perf.get("ann_return")),
                "strategy_sharpe": _safe_float(perf.get("sharpe")),
                "benchmark_total_return": _safe_float(bench.get("total_return")),
                "alpha_total_return": _safe_float(perf.get("total_return")) - _safe_float(bench.get("total_return")),
                "operation_days": int(pd.to_numeric(part["operation_day"], errors="coerce").fillna(0.0).sum()),
                "avg_turnover_daily": float(pd.to_numeric(part["turnover"], errors="coerce").fillna(0.0).mean()),
            }
        )
    return out


def scenario_grid() -> list[dict[str, Any]]:
    return [
        {"scenario": "base_daily", "rebalance_frequency": "daily", "delay_days": 0, "extra_cost_bps": 0.0, "extra_spread_bps": 0.0, "extra_slippage_bps": 0.0, "class": "base"},
        {"scenario": "daily_delay1", "rebalance_frequency": "daily", "delay_days": 1, "extra_cost_bps": 0.0, "extra_spread_bps": 0.0, "extra_slippage_bps": 0.0, "class": "delay"},
        {"scenario": "daily_delay2", "rebalance_frequency": "daily", "delay_days": 2, "extra_cost_bps": 0.0, "extra_spread_bps": 0.0, "extra_slippage_bps": 0.0, "class": "delay"},
        {"scenario": "weekly_realistic", "rebalance_frequency": "weekly", "delay_days": 1, "extra_cost_bps": 10.0, "extra_spread_bps": 5.0, "extra_slippage_bps": 5.0, "class": "realistic"},
        {"scenario": "biweekly_realistic", "rebalance_frequency": "biweekly", "delay_days": 1, "extra_cost_bps": 15.0, "extra_spread_bps": 8.0, "extra_slippage_bps": 8.0, "class": "realistic"},
        {"scenario": "monthly_realistic", "rebalance_frequency": "monthly", "delay_days": 1, "extra_cost_bps": 15.0, "extra_spread_bps": 10.0, "extra_slippage_bps": 10.0, "class": "realistic"},
        {"scenario": "weekly_hard", "rebalance_frequency": "weekly", "delay_days": 1, "extra_cost_bps": 20.0, "extra_spread_bps": 15.0, "extra_slippage_bps": 15.0, "class": "hard"},
        {"scenario": "biweekly_hard", "rebalance_frequency": "biweekly", "delay_days": 2, "extra_cost_bps": 25.0, "extra_spread_bps": 15.0, "extra_slippage_bps": 20.0, "class": "hard"},
        {"scenario": "monthly_hard", "rebalance_frequency": "monthly", "delay_days": 2, "extra_cost_bps": 25.0, "extra_spread_bps": 20.0, "extra_slippage_bps": 20.0, "class": "hard"},
        {"scenario": "weekly_brutal", "rebalance_frequency": "weekly", "delay_days": 2, "extra_cost_bps": 40.0, "extra_spread_bps": 25.0, "extra_slippage_bps": 25.0, "class": "brutal"},
        {"scenario": "monthly_brutal", "rebalance_frequency": "monthly", "delay_days": 2, "extra_cost_bps": 50.0, "extra_spread_bps": 30.0, "extra_slippage_bps": 30.0, "class": "brutal"},
    ]


def _human_label(candidate_id: str) -> str:
    mapping = {
        "meta_major8_eq_a2r1": "Modo principal de lucro",
        "alpha_attack_major8_equity25": "Modo ataque de lucro máximo",
        "meta_major8_eq_a2r1_mc_guard": "Modo principal com guarda de Monte Carlo",
        "alpha_attack_major8_equity25_mc_guard": "Modo ataque com guarda de Monte Carlo",
    }
    return mapping.get(str(candidate_id), str(candidate_id))


def main() -> None:
    ap = argparse.ArgumentParser(description="Grade dura de execução e simulação por fase de mercado para os finalistas de lucro.")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--capital-brl", type=float, default=10000.0)
    ap.add_argument("--outdir-root", default="results/validation/profit_execution_phase_suite")
    args = ap.parse_args()

    outdir = (ROOT / args.outdir_root / _run_id()).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    built = _build_candidates(
        prices_dir=(ROOT / args.prices_dir).resolve(),
        crypto_groups=(ROOT / args.crypto_asset_groups).resolve(),
        crypto_meta=(ROOT / args.crypto_asset_metadata).resolve(),
        equity_groups=(ROOT / args.equity_asset_groups).resolve(),
        equity_meta=(ROOT / args.equity_asset_metadata).resolve(),
        benchmark_crypto=str(args.benchmark_crypto),
        benchmark_equity=str(args.benchmark_equity),
    )

    allocations: dict[str, AllocationBundle] = built["allocations"]
    sleeve_returns: dict[str, pd.DataFrame] = built["sleeve_returns"]

    scenario_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    year_rows: list[dict[str, Any]] = []
    histories: dict[str, Path] = {}

    for allocation_key, allocation in allocations.items():
        candidate_id = str(allocation.bundle.result.candidate_id)
        bench = pd.to_numeric(allocation.bundle.benchmark_gross_ret, errors="coerce").fillna(0.0).astype(float)
        for scenario in scenario_grid():
            history = simulate_allocation_execution(
                allocation=allocation,
                sleeve_returns=sleeve_returns[allocation_key],
                benchmark_returns=bench,
                rebalance_frequency=str(scenario["rebalance_frequency"]),
                delay_days=int(scenario["delay_days"]),
                extra_cost_bps=float(scenario["extra_cost_bps"]),
                extra_spread_bps=float(scenario["extra_spread_bps"]),
                extra_slippage_bps=float(scenario["extra_slippage_bps"]),
                initial_capital=float(args.capital_brl),
            )
            summary = summarize_execution_history(history)
            scenario_rows.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_label": _human_label(candidate_id),
                    **scenario,
                    **summary,
                }
            )
            if not history.empty:
                hist_path = outdir / f"history__{candidate_id}__{scenario['scenario']}.csv"
                history.to_csv(hist_path, index=False)
                histories[f"{candidate_id}::{scenario['scenario']}"] = hist_path
            phase_df = summarize_market_phases(history)
            if not phase_df.empty:
                phase_df.insert(0, "scenario", str(scenario["scenario"]))
                phase_df.insert(0, "candidate_label", _human_label(candidate_id))
                phase_df.insert(0, "candidate_id", candidate_id)
                phase_rows.extend(phase_df.to_dict(orient="records"))
            year_rows.extend(
                calendar_year_rows(
                    history,
                    candidate_id=candidate_id,
                    candidate_label=_human_label(candidate_id),
                    scenario=str(scenario["scenario"]),
                )
            )

    scenario_df = pd.DataFrame(scenario_rows).sort_values(
        ["scenario", "portfolio_total_return", "portfolio_ann_return"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    phase_df = pd.DataFrame(phase_rows).sort_values(
        ["scenario", "phase", "alpha_total_return"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    year_df = pd.DataFrame(year_rows).sort_values(
        ["year", "scenario", "profit_brl"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    scenario_df.to_csv(outdir / "execution_grid.csv", index=False)
    phase_df.to_csv(outdir / "market_phase_compare.csv", index=False)
    year_df.to_csv(outdir / "calendar_year_compare.csv", index=False)

    realistic_df = scenario_df[scenario_df["class"].astype(str) != "base"].copy()
    realistic_score = (
        realistic_df.groupby(["candidate_id", "candidate_label"], as_index=False)
        .agg(
            realistic_scenarios=("scenario", "count"),
            mean_total_return=("portfolio_total_return", "mean"),
            mean_ann_return=("portfolio_ann_return", "mean"),
            mean_sharpe=("portfolio_sharpe", "mean"),
            worst_drawdown=("portfolio_max_drawdown", "min"),
            positive_realistic_scenarios=("edge_total_return", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0.0) > 0.0).sum())),
        )
        .sort_values(["mean_total_return", "mean_ann_return", "mean_sharpe"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    realistic_score.to_csv(outdir / "candidate_score.csv", index=False)

    annual_winners = (
        year_df[year_df["scenario"].astype(str).isin(realistic_df["scenario"].astype(str).unique())]
        .groupby(["year", "candidate_id", "candidate_label"], as_index=False)
        .agg(
            mean_profit_brl=("profit_brl", "mean"),
            mean_alpha_total_return=("alpha_total_return", "mean"),
            mean_ann_return=("strategy_ann_return", "mean"),
            mean_operation_days=("operation_days", "mean"),
        )
        .sort_values(["year", "mean_profit_brl", "mean_ann_return"], ascending=[True, False, False])
        .groupby("year", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    annual_winners.to_csv(outdir / "annual_winners.csv", index=False)

    phase_winners = (
        phase_df.groupby(["phase", "candidate_id", "candidate_label"], as_index=False)
        .agg(
            mean_alpha_total_return=("alpha_total_return", "mean"),
            mean_alpha_ann_return=("alpha_ann_return", "mean"),
            mean_strategy_ann_return=("strategy_ann_return", "mean"),
        )
        .sort_values(["phase", "mean_alpha_total_return", "mean_strategy_ann_return"], ascending=[True, False, False])
        .groupby("phase", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    phase_winners.to_csv(outdir / "phase_winners.csv", index=False)

    base_winner = scenario_df[scenario_df["scenario"].astype(str) == "base_daily"].head(1)
    realistic_winner = realistic_score.head(1)
    latest_year = int(year_df["year"].max()) if not year_df.empty else None
    recent_year_winner = annual_winners[annual_winners["year"] == latest_year].head(1) if latest_year is not None else pd.DataFrame()

    research_rows = []
    for allocation_key, allocation in allocations.items():
        candidate_id = str(allocation.bundle.result.candidate_id)
        candidate_realistic = realistic_df[realistic_df["candidate_id"].astype(str) == str(candidate_id)].copy()
        if candidate_realistic.empty:
            continue
        best_row = candidate_realistic.sort_values(
            ["portfolio_total_return", "portfolio_ann_return", "portfolio_sharpe"],
            ascending=[False, False, False],
        ).iloc[0]
        result = replace(
            allocation.bundle.result,
            net_ann_return=float(best_row["portfolio_ann_return"]),
            net_total_return=float(best_row["portfolio_total_return"]),
            net_sharpe=float(best_row["portfolio_sharpe"]),
            net_max_drawdown=float(best_row["portfolio_max_drawdown"]),
            edge_vs_benchmark=float(best_row["edge_total_return"]),
        )
        research_rows.append(
            _research_row(
                result,
                outdir=outdir,
                status="keep" if str(candidate_id) == str(realistic_winner.iloc[0]["candidate_id"]) else "watch",
                methodology="alpha_execution_phase_suite",
                label=f"{_human_label(candidate_id)} sob atrito realista",
            )
        )
    (outdir / "profit_research_rows.json").write_text(json.dumps(research_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "outdir": str(outdir),
        "inputs": {
            "capital_brl": float(args.capital_brl),
            "benchmark_crypto": str(args.benchmark_crypto),
            "benchmark_equity": str(args.benchmark_equity),
            "scenarios": scenario_grid(),
        },
        "base_winner": base_winner.iloc[0].to_dict() if not base_winner.empty else {},
        "realistic_execution_winner": realistic_winner.iloc[0].to_dict() if not realistic_winner.empty else {},
        "recent_year_winner": recent_year_winner.iloc[0].to_dict() if not recent_year_winner.empty else {},
        "phase_winners": phase_winners.to_dict(orient="records"),
        "annual_winners": annual_winners.to_dict(orient="records"),
        "insights": [
            "Todos os finalistas foram testados com atraso, custo, spread, slippage e frequências diferentes de rebalanceamento sem reotimização no meio do caminho.",
            "A grade dura mostra se o modo de lucro máximo continua vivo quando o mundo real atrapalha.",
            "A leitura por fase de mercado separa onde cada modo realmente ganha dinheiro e onde só sobrevive.",
        ],
        "artifacts": {
            "execution_grid_csv": str(outdir / "execution_grid.csv"),
            "market_phase_compare_csv": str(outdir / "market_phase_compare.csv"),
            "calendar_year_compare_csv": str(outdir / "calendar_year_compare.csv"),
            "candidate_score_csv": str(outdir / "candidate_score.csv"),
            "annual_winners_csv": str(outdir / "annual_winners.csv"),
            "phase_winners_csv": str(outdir / "phase_winners.csv"),
            "profit_research_rows_json": str(outdir / "profit_research_rows.json"),
        },
    }
    _write_json(outdir / "summary.json", summary)
    write_run_manifest(
        outdir,
        script="scripts/bench/validation/run_profit_execution_phase_suite.py",
        params=summary["inputs"],
        paths=summary["artifacts"],
        extra={"summary_json": str(outdir / "summary.json")},
    )


if __name__ == "__main__":
    main()
