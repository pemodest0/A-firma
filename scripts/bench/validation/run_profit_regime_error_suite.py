#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.structural.run_manifest import write_run_manifest  # noqa: E402
from scripts.bench.validation.run_profit_alpha_hardening_suite import _build_candidates  # noqa: E402
from scripts.bench.validation.run_profit_frontier_expansion_suite import (  # noqa: E402
    _evaluate_net,
    _run_id,
    _safe_float,
    _write_json,
)


RISK_ON = {"stable", "dispersion"}
REGIME_ERROR_MAP = {
    "stable": "transition",
    "dispersion": "stress",
    "transition": "stable",
    "stress": "dispersion",
}


def _normalize_regime(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"stable", "dispersion", "transition", "stress"}:
        return raw
    return "transition"


def _normalize_regime_series(series: pd.Series, idx: pd.Index) -> pd.Series:
    out = pd.Series(series, index=series.index).reindex(idx).ffill().bfill()
    return out.map(_normalize_regime).astype(str)


def _delay_regime(series: pd.Series, delay_days: int) -> pd.Series:
    if delay_days <= 0:
        return series.copy()
    delayed = series.shift(int(delay_days))
    first = str(series.iloc[0]) if len(series) else "transition"
    return delayed.ffill().fillna(first).astype(str)


def _flip_regime(series: pd.Series, flip_frac: float, seed: int) -> pd.Series:
    if flip_frac <= 0.0:
        return series.copy()
    rng = np.random.default_rng(int(seed))
    out = series.copy()
    mask = rng.random(len(out)) < float(flip_frac)
    for dt in out.index[mask]:
        out.loc[dt] = REGIME_ERROR_MAP.get(str(out.loc[dt]), "transition")
    return out.astype(str)


def _selector_choice(series: pd.Series) -> pd.Series:
    return series.map(lambda state: "attack" if str(state).lower() in RISK_ON else "protect").astype(str)


def _selector_with_inertia(choice: pd.Series, *, min_hold_days: int = 0, confirm_days: int = 1) -> pd.Series:
    base = choice.astype(str)
    if base.empty:
        return base
    hold = max(0, int(min_hold_days))
    confirm = max(1, int(confirm_days))
    out = pd.Series(index=base.index, dtype=object)
    current = str(base.iloc[0])
    candidate = current
    candidate_streak = 0
    held_days = 0
    for dt, proposed in base.items():
        proposed = str(proposed)
        if proposed == current:
            candidate = current
            candidate_streak = 0
            held_days += 1
            out.loc[dt] = current
            continue
        if proposed != candidate:
            candidate = proposed
            candidate_streak = 1
        else:
            candidate_streak += 1
        if held_days >= hold and candidate_streak >= confirm:
            current = proposed
            held_days = 1
            candidate = current
            candidate_streak = 0
        else:
            held_days += 1
        out.loc[dt] = current
    return out.astype(str)


def _combine_candidate(
    *,
    candidate_id: str,
    choice: pd.Series,
    attack_alloc,
    protect_alloc,
    profile,
) -> dict[str, Any]:
    attack_gross = pd.to_numeric(attack_alloc.bundle.result.gross_ret, errors="coerce").astype(float)
    protect_gross = pd.to_numeric(protect_alloc.bundle.result.gross_ret, errors="coerce").astype(float)
    idx = (
        attack_gross.index.intersection(protect_gross.index)
        .intersection(choice.index)
        .intersection(attack_alloc.weights.index)
        .intersection(protect_alloc.weights.index)
    )
    attack_gross = attack_gross.reindex(idx).fillna(0.0)
    protect_gross = protect_gross.reindex(idx).fillna(0.0)
    decision = choice.reindex(idx).ffill().bfill().astype(str)

    gross = pd.Series(
        np.where(decision.eq("attack"), attack_gross.to_numpy(dtype=float), protect_gross.to_numpy(dtype=float)),
        index=idx,
        dtype=float,
    )

    attack_w = attack_alloc.weights.reindex(idx).fillna(0.0).astype(float)
    protect_w = protect_alloc.weights.reindex(idx).fillna(0.0).astype(float)
    selected_weights = attack_w.where(decision.eq("attack"), protect_w)
    turnover = (
        selected_weights[["crypto", "equity", "cash"]]
        .diff()
        .abs()
        .sum(axis=1)
        .fillna(selected_weights[["crypto", "equity", "cash"]].abs().sum(axis=1))
        / 2.0
    )

    benchmark = (
        pd.to_numeric(attack_alloc.bundle.benchmark_gross_ret, errors="coerce")
        .reindex(idx)
        .fillna(pd.to_numeric(protect_alloc.bundle.benchmark_gross_ret, errors="coerce").reindex(idx))
        .fillna(0.0)
        .astype(float)
    )
    perf = _evaluate_net(
        gross_ret=gross,
        turnover=turnover,
        profile=profile,
        benchmark_ret=benchmark,
        benchmark_profile=profile,
    )
    month = pd.DataFrame({"net": perf["net_ret"], "bench": perf["benchmark_net_ret"]}).resample("ME").apply(
        lambda frame: pd.Series(
            {
                "net": float(np.prod(1.0 + frame["net"].to_numpy(dtype=float)) - 1.0),
                "bench": float(np.prod(1.0 + frame["bench"].to_numpy(dtype=float)) - 1.0),
            }
        )
    )
    quarter = pd.DataFrame({"net": perf["net_ret"], "bench": perf["benchmark_net_ret"]}).resample("QE").apply(
        lambda frame: pd.Series(
            {
                "net": float(np.prod(1.0 + frame["net"].to_numpy(dtype=float)) - 1.0),
                "bench": float(np.prod(1.0 + frame["bench"].to_numpy(dtype=float)) - 1.0),
            }
        )
    )
    return {
        "candidate_id": candidate_id,
        "gross_ret": gross,
        "turnover": turnover,
        "net_ret": perf["net_ret"],
        "benchmark_net_ret": perf["benchmark_net_ret"],
        "net_ann_return": _safe_float(perf["net_ann_return"]),
        "net_total_return": _safe_float(perf["net_total_return"]),
        "net_sharpe": _safe_float(perf["net_sharpe"]),
        "net_max_drawdown": _safe_float(perf["net_max_drawdown"]),
        "edge_vs_benchmark": _safe_float(perf["edge_vs_benchmark"]),
        "avg_turnover_daily": _safe_float(perf["avg_turnover_daily"]),
        "month_wins": int((month["net"] > month["bench"]).sum()),
        "month_losses": int((month["net"] <= month["bench"]).sum()),
        "quarter_wins": int((quarter["net"] > quarter["bench"]).sum()),
        "quarter_losses": int((quarter["net"] <= quarter["bench"]).sum()),
        "selected_weights": selected_weights,
        "selected_mode": decision,
    }


def _year_rows(candidate: dict[str, Any], notional_brl: float) -> list[dict[str, Any]]:
    net_ret = pd.to_numeric(candidate["net_ret"], errors="coerce").dropna().astype(float)
    bench_ret = pd.to_numeric(candidate["benchmark_net_ret"], errors="coerce").reindex(net_ret.index).fillna(0.0).astype(float)
    turnover = pd.to_numeric(candidate["turnover"], errors="coerce").reindex(net_ret.index).fillna(0.0).astype(float)
    if net_ret.empty:
        return []
    wealth = (1.0 + net_ret).cumprod()
    rows: list[dict[str, Any]] = []
    for year, sub in net_ret.groupby(net_ret.index.year):
        idx = sub.index
        bench_sub = bench_ret.loc[idx]
        turn_sub = turnover.loc[idx]
        year_total = float(np.prod(1.0 + sub.to_numpy(dtype=float)) - 1.0)
        bench_total = float(np.prod(1.0 + bench_sub.to_numpy(dtype=float)) - 1.0)
        start_wealth = float(wealth.shift(1).reindex(idx).ffill().iloc[0]) if idx[0] != wealth.index[0] else 1.0
        end_wealth = float(wealth.loc[idx[-1]])
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "year": int(year),
                "days": int(len(idx)),
                "year_total_return": year_total,
                "benchmark_total_return": bench_total,
                "edge_total_return": year_total - bench_total,
                "profit_brl_rebased_10000": year_total * float(notional_brl),
                "running_profit_brl_10000": (end_wealth - start_wealth) * float(notional_brl),
                "operation_days": int((turn_sub > 1e-8).sum()),
                "avg_turnover_daily": float(turn_sub.mean()) if len(turn_sub) else 0.0,
            }
        )
    return rows


def _period_win_loss(net_ret: pd.Series, bench_ret: pd.Series, freq: str) -> tuple[int, int]:
    frame = pd.DataFrame({"net": net_ret, "bench": bench_ret}).dropna(how="all")
    if frame.empty:
        return 0, 0
    rolled = frame.resample(freq).apply(
        lambda part: pd.Series(
            {
                "net": float(np.prod(1.0 + part["net"].fillna(0.0).to_numpy(dtype=float)) - 1.0),
                "bench": float(np.prod(1.0 + part["bench"].fillna(0.0).to_numpy(dtype=float)) - 1.0),
            }
        )
    )
    wins = int((rolled["net"] > rolled["bench"]).sum())
    losses = int((rolled["net"] <= rolled["bench"]).sum())
    return wins, losses


def main() -> None:
    ap = argparse.ArgumentParser(description="Mede quanto o lucro cai quando o regime estrutural atrasa ou erra.")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--capital-brl", type=float, default=10000.0)
    ap.add_argument("--outdir-root", default="results/validation/profit_regime_error_suite")
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
    attack_alloc = built["allocations"]["attack"]
    protect_alloc = built["allocations"]["baseline_guard"]
    profile = built["context"]["profiles"]["blended"]
    regime_series = _normalize_regime_series(
        pd.Series(built["context"]["regime_series"]),
        attack_alloc.weights.index.intersection(protect_alloc.weights.index),
    )

    scenarios: list[tuple[str, pd.Series]] = [
        ("selector_clean", regime_series),
        ("selector_delay_5", _delay_regime(regime_series, 5)),
        ("selector_delay_10", _delay_regime(regime_series, 10)),
        ("selector_delay_21", _delay_regime(regime_series, 21)),
        ("selector_flip_10", _flip_regime(regime_series, 0.10, 17)),
        ("selector_flip_20", _flip_regime(regime_series, 0.20, 23)),
        ("selector_delay10_flip10", _flip_regime(_delay_regime(regime_series, 10), 0.10, 29)),
    ]

    inertia_scenarios = [
        ("selector_inertia_hold3", _selector_with_inertia(_selector_choice(regime_series), min_hold_days=3, confirm_days=1)),
        ("selector_inertia_hold5", _selector_with_inertia(_selector_choice(regime_series), min_hold_days=5, confirm_days=1)),
        ("selector_inertia_hold10", _selector_with_inertia(_selector_choice(regime_series), min_hold_days=10, confirm_days=1)),
        ("selector_inertia_confirm3", _selector_with_inertia(_selector_choice(regime_series), min_hold_days=0, confirm_days=3)),
        ("selector_inertia_hold5_confirm3", _selector_with_inertia(_selector_choice(regime_series), min_hold_days=5, confirm_days=3)),
    ]

    results = [
        _combine_candidate(
            candidate_id=name,
            choice=_selector_choice(perturbed),
            attack_alloc=attack_alloc,
            protect_alloc=protect_alloc,
            profile=profile,
        )
        for name, perturbed in scenarios
    ]
    results.extend(
        [
            _combine_candidate(
                candidate_id=name,
                choice=decision,
                attack_alloc=attack_alloc,
                protect_alloc=protect_alloc,
                profile=profile,
            )
            for name, decision in inertia_scenarios
        ]
    )
    results.extend(
        [
            {
                "candidate_id": "attack_puro",
                "net_ret": built["attack"].result.net_ret,
                "benchmark_net_ret": built["attack"].result.benchmark_net_ret,
                "turnover": built["attack"].result.turnover,
                "net_ann_return": built["attack"].result.net_ann_return,
                "net_total_return": built["attack"].result.net_total_return,
                "net_sharpe": built["attack"].result.net_sharpe,
                "net_max_drawdown": built["attack"].result.net_max_drawdown,
                "edge_vs_benchmark": built["attack"].result.edge_vs_benchmark,
                "avg_turnover_daily": built["attack"].result.avg_turnover_daily,
                "month_wins": _period_win_loss(built["attack"].result.net_ret, built["attack"].result.benchmark_net_ret, "ME")[0],
                "month_losses": _period_win_loss(built["attack"].result.net_ret, built["attack"].result.benchmark_net_ret, "ME")[1],
                "quarter_wins": _period_win_loss(built["attack"].result.net_ret, built["attack"].result.benchmark_net_ret, "QE")[0],
                "quarter_losses": _period_win_loss(built["attack"].result.net_ret, built["attack"].result.benchmark_net_ret, "QE")[1],
            },
            {
                "candidate_id": "protecao_pura",
                "net_ret": built["baseline_guard"].result.net_ret,
                "benchmark_net_ret": built["baseline_guard"].result.benchmark_net_ret,
                "turnover": built["baseline_guard"].result.turnover,
                "net_ann_return": built["baseline_guard"].result.net_ann_return,
                "net_total_return": built["baseline_guard"].result.net_total_return,
                "net_sharpe": built["baseline_guard"].result.net_sharpe,
                "net_max_drawdown": built["baseline_guard"].result.net_max_drawdown,
                "edge_vs_benchmark": built["baseline_guard"].result.edge_vs_benchmark,
                "avg_turnover_daily": built["baseline_guard"].result.avg_turnover_daily,
                "month_wins": _period_win_loss(built["baseline_guard"].result.net_ret, built["baseline_guard"].result.benchmark_net_ret, "ME")[0],
                "month_losses": _period_win_loss(built["baseline_guard"].result.net_ret, built["baseline_guard"].result.benchmark_net_ret, "ME")[1],
                "quarter_wins": _period_win_loss(built["baseline_guard"].result.net_ret, built["baseline_guard"].result.benchmark_net_ret, "QE")[0],
                "quarter_losses": _period_win_loss(built["baseline_guard"].result.net_ret, built["baseline_guard"].result.benchmark_net_ret, "QE")[1],
            },
        ]
    )

    compare_rows = []
    year_rows = []
    clean_total = next((row["net_total_return"] for row in results if row["candidate_id"] == "selector_clean"), None)
    for row in results:
        compare_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "net_ann_return": row["net_ann_return"],
                "net_total_return": row["net_total_return"],
                "net_sharpe": row["net_sharpe"],
                "net_max_drawdown": row["net_max_drawdown"],
                "edge_vs_benchmark": row["edge_vs_benchmark"],
                "avg_turnover_daily": row["avg_turnover_daily"],
                "month_wins": row["month_wins"],
                "month_losses": row["month_losses"],
                "quarter_wins": row["quarter_wins"],
                "quarter_losses": row["quarter_losses"],
                "retention_vs_clean": (
                    float(row["net_total_return"]) / float(clean_total)
                    if clean_total not in (None, 0.0) and np.isfinite(clean_total)
                    else float("nan")
                ),
            }
        )
        year_rows.extend(_year_rows(row, float(args.capital_brl)))

    compare_df = pd.DataFrame(compare_rows).sort_values(["net_total_return", "net_ann_return"], ascending=False)
    year_df = pd.DataFrame(year_rows).sort_values(["candidate_id", "year"])
    compare_df.to_csv(outdir / "candidate_compare.csv", index=False)
    year_df.to_csv(outdir / "calendar_year_compare.csv", index=False)

    clean = compare_df.loc[compare_df["candidate_id"] == "selector_clean"].iloc[0].to_dict()
    delay = compare_df.loc[compare_df["candidate_id"] == "selector_delay_10"].iloc[0].to_dict()
    flip = compare_df.loc[compare_df["candidate_id"] == "selector_flip_20"].iloc[0].to_dict()
    inertia = compare_df.loc[compare_df["candidate_id"] == "selector_inertia_hold5"].iloc[0].to_dict()
    attack = compare_df.loc[compare_df["candidate_id"] == "attack_puro"].iloc[0].to_dict()
    protect = compare_df.loc[compare_df["candidate_id"] == "protecao_pura"].iloc[0].to_dict()

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_candidate": compare_df.iloc[0]["candidate_id"] if not compare_df.empty else "n/d",
        "clean_selector": clean,
        "delay_10d": delay,
        "flip_20pct": flip,
        "inertia_hold5": inertia,
        "attack_pure": attack,
        "protection_pure": protect,
        "verdict": {
            "selector_is_useful": bool(clean.get("net_total_return", 0.0) > protect.get("net_total_return", 0.0)),
            "delay_hurts_materially": bool(clean.get("retention_vs_clean", 1.0) - delay.get("retention_vs_clean", 1.0) > 0.10),
            "wrong_regime_hurts_materially": bool(clean.get("retention_vs_clean", 1.0) - flip.get("retention_vs_clean", 1.0) > 0.10),
            "small_inertia_helps": bool(inertia.get("net_total_return", 0.0) > clean.get("net_total_return", 0.0)),
            "summary": (
                "Quando o regime atrasa ou erra bastante, o lucro cai de forma relevante; "
                "o seletor entre ataque e proteção agrega valor, mas depende de um cérebro estrutural minimamente correto. "
                "Uma inércia curta pode ajudar a reduzir zigue-zague; inércia longa demais começa a atrapalhar."
            ),
        },
    }
    _write_json(outdir / "summary.json", summary)
    write_run_manifest(
        outdir,
        script="scripts/bench/validation/run_profit_regime_error_suite.py",
        params={
            "capital_brl": float(args.capital_brl),
            "benchmark_crypto": str(args.benchmark_crypto),
            "benchmark_equity": str(args.benchmark_equity),
        },
        paths={
            "candidate_compare": "candidate_compare.csv",
            "calendar_year_compare": "calendar_year_compare.csv",
            "summary": "summary.json",
        },
        extra={
            "suite": "profit_regime_error_suite",
            "best_candidate": summary["best_candidate"],
        },
    )


if __name__ == "__main__":
    main()
