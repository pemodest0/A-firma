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
    build_daily_replay_with_rebalance,
    summarize_replay,
)
from scripts.bench.validation.run_profit_shadow_realism_battery import (  # noqa: E402
    _build_candidate_context,
    _json_weight_map,
    _load_price_returns,
    _perf_from_simple_returns,
    _resolve_path,
    _safe_float,
    _weight_json,
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


def _month_labels(series: pd.Series) -> dict[str, str]:
    labels = classify_market_slices(series)
    if labels.empty:
        return {}
    by_month = (
        pd.DataFrame({"ym": labels.index.to_period("M").astype(str), "label": labels.astype(str)})
        .groupby("ym", as_index=True)["label"]
        .last()
    )
    return {str(k): str(v) for k, v in by_month.items()}


def _prev_month(ym: str) -> str:
    return (pd.Period(str(ym), freq="M") - 1).strftime("%Y-%m")


def _trailing_total(monthly_ret: pd.Series, end_pos: int, lookback: int) -> float:
    if end_pos <= 0:
        return float("nan")
    start = max(0, int(end_pos) - int(max(1, lookback)))
    window = pd.to_numeric(monthly_ret.iloc[start:end_pos], errors="coerce").fillna(0.0).astype(float)
    if window.empty:
        return float("nan")
    return float(np.prod(1.0 + window.to_numpy(dtype=float)) - 1.0)


def choose_meta_source(
    *,
    ym: str,
    pos: int,
    market_state_by_month: dict[str, str],
    main_ret: pd.Series,
    challenger_ret: pd.Series,
    lookback_months: int,
) -> str:
    prev_state = str(market_state_by_month.get(_prev_month(ym), "warmup")).strip().lower()
    if prev_state in {"bull", "recovery"}:
        return "challenger"
    if prev_state == "bear":
        return "main"
    if prev_state == "sideways":
        main_trail = _trailing_total(main_ret, pos, lookback_months)
        challenger_trail = _trailing_total(challenger_ret, pos, lookback_months)
        if np.isfinite(challenger_trail) and (not np.isfinite(main_trail) or challenger_trail > main_trail):
            return "challenger"
    return "main"


def _combine_rows(row_main: pd.Series, row_challenger: pd.Series, alpha_main: float) -> dict[str, Any]:
    wa = _json_weight_map(row_main.get("executed_weights_json", "{}"))
    wb = _json_weight_map(row_challenger.get("executed_weights_json", "{}"))
    weights: dict[str, float] = {}
    alpha = float(alpha_main)
    for asset in sorted(set(wa) | set(wb)):
        weight = alpha * float(wa.get(asset, 0.0)) + (1.0 - alpha) * float(wb.get(asset, 0.0))
        if abs(weight) > 1e-14:
            weights[str(asset)] = float(weight)
    return {
        "ym": str(row_main["ym"]),
        "risk_bucket": "blend",
        "source_candidate": "blend_50_50",
        "executed_weights_json": _weight_json(weights),
        "executed_assets": ",".join(sorted(weights.keys())),
        "selected_assets": ",".join(sorted(weights.keys())),
        "cash_weight": float(alpha * _safe_float(row_main.get("cash_weight"), 0.0) + (1.0 - alpha) * _safe_float(row_challenger.get("cash_weight"), 0.0)),
        "hedge_weight": float(alpha * _safe_float(row_main.get("hedge_weight"), 0.0) + (1.0 - alpha) * _safe_float(row_challenger.get("hedge_weight"), 0.0)),
        "n_selected": int(len(weights)),
    }


def build_combo_monthly(
    *,
    main_monthly: pd.DataFrame,
    challenger_monthly: pd.DataFrame,
    benchmark_daily: pd.Series,
    lookback_months: int,
) -> dict[str, pd.DataFrame]:
    main = main_monthly.copy().drop_duplicates(subset=["ym"], keep="last").sort_values("ym").reset_index(drop=True)
    challenger = challenger_monthly.copy().drop_duplicates(subset=["ym"], keep="last").sort_values("ym").reset_index(drop=True)
    common = sorted(set(main["ym"].astype(str)) & set(challenger["ym"].astype(str)))
    main = main[main["ym"].astype(str).isin(common)].copy().reset_index(drop=True)
    challenger = challenger[challenger["ym"].astype(str).isin(common)].copy().reset_index(drop=True)

    main_ret = pd.to_numeric(main["ret"], errors="coerce").fillna(0.0).astype(float)
    challenger_ret = pd.to_numeric(challenger["ret"], errors="coerce").fillna(0.0).astype(float)
    market_state = _month_labels(benchmark_daily)
    main_map = {str(row["ym"]): row for _, row in main.iterrows()}
    challenger_map = {str(row["ym"]): row for _, row in challenger.iterrows()}

    blend_rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    for pos, ym in enumerate(common):
        row_main = main_map[ym]
        row_challenger = challenger_map[ym]
        blend_rows.append(_combine_rows(row_main, row_challenger, alpha_main=0.5))
        source = choose_meta_source(
            ym=ym,
            pos=pos,
            market_state_by_month=market_state,
            main_ret=main_ret,
            challenger_ret=challenger_ret,
            lookback_months=lookback_months,
        )
        source_row = row_main if source == "main" else row_challenger
        payload = dict(source_row)
        payload["source_candidate"] = source
        meta_rows.append(payload)
    return {
        "main": main,
        "challenger": challenger,
        "blend_50_50": pd.DataFrame(blend_rows),
        "causal_meta_switch": pd.DataFrame(meta_rows),
    }


def _evaluate_delay(
    *,
    label: str,
    monthly_eval: pd.DataFrame,
    returns_wide: pd.DataFrame,
    benchmark_returns: pd.Series,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for delay in [0, 1]:
        history = build_daily_replay_with_delay(
            monthly_eval=monthly_eval,
            returns_wide=returns_wide,
            benchmark_returns=benchmark_returns,
            initial_capital=10000.0,
            execution_delay_days=delay,
        )
        perf = _perf_from_simple_returns(pd.to_numeric(history["portfolio_return"], errors="coerce"))
        bench = _perf_from_simple_returns(pd.to_numeric(history["benchmark_return"], errors="coerce"))
        rows.append(
            {
                "profile": label,
                "scenario": f"delay_d{delay}",
                "delay_days": int(delay),
                "cost_bps": 0.0,
                "rebalance_frequency": "native_monthly",
                "ann_return": _safe_float(perf.get("ann_return")),
                "sharpe": _safe_float(perf.get("sharpe")),
                "max_drawdown": _safe_float(perf.get("max_drawdown")),
                "total_return": _safe_float(perf.get("total_return")),
                "edge_total_return": _safe_float(perf.get("total_return")) - _safe_float(bench.get("total_return")),
            }
        )
    return rows


def _evaluate_cost(
    *,
    label: str,
    monthly_eval: pd.DataFrame,
    returns_wide: pd.DataFrame,
    benchmark_symbol: str,
    benchmark_returns: pd.Series,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scenarios = [("monthly", 10.0), ("monthly", 30.0), ("weekly", 30.0)]
    for freq, cost_bps in scenarios:
        history = build_daily_replay_with_rebalance(
            monthly_eval=monthly_eval,
            returns_wide=returns_wide,
            benchmark_symbol=benchmark_symbol,
            benchmark_returns=benchmark_returns,
            initial_capital=10000.0,
            cost_bps=float(cost_bps),
            rebalance_frequency=freq,
        )
        summary = summarize_replay(history, return_col="net_return")
        port = (summary.get("portfolio") or {}) if isinstance(summary.get("portfolio"), dict) else {}
        rows.append(
            {
                "profile": label,
                "scenario": f"{freq}_{int(cost_bps)}bps",
                "delay_days": 0,
                "cost_bps": float(cost_bps),
                "rebalance_frequency": freq,
                "ann_return": _safe_float(port.get("ann_return")),
                "sharpe": _safe_float(port.get("sharpe")),
                "max_drawdown": _safe_float(port.get("max_drawdown")),
                "total_return": _safe_float(port.get("total_return")),
                "edge_total_return": _safe_float(summary.get("edge_vs_benchmark_total_return")),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Test profit-shadow profile combinations under causal switching and blended execution.")
    ap.add_argument("--lock-path", required=True)
    ap.add_argument("--prices-dir", default=str(ROOT / "data" / "raw" / "finance" / "yfinance_daily"))
    ap.add_argument("--lookback-months", type=int, default=3)
    ap.add_argument("--outdir", default="")
    args = ap.parse_args()

    lock_path = Path(args.lock_path).resolve()
    lock = _read_json(lock_path)
    if not lock:
        raise SystemExit(f"missing lock: {lock_path}")
    outdir = _resolve_path(args.outdir) or (ROOT / "results" / "validation" / "profit_shadow_combo_validation" / _run_id())
    outdir.mkdir(parents=True, exist_ok=True)
    prices_dir = _resolve_path(args.prices_dir)
    if prices_dir is None or not prices_dir.exists():
        raise SystemExit(f"missing prices_dir: {args.prices_dir}")

    main_ctx = _build_candidate_context(lock.get("main", {}))
    challenger_ctx = _build_candidate_context(lock.get("challenger", {}))
    benchmark_symbol = str(main_ctx["benchmark_symbol"])
    benchmark_daily = _load_price_returns(prices_dir, benchmark_symbol)
    if benchmark_daily.empty:
        benchmark_daily = pd.Series(np.zeros(len(main_ctx["returns_wide"]), dtype=float), index=main_ctx["returns_wide"].index, dtype=float)

    combo_monthly = build_combo_monthly(
        main_monthly=main_ctx["monthly"],
        challenger_monthly=challenger_ctx["monthly"],
        benchmark_daily=benchmark_daily,
        lookback_months=int(args.lookback_months),
    )
    returns_wide = main_ctx["returns_wide"]

    results_rows: list[dict[str, Any]] = []
    for label, monthly_eval in combo_monthly.items():
        monthly_eval.to_csv(outdir / f"{label}_monthly_eval.csv", index=False)
        results_rows.extend(
            _evaluate_delay(
                label=label,
                monthly_eval=monthly_eval,
                returns_wide=returns_wide,
                benchmark_returns=benchmark_daily,
            )
        )
        results_rows.extend(
            _evaluate_cost(
                label=label,
                monthly_eval=monthly_eval,
                returns_wide=returns_wide,
                benchmark_symbol=benchmark_symbol,
                benchmark_returns=benchmark_daily,
            )
        )

    results_df = pd.DataFrame(results_rows).sort_values(["scenario", "profile"]).reset_index(drop=True)
    results_df.to_csv(outdir / "combo_results.csv", index=False)

    meta_monthly = combo_monthly["causal_meta_switch"].copy()
    source_counts = (
        meta_monthly["source_candidate"].astype(str).value_counts().rename_axis("source_candidate").reset_index(name="months_selected")
        if "source_candidate" in meta_monthly.columns
        else pd.DataFrame(columns=["source_candidate", "months_selected"])
    )
    source_counts.to_csv(outdir / "meta_switch_source_counts.csv", index=False)

    pivot = results_df.pivot_table(index="profile", columns="scenario", values="ann_return", aggfunc="first")
    best_delay0 = ""
    if "delay_d0" in pivot.columns:
        x = pivot["delay_d0"].dropna()
        if not x.empty:
            best_delay0 = str(x.idxmax())

    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "lock_path": str(lock_path),
        "outdir": str(outdir),
        "lookback_months": int(args.lookback_months),
        "best_profile_delay_d0_by_ann_return": best_delay0,
        "meta_switch_source_counts": source_counts.to_dict(orient="records"),
        "rows": results_rows,
    }
    _write_json(outdir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
