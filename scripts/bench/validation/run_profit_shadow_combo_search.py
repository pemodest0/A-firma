#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
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
    summarize_replay,
)
from scripts.bench.validation.run_profit_shadow_combo_validation import (  # noqa: E402
    _month_labels,
    _prev_month,
)
from scripts.bench.validation.run_profit_shadow_realism_battery import (  # noqa: E402
    _build_candidate_context,
    _json_weight_map,
    _load_price_returns,
    _resolve_path,
    _safe_float,
    _weight_json,
    build_daily_replay_with_delay,
)


BULL_RULES = ["challenger", "blend25", "blend50", "dynamic"]
RECOVERY_RULES = ["challenger", "blend25", "blend50", "dynamic"]
BEAR_RULES = ["main", "blend75", "blend50", "dynamic"]
SIDEWAYS_RULES = ["main", "challenger", "trailing_best", "blend50", "dynamic"]
LOOKBACK_OPTIONS = list(range(1, 13))
DYNAMIC_BASE_OPTIONS = [0.25, 0.50, 0.75]
DYNAMIC_STRENGTH_OPTIONS = [0.50, 1.00, 1.50]


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


def _trailing_total(monthly_ret: pd.Series, end_pos: int, lookback: int) -> float:
    if end_pos <= 0:
        return float("nan")
    start = max(0, int(end_pos) - int(max(1, lookback)))
    window = pd.to_numeric(monthly_ret.iloc[start:end_pos], errors="coerce").fillna(0.0).astype(float)
    if window.empty:
        return float("nan")
    return float(np.prod(1.0 + window.to_numpy(dtype=float)) - 1.0)


def _alpha_from_rule(
    *,
    rule: str,
    pos: int,
    ym: str,
    market_state_by_month: dict[str, str],
    main_ret: pd.Series,
    challenger_ret: pd.Series,
    lookback_months: int,
    dynamic_base_alpha_main: float,
    dynamic_strength: float,
) -> float:
    token = str(rule).strip().lower()
    if token == "main":
        return 1.0
    if token == "challenger":
        return 0.0
    if token == "blend25":
        return 0.25
    if token == "blend50":
        return 0.50
    if token == "blend75":
        return 0.75
    if token == "trailing_best":
        main_trail = _trailing_total(main_ret, pos, lookback_months)
        challenger_trail = _trailing_total(challenger_ret, pos, lookback_months)
        if np.isfinite(challenger_trail) and (not np.isfinite(main_trail) or challenger_trail > main_trail):
            return 0.0
        return 1.0
    if token == "dynamic":
        main_trail = _trailing_total(main_ret, pos, lookback_months)
        challenger_trail = _trailing_total(challenger_ret, pos, lookback_months)
        spread = challenger_trail - main_trail if np.isfinite(challenger_trail) and np.isfinite(main_trail) else 0.0
        prev_state = str(market_state_by_month.get(_prev_month(ym), "warmup")).strip().lower()
        bias = 0.0
        if prev_state in {"bull", "recovery"}:
            bias = -0.10
        elif prev_state == "bear":
            bias = 0.10
        alpha = float(dynamic_base_alpha_main) - float(dynamic_strength) * float(spread) + float(bias)
        return float(np.clip(alpha, 0.0, 1.0))
    raise ValueError(f"unsupported rule: {rule}")


def _rule_for_state(state: str, cfg: dict[str, Any]) -> str:
    state_l = str(state).strip().lower()
    if state_l == "bull":
        return str(cfg["bull_rule"])
    if state_l == "recovery":
        return str(cfg["recovery_rule"])
    if state_l == "bear":
        return str(cfg["bear_rule"])
    return str(cfg["sideways_rule"])


def _combine_month_row(row_main: pd.Series, row_challenger: pd.Series, alpha_main: float, source_rule: str) -> dict[str, Any]:
    wa = _json_weight_map(row_main.get("executed_weights_json", "{}"))
    wb = _json_weight_map(row_challenger.get("executed_weights_json", "{}"))
    alpha = float(alpha_main)
    weights: dict[str, float] = {}
    for asset in sorted(set(wa) | set(wb)):
        weight = alpha * float(wa.get(asset, 0.0)) + (1.0 - alpha) * float(wb.get(asset, 0.0))
        if abs(weight) > 1e-14:
            weights[str(asset)] = float(weight)
    return {
        "ym": str(row_main["ym"]),
        "risk_bucket": "combo_search",
        "source_rule": str(source_rule),
        "alpha_main": float(alpha),
        "ret": float(alpha * _safe_float(row_main.get("ret"), 0.0) + (1.0 - alpha) * _safe_float(row_challenger.get("ret"), 0.0)),
        "eqw_ret": float(_safe_float(row_main.get("eqw_ret"), 0.0)),
        "mkt_ret": float(_safe_float(row_main.get("mkt_ret"), 0.0)),
        "executed_weights_json": _weight_json(weights),
        "executed_assets": ",".join(sorted(weights.keys())),
        "selected_assets": ",".join(sorted(weights.keys())),
        "cash_weight": float(alpha * _safe_float(row_main.get("cash_weight"), 0.0) + (1.0 - alpha) * _safe_float(row_challenger.get("cash_weight"), 0.0)),
        "hedge_weight": float(alpha * _safe_float(row_main.get("hedge_weight"), 0.0) + (1.0 - alpha) * _safe_float(row_challenger.get("hedge_weight"), 0.0)),
        "n_selected": int(len(weights)),
    }


def build_combo_monthly_for_cfg(
    *,
    main_monthly: pd.DataFrame,
    challenger_monthly: pd.DataFrame,
    market_state_by_month: dict[str, str],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    main = main_monthly.copy().drop_duplicates(subset=["ym"], keep="last").sort_values("ym").reset_index(drop=True)
    challenger = challenger_monthly.copy().drop_duplicates(subset=["ym"], keep="last").sort_values("ym").reset_index(drop=True)
    common = sorted(set(main["ym"].astype(str)) & set(challenger["ym"].astype(str)))
    main = main[main["ym"].astype(str).isin(common)].copy().reset_index(drop=True)
    challenger = challenger[challenger["ym"].astype(str).isin(common)].copy().reset_index(drop=True)
    main_ret = pd.to_numeric(main["ret"], errors="coerce").fillna(0.0).astype(float)
    challenger_ret = pd.to_numeric(challenger["ret"], errors="coerce").fillna(0.0).astype(float)
    main_map = {str(row["ym"]): row for _, row in main.iterrows()}
    challenger_map = {str(row["ym"]): row for _, row in challenger.iterrows()}

    rows: list[dict[str, Any]] = []
    for pos, ym in enumerate(common):
        state = str(market_state_by_month.get(_prev_month(ym), "sideways")).strip().lower()
        rule = _rule_for_state(state, cfg)
        alpha_main = _alpha_from_rule(
            rule=rule,
            pos=pos,
            ym=ym,
            market_state_by_month=market_state_by_month,
            main_ret=main_ret,
            challenger_ret=challenger_ret,
            lookback_months=int(cfg["lookback_months"]),
            dynamic_base_alpha_main=float(cfg["dynamic_base_alpha_main"]),
            dynamic_strength=float(cfg["dynamic_strength"]),
        )
        rows.append(_combine_month_row(main_map[ym], challenger_map[ym], alpha_main, rule))
    return pd.DataFrame(rows)


def _monthly_turnover(monthly_eval: pd.DataFrame) -> pd.Series:
    turns: list[float] = []
    prev_core: dict[str, float] = {}
    prev_cash = 1.0
    for _, row in monthly_eval.iterrows():
        target_core = _json_weight_map(row.get("executed_weights_json", "{}"))
        target_cash = float(max(0.0, _safe_float(row.get("cash_weight"), 0.0)))
        turns.append(float(_l1_turnover(prev_core, prev_cash, target_core, target_cash)))
        prev_core, prev_cash = dict(target_core), float(target_cash)
    return pd.Series(turns, index=monthly_eval.index, dtype=float)


def _monthly_metrics(monthly_eval: pd.DataFrame, *, cost_bps: float) -> dict[str, float]:
    gross = pd.to_numeric(monthly_eval["ret"], errors="coerce").fillna(0.0).astype(float)
    turnover = _monthly_turnover(monthly_eval)
    net = gross - turnover * float(max(0.0, cost_bps) / 10000.0)
    perf = _perf_from_simple_returns(net, periods_per_year=12.0)
    return {
        "ann_return": _safe_float(perf.get("ann_return")),
        "sharpe": _safe_float(perf.get("sharpe")),
        "max_drawdown": _safe_float(perf.get("max_drawdown")),
        "total_return": _safe_float(perf.get("total_return")),
        "avg_turnover_monthly": float(turnover.mean()) if not turnover.empty else float("nan"),
    }


def _cfg_id(cfg: dict[str, Any]) -> str:
    return (
        f"lb{int(cfg['lookback_months']):02d}"
        f"_b-{cfg['bull_rule']}_r-{cfg['recovery_rule']}_be-{cfg['bear_rule']}_s-{cfg['sideways_rule']}"
        f"_a{int(round(float(cfg['dynamic_base_alpha_main']) * 100)):02d}"
        f"_k{int(round(float(cfg['dynamic_strength']) * 100)):03d}"
    )


def _parse_csv_tokens(raw: str | None, default: list[Any], *, cast: type = str) -> list[Any]:
    if raw is None or not str(raw).strip():
        return list(default)
    tokens = [part.strip() for part in str(raw).split(",")]
    values = [token for token in tokens if token]
    if not values:
        return list(default)
    if cast is int:
        return [int(token) for token in values]
    if cast is float:
        return [float(token) for token in values]
    return [cast(token) for token in values]


def _candidate_grid(
    *,
    lookbacks: list[int] | None = None,
    bull_rules: list[str] | None = None,
    recovery_rules: list[str] | None = None,
    bear_rules: list[str] | None = None,
    sideways_rules: list[str] | None = None,
    dynamic_bases: list[float] | None = None,
    dynamic_strengths: list[float] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lookbacks = list(lookbacks or LOOKBACK_OPTIONS)
    bull_rules = list(bull_rules or BULL_RULES)
    recovery_rules = list(recovery_rules or RECOVERY_RULES)
    bear_rules = list(bear_rules or BEAR_RULES)
    sideways_rules = list(sideways_rules or SIDEWAYS_RULES)
    dynamic_bases = list(dynamic_bases or DYNAMIC_BASE_OPTIONS)
    dynamic_strengths = list(dynamic_strengths or DYNAMIC_STRENGTH_OPTIONS)
    for lookback_months, bull_rule, recovery_rule, bear_rule, sideways_rule in itertools.product(
        lookbacks,
        bull_rules,
        recovery_rules,
        bear_rules,
        sideways_rules,
    ):
        uses_dynamic = "dynamic" in {bull_rule, recovery_rule, bear_rule, sideways_rule}
        base_options = dynamic_bases if uses_dynamic else [0.50]
        strength_options = dynamic_strengths if uses_dynamic else [1.00]
        for base_alpha, strength in itertools.product(base_options, strength_options):
            cfg = {
                "lookback_months": int(lookback_months),
                "bull_rule": str(bull_rule),
                "recovery_rule": str(recovery_rule),
                "bear_rule": str(bear_rule),
                "sideways_rule": str(sideways_rule),
                "dynamic_base_alpha_main": float(base_alpha),
                "dynamic_strength": float(strength),
            }
            rows.append(cfg)
    return rows


def _exact_eval_rows(
    *,
    label: str,
    monthly_eval: pd.DataFrame,
    returns_wide: pd.DataFrame,
    benchmark_symbol: str,
    benchmark_daily: pd.Series,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for delay in [0, 1]:
        history = build_daily_replay_with_delay(
            monthly_eval=monthly_eval,
            returns_wide=returns_wide,
            benchmark_returns=benchmark_daily,
            initial_capital=10000.0,
            execution_delay_days=delay,
        )
        perf = _perf_from_simple_returns(pd.to_numeric(history["portfolio_return"], errors="coerce"))
        bench = _perf_from_simple_returns(pd.to_numeric(history["benchmark_return"], errors="coerce"))
        rows.append(
            {
                "candidate_id": label,
                "scenario": f"delay_d{delay}",
                "ann_return": _safe_float(perf.get("ann_return")),
                "sharpe": _safe_float(perf.get("sharpe")),
                "max_drawdown": _safe_float(perf.get("max_drawdown")),
                "total_return": _safe_float(perf.get("total_return")),
                "edge_total_return": _safe_float(perf.get("total_return")) - _safe_float(bench.get("total_return")),
            }
        )
    for freq, cost_bps in [("monthly", 10.0), ("monthly", 30.0), ("weekly", 30.0)]:
        history = build_daily_replay_with_rebalance(
            monthly_eval=monthly_eval,
            returns_wide=returns_wide,
            benchmark_symbol=benchmark_symbol,
            benchmark_returns=benchmark_daily,
            initial_capital=10000.0,
            cost_bps=float(cost_bps),
            rebalance_frequency=freq,
        )
        summary = summarize_replay(history, return_col="net_return")
        port = (summary.get("portfolio") or {}) if isinstance(summary.get("portfolio"), dict) else {}
        rows.append(
            {
                "candidate_id": label,
                "scenario": f"{freq}_{int(cost_bps)}bps",
                "ann_return": _safe_float(port.get("ann_return")),
                "sharpe": _safe_float(port.get("sharpe")),
                "max_drawdown": _safe_float(port.get("max_drawdown")),
                "total_return": _safe_float(port.get("total_return")),
                "edge_total_return": _safe_float(summary.get("edge_vs_benchmark_total_return")),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Search the profit-shadow combo space for maximum profit.")
    ap.add_argument("--lock-path", required=True)
    ap.add_argument("--prices-dir", default=str(ROOT / "data" / "raw" / "finance" / "yfinance_daily"))
    ap.add_argument("--top-k-exact", type=int, default=24)
    ap.add_argument("--outdir", default="")
    ap.add_argument("--lookbacks", default="")
    ap.add_argument("--bull-rules", default="")
    ap.add_argument("--recovery-rules", default="")
    ap.add_argument("--bear-rules", default="")
    ap.add_argument("--sideways-rules", default="")
    ap.add_argument("--dynamic-bases", default="")
    ap.add_argument("--dynamic-strengths", default="")
    args = ap.parse_args()

    lock_path = Path(args.lock_path).resolve()
    lock = _read_json(lock_path)
    if not lock:
        raise SystemExit(f"missing lock: {lock_path}")
    outdir = _resolve_path(args.outdir) or (ROOT / "results" / "validation" / "profit_shadow_combo_search" / _run_id())
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
    market_state_by_month = _month_labels(benchmark_daily)

    grid = _candidate_grid(
        lookbacks=_parse_csv_tokens(args.lookbacks, LOOKBACK_OPTIONS, cast=int),
        bull_rules=_parse_csv_tokens(args.bull_rules, BULL_RULES, cast=str),
        recovery_rules=_parse_csv_tokens(args.recovery_rules, RECOVERY_RULES, cast=str),
        bear_rules=_parse_csv_tokens(args.bear_rules, BEAR_RULES, cast=str),
        sideways_rules=_parse_csv_tokens(args.sideways_rules, SIDEWAYS_RULES, cast=str),
        dynamic_bases=_parse_csv_tokens(args.dynamic_bases, DYNAMIC_BASE_OPTIONS, cast=float),
        dynamic_strengths=_parse_csv_tokens(args.dynamic_strengths, DYNAMIC_STRENGTH_OPTIONS, cast=float),
    )

    search_rows: list[dict[str, Any]] = []
    cfg_lookup: dict[str, dict[str, Any]] = {}
    for cfg in grid:
        combo = build_combo_monthly_for_cfg(
            main_monthly=main_ctx["monthly"],
            challenger_monthly=challenger_ctx["monthly"],
            market_state_by_month=market_state_by_month,
            cfg=cfg,
        )
        cid = _cfg_id(cfg)
        cfg_lookup[cid] = dict(cfg)
        gross = _monthly_metrics(combo, cost_bps=0.0)
        net10 = _monthly_metrics(combo, cost_bps=10.0)
        net30 = _monthly_metrics(combo, cost_bps=30.0)
        search_rows.append(
            {
                "candidate_id": cid,
                **cfg,
                "gross_ann_return": gross["ann_return"],
                "gross_sharpe": gross["sharpe"],
                "gross_max_drawdown": gross["max_drawdown"],
                "gross_total_return": gross["total_return"],
                "avg_turnover_monthly": gross["avg_turnover_monthly"],
                "net10_ann_return": net10["ann_return"],
                "net10_sharpe": net10["sharpe"],
                "net10_max_drawdown": net10["max_drawdown"],
                "net10_total_return": net10["total_return"],
                "net30_ann_return": net30["ann_return"],
                "net30_sharpe": net30["sharpe"],
                "net30_max_drawdown": net30["max_drawdown"],
                "net30_total_return": net30["total_return"],
            }
        )

    search_df = pd.DataFrame(search_rows).sort_values(["gross_ann_return", "gross_sharpe"], ascending=False).reset_index(drop=True)
    search_df.to_csv(outdir / "search_results.csv", index=False)

    top_exact_ids = set(search_df.head(int(args.top_k_exact))["candidate_id"].astype(str))
    top_exact_ids.update(search_df.sort_values(["net10_ann_return", "net10_sharpe"], ascending=False).head(int(args.top_k_exact))["candidate_id"].astype(str))
    exact_rows: list[dict[str, Any]] = []
    top_monthly_dir = outdir / "top_monthly_eval"
    top_monthly_dir.mkdir(parents=True, exist_ok=True)
    for cid in sorted(top_exact_ids):
        cfg = cfg_lookup[cid]
        combo = build_combo_monthly_for_cfg(
            main_monthly=main_ctx["monthly"],
            challenger_monthly=challenger_ctx["monthly"],
            market_state_by_month=market_state_by_month,
            cfg=cfg,
        )
        combo.to_csv(top_monthly_dir / f"{cid}.csv", index=False)
        exact_rows.extend(
            _exact_eval_rows(
                label=cid,
                monthly_eval=combo,
                returns_wide=main_ctx["returns_wide"],
                benchmark_symbol=benchmark_symbol,
                benchmark_daily=benchmark_daily,
            )
        )
    exact_df = pd.DataFrame(exact_rows).sort_values(["scenario", "ann_return"], ascending=[True, False]).reset_index(drop=True)
    exact_df.to_csv(outdir / "exact_top_results.csv", index=False)

    winners: dict[str, dict[str, Any]] = {}
    if not exact_df.empty:
        for scenario, group in exact_df.groupby("scenario"):
            best = group.sort_values(["ann_return", "sharpe"], ascending=False).iloc[0]
            cid = str(best["candidate_id"])
            cfg_row = search_df[search_df["candidate_id"] == cid].iloc[0].to_dict()
            winners[str(scenario)] = {
                "candidate_id": cid,
                "ann_return": _safe_float(best["ann_return"]),
                "sharpe": _safe_float(best["sharpe"]),
                "max_drawdown": _safe_float(best["max_drawdown"]),
                "config": cfg_row,
            }

    top_gross = search_df.iloc[0].to_dict() if not search_df.empty else {}
    top_net10 = search_df.sort_values(["net10_ann_return", "net10_sharpe"], ascending=False).iloc[0].to_dict() if not search_df.empty else {}
    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "lock_path": str(lock_path),
        "outdir": str(outdir),
        "search_args": {
            "lookbacks": _parse_csv_tokens(args.lookbacks, LOOKBACK_OPTIONS, cast=int),
            "bull_rules": _parse_csv_tokens(args.bull_rules, BULL_RULES, cast=str),
            "recovery_rules": _parse_csv_tokens(args.recovery_rules, RECOVERY_RULES, cast=str),
            "bear_rules": _parse_csv_tokens(args.bear_rules, BEAR_RULES, cast=str),
            "sideways_rules": _parse_csv_tokens(args.sideways_rules, SIDEWAYS_RULES, cast=str),
            "dynamic_bases": _parse_csv_tokens(args.dynamic_bases, DYNAMIC_BASE_OPTIONS, cast=float),
            "dynamic_strengths": _parse_csv_tokens(args.dynamic_strengths, DYNAMIC_STRENGTH_OPTIONS, cast=float),
        },
        "search_space_size": int(search_df.shape[0]),
        "top_gross_monthly": top_gross,
        "top_net10_monthly": top_net10,
        "scenario_winners_exact": winners,
        "artifacts": {
            "search_results_csv": str(outdir / "search_results.csv"),
            "exact_top_results_csv": str(outdir / "exact_top_results.csv"),
            "top_monthly_eval_dir": str(top_monthly_dir),
        },
    }
    _write_json(outdir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
