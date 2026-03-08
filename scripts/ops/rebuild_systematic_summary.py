#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _safe_float(x: Any) -> float:
    try:
        y = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return y if np.isfinite(y) else float("nan")


def _first_existing(df: pd.DataFrame, names: list[str]) -> str:
    cols = set(df.columns)
    for n in names:
        if n in cols:
            return n
    raise KeyError(f"none of columns found: {names}")


def _cum_total(ret: pd.Series) -> float:
    x = pd.to_numeric(ret, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if x.size <= 0:
        return float("nan")
    return float(np.prod(1.0 + x) - 1.0)


def _ann(total_ret: float, months: int) -> float:
    if not np.isfinite(total_ret) or int(months) <= 0:
        return float("nan")
    return float((1.0 + float(total_ret)) ** (12.0 / float(months)) - 1.0)


def _max_drawdown(ret: pd.Series) -> float:
    x = pd.to_numeric(ret, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if x.size <= 0:
        return float("nan")
    eq = np.cumprod(1.0 + x)
    peak = np.maximum.accumulate(eq)
    dd = eq / np.where(peak == 0.0, np.nan, peak) - 1.0
    dd = dd[np.isfinite(dd)]
    if dd.size <= 0:
        return float("nan")
    return float(np.min(dd))


def _mean_margin95(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return float("nan"), float("nan")
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, 0.0
    std = float(arr.std(ddof=1))
    margin = float(1.96 * std / np.sqrt(float(arr.size)))
    return mean, margin


def _bool_rate(values: pd.Series) -> float:
    arr = pd.Series(values).dropna().astype(bool)
    if arr.empty:
        return float("nan")
    return float(arr.mean())


def rebuild(yearly_dir: Path) -> dict[str, Any]:
    monthly_csv = yearly_dir / "monthly_systematic_eval.csv"
    yearly_csv = yearly_dir / "yearly_systematic_eval.csv"
    if not monthly_csv.exists():
        raise FileNotFoundError(f"missing file: {monthly_csv}")
    if not yearly_csv.exists():
        raise FileNotFoundError(f"missing file: {yearly_csv}")

    monthly = pd.read_csv(monthly_csv)
    yearly = pd.read_csv(yearly_csv)
    if monthly.empty:
        raise ValueError("monthly_systematic_eval.csv is empty")

    if "year" not in monthly.columns:
        ym_col = _first_existing(monthly, ["ym", "month", "month_id"])
        monthly["year"] = pd.to_datetime(monthly[ym_col].astype(str) + "-01", errors="coerce").dt.year
    monthly = monthly.dropna(subset=["year"]).copy()
    monthly["year"] = monthly["year"].astype(int)

    ret_col = _first_existing(monthly, ["ret", "strategy_ret"])
    eqw_col = _first_existing(monthly, ["eqw_ret", "bench_eqw_ret"])
    mkt_col = _first_existing(monthly, ["mkt_ret", "market_ret"])
    motor_col = _first_existing(monthly, ["motor_ret"])

    months_total = int(len(monthly))
    strategy_total = _cum_total(monthly[ret_col])
    eqw_total = _cum_total(monthly[eqw_col])
    market_total = _cum_total(monthly[mkt_col])
    motor_total = _cum_total(monthly[motor_col])

    alpha_eqw = pd.to_numeric(monthly[ret_col], errors="coerce") - pd.to_numeric(monthly[eqw_col], errors="coerce")
    alpha_mkt = pd.to_numeric(monthly[ret_col], errors="coerce") - pd.to_numeric(monthly[mkt_col], errors="coerce")
    alpha_eqw_arr = alpha_eqw.to_numpy(dtype=float)
    alpha_mkt_arr = alpha_mkt.to_numpy(dtype=float)
    mean_alpha_eqw, margin_alpha_eqw = _mean_margin95(alpha_eqw_arr)
    mean_alpha_mkt, margin_alpha_mkt = _mean_margin95(alpha_mkt_arr)

    year_col = _first_existing(yearly, ["year"])
    years_tested = sorted(pd.to_numeric(yearly[year_col], errors="coerce").dropna().astype(int).unique().tolist())
    if "worth_it_vs_eqw" in yearly.columns:
        worth_eqw = _bool_rate(yearly["worth_it_vs_eqw"])
    else:
        y_strategy_col = _first_existing(yearly, ["strategy_total", "strategy_return"])
        y_eqw_col = _first_existing(yearly, ["eqw_total", "eqw_return"])
        worth_eqw = _bool_rate(pd.to_numeric(yearly[y_strategy_col], errors="coerce") > pd.to_numeric(yearly[y_eqw_col], errors="coerce"))
    if "worth_it_vs_market" in yearly.columns:
        worth_mkt = _bool_rate(yearly["worth_it_vs_market"])
    else:
        y_strategy_col = _first_existing(yearly, ["strategy_total", "strategy_return"])
        y_mkt_col = _first_existing(yearly, ["market_total", "market_return"])
        worth_mkt = _bool_rate(pd.to_numeric(yearly[y_strategy_col], errors="coerce") > pd.to_numeric(yearly[y_mkt_col], errors="coerce"))

    summary = {
        "years_tested": years_tested,
        "months_total": months_total,
        "strategy_total": strategy_total,
        "eqw_total": eqw_total,
        "market_total": market_total,
        "motor_total": motor_total,
        "alpha_total_vs_eqw": _safe_float(strategy_total - eqw_total),
        "alpha_total_vs_market": _safe_float(strategy_total - market_total),
        "strategy_ann": _ann(strategy_total, months_total),
        "eqw_ann": _ann(eqw_total, months_total),
        "market_ann": _ann(market_total, months_total),
        "strategy_max_drop": _max_drawdown(monthly[ret_col]),
        "eqw_max_drop": _max_drawdown(monthly[eqw_col]),
        "market_max_drop": _max_drawdown(monthly[mkt_col]),
        "monthly_alpha_mean_vs_eqw": mean_alpha_eqw,
        "monthly_alpha_margin95_vs_eqw": margin_alpha_eqw,
        "monthly_alpha_ci95_vs_eqw": [
            _safe_float(mean_alpha_eqw - margin_alpha_eqw),
            _safe_float(mean_alpha_eqw + margin_alpha_eqw),
        ],
        "monthly_alpha_prob_positive_vs_eqw": _safe_float(np.mean(alpha_eqw_arr > 0.0)),
        "monthly_alpha_mean_vs_market": mean_alpha_mkt,
        "monthly_alpha_margin95_vs_market": margin_alpha_mkt,
        "monthly_alpha_ci95_vs_market": [
            _safe_float(mean_alpha_mkt - margin_alpha_mkt),
            _safe_float(mean_alpha_mkt + margin_alpha_mkt),
        ],
        "monthly_alpha_prob_positive_vs_market": _safe_float(np.mean(alpha_mkt_arr > 0.0)),
        "worth_it_rate_vs_eqw": worth_eqw,
        "worth_it_rate_vs_market": worth_mkt,
        "rebuilt_at_utc": datetime.now(timezone.utc).isoformat(),
        "rebuilt_from": {
            "monthly_csv": str(monthly_csv),
            "yearly_csv": str(yearly_csv),
        },
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild systematic_summary.json from monthly/yearly eval CSV files.")
    ap.add_argument("--yearly-dir", required=True, help="Directory containing monthly_systematic_eval.csv and yearly_systematic_eval.csv")
    ap.add_argument("--output", default="", help="Optional output path for summary JSON (default: <yearly-dir>/systematic_summary.json)")
    args = ap.parse_args()

    yearly_dir = Path(args.yearly_dir).resolve()
    summary = rebuild(yearly_dir)
    output = Path(args.output).resolve() if str(args.output).strip() else (yearly_dir / "systematic_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(output), "monthly_alpha_prob_positive_vs_eqw": summary["monthly_alpha_prob_positive_vs_eqw"]}))


if __name__ == "__main__":
    main()

