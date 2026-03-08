#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _parse_int_list(s: str) -> list[int]:
    vals = [int(x.strip()) for x in str(s).split(",") if str(x).strip()]
    if not vals:
        raise ValueError("empty int list")
    return vals


def _parse_float_list(s: str) -> list[float]:
    vals = [float(x.strip()) for x in str(s).split(",") if str(x).strip()]
    if not vals:
        raise ValueError("empty float list")
    return vals


def _latest_systematic_run() -> Path:
    base = ROOT / "results" / "portfolio_sim"
    if not base.exists():
        raise FileNotFoundError(f"missing dir: {base}")
    runs = sorted(
        [p for p in base.iterdir() if p.is_dir() and p.name.endswith("_systematic_yearly")],
        key=lambda p: p.name,
        reverse=True,
    )
    for run in runs:
        if (run / "simulation_summary.json").exists():
            return run
    raise FileNotFoundError("no systematic run with simulation_summary.json")


def _run_canonical_for_topk(
    *,
    impact_dir: Path,
    returns_csv: Path,
    top_k: int,
    train_end: str,
    outdir: Path,
) -> Path:
    py = sys.executable
    script = ROOT / "scripts" / "ops" / "run_canonical_systematic_eval.py"
    cmd = [
        py,
        str(script),
        "--impact-dir",
        str(impact_dir),
        "--returns-csv",
        str(returns_csv),
        "--outdir",
        str(outdir),
        "--train-end",
        str(train_end),
        "--start-ym",
        "2019-01",
        "--top-k-options",
        str(int(top_k)),
        "--impact-power-options",
        "0",
        "--wmax-options",
        "0.1",
        "--mom-lookback-options",
        "0",
        "--mom-threshold-options",
        "-0.02",
        "--modes",
        "const",
        "--defense-enabled",
        "1",
        "--defense-multiplier",
        "0.85",
        "--decel-enabled",
        "1",
        "--decel-lookback-months",
        "6",
        "--decel-min-streak",
        "2",
        "--decel-multiplier",
        "0.95",
        "--decel-topk-multiplier",
        "0.85",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    monthly = outdir / "monthly_systematic_eval.csv"
    if not monthly.exists():
        raise FileNotFoundError(f"missing {monthly}")
    return monthly


def _mdd_from_series(ret: pd.Series) -> float:
    s = pd.to_numeric(ret, errors="coerce").fillna(0.0).astype(float)
    if s.empty:
        return float("nan")
    eq = np.cumprod(1.0 + s.to_numpy(dtype=float))
    peak = np.maximum.accumulate(eq)
    dd = eq / np.where(peak == 0.0, np.nan, peak) - 1.0
    dd = dd[np.isfinite(dd)]
    return float(np.min(dd)) if dd.size > 0 else float("nan")


def _total_return(ret: pd.Series) -> float:
    s = pd.to_numeric(ret, errors="coerce").dropna().astype(float)
    if s.empty:
        return float("nan")
    return float(np.prod(1.0 + s.to_numpy(dtype=float)) - 1.0)


def _annualized_return(ret: pd.Series) -> float:
    s = pd.to_numeric(ret, errors="coerce").dropna().astype(float)
    if s.empty:
        return float("nan")
    total = float(np.prod(1.0 + s.to_numpy(dtype=float)) - 1.0)
    return float((1.0 + total) ** (12.0 / float(len(s))) - 1.0)


def _signal_mask(
    df: pd.DataFrame,
    *,
    include_transition: bool = False,
    risk_budget_threshold: float | None = None,
) -> pd.Series:
    rbucket = df["risk_bucket"].astype(str).str.lower()
    if include_transition:
        risk_stress = rbucket.isin(["stress", "transition"])
    else:
        risk_stress = rbucket.eq("stress")
    defense_on = pd.to_numeric(df["defense_active"], errors="coerce").fillna(0).astype(int).gt(0)
    decel_on = pd.to_numeric(df["decel_active"], errors="coerce").fillna(0).astype(int).gt(0)
    if risk_budget_threshold is None:
        risk_budget_on = pd.Series(False, index=df.index)
    else:
        rb = pd.to_numeric(df.get("risk_budget", np.nan), errors="coerce")
        risk_budget_on = rb.lt(float(risk_budget_threshold)).fillna(False)
    return risk_stress | defense_on | decel_on | risk_budget_on


@dataclass(frozen=True)
class RealizeRule:
    realize_pct: float
    reinvest_delay_off_months: int


def _simulate_with_realization(
    df: pd.DataFrame,
    *,
    rule: RealizeRule,
    initial_capital: float,
    signal_include_transition: bool,
    signal_risk_budget_threshold: float | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    d = df.copy().sort_values("ym").reset_index(drop=True)
    d["signal_on"] = _signal_mask(
        d,
        include_transition=bool(signal_include_transition),
        risk_budget_threshold=signal_risk_budget_threshold,
    ).astype(int)
    d["ret"] = pd.to_numeric(d["ret"], errors="coerce").fillna(0.0)

    invested = float(initial_capital)
    cash = 0.0
    off_streak = 0
    rows: list[dict[str, Any]] = []

    for _, row in d.iterrows():
        ym = str(row["ym"])
        signal_on = int(row["signal_on"]) == 1

        if signal_on:
            off_streak = 0
        else:
            off_streak += 1
            if cash > 0 and off_streak >= int(rule.reinvest_delay_off_months):
                invested += cash
                cash = 0.0

        invested_before = invested
        realized_now = 0.0
        # Signal is known at month start (built from prior information), so realization
        # must happen before applying current-month return.
        if signal_on and invested > 0:
            realized_now = max(0.0, float(rule.realize_pct) * invested)
            invested -= realized_now
            cash += realized_now

        ret_m = float(row["ret"])
        invested *= (1.0 + ret_m)

        total_value = invested + cash
        rows.append(
            {
                "ym": ym,
                "ret": ret_m,
                "signal_on": int(signal_on),
                "invested_before": invested_before,
                "realized_now": realized_now,
                "invested_after": invested,
                "cash_after": cash,
                "total_value": total_value,
            }
        )

    hist = pd.DataFrame(rows)
    if hist.empty:
        return hist, {"status": "empty"}

    # Convert value path to equivalent monthly returns for metrics.
    v = pd.to_numeric(hist["total_value"], errors="coerce").astype(float)
    ret_equiv = v.pct_change().fillna((v.iloc[0] / float(initial_capital)) - 1.0)

    summary = {
        "status": "ok",
        "initial_capital": float(initial_capital),
        "final_value": float(v.iloc[-1]),
        "total_return": float((v.iloc[-1] / float(initial_capital)) - 1.0),
        "annualized_return": _annualized_return(ret_equiv),
        "max_drawdown": _mdd_from_series(ret_equiv),
        "realization_events": int((hist["realized_now"] > 0).sum()),
        "realized_cash_total_flow": float(hist["realized_now"].sum()),
    }
    return hist, summary


def _metrics_for_slice(df: pd.DataFrame, start_ym: str | None, end_ym: str | None) -> pd.DataFrame:
    d = df.copy()
    if start_ym is not None:
        d = d[d["ym"] >= str(start_ym)]
    if end_ym is not None:
        d = d[d["ym"] <= str(end_ym)]
    return d.reset_index(drop=True)


def _next_month_observed(values: pd.Series) -> pd.Series:
    s = pd.to_numeric(values, errors="coerce")
    arr = s.to_numpy(dtype=float)
    out = np.full(arr.shape[0], np.nan, dtype=float)
    if arr.shape[0] >= 2:
        out[:-1] = arr[1:]
    return pd.Series(out, index=values.index)


def _pick_rule_on_train(
    df: pd.DataFrame,
    candidates: list[RealizeRule],
    *,
    train_end: str,
    dd_penalty: float,
    signal_include_transition: bool,
    signal_risk_budget_threshold: float | None,
) -> tuple[RealizeRule, pd.DataFrame]:
    train = _metrics_for_slice(df, None, train_end)
    rows: list[dict[str, Any]] = []
    if train.empty:
        raise RuntimeError("train slice is empty")

    # Objective: prioritize downside reduction without killing return.
    for rule in candidates:
        _, summ = _simulate_with_realization(
            train,
            rule=rule,
            initial_capital=1.0,
            signal_include_transition=signal_include_transition,
            signal_risk_budget_threshold=signal_risk_budget_threshold,
        )
        total = _safe_float(summ.get("total_return"))
        mdd = _safe_float(summ.get("max_drawdown"))
        score = total - float(dd_penalty) * abs(mdd) if np.isfinite(total) and np.isfinite(mdd) else float("-inf")
        rows.append(
            {
                "realize_pct": float(rule.realize_pct),
                "reinvest_delay_off_months": int(rule.reinvest_delay_off_months),
                "train_total_return": total,
                "train_max_drawdown": mdd,
                "train_score": score,
            }
        )

    table = pd.DataFrame(rows).sort_values("train_score", ascending=False).reset_index(drop=True)
    best = table.iloc[0]
    return (
        RealizeRule(
            realize_pct=float(best["realize_pct"]),
            reinvest_delay_off_months=int(best["reinvest_delay_off_months"]),
        ),
        table,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Simulate investment scenarios with capital X, asset count, and realization timing guide.")
    ap.add_argument("--base-run-dir", default="", help="Base systematic yearly run; default latest.")
    ap.add_argument("--top-k-list", default="20,32,52,64")
    ap.add_argument("--capitals", default="1000,5000,10000,50000")
    ap.add_argument("--train-end", default="2023-12")
    ap.add_argument("--test-start", default="2024-01")
    ap.add_argument("--test-end", default="")
    ap.add_argument("--realize-pcts", default="0.10,0.20,0.30,0.40,0.50,0.60,0.80,1.00")
    ap.add_argument("--reinvest-delays", default="1,2,3,4,6,9,12")
    ap.add_argument("--train-dd-penalty", type=float, default=0.8)
    ap.add_argument("--topk-robust-lookback-years", type=int, default=3)
    ap.add_argument("--signal-include-transition", type=int, default=0)
    ap.add_argument("--signal-risk-budget-threshold", type=float, default=-1.0)
    ap.add_argument("--preservation-mode", type=int, default=0, help="If 1, prioritize near-zero loss (capital preservation).")
    ap.add_argument(
        "--select-topk-on",
        default="train",
        choices=["train", "test"],
        help="Choose best top_k by train metrics (causal) or test metrics (legacy).",
    )
    ap.add_argument(
        "--allow-leaky-test-selection",
        type=int,
        default=0,
        help="Must be 1 to allow --select-topk-on test (for research-only comparisons).",
    )
    ap.add_argument("--outdir", default="", help="Output dir (default: results/portfolio_sim/<runid>_invest_scenarios)")
    args = ap.parse_args()
    if str(args.select_topk_on).strip().lower() == "test" and int(args.allow_leaky_test_selection) != 1:
        raise ValueError(
            "Leakage guard: --select-topk-on test is blocked by default. "
            "Use --allow-leaky-test-selection 1 only for research diagnostics."
        )

    base_run = Path(args.base_run_dir).resolve() if args.base_run_dir.strip() else _latest_systematic_run()
    sim = json.loads((base_run / "simulation_summary.json").read_text(encoding="utf-8"))
    impact_dir = Path(sim["impact_dir"])
    returns_csv = Path(sim["returns_csv"])

    top_ks = _parse_int_list(args.top_k_list)
    capitals = _parse_float_list(args.capitals)
    train_end_ts = pd.to_datetime(f"{str(args.train_end)}-01", errors="coerce")
    if pd.isna(train_end_ts):
        raise ValueError(f"invalid --train-end: {args.train_end}")
    train_end_date = str(train_end_ts.to_period("M").end_time.date())

    run_id = _run_id()
    outdir = Path(args.outdir).resolve() if args.outdir.strip() else (ROOT / "results" / "portfolio_sim" / f"{run_id}_invest_scenarios")
    outdir.mkdir(parents=True, exist_ok=True)

    scen_root = outdir / "scenario_runs"
    scen_root.mkdir(parents=True, exist_ok=True)

    realize_pcts = _parse_float_list(args.realize_pcts)
    reinvest_delays = _parse_int_list(args.reinvest_delays)
    candidate_rules = [RealizeRule(float(p), int(d)) for p in realize_pcts for d in reinvest_delays]
    signal_include_transition = bool(int(args.signal_include_transition))
    preservation_mode = bool(int(args.preservation_mode))
    train_dd_penalty = float(args.train_dd_penalty)
    signal_risk_budget_threshold = float(args.signal_risk_budget_threshold)
    if not np.isfinite(signal_risk_budget_threshold) or signal_risk_budget_threshold < 0.0:
        signal_risk_budget_threshold = None
    if preservation_mode:
        train_dd_penalty = max(train_dd_penalty, 5.0)
        if signal_risk_budget_threshold is None:
            signal_risk_budget_threshold = 0.75

    scenario_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    realize_calendar_rows: list[dict[str, Any]] = []
    test_start_year = int(str(args.test_start).split("-")[0])
    robust_lookback_years = int(max(1, args.topk_robust_lookback_years))

    for top_k in top_ks:
        run_dir = scen_root / f"topk_{int(top_k)}"
        monthly_path = _run_canonical_for_topk(
            impact_dir=impact_dir,
            returns_csv=returns_csv,
            top_k=int(top_k),
            train_end=train_end_date,
            outdir=run_dir,
        )
        d = pd.read_csv(monthly_path).sort_values("ym").reset_index(drop=True)
        d["ym"] = d["ym"].astype(str)

        best_rule, train_table = _pick_rule_on_train(
            d,
            candidate_rules,
            train_end=str(args.train_end),
            dd_penalty=float(train_dd_penalty),
            signal_include_transition=signal_include_transition,
            signal_risk_budget_threshold=signal_risk_budget_threshold,
        )
        train_table.to_csv(run_dir / "realize_rule_train_grid.csv", index=False)
        train_best = train_table.iloc[0]

        full_base = _simulate_with_realization(
            d,
            rule=RealizeRule(0.0, 1),
            initial_capital=1.0,
            signal_include_transition=signal_include_transition,
            signal_risk_budget_threshold=signal_risk_budget_threshold,
        )[1]
        full_real = _simulate_with_realization(
            d,
            rule=best_rule,
            initial_capital=1.0,
            signal_include_transition=signal_include_transition,
            signal_risk_budget_threshold=signal_risk_budget_threshold,
        )[1]

        test_end = str(args.test_end).strip() if str(args.test_end).strip() else None
        test_df = _metrics_for_slice(d, str(args.test_start), test_end)
        test_base = _simulate_with_realization(
            test_df,
            rule=RealizeRule(0.0, 1),
            initial_capital=1.0,
            signal_include_transition=signal_include_transition,
            signal_risk_budget_threshold=signal_risk_budget_threshold,
        )[1]
        test_hist, test_real = _simulate_with_realization(
            test_df,
            rule=best_rule,
            initial_capital=1.0,
            signal_include_transition=signal_include_transition,
            signal_risk_budget_threshold=signal_risk_budget_threshold,
        )
        test_hist.to_csv(run_dir / "test_realization_path.csv", index=False)

        # Robustness proxy for top_k selection: yearly performance in prior years only.
        pre_start_ym = f"{test_start_year - robust_lookback_years}-01"
        pre_end_ym = f"{test_start_year - 1}-12"
        pre_df = _metrics_for_slice(d, pre_start_ym, pre_end_ym)
        pre_worst_year_ret = float("nan")
        pre_median_year_ret = float("nan")
        pre_years_count = 0
        if not pre_df.empty:
            pre_hist, _ = _simulate_with_realization(
                pre_df,
                rule=best_rule,
                initial_capital=1.0,
                signal_include_transition=signal_include_transition,
                signal_risk_budget_threshold=signal_risk_budget_threshold,
            )
            if not pre_hist.empty:
                z = pre_hist.copy()
                z["year"] = z["ym"].astype(str).str[:4]
                v = pd.to_numeric(z["total_value"], errors="coerce").astype(float)
                z["ret_equiv"] = v.pct_change().fillna((v.iloc[0] / 1.0) - 1.0)
                yret = z.groupby("year", as_index=False).agg(year_total_return=("ret_equiv", _total_return))
                if not yret.empty:
                    pre_worst_year_ret = _safe_float(yret["year_total_return"].min())
                    pre_median_year_ret = _safe_float(yret["year_total_return"].median())
                    pre_years_count = int(yret.shape[0])

        # Guidance quality: next-month expectation conditioned on current signal
        t = test_df.copy()
        t["signal_on"] = _signal_mask(
            t,
            include_transition=signal_include_transition,
            risk_budget_threshold=signal_risk_budget_threshold,
        ).astype(int)
        t["next_ret"] = _next_month_observed(t["ret"])
        next_when_signal = _safe_float(t.loc[t["signal_on"] == 1, "next_ret"].mean())
        next_when_off = _safe_float(t.loc[t["signal_on"] == 0, "next_ret"].mean())

        scenario_rows.append(
            {
                "top_k": int(top_k),
                "best_realize_pct": float(best_rule.realize_pct),
                "best_reinvest_delay_off_months": int(best_rule.reinvest_delay_off_months),
                "train_rule_score": _safe_float(train_best.get("train_score")),
                "train_rule_total_return": _safe_float(train_best.get("train_total_return")),
                "train_rule_max_drawdown": _safe_float(train_best.get("train_max_drawdown")),
                "pre_worst_year_return": pre_worst_year_ret,
                "pre_median_year_return": pre_median_year_ret,
                "pre_years_count": int(pre_years_count),
                "full_base_total_return": _safe_float(full_base.get("total_return")),
                "full_real_total_return": _safe_float(full_real.get("total_return")),
                "full_base_max_drawdown": _safe_float(full_base.get("max_drawdown")),
                "full_real_max_drawdown": _safe_float(full_real.get("max_drawdown")),
                "test_base_total_return": _safe_float(test_base.get("total_return")),
                "test_real_total_return": _safe_float(test_real.get("total_return")),
                "test_base_max_drawdown": _safe_float(test_base.get("max_drawdown")),
                "test_real_max_drawdown": _safe_float(test_real.get("max_drawdown")),
                "test_next_month_ret_when_signal_on": next_when_signal,
                "test_next_month_ret_when_signal_off": next_when_off,
            }
        )

        for cap in capitals:
            # Capital projection aligned with test window (year-specific when test_start/test_end are provided).
            base_proj_test = _simulate_with_realization(
                test_df,
                rule=RealizeRule(0.0, 1),
                initial_capital=float(cap),
                signal_include_transition=signal_include_transition,
                signal_risk_budget_threshold=signal_risk_budget_threshold,
            )[1]
            real_proj_test = _simulate_with_realization(
                test_df,
                rule=best_rule,
                initial_capital=float(cap),
                signal_include_transition=signal_include_transition,
                signal_risk_budget_threshold=signal_risk_budget_threshold,
            )[1]
            # Keep full-horizon projection for reference.
            base_proj_full = _simulate_with_realization(
                d,
                rule=RealizeRule(0.0, 1),
                initial_capital=float(cap),
                signal_include_transition=signal_include_transition,
                signal_risk_budget_threshold=signal_risk_budget_threshold,
            )[1]
            real_proj_full = _simulate_with_realization(
                d,
                rule=best_rule,
                initial_capital=float(cap),
                signal_include_transition=signal_include_transition,
                signal_risk_budget_threshold=signal_risk_budget_threshold,
            )[1]
            projection_rows.append(
                {
                    "top_k": int(top_k),
                    "capital_inicial": float(cap),
                    # Backward-compatible names now point to test-window projection.
                    "capital_final_base": _safe_float(base_proj_test.get("final_value")),
                    "capital_final_realizacao": _safe_float(real_proj_test.get("final_value")),
                    "ganho_liquido_base": _safe_float(base_proj_test.get("final_value")) - float(cap),
                    "ganho_liquido_realizacao": _safe_float(real_proj_test.get("final_value")) - float(cap),
                    "diferenca_realizacao_menos_base": _safe_float(real_proj_test.get("final_value"))
                    - _safe_float(base_proj_test.get("final_value")),
                    "capital_final_base_full": _safe_float(base_proj_full.get("final_value")),
                    "capital_final_realizacao_full": _safe_float(real_proj_full.get("final_value")),
                }
            )

        # Calendar of suggested realize moments (test period).
        test_hist = test_hist.copy()
        test_hist["next_ret"] = _next_month_observed(test_hist["ret"])
        for _, r in test_hist.iterrows():
            if int(r.get("signal_on", 0)) != 1:
                continue
            realize_calendar_rows.append(
                {
                    "top_k": int(top_k),
                    "ym": str(r["ym"]),
                    "realized_now_unit_capital": _safe_float(r.get("realized_now")),
                    "total_value_after": _safe_float(r.get("total_value")),
                    "next_month_ret_observed": _safe_float(r.get("next_ret")),
                }
            )

    scen_df = pd.DataFrame(scenario_rows).sort_values("top_k").reset_index(drop=True)
    proj_df = pd.DataFrame(projection_rows).sort_values(["top_k", "capital_inicial"]).reset_index(drop=True)
    if realize_calendar_rows:
        cal_df = pd.DataFrame(realize_calendar_rows).sort_values(["top_k", "ym"]).reset_index(drop=True)
    else:
        cal_df = pd.DataFrame(
            columns=[
                "top_k",
                "ym",
                "realized_now_unit_capital",
                "total_value_after",
                "next_month_ret_observed",
            ]
        )

    scen_path = outdir / "scenario_summary.csv"
    proj_path = outdir / "capital_projection.csv"
    cal_path = outdir / "realize_calendar_test.csv"
    scen_df.to_csv(scen_path, index=False)
    proj_df.to_csv(proj_path, index=False)
    cal_df.to_csv(cal_path, index=False)

    if str(args.select_topk_on).strip().lower() == "train":
        # Causal selection with robustness: prioritize lower train drawdown, then stable pre-test yearly behavior.
        best = scen_df.sort_values(
            ["train_rule_max_drawdown", "pre_worst_year_return", "pre_median_year_return", "train_rule_score", "train_rule_total_return", "top_k"],
            ascending=[False, False, False, False, False, True],
        ).iloc[0].to_dict()
    else:
        # Legacy mode kept for comparability with older artifacts.
        best = scen_df.sort_values(
            ["test_real_total_return", "test_real_max_drawdown"],
            ascending=[False, False],
        ).iloc[0].to_dict()

    payload = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_base_run_dir": str(base_run),
        "assumptions": {
            "train_end": str(args.train_end),
            "test_start": str(args.test_start),
            "test_end": (str(args.test_end) if str(args.test_end).strip() else None),
            "signal_definition": {
                "include_stress": True,
                "include_transition": bool(signal_include_transition),
                "include_defense_active": True,
                "include_decel_active": True,
                "risk_budget_threshold": (None if signal_risk_budget_threshold is None else float(signal_risk_budget_threshold)),
            },
            "preservation_mode": bool(preservation_mode),
            "train_dd_penalty": float(train_dd_penalty),
            "topk_robust_lookback_years": int(robust_lookback_years),
            "signal_include_transition": bool(signal_include_transition),
            "signal_risk_budget_threshold": (None if signal_risk_budget_threshold is None else float(signal_risk_budget_threshold)),
            "select_topk_on": str(args.select_topk_on),
            "allow_leaky_test_selection": int(args.allow_leaky_test_selection),
            "realization_rule_candidates": [
                {"realize_pct": float(r.realize_pct), "reinvest_delay_off_months": int(r.reinvest_delay_off_months)}
                for r in candidate_rules
            ],
        },
        "best_scenario": best,
        "artifacts": {
            "scenario_summary_csv": str(scen_path),
            "capital_projection_csv": str(proj_path),
            "realize_calendar_test_csv": str(cal_path),
        },
    }
    out_json = outdir / "investment_scenarios_summary.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "outdir": str(outdir),
                "summary_json": str(out_json),
                "best_top_k": int(best["top_k"]),
                "best_test_real_total_return": _safe_float(best["test_real_total_return"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
