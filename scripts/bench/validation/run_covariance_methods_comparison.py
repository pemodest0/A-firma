#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.lab.run_corr_macro_offline import _ensure_cols, _find_latest_finance_run, _process_window


def _build_core_returns(
    *,
    panel_path: Path,
    start: str,
    end: str,
    business_days_only: bool,
    coverage_core: float,
    max_core_assets: int,
    min_assets: int,
) -> tuple[pd.DataFrame, dict[str, str]]:
    panel = pd.read_csv(panel_path)
    _ensure_cols(panel, ["date", "ticker", "sector", "r"], "panel")
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel["r"] = pd.to_numeric(panel["r"], errors="coerce")
    panel = panel.dropna(subset=["date", "ticker", "sector", "r"]).copy()
    panel = panel[(panel["date"] >= pd.Timestamp(start)) & (panel["date"] <= pd.Timestamp(end))]
    if bool(business_days_only):
        panel = panel[panel["date"].dt.dayofweek < 5].copy()
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    if panel.empty:
        raise RuntimeError("panel vazio após filtros")

    trading_days = pd.DatetimeIndex(sorted(panel["date"].drop_duplicates().to_list()))
    returns_wide = panel.pivot_table(index="date", columns="ticker", values="r", aggfunc="last").reindex(trading_days).sort_index()
    sector_map = (
        panel[["ticker", "sector"]]
        .drop_duplicates(subset=["ticker"], keep="last")
        .set_index("ticker")["sector"]
        .astype(str)
        .to_dict()
    )
    coverage = returns_wide.notna().mean(axis=0)
    core = coverage[coverage >= float(coverage_core)].index.astype(str).tolist()
    core = sorted(core)
    cap = int(max(0, max_core_assets))
    if cap > 0:
        core = core[:cap]
    if len(core) < int(min_assets):
        raise RuntimeError(f"core universo insuficiente: {len(core)} < {int(min_assets)}")
    return returns_wide[core].copy(), sector_map


def _switch_rate(regimes: pd.Series) -> float:
    s = regimes.dropna().astype(str)
    if s.shape[0] <= 1:
        return float("nan")
    sw = int((s != s.shift(1)).sum() - 1)
    return float(100.0 * sw / max(s.shape[0] - 1, 1))


def _simple_regime_series(df: pd.DataFrame) -> pd.Series:
    x = df.copy()
    p1_hi = float(pd.to_numeric(x["p1"], errors="coerce").quantile(0.80))
    p1_lo = float(pd.to_numeric(x["p1"], errors="coerce").quantile(0.20))
    d_lo = float(pd.to_numeric(x["deff"], errors="coerce").quantile(0.20))
    d_hi = float(pd.to_numeric(x["deff"], errors="coerce").quantile(0.80))
    dp = pd.to_numeric(x["p1"], errors="coerce").diff(5).abs()
    dd = pd.to_numeric(x["deff"], errors="coerce").diff(5).abs()
    dp_thr = float(dp.quantile(0.80))
    dd_thr = float(dd.quantile(0.80))
    out: list[str] = []
    for _, r in x.iterrows():
        p1 = float(r.get("p1", np.nan))
        deff = float(r.get("deff", np.nan))
        dpv = abs(float(r.get("dp1_5", np.nan))) if "dp1_5" in x.columns else float("nan")
        ddv = abs(float(r.get("ddeff_5", np.nan))) if "ddeff_5" in x.columns else float("nan")
        if np.isfinite(p1) and np.isfinite(deff) and p1 >= p1_hi and deff <= d_lo:
            out.append("stress")
        elif np.isfinite(p1) and np.isfinite(deff) and p1 <= p1_lo and deff >= d_hi:
            out.append("dispersion")
        elif (np.isfinite(dpv) and np.isfinite(dp_thr) and dpv >= dp_thr) or (np.isfinite(ddv) and np.isfinite(dd_thr) and ddv >= dd_thr):
            out.append("transition")
        else:
            out.append("stable")
    return pd.Series(out, index=x.index, dtype=str)


def _profile_row(name: str, ts: pd.DataFrame) -> dict[str, Any]:
    d = ts[~ts["insufficient_universe"]].copy()
    if d.empty:
        return {
            "profile": str(name),
            "n_rows": int(ts.shape[0]),
            "n_sufficient": 0,
            "p1_std": float("nan"),
            "deff_std": float("nan"),
            "overlap_median": float("nan"),
            "overlap_p10": float("nan"),
            "regime_switch_per_100d": float("nan"),
            "corr_cond_median": float("nan"),
            "corr_cond_p90": float("nan"),
        }
    d["dp1_5"] = pd.to_numeric(d["p1"], errors="coerce").diff(5)
    d["ddeff_5"] = pd.to_numeric(d["deff"], errors="coerce").diff(5)
    regimes = _simple_regime_series(d)
    cond = pd.to_numeric(d.get("corr_cond"), errors="coerce").replace([np.inf, -np.inf], np.nan)
    overlap = pd.to_numeric(d.get("eigvec_overlap_1d"), errors="coerce")
    return {
        "profile": str(name),
        "n_rows": int(ts.shape[0]),
        "n_sufficient": int(d.shape[0]),
        "p1_std": float(pd.to_numeric(d["p1"], errors="coerce").std(ddof=0)),
        "deff_std": float(pd.to_numeric(d["deff"], errors="coerce").std(ddof=0)),
        "overlap_median": float(overlap.median(skipna=True)),
        "overlap_p10": float(overlap.quantile(0.10)),
        "regime_switch_per_100d": _switch_rate(regimes),
        "corr_cond_median": float(cond.median(skipna=True)),
        "corr_cond_p90": float(cond.quantile(0.90)),
    }


def _gate_profile(row: dict[str, Any], base: dict[str, Any]) -> tuple[bool, list[str], float]:
    reasons: list[str] = []
    if int(row.get("n_sufficient", 0)) < 100:
        reasons.append("n_sufficient_lt_100")
    if np.isfinite(float(base.get("overlap_median", np.nan))) and np.isfinite(float(row.get("overlap_median", np.nan))):
        if float(row["overlap_median"]) + 1e-12 < float(base["overlap_median"]):
            reasons.append("overlap_median_below_baseline")
    if np.isfinite(float(base.get("regime_switch_per_100d", np.nan))) and np.isfinite(float(row.get("regime_switch_per_100d", np.nan))):
        if float(row["regime_switch_per_100d"]) > (1.10 * float(base["regime_switch_per_100d"])):
            reasons.append("switch_rate_above_baseline_10pct")
    if np.isfinite(float(base.get("corr_cond_median", np.nan))) and np.isfinite(float(row.get("corr_cond_median", np.nan))):
        if float(row["corr_cond_median"]) > float(base["corr_cond_median"]):
            reasons.append("corr_condition_not_improved")

    overlap_gain = float(row.get("overlap_median", np.nan)) - float(base.get("overlap_median", np.nan))
    switch_delta = float(base.get("regime_switch_per_100d", np.nan)) - float(row.get("regime_switch_per_100d", np.nan))
    cond_gain = np.log1p(max(float(base.get("corr_cond_median", np.nan)), 0.0)) - np.log1p(max(float(row.get("corr_cond_median", np.nan)), 0.0))
    score = float(np.nan_to_num(overlap_gain, nan=0.0) + 0.25 * np.nan_to_num(switch_delta, nan=0.0) + 0.15 * np.nan_to_num(cond_gain, nan=0.0))
    return len(reasons) == 0, reasons, score


def main() -> None:
    ap = argparse.ArgumentParser(description="Compara estimadores de covariância/correlação no Eigen Engine.")
    ap.add_argument("--panel-path", type=str, default="")
    ap.add_argument("--outdir", type=str, default=str(ROOT / "results" / "validation" / "covariance_methods"))
    ap.add_argument("--start", type=str, default="2018-01-01")
    ap.add_argument("--end", type=str, default="2026-02-12")
    ap.add_argument("--coverage-core", type=float, default=0.95)
    ap.add_argument("--max-core-assets", type=int, default=180)
    ap.add_argument("--coverage-window", type=float, default=0.98)
    ap.add_argument("--official-window", type=int, default=120)
    ap.add_argument("--min-assets", type=int, default=25)
    ap.add_argument("--business-days-only", type=int, default=1)
    ap.add_argument("--date-step", type=int, default=1, help="Use every k-th date row for faster benchmark (1=all rows).")
    ap.add_argument("--noise-step", type=int, default=10)
    ap.add_argument("--bootstrap-block", type=int, default=10)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument(
        "--profiles",
        type=str,
        default="all",
        help="all | core3 | comma-separated profile names",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.panel_path:
        panel_path = Path(args.panel_path)
    else:
        panel_path = _find_latest_finance_run() / "panel_long_sector.csv"
    if not panel_path.exists():
        raise SystemExit(f"panel path not found: {panel_path}")

    returns_wide, sector_map = _build_core_returns(
        panel_path=panel_path,
        start=str(args.start),
        end=str(args.end),
        business_days_only=bool(int(args.business_days_only)),
        coverage_core=float(args.coverage_core),
        max_core_assets=int(args.max_core_assets),
        min_assets=int(args.min_assets),
    )
    step = int(max(1, int(args.date_step)))
    if step > 1:
        returns_wide = returns_wide.iloc[::step, :].copy()
        if returns_wide.shape[0] <= int(args.official_window):
            raise SystemExit(
                f"date-step too large for window={int(args.official_window)}; rows={int(returns_wide.shape[0])}"
            )
    tickers = returns_wide.columns.astype(str).tolist()
    sector_by_ticker = {t: sector_map.get(t, "unknown") for t in tickers}

    all_profiles = [
        {"name": "sample", "cov_estimator": "sample", "rmt_cleaning": False, "rmt_cleaning_mode": "clip", "rmt_keep_top_k": 0},
        {"name": "sample_mpclip", "cov_estimator": "sample", "rmt_cleaning": True, "rmt_cleaning_mode": "clip", "rmt_keep_top_k": 0},
        {"name": "ewma", "cov_estimator": "ewma", "rmt_cleaning": False, "rmt_cleaning_mode": "clip", "rmt_keep_top_k": 0},
        {"name": "ledoit_wolf", "cov_estimator": "ledoit_wolf", "rmt_cleaning": False, "rmt_cleaning_mode": "clip", "rmt_keep_top_k": 0},
        {"name": "ledoit_wolf_mpclip", "cov_estimator": "ledoit_wolf", "rmt_cleaning": True, "rmt_cleaning_mode": "clip", "rmt_keep_top_k": 0},
        {"name": "oas", "cov_estimator": "oas", "rmt_cleaning": False, "rmt_cleaning_mode": "clip", "rmt_keep_top_k": 0},
    ]
    p_arg = str(args.profiles).strip().lower()
    if p_arg == "all":
        profiles = all_profiles
    elif p_arg == "core3":
        keep = {"sample", "sample_mpclip", "ledoit_wolf_mpclip"}
        profiles = [p for p in all_profiles if str(p["name"]) in keep]
    else:
        keep = {x.strip() for x in str(args.profiles).split(",") if x.strip()}
        profiles = [p for p in all_profiles if str(p["name"]) in keep]
    if not profiles:
        raise SystemExit(f"no valid profiles selected from --profiles={args.profiles!r}")

    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for prof in profiles:
        ts, _, _, _ = _process_window(
            returns_wide=returns_wide,
            sector_by_ticker=sector_by_ticker,
            window=int(args.official_window),
            cov_window=float(args.coverage_window),
            min_assets=int(args.min_assets),
            noise_step=int(args.noise_step),
            bootstrap_block=int(args.bootstrap_block),
            overlap_step=1,
            seed=int(args.seed),
            cov_estimator=str(prof["cov_estimator"]),
            cov_ewma_lambda=0.94,
            rmt_cleaning=bool(prof["rmt_cleaning"]),
            rmt_cleaning_mode=str(prof["rmt_cleaning_mode"]),
            rmt_keep_top_k=int(prof["rmt_keep_top_k"]),
            compute_forman=False,
            forman_topk=10,
            capture_v1=False,
            universe_name="bench",
        )
        row = _profile_row(str(prof["name"]), ts)
        rows.append({**row, **prof})
        details[str(prof["name"])] = {"n_total_rows": int(ts.shape[0]), "n_sufficient_rows": int((~ts["insufficient_universe"]).sum())}

    table = pd.DataFrame(rows).sort_values("profile").reset_index(drop=True)
    table.to_csv(outdir / "covariance_methods_comparison.csv", index=False)
    if table.empty:
        payload = {"status": "fail", "reason": "no_profiles_evaluated"}
        (outdir / "covariance_methods_comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload))
        return

    base_row = table[table["profile"] == "sample"]
    base = base_row.iloc[0].to_dict() if not base_row.empty else table.iloc[0].to_dict()

    gated: list[dict[str, Any]] = []
    for _, r in table.iterrows():
        row = r.to_dict()
        passed, reasons, score = _gate_profile(row=row, base=base)
        gated.append(
            {
                "profile": str(row["profile"]),
                "passed": bool(passed),
                "reasons": reasons,
                "score": float(score),
                "delta_overlap_median_vs_base": float(row.get("overlap_median", np.nan)) - float(base.get("overlap_median", np.nan)),
                "delta_switch_per_100d_vs_base": float(row.get("regime_switch_per_100d", np.nan))
                - float(base.get("regime_switch_per_100d", np.nan)),
                "delta_corr_cond_median_vs_base": float(row.get("corr_cond_median", np.nan)) - float(base.get("corr_cond_median", np.nan)),
            }
        )
    gate_df = pd.DataFrame(gated).sort_values(["passed", "score"], ascending=[False, False]).reset_index(drop=True)
    gate_df.to_csv(outdir / "covariance_methods_gate.csv", index=False)

    best = gate_df.iloc[0].to_dict() if not gate_df.empty else {}
    final_gate = {
        "passed": bool(best.get("passed", False) and float(best.get("score", float("-inf"))) > 0.0),
        "best_profile": str(best.get("profile", "")),
        "best_score": float(best.get("score", float("nan"))),
        "reasons": [] if bool(best.get("passed", False) and float(best.get("score", float("-inf"))) > 0.0) else ["no_profile_passed_with_positive_score"],
        "thresholds": {
            "min_score": 0.0,
            "min_n_sufficient": 100,
            "switch_rate_vs_base_max_ratio": 1.10,
            "overlap_median_vs_base_min": 1.0,
        },
    }
    payload = {
        "status": "ok",
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "panel_path": str(panel_path),
        "period": {"start": str(args.start), "end": str(args.end)},
        "window": int(args.official_window),
        "date_step": int(step),
        "baseline_profile": "sample",
        "best_profile": best,
        "final_gate": final_gate,
        "profiles": table.to_dict(orient="records"),
        "gates": gate_df.to_dict(orient="records"),
        "details": details,
    }
    (outdir / "covariance_methods_comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "best_profile": best.get("profile"), "best_score": best.get("score")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
