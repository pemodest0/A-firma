#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _total(ret: pd.Series) -> float:
    s = pd.to_numeric(ret, errors="coerce").dropna().astype(float)
    if s.empty:
        return float("nan")
    return float(np.prod(1.0 + s.to_numpy(dtype=float)) - 1.0)


def _ann(ret: pd.Series) -> float:
    s = pd.to_numeric(ret, errors="coerce").dropna().astype(float)
    if s.empty:
        return float("nan")
    tot = float(np.prod(1.0 + s.to_numpy(dtype=float)) - 1.0)
    return float((1.0 + tot) ** (12.0 / float(len(s))) - 1.0)


def _mdd(ret: pd.Series) -> float:
    s = pd.to_numeric(ret, errors="coerce").fillna(0.0).astype(float)
    if s.empty:
        return float("nan")
    eq = np.cumprod(1.0 + s.to_numpy(dtype=float))
    peak = np.maximum.accumulate(eq)
    dd = eq / np.where(peak == 0.0, np.nan, peak) - 1.0
    dd = dd[np.isfinite(dd)]
    return float(np.min(dd)) if dd.size > 0 else float("nan")


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
        required = ["monthly_systematic_eval.csv", "yearly_systematic_eval.csv", "simulation_summary.json"]
        if all((run / f).exists() for f in required):
            return run
    raise FileNotFoundError("no systematic yearly run with required artifacts")


def _build_monthly_returns(returns_csv: Path) -> pd.DataFrame:
    d = pd.read_csv(returns_csv)
    if "date" not in d.columns:
        raise ValueError(f"missing date column in {returns_csv}")
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    assets = [c for c in d.columns if c != "date"]
    if not assets:
        raise ValueError(f"no asset columns in {returns_csv}")
    for c in assets:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
    ym = d["date"].dt.to_period("M").astype(str)
    labels: list[str] = []
    rows: list[np.ndarray] = []
    for label, g in d.groupby(ym):
        arr = g[assets].to_numpy(dtype=float)
        rows.append(np.prod(1.0 + arr, axis=0) - 1.0)
        labels.append(str(label))
    return pd.DataFrame(np.vstack(rows), index=labels, columns=assets).sort_index()


def _build_snapshots(impact_csv: Path, *, max_assets_per_month: int) -> dict[str, pd.DataFrame]:
    d = pd.read_csv(impact_csv, usecols=["date", "asset_id", "impact_global"])
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date", "asset_id"]).sort_values("date").reset_index(drop=True)
    d["impact_global"] = pd.to_numeric(d["impact_global"], errors="coerce").fillna(0.0)
    d["ym"] = d["date"].dt.to_period("M").astype(str)
    snap = d.groupby(["ym", "asset_id"], as_index=False).tail(1).copy()
    snap = (
        snap.sort_values(["ym", "impact_global"], ascending=[True, False])
        .groupby("ym", as_index=False)
        .head(int(max_assets_per_month))
        .reset_index(drop=True)
    )
    out: dict[str, pd.DataFrame] = {}
    for ym, g in snap.groupby("ym"):
        out[str(ym)] = g[["asset_id", "impact_global"]].sort_values("impact_global", ascending=False).reset_index(drop=True)
    return out


def _step1_cost_stress_and_turnover(
    monthly: pd.DataFrame,
    mret: pd.DataFrame,
    snap_by_month: dict[str, pd.DataFrame],
    *,
    real_cost_bps: float,
    stress_multiplier: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    months = monthly["ym"].astype(str).tolist()
    prev_months = [months[i - 1] if i > 0 else None for i in range(len(months))]

    pre_w: dict[str, float] = {}
    cash_pre = 1.0
    turnovers: list[float] = []

    for i, row in monthly.iterrows():
        ym_cur = str(row["ym"])
        ym_prev = prev_months[i]
        rb = _safe_float(row.get("risk_budget", 0.0))
        nsel = int(_safe_float(row.get("n_selected", 0.0)) if np.isfinite(_safe_float(row.get("n_selected", 0.0))) else 0)
        eff_k = int(
            _safe_float(row.get("effective_top_k", nsel))
            if np.isfinite(_safe_float(row.get("effective_top_k", nsel)))
            else nsel
        )

        target: dict[str, float] = {}
        if ym_prev and ym_prev in snap_by_month and nsel > 0 and rb > 0:
            sel = [a for a in snap_by_month[ym_prev].head(eff_k)["asset_id"].tolist() if a in mret.columns][:nsel]
            if sel:
                w_each = float(rb) / float(len(sel))
                for a in sel:
                    target[a] = w_each
        cash_target = max(0.0, 1.0 - float(sum(target.values())))

        keys = set(pre_w.keys()) | set(target.keys())
        l1 = sum(abs(float(target.get(a, 0.0)) - float(pre_w.get(a, 0.0))) for a in keys) + abs(cash_target - cash_pre)
        turnover = 0.5 * l1
        turnovers.append(float(turnover))

        gross_ret = _safe_float(row.get("ret", 0.0))
        denom = 1.0 + (gross_ret if np.isfinite(gross_ret) else 0.0)
        if denom <= 0:
            pre_w, cash_pre = {}, 1.0
            continue

        nxt: dict[str, float] = {}
        for a, w in target.items():
            r = _safe_float(mret.at[ym_cur, a]) if (ym_cur in mret.index and a in mret.columns) else 0.0
            v = float(w) * (1.0 + (r if np.isfinite(r) else 0.0)) / denom
            if abs(v) > 1e-12:
                nxt[a] = float(v)
        pre_w = nxt
        cash_pre = float(cash_target / denom)

    out = monthly.copy()
    out["turnover"] = turnovers
    real_rate = float(real_cost_bps) / 10000.0
    stress_rate = real_rate * float(stress_multiplier)
    out["ret_cost_real"] = out["ret"] - out["turnover"] * real_rate
    out["ret_cost_stress"] = out["ret"] - out["turnover"] * stress_rate

    payload = {
        "assumption": {
            "cost_model": "monthly_turnover_cost",
            "turnover_definition": "0.5 * L1(target_weights - pre_rebalance_weights)",
            "real_cost_bps": float(real_cost_bps),
            "stress_cost_bps": float(real_cost_bps * stress_multiplier),
        },
        "avg_monthly_turnover": _safe_float(out["turnover"].mean()),
        "strategy_no_cost": {
            "total_return": _total(out["ret"]),
            "annualized_return": _ann(out["ret"]),
            "max_drawdown": _mdd(out["ret"]),
        },
        "strategy_cost_real": {
            "total_return": _total(out["ret_cost_real"]),
            "annualized_return": _ann(out["ret_cost_real"]),
            "max_drawdown": _mdd(out["ret_cost_real"]),
        },
        "strategy_cost_stress_2x": {
            "total_return": _total(out["ret_cost_stress"]),
            "annualized_return": _ann(out["ret_cost_stress"]),
            "max_drawdown": _mdd(out["ret_cost_stress"]),
        },
    }
    return out, payload


def _step3_random_baseline_same_exposure(
    monthly: pd.DataFrame,
    mret: pd.DataFrame,
    snap_by_month: dict[str, pd.DataFrame],
    *,
    random_iterations: int,
    random_seed: int,
    pool_max_assets: int,
) -> dict[str, Any]:
    months = monthly["ym"].astype(str).tolist()
    prev_months = [months[i - 1] if i > 0 else None for i in range(len(months))]
    rows = monthly.to_dict("records")

    rng = np.random.default_rng(int(random_seed))
    totals: list[float] = []
    anns: list[float] = []
    mdds: list[float] = []

    for _ in range(int(random_iterations)):
        rr: list[float] = []
        for i, row in enumerate(rows):
            ym_cur = str(row["ym"])
            ym_prev = prev_months[i]
            rb = _safe_float(row.get("risk_budget", 0.0))
            nsel = int(_safe_float(row.get("n_selected", 0.0)) if np.isfinite(_safe_float(row.get("n_selected", 0.0))) else 0)
            if not ym_prev or ym_prev not in snap_by_month or nsel <= 0 or not np.isfinite(rb) or rb <= 0:
                rr.append(0.0)
                continue
            pool = [a for a in snap_by_month[ym_prev].head(int(pool_max_assets))["asset_id"].tolist() if a in mret.columns]
            if not pool:
                rr.append(0.0)
                continue
            take = min(nsel, len(pool))
            sel = rng.choice(pool, size=take, replace=False)
            vals = [float(mret.at[ym_cur, a]) if ym_cur in mret.index else 0.0 for a in sel]
            base = float(np.mean(vals)) if vals else 0.0
            rr.append(float(rb) * base)
        s = pd.Series(rr, dtype=float)
        totals.append(_total(s))
        anns.append(_ann(s))
        mdds.append(_mdd(s))

    strat_total = _total(monthly["ret"])
    strat_ann = _ann(monthly["ret"])
    strat_mdd = _mdd(monthly["ret"])
    mean_total = float(np.nanmean(np.asarray(totals, dtype=float)))

    return {
        "assumption": {
            "same_risk_budget_path": True,
            "same_n_selected_path": True,
            "random_pool": f"previous_month_snapshot_top{int(pool_max_assets)}",
            "iterations": int(random_iterations),
            "seed": int(random_seed),
        },
        "strategy": {
            "total_return": strat_total,
            "annualized_return": strat_ann,
            "max_drawdown": strat_mdd,
        },
        "random_distribution": {
            "mean_total_return": mean_total,
            "p10_total_return": float(np.nanquantile(np.asarray(totals, dtype=float), 0.10)),
            "p50_total_return": float(np.nanquantile(np.asarray(totals, dtype=float), 0.50)),
            "p90_total_return": float(np.nanquantile(np.asarray(totals, dtype=float), 0.90)),
            "mean_annualized_return": float(np.nanmean(np.asarray(anns, dtype=float))),
            "mean_max_drawdown": float(np.nanmean(np.asarray(mdds, dtype=float))),
        },
        "lift_vs_random_mean_total": float(strat_total / mean_total) if np.isfinite(mean_total) and mean_total > 0 else float("nan"),
        "prob_strategy_beats_random": float(np.mean(np.asarray(totals, dtype=float) < strat_total)),
    }


def _step2_walkforward_blocks(
    *,
    impact_dir: Path,
    returns_csv: Path,
    outdir: Path,
    train_end_years: list[int],
    top_k_options: str,
    impact_power_options: str,
    wmax_options: str,
    mom_lookback_options: str,
    mom_threshold_options: str,
    modes: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    py = sys.executable
    script = ROOT / "scripts" / "ops" / "run_canonical_systematic_eval.py"
    wf_root = outdir / "walkforward_runs"
    wf_root.mkdir(parents=True, exist_ok=True)

    for y in train_end_years:
        train_end = f"{int(y)}-12-31"
        test_year = int(y) + 1
        run_out = wf_root / f"train_to_{y}"
        cmd = [
            py,
            str(script),
            "--impact-dir",
            str(impact_dir),
            "--returns-csv",
            str(returns_csv),
            "--outdir",
            str(run_out),
            "--train-end",
            train_end,
            "--start-ym",
            "2019-01",
            "--top-k-options",
            top_k_options,
            "--impact-power-options",
            impact_power_options,
            "--wmax-options",
            wmax_options,
            "--mom-lookback-options",
            mom_lookback_options,
            "--mom-threshold-options",
            mom_threshold_options,
            "--modes",
            modes,
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
        subprocess.run(cmd, check=True, cwd=ROOT)

        ycsv = run_out / "yearly_systematic_eval.csv"
        if not ycsv.exists():
            rows.append(
                {
                    "train_end_year": int(y),
                    "test_year": int(test_year),
                    "status": "missing_yearly_eval",
                }
            )
            continue
        yd = pd.read_csv(ycsv)
        z = yd[yd["year"] == int(test_year)].copy()
        if z.empty:
            rows.append(
                {
                    "train_end_year": int(y),
                    "test_year": int(test_year),
                    "status": "no_test_row",
                }
            )
            continue
        r = z.iloc[0]
        rows.append(
            {
                "train_end_year": int(y),
                "test_year": int(test_year),
                "status": "ok",
                "strategy_total": _safe_float(r.get("strategy_total")),
                "eqw_total": _safe_float(r.get("eqw_total")),
                "alpha_total_vs_eqw": _safe_float(r.get("alpha_total_vs_eqw")),
                "worth_it_vs_eqw": bool(r.get("worth_it_vs_eqw")) if str(r.get("worth_it_vs_eqw")).strip() != "" else False,
                "source_dir": str(run_out),
            }
        )

    out = pd.DataFrame(rows)
    ok = out[out["status"] == "ok"].copy()
    payload = {
        "blocks_requested": int(len(train_end_years)),
        "blocks_ok": int(ok.shape[0]),
        "blocks_fail": int(len(train_end_years) - ok.shape[0]),
        "beat_rate_vs_eqw": _safe_float(ok["worth_it_vs_eqw"].mean()) if not ok.empty else float("nan"),
        "mean_alpha_total_vs_eqw": _safe_float(ok["alpha_total_vs_eqw"].mean()) if not ok.empty else float("nan"),
        "median_alpha_total_vs_eqw": _safe_float(ok["alpha_total_vs_eqw"].median()) if not ok.empty else float("nan"),
        "min_alpha_total_vs_eqw": _safe_float(ok["alpha_total_vs_eqw"].min()) if not ok.empty else float("nan"),
        "max_alpha_total_vs_eqw": _safe_float(ok["alpha_total_vs_eqw"].max()) if not ok.empty else float("nan"),
    }
    return out, payload


def _parse_int_list(s: str) -> list[int]:
    vals = [int(x.strip()) for x in str(s).split(",") if str(x).strip()]
    if not vals:
        raise ValueError("empty int list")
    return vals


def main() -> None:
    ap = argparse.ArgumentParser(description="Run robustness tests 1/2/3: costs, walk-forward blocks, random baseline.")
    ap.add_argument("--base-run-dir", default="", help="Systematic yearly run dir; default latest.")
    ap.add_argument("--outdir", default="", help="Output dir (default: results/portfolio_sim/<runid>_tests_123)")
    ap.add_argument("--real-cost-bps", type=float, default=10.0)
    ap.add_argument("--stress-cost-multiplier", type=float, default=2.0)
    ap.add_argument("--random-iterations", type=int, default=500)
    ap.add_argument("--random-seed", type=int, default=23)
    ap.add_argument("--random-pool-max-assets", type=int, default=80)
    ap.add_argument("--train-end-years", default="2020,2021,2022,2023,2024,2025")
    ap.add_argument("--top-k-options", default="44,52,64")
    ap.add_argument("--impact-power-options", default="0")
    ap.add_argument("--wmax-options", default="0.1")
    ap.add_argument("--mom-lookback-options", default="0")
    ap.add_argument("--mom-threshold-options", default="-0.02")
    ap.add_argument("--modes", default="const")
    args = ap.parse_args()

    base_run = Path(args.base_run_dir).resolve() if args.base_run_dir.strip() else _latest_systematic_run()
    monthly_csv = base_run / "monthly_systematic_eval.csv"
    sim_summary = base_run / "simulation_summary.json"
    if not monthly_csv.exists() or not sim_summary.exists():
        raise FileNotFoundError(f"missing monthly/simulation in {base_run}")

    run_id = _run_id()
    outdir = Path(args.outdir).resolve() if args.outdir.strip() else (ROOT / "results" / "portfolio_sim" / f"{run_id}_tests_123")
    outdir.mkdir(parents=True, exist_ok=True)

    monthly = pd.read_csv(monthly_csv).sort_values("ym").reset_index(drop=True)
    sim = json.loads(sim_summary.read_text(encoding="utf-8"))
    impact_dir = Path(sim["impact_dir"])
    returns_csv = Path(sim["returns_csv"])
    impact_csv = impact_dir / "impact_training_dataset.csv"
    if not impact_csv.exists() or not returns_csv.exists():
        raise FileNotFoundError("impact_training_dataset.csv or returns_wide_core.csv not found")

    mret = _build_monthly_returns(returns_csv)
    snap_by_month = _build_snapshots(impact_csv, max_assets_per_month=int(args.random_pool_max_assets))

    monthly_cost, step1 = _step1_cost_stress_and_turnover(
        monthly,
        mret,
        snap_by_month,
        real_cost_bps=float(args.real_cost_bps),
        stress_multiplier=float(args.stress_cost_multiplier),
    )
    step3 = _step3_random_baseline_same_exposure(
        monthly,
        mret,
        snap_by_month,
        random_iterations=int(args.random_iterations),
        random_seed=int(args.random_seed),
        pool_max_assets=int(args.random_pool_max_assets),
    )
    wf_df, step2 = _step2_walkforward_blocks(
        impact_dir=impact_dir,
        returns_csv=returns_csv,
        outdir=outdir,
        train_end_years=_parse_int_list(args.train_end_years),
        top_k_options=str(args.top_k_options),
        impact_power_options=str(args.impact_power_options),
        wmax_options=str(args.wmax_options),
        mom_lookback_options=str(args.mom_lookback_options),
        mom_threshold_options=str(args.mom_threshold_options),
        modes=str(args.modes),
    )

    monthly_cost_csv = outdir / "monthly_with_turnover_and_costs.csv"
    wf_csv = outdir / "walkforward_blocks.csv"
    summary_json = outdir / "tests_123_summary.json"
    monthly_cost.to_csv(monthly_cost_csv, index=False)
    wf_df.to_csv(wf_csv, index=False)

    payload = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_dir": str(base_run),
        "tests": {
            "1_cost_stress": step1,
            "2_walkforward_blocks": step2,
            "3_random_baseline_same_exposure": step3,
        },
        "artifacts": {
            "monthly_with_turnover_and_costs_csv": str(monthly_cost_csv),
            "walkforward_blocks_csv": str(wf_csv),
        },
    }
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "ok", "outdir": str(outdir), "summary_json": str(summary_json)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
