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

from engine.structural.run_manifest import write_run_manifest  # noqa: E402
from execution.cost_model import summarize_return_series  # noqa: E402


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _latest_attack25_dir() -> Path:
    base = ROOT / "results" / "portfolio_sim"
    if not base.exists():
        raise FileNotFoundError(f"missing base dir: {base}")
    runs = sorted(
        [p for p in base.iterdir() if p.is_dir() and "ultra_return_compact_attack25" in p.name],
        key=lambda p: p.name,
        reverse=True,
    )
    for run in runs:
        if (run / "monthly_systematic_eval.csv").exists() and (run / "simulation_summary.json").exists():
            return run
    raise FileNotFoundError("no attack25 profile dir found")


def _build_monthly_returns(returns_csv: Path) -> pd.DataFrame:
    d = pd.read_csv(returns_csv)
    if "date" not in d.columns:
        raise ValueError(f"missing date column: {returns_csv}")
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    assets = [c for c in d.columns if c != "date"]
    if not assets:
        raise ValueError(f"no assets in {returns_csv}")
    d = d[["date", *assets]].copy()
    d[assets] = d[assets].apply(pd.to_numeric, errors="coerce")
    d = d.copy()
    d["ym"] = d["date"].dt.to_period("M").astype(str)
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for ym, g in d.groupby("ym"):
        arr = g[assets].to_numpy(dtype=float)
        rows.append(np.prod(1.0 + np.nan_to_num(arr, nan=0.0), axis=0) - 1.0)
        labels.append(str(ym))
    out = pd.DataFrame(np.vstack(rows), index=labels, columns=assets).sort_index()
    return out


def _calendar_total(ret: pd.Series, labels: pd.Series) -> pd.Series:
    d = pd.DataFrame({"label": labels.astype(str), "ret": pd.to_numeric(ret, errors="coerce").fillna(0.0).astype(float)})
    return d.groupby("label", as_index=True)["ret"].apply(lambda s: float(np.prod(1.0 + s.to_numpy(dtype=float)) - 1.0))


def _year_and_semester(labels_ym: pd.Series) -> tuple[pd.Series, pd.Series]:
    dt = pd.to_datetime(labels_ym.astype(str) + "-01", errors="coerce")
    year = dt.dt.year.astype("Int64").astype(str)
    sem = year + "H" + (dt.dt.month.apply(lambda m: "1" if int(m) <= 6 else "2").astype(str))
    return year, sem


def _block_bootstrap_alpha(alpha: pd.Series, *, n_boot: int, block_len: int, seed: int) -> dict[str, float]:
    a = pd.to_numeric(alpha, errors="coerce").dropna().astype(float).to_numpy(dtype=float)
    n = int(len(a))
    if n <= 0:
        return {"alpha_mean": float("nan"), "alpha_ci95_lo": float("nan"), "alpha_ci95_hi": float("nan"), "prob_alpha_positive": float("nan")}
    lb = int(max(2, block_len))
    rng = np.random.default_rng(int(seed))
    means: list[float] = []
    for _ in range(int(max(100, n_boot))):
        idxs: list[int] = []
        while len(idxs) < n:
            st = int(rng.integers(0, max(1, n - lb + 1)))
            idxs.extend(range(st, min(n, st + lb)))
        idx = np.asarray(idxs[:n], dtype=int)
        means.append(float(np.mean(a[idx])))
    arr = np.asarray(means, dtype=float)
    return {
        "alpha_mean": float(np.mean(arr)),
        "alpha_ci95_lo": float(np.quantile(arr, 0.025)),
        "alpha_ci95_hi": float(np.quantile(arr, 0.975)),
        "prob_alpha_positive": float(np.mean(arr > 0.0)),
    }


def _momentum_baseline(
    monthly: pd.DataFrame,
    mret: pd.DataFrame,
    *,
    lookback: int,
    same_risk_budget: bool,
) -> pd.Series:
    yms = monthly["ym"].astype(str).tolist()
    vals: list[float] = []
    lb = int(max(1, lookback))
    for i, ym in enumerate(yms):
        if ym not in mret.index:
            vals.append(0.0)
            continue
        n_sel = int(max(1, pd.to_numeric(monthly.iloc[i].get("n_selected", 0), errors="coerce") or 1))
        rb = float(pd.to_numeric(monthly.iloc[i].get("risk_budget", 1.0), errors="coerce"))
        if i < lb:
            vals.append(0.0)
            continue
        hist_idx = [x for x in yms[i - lb : i] if x in mret.index]
        if len(hist_idx) < lb:
            vals.append(0.0)
            continue
        hist = mret.loc[hist_idx]
        scores = (1.0 + hist).prod(axis=0) - 1.0
        scores = pd.to_numeric(scores, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if scores.empty:
            vals.append(0.0)
            continue
        sel = scores.sort_values(ascending=False).head(n_sel).index.tolist()
        r = pd.to_numeric(mret.loc[ym, sel], errors="coerce").dropna().astype(float)
        base = float(r.mean()) if not r.empty else 0.0
        vals.append(float(rb) * base if same_risk_budget else base)
    return pd.Series(vals, index=monthly.index, dtype=float)


def _random_baseline_distribution(
    monthly: pd.DataFrame,
    mret: pd.DataFrame,
    *,
    n_iter: int,
    same_risk_budget: bool,
    seed: int,
) -> tuple[pd.Series, dict[str, Any]]:
    yms = monthly["ym"].astype(str).tolist()
    rng = np.random.default_rng(int(seed))
    mat = np.zeros((int(n_iter), len(yms)), dtype=float)
    assets = mret.columns.tolist()
    if not assets:
        return pd.Series(np.zeros(len(yms)), index=monthly.index, dtype=float), {
            "n_iter": int(n_iter),
            "status": "empty_pool",
        }

    for j in range(int(n_iter)):
        for i, ym in enumerate(yms):
            if ym not in mret.index:
                continue
            n_sel = int(max(1, pd.to_numeric(monthly.iloc[i].get("n_selected", 0), errors="coerce") or 1))
            rb = float(pd.to_numeric(monthly.iloc[i].get("risk_budget", 1.0), errors="coerce"))
            row = pd.to_numeric(mret.loc[ym, assets], errors="coerce").dropna().astype(float)
            if row.empty:
                continue
            take = int(min(n_sel, len(row)))
            sel = rng.choice(row.index.to_numpy(dtype=object), size=take, replace=False)
            base = float(pd.to_numeric(row.loc[sel], errors="coerce").mean())
            mat[j, i] = float(rb) * base if same_risk_budget else base

    mean_path = pd.Series(np.mean(mat, axis=0), index=monthly.index, dtype=float)
    totals = np.asarray([float(np.prod(1.0 + mat[k, :]) - 1.0) for k in range(mat.shape[0])], dtype=float)
    strat_total = float(np.prod(1.0 + pd.to_numeric(monthly["ret"], errors="coerce").fillna(0.0).to_numpy(dtype=float)) - 1.0)
    payload = {
        "n_iter": int(n_iter),
        "seed": int(seed),
        "total_return_mean": float(np.mean(totals)),
        "total_return_p10": float(np.quantile(totals, 0.10)),
        "total_return_p50": float(np.quantile(totals, 0.50)),
        "total_return_p90": float(np.quantile(totals, 0.90)),
        "prob_strategy_beats_random_total": float(np.mean(strat_total > totals)),
    }
    return mean_path, payload


def _compare_vs_strategy(
    strategy: pd.Series,
    baseline: pd.Series,
    *,
    ym_labels: pd.Series,
    bootstrap_iter: int,
    bootstrap_block_len: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    alpha = pd.to_numeric(strategy, errors="coerce").fillna(0.0).astype(float) - pd.to_numeric(baseline, errors="coerce").fillna(0.0).astype(float)
    year, sem = _year_and_semester(ym_labels)
    s_y = _calendar_total(strategy, year)
    b_y = _calendar_total(baseline, year)
    s_h = _calendar_total(strategy, sem)
    b_h = _calendar_total(baseline, sem)
    common_y = s_y.index.intersection(b_y.index)
    common_h = s_h.index.intersection(b_h.index)
    year_win = float(np.mean((s_y.loc[common_y] > b_y.loc[common_y]).to_numpy(dtype=bool))) if len(common_y) > 0 else float("nan")
    sem_win = float(np.mean((s_h.loc[common_h] > b_h.loc[common_h]).to_numpy(dtype=bool))) if len(common_h) > 0 else float("nan")
    boot = _block_bootstrap_alpha(alpha, n_boot=int(bootstrap_iter), block_len=int(bootstrap_block_len), seed=int(bootstrap_seed))
    return {
        "alpha_monthly_mean": float(alpha.mean()),
        "alpha_monthly_positive_rate": float((alpha > 0.0).mean()),
        "outperformance_year_win_rate": year_win,
        "outperformance_semester_win_rate": sem_win,
        "bootstrap": boot,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Baselines completos para um perfil sistemático (attack25).")
    ap.add_argument("--profile-dir", default="", help="Diretório do perfil (default: latest ultra_return_compact_attack25)")
    ap.add_argument("--outdir", default="", help="Diretório de saída (default: <profile-dir>/baselines_attack25_<run_id>)")
    ap.add_argument("--momentum-lookback", type=int, default=12)
    ap.add_argument("--same-risk-budget", type=int, default=1)
    ap.add_argument("--random-iter", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--bootstrap-iter", type=int, default=5000)
    ap.add_argument("--bootstrap-block-len", type=int, default=3)
    args = ap.parse_args()

    profile_dir = Path(args.profile_dir).resolve() if str(args.profile_dir).strip() else _latest_attack25_dir()
    monthly_csv = profile_dir / "monthly_systematic_eval.csv"
    sim_json = profile_dir / "simulation_summary.json"
    if not monthly_csv.exists() or not sim_json.exists():
        raise FileNotFoundError(f"missing required files in {profile_dir}")

    sim = json.loads(sim_json.read_text(encoding="utf-8"))
    returns_csv = Path(str(sim.get("returns_csv", ""))).resolve()
    if not returns_csv.exists():
        raise FileNotFoundError(f"returns_csv not found: {returns_csv}")

    outdir = Path(args.outdir).resolve() if str(args.outdir).strip() else (profile_dir / f"baselines_attack25_{_run_id()}")
    outdir.mkdir(parents=True, exist_ok=True)

    monthly = pd.read_csv(monthly_csv)
    for c in ["ret", "eqw_ret", "mkt_ret", "risk_budget", "n_selected"]:
        if c in monthly.columns:
            monthly[c] = pd.to_numeric(monthly[c], errors="coerce")
    monthly["ym"] = monthly["ym"].astype(str)
    mret = _build_monthly_returns(returns_csv)
    mret = mret.reindex(monthly["ym"].tolist()).fillna(0.0)

    same_budget = bool(int(args.same_risk_budget))
    strategy = pd.to_numeric(monthly["ret"], errors="coerce").fillna(0.0).astype(float)
    eqw_full = pd.to_numeric(monthly["eqw_ret"], errors="coerce").fillna(0.0).astype(float)
    market_full = pd.to_numeric(monthly["mkt_ret"], errors="coerce").fillna(0.0).astype(float)
    rb = pd.to_numeric(monthly["risk_budget"], errors="coerce").fillna(1.0).astype(float)
    eqw_budget = rb * eqw_full if same_budget else eqw_full.copy()
    mkt_budget = rb * market_full if same_budget else market_full.copy()
    momentum = _momentum_baseline(
        monthly=monthly,
        mret=mret,
        lookback=int(args.momentum_lookback),
        same_risk_budget=same_budget,
    )
    random_mean, random_payload = _random_baseline_distribution(
        monthly=monthly,
        mret=mret,
        n_iter=int(args.random_iter),
        same_risk_budget=same_budget,
        seed=int(args.seed),
    )

    baselines = {
        "strategy_profile": strategy,
        "buy_hold_market_full": market_full,
        "equal_weight_full": eqw_full,
        "equal_weight_same_budget": eqw_budget,
        "market_same_budget": mkt_budget,
        "momentum_same_budget": momentum,
        "random_same_budget_mean_path": random_mean,
    }

    metrics: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    for name, series in baselines.items():
        metrics[name] = summarize_return_series(series)
        if name != "strategy_profile":
            comparisons[name] = _compare_vs_strategy(
                strategy,
                series,
                ym_labels=monthly["ym"],
                bootstrap_iter=int(args.bootstrap_iter),
                bootstrap_block_len=int(args.bootstrap_block_len),
                bootstrap_seed=int(args.seed),
            )

    monthly_out = pd.DataFrame({"ym": monthly["ym"].astype(str)})
    for name, series in baselines.items():
        monthly_out[name] = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float).to_numpy(dtype=float)
    monthly_out_path = outdir / "baselines_monthly.csv"
    monthly_out.to_csv(monthly_out_path, index=False)

    payload = {
        "status": "ok",
        "profile_dir": str(profile_dir),
        "returns_csv": str(returns_csv),
        "assumptions": {
            "same_risk_budget": bool(same_budget),
            "momentum_lookback_months": int(args.momentum_lookback),
            "random_iterations": int(args.random_iter),
            "bootstrap_iterations": int(args.bootstrap_iter),
            "bootstrap_block_len": int(args.bootstrap_block_len),
            "seed": int(args.seed),
        },
        "metrics": metrics,
        "comparisons_vs_strategy": comparisons,
        "random_distribution": random_payload,
        "paths": {
            "monthly_baselines_csv": str(monthly_out_path),
        },
    }
    report_path = outdir / "baselines_report.json"
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    write_run_manifest(
        outdir=outdir,
        script="scripts/bench/portfolio/run_baselines_for_profile.py",
        params={
            "profile_dir": str(profile_dir),
            "momentum_lookback": int(args.momentum_lookback),
            "same_risk_budget": int(args.same_risk_budget),
            "random_iter": int(args.random_iter),
            "seed": int(args.seed),
            "bootstrap_iter": int(args.bootstrap_iter),
            "bootstrap_block_len": int(args.bootstrap_block_len),
        },
        paths={"report_json": str(report_path), "monthly_baselines_csv": str(monthly_out_path)},
        gates={"report_created": report_path.exists(), "monthly_csv_created": monthly_out_path.exists()},
    )

    print(json.dumps({"status": "ok", "outdir": str(outdir), "report": str(report_path)}))


if __name__ == "__main__":
    main()
