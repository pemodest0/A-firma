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
    t = float(np.prod(1.0 + s.to_numpy(dtype=float)) - 1.0)
    return float((1.0 + t) ** (12.0 / float(len(s))) - 1.0)


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
        if (run / "simulation_summary.json").exists():
            return run
    raise FileNotFoundError("no systematic yearly run found")


def _parse_int_list(s: str) -> list[int]:
    vals = [int(x.strip()) for x in str(s).split(",") if str(x).strip()]
    if not vals:
        raise ValueError("empty int list")
    return vals


def _run_canonical_topk(*, impact_dir: Path, returns_csv: Path, top_k: int, outdir: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "ops" / "run_canonical_systematic_eval.py"),
        "--impact-dir",
        str(impact_dir),
        "--returns-csv",
        str(returns_csv),
        "--outdir",
        str(outdir),
        "--train-end",
        "2023-12-31",
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
    summary = json.loads((outdir / "systematic_summary.json").read_text(encoding="utf-8"))
    sim = json.loads((outdir / "simulation_summary.json").read_text(encoding="utf-8"))
    metrics = sim.get("best_metrics", {}) if isinstance(sim, dict) else {}
    return {
        "top_k": int(top_k),
        "run_dir": str(outdir),
        "total_return": _safe_float(summary.get("strategy_total")),
        "ann_return": _safe_float(summary.get("strategy_ann")),
        "max_drawdown": _safe_float(summary.get("strategy_max_drop")),
        "worth_it_rate_vs_eqw": _safe_float(summary.get("worth_it_rate_vs_eqw")),
        "monthly_alpha_prob_positive_vs_eqw": _safe_float(summary.get("monthly_alpha_prob_positive_vs_eqw")),
        "full_alpha_recent6": _safe_float(metrics.get("full_alpha_recent6")),
    }


def _build_snapshots(impact_csv: Path, *, max_assets_per_month: int = 120) -> dict[str, pd.DataFrame]:
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
    return {
        str(ym): g[["asset_id", "impact_global"]].sort_values("impact_global", ascending=False).reset_index(drop=True)
        for ym, g in snap.groupby("ym")
    }


def _build_monthly_returns(returns_csv: Path) -> pd.DataFrame:
    d = pd.read_csv(returns_csv)
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    assets = [c for c in d.columns if c != "date"]
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


def _apply_min_streak(sig: pd.Series, min_streak: int) -> pd.Series:
    s = pd.to_numeric(sig, errors="coerce").fillna(0.0).astype(int).clip(lower=0, upper=1)
    if int(min_streak) <= 1:
        return s.astype(int)
    run = 0
    out: list[int] = []
    for v in s.to_numpy(dtype=int):
        run = (run + 1) if int(v) == 1 else 0
        out.append(1 if run >= int(min_streak) else 0)
    return pd.Series(out, index=s.index, dtype=int)


def _build_signal(monthly: pd.DataFrame, *, rule: str, min_streak: int) -> pd.Series:
    rb_stress = monthly["risk_bucket"].astype(str).str.lower().eq("stress")
    defense = pd.to_numeric(monthly["defense_active"], errors="coerce").fillna(0.0) > 0
    decel = pd.to_numeric(monthly["decel_active"], errors="coerce").fillna(0.0) > 0

    key = str(rule).strip().lower()
    if key in {"default", "stress_or_defense_or_decel"}:
        raw = rb_stress | defense | decel
    elif key in {"defense_or_decel", "def_or_decel"}:
        raw = defense | decel
    elif key in {"defense_and_decel", "def_and_decel"}:
        raw = defense & decel
    elif key in {"defense_only", "def_only"}:
        raw = defense
    elif key in {"decel_only"}:
        raw = decel
    elif key in {"stress_only"}:
        raw = rb_stress
    else:
        raise ValueError(f"unsupported signal rule: {rule}")

    return _apply_min_streak(raw.astype(int), int(min_streak))


def _lag_test(monthly: pd.DataFrame, lags: list[int], *, signal_rule: str, signal_min_streak: int) -> pd.DataFrame:
    d = monthly.copy().sort_values("ym").reset_index(drop=True)
    d["ret"] = pd.to_numeric(d["ret"], errors="coerce").fillna(0.0)
    d["eqw_ret"] = pd.to_numeric(d["eqw_ret"], errors="coerce").fillna(0.0)
    d["motor_ret"] = pd.to_numeric(d["motor_ret"], errors="coerce").fillna(0.0)
    signal = _build_signal(d, rule=signal_rule, min_streak=int(signal_min_streak)).astype(int)

    rows: list[dict[str, Any]] = []
    for lag in lags:
        used = signal.shift(int(lag)).fillna(0).astype(int)
        guided = np.where(used.to_numpy(dtype=int) == 1, d["motor_ret"].to_numpy(dtype=float), d["ret"].to_numpy(dtype=float))
        gr = pd.Series(guided, index=d.index, dtype=float)
        alpha = gr - d["eqw_ret"]
        rows.append(
            {
                "lag_months": int(lag),
                "guided_total_return": _total(gr),
                "guided_ann_return": _ann(gr),
                "guided_max_drawdown": _mdd(gr),
                "guided_alpha_total_vs_eqw": _total(gr) - _total(d["eqw_ret"]),
                "guided_alpha_positive_rate_vs_eqw": _safe_float((alpha > 0.0).mean()),
                "signal_rate_used": _safe_float(used.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("lag_months").reset_index(drop=True)


def _null_shuffle_test(
    monthly: pd.DataFrame,
    *,
    lag: int,
    n_iter: int,
    seed: int,
    signal_rule: str,
    signal_min_streak: int,
) -> dict[str, Any]:
    d = monthly.copy().sort_values("ym").reset_index(drop=True)
    d["ret"] = pd.to_numeric(d["ret"], errors="coerce").fillna(0.0)
    d["eqw_ret"] = pd.to_numeric(d["eqw_ret"], errors="coerce").fillna(0.0)
    d["motor_ret"] = pd.to_numeric(d["motor_ret"], errors="coerce").fillna(0.0)

    signal = _build_signal(d, rule=signal_rule, min_streak=int(signal_min_streak)).astype(int)
    used_true = signal.shift(int(lag)).fillna(0).astype(int)
    guided_true = pd.Series(
        np.where(used_true.to_numpy(dtype=int) == 1, d["motor_ret"].to_numpy(dtype=float), d["ret"].to_numpy(dtype=float)),
        index=d.index,
        dtype=float,
    )
    alpha_true = _total(guided_true) - _total(d["eqw_ret"])

    rng = np.random.default_rng(int(seed))
    alpha_null: list[float] = []
    for _ in range(int(n_iter)):
        shuffled = used_true.sample(frac=1.0, replace=False, random_state=int(rng.integers(0, 2_000_000))).to_numpy(dtype=int)
        g = pd.Series(np.where(shuffled == 1, d["motor_ret"].to_numpy(dtype=float), d["ret"].to_numpy(dtype=float)), index=d.index, dtype=float)
        alpha_null.append(_total(g) - _total(d["eqw_ret"]))

    arr = np.asarray(alpha_null, dtype=float)
    return {
        "lag_months": int(lag),
        "iterations": int(n_iter),
        "alpha_total_true_vs_eqw": float(alpha_true),
        "alpha_total_null_mean": float(np.mean(arr)),
        "alpha_total_null_p90": float(np.quantile(arr, 0.90)),
        "alpha_total_null_p95": float(np.quantile(arr, 0.95)),
        "prob_true_beats_null": float(np.mean(arr < alpha_true)),
    }


def _cost_3x_test(monthly: pd.DataFrame, mret: pd.DataFrame, snaps: dict[str, pd.DataFrame], *, cost_bps: float) -> dict[str, Any]:
    d = monthly.copy().sort_values("ym").reset_index(drop=True)
    d["ret"] = pd.to_numeric(d["ret"], errors="coerce").fillna(0.0)
    d["eqw_ret"] = pd.to_numeric(d["eqw_ret"], errors="coerce").fillna(0.0)

    months = d["ym"].astype(str).tolist()
    prev_months = [months[i - 1] if i > 0 else None for i in range(len(months))]
    pre_w: dict[str, float] = {}
    cash_pre = 1.0
    turnovers: list[float] = []

    for i, row in d.iterrows():
        ym_cur = str(row["ym"])
        ym_prev = prev_months[i]
        rb = _safe_float(row.get("risk_budget", 0.0))
        nsel = int(_safe_float(row.get("n_selected", 0.0)) if np.isfinite(_safe_float(row.get("n_selected", 0.0))) else 0)
        eff_k = int(_safe_float(row.get("effective_top_k", nsel)) if np.isfinite(_safe_float(row.get("effective_top_k", nsel))) else nsel)

        target: dict[str, float] = {}
        if ym_prev and ym_prev in snaps and nsel > 0 and rb > 0:
            sel = [a for a in snaps[ym_prev].head(eff_k)["asset_id"].tolist() if a in mret.columns][:nsel]
            if sel:
                w_each = float(rb) / float(len(sel))
                for a in sel:
                    target[a] = w_each
        cash_target = max(0.0, 1.0 - float(sum(target.values())))

        keys = set(pre_w.keys()) | set(target.keys())
        l1 = sum(abs(float(target.get(a, 0.0)) - float(pre_w.get(a, 0.0))) for a in keys) + abs(cash_target - cash_pre)
        turnover = 0.5 * l1
        turnovers.append(float(turnover))

        gross_ret = float(row["ret"])
        denom = 1.0 + gross_ret
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

    d["turnover"] = turnovers
    cost_rate = float(cost_bps) / 10000.0
    d["ret_cost_3x"] = d["ret"] - d["turnover"] * cost_rate
    return {
        "cost_bps": float(cost_bps),
        "avg_turnover": _safe_float(d["turnover"].mean()),
        "strategy_total_no_cost": _total(d["ret"]),
        "strategy_total_cost_3x": _total(d["ret_cost_3x"]),
        "strategy_ann_no_cost": _ann(d["ret"]),
        "strategy_ann_cost_3x": _ann(d["ret_cost_3x"]),
        "strategy_mdd_no_cost": _mdd(d["ret"]),
        "strategy_mdd_cost_3x": _mdd(d["ret_cost_3x"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run gain tests 1..4: lag, null shuffle, cost 3x, subuniverses.")
    ap.add_argument("--base-run-dir", default="", help="Base systematic run; default latest.")
    ap.add_argument("--subuniverses", default="10,20,40,80")
    ap.add_argument("--lags", default="0,1,2")
    ap.add_argument("--signal-rule", default="stress_or_defense_or_decel")
    ap.add_argument("--signal-min-streak", type=int, default=1)
    ap.add_argument("--null-lag", type=int, default=1)
    ap.add_argument("--null-iterations", type=int, default=800)
    ap.add_argument("--null-seed", type=int, default=23)
    ap.add_argument("--cost-3x-bps", type=float, default=30.0)
    ap.add_argument("--outdir", default="", help="Output dir (default: results/portfolio_sim/<runid>_gain_tests_1234)")
    args = ap.parse_args()

    base_run = Path(args.base_run_dir).resolve() if args.base_run_dir.strip() else _latest_systematic_run()
    sim = json.loads((base_run / "simulation_summary.json").read_text(encoding="utf-8"))
    impact_dir = Path(sim["impact_dir"])
    returns_csv = Path(sim["returns_csv"])
    impact_csv = impact_dir / "impact_training_dataset.csv"
    if not impact_csv.exists():
        raise FileNotFoundError(f"missing {impact_csv}")

    run_id = _run_id()
    outdir = Path(args.outdir).resolve() if args.outdir.strip() else (ROOT / "results" / "portfolio_sim" / f"{run_id}_gain_tests_1234")
    outdir.mkdir(parents=True, exist_ok=True)
    sub_root = outdir / "subuniverses"
    sub_root.mkdir(parents=True, exist_ok=True)

    # Test 4: subuniverses
    sub_list = _parse_int_list(args.subuniverses)
    sub_rows: list[dict[str, Any]] = []
    monthly_by_topk: dict[int, pd.DataFrame] = {}
    for top_k in sub_list:
        rdir = sub_root / f"topk_{int(top_k)}"
        m = _run_canonical_topk(impact_dir=impact_dir, returns_csv=returns_csv, top_k=int(top_k), outdir=rdir)
        sub_rows.append(m)
        monthly_by_topk[int(top_k)] = pd.read_csv(rdir / "monthly_systematic_eval.csv")
    sub_df = pd.DataFrame(sub_rows).sort_values("top_k").reset_index(drop=True)
    sub_df.to_csv(outdir / "subuniverses_summary.csv", index=False)

    # Select best by total return then drawdown.
    best_row = sub_df.sort_values(["total_return", "max_drawdown"], ascending=[False, False]).iloc[0].to_dict()
    best_topk = int(best_row["top_k"])
    best_monthly = monthly_by_topk[best_topk]

    # Test 1: lag robustness
    lag_df = _lag_test(
        best_monthly,
        _parse_int_list(args.lags),
        signal_rule=str(args.signal_rule),
        signal_min_streak=int(args.signal_min_streak),
    )
    lag_df.to_csv(outdir / "lag_test_summary.csv", index=False)

    # Test 2: null shuffled signal
    null_payload = _null_shuffle_test(
        best_monthly,
        lag=int(args.null_lag),
        n_iter=int(args.null_iterations),
        seed=int(args.null_seed),
        signal_rule=str(args.signal_rule),
        signal_min_streak=int(args.signal_min_streak),
    )

    # Test 3: cost 3x
    snaps = _build_snapshots(impact_csv, max_assets_per_month=120)
    mret = _build_monthly_returns(returns_csv)
    cost_payload = _cost_3x_test(
        best_monthly,
        mret,
        snaps,
        cost_bps=float(args.cost_3x_bps),
    )

    launch_flags = {
        "test1_lag1_alpha_non_negative": bool(
            _safe_float(lag_df.loc[lag_df["lag_months"] == 1, "guided_alpha_total_vs_eqw"].iloc[0])
            >= 0.0
            if (lag_df["lag_months"] == 1).any()
            else False
        ),
        "test1_lag2_alpha_non_negative": bool(
            _safe_float(lag_df.loc[lag_df["lag_months"] == 2, "guided_alpha_total_vs_eqw"].iloc[0])
            >= 0.0
            if (lag_df["lag_months"] == 2).any()
            else False
        ),
        "test2_true_signal_beats_null_p90": bool(
            _safe_float(null_payload.get("alpha_total_true_vs_eqw")) > _safe_float(null_payload.get("alpha_total_null_p90"))
        ),
        "test3_cost3x_total_positive": bool(_safe_float(cost_payload.get("strategy_total_cost_3x")) > 0.0),
        "test4_subuniverse_majority_positive_total": bool((_safe_float((sub_df["total_return"] > 0).mean())) >= 0.5),
    }

    ready = all(bool(v) for v in launch_flags.values())
    payload = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_base_run_dir": str(base_run),
        "selected_best_top_k": int(best_topk),
        "signal_config": {
            "rule": str(args.signal_rule),
            "min_streak": int(args.signal_min_streak),
        },
        "tests": {
            "1_lag_robustness": {
                "csv": str(outdir / "lag_test_summary.csv"),
                "rows": lag_df.to_dict(orient="records"),
            },
            "2_null_shuffle": null_payload,
            "3_cost_stress_3x": cost_payload,
            "4_subuniverses": {
                "csv": str(outdir / "subuniverses_summary.csv"),
                "rows": sub_df.to_dict(orient="records"),
            },
        },
        "launch_flags_1_to_4": launch_flags,
        "launch_ready_round_1_to_4": bool(ready),
    }
    out_json = outdir / "gain_tests_1234_summary.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "ok", "outdir": str(outdir), "summary_json": str(out_json), "launch_ready_round_1_to_4": bool(ready)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
