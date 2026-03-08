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


def _parse_int_list(s: str) -> list[int]:
    vals = [int(x.strip()) for x in str(s).split(",") if str(x).strip()]
    if not vals:
        raise ValueError("empty int list")
    return vals


def _parse_str_list(s: str) -> list[str]:
    vals = [str(x).strip() for x in str(s).split(",") if str(x).strip()]
    if not vals:
        raise ValueError("empty str list")
    return vals


def _total(ret: pd.Series | np.ndarray) -> float:
    s = pd.Series(ret)
    s = pd.to_numeric(s, errors="coerce").dropna().astype(float)
    if s.empty:
        return float("nan")
    return float(np.prod(1.0 + s.to_numpy(dtype=float)) - 1.0)


def _ann(ret: pd.Series | np.ndarray) -> float:
    s = pd.Series(ret)
    s = pd.to_numeric(s, errors="coerce").dropna().astype(float)
    if s.empty:
        return float("nan")
    t = float(np.prod(1.0 + s.to_numpy(dtype=float)) - 1.0)
    return float((1.0 + t) ** (12.0 / float(len(s))) - 1.0)


def _mdd(ret: pd.Series | np.ndarray) -> float:
    s = pd.Series(ret)
    s = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
    if s.empty:
        return float("nan")
    eq = np.cumprod(1.0 + s.to_numpy(dtype=float))
    peak = np.maximum.accumulate(eq)
    dd = eq / np.where(peak == 0.0, np.nan, peak) - 1.0
    dd = dd[np.isfinite(dd)]
    return float(np.min(dd)) if dd.size > 0 else float("nan")


def _infer_subuniverse_summary_from_base_run(base_run: Path) -> Path | None:
    # Keep test-5 tied to the same evaluation lineage, avoiding global "latest" bleed.
    cur = base_run.resolve()
    for _ in range(10):
        candidate = cur / "subuniverses_summary.csv"
        if candidate.exists():
            return candidate
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def _autobuild_subuniverse_summary(base_run: Path, outdir: Path) -> Path | None:
    auto_root = outdir / "autogen_gain_tests_1234"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "ops" / "run_gain_tests_1234.py"),
        "--base-run-dir",
        str(base_run),
        "--subuniverses",
        "10,20,40,80",
        "--outdir",
        str(auto_root),
    ]
    try:
        subprocess.run(cmd, cwd=ROOT, check=True)
    except Exception:
        return None
    p = auto_root / "subuniverses_summary.csv"
    return p if p.exists() else None


def _signal_on(monthly: pd.DataFrame) -> pd.Series:
    return _build_signal(
        monthly,
        rule="defense_or_decel",
        min_streak=1,
    )


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


def _test1_walkforward(
    *,
    impact_dir: Path,
    returns_csv: Path,
    outdir: Path,
    train_end_years: list[int],
    top_k: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    wf_root = outdir / "walkforward"
    wf_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for y in train_end_years:
        run_out = wf_root / f"train_to_{int(y)}"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "ops" / "run_canonical_systematic_eval.py"),
            "--impact-dir",
            str(impact_dir),
            "--returns-csv",
            str(returns_csv),
            "--outdir",
            str(run_out),
            "--train-end",
            f"{int(y)}-12-31",
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
        ycsv = run_out / "yearly_systematic_eval.csv"
        if not ycsv.exists():
            rows.append({"train_end_year": int(y), "test_year": int(y) + 1, "status": "missing_yearly_eval"})
            continue
        d = pd.read_csv(ycsv)
        z = d[d["year"] == (int(y) + 1)]
        if z.empty:
            rows.append({"train_end_year": int(y), "test_year": int(y) + 1, "status": "no_test_row"})
            continue
        r = z.iloc[0]
        rows.append(
            {
                "train_end_year": int(y),
                "test_year": int(y) + 1,
                "status": "ok",
                "strategy_total": _safe_float(r.get("strategy_total")),
                "eqw_total": _safe_float(r.get("eqw_total")),
                "alpha_total_vs_eqw": _safe_float(r.get("alpha_total_vs_eqw")),
                "worth_it_vs_eqw": bool(r.get("worth_it_vs_eqw")),
                "source_dir": str(run_out),
            }
        )
    df = pd.DataFrame(rows)
    ok = df[df["status"] == "ok"].copy()
    summary = {
        "blocks_requested": int(len(train_end_years)),
        "blocks_ok": int(ok.shape[0]),
        "blocks_fail": int(len(train_end_years) - ok.shape[0]),
        "beat_rate_vs_eqw": _safe_float(ok["worth_it_vs_eqw"].mean()) if not ok.empty else float("nan"),
        "mean_alpha_total_vs_eqw": _safe_float(ok["alpha_total_vs_eqw"].mean()) if not ok.empty else float("nan"),
        "min_alpha_total_vs_eqw": _safe_float(ok["alpha_total_vs_eqw"].min()) if not ok.empty else float("nan"),
        "max_alpha_total_vs_eqw": _safe_float(ok["alpha_total_vs_eqw"].max()) if not ok.empty else float("nan"),
    }
    return df, summary


def _test2_crisis(monthly: pd.DataFrame) -> dict[str, Any]:
    d = monthly.copy()
    d["ret"] = pd.to_numeric(d["ret"], errors="coerce")
    d["eqw_ret"] = pd.to_numeric(d["eqw_ret"], errors="coerce")
    d = d.dropna(subset=["ret", "eqw_ret"]).copy()
    d["alpha"] = d["ret"] - d["eqw_ret"]
    q20 = float(np.quantile(d["eqw_ret"].to_numpy(dtype=float), 0.2))
    c1 = d[d["eqw_ret"] <= q20].copy()
    c2 = d[d["eqw_ret"] <= -0.05].copy()

    def pack(z: pd.DataFrame) -> dict[str, Any]:
        if z.empty:
            return {"months": 0}
        return {
            "months": int(len(z)),
            "strategy_total": _total(z["ret"]),
            "eqw_total": _total(z["eqw_ret"]),
            "alpha_total": _total(z["ret"]) - _total(z["eqw_ret"]),
            "alpha_mean": _safe_float(z["alpha"].mean()),
            "alpha_positive_rate": _safe_float((z["alpha"] > 0.0).mean()),
        }

    return {
        "worst_20pct_eqw_months": pack(c1),
        "eqw_le_minus_5pct_months": pack(c2),
    }


def _test3_null(monthly: pd.DataFrame, *, signal: pd.Series, lag: int, n_iter: int, seed: int) -> dict[str, Any]:
    d = monthly.copy().sort_values("ym").reset_index(drop=True)
    d["ret"] = pd.to_numeric(d["ret"], errors="coerce").fillna(0.0)
    d["eqw_ret"] = pd.to_numeric(d["eqw_ret"], errors="coerce").fillna(0.0)
    d["motor_ret"] = pd.to_numeric(d["motor_ret"], errors="coerce").fillna(0.0)
    used = pd.to_numeric(signal, errors="coerce").fillna(0.0).astype(int).shift(int(lag)).fillna(0).astype(int)
    true_g = np.where(used.to_numpy(dtype=int) == 1, d["motor_ret"].to_numpy(dtype=float), d["ret"].to_numpy(dtype=float))
    alpha_true = _total(true_g) - _total(d["eqw_ret"])

    rng = np.random.default_rng(int(seed))
    vals: list[float] = []
    for _ in range(int(n_iter)):
        sh = used.sample(frac=1.0, replace=False, random_state=int(rng.integers(0, 3_000_000))).to_numpy(dtype=int)
        g = np.where(sh == 1, d["motor_ret"].to_numpy(dtype=float), d["ret"].to_numpy(dtype=float))
        vals.append(_total(g) - _total(d["eqw_ret"]))
    arr = np.asarray(vals, dtype=float)
    return {
        "lag_months": int(lag),
        "iterations": int(n_iter),
        "signal_rate": _safe_float(used.mean()),
        "alpha_total_true_vs_eqw": float(alpha_true),
        "alpha_total_null_mean": float(np.mean(arr)),
        "alpha_total_null_p90": float(np.quantile(arr, 0.90)),
        "alpha_total_null_p95": float(np.quantile(arr, 0.95)),
        "prob_true_beats_null": float(np.mean(arr < alpha_true)),
    }


def _test3_null_by_year(
    monthly: pd.DataFrame,
    *,
    signal: pd.Series,
    lag: int,
    n_iter: int,
    seed: int,
) -> list[dict[str, Any]]:
    d = monthly.copy().sort_values("ym").reset_index(drop=True)
    d["ret"] = pd.to_numeric(d["ret"], errors="coerce").fillna(0.0)
    d["eqw_ret"] = pd.to_numeric(d["eqw_ret"], errors="coerce").fillna(0.0)
    d["motor_ret"] = pd.to_numeric(d["motor_ret"], errors="coerce").fillna(0.0)
    d["year"] = pd.to_numeric(d["year"], errors="coerce").astype("Int64")
    used_global = pd.to_numeric(signal, errors="coerce").fillna(0.0).astype(int).shift(int(lag)).fillna(0).astype(int)
    rows: list[dict[str, Any]] = []
    for y, g in d.groupby("year", dropna=True):
        g = g.reset_index(drop=True)
        if len(g) < 6:
            rows.append({"year": int(y), "status": "insufficient_months", "months": int(len(g))})
            continue
        idx = d[d["year"] == y].index.to_numpy(dtype=int)
        used = used_global.iloc[idx].to_numpy(dtype=int)
        alpha_true = _total(np.where(used == 1, g["motor_ret"].to_numpy(dtype=float), g["ret"].to_numpy(dtype=float))) - _total(
            g["eqw_ret"]
        )
        rng = np.random.default_rng(int(seed) + int(y))
        vals = np.empty(int(n_iter), dtype=float)
        for i in range(int(n_iter)):
            sh = np.array(used, copy=True)
            rng.shuffle(sh)
            vals[i] = _total(np.where(sh == 1, g["motor_ret"].to_numpy(dtype=float), g["ret"].to_numpy(dtype=float))) - _total(
                g["eqw_ret"]
            )
        rows.append({"year": int(y), "status": "ok", "months": int(len(g))})
        rows[-1].update(
            {
                "lag_months": int(lag),
                "iterations": int(n_iter),
                "signal_rate": float(np.mean(used)),
                "alpha_total_true_vs_eqw": float(alpha_true),
                "alpha_total_null_mean": float(np.mean(vals)),
                "alpha_total_null_p90": float(np.quantile(vals, 0.90)),
                "alpha_total_null_p95": float(np.quantile(vals, 0.95)),
                "prob_true_beats_null": float(np.mean(vals < alpha_true)),
            }
        )
    return rows


def _calibrate_signal_train(
    monthly: pd.DataFrame,
    *,
    train_end_ym: str,
    rule_options: list[str],
    lag_options: list[int],
    streak_options: list[int],
    n_iter: int,
    seed: int,
    min_rate: float,
    max_rate: float,
) -> dict[str, Any]:
    d = monthly.copy().sort_values("ym").reset_index(drop=True)
    d["date"] = pd.to_datetime(d["ym"].astype(str) + "-01", errors="coerce")
    train_end_date = pd.to_datetime(str(train_end_ym) + "-01", errors="coerce")
    z = d[d["date"] <= train_end_date].reset_index(drop=True)
    if z.empty:
        raise ValueError(f"empty train slice for train_end_ym={train_end_ym}")

    rows: list[dict[str, Any]] = []
    for rule in rule_options:
        for lag in lag_options:
            for streak in streak_options:
                sig = _build_signal(z, rule=rule, min_streak=int(streak))
                null_payload = _test3_null(
                    z,
                    signal=sig,
                    lag=int(lag),
                    n_iter=int(n_iter),
                    seed=int(seed),
                )
                rate = _safe_float(null_payload.get("signal_rate"))
                margin = _safe_float(null_payload.get("alpha_total_true_vs_eqw")) - _safe_float(
                    null_payload.get("alpha_total_null_p90")
                )
                rate_ok = bool(np.isfinite(rate) and (rate >= float(min_rate)) and (rate <= float(max_rate)))
                rows.append(
                    {
                        "rule": str(rule),
                        "lag_months": int(lag),
                        "min_streak": int(streak),
                        "signal_rate": float(rate) if np.isfinite(rate) else float("nan"),
                        "rate_ok": bool(rate_ok),
                        "alpha_total_true_vs_eqw": _safe_float(null_payload.get("alpha_total_true_vs_eqw")),
                        "alpha_total_null_p90": _safe_float(null_payload.get("alpha_total_null_p90")),
                        "margin_true_minus_null_p90": float(margin) if np.isfinite(margin) else float("nan"),
                        "prob_true_beats_null": _safe_float(null_payload.get("prob_true_beats_null")),
                    }
                )
    cands = pd.DataFrame(rows)
    if cands.empty:
        raise RuntimeError("no calibration candidates")
    cands["score"] = np.where(cands["rate_ok"], cands["margin_true_minus_null_p90"], -1e9)
    cands = cands.sort_values(
        ["score", "prob_true_beats_null", "alpha_total_true_vs_eqw"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    best = cands.iloc[0].to_dict()
    return {
        "train_end_ym": str(train_end_ym),
        "grid_size": int(len(cands)),
        "best": best,
        "top10": cands.head(10).to_dict(orient="records"),
    }


def _compute_turnover(monthly: pd.DataFrame, mret: pd.DataFrame, snaps: dict[str, pd.DataFrame]) -> pd.Series:
    d = monthly.copy().sort_values("ym").reset_index(drop=True)
    months = d["ym"].astype(str).tolist()
    prev_months = [months[i - 1] if i > 0 else None for i in range(len(months))]
    pre_w: dict[str, float] = {}
    cash_pre = 1.0
    out: list[float] = []
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
        out.append(float(turnover))

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
    return pd.Series(out, index=d.index, dtype=float)


def _test4_cost_ladder(monthly: pd.DataFrame, mret: pd.DataFrame, snaps: dict[str, pd.DataFrame], cost_bps_list: list[int]) -> pd.DataFrame:
    d = monthly.copy().sort_values("ym").reset_index(drop=True)
    d["ret"] = pd.to_numeric(d["ret"], errors="coerce").fillna(0.0)
    d["turnover"] = _compute_turnover(d, mret, snaps)
    rows: list[dict[str, Any]] = []
    for bps in cost_bps_list:
        rate = float(bps) / 10000.0
        rc = d["ret"] - d["turnover"] * rate
        rows.append(
            {
                "cost_bps": int(bps),
                "avg_turnover": _safe_float(d["turnover"].mean()),
                "total_return": _total(rc),
                "annualized_return": _ann(rc),
                "max_drawdown": _mdd(rc),
            }
        )
    return pd.DataFrame(rows).sort_values("cost_bps").reset_index(drop=True)


def _test6_synthetic_shocks(monthly: pd.DataFrame, *, seed: int = 23) -> dict[str, Any]:
    d = monthly.copy().sort_values("ym").reset_index(drop=True)
    d["ret"] = pd.to_numeric(d["ret"], errors="coerce").fillna(0.0)
    d["eqw_ret"] = pd.to_numeric(d["eqw_ret"], errors="coerce").fillna(0.0)
    d["risk_budget"] = pd.to_numeric(d["risk_budget"], errors="coerce").fillna(1.0).clip(lower=0.0, upper=1.5)

    n = len(d)
    if n == 0:
        return {"status": "empty"}

    rng = np.random.default_rng(int(seed))
    # Scenario A: random independent shocks
    mask_a = rng.random(n) < 0.12
    amp_a = -0.12
    s_a = d["ret"] + (amp_a * d["risk_budget"] * mask_a.astype(float))
    b_a = d["eqw_ret"] + (amp_a * 1.0 * mask_a.astype(float))

    # Scenario B: clustered 3-month cascades
    mask_b = np.zeros(n, dtype=bool)
    starts = [max(0, int(n * 0.20)), max(0, int(n * 0.48)), max(0, int(n * 0.74))]
    for st in starts:
        mask_b[st : min(n, st + 3)] = True
    amp_b = -0.15
    s_b = d["ret"] + (amp_b * d["risk_budget"] * mask_b.astype(float))
    b_b = d["eqw_ret"] + (amp_b * 1.0 * mask_b.astype(float))

    def pack(s: pd.Series, b: pd.Series, name: str, n_shocks: int, amp: float) -> dict[str, Any]:
        return {
            "scenario": name,
            "n_shock_months": int(n_shocks),
            "shock_amplitude": float(amp),
            "strategy_total_return": _total(s),
            "eqw_total_return": _total(b),
            "alpha_total_vs_eqw": _total(s) - _total(b),
            "strategy_max_drawdown": _mdd(s),
            "eqw_max_drawdown": _mdd(b),
            "alpha_mean_monthly": _safe_float((pd.Series(s) - pd.Series(b)).mean()),
        }

    return {
        "status": "ok",
        "random_shocks": pack(s_a, b_a, "random_independent_shocks", int(mask_a.sum()), amp_a),
        "clustered_shocks": pack(s_b, b_b, "clustered_3month_cascades", int(mask_b.sum()), amp_b),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Validation battery 1..6 (walk-forward, crisis, null, costs, subuniverses, synthetic shocks).")
    ap.add_argument("--base-topk20-run", default="results/portfolio_sim/20260302T171626Z_gain_tests_1234/subuniverses/topk_20")
    ap.add_argument("--subuniverse-summary-csv", default="", help="Optional existing subuniverses summary csv.")
    ap.add_argument("--train-end-years", default="2019,2020,2021,2022,2023,2024,2025")
    ap.add_argument("--null-iterations", type=int, default=800)
    ap.add_argument("--null-seed", type=int, default=23)
    ap.add_argument("--null-by-year-iterations", type=int, default=1200)
    ap.add_argument("--null-by-year-seed", type=int, default=131)
    ap.add_argument("--cost-bps-list", default="10,20,30,50")
    ap.add_argument("--signal-rule", default="defense_or_decel")
    ap.add_argument("--signal-min-streak", type=int, default=1)
    ap.add_argument("--signal-lag-months", type=int, default=1)
    ap.add_argument("--calibrate-signal-on-train", type=int, default=0)
    ap.add_argument("--signal-train-end-ym", default="2023-12")
    ap.add_argument("--signal-rule-options", default="defense_or_decel,defense_and_decel,decel_only,defense_only")
    ap.add_argument("--signal-lag-options", default="1,2,3")
    ap.add_argument("--signal-streak-options", default="1,2")
    ap.add_argument("--signal-calibration-iterations", type=int, default=800)
    ap.add_argument("--signal-calibration-seed", type=int, default=31)
    ap.add_argument("--signal-min-rate", type=float, default=0.05)
    ap.add_argument("--signal-max-rate", type=float, default=0.70)
    ap.add_argument("--outdir", default="", help="Output dir (default: results/portfolio_sim/<runid>_validation_1_6)")
    args = ap.parse_args()

    base_run = Path(args.base_topk20_run).resolve()
    monthly_csv = base_run / "monthly_systematic_eval.csv"
    sim_json = base_run / "simulation_summary.json"
    if not monthly_csv.exists() or not sim_json.exists():
        raise FileNotFoundError(f"missing monthly/simulation at {base_run}")
    sim = json.loads(sim_json.read_text(encoding="utf-8"))
    impact_dir = Path(sim["impact_dir"])
    returns_csv = Path(sim["returns_csv"])
    impact_csv = impact_dir / "impact_training_dataset.csv"
    if not impact_csv.exists():
        raise FileNotFoundError(f"missing {impact_csv}")

    run_id = _run_id()
    outdir = Path(args.outdir).resolve() if args.outdir.strip() else (ROOT / "results" / "portfolio_sim" / f"{run_id}_validation_1_6")
    outdir.mkdir(parents=True, exist_ok=True)

    monthly = pd.read_csv(monthly_csv).sort_values("ym").reset_index(drop=True)
    mret = _build_monthly_returns(returns_csv)
    snaps = _build_snapshots(impact_csv, max_assets_per_month=120)

    signal_cfg: dict[str, Any] = {
        "rule": str(args.signal_rule),
        "min_streak": int(args.signal_min_streak),
        "lag_months": int(args.signal_lag_months),
    }
    signal_calib: dict[str, Any] | None = None
    if int(args.calibrate_signal_on_train) == 1:
        signal_calib = _calibrate_signal_train(
            monthly,
            train_end_ym=str(args.signal_train_end_ym),
            rule_options=_parse_str_list(args.signal_rule_options),
            lag_options=_parse_int_list(args.signal_lag_options),
            streak_options=_parse_int_list(args.signal_streak_options),
            n_iter=int(args.signal_calibration_iterations),
            seed=int(args.signal_calibration_seed),
            min_rate=float(args.signal_min_rate),
            max_rate=float(args.signal_max_rate),
        )
        best = signal_calib.get("best") or {}
        signal_cfg = {
            "rule": str(best.get("rule", signal_cfg["rule"])),
            "min_streak": int(_safe_float(best.get("min_streak", signal_cfg["min_streak"]))),
            "lag_months": int(_safe_float(best.get("lag_months", signal_cfg["lag_months"]))),
        }

    signal = _build_signal(
        monthly,
        rule=str(signal_cfg["rule"]),
        min_streak=int(signal_cfg["min_streak"]),
    )

    # 1) Walk-forward harder
    wf_df, wf_summary = _test1_walkforward(
        impact_dir=impact_dir,
        returns_csv=returns_csv,
        outdir=outdir,
        train_end_years=_parse_int_list(args.train_end_years),
        top_k=20,
    )
    wf_csv = outdir / "test1_walkforward_blocks.csv"
    wf_df.to_csv(wf_csv, index=False)

    # 2) Crisis windows
    crisis = _test2_crisis(monthly)

    # 3) Null test
    null = _test3_null(
        monthly,
        signal=signal,
        lag=int(signal_cfg["lag_months"]),
        n_iter=int(args.null_iterations),
        seed=int(args.null_seed),
    )
    null_by_year = _test3_null_by_year(
        monthly,
        signal=signal,
        lag=int(signal_cfg["lag_months"]),
        n_iter=int(args.null_by_year_iterations),
        seed=int(args.null_by_year_seed),
    )

    # 4) Cost ladder 1x/2x/3x/5x
    cost_df = _test4_cost_ladder(monthly, mret, snaps, _parse_int_list(args.cost_bps_list))
    cost_csv = outdir / "test4_cost_ladder.csv"
    cost_df.to_csv(cost_csv, index=False)

    # 5) Subuniverses stability (must come from the same run lineage)
    sub_csv: Path | None = None
    if args.subuniverse_summary_csv.strip():
        p = Path(args.subuniverse_summary_csv).resolve()
        if p.exists():
            sub_csv = p
    if sub_csv is None:
        sub_csv = _infer_subuniverse_summary_from_base_run(base_run)
    if sub_csv is None:
        sub_csv = _autobuild_subuniverse_summary(base_run, outdir)
    if sub_csv is None:
        sub_df = pd.DataFrame()
        sub_summary = {
            "status": "missing",
            "reason": "subuniverses_summary_not_found_for_same_run",
            "base_run": str(base_run),
        }
    else:
        sub_df = pd.read_csv(sub_csv)
        sub_summary = {
            "status": "ok",
            "source_csv": str(sub_csv),
            "rows": sub_df.to_dict(orient="records"),
            "positive_total_rate": _safe_float((pd.to_numeric(sub_df["total_return"], errors="coerce") > 0.0).mean()),
            "worth_rate_ge_060": _safe_float((pd.to_numeric(sub_df["worth_it_rate_vs_eqw"], errors="coerce") >= 0.60).mean()),
        }

    # 6) Synthetic shocks
    synthetic = _test6_synthetic_shocks(monthly, seed=23)

    flags = {
        "test1_walkforward_beat_rate_ge_065": bool(_safe_float(wf_summary.get("beat_rate_vs_eqw")) >= 0.65),
        "test2_crisis_alpha_mean_non_negative": bool(
            _safe_float((crisis.get("worst_20pct_eqw_months") or {}).get("alpha_mean")) >= 0.0
        ),
        "test3_null_true_above_null_p90": bool(
            _safe_float(null.get("alpha_total_true_vs_eqw")) > _safe_float(null.get("alpha_total_null_p90"))
        ),
        "test4_cost_5x_total_positive": bool(
            (
                not cost_df[cost_df["cost_bps"] == 50].empty
                and _safe_float(cost_df[cost_df["cost_bps"] == 50]["total_return"].iloc[0]) > 0.0
            )
        ),
        "test5_subuniverse_positive_rate_ge_075": bool(_safe_float(sub_summary.get("positive_total_rate")) >= 0.75)
        if sub_summary.get("status") == "ok"
        else False,
        "test6_synthetic_cluster_alpha_non_negative": bool(
            _safe_float((synthetic.get("clustered_shocks") or {}).get("alpha_total_vs_eqw")) >= 0.0
        ),
    }
    ready = all(bool(v) for v in flags.values())

    payload = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_topk20_run": str(base_run),
        "tests": {
            "1_walkforward_harder": {"summary": wf_summary, "csv": str(wf_csv)},
            "2_crisis_windows": crisis,
            "3_null_test": null,
            "3b_null_test_by_year": null_by_year,
            "4_cost_ladder": {"csv": str(cost_csv), "rows": cost_df.to_dict(orient="records")},
            "5_subuniverse_stability": sub_summary,
            "6_synthetic_shocks": synthetic,
        },
        "signal_config_selected": signal_cfg,
        "signal_calibration": signal_calib,
        "flags_1_to_6": flags,
        "ready_1_to_6": bool(ready),
    }
    out_json = outdir / "validation_battery_1_6_summary.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "ok", "outdir": str(outdir), "summary_json": str(out_json), "ready_1_to_6": bool(ready)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
