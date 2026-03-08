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
    tot = float(np.prod(1.0 + s.to_numpy(dtype=float)) - 1.0)
    return float((1.0 + tot) ** (12.0 / float(len(s))) - 1.0)


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
        required = ["monthly_systematic_eval.csv", "simulation_summary.json", "systematic_summary.json"]
        if all((run / f).exists() for f in required):
            return run
    raise FileNotFoundError("no systematic run with required artifacts")


def _run_canonical(
    *,
    impact_dir: Path,
    returns_csv: Path,
    outdir: Path,
    defense_enabled: int,
    defense_multiplier: float,
    defense_corr_q: float,
    defense_vol_q: float,
    decel_enabled: int,
    decel_alpha_threshold: float,
    top_k_options: str,
    mode: str,
) -> dict[str, Any]:
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
        "2023-12-31",
        "--start-ym",
        "2019-01",
        "--top-k-options",
        top_k_options,
        "--impact-power-options",
        "0",
        "--wmax-options",
        "0.1",
        "--mom-lookback-options",
        "0",
        "--mom-threshold-options",
        "-0.02",
        "--modes",
        mode,
        "--defense-enabled",
        str(int(defense_enabled)),
        "--defense-multiplier",
        str(float(defense_multiplier)),
        "--defense-corr-quantile",
        str(float(defense_corr_q)),
        "--defense-vol-quantile",
        str(float(defense_vol_q)),
        "--decel-enabled",
        str(int(decel_enabled)),
        "--decel-lookback-months",
        "6",
        "--decel-alpha-threshold",
        str(float(decel_alpha_threshold)),
        "--decel-min-streak",
        "2",
        "--decel-multiplier",
        "0.95",
        "--decel-topk-multiplier",
        "0.85",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    sys_json = outdir / "systematic_summary.json"
    sim_json = outdir / "simulation_summary.json"
    if not sys_json.exists() or not sim_json.exists():
        raise FileNotFoundError(f"missing outputs in {outdir}")
    summary = json.loads(sys_json.read_text(encoding="utf-8"))
    sim = json.loads(sim_json.read_text(encoding="utf-8"))
    metrics = sim.get("best_metrics", {}) if isinstance(sim, dict) else {}
    return {
        "outdir": str(outdir),
        "worth_it_rate_vs_eqw": _safe_float(summary.get("worth_it_rate_vs_eqw")),
        "monthly_alpha_prob_positive_vs_eqw": _safe_float(summary.get("monthly_alpha_prob_positive_vs_eqw")),
        "strategy_max_drop": _safe_float(summary.get("strategy_max_drop")),
        "full_alpha_recent6": _safe_float(metrics.get("full_alpha_recent6")),
    }


def _test4_crisis(monthly: pd.DataFrame) -> dict[str, Any]:
    d = monthly.copy()
    d["alpha"] = pd.to_numeric(d["ret"], errors="coerce") - pd.to_numeric(d["eqw_ret"], errors="coerce")
    d = d.dropna(subset=["ret", "eqw_ret", "alpha"]).copy()
    if d.empty:
        return {"status": "empty"}

    q20 = float(np.quantile(pd.to_numeric(d["eqw_ret"], errors="coerce").to_numpy(dtype=float), 0.20))
    c1 = d[d["eqw_ret"] <= q20].copy()
    c2 = d[d["eqw_ret"] <= -0.05].copy()

    def pack(z: pd.DataFrame, tag: str) -> dict[str, Any]:
        if z.empty:
            return {"name": tag, "months": 0}
        eqw_losses = z[z["eqw_ret"] < 0].copy()
        loss_ratio = float(np.mean(np.abs(eqw_losses["ret"]) / np.abs(eqw_losses["eqw_ret"]))) if not eqw_losses.empty else float("nan")
        return {
            "name": tag,
            "months": int(len(z)),
            "strategy_total": _total(z["ret"]),
            "eqw_total": _total(z["eqw_ret"]),
            "alpha_total": _total(z["ret"]) - _total(z["eqw_ret"]),
            "alpha_mean": _safe_float(z["alpha"].mean()),
            "alpha_positive_rate": _safe_float((z["alpha"] > 0).mean()),
            "loss_ratio_abs_strategy_over_eqw": loss_ratio,
        }

    return {
        "status": "ok",
        "crisis_by_worst_20pct_eqw_months": pack(c1, "worst_20pct_eqw_months"),
        "crisis_by_eqw_le_minus_5pct": pack(c2, "eqw_le_minus_5pct"),
    }


def _test5_ablation(base_run: Path, impact_dir: Path, returns_csv: Path, outdir: Path) -> dict[str, Any]:
    ab_root = outdir / "ablation_runs"
    ab_root.mkdir(parents=True, exist_ok=True)
    variants = [
        ("full", 1, 0.85, 1, 0.0),
        ("no_defense", 0, 1.0, 1, 0.0),
        ("no_decel", 1, 0.85, 0, 0.0),
        ("no_defense_no_decel", 0, 1.0, 0, 0.0),
    ]
    rows: list[dict[str, Any]] = []
    for name, defense_en, defense_mult, decel_en, decel_alpha in variants:
        run_out = ab_root / name
        m = _run_canonical(
            impact_dir=impact_dir,
            returns_csv=returns_csv,
            outdir=run_out,
            defense_enabled=defense_en,
            defense_multiplier=defense_mult,
            defense_corr_q=0.8,
            defense_vol_q=0.8,
            decel_enabled=decel_en,
            decel_alpha_threshold=decel_alpha,
            top_k_options="52",
            mode="const",
        )
        m["variant"] = name
        rows.append(m)

    df = pd.DataFrame(rows)
    base = df[df["variant"] == "full"].iloc[0].to_dict()
    for c in ["worth_it_rate_vs_eqw", "monthly_alpha_prob_positive_vs_eqw", "strategy_max_drop", "full_alpha_recent6"]:
        b = _safe_float(base.get(c))
        if np.isfinite(b):
            df[f"delta_vs_full_{c}"] = pd.to_numeric(df[c], errors="coerce") - b
        else:
            df[f"delta_vs_full_{c}"] = float("nan")
    csv_path = outdir / "ablation_summary.csv"
    df.to_csv(csv_path, index=False)
    return {
        "status": "ok",
        "csv": str(csv_path),
        "rows": rows,
    }


def _test6_sensitivity(impact_dir: Path, returns_csv: Path, outdir: Path) -> dict[str, Any]:
    sens_root = outdir / "sensitivity_runs"
    sens_root.mkdir(parents=True, exist_ok=True)
    qs = [0.75, 0.80, 0.85, 0.90, 0.95]
    rows: list[dict[str, Any]] = []
    for q in qs:
        run_out = sens_root / f"q_{str(q).replace('.', '')}"
        m = _run_canonical(
            impact_dir=impact_dir,
            returns_csv=returns_csv,
            outdir=run_out,
            defense_enabled=1,
            defense_multiplier=0.85,
            defense_corr_q=q,
            defense_vol_q=q,
            decel_enabled=1,
            decel_alpha_threshold=0.0,
            top_k_options="52",
            mode="const",
        )
        m["q"] = q
        rows.append(m)

    df = pd.DataFrame(rows).sort_values("q").reset_index(drop=True)
    csv_path = outdir / "sensitivity_q_summary.csv"
    df.to_csv(csv_path, index=False)

    pass_flags = (
        (pd.to_numeric(df["worth_it_rate_vs_eqw"], errors="coerce") >= 0.55)
        & (pd.to_numeric(df["monthly_alpha_prob_positive_vs_eqw"], errors="coerce") >= 0.55)
        & (pd.to_numeric(df["strategy_max_drop"], errors="coerce") >= -0.35)
        & (pd.to_numeric(df["full_alpha_recent6"], errors="coerce") >= -0.003)
    )
    return {
        "status": "ok",
        "csv": str(csv_path),
        "q_values": qs,
        "pass_count": int(pass_flags.sum()),
        "total_count": int(len(pass_flags)),
        "std_worth_it_rate_vs_eqw": _safe_float(pd.to_numeric(df["worth_it_rate_vs_eqw"], errors="coerce").std(ddof=0)),
        "std_monthly_alpha_prob_positive_vs_eqw": _safe_float(
            pd.to_numeric(df["monthly_alpha_prob_positive_vs_eqw"], errors="coerce").std(ddof=0)
        ),
        "std_strategy_max_drop": _safe_float(pd.to_numeric(df["strategy_max_drop"], errors="coerce").std(ddof=0)),
        "rows": rows,
    }


def _test7_bootstrap(monthly: pd.DataFrame, *, n_boot: int, block_len: int, seed: int) -> dict[str, Any]:
    d = monthly.copy()
    d["ret"] = pd.to_numeric(d["ret"], errors="coerce")
    d["eqw_ret"] = pd.to_numeric(d["eqw_ret"], errors="coerce")
    d = d.dropna(subset=["ret", "eqw_ret"]).reset_index(drop=True)
    if d.empty:
        return {"status": "empty"}

    strat = d["ret"].to_numpy(dtype=float)
    eqw = d["eqw_ret"].to_numpy(dtype=float)
    n = int(len(d))
    lb = int(max(2, block_len))
    rng = np.random.default_rng(int(seed))

    alpha_totals: list[float] = []
    alpha_means: list[float] = []
    alpha_prob_pos: list[float] = []
    for _ in range(int(n_boot)):
        idxs: list[int] = []
        while len(idxs) < n:
            start = int(rng.integers(0, max(1, n - lb + 1)))
            idxs.extend(range(start, min(n, start + lb)))
        idx = np.asarray(idxs[:n], dtype=int)
        s = strat[idx]
        b = eqw[idx]
        alpha = s - b
        alpha_totals.append(float(np.prod(1.0 + s) - np.prod(1.0 + b)))
        alpha_means.append(float(np.mean(alpha)))
        alpha_prob_pos.append(float(np.mean(alpha > 0.0)))

    def ci95(vals: list[float]) -> list[float]:
        arr = np.asarray(vals, dtype=float)
        return [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))]

    return {
        "status": "ok",
        "n_boot": int(n_boot),
        "block_len_months": int(lb),
        "alpha_total_ci95": ci95(alpha_totals),
        "alpha_mean_monthly_ci95": ci95(alpha_means),
        "alpha_prob_positive_ci95": ci95(alpha_prob_pos),
        "prob_alpha_total_positive": float(np.mean(np.asarray(alpha_totals, dtype=float) > 0.0)),
    }


def _test8_shadow(monthly: pd.DataFrame, *, last_n_months: int) -> dict[str, Any]:
    d = monthly.copy()
    d["alpha"] = pd.to_numeric(d["ret"], errors="coerce") - pd.to_numeric(d["eqw_ret"], errors="coerce")
    d = d.sort_values("ym").reset_index(drop=True)
    tail = d.tail(int(last_n_months)).copy()
    if tail.empty:
        return {"status": "empty"}

    return {
        "status": "ok",
        "window_months": int(len(tail)),
        "start_ym": str(tail["ym"].iloc[0]),
        "end_ym": str(tail["ym"].iloc[-1]),
        "strategy_total": _total(tail["ret"]),
        "eqw_total": _total(tail["eqw_ret"]),
        "alpha_total": _total(tail["ret"]) - _total(tail["eqw_ret"]),
        "alpha_positive_rate": _safe_float((tail["alpha"] > 0.0).mean()),
        "avg_risk_budget": _safe_float(pd.to_numeric(tail["risk_budget"], errors="coerce").mean()),
        "avg_selected_assets": _safe_float(pd.to_numeric(tail["n_selected"], errors="coerce").mean()),
        "shadow_ready": bool(
            len(tail) >= max(3, int(last_n_months) // 2)
            and np.isfinite(_safe_float((tail["alpha"] > 0.0).mean()))
            and _safe_float(pd.to_numeric(tail["n_selected"], errors="coerce").mean()) > 0
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run robustness tests 4-8 on systematic strategy.")
    ap.add_argument("--base-run-dir", default="", help="Base systematic run dir; default latest.")
    ap.add_argument("--outdir", default="", help="Output dir (default: results/portfolio_sim/<runid>_tests_45678).")
    ap.add_argument("--bootstrap-iterations", type=int, default=2000)
    ap.add_argument("--bootstrap-block-len", type=int, default=6)
    ap.add_argument("--bootstrap-seed", type=int, default=23)
    ap.add_argument("--shadow-last-months", type=int, default=6)
    args = ap.parse_args()

    base_run = Path(args.base_run_dir).resolve() if args.base_run_dir.strip() else _latest_systematic_run()
    monthly_csv = base_run / "monthly_systematic_eval.csv"
    sim_json = base_run / "simulation_summary.json"
    if not monthly_csv.exists() or not sim_json.exists():
        raise FileNotFoundError(f"missing monthly/simulation in {base_run}")

    sim = json.loads(sim_json.read_text(encoding="utf-8"))
    impact_dir = Path(sim["impact_dir"])
    returns_csv = Path(sim["returns_csv"])
    monthly = pd.read_csv(monthly_csv)

    run_id = _run_id()
    outdir = Path(args.outdir).resolve() if args.outdir.strip() else (ROOT / "results" / "portfolio_sim" / f"{run_id}_tests_45678")
    outdir.mkdir(parents=True, exist_ok=True)

    t4 = _test4_crisis(monthly)
    t5 = _test5_ablation(base_run, impact_dir, returns_csv, outdir)
    t6 = _test6_sensitivity(impact_dir, returns_csv, outdir)
    t7 = _test7_bootstrap(
        monthly,
        n_boot=int(args.bootstrap_iterations),
        block_len=int(args.bootstrap_block_len),
        seed=int(args.bootstrap_seed),
    )
    t8 = _test8_shadow(monthly, last_n_months=int(args.shadow_last_months))

    launch_gate = {
        "t4_crisis_alpha_non_negative": bool(
            _safe_float((t4.get("crisis_by_worst_20pct_eqw_months") or {}).get("alpha_mean")) >= 0.0
        ),
        "t5_ablation_full_not_dominated": True,
        "t6_sensitivity_majority_pass": bool(int(t6.get("pass_count", 0)) >= max(3, int(t6.get("total_count", 0)) // 2 + 1)),
        "t7_bootstrap_alpha_total_positive_prob_ge_070": bool(_safe_float(t7.get("prob_alpha_total_positive")) >= 0.70),
        "t8_shadow_ready": bool(t8.get("shadow_ready") is True),
    }

    # Check if any ablation variant strictly dominates full on all key metrics.
    try:
        ab_df = pd.read_csv(Path(t5["csv"]))
        full = ab_df[ab_df["variant"] == "full"].iloc[0]
        dominated = False
        for _, row in ab_df.iterrows():
            if row["variant"] == "full":
                continue
            better_or_equal = (
                _safe_float(row["worth_it_rate_vs_eqw"]) >= _safe_float(full["worth_it_rate_vs_eqw"])
                and _safe_float(row["monthly_alpha_prob_positive_vs_eqw"]) >= _safe_float(full["monthly_alpha_prob_positive_vs_eqw"])
                and _safe_float(row["strategy_max_drop"]) >= _safe_float(full["strategy_max_drop"])
                and _safe_float(row["full_alpha_recent6"]) >= _safe_float(full["full_alpha_recent6"])
            )
            strictly_better = (
                _safe_float(row["worth_it_rate_vs_eqw"]) > _safe_float(full["worth_it_rate_vs_eqw"])
                or _safe_float(row["monthly_alpha_prob_positive_vs_eqw"]) > _safe_float(full["monthly_alpha_prob_positive_vs_eqw"])
                or _safe_float(row["strategy_max_drop"]) > _safe_float(full["strategy_max_drop"])
                or _safe_float(row["full_alpha_recent6"]) > _safe_float(full["full_alpha_recent6"])
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        launch_gate["t5_ablation_full_not_dominated"] = not dominated
    except Exception:
        launch_gate["t5_ablation_full_not_dominated"] = False

    ready = all(bool(v) for v in launch_gate.values())
    payload = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_dir": str(base_run),
        "tests": {
            "4_crisis_windows": t4,
            "5_ablation": t5,
            "6_threshold_sensitivity": t6,
            "7_bootstrap_temporal": t7,
            "8_shadow_paper_mode": t8,
        },
        "launch_gate_4_to_8": launch_gate,
        "launch_ready_round_4_to_8": bool(ready),
    }
    out_json = outdir / "tests_45678_summary.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "ok", "outdir": str(outdir), "summary_json": str(out_json), "launch_ready_round_4_to_8": bool(ready)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
