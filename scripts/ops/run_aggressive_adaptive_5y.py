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


def _latest_nonempty_impact() -> tuple[Path, Path]:
    base = ROOT / "results" / "lab_corr_macro"
    if not base.exists():
        raise FileNotFoundError(f"missing dir: {base}")
    runs = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
    for run in runs:
        returns = run / "returns_wide_core.csv"
        if not returns.exists():
            continue
        hier = run / "hierarchical"
        if not hier.exists():
            continue
        for cand in sorted([p for p in hier.glob("impact_learning*") if p.is_dir()], key=lambda p: p.name, reverse=True):
            f = cand / "impact_training_dataset.csv"
            if not f.exists():
                continue
            try:
                n = int(pd.read_csv(f, usecols=["date"]).shape[0])
            except Exception:
                n = 0
            if n > 0:
                return cand, returns
    raise FileNotFoundError("no non-empty impact dataset found")


def _run(cmd: list[str], *, cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


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
    t = _total(s)
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


def _cagr(ret: pd.Series | np.ndarray, n_months: int) -> float:
    t = _total(ret)
    if not np.isfinite(t) or int(n_months) <= 0:
        return float("nan")
    years = float(n_months) / 12.0
    if years <= 0:
        return float("nan")
    return float((1.0 + t) ** (1.0 / years) - 1.0)


def _monthly_win_rate(ret: pd.Series | np.ndarray) -> float:
    s = pd.Series(ret)
    s = pd.to_numeric(s, errors="coerce").dropna().astype(float)
    if s.empty:
        return float("nan")
    return float((s > 0.0).mean())


def _max_assets_for_bps(top_k: int) -> int:
    return int(max(80, min(220, int(top_k) * 12)))


def _candidate_id(top_k: int, w_max: float, mode: str, mom_lb: int, impact_power: float) -> str:
    return f"k{int(top_k)}_w{float(w_max):.2f}_m{str(mode)}_lb{int(mom_lb)}_ip{float(impact_power):.1f}"


def _run_candidate(
    *,
    impact_dir: Path,
    returns_csv: Path,
    train_end_date: str,
    start_ym: str,
    top_k: int,
    w_max: float,
    mode: str,
    mom_lb: int,
    impact_power: float,
    outdir: Path,
) -> Path:
    monthly_csv = outdir / "monthly_systematic_eval.csv"
    if monthly_csv.exists():
        return monthly_csv
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
        str(train_end_date),
        "--start-ym",
        str(start_ym),
        "--max-assets-per-month",
        str(_max_assets_for_bps(int(top_k))),
        "--top-k-options",
        str(int(top_k)),
        "--impact-power-options",
        str(float(impact_power)),
        "--wmax-options",
        str(float(w_max)),
        "--mom-lookback-options",
        str(int(mom_lb)),
        "--mom-threshold-options",
        "-0.01",
        "--modes",
        str(mode),
        "--defense-enabled",
        "1",
        "--defense-multiplier",
        "0.95",
        "--defense-corr-quantile",
        "0.90",
        "--defense-vol-quantile",
        "0.90",
        "--decel-enabled",
        "1",
        "--decel-lookback-months",
        "6",
        "--decel-alpha-threshold",
        "0.0",
        "--decel-min-streak",
        "3",
        "--decel-multiplier",
        "0.98",
        "--decel-topk-multiplier",
        "0.95",
    ]
    _run(cmd, cwd=ROOT)
    if not monthly_csv.exists():
        raise FileNotFoundError(f"missing candidate monthly csv: {monthly_csv}")
    return monthly_csv


def _candidate_score(train_ret: pd.Series) -> float:
    total = _total(train_ret)
    dd = _mdd(train_ret)
    wr = _monthly_win_rate(train_ret)
    if not np.isfinite(total) or not np.isfinite(dd):
        return float("-inf")
    return float(total - 0.90 * abs(dd) + 0.08 * (wr if np.isfinite(wr) else 0.0))


@dataclass(frozen=True)
class LeverageProfile:
    name: str
    lev_base: float
    dd1: float
    dd2: float
    dd3: float
    lev1: float
    lev2: float
    lev3: float
    signal_brake: float


def _apply_leverage(base_ret: pd.Series, signal_on: pd.Series, p: LeverageProfile) -> tuple[pd.Series, pd.Series]:
    r = pd.to_numeric(base_ret, errors="coerce").fillna(0.0).astype(float)
    sig = pd.to_numeric(signal_on, errors="coerce").fillna(0.0).astype(int)
    lev = []
    out = []
    eq = 1.0
    peak = 1.0
    for i, rv in enumerate(r.to_numpy(dtype=float)):
        dd = (eq / peak) - 1.0
        if dd <= p.dd3:
            cur = p.lev3
        elif dd <= p.dd2:
            cur = p.lev2
        elif dd <= p.dd1:
            cur = p.lev1
        else:
            cur = p.lev_base
        if int(sig.iloc[i]) == 1 and float(p.signal_brake) < 1.0:
            cur = cur * float(p.signal_brake)
        rr = float(np.clip(cur * rv, -0.95, 3.00))
        eq = eq * (1.0 + rr)
        peak = max(peak, eq)
        lev.append(cur)
        out.append(rr)
    return pd.Series(out, index=r.index, dtype=float), pd.Series(lev, index=r.index, dtype=float)


def _yearly(ret: pd.Series, ym: pd.Series) -> pd.DataFrame:
    d = pd.DataFrame({"ym": ym.astype(str), "ret": pd.to_numeric(ret, errors="coerce").fillna(0.0)})
    d["year"] = d["ym"].str.slice(0, 4).astype(int)
    y = d.groupby("year", as_index=False).agg(total_return=("ret", _total), ann_return=("ret", _ann), mdd=("ret", _mdd))
    return y


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggressive adaptive 5-year scan with monthly re-evaluation and leverage control.")
    ap.add_argument("--impact-dir", default="")
    ap.add_argument("--returns-csv", default="")
    ap.add_argument("--test-start-ym", default="")
    ap.add_argument("--test-end-ym", default="")
    ap.add_argument("--train-lookback-months", type=int, default=36)
    ap.add_argument("--min-train-months", type=int, default=24)
    ap.add_argument("--reeval-every-months", default="1,3")
    ap.add_argument("--top-k-options", default="4,6,8,10,12,16")
    ap.add_argument("--wmax-options", default="0.25,0.40")
    ap.add_argument("--mode-options", default="const,score")
    ap.add_argument("--mom-lookback-options", default="0")
    ap.add_argument("--impact-power-options", default="0")
    ap.add_argument("--capitals", default="1000,5000,10000,25000,50000,100000")
    ap.add_argument("--outdir", default="")
    args = ap.parse_args()

    if str(args.impact_dir).strip():
        impact_dir = Path(args.impact_dir).resolve()
        returns_csv = Path(args.returns_csv).resolve() if str(args.returns_csv).strip() else impact_dir.parents[1] / "returns_wide_core.csv"
    else:
        impact_dir, returns_csv = _latest_nonempty_impact()
    if not impact_dir.exists():
        raise FileNotFoundError(f"missing impact_dir: {impact_dir}")
    if not returns_csv.exists():
        raise FileNotFoundError(f"missing returns_csv: {returns_csv}")

    run_id = _run_id()
    outdir = Path(args.outdir).resolve() if str(args.outdir).strip() else (ROOT / "results" / "portfolio_sim" / f"{run_id}_aggressive_adaptive_5y")
    outdir.mkdir(parents=True, exist_ok=True)
    cand_root = outdir / "candidate_runs"
    cand_root.mkdir(parents=True, exist_ok=True)

    reeval_steps = _parse_int_list(args.reeval_every_months)
    top_ks = _parse_int_list(args.top_k_options)
    wmaxs = _parse_float_list(args.wmax_options)
    modes = [x.strip().lower() for x in str(args.mode_options).split(",") if x.strip()]
    mom_lbs = _parse_int_list(args.mom_lookback_options)
    impact_powers = _parse_float_list(args.impact_power_options)
    capitals = _parse_float_list(args.capitals)

    # Use monthly span from data to define last 5 years if not provided.
    probe = pd.read_csv(returns_csv, usecols=["date"])
    probe["date"] = pd.to_datetime(probe["date"], errors="coerce")
    probe = probe.dropna(subset=["date"]).sort_values("date")
    max_ym = str(probe["date"].iloc[-1].to_period("M"))
    end_ym = str(args.test_end_ym).strip() if str(args.test_end_ym).strip() else max_ym
    end_ts = pd.to_datetime(f"{end_ym}-01", errors="coerce")
    start_ym = str(args.test_start_ym).strip() if str(args.test_start_ym).strip() else str((end_ts - pd.DateOffset(months=59)).to_period("M"))
    start_ts = pd.to_datetime(f"{start_ym}-01", errors="coerce")
    train_end_date = str((start_ts - pd.DateOffset(months=1)).to_period("M").end_time.date())

    candidates: list[dict[str, Any]] = []
    for top_k in top_ks:
        for w_max in wmaxs:
            for mode in modes:
                for mom_lb in mom_lbs:
                    for impact_power in impact_powers:
                        cid = _candidate_id(top_k, w_max, mode, mom_lb, impact_power)
                        cdir = cand_root / cid
                        monthly_csv = _run_candidate(
                            impact_dir=impact_dir,
                            returns_csv=returns_csv,
                            train_end_date=train_end_date,
                            start_ym=start_ym,
                            top_k=int(top_k),
                            w_max=float(w_max),
                            mode=str(mode),
                            mom_lb=int(mom_lb),
                            impact_power=float(impact_power),
                            outdir=cdir,
                        )
                        d = pd.read_csv(monthly_csv)
                        d["ym"] = d["ym"].astype(str)
                        for c in ["ret", "eqw_ret", "defense_active", "decel_active"]:
                            d[c] = pd.to_numeric(d[c], errors="coerce")
                        d = d[(d["ym"] >= start_ym) & (d["ym"] <= end_ym)].copy()
                        if d.empty:
                            continue
                        candidates.append({"id": cid, "params": {"top_k": int(top_k), "w_max": float(w_max), "mode": str(mode), "mom_lb": int(mom_lb), "impact_power": float(impact_power)}, "monthly": d})

    if not candidates:
        raise RuntimeError("no candidates with monthly data in test range")

    # Build aligned panel.
    months = sorted(set().union(*[set(c["monthly"]["ym"].tolist()) for c in candidates]))
    panel_rows: list[dict[str, Any]] = []
    for c in candidates:
        s = c["monthly"].set_index("ym")
        for ym in months:
            if ym not in s.index:
                continue
            panel_rows.append(
                {
                    "ym": ym,
                    "candidate_id": c["id"],
                    "ret": _safe_float(s.at[ym, "ret"]),
                    "eqw_ret": _safe_float(s.at[ym, "eqw_ret"]),
                    "signal_on": int(((_safe_float(s.at[ym, "defense_active"]) > 0.0) or (_safe_float(s.at[ym, "decel_active"]) > 0.0))),
                }
            )
    panel = pd.DataFrame(panel_rows).sort_values(["ym", "candidate_id"]).reset_index(drop=True)

    months_sorted = sorted(panel["ym"].unique().tolist())
    cand_ids = sorted(panel["candidate_id"].unique().tolist())

    # Adaptive selection + leverage profiles.
    profiles = [
        LeverageProfile("balanced_130", 1.30, -0.10, -0.20, -0.30, 1.10, 0.90, 0.60, 0.90),
        LeverageProfile("aggressive_160", 1.60, -0.10, -0.20, -0.30, 1.30, 1.00, 0.70, 0.92),
        LeverageProfile("aggressive_180", 1.80, -0.10, -0.20, -0.30, 1.40, 1.05, 0.75, 0.95),
        LeverageProfile("ultra_200", 2.00, -0.10, -0.20, -0.30, 1.50, 1.10, 0.80, 0.95),
        LeverageProfile("guard_150", 1.50, -0.08, -0.15, -0.25, 1.00, 0.80, 0.50, 0.85),
    ]

    combo_rows: list[dict[str, Any]] = []
    best_combo: dict[str, Any] | None = None
    best_obj = float("-inf")
    best_path = pd.DataFrame()

    for reeval_m in reeval_steps:
        selected_cid = cand_ids[0]
        path_rows: list[dict[str, Any]] = []
        for i, ym in enumerate(months_sorted):
            if i == 0 or (int(reeval_m) > 0 and (i % int(reeval_m) == 0)):
                train_start_i = max(0, i - int(args.train_lookback_months))
                train_months = months_sorted[train_start_i:i]
                if len(train_months) >= int(args.min_train_months):
                    scores = []
                    for cid in cand_ids:
                        z = panel[(panel["candidate_id"] == cid) & (panel["ym"].isin(train_months))].copy()
                        if z.empty:
                            continue
                        scores.append((cid, _candidate_score(z["ret"])))
                    if scores:
                        scores = sorted(scores, key=lambda x: x[1], reverse=True)
                        selected_cid = str(scores[0][0])
            row = panel[(panel["candidate_id"] == selected_cid) & (panel["ym"] == ym)]
            if row.empty:
                # fallback if sparse
                row = panel[panel["ym"] == ym].head(1)
                selected_cid = str(row.iloc[0]["candidate_id"])
            rr = row.iloc[0]
            path_rows.append(
                {
                    "ym": ym,
                    "candidate_id": selected_cid,
                    "base_ret": _safe_float(rr["ret"]),
                    "eqw_ret": _safe_float(rr["eqw_ret"]),
                    "signal_on": int(rr["signal_on"]),
                }
            )
        base_path = pd.DataFrame(path_rows)

        for lp in profiles:
            ret_lev, lev_used = _apply_leverage(base_path["base_ret"], base_path["signal_on"], lp)
            eqw = pd.to_numeric(base_path["eqw_ret"], errors="coerce").fillna(0.0)
            total = _total(ret_lev)
            dd = _mdd(ret_lev)
            wr = _monthly_win_rate(ret_lev)
            cagr = _cagr(ret_lev, len(ret_lev))
            obj = _safe_float(total) - 0.70 * abs(_safe_float(dd))
            combo = {
                "reeval_every_months": int(reeval_m),
                "profile": lp.name,
                "total_return": total,
                "cagr": cagr,
                "max_drawdown": dd,
                "monthly_win_rate": wr,
                "eqw_total_return": _total(eqw),
                "alpha_vs_eqw": _total(ret_lev) - _total(eqw),
                "objective": obj,
            }
            combo_rows.append(combo)
            if obj > best_obj:
                best_obj = obj
                best_combo = combo
                best_path = base_path.copy()
                best_path["ret_aggressive"] = ret_lev
                best_path["lev_used"] = lev_used

    if best_combo is None or best_path.empty:
        raise RuntimeError("failed to select best aggressive combo")

    # Best combo yearly and capital projection.
    yearly = _yearly(best_path["ret_aggressive"], best_path["ym"])
    yearly_eqw = _yearly(best_path["eqw_ret"], best_path["ym"]).rename(
        columns={"total_return": "eqw_total_return_y", "ann_return": "eqw_ann_return_y", "mdd": "eqw_mdd_y"}
    )
    yearly = yearly.merge(yearly_eqw, on="year", how="left")
    yearly["alpha_vs_eqw_y"] = yearly["total_return"] - yearly["eqw_total_return_y"]

    cap_rows: list[dict[str, Any]] = []
    for cap in capitals:
        v = float(cap)
        for r in pd.to_numeric(best_path["ret_aggressive"], errors="coerce").fillna(0.0).to_numpy(dtype=float):
            v *= (1.0 + float(r))
        vb = float(cap)
        for r in pd.to_numeric(best_path["eqw_ret"], errors="coerce").fillna(0.0).to_numpy(dtype=float):
            vb *= (1.0 + float(r))
        cap_rows.append(
            {
                "capital_inicial": float(cap),
                "capital_final_aggressive": float(v),
                "capital_final_eqw": float(vb),
                "ganho_aggressive": float(v - float(cap)),
                "ganho_eqw": float(vb - float(cap)),
                "delta_aggressive_menos_eqw": float(v - vb),
            }
        )

    combo_df = pd.DataFrame(combo_rows).sort_values("objective", ascending=False).reset_index(drop=True)
    combo_csv = outdir / "aggressive_combo_ranking.csv"
    path_csv = outdir / "aggressive_best_path.csv"
    yearly_csv = outdir / "aggressive_best_yearly.csv"
    cap_csv = outdir / "aggressive_capital_projection.csv"
    combo_df.to_csv(combo_csv, index=False)
    best_path.to_csv(path_csv, index=False)
    yearly.to_csv(yearly_csv, index=False)
    pd.DataFrame(cap_rows).to_csv(cap_csv, index=False)

    payload = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "impact_dir": str(impact_dir),
            "returns_csv": str(returns_csv),
            "test_start_ym": start_ym,
            "test_end_ym": end_ym,
            "train_end_for_candidate_generation": train_end_date,
            "train_lookback_months": int(args.train_lookback_months),
            "min_train_months": int(args.min_train_months),
            "reeval_every_months": reeval_steps,
            "candidates_count": int(len(candidates)),
        },
        "best_combo": best_combo,
        "over_100pct_total_return": bool(_safe_float(best_combo.get("total_return")) >= 1.0),
        "artifacts": {
            "combo_ranking_csv": str(combo_csv),
            "best_path_csv": str(path_csv),
            "best_yearly_csv": str(yearly_csv),
            "capital_projection_csv": str(cap_csv),
        },
    }
    out_json = outdir / "aggressive_adaptive_5y_summary.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "outdir": str(outdir),
                "summary_json": str(out_json),
                "candidates_count": int(len(candidates)),
                "best_total_return": _safe_float(best_combo.get("total_return")),
                "best_max_drawdown": _safe_float(best_combo.get("max_drawdown")),
                "over_100pct_total_return": bool(_safe_float(best_combo.get("total_return")) >= 1.0),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

