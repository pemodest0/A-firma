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


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if np.isfinite(v) else float("nan")


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
    raise FileNotFoundError("no non-empty impact_training_dataset.csv found under results/lab_corr_macro/*/hierarchical/impact_learning*")


def _run(cmd: list[str], *, cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def _build_base_run(
    *,
    impact_dir: Path,
    returns_csv: Path,
    train_end_date: str,
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
        str(train_end_date),
        "--start-ym",
        "2019-01",
        "--top-k-options",
        "20",
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
    _run(cmd, cwd=ROOT)
    if not (outdir / "simulation_summary.json").exists():
        raise FileNotFoundError(f"missing simulation_summary.json in {outdir}")
    return outdir


def _run_investment_for_year(
    *,
    base_run_dir: Path,
    top_k_list: str,
    capitals: str,
    train_end_ym: str,
    test_start_ym: str,
    test_end_ym: str,
    outdir: Path,
) -> dict[str, Any]:
    py = sys.executable
    script = ROOT / "scripts" / "ops" / "run_investment_scenarios.py"
    cmd = [
        py,
        str(script),
        "--base-run-dir",
        str(base_run_dir),
        "--top-k-list",
        str(top_k_list),
        "--capitals",
        str(capitals),
        "--train-end",
        str(train_end_ym),
        "--test-start",
        str(test_start_ym),
        "--test-end",
        str(test_end_ym),
        "--select-topk-on",
        "train",
        "--outdir",
        str(outdir),
    ]
    _run(cmd, cwd=ROOT)
    summary = json.loads((outdir / "investment_scenarios_summary.json").read_text(encoding="utf-8"))
    scen = pd.read_csv(outdir / "scenario_summary.csv")
    proj = pd.read_csv(outdir / "capital_projection.csv")
    return {"summary": summary, "scenario_df": scen, "projection_df": proj}


def _direction_scan(
    *,
    impact_csv: Path,
    year: int,
    outdir: Path,
) -> dict[str, Any]:
    py = sys.executable
    script = ROOT / "scripts" / "ops" / "analyze_sector_global_direction.py"
    start_date = f"{int(year)}-01-01"
    end_date = f"{int(year)}-12-31"
    cmd = [
        py,
        str(script),
        "--impact-csv",
        str(impact_csv),
        "--sector-kind",
        "gics",
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--lead-days",
        "5",
        "--min-obs",
        "30",
        "--outdir",
        str(outdir),
    ]
    _run(cmd, cwd=ROOT)
    return json.loads((outdir / "sector_global_direction_summary.json").read_text(encoding="utf-8"))


def _pick_oracle_row(scen: pd.DataFrame) -> dict[str, Any]:
    x = scen.copy()
    for c in [
        "test_real_total_return",
        "test_real_max_drawdown",
        "test_base_total_return",
        "test_base_max_drawdown",
        "full_real_total_return",
        "full_real_max_drawdown",
    ]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    safe = x[x["test_real_total_return"] > 0.0].copy()
    if safe.empty:
        # fallback: least drawdown even if return <= 0
        best = x.sort_values(["test_real_max_drawdown", "test_real_total_return"], ascending=[False, False]).iloc[0]
    else:
        best = safe.sort_values(["test_real_max_drawdown", "test_real_total_return"], ascending=[False, False]).iloc[0]
    return best.to_dict()


def _projection_for_topk(proj: pd.DataFrame, top_k: int) -> list[dict[str, Any]]:
    z = proj[pd.to_numeric(proj["top_k"], errors="coerce").fillna(-1).astype(int) == int(top_k)].copy()
    if z.empty:
        return []
    z["capital_inicial"] = pd.to_numeric(z["capital_inicial"], errors="coerce")
    z = z.sort_values("capital_inicial")
    return z.to_dict(orient="records")


def main() -> None:
    ap = argparse.ArgumentParser(description="Long scenario scan for yearly investment contexts and sector/global direction.")
    ap.add_argument("--impact-dir", default="")
    ap.add_argument("--returns-csv", default="")
    ap.add_argument("--years", default="2025,2026")
    ap.add_argument("--top-k-list", default="10,16,20,24,32,40,52,64,80")
    ap.add_argument("--capitals", default="1000,5000,10000,25000,50000,100000")
    ap.add_argument("--outdir", default="")
    args = ap.parse_args()

    if str(args.impact_dir).strip():
        impact_dir = Path(args.impact_dir).resolve()
        returns_csv = Path(args.returns_csv).resolve() if str(args.returns_csv).strip() else impact_dir.parents[1] / "returns_wide_core.csv"
    else:
        impact_dir, returns_csv = _latest_nonempty_impact()
    impact_csv = impact_dir / "impact_training_dataset.csv"
    if not impact_csv.exists():
        raise FileNotFoundError(f"missing file: {impact_csv}")
    if not returns_csv.exists():
        raise FileNotFoundError(f"missing file: {returns_csv}")

    years = _parse_int_list(args.years)
    _parse_int_list(args.top_k_list)
    _parse_float_list(args.capitals)

    run_id = _run_id()
    outdir = Path(args.outdir).resolve() if str(args.outdir).strip() else (ROOT / "results" / "portfolio_sim" / f"{run_id}_long_scenario_scan")
    outdir.mkdir(parents=True, exist_ok=True)

    year_rows: list[dict[str, Any]] = []
    for year in years:
        train_year = int(year) - 1
        train_end_ym = f"{train_year}-12"
        train_end_date = str(pd.to_datetime(f"{train_end_ym}-01").to_period("M").end_time.date())
        test_start_ym = f"{int(year)}-01"
        test_end_ym = f"{int(year)}-12"

        year_dir = outdir / f"year_{int(year)}"
        base_dir = year_dir / "base_systematic"
        scen_dir = year_dir / "investment_scenarios"
        dir_dir = year_dir / "sector_global_direction"
        year_dir.mkdir(parents=True, exist_ok=True)

        _build_base_run(
            impact_dir=impact_dir,
            returns_csv=returns_csv,
            train_end_date=train_end_date,
            outdir=base_dir,
        )
        inv = _run_investment_for_year(
            base_run_dir=base_dir,
            top_k_list=str(args.top_k_list),
            capitals=str(args.capitals),
            train_end_ym=train_end_ym,
            test_start_ym=test_start_ym,
            test_end_ym=test_end_ym,
            outdir=scen_dir,
        )
        direction = _direction_scan(
            impact_csv=impact_csv,
            year=int(year),
            outdir=dir_dir,
        )

        best = (inv["summary"] or {}).get("best_scenario", {})
        best_top_k = int(_safe_float(best.get("top_k", 20)))
        oracle = _pick_oracle_row(inv["scenario_df"])
        proj_best = _projection_for_topk(inv["projection_df"], best_top_k)
        top_sector_leads = (direction.get("top_sector_leads_global") or [])[:5]
        top_global_leads = (direction.get("top_global_leads_sector") or [])[:5]

        year_rows.append(
            {
                "year": int(year),
                "train_end": str(train_end_ym),
                "test_start": str(test_start_ym),
                "test_end": str(test_end_ym),
                "best_top_k": best_top_k,
                "best_test_return": _safe_float(best.get("test_real_total_return")),
                "best_test_max_drawdown": _safe_float(best.get("test_real_max_drawdown")),
                "best_base_test_return": _safe_float(best.get("test_base_total_return")),
                "best_base_test_max_drawdown": _safe_float(best.get("test_base_max_drawdown")),
                "oracle_top_k": int(_safe_float(oracle.get("top_k", np.nan))) if np.isfinite(_safe_float(oracle.get("top_k", np.nan))) else None,
                "oracle_test_return": _safe_float(oracle.get("test_real_total_return")),
                "oracle_test_max_drawdown": _safe_float(oracle.get("test_real_max_drawdown")),
                "oracle_base_test_return": _safe_float(oracle.get("test_base_total_return")),
                "oracle_base_test_max_drawdown": _safe_float(oracle.get("test_base_max_drawdown")),
                "capital_projection_best_top_k": proj_best,
                "top_sector_leads_global": top_sector_leads,
                "top_global_leads_sector": top_global_leads,
                "artifacts": {
                    "base_dir": str(base_dir),
                    "scenarios_dir": str(scen_dir),
                    "direction_dir": str(dir_dir),
                    "scenario_summary_csv": str(scen_dir / "scenario_summary.csv"),
                    "capital_projection_csv": str(scen_dir / "capital_projection.csv"),
                    "direction_summary_json": str(dir_dir / "sector_global_direction_summary.json"),
                },
            }
        )

    yearly_df = pd.DataFrame(year_rows)
    yearly_csv = outdir / "yearly_long_scenario_summary.csv"
    yearly_df.to_csv(yearly_csv, index=False)

    payload = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "impact_dir": str(impact_dir),
            "impact_csv": str(impact_csv),
            "returns_csv": str(returns_csv),
            "years": years,
            "top_k_list": str(args.top_k_list),
            "capitals": str(args.capitals),
        },
        "leakage_note": "oracle_* fields are test-based diagnostics only; operational selection is best_top_k from causal train selection in run_investment_scenarios.",
        "year_results": year_rows,
        "artifacts": {
            "yearly_summary_csv": str(yearly_csv),
        },
    }
    out_json = outdir / "long_scenario_scan_summary.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "outdir": str(outdir),
                "summary_json": str(out_json),
                "years": years,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
