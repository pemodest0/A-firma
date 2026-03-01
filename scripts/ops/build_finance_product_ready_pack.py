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


ROOT = Path(__file__).resolve().parents[2]


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_float(x: Any) -> float:
    try:
        y = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return y if np.isfinite(y) else float("nan")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _sanitize_json_value(x: Any) -> Any:
    if isinstance(x, float):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): _sanitize_json_value(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_sanitize_json_value(v) for v in x]
    return x


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=str(ROOT), check=True)  # noqa: S603


def _resolve_latest_finance_run() -> Path:
    pointer = ROOT / "results" / "lab_corr_macro" / "latest_release.json"
    p = _read_json(pointer)
    run_dir = Path(str(p.get("run_dir", "")).strip()) if str(p.get("run_dir", "")).strip() else None
    if run_dir is not None and run_dir.exists() and (run_dir / "hierarchical").exists():
        return run_dir
    base = ROOT / "results" / "lab_corr_macro"
    runs = sorted([d for d in base.iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True)
    for d in runs:
        if (d / "hierarchical" / "impact_learning_2015_2026_compare").exists():
            return d
    for d in runs:
        if (d / "hierarchical").exists():
            return d
    raise FileNotFoundError("no finance run with hierarchical artifacts found")


def _resolve_impact_dir(run_dir: Path, raw: str) -> Path:
    if str(raw).strip():
        p = Path(str(raw))
        if not p.is_absolute():
            p = ROOT / str(raw)
        return p
    primary = run_dir / "hierarchical" / "impact_learning_2015_2026_compare"
    fallback = run_dir / "hierarchical" / "impact_learning"
    return primary if primary.exists() else fallback


def _check(condition: bool, name: str, detail: str) -> dict[str, Any]:
    return {"name": str(name), "ok": bool(condition), "detail": str(detail)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build finance product-readiness pack (state + evidence + operational brief).")
    ap.add_argument("--run-dir", type=str, default="")
    ap.add_argument("--impact-dir", type=str, default="")
    ap.add_argument("--alert-budget", type=float, default=0.15)
    ap.add_argument("--alert-budget-sweep", type=str, default="0.10,0.15,0.20")
    ap.add_argument("--alert-dedupe-days", type=int, default=20)
    ap.add_argument("--lead-window-days", type=int, default=30)
    ap.add_argument("--min-event-gap-days", type=int, default=20)
    ap.add_argument("--ai-outdir", type=str, default="results/ops/ai_knowledge")
    ap.add_argument("--outdir", type=str, default="results/ops/finance_product_ready")
    args = ap.parse_args()

    if str(args.run_dir).strip():
        run_dir = Path(str(args.run_dir))
        if not run_dir.is_absolute():
            run_dir = ROOT / str(args.run_dir)
    else:
        run_dir = _resolve_latest_finance_run()
    run_dir = run_dir.resolve()
    if not run_dir.exists():
        raise SystemExit(f"run dir not found: {run_dir}")

    impact_dir = _resolve_impact_dir(run_dir=run_dir, raw=str(args.impact_dir)).resolve()
    if not impact_dir.exists():
        raise SystemExit(f"impact dir not found: {impact_dir}")

    ai_outdir = Path(str(args.ai_outdir))
    if not ai_outdir.is_absolute():
        ai_outdir = ROOT / str(args.ai_outdir)
    ai_outdir.mkdir(parents=True, exist_ok=True)

    outdir = Path(str(args.outdir))
    if not outdir.is_absolute():
        outdir = ROOT / str(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    _run(
        [
            sys.executable,
            "scripts/structural/build_historical_structure_assessment.py",
            "--run-dir",
            str(run_dir),
            "--impact-dir",
            str(impact_dir),
            "--alert-budget",
            str(float(args.alert_budget)),
            "--alert-budget-sweep",
            str(args.alert_budget_sweep),
            "--alert-dedupe-days",
            str(int(args.alert_dedupe_days)),
            "--lead-window-days",
            str(int(args.lead_window_days)),
            "--min-event-gap-days",
            str(int(args.min_event_gap_days)),
        ]
    )

    _run(
        [
            sys.executable,
            "scripts/ops/build_ai_operational_brief.py",
            "--run-dir",
            str(run_dir),
            "--impact-dir",
            str(impact_dir),
            "--outdir",
            str(ai_outdir),
        ]
    )

    hist_summary_path = impact_dir / "historical_structure_summary.json"
    hist_next_path = impact_dir / "historical_structure_next_month_indication.json"
    hist_budget_sweep = impact_dir / "historical_structure_stress_prealert_budget_sweep.csv"
    ai_latest_path = ai_outdir / "latest_operational_brief.json"
    ai_latest = _read_json(ai_latest_path)
    ai_brief_path = Path(str(ai_latest.get("operational_brief_path", ""))).resolve() if str(ai_latest.get("operational_brief_path", "")).strip() else None
    ai_brief = _read_json(ai_brief_path) if ai_brief_path is not None else {}
    hist_summary = _read_json(hist_summary_path)
    next_month = _read_json(hist_next_path)

    reg_hw = (
        (((ai_brief.get("model_evidence") or {}).get("horizon_winners_global") or {}).get("winners") or {}).get(
            "regime_entry", {}
        )
    )
    reg_f1 = _safe_float((reg_hw or {}).get("f1_mean"))
    freshness_status = str(((ai_brief.get("freshness") or {}).get("status", "")).strip().lower())
    data_last_date = str(ai_brief.get("data_last_date") or hist_summary.get("data_last_date") or "")

    checks = [
        _check(hist_summary_path.exists(), "historical_summary_exists", str(hist_summary_path)),
        _check(hist_next_path.exists(), "next_month_exists", str(hist_next_path)),
        _check(hist_budget_sweep.exists(), "budget_sweep_exists", str(hist_budget_sweep)),
        _check(ai_latest_path.exists(), "ai_latest_exists", str(ai_latest_path)),
        _check(bool(data_last_date), "data_last_date_present", data_last_date),
        _check(str(next_month.get("status", "")) == "ok", "next_month_status_ok", str(next_month.get("status", ""))),
        _check(len(hist_summary.get("stress_prealert_summary_top10", []) or []) > 0, "stress_prealert_summary_non_empty", "top10 available"),
        _check(np.isfinite(reg_f1), "regime_horizon_f1_available", f"{reg_f1}"),
    ]

    warnings: list[str] = []
    if freshness_status == "stale":
        warnings.append("data freshness is stale (>7 days lag)")
    if np.isfinite(reg_f1) and reg_f1 < 0.55:
        warnings.append("regime horizon f1_mean below 0.55")

    required_ok = all(bool(c.get("ok")) for c in checks)
    if not required_ok:
        overall = "fail"
    elif warnings:
        overall = "warn"
    else:
        overall = "pass"

    rid = _run_id()
    payload = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_readiness": overall,
        "run_dir": str(run_dir),
        "impact_dir": str(impact_dir),
        "data_last_date": data_last_date,
        "risk_level_next_month": str(((ai_brief.get("operational_signal") or {}).get("risk_level_next_month", ""))),
        "operational_state": str(((ai_brief.get("operational_signal") or {}).get("operational_state", ""))),
        "confidence_score": _safe_float(((ai_brief.get("operational_signal") or {}).get("confidence_score"))),
        "selected_alert_budget": _safe_float(((hist_summary.get("lead_alert_config") or {}).get("selected_alert_budget"))),
        "regime_horizon_f1_mean": reg_f1,
        "checks": checks,
        "warnings": warnings,
        "artifacts": {
            "historical_summary_json": str(hist_summary_path),
            "historical_next_month_json": str(hist_next_path),
            "historical_budget_sweep_csv": str(hist_budget_sweep),
            "ai_latest_json": str(ai_latest_path),
            "ai_brief_json": str(ai_brief_path) if ai_brief_path is not None else "",
        },
    }
    payload = _sanitize_json_value(payload)

    out_json = outdir / f"finance_product_ready_{rid}.json"
    out_md = outdir / f"finance_product_ready_{rid}.md"
    latest = outdir / "latest_finance_product_ready.json"

    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Finance Product Readiness",
        "",
        f"- generated_at_utc: {payload.get('generated_at_utc')}",
        f"- overall_readiness: {payload.get('overall_readiness')}",
        f"- run_dir: {payload.get('run_dir')}",
        f"- data_last_date: {payload.get('data_last_date')}",
        f"- risk_level_next_month: {payload.get('risk_level_next_month')}",
        f"- operational_state: {payload.get('operational_state')}",
        f"- confidence_score: {payload.get('confidence_score')}",
        f"- selected_alert_budget: {payload.get('selected_alert_budget')}",
        f"- regime_horizon_f1_mean: {payload.get('regime_horizon_f1_mean')}",
        "",
        "## Checks",
    ]
    for c in payload.get("checks", []):
        mark = "PASS" if c.get("ok") else "FAIL"
        lines.append(f"- [{mark}] {c.get('name')}: {c.get('detail')}")
    if payload.get("warnings"):
        lines.append("")
        lines.append("## Warnings")
        for w in payload.get("warnings", []):
            lines.append(f"- {w}")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    latest.write_text(
        json.dumps(
            _sanitize_json_value(
                {
                    "status": "ok",
                    "generated_at_utc": payload.get("generated_at_utc"),
                    "overall_readiness": payload.get("overall_readiness"),
                    "run_dir": payload.get("run_dir"),
                    "data_last_date": payload.get("data_last_date"),
                    "finance_product_ready_json": str(out_json),
                    "finance_product_ready_md": str(out_md),
                }
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "overall_readiness": payload.get("overall_readiness"),
                "out_json": str(out_json),
                "out_md": str(out_md),
                "latest": str(latest),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
