#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _collect_gate_results() -> pd.DataFrame:
    base = ROOT / "results" / "portfolio_sim"
    rows: list[dict[str, Any]] = []

    # Gate 1-3
    for p in base.glob("*/tests_123_summary.json"):
        j = _read_json(p)
        source_run_dir = str(j.get("source_run_dir", "")).strip()
        if not source_run_dir:
            continue
        run_dir = Path(source_run_dir).resolve()
        run = run_dir.name
        t = j.get("tests") or {}
        s2 = t.get("2_walkforward_blocks") or {}
        s3 = t.get("3_random_baseline_same_exposure") or {}
        gate_123_ok = (
            float(s2.get("blocks_ok", 0)) >= float(s2.get("blocks_requested", 0))
            and float(s3.get("prob_strategy_beats_random", 0.0)) >= 0.70
        )
        rows.append(
            {
                "run": run,
                "run_dir": str(run_dir),
                "gate_123_ok": bool(gate_123_ok),
                "gate_45678_ok": False,
                "source_123": str(p),
                "source_45678": "",
            }
        )

    # Gate 4-8
    for p in base.glob("*/tests_45678_summary.json"):
        j = _read_json(p)
        source_run_dir = str(j.get("source_run_dir", "")).strip()
        if not source_run_dir:
            continue
        run_dir = Path(source_run_dir).resolve()
        run = run_dir.name
        gate_45678_ok = bool(j.get("launch_ready_round_4_to_8", False))
        rows.append(
            {
                "run": run,
                "run_dir": str(run_dir),
                "gate_123_ok": False,
                "gate_45678_ok": bool(gate_45678_ok),
                "source_123": "",
                "source_45678": str(p),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["run", "run_dir", "gate_123_ok", "gate_45678_ok", "source_123", "source_45678"])

    d = pd.DataFrame(rows)
    agg = (
        d.groupby(["run", "run_dir"], as_index=False)
        .agg(
            gate_123_ok=("gate_123_ok", "max"),
            gate_45678_ok=("gate_45678_ok", "max"),
            source_123=("source_123", lambda x: next((s for s in x if str(s).strip()), "")),
            source_45678=("source_45678", lambda x: next((s for s in x if str(s).strip()), "")),
        )
        .reset_index(drop=True)
    )
    agg["all_8_ok"] = agg["gate_123_ok"] & agg["gate_45678_ok"]
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Select best personal strategy release among runs that pass all 8 gates."
    )
    ap.add_argument(
        "--out",
        default="results/portfolio_sim/personal_latest_release.json",
        help="Output JSON pointer for personal release.",
    )
    args = ap.parse_args()

    gates = _collect_gate_results()
    if gates.empty:
        raise RuntimeError("No gate summaries found in results/portfolio_sim.")

    candidates = gates[gates["all_8_ok"]].copy()
    if candidates.empty:
        raise RuntimeError("No run passes all 8 gates.")

    # Join with strategy metrics.
    metrics_rows: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        run = str(row.get("run", "")).strip()
        run_dir = Path(str(row.get("run_dir", ""))).resolve()
        ss_primary = run_dir / "systematic_summary.json"
        ss_fallback = ROOT / "results" / "portfolio_sim" / run / "systematic_summary.json"
        ss = ss_primary if ss_primary.exists() else ss_fallback
        j = _read_json(ss)
        metrics_rows.append(
            {
                "run": run,
                "run_dir": str(run_dir),
                "strategy_total": float(j.get("strategy_total", float("nan"))),
                "strategy_ann": float(j.get("strategy_ann", float("nan"))),
                "strategy_max_drop": float(j.get("strategy_max_drop", float("nan"))),
                "worth_it_rate_vs_eqw": float(j.get("worth_it_rate_vs_eqw", float("nan"))),
                "monthly_alpha_prob_positive_vs_eqw": float(
                    j.get("monthly_alpha_prob_positive_vs_eqw", float("nan"))
                ),
                "systematic_summary": str(ss),
            }
        )

    metrics = pd.DataFrame(metrics_rows)
    merged = candidates.merge(metrics, on=["run", "run_dir"], how="left")
    merged = merged[pd.to_numeric(merged["strategy_total"], errors="coerce").notna()].copy()
    if merged.empty:
        raise RuntimeError("No all-8-gates candidate has valid systematic_summary metrics.")
    merged = merged.sort_values(["strategy_total", "strategy_ann"], ascending=[False, False]).reset_index(drop=True)
    champion = merged.iloc[0].to_dict()

    payload = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "personal_release_requires_all_8_gates",
        "champion": champion,
        "all_gate_passed_candidates": merged.to_dict(orient="records"),
        "failed_or_incomplete": gates[~gates["all_8_ok"]].sort_values("run").to_dict(orient="records"),
    }

    out = (ROOT / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "out": str(out),
                "champion_run": str(champion.get("run", "")),
                "champion_strategy_total": float(champion.get("strategy_total", float("nan"))),
                "n_candidates_all_8": int(merged.shape[0]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
