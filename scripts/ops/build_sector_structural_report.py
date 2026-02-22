#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_float(value: Any) -> float | None:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    return n if math.isfinite(n) else None


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _mean(values: list[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return float("nan")
    return float(sum(clean) / len(clean))


def _load_levels(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            level = str(raw.get("alert_level", "verde")).strip().lower()
            if level not in {"verde", "amarelo", "vermelho"}:
                level = "verde"
            rows.append(
                {
                    "sector": str(raw.get("sector", "unknown")).strip(),
                    "level": level,
                    "sector_score": _safe_float(raw.get("sector_score")),
                    "share_unstable": _safe_float(raw.get("share_unstable")),
                    "share_transition": _safe_float(raw.get("share_transition")),
                    "mean_confidence": _safe_float(raw.get("mean_confidence")),
                }
            )
    return rows


def _clarity_label(score: float) -> str:
    if score >= 0.70:
        return "high"
    if score >= 0.50:
        return "medium"
    return "low"


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(x) for x in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def build_report(
    *,
    run_id: str,
    levels_csv: Path,
    weekly_compare_json: Path,
    drift_json: Path,
) -> dict[str, Any]:
    levels = _load_levels(levels_csv)
    if not levels:
        return {
            "status": "fail",
            "run_id": run_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "reason": "sector_alert_levels_latest.csv sem dados",
            "sources": {
                "levels_csv": str(levels_csv),
                "weekly_compare_json": str(weekly_compare_json),
                "drift_json": str(drift_json),
            },
        }

    total = float(len(levels))
    counts_abs = {
        "vermelho": int(sum(1 for r in levels if r["level"] == "vermelho")),
        "amarelo": int(sum(1 for r in levels if r["level"] == "amarelo")),
        "verde": int(sum(1 for r in levels if r["level"] == "verde")),
        "total": int(total),
    }
    counts_ratio = {
        "red_ratio": float(counts_abs["vermelho"] / total),
        "yellow_ratio": float(counts_abs["amarelo"] / total),
        "green_ratio": float(counts_abs["verde"] / total),
    }

    mean_score = _mean([float(r["sector_score"]) for r in levels if r["sector_score"] is not None])
    mean_unstable = _mean([float(r["share_unstable"]) for r in levels if r["share_unstable"] is not None])
    mean_transition = _mean([float(r["share_transition"]) for r in levels if r["share_transition"] is not None])
    mean_confidence = _mean([float(r["mean_confidence"]) for r in levels if r["mean_confidence"] is not None])

    instability_component = 1.0 - _clip01(0.60 * max(0.0, mean_unstable) + 0.40 * max(0.0, mean_transition))
    confidence_component = _clip01(max(0.0, mean_confidence))
    green_component = _clip01(counts_ratio["green_ratio"])
    red_penalty = _clip01(counts_ratio["red_ratio"])
    clarity_score = _clip01(
        0.45 * instability_component
        + 0.35 * confidence_component
        + 0.20 * green_component
        - 0.20 * red_penalty
    )

    weekly_compare = _read_json(weekly_compare_json)
    weekly_summary = weekly_compare.get("summary") if isinstance(weekly_compare.get("summary"), dict) else {}
    changed_up = int(weekly_summary.get("changed_up") or 0)
    changed_down = int(weekly_summary.get("changed_down") or 0)
    unchanged = int(weekly_summary.get("unchanged") or 0)
    changed_total = max(0, changed_up + changed_down)
    change_rate = float(changed_total / max(1, int(total)))

    drift = _read_json(drift_json)
    drift_level = str(drift.get("drift_level", "unknown")).lower()
    drift_score = _safe_float(drift.get("drift_score"))
    drift_reasons = drift.get("reasons") if isinstance(drift.get("reasons"), list) else []
    baseline_window = drift.get("baseline_window") if isinstance(drift.get("baseline_window"), dict) else {}
    baseline_runs = int(baseline_window.get("n_runs") or 0)

    gate_status = "ok"
    gate_reasons: list[str] = []
    if drift_level == "block":
        gate_status = "block"
        gate_reasons.append("drift_level_block")
    elif drift_level == "watch":
        gate_status = "watch"
        gate_reasons.append("drift_level_watch")

    if clarity_score < 0.45 and counts_ratio["red_ratio"] >= 0.30:
        if gate_status != "block":
            gate_status = "watch"
        gate_reasons.append("low_clarity_with_high_red_ratio")
    if change_rate >= 0.35:
        if gate_status != "block":
            gate_status = "watch"
        gate_reasons.append("high_weekly_level_churn")

    payload = {
        "status": "ok",
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {
            **counts_abs,
            **counts_ratio,
        },
        "structural_clarity": {
            "score": clarity_score,
            "label": _clarity_label(clarity_score),
            "mean_sector_score": mean_score,
            "mean_unstable": mean_unstable,
            "mean_transition": mean_transition,
            "mean_confidence": mean_confidence,
            "instability_component": instability_component,
            "confidence_component": confidence_component,
            "green_component": green_component,
            "red_penalty": red_penalty,
        },
        "weekly_change": {
            "reference_run_id": weekly_compare.get("reference_run_id"),
            "changed_up": changed_up,
            "changed_down": changed_down,
            "unchanged": unchanged,
            "change_rate": change_rate,
        },
        "drift": {
            "level": drift_level,
            "score": drift_score,
            "baseline_runs": baseline_runs,
            "reasons": [str(x) for x in drift_reasons],
        },
        "gate_hint": {
            "status": gate_status,
            "reasons": sorted(set(gate_reasons)),
        },
        "sources": {
            "levels_csv": str(levels_csv),
            "weekly_compare_json": str(weekly_compare_json),
            "drift_json": str(drift_json),
        },
    }
    return _clean(payload)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build daily sector structural clarity + drift report.")
    ap.add_argument("--run-id", type=str, default="")
    ap.add_argument("--levels-csv", type=str, required=True)
    ap.add_argument("--weekly-compare-json", type=str, default="")
    ap.add_argument("--drift-json", type=str, default="")
    ap.add_argument("--out-json", type=str, default="")
    args = ap.parse_args()

    levels_csv = Path(str(args.levels_csv))
    if not levels_csv.is_absolute():
        levels_csv = ROOT / str(levels_csv)

    weekly_compare_json = Path(str(args.weekly_compare_json).strip()) if str(args.weekly_compare_json).strip() else (
        levels_csv.parent / "weekly_compare.json"
    )
    if not weekly_compare_json.is_absolute():
        weekly_compare_json = ROOT / str(weekly_compare_json)

    drift_json = Path(str(args.drift_json).strip()) if str(args.drift_json).strip() else (levels_csv.parent / "drift_monitor.json")
    if not drift_json.is_absolute():
        drift_json = ROOT / str(drift_json)

    run_id = str(args.run_id).strip() or levels_csv.parent.name
    payload = build_report(
        run_id=run_id,
        levels_csv=levels_csv,
        weekly_compare_json=weekly_compare_json,
        drift_json=drift_json,
    )

    out_json = Path(str(args.out_json).strip()) if str(args.out_json).strip() else (levels_csv.parent / "sector_structural_report.json")
    if not out_json.is_absolute():
        out_json = ROOT / str(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": payload.get("status", "fail"),
                "run_id": run_id,
                "clarity_score": ((payload.get("structural_clarity") or {}).get("score") if isinstance(payload, dict) else None),
                "drift_level": ((payload.get("drift") or {}).get("level") if isinstance(payload, dict) else None),
                "out_json": str(out_json),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
