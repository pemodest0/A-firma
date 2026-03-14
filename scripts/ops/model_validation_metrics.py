from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def _latest_run_dir(root: Path, suite_name: str) -> Path | None:
    base = root / "results" / "validation" / suite_name
    if not base.exists():
        return None
    runs = sorted([path for path in base.iterdir() if path.is_dir()], reverse=True)
    return runs[0] if runs else None


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def _worst_verdict(values: list[str]) -> str:
    order = {
        "robusto": 0,
        "aceitavel": 1,
        "provavel_overfit": 2,
        "desconhecido": 3,
        "": 3,
    }
    cleaned = [str(v or "").strip().lower() for v in values if str(v or "").strip()]
    if not cleaned:
        return "desconhecido"
    return max(cleaned, key=lambda item: order.get(item, 3))


def resolve_live_validation_metrics(*, root: Path, candidate_id: str) -> dict[str, Any]:
    candidate_key = str(candidate_id or "").strip()
    if not candidate_key:
        return {
            "candidate_id": "",
            "source_suite": "",
            "underperform_prob_63": None,
            "top3_total_retention": None,
            "pbo_verdict": "desconhecido",
        }

    champion_run = _latest_run_dir(root, "profit_champion_timing_robustness_suite")
    if champion_run is not None:
        compare_rows = _read_csv_rows(champion_run / "candidate_compare.csv")
        row = next((item for item in compare_rows if str(item.get("candidate_id") or "").strip() == candidate_key), None)
        summary = read_json(champion_run / "summary.json")
        pbo = summary.get("pbo_overall", {}) if isinstance(summary.get("pbo_overall"), dict) else {}
        verdict = _worst_verdict(
            [
                ((pbo.get("total_return") or {}).get("verdict") if isinstance(pbo.get("total_return"), dict) else ""),
                ((pbo.get("sharpe") or {}).get("verdict") if isinstance(pbo.get("sharpe"), dict) else ""),
            ]
        )
        if row is not None:
            return {
                "candidate_id": candidate_key,
                "source_suite": "profit_champion_timing_robustness_suite",
                "underperform_prob_63": _safe_float(row.get("underperform_prob_63")),
                "top3_total_retention": _safe_float(row.get("top3_total_retention")),
                "pbo_verdict": verdict,
            }

    pbo_run = _latest_run_dir(root, "profit_pbo_suite")
    resilience_run = _latest_run_dir(root, "profit_universe_resilience_suite")
    pbo_summary = read_json(pbo_run / "summary.json") if pbo_run is not None else {}
    resilience_summary = read_json(resilience_run / "summary.json") if resilience_run is not None else {}

    underperform = None
    rows = resilience_summary.get("attack_mc_base")
    if isinstance(rows, list):
        for row in rows:
            if int(row.get("horizon_days") or 0) == 63:
                underperform = _safe_float(row.get("underperform_prob"))
                break

    retention = None
    rows = resilience_summary.get("attack_retention_worst_nonbase")
    if isinstance(rows, list):
        preferred = next((row for row in rows if str(row.get("scenario") or "") == "drop_top3_crypto"), rows[0] if rows else None)
        if preferred is not None:
            retention = _safe_float(preferred.get("total_retention"))

    return {
        "candidate_id": candidate_key,
        "source_suite": "profit_universe_resilience_suite",
        "underperform_prob_63": underperform,
        "top3_total_retention": retention,
        "pbo_verdict": str(pbo_summary.get("overall_verdict") or "desconhecido").strip() or "desconhecido",
    }
