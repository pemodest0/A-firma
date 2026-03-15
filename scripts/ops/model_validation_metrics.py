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


def _latest_candidate_row(root: Path, suite_name: str, candidate_id: str) -> tuple[Path | None, dict[str, Any]]:
    run_dir = _latest_run_dir(root, suite_name)
    if run_dir is None:
        return None, {}
    rows = _read_csv_rows(run_dir / "candidate_compare.csv")
    row = next((item for item in rows if str(item.get("candidate_id") or "").strip() == str(candidate_id or "").strip()), None)
    return run_dir, (row or {})


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

    for suite_name in [
        "profit_official_post_fiscal_validation",
        "profit_champion_timing_robustness_suite",
        "profit_champion_extension_suite",
        "profit_champion_drawdown_suite",
        "profit_champion_selection_rotation_suite",
    ]:
        run_dir, row = _latest_candidate_row(root, suite_name, candidate_key)
        if run_dir is not None and row:
            summary = read_json(run_dir / "summary.json")
            pbo = summary.get("pbo_overall", {}) if isinstance(summary.get("pbo_overall"), dict) else {}
            verdict = _worst_verdict(
                [
                    ((pbo.get("total_return") or {}).get("verdict") if isinstance(pbo.get("total_return"), dict) else ""),
                    ((pbo.get("sharpe") or {}).get("verdict") if isinstance(pbo.get("sharpe"), dict) else ""),
                    str(summary.get("overall_verdict") or ""),
                ]
            )
            return {
                "candidate_id": candidate_key,
                "source_suite": suite_name,
                "underperform_prob_63": _safe_float(row.get("underperform_prob_63")),
                "top3_total_retention": _safe_float(row.get("top3_total_retention")),
                "pbo_verdict": verdict,
            }

    return {
        "candidate_id": candidate_key,
        "source_suite": "",
        "underperform_prob_63": None,
        "top3_total_retention": None,
        "pbo_verdict": "desconhecido",
    }
