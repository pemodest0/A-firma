#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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


def _resolve_path(raw: str | Path | None) -> Path | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    p = Path(text)
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return p


def _extract_gt_best(ground_truth_summary: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"status": "empty", "best_by_ground_truth": {}, "rows": 0}
    rows: list[dict[str, Any]] = []
    for hs in ground_truth_summary.get("horizon_summaries", []) or []:
        h = int(hs.get("horizon_days", 0) or 0)
        tm = hs.get("test_metrics", {}) or {}
        for gt_name in ["ground_truth_drawdown", "ground_truth_regime_entry"]:
            per_model = tm.get(gt_name, {}) or {}
            for model_name, metrics in per_model.items():
                rows.append(
                    {
                        "ground_truth": str(gt_name),
                        "horizon_days": int(h),
                        "model": str(model_name),
                        "f1": _safe_float((metrics or {}).get("f1")),
                        "recall": _safe_float((metrics or {}).get("recall")),
                        "precision": _safe_float((metrics or {}).get("precision")),
                        "event_rate": _safe_float((metrics or {}).get("event_rate")),
                        "alert_rate": _safe_float((metrics or {}).get("alert_rate")),
                    }
                )
    if not rows:
        return out
    df = pd.DataFrame(rows)
    out["rows"] = int(df.shape[0])
    best_map: dict[str, Any] = {}
    for gt_name in sorted(df["ground_truth"].unique().tolist()):
        d = df[df["ground_truth"] == gt_name].copy()
        d = d.sort_values(["f1", "recall", "precision"], ascending=[False, False, False]).reset_index(drop=True)
        if d.empty:
            continue
        top = d.iloc[0]
        best_map[str(gt_name)] = {
            "horizon_days": int(top["horizon_days"]),
            "model": str(top["model"]),
            "f1": _safe_float(top["f1"]),
            "recall": _safe_float(top["recall"]),
            "precision": _safe_float(top["precision"]),
            "event_rate": _safe_float(top["event_rate"]),
            "alert_rate": _safe_float(top["alert_rate"]),
        }
    out["status"] = "ok"
    out["best_by_ground_truth"] = best_map
    return out


def _load_horizon_winners(run_dir: Path) -> dict[str, Any]:
    out = {"status": "missing", "path": "", "winners": {}}
    p = run_dir / "hierarchical" / "epistemic_horizon_sweep_global" / "horizon_winners_global.csv"
    if not p.exists():
        return out
    d = pd.read_csv(p)
    if d.empty:
        out["status"] = "empty"
        out["path"] = str(p)
        return out
    winners: dict[str, Any] = {}
    for gt_name in sorted(d["ground_truth"].astype(str).unique().tolist()):
        g = d[d["ground_truth"].astype(str) == gt_name].copy()
        if g.empty:
            continue
        x = g.iloc[0]
        winners[str(gt_name)] = {
            "horizon_days": int(float(x.get("horizon_days", float("nan")))),
            "model": str(x.get("model", "")),
            "f1_mean": _safe_float(x.get("f1_mean")),
            "recall_mean": _safe_float(x.get("recall_mean")),
            "lift_precision_mean": _safe_float(x.get("lift_precision_mean")),
            "lead_time_median": _safe_float(x.get("lead_time_median")),
        }
    return {"status": "ok", "path": str(p), "winners": winners}


def _risk_to_action(risk_level: str) -> dict[str, Any]:
    rl = str(risk_level).strip().lower()
    if rl == "alto":
        return {
            "operational_state": "defensivo",
            "action_hint": "reduzir risco, priorizar protecao e diminuir agressividade de alocacao",
            "priority": "alta",
        }
    if rl == "moderado":
        return {
            "operational_state": "cautela_ativa",
            "action_hint": "manter monitoramento reforcado e escalonar risco com gatilhos de confirmacao",
            "priority": "media",
        }
    return {
        "operational_state": "monitoramento_normal",
        "action_hint": "manter operacao normal com vigilancia de mudanca estrutural",
        "priority": "normal",
    }


def _parse_date(text: Any) -> date | None:
    s = str(text or "").strip()
    if not s:
        return None
    try:
        return pd.Timestamp(s).date()
    except Exception:
        return None


def _freshness_label(last_date: date | None, today: date) -> dict[str, Any]:
    if last_date is None:
        return {"status": "unknown", "days_lag": None}
    lag = int((today - last_date).days)
    if lag <= 3:
        st = "fresh"
    elif lag <= 7:
        st = "attention"
    else:
        st = "stale"
    return {"status": st, "days_lag": lag}


def _build_insights(*, summary_hist: dict[str, Any], horizon_winners: dict[str, Any], gt_best: dict[str, Any]) -> list[dict[str, Any]]:
    insights: list[dict[str, Any]] = []

    nw = (summary_hist.get("next_month_indication") or {}) if isinstance(summary_hist, dict) else {}
    risk = str(nw.get("risk_level_next_month", "unknown"))
    data_last = str(summary_hist.get("data_last_date", "") or nw.get("data_last_date", ""))
    if data_last:
        insights.append(
            {
                "id": "data_last_date",
                "level": "info",
                "message": f"ultima data efetiva da base: {data_last}",
                "evidence": {"data_last_date": data_last},
            }
        )

    if risk:
        insights.append(
            {
                "id": "next_month_risk",
                "level": "info",
                "message": f"estado operacional projetado para o proximo mes: {risk}",
                "evidence": {"risk_level_next_month": risk},
            }
        )

    hw = (horizon_winners.get("winners") or {}) if isinstance(horizon_winners, dict) else {}
    reg = hw.get("regime_entry")
    if isinstance(reg, dict):
        insights.append(
            {
                "id": "best_horizon_regime_entry",
                "level": "high",
                "message": f"horizonte mais forte para estado proximo (regime_entry): h={reg.get('horizon_days')}d",
                "evidence": reg,
            }
        )
    dd = hw.get("drawdown")
    if isinstance(dd, dict):
        insights.append(
            {
                "id": "best_horizon_drawdown",
                "level": "medium",
                "message": f"horizonte selecionado para drawdown no sweep atual: h={dd.get('horizon_days')}d",
                "evidence": dd,
            }
        )

    gt = (gt_best.get("best_by_ground_truth") or {}) if isinstance(gt_best, dict) else {}
    gt_reg = gt.get("ground_truth_regime_entry")
    if isinstance(gt_reg, dict):
        insights.append(
            {
                "id": "ground_truth_regime_entry",
                "level": "high",
                "message": "ground truth confirma sinal mais forte para regime_entry do que para drawdown",
                "evidence": gt_reg,
            }
        )
    return insights


def main() -> None:
    ap = argparse.ArgumentParser(description="Build unified operational brief for AI from Assyntrax artifacts.")
    ap.add_argument("--run-dir", type=str, default="")
    ap.add_argument("--impact-dir", type=str, default="")
    ap.add_argument("--outdir", type=str, default="results/ops/ai_knowledge")
    args = ap.parse_args()

    outdir = (ROOT / str(args.outdir).strip()) if not Path(str(args.outdir)).is_absolute() else Path(str(args.outdir))
    outdir.mkdir(parents=True, exist_ok=True)

    latest_gt_ptr = _read_json(ROOT / "results" / "ops" / "ai_knowledge" / "latest_ground_truth.json")
    latest_impact_ptr = _read_json(ROOT / "results" / "ops" / "ai_knowledge" / "latest_structural_impact.json")

    run_dir = _resolve_path(args.run_dir) if str(args.run_dir).strip() else _resolve_path(latest_impact_ptr.get("source_run_dir"))
    if run_dir is None:
        raise SystemExit("missing run_dir (provide --run-dir or latest_structural_impact.json)")
    impact_dir = _resolve_path(args.impact_dir) if str(args.impact_dir).strip() else _resolve_path(Path(run_dir) / "hierarchical" / "impact_learning_2015_2026_compare")
    if impact_dir is None or (not Path(impact_dir).exists()):
        fallback = Path(run_dir) / "hierarchical" / "impact_learning"
        impact_dir = fallback if fallback.exists() else impact_dir
    if impact_dir is None or (not Path(impact_dir).exists()):
        ptr_run = _resolve_path(latest_impact_ptr.get("source_run_dir"))
        if ptr_run is not None:
            ptr_primary = ptr_run / "hierarchical" / "impact_learning_2015_2026_compare"
            ptr_fallback = ptr_run / "hierarchical" / "impact_learning"
            if ptr_primary.exists():
                run_dir = ptr_run
                impact_dir = ptr_primary
            elif ptr_fallback.exists():
                run_dir = ptr_run
                impact_dir = ptr_fallback

    gt_summary_path = _resolve_path(latest_gt_ptr.get("summary_path"))
    impact_summary_path = _resolve_path(latest_impact_ptr.get("summary_path"))
    gt_summary = _read_json(gt_summary_path) if gt_summary_path is not None else {}
    impact_summary = _read_json(impact_summary_path) if impact_summary_path is not None else {}

    hist_summary_path = Path(impact_dir) / "historical_structure_summary.json"
    hist_summary = _read_json(hist_summary_path)
    next_month_path = Path(impact_dir) / "historical_structure_next_month_indication.json"
    next_month = _read_json(next_month_path)
    if hist_summary and ("next_month_indication" not in hist_summary):
        hist_summary["next_month_indication"] = next_month

    rankings_latest_path = Path(run_dir) / "platform" / "rankings_latest.json"
    rankings_latest = _read_json(rankings_latest_path)

    horizon_winners = _load_horizon_winners(Path(run_dir))
    gt_best = _extract_gt_best(gt_summary)

    data_last_date = str(hist_summary.get("data_last_date") or next_month.get("data_last_date") or next_month.get("as_of_date") or "")
    today = datetime.now(timezone.utc).date()
    freshness = _freshness_label(_parse_date(data_last_date), today=today)
    risk_level = str((next_month or {}).get("risk_level_next_month", "unknown"))
    action = _risk_to_action(risk_level)

    insights = _build_insights(summary_hist=hist_summary, horizon_winners=horizon_winners, gt_best=gt_best)

    confidence_parts: list[float] = []
    if freshness.get("status") == "fresh":
        confidence_parts.append(1.0)
    elif freshness.get("status") == "attention":
        confidence_parts.append(0.7)
    elif freshness.get("status") == "stale":
        confidence_parts.append(0.4)
    reg_hw = (horizon_winners.get("winners") or {}).get("regime_entry", {})
    reg_f1 = _safe_float(reg_hw.get("f1_mean"))
    if np.isfinite(reg_f1):
        if reg_f1 >= 0.75:
            confidence_parts.append(1.0)
        elif reg_f1 >= 0.55:
            confidence_parts.append(0.7)
        else:
            confidence_parts.append(0.4)
    confidence_score = float(np.mean(confidence_parts)) if confidence_parts else float("nan")

    brief = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_last_date": data_last_date,
        "freshness": freshness,
        "run_context": {
            "run_dir": str(run_dir),
            "impact_dir": str(impact_dir) if impact_dir is not None else "",
        },
        "sources": {
            "latest_ground_truth_json": str(ROOT / "results" / "ops" / "ai_knowledge" / "latest_ground_truth.json"),
            "latest_structural_impact_json": str(ROOT / "results" / "ops" / "ai_knowledge" / "latest_structural_impact.json"),
            "ground_truth_summary_json": str(gt_summary_path) if gt_summary_path is not None else "",
            "impact_summary_json": str(impact_summary_path) if impact_summary_path is not None else "",
            "historical_structure_summary_json": str(hist_summary_path),
            "historical_structure_next_month_json": str(next_month_path),
            "rankings_latest_json": str(rankings_latest_path) if rankings_latest_path.exists() else "",
            "horizon_winners_global_csv": str(horizon_winners.get("path", "")),
        },
        "operational_signal": {
            "risk_level_next_month": risk_level,
            "operational_state": action["operational_state"],
            "action_hint": action["action_hint"],
            "priority": action["priority"],
            "confidence_score": confidence_score,
        },
        "model_evidence": {
            "ground_truth_best": gt_best,
            "horizon_winners_global": horizon_winners,
            "historical_verdict_counts_expanding": (hist_summary.get("verdict_counts_expanding") or {}),
            "stress_prealert_top": (hist_summary.get("stress_prealert_summary_top10") or [])[:5],
        },
        "state_snapshot": {
            "next_month_indication": next_month,
            "top_sectors_global_mode": rankings_latest.get("top_sectors_global_mode", []),
            "top_assets_global_mode": rankings_latest.get("top_assets_global_mode", []),
            "sector_global_overlap": rankings_latest.get("sector_global_overlap", []),
            "global_state": rankings_latest.get("global_state", {}),
        },
        "insights": insights,
        "guardrails": [
            "nao afirmar direcao de preco; usar linguagem de estado estrutural",
            "citar sempre data_last_date e run_id/arquivo fonte",
            "se data_last_date estiver stale, reduzir confianca e avisar explicitamente",
            "para operacao, usar regime/horizon como gatilho e drawdown como confirmacao secundaria",
        ],
    }
    brief = _sanitize_json_value(brief)

    out_path = outdir / f"operational_brief_{_run_id()}.json"
    out_path.write_text(json.dumps(brief, indent=2, ensure_ascii=False), encoding="utf-8")
    latest_path = outdir / "latest_operational_brief.json"
    latest_payload = {
        "status": "ok",
        "generated_at_utc": brief["generated_at_utc"],
        "data_last_date": brief["data_last_date"],
        "run_dir": brief["run_context"]["run_dir"],
        "source_run_dir": brief["run_context"]["run_dir"],
        "risk_level_next_month": (brief.get("operational_signal") or {}).get("risk_level_next_month"),
        "operational_state": (brief.get("operational_signal") or {}).get("operational_state"),
        "confidence_score": (brief.get("operational_signal") or {}).get("confidence_score"),
        "operational_brief_path": str(out_path),
    }
    latest_payload = _sanitize_json_value(latest_payload)
    latest_path.write_text(
        json.dumps(latest_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "data_last_date": brief["data_last_date"],
                "operational_state": brief["operational_signal"]["operational_state"],
                "confidence_score": brief["operational_signal"]["confidence_score"],
                "out_path": str(out_path),
                "latest_pointer": str(latest_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
