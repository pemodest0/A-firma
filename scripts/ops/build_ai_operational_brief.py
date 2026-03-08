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
DOMAIN_EVENT_CATALOGS = {
    "finance": ROOT / "config" / "event_catalog_finance_macro.json",
    "energy": ROOT / "config" / "event_catalog_energy_br.json",
    "agro": ROOT / "config" / "event_catalog_agro_br.json",
}


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


def _cum_return_from_series(values: pd.Series) -> float:
    s = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if s.empty:
        return float("nan")
    return float(np.prod(1.0 + s.to_numpy(dtype=float)) - 1.0)


def _mdd_from_series(values: pd.Series) -> float:
    s = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if s.empty:
        return float("nan")
    eq = np.cumprod(1.0 + s.to_numpy(dtype=float))
    peak = np.maximum.accumulate(eq)
    dd = eq / np.where(peak == 0.0, np.nan, peak) - 1.0
    dd = dd[np.isfinite(dd)]
    return float(np.min(dd)) if dd.size else float("nan")


def _load_latest_systematic_snapshot() -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": "missing",
        "run_dir": "",
        "summary_path": "",
        "simulation_summary_path": "",
        "monthly_path": "",
        "summary": {},
        "simulation_summary": {},
        "monthly_stats": {},
    }
    root = ROOT / "results" / "portfolio_sim"
    if not root.exists():
        return out

    runs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.endswith("_systematic_yearly")], key=lambda p: p.name, reverse=True)
    for run in runs:
        summary_path = run / "systematic_summary.json"
        sim_path = run / "simulation_summary.json"
        monthly_path = run / "monthly_systematic_eval.csv"
        if not summary_path.exists():
            continue
        summary = _read_json(summary_path)
        if not summary:
            continue
        simulation_summary = _read_json(sim_path) if sim_path.exists() else {}
        monthly_stats: dict[str, Any] = {}
        if monthly_path.exists():
            try:
                d = pd.read_csv(monthly_path)
                if not d.empty:
                    d["ret"] = pd.to_numeric(d.get("ret"), errors="coerce")
                    d["eqw_ret"] = pd.to_numeric(d.get("eqw_ret"), errors="coerce")
                    d["risk_budget"] = pd.to_numeric(d.get("risk_budget"), errors="coerce")
                    d["alpha"] = d["ret"] - d["eqw_ret"]
                    tail = d.tail(6).copy()
                    monthly_stats = {
                        "rows_total": int(d.shape[0]),
                        "rows_tail": int(tail.shape[0]),
                        "last_ym": str(d.iloc[-1].get("ym", "")),
                        "alpha_mean_6m": _safe_float(tail["alpha"].mean()),
                        "alpha_positive_rate_6m": _safe_float((tail["alpha"] > 0).mean()),
                        "alpha_total_6m": _cum_return_from_series(tail["alpha"]),
                        "strategy_total_6m": _cum_return_from_series(tail["ret"]),
                        "eqw_total_6m": _cum_return_from_series(tail["eqw_ret"]),
                        "strategy_mdd_6m": _mdd_from_series(tail["ret"]),
                        "avg_risk_budget_6m": _safe_float(tail["risk_budget"].mean()),
                    }
            except Exception:
                monthly_stats = {}

        out = {
            "status": "ok",
            "run_dir": str(run),
            "summary_path": str(summary_path),
            "simulation_summary_path": str(sim_path) if sim_path.exists() else "",
            "monthly_path": str(monthly_path) if monthly_path.exists() else "",
            "summary": summary,
            "simulation_summary": simulation_summary,
            "monthly_stats": monthly_stats,
        }
        return out
    return out


def _build_dynamic_allocation_policy(
    *,
    risk_level: str,
    confidence_score: float,
    domain_gate: dict[str, Any],
    systematic_snapshot: dict[str, Any],
) -> dict[str, Any]:
    rl = str(risk_level or "").strip().lower()
    if rl in {"alto", "high", "stress"}:
        base_exposure = 0.35
    elif rl in {"moderado", "medium", "transition"}:
        base_exposure = 0.55
    else:
        base_exposure = 0.72

    summary = (systematic_snapshot.get("summary") or {}) if isinstance(systematic_snapshot, dict) else {}
    sim_summary = (systematic_snapshot.get("simulation_summary") or {}) if isinstance(systematic_snapshot, dict) else {}
    sim_best = (sim_summary.get("best_metrics") or {}) if isinstance(sim_summary, dict) else {}
    monthly_stats = (systematic_snapshot.get("monthly_stats") or {}) if isinstance(systematic_snapshot, dict) else {}

    prob_positive = _safe_float(summary.get("monthly_alpha_prob_positive_vs_eqw"))
    worth_rate = _safe_float(summary.get("worth_it_rate_vs_eqw"))
    alpha_recent6 = _safe_float(sim_best.get("full_alpha_recent6"))
    alpha_total_6m = _safe_float(monthly_stats.get("alpha_total_6m"))
    strategy_max_drop = _safe_float(summary.get("strategy_max_drop"))
    conf = _safe_float(confidence_score)
    all_domains_publishable = bool((domain_gate.get("all_domains_publishable") if isinstance(domain_gate, dict) else False))

    adj = 0.0
    rationale: list[str] = []

    if np.isfinite(prob_positive):
        if prob_positive >= 0.60:
            adj += 0.05
            rationale.append("probabilidade mensal positiva acima de 60%: aumentar exposicao")
        elif prob_positive <= 0.52:
            adj -= 0.06
            rationale.append("probabilidade mensal positiva fraca: reduzir exposicao")

    if np.isfinite(worth_rate):
        if worth_rate >= 0.70:
            adj += 0.05
            rationale.append("taxa de anos vantajosos alta: aumentar exposicao")
        elif worth_rate <= 0.50:
            adj -= 0.05
            rationale.append("taxa de anos vantajosos baixa: reduzir exposicao")

    if np.isfinite(alpha_recent6):
        if alpha_recent6 >= 0.004:
            adj += 0.06
            rationale.append("vantagem recente (6m) positiva: aumentar exposicao")
        elif alpha_recent6 <= -0.002:
            adj -= 0.08
            rationale.append("vantagem recente (6m) negativa: reduzir exposicao")

    if np.isfinite(alpha_total_6m):
        if alpha_total_6m >= 0.03:
            adj += 0.04
            rationale.append("ganho acumulado recente acima do baseline: aumentar exposicao")
        elif alpha_total_6m <= 0.0:
            adj -= 0.04
            rationale.append("ganho acumulado recente nao confirmou: reduzir exposicao")

    if np.isfinite(conf):
        if conf >= 0.80:
            adj += 0.03
            rationale.append("confianca operacional alta: ampliar faixa de risco")
        elif conf < 0.60:
            adj -= 0.04
            rationale.append("confianca operacional baixa: reduzir faixa de risco")

    if not all_domains_publishable:
        adj -= 0.06
        rationale.append("nem todos os dominios estao publicaveis: reduzir exposicao")

    if np.isfinite(strategy_max_drop) and strategy_max_drop <= -0.35:
        adj -= 0.06
        rationale.append("queda historica profunda no sistematico: impor teto defensivo")

    target = float(np.clip(base_exposure + adj, 0.15, 0.95))
    hard_max = 0.95
    if rl in {"alto", "high", "stress"}:
        hard_max = min(hard_max, 0.45)
    elif rl in {"moderado", "medium", "transition"}:
        hard_max = min(hard_max, 0.75)
    if np.isfinite(strategy_max_drop):
        if strategy_max_drop <= -0.45:
            hard_max = min(hard_max, 0.55)
        elif strategy_max_drop <= -0.35:
            hard_max = min(hard_max, 0.70)
    if not all_domains_publishable:
        hard_max = min(hard_max, 0.65)

    target = min(target, hard_max)
    band = 0.12 if target >= 0.70 else 0.10
    range_min = float(max(0.10, target - band))
    range_max = float(min(hard_max, target + band))

    if target >= 0.75:
        mode = "expansivo"
    elif target >= 0.55:
        mode = "equilibrado"
    else:
        mode = "defensivo"

    should_increase_with_profit = bool(
        (np.isfinite(alpha_recent6) and alpha_recent6 > 0.0)
        or (np.isfinite(alpha_total_6m) and alpha_total_6m > 0.0)
    )

    return {
        "status": "ok",
        "update_frequency": "daily",
        "mode": mode,
        "target_exposure": float(target),
        "range_min": range_min,
        "range_max": range_max,
        "hard_cap_max": float(hard_max),
        "hard_cap_min": 0.10,
        "profit_reinforcement_enabled": should_increase_with_profit,
        "rationale": rationale[:8],
        "signals": {
            "risk_level_next_month": risk_level,
            "confidence_score": conf if np.isfinite(conf) else None,
            "monthly_alpha_prob_positive_vs_eqw": prob_positive if np.isfinite(prob_positive) else None,
            "worth_it_rate_vs_eqw": worth_rate if np.isfinite(worth_rate) else None,
            "alpha_recent6": alpha_recent6 if np.isfinite(alpha_recent6) else None,
            "alpha_total_6m": alpha_total_6m if np.isfinite(alpha_total_6m) else None,
            "strategy_max_drop": strategy_max_drop if np.isfinite(strategy_max_drop) else None,
            "all_domains_publishable": all_domains_publishable,
        },
        "source": {
            "systematic_run_dir": str(systematic_snapshot.get("run_dir", "")),
            "systematic_summary_json": str(systematic_snapshot.get("summary_path", "")),
            "systematic_simulation_summary_json": str(systematic_snapshot.get("simulation_summary_path", "")),
            "systematic_monthly_csv": str(systematic_snapshot.get("monthly_path", "")),
        },
    }


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


def _latest_by_glob(pattern: str) -> Path | None:
    candidates = sorted(ROOT.glob(pattern), key=lambda p: str(p), reverse=True)
    for p in candidates:
        if p.exists():
            return p
    return None


def _extract_temporal_validation_stats(payload: dict[str, Any]) -> dict[str, Any]:
    blocks = payload.get("blocks", []) if isinstance(payload, dict) else []
    if not isinstance(blocks, list) or (not blocks):
        return {"status": "missing"}
    rows = pd.DataFrame(blocks)
    if rows.empty:
        return {"status": "empty"}

    def _m(col: str) -> float:
        s = pd.to_numeric(rows.get(col), errors="coerce")
        s = s[np.isfinite(s)]
        return float(s.mean()) if s.shape[0] > 0 else float("nan")

    return {
        "status": "ok",
        "blocks": int(rows.shape[0]),
        "precision_mean": _m("precision"),
        "recall_mean": _m("recall"),
        "f1_mean": _m("f1"),
        "alert_rate_mean": _m("alert_rate"),
        "lift_precision_vs_random_mean": _m("lift_vs_random_precision"),
        "lift_f1_vs_random_mean": _m("lift_vs_random_f1"),
    }


def _load_event_catalog_rows(path: Path) -> list[dict[str, Any]]:
    data = _read_json(path)
    rows = data.get("events", []) if isinstance(data, dict) else []
    out: list[dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        dt = pd.to_datetime(r.get("date"), errors="coerce")
        if pd.isna(dt):
            continue
        out.append(
            {
                "id": str(r.get("id", "")).strip(),
                "date": str(dt.date()),
                "title": str(r.get("title", "")).strip(),
                "type": str(r.get("type", "")).strip(),
                "source": str(r.get("source", "")).strip(),
                "_dt": dt.normalize(),
            }
        )
    out = sorted(out, key=lambda x: str(x.get("date", "")))
    return out


def _build_macro_event_context(data_last_date: str) -> dict[str, Any]:
    out: dict[str, Any] = {"status": "ok", "as_of_date": str(data_last_date or ""), "domains": {}}
    as_of = pd.to_datetime(data_last_date, errors="coerce")
    as_of_norm = as_of.normalize() if not pd.isna(as_of) else None

    for domain, path in DOMAIN_EVENT_CATALOGS.items():
        events = _load_event_catalog_rows(path)
        latest_past: dict[str, Any] | None = None
        next_event: dict[str, Any] | None = None
        recent_24m: list[dict[str, Any]] = []
        if as_of_norm is not None:
            for ev in events:
                dt = ev.get("_dt")
                if dt is None:
                    continue
                if dt <= as_of_norm:
                    latest_past = ev
                    delta_days = int((as_of_norm - dt).days)
                    if 0 <= delta_days <= 730:
                        recent_24m.append(ev)
                elif next_event is None:
                    next_event = ev
        clean_recent = [{k: v for k, v in ev.items() if k != "_dt"} for ev in recent_24m[-5:]]
        latest_clean = {k: v for k, v in latest_past.items() if k != "_dt"} if isinstance(latest_past, dict) else {}
        next_clean = {k: v for k, v in next_event.items() if k != "_dt"} if isinstance(next_event, dict) else {}
        out["domains"][domain] = {
            "catalog_path": str(path),
            "catalog_exists": bool(path.exists()),
            "events_total": int(len(events)),
            "events_recent_24m": int(len(recent_24m)),
            "latest_event": latest_clean,
            "next_event": next_clean,
            "recent_events_top5": clean_recent,
        }
    return out


def _load_domain_snapshot(domain: str) -> dict[str, Any]:
    slug = str(domain).strip().lower()
    release_path = ROOT / "results" / f"{slug}_br" / f"latest_release_{slug}_br.json"
    release = _read_json(release_path)
    latest_dir = Path(str(release.get("latest_dir", "")).strip()) if release else Path()
    if (not latest_dir.exists()) and (ROOT / "results" / f"{slug}_br" / "latest").exists():
        latest_dir = ROOT / "results" / f"{slug}_br" / "latest"

    evidence_path = latest_dir / f"historical_structure_summary_{slug}_br.json" if latest_dir else Path()
    evidence = _read_json(evidence_path) if evidence_path.exists() else {}

    tuning_path = _latest_by_glob(f"results/{slug}_br/latest/threshold_tuning_*/threshold_sweep_recommendation.json")
    tuning = _read_json(tuning_path) if tuning_path is not None else {}

    temporal_path = _latest_by_glob(f"results/{slug}_br/latest/temporal_validation_*/temporal_validation_summary.json")
    temporal = _read_json(temporal_path) if temporal_path is not None else {}
    temporal_stats = _extract_temporal_validation_stats(temporal)

    epistemic_path = _resolve_path((((release.get("validation") or {}).get("epistemic") or {}).get("summary_json")))
    if epistemic_path is None or (not epistemic_path.exists()):
        fallback = latest_dir / "epistemic_global" / "epistemic_diagnostics_summary.json" if latest_dir else Path()
        epistemic_path = fallback if fallback.exists() else None
    epistemic = _read_json(epistemic_path) if epistemic_path is not None else {}

    evidence_obj = evidence.get("evidence", {}) if isinstance(evidence, dict) else {}
    best_tuning = tuning.get("best", {}) if isinstance(tuning, dict) else {}
    schema_all_ok = bool(((release.get("schema_checks") or {}).get("all_ok")) if isinstance(release, dict) else False)

    return {
        "domain": slug,
        "release_path": str(release_path),
        "latest_dir": str(latest_dir) if latest_dir else "",
        "status": str(release.get("status", "missing")) if isinstance(release, dict) else "missing",
        "schema_all_ok": schema_all_ok,
        "run_dir": str(release.get("run_dir", "")) if isinstance(release, dict) else "",
        "data_last_date": str(evidence.get("last_date", "")) if isinstance(evidence, dict) else "",
        "pre_signal_rate": _safe_float(evidence_obj.get("pre_signal_rate")),
        "pre_signal_count": _safe_float(evidence_obj.get("pre_signal_count")),
        "events_valid": _safe_float(evidence_obj.get("events_valid")),
        "selected_thresholds": {
            "score_z_threshold": _safe_float(best_tuning.get("score_z_threshold")),
            "phi_z_threshold": _safe_float(best_tuning.get("phi_z_threshold")),
            "deff_z_threshold": _safe_float(best_tuning.get("deff_z_threshold")),
        },
        "temporal_validation": temporal_stats,
        "epistemic_summary_status": str(epistemic.get("status", "missing")) if isinstance(epistemic, dict) else "missing",
        "artifacts": {
            "release_json": str(release_path),
            "evidence_json": str(evidence_path) if evidence_path else "",
            "tuning_json": str(tuning_path) if tuning_path is not None else "",
            "temporal_validation_json": str(temporal_path) if temporal_path is not None else "",
            "epistemic_summary_json": str(epistemic_path) if epistemic_path is not None else "",
        },
    }


def _load_finance_ready_snapshot() -> dict[str, Any]:
    latest_path = ROOT / "results" / "ops" / "finance_product_ready" / "latest_finance_product_ready.json"
    latest = _read_json(latest_path)
    report_path = _resolve_path(latest.get("finance_product_ready_json"))
    report = _read_json(report_path) if report_path is not None else {}
    return {
        "latest_json": str(latest_path),
        "report_json": str(report_path) if report_path is not None else "",
        "overall_readiness": str(report.get("overall_readiness", latest.get("overall_readiness", "missing"))),
        "data_last_date": str(report.get("data_last_date", latest.get("data_last_date", ""))),
        "selected_alert_budget": _safe_float(report.get("selected_alert_budget")),
        "regime_horizon_f1_mean": _safe_float(report.get("regime_horizon_f1_mean")),
        "status": str(report.get("status", latest.get("status", "missing"))),
        "warnings": report.get("warnings", []),
    }


def _build_domain_gate(*, finance: dict[str, Any], energy: dict[str, Any], agro: dict[str, Any], freshness: dict[str, Any]) -> dict[str, Any]:
    finance_ok = str(finance.get("overall_readiness", "")).lower() in {"pass", "warn"} and str(freshness.get("status", "")) != "stale"
    energy_ok = (
        str(energy.get("status", "")).lower() == "ok"
        and bool(energy.get("schema_all_ok"))
        and np.isfinite(_safe_float(energy.get("pre_signal_rate")))
    )
    agro_ok = (
        str(agro.get("status", "")).lower() == "ok"
        and bool(agro.get("schema_all_ok"))
        and np.isfinite(_safe_float(agro.get("pre_signal_rate")))
    )

    gates = {
        "finance": {
            "publishable": bool(finance_ok),
            "reason": "ok" if finance_ok else "readiness_or_freshness_failed",
        },
        "energy": {
            "publishable": bool(energy_ok),
            "reason": "ok" if energy_ok else "schema_or_evidence_failed",
        },
        "agro": {
            "publishable": bool(agro_ok),
            "reason": "ok" if agro_ok else "schema_or_evidence_failed",
        },
    }
    gates["all_domains_publishable"] = bool(finance_ok and energy_ok and agro_ok)
    return gates


def _build_insights(
    *,
    summary_hist: dict[str, Any],
    horizon_winners: dict[str, Any],
    gt_best: dict[str, Any],
    macro_event_context: dict[str, Any],
) -> list[dict[str, Any]]:
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
    macro_domains = (macro_event_context.get("domains") or {}) if isinstance(macro_event_context, dict) else {}
    for domain in ["finance", "energy", "agro"]:
        d = macro_domains.get(domain, {})
        if not isinstance(d, dict):
            continue
        latest_event = d.get("latest_event", {}) if isinstance(d.get("latest_event"), dict) else {}
        if not latest_event:
            continue
        insights.append(
            {
                "id": f"macro_context_{domain}",
                "level": "info",
                "message": f"contexto macro {domain}: ultimo evento catalogado {latest_event.get('date')} ({latest_event.get('title')})",
                "evidence": {
                    "domain": domain,
                    "events_total": d.get("events_total"),
                    "events_recent_24m": d.get("events_recent_24m"),
                    "latest_event": latest_event,
                },
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

    macro_event_context = _build_macro_event_context(data_last_date)
    insights = _build_insights(
        summary_hist=hist_summary,
        horizon_winners=horizon_winners,
        gt_best=gt_best,
        macro_event_context=macro_event_context,
    )

    finance_snapshot = _load_finance_ready_snapshot()
    energy_snapshot = _load_domain_snapshot("energy")
    agro_snapshot = _load_domain_snapshot("agro")
    domain_gate = _build_domain_gate(
        finance=finance_snapshot,
        energy=energy_snapshot,
        agro=agro_snapshot,
        freshness=freshness,
    )

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
    systematic_snapshot = _load_latest_systematic_snapshot()
    allocation_policy = _build_dynamic_allocation_policy(
        risk_level=risk_level,
        confidence_score=confidence_score,
        domain_gate=domain_gate,
        systematic_snapshot=systematic_snapshot,
    )

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
            "systematic_summary_json": str(systematic_snapshot.get("summary_path", "")),
            "systematic_simulation_summary_json": str(systematic_snapshot.get("simulation_summary_path", "")),
            "systematic_monthly_csv": str(systematic_snapshot.get("monthly_path", "")),
        },
        "operational_signal": {
            "risk_level_next_month": risk_level,
            "operational_state": action["operational_state"],
            "action_hint": action["action_hint"],
            "priority": action["priority"],
            "confidence_score": confidence_score,
            "target_exposure": allocation_policy.get("target_exposure"),
            "target_exposure_min": allocation_policy.get("range_min"),
            "target_exposure_max": allocation_policy.get("range_max"),
            "allocation_mode": allocation_policy.get("mode"),
        },
        "model_evidence": {
            "ground_truth_best": gt_best,
            "horizon_winners_global": horizon_winners,
            "historical_verdict_counts_expanding": (hist_summary.get("verdict_counts_expanding") or {}),
            "stress_prealert_top": (hist_summary.get("stress_prealert_summary_top10") or [])[:5],
            "systematic_snapshot": {
                "status": systematic_snapshot.get("status"),
                "run_dir": systematic_snapshot.get("run_dir"),
                "summary_metrics": {
                    "monthly_alpha_prob_positive_vs_eqw": _safe_float((systematic_snapshot.get("summary") or {}).get("monthly_alpha_prob_positive_vs_eqw")),
                    "worth_it_rate_vs_eqw": _safe_float((systematic_snapshot.get("summary") or {}).get("worth_it_rate_vs_eqw")),
                    "strategy_max_drop": _safe_float((systematic_snapshot.get("summary") or {}).get("strategy_max_drop")),
                },
            },
        },
        "state_snapshot": {
            "next_month_indication": next_month,
            "top_sectors_global_mode": rankings_latest.get("top_sectors_global_mode", []),
            "top_assets_global_mode": rankings_latest.get("top_assets_global_mode", []),
            "sector_global_overlap": rankings_latest.get("sector_global_overlap", []),
            "global_state": rankings_latest.get("global_state", {}),
        },
        "allocation_policy": allocation_policy,
        "macro_event_context": macro_event_context,
        "domain_snapshots": {
            "finance": finance_snapshot,
            "energy": energy_snapshot,
            "agro": agro_snapshot,
        },
        "domain_gate": domain_gate,
        "insights": insights,
        "guardrails": [
            "nao afirmar direcao de preco; usar linguagem de estado estrutural",
            "citar sempre data_last_date e run_id/arquivo fonte",
            "se data_last_date estiver stale, reduzir confianca e avisar explicitamente",
            "para operacao, usar regime/horizon como gatilho e drawdown como confirmacao secundaria",
            "se domain_gate.all_domains_publishable=false, explicitar limite de cobertura por dominio",
        ],
    }
    brief = _sanitize_json_value(brief)

    run_id = _run_id()
    out_path = outdir / f"operational_brief_{run_id}.json"
    out_path.write_text(json.dumps(brief, indent=2, ensure_ascii=False), encoding="utf-8")

    exposure_policy = {
        "status": "ok",
        "generated_at_utc": brief["generated_at_utc"],
        "data_last_date": brief["data_last_date"],
        "risk_level_next_month": (brief.get("operational_signal") or {}).get("risk_level_next_month"),
        "allocation_policy": brief.get("allocation_policy"),
        "run_context": brief.get("run_context"),
    }
    exposure_policy = _sanitize_json_value(exposure_policy)
    exposure_path = outdir / f"exposure_policy_{run_id}.json"
    exposure_path.write_text(json.dumps(exposure_policy, indent=2, ensure_ascii=False), encoding="utf-8")

    exposure_latest_path = outdir / "latest_exposure_policy.json"
    exposure_latest_payload = {
        "status": "ok",
        "generated_at_utc": brief["generated_at_utc"],
        "data_last_date": brief["data_last_date"],
        "allocation_mode": (brief.get("allocation_policy") or {}).get("mode"),
        "target_exposure": (brief.get("allocation_policy") or {}).get("target_exposure"),
        "range_min": (brief.get("allocation_policy") or {}).get("range_min"),
        "range_max": (brief.get("allocation_policy") or {}).get("range_max"),
        "exposure_policy_path": str(exposure_path),
    }
    exposure_latest_payload = _sanitize_json_value(exposure_latest_payload)
    exposure_latest_path.write_text(
        json.dumps(exposure_latest_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    assistant_ground_truth = {
        "status": "ok",
        "schema_version": "assistant_ground_truth_v2",
        "generated_at_utc": brief["generated_at_utc"],
        "assistant_name": "Eigen Engine Assistant",
        "data_last_date": brief["data_last_date"],
        "freshness": brief["freshness"],
        "operational_signal": brief["operational_signal"],
        "domain_gate": brief["domain_gate"],
        "domains": brief["domain_snapshots"],
        "sources": brief["sources"],
    }
    assistant_ground_truth = _sanitize_json_value(assistant_ground_truth)
    assistant_pack_path = outdir / f"assistant_ground_truth_{run_id}.json"
    assistant_pack_path.write_text(
        json.dumps(assistant_ground_truth, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

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
        "allocation_mode": (brief.get("allocation_policy") or {}).get("mode"),
        "target_exposure": (brief.get("allocation_policy") or {}).get("target_exposure"),
        "target_exposure_min": (brief.get("allocation_policy") or {}).get("range_min"),
        "target_exposure_max": (brief.get("allocation_policy") or {}).get("range_max"),
        "operational_brief_path": str(out_path),
        "assistant_ground_truth_path": str(assistant_pack_path),
        "exposure_policy_path": str(exposure_path),
        "all_domains_publishable": bool((brief.get("domain_gate") or {}).get("all_domains_publishable")),
    }
    latest_payload = _sanitize_json_value(latest_payload)
    latest_path.write_text(
        json.dumps(latest_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    assistant_latest_path = outdir / "latest_assistant_ground_truth.json"
    assistant_latest_payload = {
        "status": "ok",
        "generated_at_utc": brief["generated_at_utc"],
        "data_last_date": brief["data_last_date"],
        "assistant_ground_truth_path": str(assistant_pack_path),
    }
    assistant_latest_payload = _sanitize_json_value(assistant_latest_payload)
    assistant_latest_path.write_text(
        json.dumps(assistant_latest_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "data_last_date": brief["data_last_date"],
                "operational_state": brief["operational_signal"]["operational_state"],
                "confidence_score": brief["operational_signal"]["confidence_score"],
                "target_exposure": brief["operational_signal"].get("target_exposure"),
                "out_path": str(out_path),
                "latest_pointer": str(latest_path),
                "assistant_ground_truth_path": str(assistant_pack_path),
                "assistant_latest_pointer": str(assistant_latest_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
