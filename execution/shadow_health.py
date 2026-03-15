from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


LEGACY_SHADOWS: tuple[dict[str, str], ...] = (
    {
        "shadow_key": "profit_shadow",
        "summary_path": "results/ops/profit_shadow/latest_summary.json",
        "launch_agent": "com.assyntrax.profit-shadow.plist",
    },
    {
        "shadow_key": "profit_shadow_target_800_attack",
        "summary_path": "results/ops/profit_shadow_target_800_attack/latest_summary.json",
        "launch_agent": "com.assyntrax.profit-shadow-target800.plist",
    },
    {
        "shadow_key": "profit_shadow_target_800_attack_ensemble",
        "summary_path": "results/ops/profit_shadow_target_800_attack_ensemble/latest_summary.json",
        "launch_agent": "com.assyntrax.profit-shadow-target800.plist",
    },
    {
        "shadow_key": "invest_shadow",
        "summary_path": "results/ops/invest_shadow/latest_summary.json",
        "launch_agent": "com.assyntrax.investment-shadow.plist",
    },
)


def _read_json_dict(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_dt(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _best_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    best = payload.get("best_by_profit", {}) if isinstance(payload.get("best_by_profit"), dict) else {}
    latest_signal = best.get("latest_signal", {}) if isinstance(best.get("latest_signal"), dict) else {}
    return {
        "profile": best.get("profile"),
        "daily_total_return": _safe_float(best.get("daily_total_return")),
        "daily_ann_return": _safe_float(best.get("daily_ann_return")),
        "daily_sharpe": _safe_float(best.get("daily_sharpe")),
        "daily_max_drawdown": _safe_float(best.get("daily_max_drawdown")),
        "daily_edge_vs_benchmark": _safe_float(best.get("daily_edge_vs_benchmark")),
        "worth_it_rate_vs_eqw": _safe_float(best.get("systematic_worth_it_rate_vs_eqw")),
        "monthly_alpha_prob_positive_vs_eqw": _safe_float(best.get("systematic_monthly_alpha_prob_positive_vs_eqw")),
        "latest_risk_bucket": latest_signal.get("risk_bucket"),
        "latest_assets": latest_signal.get("executed_assets"),
    }


def _legacy_recommendation(metrics: dict[str, Any], stale_days: float, stale_threshold_days: float) -> tuple[str, str]:
    edge = metrics.get("daily_edge_vs_benchmark")
    alpha_prob = metrics.get("monthly_alpha_prob_positive_vs_eqw")
    if stale_days >= stale_threshold_days:
        if edge is not None and edge > 0.0 and alpha_prob is not None and alpha_prob >= 0.5:
            return ("archive_only", "Historicamente interessante, mas stale para operacao diaria.")
        return ("retire", "Stale e sem ganho operacional claro para a arquitetura atual.")
    if edge is not None and edge <= 0.0:
        return ("retire", "Nao mostrou edge suficiente contra benchmark.")
    if alpha_prob is not None and alpha_prob < 0.5:
        return ("retire", "Alpha mensal positivo insuficiente para justificar shadow ativo.")
    return ("keep_baseline", "Ainda serve como baseline de comparacao.")


def build_shadow_health_summary(
    repo_root: str | Path,
    *,
    watchlist_path: str | Path | None = None,
    launch_agents_dir: str | Path | None = None,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    watchlist_file = Path(watchlist_path).resolve() if watchlist_path else root / "config/live_shadow_watchlist.json"
    watchlist = _read_json_dict(watchlist_file)
    agents_dir = Path(launch_agents_dir).expanduser().resolve() if launch_agents_dir else (Path.home() / "Library/LaunchAgents").resolve()
    now = now_utc or datetime.now(UTC)
    stale_threshold = float(((watchlist.get("policy") or {}).get("retire_legacy_shadows_when_stale_days_gte")) or 3.0)

    legacy_status: list[dict[str, Any]] = []
    for item in LEGACY_SHADOWS:
        summary_path = root / item["summary_path"]
        if not summary_path.exists():
            legacy_status.append(
                {
                    "shadow_key": item["shadow_key"],
                    "summary_path": str(summary_path),
                    "exists": False,
                    "launch_agent_installed": (agents_dir / item["launch_agent"]).exists(),
                    "recommendation": "retire",
                    "reason": "Summary inexistente.",
                }
            )
            continue
        payload = _read_json_dict(summary_path)
        generated_at = _parse_dt(payload.get("generated_at_utc"))
        stale_days = None
        if generated_at:
            stale_days = (now - generated_at).total_seconds() / 86400.0
        metrics = _best_metrics(payload)
        recommendation, reason = _legacy_recommendation(metrics, stale_days or 9999.0, stale_threshold)
        legacy_status.append(
            {
                "shadow_key": item["shadow_key"],
                "shadow_name": payload.get("shadow_name"),
                "summary_path": str(summary_path),
                "exists": True,
                "status": payload.get("status"),
                "run_id": payload.get("run_id"),
                "generated_at_utc": payload.get("generated_at_utc"),
                "stale_days": stale_days,
                "launch_agent_installed": (agents_dir / item["launch_agent"]).exists(),
                "metrics": metrics,
                "recommendation": recommendation,
                "reason": reason,
            }
        )

    active_watchlist = watchlist.get("active_watchlist", []) if isinstance(watchlist.get("active_watchlist"), list) else []
    recommended_active = [item for item in active_watchlist if isinstance(item, dict) and str(item.get("status") or "").strip().lower() == "shadow"]
    recommendation_counts: dict[str, int] = {}
    for item in legacy_status:
        key = str(item.get("recommendation") or "unknown")
        recommendation_counts[key] = recommendation_counts.get(key, 0) + 1

    return {
        "status": "ok",
        "generated_at_utc": now.isoformat(),
        "watchlist_path": str(watchlist_file),
        "launch_agents_dir": str(agents_dir),
        "legacy_shadows": legacy_status,
        "recommended_active_watchlist": recommended_active,
        "summary": {
            "legacy_count": len(legacy_status),
            "recommended_active_count": len(recommended_active),
            "recommendation_counts": recommendation_counts,
            "all_legacy_stale": all((item.get("stale_days") or 0.0) >= stale_threshold for item in legacy_status if item.get("exists")),
        },
    }


def write_shadow_health_outputs(
    repo_root: str | Path,
    *,
    outdir: str | Path,
    watchlist_path: str | Path | None = None,
    launch_agents_dir: str | Path | None = None,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    out_path = Path(outdir).resolve()
    summary = build_shadow_health_summary(
        root,
        watchlist_path=watchlist_path,
        launch_agents_dir=launch_agents_dir,
        now_utc=now_utc,
    )
    _write_json(out_path / "summary.json", summary)
    _write_json(root / "results/ops/shadow_health/latest_summary.json", summary)
    return summary
