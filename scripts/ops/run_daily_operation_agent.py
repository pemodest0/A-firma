#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.portfolio import decide_attack_vs_protection  # noqa: E402
from scripts.bench.validation.run_profit_marketmode_criticality_suite import (  # noqa: E402
    build_official_mode_allocations,
)


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def _latest_weights(weights: pd.DataFrame) -> dict[str, float]:
    if weights.empty:
        return {"crypto": 0.0, "equity": 0.0, "cash": 1.0}
    row = weights.tail(1).iloc[0]
    return {
        "crypto": float(pd.to_numeric(row.get("crypto", 0.0), errors="coerce") or 0.0),
        "equity": float(pd.to_numeric(row.get("equity", 0.0), errors="coerce") or 0.0),
        "cash": float(pd.to_numeric(row.get("cash", 0.0), errors="coerce") or 0.0),
    }


def _latest_source(source: pd.Series) -> str:
    if source.empty:
        return "cash"
    return str(source.tail(1).iloc[0] or "cash").strip() or "cash"


def _latest_validation_summary(name: str) -> dict[str, Any]:
    root = ROOT / "results" / "validation" / name
    if not root.exists():
        return {}
    runs = sorted([path for path in root.iterdir() if path.is_dir()], reverse=True)
    if not runs:
        return {}
    return _read_json(runs[0] / "summary.json")


def _finance_ready_details() -> dict[str, Any]:
    latest = _read_json(ROOT / "results" / "ops" / "finance_product_ready" / "latest_finance_product_ready.json")
    detail_path = str(latest.get("finance_product_ready_json") or "").strip()
    if not detail_path:
        return {}
    return _read_json(Path(detail_path))


def _normalize_structural_level(finance_ready: dict[str, Any]) -> str:
    raw_risk = str(finance_ready.get("risk_level_next_month") or "").strip().lower()
    if raw_risk in {"baixo", "low"}:
        return "stable"
    if raw_risk in {"medio", "médio", "medium"}:
        return "transition"
    if raw_risk in {"alto", "high"}:
        return "stress"
    raw_state = str(finance_ready.get("operational_state") or "").strip().lower()
    if raw_state in {"monitoramento_normal", "normal", "stable"}:
        return "stable"
    if raw_state in {"atencao", "atenção", "transition"}:
        return "transition"
    if raw_state in {"defesa", "stress", "estresse"}:
        return "stress"
    return "transition"


def _attack_underperform_prob(summary: dict[str, Any], *, horizon: int) -> float | None:
    rows = summary.get("attack_mc_base")
    if not isinstance(rows, list):
        return None
    for row in rows:
        if int(row.get("horizon_days") or 0) == int(horizon):
            return _safe_float(row.get("underperform_prob"))
    return None


def _attack_top3_retention(summary: dict[str, Any]) -> float | None:
    rows = summary.get("attack_retention_worst_nonbase")
    if not isinstance(rows, list):
        return None
    for row in rows:
        if str(row.get("scenario") or "") == "drop_top3_crypto":
            return _safe_float(row.get("total_retention"))
    if rows:
        return _safe_float(rows[0].get("total_retention"))
    return None


def _execution_winner_label(summary: dict[str, Any]) -> str:
    capital_scaling = summary.get("capital_scaling")
    if not isinstance(capital_scaling, dict):
        return ""
    best_small = capital_scaling.get("best_profit_at_small_capital")
    if not isinstance(best_small, dict):
        return ""
    return str(best_small.get("candidate_label") or "").strip()


def _mode_payload(*, label: str, allocation) -> dict[str, Any]:
    result = allocation.bundle.result
    ret_idx = result.net_ret.index if isinstance(result.net_ret, pd.Series) else pd.Index([])
    last_date = str(ret_idx[-1].date()) if len(ret_idx) else ""
    weights = _latest_weights(allocation.weights)
    gross = max(0.0, float(weights.get("crypto", 0.0)) + float(weights.get("equity", 0.0)))
    return {
        "label": str(label),
        "candidate_id": str(result.candidate_id),
        "latest_date": last_date,
        "latest_source": _latest_source(allocation.source),
        "weights": weights,
        "gross_exposure": gross,
        "net_ann_return": _safe_float(result.net_ann_return),
        "net_total_return": _safe_float(result.net_total_return),
        "net_sharpe": _safe_float(result.net_sharpe),
        "net_max_drawdown": _safe_float(result.net_max_drawdown),
        "avg_turnover_daily": _safe_float(result.avg_turnover_daily),
        "notes": str(result.notes or ""),
    }


def _run_step(cmd: list[str], *, timeout_sec: float) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    return {
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "stdout": (proc.stdout or "")[-2000:],
        "stderr": (proc.stderr or "")[-2000:],
        "ok": proc.returncode == 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Agente diario de operacao do motor de lucro.")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--outdir-root", default="results/ops/agents/daily_operation")
    args = ap.parse_args()

    outdir = (ROOT / args.outdir_root / _run_id()).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    official = build_official_mode_allocations(
        prices_dir=(ROOT / args.prices_dir).resolve(),
        crypto_groups=(ROOT / args.crypto_asset_groups).resolve(),
        crypto_meta=(ROOT / args.crypto_asset_metadata).resolve(),
        equity_groups=(ROOT / args.equity_asset_groups).resolve(),
        equity_meta=(ROOT / args.equity_asset_metadata).resolve(),
        benchmark_crypto=str(args.benchmark_crypto),
        benchmark_equity=str(args.benchmark_equity),
    )
    built = official["built"]

    operation = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode_attack": _mode_payload(label="Modo ataque", allocation=official["official_attack"]),
        "mode_main": _mode_payload(label="Modo principal", allocation=official["official_main"]),
        "mode_attack_guard": _mode_payload(label="Modo ataque com guarda", allocation=official["official_attack_guard"]),
        "mode_main_guard": _mode_payload(label="Modo principal com guarda", allocation=official["official_main_guard"]),
        "artifacts": {
            "prices_dir": str((ROOT / args.prices_dir).resolve()),
            "crypto_asset_groups": str((ROOT / args.crypto_asset_groups).resolve()),
            "equity_asset_groups": str((ROOT / args.equity_asset_groups).resolve()),
        },
    }

    finance_ready = _finance_ready_details()
    vigilance = _read_json(ROOT / "results" / "ops" / "agents" / "daily_vigilance" / "latest_summary.json")
    pbo_summary = _latest_validation_summary("profit_pbo_suite")
    resilience_summary = _latest_validation_summary("profit_universe_resilience_suite")
    execution_summary = _latest_validation_summary("profit_execution_resilience_suite")

    mode_confidence = decide_attack_vs_protection(
        structural_risk_level=_normalize_structural_level(finance_ready),
        structural_confidence_score=_safe_float(finance_ready.get("confidence_score")),
        vigilance_status=str(vigilance.get("status") or ""),
        vigilance_alert_count=len(vigilance.get("alerts") or []) if isinstance(vigilance.get("alerts"), list) else 0,
        pbo_verdict=str(pbo_summary.get("overall_verdict") or ""),
        attack_underperform_prob_63=_attack_underperform_prob(resilience_summary, horizon=63),
        attack_top3_retention=_attack_top3_retention(resilience_summary),
        attack_drawdown=_safe_float(operation["mode_attack"].get("net_max_drawdown")),
        protection_drawdown=_safe_float(operation["mode_main_guard"].get("net_max_drawdown")),
        execution_winner=_execution_winner_label(execution_summary),
    )
    operation["mode_confidence"] = mode_confidence.to_dict()
    operation["recommended_live_mode"] = {
        "mode": mode_confidence.recommended_mode,
        "label": "Modo ataque" if mode_confidence.recommended_mode == "ataque" else "Modo principal com guarda",
        "confidence_level": mode_confidence.confidence_level,
        "confidence_score": mode_confidence.confidence_score,
    }
    operation["confidence_notes"] = [
        "O ataque agora combina criticidade estrutural com um freio leve de reorganizacao para evitar euforia e giro desnecessario.",
        "A protecao continua preferivel quando o mercado inteiro aperta junto, a confianca cai e o atrito operacional sobe.",
    ]

    registry_step = _run_step([sys.executable, "scripts/ops/build_profit_research_registry.py"], timeout_sec=1200.0)
    snapshot_step = _run_step([sys.executable, "scripts/ops/build_site_finance_snapshot.py"], timeout_sec=1200.0)
    operation["post_steps"] = {
        "registry": registry_step,
        "site_snapshot": snapshot_step,
    }
    operation["publish_ready"] = bool(registry_step["ok"] and snapshot_step["ok"])

    _write_json(outdir / "summary.json", operation)
    latest_dir = (ROOT / args.outdir_root).resolve()
    _write_json(latest_dir / "latest_summary.json", operation)
    _write_json(latest_dir / "latest_operation.json", operation)
    print(json.dumps(operation, ensure_ascii=False))


if __name__ == "__main__":
    main()
