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

from scripts.ops.agent_guides import attach_agent_guide  # noqa: E402
from scripts.ops.cycle_context import attach_cycle_context, resolve_cycle_run_id, utc_now_iso, utc_run_id  # noqa: E402
from scripts.ops.model_validation_metrics import resolve_live_validation_metrics  # noqa: E402
from engine.portfolio import decide_attack_vs_protection  # noqa: E402
from scripts.bench.validation.run_profit_marketmode_criticality_suite import (  # noqa: E402
    build_official_mode_allocations,
)


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


def _current_code_revision() -> str:
    step = _run_step(["git", "rev-parse", "--short", "HEAD"], timeout_sec=10.0)
    if not step["ok"]:
        return ""
    revision = str(step["stdout"] or "").strip().splitlines()[-1].strip()
    dirty = _run_step(["git", "status", "--short"], timeout_sec=10.0)
    if dirty["ok"] and str(dirty.get("stdout") or "").strip():
        revision = f"{revision}-dirty"
    return revision


def _latest_ingestion_as_of_date() -> str:
    latest = _read_json(ROOT / "results" / "ops" / "agents" / "daily_ingestion" / "latest_summary.json")
    return str(latest.get("max_latest_date") or "").strip()


def _parse_date(text: Any):
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw).date()
    except ValueError:
        return None


def _write_official_structural_regime(*, operation: dict[str, Any], official: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "status": "ok",
        "generated_at_utc": operation.get("generated_at_utc"),
        "cycle_run_id": operation.get("cycle_run_id"),
        "agent_run_id": operation.get("agent_run_id"),
        "as_of_date": ((official.get("official_structural_now") or {}).get("as_of_date") or ""),
        "regime": ((official.get("official_structural_now") or {}).get("regime") or ""),
        "criticality": ((official.get("official_structural_now") or {}).get("criticality")),
        "structural_stress": ((official.get("official_structural_now") or {}).get("structural_stress")),
        "market_mode_share_pct": ((official.get("official_structural_now") or {}).get("market_mode_share_pct")),
        "classification_method": ((official.get("official_structural_now") or {}).get("classification_method") or ""),
        "base_run_dir": str((((official.get("built") or {}).get("context") or {}).get("structural_regime_meta") or {}).get("run_dir") or ""),
        "base_regime_source": str((((official.get("built") or {}).get("context") or {}).get("structural_regime_meta") or {}).get("source") or ""),
    }
    root = ROOT / "results" / "ops" / "official_structural_regime"
    run_id = str(operation.get("agent_run_id") or utc_run_id())
    _write_json(root / run_id / "latest_structural_regime.json", payload)
    _write_json(root / "latest_structural_regime.json", payload)
    return payload


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


def _finance_ready_details() -> dict[str, Any]:
    latest = _read_json(ROOT / "results" / "ops" / "finance_product_ready" / "latest_finance_product_ready.json")
    detail_path = str(latest.get("finance_product_ready_json") or "").strip()
    if not detail_path:
        return {}
    return _read_json(Path(detail_path))


def _latest_validation_summary(name: str) -> dict[str, Any]:
    root = ROOT / "results" / "validation" / name
    if not root.exists():
        return {}
    runs = sorted([path for path in root.iterdir() if path.is_dir()], reverse=True)
    if not runs:
        return {}
    return _read_json(runs[0] / "summary.json")


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


def _posture_payload(payload: dict[str, Any]) -> dict[str, Any]:
    source = str(payload.get("latest_source") or "").strip().lower()
    gross = _safe_float(payload.get("gross_exposure")) or 0.0
    if gross <= 0.05:
        posture = "quase_caixa"
    elif source in {"cash", "protect", "equity25", "equity50"}:
        posture = "defensivo"
    elif source == "attack" and gross >= 0.75:
        posture = "ataque_pleno"
    elif source == "attack":
        posture = "ataque_parcial"
    else:
        posture = "misto"
    return {
        "posture": posture,
        "gross_exposure": gross,
        "latest_source": source or "cash",
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


def _can_reuse_latest_operation(latest: dict[str, Any], *, inputs_as_of_date: str, code_revision: str) -> bool:
    if not latest:
        return False
    if str(latest.get("status") or "").strip().lower() != "ok":
        return False
    prev_inputs = str(latest.get("inputs_as_of_date") or "").strip()
    prev_revision = str(latest.get("code_revision") or "").strip()
    attack = latest.get("mode_attack")
    return bool(prev_inputs and prev_revision and attack and prev_inputs == inputs_as_of_date and prev_revision == code_revision)


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
    ap.add_argument("--cycle-run-id", default="")
    ap.add_argument("--reuse-if-fresh", action="store_true", default=True)
    ap.add_argument("--no-reuse-if-fresh", dest="reuse_if_fresh", action="store_false")
    args = ap.parse_args()
    agent_run_id = utc_run_id()
    cycle_run_id = resolve_cycle_run_id(args.cycle_run_id)

    outdir = (ROOT / args.outdir_root / agent_run_id).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    inputs_as_of_date = _latest_ingestion_as_of_date()
    code_revision = _current_code_revision()
    latest_dir = (ROOT / args.outdir_root).resolve()
    previous_operation = _read_json(latest_dir / "latest_summary.json")
    reused_previous = bool(
        args.reuse_if_fresh and _can_reuse_latest_operation(previous_operation, inputs_as_of_date=inputs_as_of_date, code_revision=code_revision)
    )

    official: dict[str, Any] = {}
    if reused_previous:
        operation = dict(previous_operation)
        operation["status"] = "ok"
        operation["generated_at_utc"] = utc_now_iso()
        operation["reuse_reason"] = "same_inputs_and_code_revision"
        operation["reused_previous_operation"] = True
    else:
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
            "generated_at_utc": utc_now_iso(),
            "mode_attack": _mode_payload(label="Modo ataque", allocation=official["official_attack"]),
            "mode_main": _mode_payload(label="Modo principal", allocation=official["official_main"]),
            "mode_attack_guard": _mode_payload(label="Modo ataque com guarda", allocation=official["official_attack_guard"]),
            "mode_main_guard": _mode_payload(label="Modo principal com guarda", allocation=official["official_main_guard"]),
            "artifacts": {
                "prices_dir": str((ROOT / args.prices_dir).resolve()),
                "crypto_asset_groups": str((ROOT / args.crypto_asset_groups).resolve()),
                "equity_asset_groups": str((ROOT / args.equity_asset_groups).resolve()),
            },
            "reused_previous_operation": False,
        }
        operation["current_posture"] = {
            "mode_attack": _posture_payload(operation["mode_attack"]),
            "mode_main": _posture_payload(operation["mode_main"]),
            "mode_attack_guard": _posture_payload(operation["mode_attack_guard"]),
            "mode_main_guard": _posture_payload(operation["mode_main_guard"]),
        }

    finance_ready = _finance_ready_details()
    vigilance = _read_json(ROOT / "results" / "ops" / "agents" / "daily_vigilance" / "latest_summary.json")
    execution_summary = _latest_validation_summary("profit_execution_resilience_suite")
    validation_metrics = resolve_live_validation_metrics(
        root=ROOT,
        candidate_id=str(operation["mode_attack"]["candidate_id"]),
    )

    mode_confidence = decide_attack_vs_protection(
        structural_risk_level=_normalize_structural_level(finance_ready),
        structural_confidence_score=_safe_float(finance_ready.get("confidence_score")),
        vigilance_status=str(vigilance.get("status") or ""),
        vigilance_alert_count=len(vigilance.get("alerts") or []) if isinstance(vigilance.get("alerts"), list) else 0,
        pbo_verdict=str(validation_metrics.get("pbo_verdict") or ""),
        attack_underperform_prob_63=_safe_float(validation_metrics.get("underperform_prob_63")),
        attack_top3_retention=_safe_float(validation_metrics.get("top3_total_retention")),
        attack_drawdown=_safe_float(operation["mode_attack"].get("net_max_drawdown")),
        protection_drawdown=_safe_float(operation["mode_main_guard"].get("net_max_drawdown")),
        execution_winner=_execution_winner_label(execution_summary),
    )
    operation["mode_confidence"] = mode_confidence.to_dict()
    operation["validation_metrics_source"] = validation_metrics
    if not reused_previous:
        operation["official_structural_regime"] = official.get("official_structural_now") if isinstance(official.get("official_structural_now"), dict) else {}
    operation["recommended_live_mode"] = {
        "mode": mode_confidence.recommended_mode,
        "label": "Modo ataque" if mode_confidence.recommended_mode == "ataque" else "Modo principal com guarda",
        "confidence_level": mode_confidence.confidence_level,
        "confidence_score": mode_confidence.confidence_score,
        "current_posture": operation["current_posture"]["mode_attack" if mode_confidence.recommended_mode == "ataque" else "mode_main_guard"],
    }
    operation["confidence_notes"] = [
        "O ataque agora combina criticidade estrutural com um freio leve de reorganização para evitar euforia e giro desnecessário.",
        "A proteção continua preferível quando o mercado inteiro aperta junto, a confiança cai e o atrito operacional sobe.",
    ]
    operation["inputs_as_of_date"] = inputs_as_of_date
    operation["code_revision"] = code_revision
    operation = attach_agent_guide(
        attach_cycle_context(operation, cycle_run_id=cycle_run_id, agent_run_id=agent_run_id),
        "daily-operation-agent",
    )
    if not reused_previous:
        operation["official_structural_regime_artifact"] = _write_official_structural_regime(operation=operation, official=official)
    else:
        operation["official_structural_regime_artifact"] = previous_operation.get("official_structural_regime_artifact", {})

    _write_json(outdir / "summary.json", operation)
    _write_json(latest_dir / "latest_summary.json", operation)
    _write_json(latest_dir / "latest_operation.json", operation)

    registry_step = _run_step([sys.executable, "scripts/ops/build_profit_research_registry.py"], timeout_sec=1200.0)
    data_quality_step = _run_step([sys.executable, "scripts/ops/run_daily_data_quality_agent.py", "--cycle-run-id", cycle_run_id], timeout_sec=1200.0)
    snapshot_step = _run_step([sys.executable, "scripts/ops/build_site_finance_snapshot.py", "--cycle-run-id", cycle_run_id], timeout_sec=1200.0)
    operation["post_steps"] = {
        "registry": registry_step,
        "data_quality": data_quality_step,
        "site_snapshot": snapshot_step,
    }
    operation["publish_ready"] = bool(registry_step["ok"] and data_quality_step["ok"] and snapshot_step["ok"])

    _write_json(outdir / "summary.json", operation)
    _write_json(latest_dir / "latest_summary.json", operation)
    _write_json(latest_dir / "latest_operation.json", operation)
    print(json.dumps(operation, ensure_ascii=False))


if __name__ == "__main__":
    main()
