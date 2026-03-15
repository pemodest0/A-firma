#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution.live_ops import (  # noqa: E402
    compile_order_tickets,
    load_last_prices,
    load_live_execution_profile,
    load_portfolio_state,
    portfolio_template_payload,
    write_json,
)
from scripts.ops.agent_guides import attach_agent_guide  # noqa: E402
from scripts.ops.cycle_context import attach_cycle_context, resolve_cycle_run_id, utc_now_iso, utc_run_id  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Compila o plano semi-automatico de execucao live.")
    ap.add_argument("--config", default="config/live_execution_profile.json")
    ap.add_argument("--operation-json", default="results/ops/agents/daily_operation/latest_summary.json")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--outdir-root", default="results/ops/execution_live")
    ap.add_argument("--cycle-run-id", default="")
    args = ap.parse_args()

    run_id = utc_run_id()
    cycle_run_id = resolve_cycle_run_id(args.cycle_run_id)
    outdir = (ROOT / args.outdir_root / run_id).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    profile = load_live_execution_profile(ROOT / args.config)
    operation = _read_json((ROOT / args.operation_json).resolve())
    paths = profile.get("paths", {}) if isinstance(profile.get("paths"), dict) else {}
    portfolio_path = (ROOT / str(paths.get("portfolio_state_json") or "data/live_execution/portfolio_state.json")).resolve()

    if not portfolio_path.exists():
        template_path = outdir / "portfolio_state_template.json"
        write_json(template_path, portfolio_template_payload())
        summary = {
            "status": "needs_portfolio_state",
            "generated_at_utc": utc_now_iso(),
            "run_id": run_id,
            "cycle_run_id": cycle_run_id,
            "config_path": str((ROOT / args.config).resolve()),
            "portfolio_state_path": str(portfolio_path),
            "portfolio_template_path": str(template_path),
            "notes": [
                "Crie a carteira real local antes de emitir ordens.",
                "Use config/live_portfolio_state.example.json como modelo inicial.",
            ],
        }
    elif not operation:
        summary = {
            "status": "needs_operation_summary",
            "generated_at_utc": utc_now_iso(),
            "run_id": run_id,
            "cycle_run_id": cycle_run_id,
            "config_path": str((ROOT / args.config).resolve()),
            "portfolio_state_path": str(portfolio_path),
            "notes": ["O agente diario de operacao ainda nao produziu latest_summary.json."],
        }
    else:
        portfolio = load_portfolio_state(portfolio_path)
        tickers = list((profile.get("execution_profile") or {}).get("allowed_tickers", []))
        prices = load_last_prices(ROOT / args.prices_dir, tickers, fx_rates=portfolio.fx_rates)
        compiled = compile_order_tickets(operation, profile, portfolio, prices)
        summary = {
            "status": str(compiled.get("status") or "ok"),
            "generated_at_utc": utc_now_iso(),
            "run_id": run_id,
            "cycle_run_id": cycle_run_id,
            "config_path": str((ROOT / args.config).resolve()),
            "portfolio_state_path": str(portfolio_path),
            "operation_summary_path": str((ROOT / args.operation_json).resolve()),
            "portfolio": portfolio.to_dict(),
            "price_context": prices,
            "selected_mode": compiled.get("selected_mode", {}),
            "target_notional_brl": compiled.get("target_notional_brl", {}),
            "target_notional_after_caps_brl": compiled.get("target_notional_after_caps_brl", {}),
            "turnover_requested_brl": compiled.get("turnover_requested_brl"),
            "turnover_cap_brl": compiled.get("turnover_cap_brl"),
            "turnover_scale_applied": compiled.get("turnover_scale_applied"),
            "tickets": compiled.get("tickets", []),
            "blocked": compiled.get("blocked", []),
            "notes": compiled.get("notes", []),
            "manual_steps": [
                "Revisar o selected_mode e confirmar se ele faz sentido para o contexto do dia.",
                "Checar se a carteira real e o saldo em caixa batem com o portfolio_state.json.",
                "Executar apenas os tickets com notional_brl acima do minimo e sem blocked.",
                "Depois da execucao, preencher execution_report.json e rodar a reconciliacao.",
            ],
        }

    summary = attach_agent_guide(
        attach_cycle_context(summary, cycle_run_id=cycle_run_id, agent_run_id=run_id),
        "daily-operation-agent",
    )
    write_json(outdir / "execution_plan.json", summary)
    write_json((ROOT / args.outdir_root / "latest_execution_plan.json").resolve(), summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
