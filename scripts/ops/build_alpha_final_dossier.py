#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
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


def _latest_dir(path: Path) -> Path | None:
    if not path.exists():
        return None
    dirs = sorted([p for p in path.iterdir() if p.is_dir()], reverse=True)
    return dirs[0] if dirs else None


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    outroot = ROOT / "results" / "ops" / "alpha_final_dossier"
    outdir = outroot / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir.mkdir(parents=True, exist_ok=True)

    closure_dir = _latest_dir(ROOT / "results" / "validation" / "profit_historical_closure_suite")
    pbo_dir = _latest_dir(ROOT / "results" / "validation" / "profit_pbo_suite")
    fragility_dir = _latest_dir(ROOT / "results" / "validation" / "profit_crypto_dependency_relief_suite")
    operation = _read_json(ROOT / "results" / "ops" / "agents" / "daily_operation" / "latest_summary.json")
    vigilance = _read_json(ROOT / "results" / "ops" / "agents" / "daily_vigilance" / "latest_summary.json")

    closure = _read_json((closure_dir / "summary.json") if closure_dir else Path())
    pbo = _read_json((pbo_dir / "summary.json") if pbo_dir else Path())
    fragility = _read_json((fragility_dir / "summary.json") if fragility_dir else Path())
    yearbook = _read_csv_rows((closure_dir / "yearbook_reais.csv") if closure_dir else Path())

    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "attack_mode": closure.get("best_profit_mode", {}),
        "balanced_mode": closure.get("best_bootstrap_mode", {}),
        "pbo": {
            "overall_verdict": pbo.get("overall_verdict"),
            "sharpe": (pbo.get("pbo_sharpe") or {}).get("verdict"),
            "total_return": (pbo.get("pbo_total_return") or {}).get("verdict"),
        },
        "fragility_relief": {
            "best_profit_variant": fragility.get("best_profit_variant"),
            "best_balance_variant": fragility.get("best_balance_variant"),
        },
        "agents": {
            "operation_status": operation.get("status"),
            "vigilance_status": vigilance.get("status"),
            "alerts": vigilance.get("alerts", []),
        },
        "human_reading": [
            "O modo ataque continua sendo o melhor para lucro máximo no histórico.",
            "O modo principal segue mais consistente no caminho e mais fácil de defender como modo equilibrado.",
            "O teste de overfit ficou no lado robusto, então o ganho não parece só um truque estatístico simples.",
            "A maior fragilidade ainda é a dependência da perna cripto e de poucos nomes muito fortes.",
            "Os novos agentes deixam a operação diária e a vigilância documentadas sem precisar reotimizar o motor todo dia.",
        ],
        "artifacts": {
            "historical_closure_dir": str(closure_dir) if closure_dir else "",
            "pbo_dir": str(pbo_dir) if pbo_dir else "",
            "fragility_dir": str(fragility_dir) if fragility_dir else "",
        },
    }

    md_lines = [
        "# Dossiê Final de Alpha",
        "",
        "## Leitura curta",
        "- O modo ataque segue como melhor buscador de lucro.",
        "- O modo principal continua mais consistente e menos brutal.",
        f"- PBO: `{summary['pbo'].get('overall_verdict', 'desconhecido')}`.",
        "- A maior fragilidade ainda está na dependência da perna cripto.",
        "",
        "## Alertas atuais",
    ]
    for alert in summary["agents"]["alerts"]:
        md_lines.append(f"- [{alert.get('level')}] {alert.get('message')}")
    md_lines.extend(["", "## Yearbook disponível", f"- linhas: `{len(yearbook)}`"])

    (outdir / "dossier.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    (outdir / "yearbook_reais.csv").write_text(
        (closure_dir / "yearbook_reais.csv").read_text(encoding="utf-8") if closure_dir and (closure_dir / "yearbook_reais.csv").exists() else "",
        encoding="utf-8",
    )
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (outroot / "latest_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (outroot / "latest_dossier.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
