#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _latest_dir(root: Path) -> Path:
    dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not dirs:
        raise FileNotFoundError(f"no dirs under {root}")
    return dirs[-1]


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main() -> None:
    ap = argparse.ArgumentParser(description="Audita falhas metodologicas da pesquisa de lucro.")
    ap.add_argument("--suite-dir", default="")
    ap.add_argument("--suite-min6-dir", default="")
    ap.add_argument("--oos-dir", default="")
    ap.add_argument("--outdir-root", default="results/validation/profit_method_failure_audit")
    args = ap.parse_args()

    suite_dir = Path(args.suite_dir).resolve() if str(args.suite_dir).strip() else _latest_dir((ROOT / "results/validation/profit_group_methodology_suite").resolve())
    min6_dir = Path(args.suite_min6_dir).resolve() if str(args.suite_min6_dir).strip() else _latest_dir((ROOT / "results/validation/profit_group_methodology_suite_min6").resolve())
    oos_dir = Path(args.oos_dir).resolve() if str(args.oos_dir).strip() else _latest_dir((ROOT / "results/validation/profit_group_oos_validation").resolve())
    outdir = (ROOT / args.outdir_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    suite = _read_json(suite_dir / "summary.json")
    suite_min6 = _read_json(min6_dir / "summary.json")
    oos = _read_json(oos_dir / "summary.json")
    group_pool = pd.read_csv(suite_dir / "group_asset_pool.csv")
    candidate_compare = pd.read_csv(suite_dir / "candidate_compare.csv")

    small_groups = (
        group_pool.groupby("asset_group")["asset_id"]
        .nunique()
        .sort_values()
        .reset_index(name="n_assets")
    )
    risky_small = small_groups[small_groups["n_assets"] < 6].copy()

    top_full = suite.get("top_candidates", {}).get("best_net_blended", {})
    top_min6 = suite_min6.get("top_candidates", {}).get("best_net_blended", {})
    top_oos = {}
    best_consistent = oos.get("best_consistent_candidates", [])
    if isinstance(best_consistent, list) and best_consistent:
        top_oos = best_consistent[0]

    findings = []
    findings.append(
        {
            "severity": "high",
            "id": "full_sample_liquidity_selection",
            "message": "A selecao de ativos por grupo ainda usa liquidity_proxy e universo construidos com historico completo, o que introduz survivorship/lookahead na definicao do sleeve.",
        }
    )
    findings.append(
        {
            "severity": "high",
            "id": "static_internal_turnover_not_modeled",
            "message": "Combos estaticos carregam turnover mensal zero; custos internos de rebalanceamento dentro do sleeve nao foram modelados, entao o lucro liquido absoluto desses combos esta inflado.",
        }
    )
    if not risky_small.empty:
        findings.append(
            {
                "severity": "high",
                "id": "small_group_depth",
                "message": f"Existem grupos com profundidade baixa no sleeve atual: {risky_small.to_dict(orient='records')}.",
            }
        )
    top1 = oos.get("top1_by_split", [])
    if isinstance(top1, list) and top1:
        failed_top1 = [row for row in top1 if _safe_float(row.get("test_edge_vs_spy_net_total_return")) < 0.0]
        if failed_top1:
            findings.append(
                {
                    "severity": "medium",
                    "id": "top1_in_sample_overfit",
                    "message": "O vencedor por treino em cada split nao generalizou; o top1 in-sample falhou OOS em todos os blocos testados.",
                    "details": failed_top1,
                }
            )

    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite_dir": str(suite_dir),
        "suite_min6_dir": str(min6_dir),
        "oos_dir": str(oos_dir),
        "top_candidates": {
            "full_sample": top_full,
            "min_group_assets_6": top_min6,
            "best_oos_consistent": top_oos,
        },
        "small_groups": risky_small.to_dict(orient="records"),
        "candidate_counts": {
            "suite_total": int(candidate_compare.shape[0]),
            "small_groups_count": int(risky_small.shape[0]),
        },
        "findings": findings,
        "verdict": {
            "promotable_now": False,
            "best_research_candidate": top_oos.get("candidate_id", top_full.get("candidate_id", "")),
            "why_not_promotable": [
                "ainda existe survivorship/lookahead na selecao do sleeve",
                "custos internos do sleeve estatico nao estao modelados",
                "grupos pequenos ainda contaminam parte do ranking"
            ],
        },
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "ok", "outdir": str(outdir), "summary_json": str(outdir / "summary.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
