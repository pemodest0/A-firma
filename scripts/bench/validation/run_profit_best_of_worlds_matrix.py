#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]


RECOMMENDATIONS: dict[str, dict[str, str]] = {
    "champion_profit_lock_partial": {
        "action": "promote",
        "role": "core",
        "confidence": "medium_good",
        "reason": "Melhor oficial atual; causal; melhor equilibrio entre retorno e uso serio.",
    },
    "criticality_free_energy_attack": {
        "action": "shadow",
        "role": "structural_anchor",
        "confidence": "medium",
        "reason": "Melhor base estrutural; util como ancora e challenger do core.",
    },
    "champion_gradual_posture_cap": {
        "action": "test_more",
        "role": "official_guard_alt",
        "confidence": "medium",
        "reason": "Menos retorno, mas bom candidato para defesa progressiva.",
    },
    "champion_fragility_u800_bridge": {
        "action": "test_more",
        "role": "official_alt",
        "confidence": "medium_low",
        "reason": "Bom challenger, mas ainda atras do oficial promovido.",
    },
    "champion_fragility_decile": {
        "action": "test_more",
        "role": "official_alt",
        "confidence": "medium_low",
        "reason": "Variante interessante de fragilidade, ainda sem motivo para promover.",
    },
    "meta_dd_combo__conv25_vol25": {
        "action": "promote",
        "role": "guard",
        "confidence": "medium_good",
        "reason": "Melhor compromisso drawdown/Sharpe entre os guards mais fortes.",
    },
    "meta_dd_voltarget__20": {
        "action": "shadow",
        "role": "guard_alt",
        "confidence": "medium",
        "reason": "Alternativa simples de controle de volatilidade com drawdown baixo.",
    },
    "meta_exit__eq_guard_br52_ed1_dd14": {
        "action": "shadow",
        "role": "guard_alt",
        "confidence": "medium",
        "reason": "Guard mais ofensivo; bom como comparador live.",
    },
    "meta_dd_voltarget__25": {
        "action": "test_more",
        "role": "guard_alt",
        "confidence": "medium_low",
        "reason": "Parecido com o voltarget 20, mas sem vantagem clara.",
    },
    "meta_exit__eq_fast_br50_ed0_dd12": {
        "action": "test_more",
        "role": "guard_alt",
        "confidence": "medium_low",
        "reason": "Bom candidato de saida rapida, mas precisa mais stress de custo.",
    },
    "all_assets__lb021__rb07__k2__mom_total__ama000__mma000__riskon": {
        "action": "promote",
        "role": "attack",
        "confidence": "medium",
        "reason": "Melhor ataque por mediana no curto e medio prazo; mais balanceado que o turbo.",
    },
    "all_assets__lb126__rb07__k2__mom_total__ama200__mma200__riskon": {
        "action": "shadow",
        "role": "attack_alt",
        "confidence": "medium",
        "reason": "Ataque mais lento e mais limpo; bom challenger do ataque principal.",
    },
    "all_assets__lb021__rb07__k3__mom_total__ama000__mma000__riskon": {
        "action": "shadow",
        "role": "attack_alt",
        "confidence": "medium_low",
        "reason": "Agressivo forte, mas sem ganho estrutural suficiente sobre o k2.",
    },
    "all_assets__lb021__rb07__k3__mom_vol_adj__ama000__mma000__relshy": {
        "action": "shadow",
        "role": "turbo_shadow",
        "confidence": "low",
        "reason": "Melhor cauda explosiva; manter em shadow por risco alto de overfit.",
    },
    "meta_mode_selector": {
        "action": "shadow",
        "role": "context_selector",
        "confidence": "medium_low",
        "reason": "Promissor para selecionar contexto, mas ainda muito laboratorio.",
    },
    "alpha_attack_major8_equity25": {
        "action": "test_more",
        "role": "legacy_meta",
        "confidence": "low",
        "reason": "Forte historicamente, mas legado e mais suspeito de otimização.",
    },
    "alpha_attack_major8_equity25_mc_guard": {
        "action": "shadow",
        "role": "meta_guard_alt",
        "confidence": "medium_low",
        "reason": "Variante guard do legado, boa para comparacao de contexto.",
    },
    "meta_major8_eq_a2r1_mc_guard": {
        "action": "test_more",
        "role": "meta_guard_alt",
        "confidence": "medium_low",
        "reason": "Menos retorno que os guards escolhidos; manter como referencia.",
    },
    "baseline_fast_entry": {
        "action": "discard_live",
        "role": "research_turbo",
        "confidence": "low",
        "reason": "Muito forte no laboratorio, mas sem evidencias suficientes para live.",
    },
    "entry_fast14_exit63_m2_h0__wrapped": {
        "action": "discard_live",
        "role": "research_turbo",
        "confidence": "low",
        "reason": "Wrapper do baseline turbo; mesmo problema de robustez.",
    },
    "critical_slowing_down": {
        "action": "test_more",
        "role": "research_structural",
        "confidence": "low",
        "reason": "Sinal fisico interessante; ainda precisa bateria cruzada antes de live.",
    },
    "structural_stress": {
        "action": "test_more",
        "role": "research_structural",
        "confidence": "low",
        "reason": "Boa hipotese de estrutura, mas ainda muito de laboratorio.",
    },
    "critical_plus_stress": {
        "action": "test_more",
        "role": "research_structural",
        "confidence": "low",
        "reason": "Combinacao promissora, ainda sem validacao operacional suficiente.",
    },
}


BEST_OF_WORLDS = {
    "core": "champion_profit_lock_partial",
    "structural_anchor": "criticality_free_energy_attack",
    "guard": "meta_dd_combo__conv25_vol25",
    "guard_alt": "meta_dd_voltarget__20",
    "attack": "all_assets__lb021__rb07__k2__mom_total__ama000__mma000__riskon",
    "attack_alt": "all_assets__lb126__rb07__k2__mom_total__ama200__mma200__riskon",
    "context_selector": "meta_mode_selector",
    "turbo_shadow": "all_assets__lb021__rb07__k3__mom_vol_adj__ama000__mma000__relshy",
    "deep_research": "critical_slowing_down",
}


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _flatten_categories(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category, candidates in summary.get("categories", {}).items():
        for row in candidates:
            out = dict(row)
            out["category"] = str(category)
            cid = str(out.get("candidate_id", ""))
            rec = RECOMMENDATIONS.get(
                cid,
                {
                    "action": "test_more",
                    "role": "unclassified",
                    "confidence": "unknown",
                    "reason": "Nao classificado manualmente ainda.",
                },
            )
            out["recommended_action"] = rec["action"]
            out["recommended_role"] = rec["role"]
            out["confidence_band"] = rec["confidence"]
            out["reason"] = rec["reason"]
            out["score_return"] = _safe_float(out.get("net_total_return"))
            out["score_short_horizon"] = _safe_float(out.get("median_return_252d"))
            out["score_drawdown"] = _safe_float(out.get("net_max_drawdown"))
            rows.append(out)
    return rows


def _best_of_worlds_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_candidate = {str(row["candidate_id"]): row for row in rows}
    out: dict[str, dict[str, Any]] = {}
    for slot, candidate_id in BEST_OF_WORLDS.items():
        chosen = dict(by_candidate.get(candidate_id, {"candidate_id": candidate_id}))
        chosen["slot"] = slot
        out[slot] = chosen
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Matriz final dos campeoes e arquitetura do melhor dos mundos.")
    ap.add_argument(
        "--catalog",
        default="results/validation/profit_champion_battery_catalog/20260315T071128Z/summary.json",
    )
    ap.add_argument("--outdir-root", default="results/validation/profit_best_of_worlds_matrix")
    args = ap.parse_args()

    summary = json.loads((ROOT / args.catalog).read_text())
    rows = _flatten_categories(summary)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = ROOT / args.outdir_root / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    _write_csv(outdir / "decision_matrix.csv", rows)
    best = _best_of_worlds_rows(rows)
    matrix_summary = {
        "suite": "profit_best_of_worlds_matrix",
        "run_id": run_id,
        "catalog_source": str(ROOT / args.catalog),
        "best_of_worlds": best,
        "notes": [
            "Promote = candidato bom o suficiente para motor live.",
            "Shadow = usar em paralelo no laboratorio operacional.",
            "Test_more = manter na bateria cruzada antes de qualquer promocao.",
            "Discard_live = muito forte no laboratorio, mas nao merece live hoje.",
        ],
    }
    (outdir / "summary.json").write_text(json.dumps(matrix_summary, ensure_ascii=False, indent=2))
    print(json.dumps({"status": "ok", "outdir": str(outdir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
