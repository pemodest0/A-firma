#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.structural.run_manifest import write_run_manifest  # noqa: E402
from scripts.bench.validation.run_profit_frontier_expansion_suite import _write_json  # noqa: E402


SUITE_SPECS: list[dict[str, Any]] = [
    {
        "order": 1,
        "suite_id": "historical_closure",
        "label": "Fechamento historico",
        "relpath": "validation/profit_historical_closure_suite",
        "summary": "summary.json",
        "compare": "frozen_overview.csv",
        "goal": "Validar dinheiro no fim, ano a ano, meses, trimestres, bootstrap e ablação.",
        "scenarios": "historico completo, ano a ano, month/quarter tournament, bootstrap, ablação",
    },
    {
        "order": 2,
        "suite_id": "pbo",
        "label": "Overfit",
        "relpath": "validation/profit_pbo_suite",
        "summary": "summary.json",
        "compare": "pbo_metric_summary.csv",
        "goal": "Medir risco de overfit usando splits combinatórios.",
        "scenarios": "PBO por Sharpe e retorno total",
    },
    {
        "order": 3,
        "suite_id": "execution_phase",
        "label": "Fases de mercado",
        "relpath": "validation/profit_execution_phase_suite",
        "summary": "summary.json",
        "compare": None,
        "goal": "Ver quem ganha em bull, bear, recovery e lateral.",
        "scenarios": "phase splits e ano recente",
    },
    {
        "order": 4,
        "suite_id": "universe_resilience",
        "label": "Resiliencia de universo",
        "relpath": "validation/profit_universe_resilience_suite",
        "summary": "summary.json",
        "compare": None,
        "goal": "Medir fragilidade a packs, grupos e remocao dos nomes mais fortes.",
        "scenarios": "packs, drop top crypto, Monte Carlo curto",
    },
    {
        "order": 5,
        "suite_id": "bad_year_defense",
        "label": "Defesa de ano ruim",
        "relpath": "validation/profit_bad_year_defense_suite",
        "summary": "summary.json",
        "compare": "candidate_compare.csv",
        "goal": "Reduzir perda anual sem matar todo o lucro.",
        "scenarios": "guards mensais/trimestrais, defesa de ano, concentracao, caixa",
    },
    {
        "order": 6,
        "suite_id": "u800_alpha",
        "label": "Universo 800",
        "relpath": "validation/profit_u800_alpha_suite",
        "summary": "summary.json",
        "compare": "candidate_compare.csv",
        "goal": "Medir se o universo de 800 ativos melhora alpha e robustez.",
        "scenarios": "packs 800, apoio de ações, fragilidade cripto",
    },
    {
        "order": 7,
        "suite_id": "marketmode_criticality",
        "label": "Modo de mercado e criticidade",
        "relpath": "validation/profit_marketmode_criticality_suite",
        "summary": "summary.json",
        "compare": "candidate_compare.csv",
        "goal": "Separar pânico/euforia geral de sinal estrutural real.",
        "scenarios": "market mode, criticidade, energia livre, atrator, direção, curvatura",
    },
    {
        "order": 8,
        "suite_id": "meta_mode_selector",
        "label": "Meta-seletor causal",
        "relpath": "validation/profit_meta_mode_selector_suite",
        "summary": "summary.json",
        "compare": "candidate_compare.csv",
        "goal": "Ver se um seletor anual/mensal de modos bate os modos fixos.",
        "scenarios": "treino ate o mes anterior, escolha causal entre poucos modos",
    },
]


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_json(path: Path) -> dict[str, Any] | list[Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, (dict, list)) else {}


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if pd.notna(out) and np.isfinite(out) else None


def _latest_run_with_file(root: Path, marker_name: str) -> Path | None:
    if not root.exists():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir() and (p / marker_name).exists()]
    if not candidates:
        return None
    return sorted(candidates)[-1]


def _load_suite_bundle(results_root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    suite_root = results_root / spec["relpath"]
    latest = _latest_run_with_file(suite_root, str(spec["summary"]))
    if latest is None:
        return {
            **spec,
            "status": "missing",
            "latest_run": None,
            "summary_payload": {},
            "compare_df": pd.DataFrame(),
        }
    summary_path = latest / str(spec["summary"])
    payload = _read_json(summary_path)
    compare_df = pd.DataFrame()
    compare_name = spec.get("compare")
    if compare_name and (latest / str(compare_name)).exists():
        compare_df = pd.read_csv(latest / str(compare_name))
    return {
        **spec,
        "status": "ok",
        "latest_run": str(latest.name),
        "latest_path": str(latest),
        "summary_path": str(summary_path),
        "summary_payload": payload if isinstance(payload, dict) else {},
        "compare_df": compare_df,
    }


def _registry_frame(results_root: Path) -> pd.DataFrame:
    path = results_root / "ops/profit_research/latest_registry.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in ["net_ann_return", "net_total_return", "sharpe", "max_drawdown"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _candidate_label(registry: pd.DataFrame, candidate_id: str) -> str:
    if registry.empty:
        return str(candidate_id)
    rows = registry[registry["candidate_id"].astype(str) == str(candidate_id)]
    if rows.empty:
        return str(candidate_id)
    return str(rows.iloc[0].get("label", candidate_id))


def _candidate_metric_from_compare(df: pd.DataFrame, candidate_id: str, column: str) -> float | None:
    if df.empty or "candidate_id" not in df.columns or column not in df.columns:
        return None
    rows = df[df["candidate_id"].astype(str) == str(candidate_id)]
    if rows.empty:
        return None
    return _safe_float(rows.iloc[0].get(column))


def _candidate_id_from_summary_entry(entry: Any, default: str = "") -> str:
    if isinstance(entry, dict):
        value = entry.get("candidate_id", default)
        return str(value) if value is not None else str(default)
    if entry is None:
        return str(default)
    return str(entry)


def _build_hypothesis_catalog(suites: dict[str, dict[str, Any]], registry: pd.DataFrame) -> list[dict[str, Any]]:
    marketmode = suites["marketmode_criticality"]["summary_payload"]
    bad_year = suites["bad_year_defense"]["summary_payload"]
    selector = suites["meta_mode_selector"]["summary_payload"]
    u800 = suites["u800_alpha"]["summary_payload"]

    mm_best = _candidate_id_from_summary_entry(
        marketmode.get("best_candidate"),
        "criticality_free_energy_attack",
    )
    mm_second = _candidate_id_from_summary_entry(
        marketmode.get("best_challenger"),
        "criticality_guard_attack",
    )
    u800_best = _candidate_id_from_summary_entry(u800.get("best_retention_candidate"), "")
    if not u800_best:
        u800_best = _candidate_id_from_summary_entry(u800.get("best_profit_candidate"), "")
    if mm_second == mm_best:
        mm_second = "criticality_guard_attack"
    worth_keep = bad_year.get("worth_keeping_candidates", [])
    period_keep = ""
    if isinstance(worth_keep, list):
        for row in worth_keep:
            if isinstance(row, dict) and str(row.get("candidate_id", "")).startswith("period_loss_guards"):
                period_keep = str(row.get("candidate_id", ""))
                break

    hypotheses = [
        {
            "order": 1,
            "id": "criticality_free_energy_attack",
            "label": "Ataque com criticidade e reorganização leve",
            "status": "promote",
            "candidate_id": mm_best,
            "why": "Melhor conjunto recente de lucro, qualidade do caminho e leitura estrutural.",
            "approval_criteria": "Ganhar do ataque atual em lucro total e manter ou melhorar qualidade do retorno sem piorar muito a pior queda.",
            "discard_criteria": "Perder para o ataque atual no lucro total ou piorar claramente a robustez sem compensação.",
            "relevant_suites": ["marketmode_criticality", "meta_mode_selector", "historical_closure", "pbo"],
        },
        {
            "order": 2,
            "id": "criticality_guard_attack",
            "label": "Freio por criticidade",
            "status": "watch",
            "candidate_id": mm_second,
            "why": "Quase campeão e bom para reduzir entrada em histeria geral.",
            "approval_criteria": "Reter a maior parte do lucro do campeão e reduzir a pior queda.",
            "discard_criteria": "Ficar longe demais do campeão em lucro sem compensar em risco.",
            "relevant_suites": ["marketmode_criticality", "historical_closure"],
        },
        {
            "order": 3,
            "id": "period_loss_guards_light",
            "label": "Travas leves por mês e trimestre",
            "status": "keep",
            "candidate_id": period_keep or "period_loss_guards_light",
            "why": "Melhor defesa de ano ruim sem desmontar o restante do motor.",
            "approval_criteria": "Reduzir a profundidade do pior ano mantendo boa parte do lucro acumulado.",
            "discard_criteria": "Matar dinheiro demais no fim ou não aliviar o pior ano.",
            "relevant_suites": ["bad_year_defense", "historical_closure"],
        },
        {
            "order": 4,
            "id": "meta_mode_selector",
            "label": "Meta-seletor causal de modos",
            "status": "watch",
            "candidate_id": "meta_mode_selector",
            "why": "Quase encosta nos modos fixos e ensina quando trocar de modo.",
            "approval_criteria": "Bater os modos fixos ou ficar muito perto com giro claramente menor.",
            "discard_criteria": "Perder do campeão sem vantagem operacional clara.",
            "relevant_suites": ["meta_mode_selector", "historical_closure"],
        },
        {
            "order": 5,
            "id": "u800_equity_support",
            "label": "Apoio das ações com universo 800",
            "status": "watch",
            "candidate_id": u800_best or "meta_major8_equities_meta__trail_switch__a2__r1",
            "why": "Melhora breadth e reduz parte da dependência do cripto, mas não é campeão de lucro.",
            "approval_criteria": "Melhorar robustez sem perder materialmente para o campeão.",
            "discard_criteria": "Perder lucro demais sem reduzir fragilidade de forma relevante.",
            "relevant_suites": ["u800_alpha", "universe_resilience"],
        },
        {
            "order": 6,
            "id": "structural_attack_driver",
            "label": "Usar sinais estruturais para mandar no ataque",
            "status": "discard",
            "candidate_id": "critical_transition_or_crowding",
            "why": "Ajuda a explicar risco, mas não ganhou dinheiro suficiente como modo principal.",
            "approval_criteria": "Bater o ataque atual em lucro ou manter lucro com redução clara de pior queda.",
            "discard_criteria": "Virar só termômetro bonito sem melhorar o conjunto.",
            "relevant_suites": ["marketmode_criticality"],
        },
    ]

    for item in hypotheses:
        item["candidate_label"] = _candidate_label(registry, item["candidate_id"])
    return hypotheses


def _suite_rows(suites: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for spec in sorted(suites.values(), key=lambda x: int(x["order"])):
        payload = spec["summary_payload"]
        insights = payload.get("insights", []) if isinstance(payload, dict) else []
        rows.append(
            {
                "order": spec["order"],
                "suite_id": spec["suite_id"],
                "label": spec["label"],
                "status": spec["status"],
                "latest_run": spec.get("latest_run"),
                "goal": spec["goal"],
                "scenarios": spec["scenarios"],
                "summary_path": spec.get("summary_path"),
                "compare_rows": int(spec["compare_df"].shape[0]) if isinstance(spec.get("compare_df"), pd.DataFrame) else 0,
                "top_insight": insights[0] if isinstance(insights, list) and insights else "",
            }
        )
    return rows


def _candidate_ids_from_hypotheses(hypotheses: list[dict[str, Any]], registry: pd.DataFrame) -> list[str]:
    ids = [str(h["candidate_id"]) for h in hypotheses if str(h.get("candidate_id", "")).strip()]
    ids.extend(
        [
            "alpha_attack_major8_equity25",
            "meta_major8_eq_a2r1_mc_guard",
            "criticality_free_energy_attack",
            "criticality_guard_attack",
            "meta_mode_selector",
        ]
    )
    if not registry.empty:
        ids.extend(registry.head(8)["candidate_id"].astype(str).tolist())
    unique = []
    seen = set()
    for cid in ids:
        if cid in seen:
            continue
        seen.add(cid)
        unique.append(cid)
    return unique


def _suite_rank(df: pd.DataFrame, candidate_id: str) -> float | None:
    if df.empty or "candidate_id" not in df.columns:
        return None
    work = df.copy()
    if "net_total_return" not in work.columns:
        return None
    work["net_total_return"] = pd.to_numeric(work["net_total_return"], errors="coerce")
    if "net_sharpe" in work.columns:
        work["net_sharpe"] = pd.to_numeric(work["net_sharpe"], errors="coerce")
    if "net_max_drawdown" in work.columns:
        work["net_max_drawdown"] = pd.to_numeric(work["net_max_drawdown"], errors="coerce")
    sort_cols = ["net_total_return"]
    ascending = [False]
    if "net_sharpe" in work.columns:
        sort_cols.append("net_sharpe")
        ascending.append(False)
    if "net_max_drawdown" in work.columns:
        sort_cols.append("net_max_drawdown")
        ascending.append(False)
    work = work.sort_values(sort_cols, ascending=ascending, na_position="last").reset_index(drop=True)
    rows = work[work["candidate_id"].astype(str) == str(candidate_id)]
    if rows.empty:
        return None
    rank = int(rows.index[0]) + 1
    return float(rank)


def _candidate_master_board(
    hypotheses: list[dict[str, Any]],
    suites: dict[str, dict[str, Any]],
    registry: pd.DataFrame,
) -> pd.DataFrame:
    candidate_ids = _candidate_ids_from_hypotheses(hypotheses, registry)
    rows: list[dict[str, Any]] = []
    for cid in candidate_ids:
        reg = registry[registry["candidate_id"].astype(str) == str(cid)] if not registry.empty else pd.DataFrame()
        row: dict[str, Any] = {
            "candidate_id": cid,
            "candidate_label": _candidate_label(registry, cid),
            "registry_status": reg.iloc[0]["status"] if not reg.empty and "status" in reg.columns else "",
            "registry_net_ann_return": _safe_float(reg.iloc[0]["net_ann_return"]) if not reg.empty else None,
            "registry_net_total_return": _safe_float(reg.iloc[0]["net_total_return"]) if not reg.empty else None,
            "registry_sharpe": _safe_float(reg.iloc[0]["sharpe"]) if not reg.empty else None,
            "registry_max_drawdown": _safe_float(reg.iloc[0]["max_drawdown"]) if not reg.empty else None,
        }
        ranks: list[float] = []
        covered: list[str] = []
        win_count = 0
        for suite_id, spec in suites.items():
            df = spec["compare_df"]
            rank = _suite_rank(df, cid) if isinstance(df, pd.DataFrame) else None
            if rank is not None:
                covered.append(suite_id)
                ranks.append(rank)
                if rank == 1:
                    win_count += 1
            if suite_id == "marketmode_criticality":
                row["marketmode_net_ann_return"] = _candidate_metric_from_compare(df, cid, "net_ann_return")
                row["marketmode_net_total_return"] = _candidate_metric_from_compare(df, cid, "net_total_return")
                row["marketmode_net_sharpe"] = _candidate_metric_from_compare(df, cid, "net_sharpe")
            if suite_id == "bad_year_defense":
                row["bad_year_total_return_retention"] = _candidate_metric_from_compare(df, cid, "total_return_retention")
                row["bad_year_worst_year_profit_brl"] = _candidate_metric_from_compare(df, cid, "worst_year_profit_brl")
            if suite_id == "historical_closure":
                row["historical_profit_total_brl"] = _candidate_metric_from_compare(df, cid, "profit_total_brl")
                row["historical_years_negative"] = _candidate_metric_from_compare(df, cid, "years_negative")
                row["historical_worst_year_profit_brl"] = _candidate_metric_from_compare(df, cid, "worst_year_profit_brl")
        row["suites_covered"] = len(covered)
        row["suite_ids"] = ",".join(covered)
        row["suite_win_count"] = win_count
        if ranks:
            avg_rank = float(np.mean(ranks))
            row["average_suite_rank"] = avg_rank
            row["lab_score"] = float(1.0 / avg_rank)
        else:
            row["average_suite_rank"] = None
            row["lab_score"] = None
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["lab_score", "registry_net_total_return", "registry_sharpe"],
            ascending=[False, False, False],
            na_position="last",
        ).reset_index(drop=True)
    return df


def _pbo_panel(results_root: Path) -> dict[str, Any]:
    suite_root = results_root / "validation/profit_pbo_suite"
    latest = _latest_run_with_file(suite_root, "summary.json")
    if latest is None:
        return {"status": "missing"}
    summary = _read_json(latest / "summary.json")
    metrics = pd.DataFrame()
    if (latest / "pbo_metric_summary.csv").exists():
        metrics = pd.read_csv(latest / "pbo_metric_summary.csv")
    out = {
        "status": "ok",
        "latest_run": latest.name,
        "summary_path": str(latest / "summary.json"),
        "overall_verdict": summary.get("overall_verdict") if isinstance(summary, dict) else None,
        "pbo_sharpe": summary.get("pbo_sharpe") if isinstance(summary, dict) else None,
        "pbo_total_return": summary.get("pbo_total_return") if isinstance(summary, dict) else None,
    }
    if not metrics.empty:
        out["metrics"] = metrics.to_dict(orient="records")
    return out


def _summary(
    hypotheses: list[dict[str, Any]],
    suite_rows: list[dict[str, Any]],
    candidates: pd.DataFrame,
    pbo_panel: dict[str, Any],
) -> dict[str, Any]:
    top = candidates.iloc[0].to_dict() if not candidates.empty else {}
    keep = [h for h in hypotheses if h["status"] in {"promote", "keep"}]
    discard = [h for h in hypotheses if h["status"] == "discard"]
    watch = [h for h in hypotheses if h["status"] in {"watch", "queued", "testing"}]
    return {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "questions_answered": {
            "hypotheses_defined": True,
            "test_order_defined": True,
            "approval_criteria_defined": True,
            "discard_criteria_defined": True,
            "heavy_battery_consolidated": True,
        },
        "top_candidate": top,
        "keep_now": [{"id": h["id"], "label": h["label"], "candidate_id": h["candidate_id"]} for h in keep],
        "discard_now": [{"id": h["id"], "label": h["label"], "candidate_id": h["candidate_id"]} for h in discard],
        "watch_queue": [{"id": h["id"], "label": h["label"], "candidate_id": h["candidate_id"]} for h in watch],
        "suite_order": [{"order": r["order"], "suite_id": r["suite_id"], "label": r["label"]} for r in suite_rows],
        "pbo": pbo_panel,
        "insights": [
            "O laboratório agora separa explicitamente o que manter, o que descartar e o que continua em observação.",
            "A ordem de teste começa pelo fechamento histórico e pelo risco de overfit, e só depois olha fases, fragilidade e challengers.",
            "O melhor candidato atual continua vindo da frente de criticidade e reorganização leve.",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Consolida as baterias pesadas do laboratorio em um board unico de hipoteses, ordem e criterios.")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--outdir-root", default="results/validation/profit_hypothesis_lab_suite")
    ap.add_argument("--publish-ops", action="store_true", help="Tambem publica latest_lab_board.json em results/ops/profit_research.")
    args = ap.parse_args()

    results_root = (ROOT / args.results_root).resolve()
    outdir = (ROOT / args.outdir_root / _run_id()).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    suites = {spec["suite_id"]: _load_suite_bundle(results_root, spec) for spec in SUITE_SPECS}
    registry = _registry_frame(results_root)
    hypotheses = _build_hypothesis_catalog(suites, registry)
    suite_rows = _suite_rows(suites)
    candidates = _candidate_master_board(hypotheses, suites, registry)
    pbo = _pbo_panel(results_root)
    summary = _summary(hypotheses, suite_rows, candidates, pbo)

    hypothesis_df = pd.DataFrame(hypotheses)
    suite_df = pd.DataFrame(suite_rows)

    hypothesis_df.to_csv(outdir / "hypothesis_board.csv", index=False)
    suite_df.to_csv(outdir / "suite_order.csv", index=False)
    candidates.to_csv(outdir / "candidate_master_board.csv", index=False)
    _write_json(outdir / "summary.json", summary)
    _write_json(outdir / "hypotheses.json", {"hypotheses": hypotheses})
    _write_json(outdir / "pbo_panel.json", pbo)

    write_run_manifest(
        outdir,
        script="scripts/bench/validation/run_profit_hypothesis_lab_suite.py",
        params={
            "results_root": args.results_root,
            "outdir_root": args.outdir_root,
            "publish_ops": bool(args.publish_ops),
        },
        paths={
            "summary_json": str(outdir / "summary.json"),
            "hypothesis_board_csv": str(outdir / "hypothesis_board.csv"),
            "suite_order_csv": str(outdir / "suite_order.csv"),
            "candidate_master_board_csv": str(outdir / "candidate_master_board.csv"),
            "pbo_panel_json": str(outdir / "pbo_panel.json"),
        },
        extra={
            "suite": "profit_hypothesis_lab_suite",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "hypothesis_count": int(hypothesis_df.shape[0]),
            "suite_count": int(suite_df.shape[0]),
            "candidate_count": int(candidates.shape[0]),
            "top_candidate_id": summary.get("top_candidate", {}).get("candidate_id"),
        },
        repo_root=ROOT,
    )

    if args.publish_ops:
        ops_dir = (results_root / "ops/profit_research").resolve()
        ops_dir.mkdir(parents=True, exist_ok=True)
        _write_json(ops_dir / "latest_lab_board.json", summary)

    print(json.dumps({"status": "ok", "outdir": str(outdir), "top_candidate": summary.get("top_candidate", {})}, ensure_ascii=False))


if __name__ == "__main__":
    main()
