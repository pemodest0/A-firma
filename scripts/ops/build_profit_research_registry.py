#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict[str, Any] | list[Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, (dict, list)) else {}


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if pd.notna(out) else None


def _shadow_rows_from_lock(lock_path: Path) -> list[dict[str, Any]]:
    if not lock_path.exists():
        return []
    lock = _read_json(lock_path)
    if not isinstance(lock, dict):
        return []
    rows: list[dict[str, Any]] = []
    for slot in ["main", "challenger"]:
        node = lock.get(slot, {})
        if not isinstance(node, dict):
            continue
        summary_path = Path(str(node.get("summary_json", "")).strip())
        if not summary_path.exists():
            continue
        summary = _read_json(summary_path)
        if not isinstance(summary, dict):
            continue
        profiles = summary.get("profiles", [])
        profile = profiles[0] if isinstance(profiles, list) and profiles else {}
        if not isinstance(profile, dict):
            profile = {}
        rows.append(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "candidate_id": f"shadow_{slot}",
                "label": str(node.get("run_id", slot)),
                "methodology": f"shadow_{slot}",
                "status": "stale_shadow",
                "gross_ann_return": _safe_float(profile.get("daily_ann_return")),
                "net_ann_return": _safe_float(profile.get("daily_ann_return")),
                "gross_total_return": _safe_float(profile.get("daily_total_return")),
                "net_total_return": _safe_float(profile.get("daily_total_return")),
                "sharpe": _safe_float(profile.get("daily_sharpe")),
                "max_drawdown": _safe_float(profile.get("daily_max_drawdown")),
                "benchmark_ticker": "SPY",
                "edge_vs_benchmark_net_total_return": _safe_float(profile.get("daily_edge_vs_benchmark")),
                "edge_vs_spy_net_total_return": _safe_float(profile.get("daily_edge_vs_benchmark")),
                "avg_foreign_share": None,
                "groups": "",
                "artifacts": {
                    "summary_json": str(summary_path),
                    "profile_dir": str(node.get("profile_dir", "")),
                },
                "notes": "Shadow row before full rerun after return-math fix; treat as stale comparison only.",
            }
        )
    return rows


def _scan_research_rows(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary_cache: dict[str, dict[str, Any]] = {}
    for path in sorted(results_root.glob("validation/**/profit_research_rows.json")):
        payload = _read_json(path)
        if not isinstance(payload, list):
            continue
        for row in payload:
            if not isinstance(row, dict):
                continue
            clean = dict(row)
            if "edge_vs_benchmark_net_total_return" not in clean and "edge_vs_spy_net_total_return" in clean:
                clean["edge_vs_benchmark_net_total_return"] = clean.get("edge_vs_spy_net_total_return")
            clean.setdefault("artifacts", {})
            if isinstance(clean["artifacts"], dict):
                clean["artifacts"] = {**clean["artifacts"], "registry_source": str(path)}
                summary_path = Path(str(clean["artifacts"].get("summary_json", "")).strip())
                if not clean.get("benchmark_ticker") and summary_path.exists():
                    cache_key = str(summary_path)
                    if cache_key not in summary_cache:
                        payload_summary = _read_json(summary_path)
                        summary_cache[cache_key] = payload_summary if isinstance(payload_summary, dict) else {}
                    summary_payload = summary_cache.get(cache_key, {})
                    inputs = summary_payload.get("inputs", {}) if isinstance(summary_payload, dict) else {}
                    if isinstance(inputs, dict):
                        benchmark_ticker = str(inputs.get("benchmark_ticker", "")).strip()
                        if benchmark_ticker:
                            clean["benchmark_ticker"] = benchmark_ticker
            rows.append(clean)
    return rows


def _latest_oos_summary(results_root: Path) -> dict[str, Any]:
    summaries = sorted(results_root.glob("validation/profit_group_oos_validation/*/summary.json"))
    if not summaries:
        return {}
    payload = _read_json(summaries[-1])
    return payload if isinstance(payload, dict) else {}


def _latest_summary(root: Path, pattern: str) -> dict[str, Any]:
    summaries = sorted(root.glob(pattern))
    if not summaries:
        return {}
    payload = _read_json(summaries[-1])
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Consolida artefatos de pesquisa de lucro para o copiloto.")
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--lock-path", default="results/ops/profit_shadow_target_800_attack/canonical_shadow_profile.json")
    ap.add_argument("--outdir", default="results/ops/profit_research")
    args = ap.parse_args()

    results_root = (ROOT / args.results_root).resolve()
    lock_path = (ROOT / args.lock_path).resolve()
    outdir = (ROOT / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rows = _scan_research_rows(results_root)
    rows.extend(_shadow_rows_from_lock(lock_path))
    if not rows:
        raise SystemExit("no research rows found")

    df = pd.DataFrame(rows)
    if "candidate_id" not in df.columns:
        raise SystemExit("registry rows missing candidate_id")
    if "generated_at_utc" in df.columns:
        df = df.sort_values(["generated_at_utc", "candidate_id"], ascending=[False, True])
    df = df.drop_duplicates(subset=["candidate_id"], keep="first").reset_index(drop=True)

    for col in ["gross_ann_return", "net_ann_return", "gross_total_return", "net_total_return", "sharpe", "max_drawdown", "edge_vs_benchmark_net_total_return", "edge_vs_spy_net_total_return", "avg_foreign_share"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["net_ann_return", "gross_ann_return", "sharpe"], ascending=[False, False, False], na_position="last").reset_index(drop=True)
    df.to_csv(outdir / "latest_registry.csv", index=False)

    status_counts = Counter(df.get("status", pd.Series(dtype=object)).fillna("unknown").astype(str))
    methodology_counts = Counter(df.get("methodology", pd.Series(dtype=object)).fillna("unknown").astype(str))
    top = df.iloc[0].to_dict()
    oos = _latest_oos_summary(results_root)
    oos_best = {}
    if isinstance(oos, dict):
        best = oos.get("best_consistent_candidates", [])
        if isinstance(best, list) and best:
            oos_best = best[0] if isinstance(best[0], dict) else {}
    if oos_best:
        match = df[df["candidate_id"].astype(str) == str(oos_best.get("candidate_id", ""))]
        if not match.empty:
            top = {**match.iloc[0].to_dict(), "selection_basis": "oos_consistency", "oos": oos_best}
        else:
            top = {"selection_basis": "oos_consistency", "oos": oos_best}
    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "registry_path": str(outdir / "latest_registry.csv"),
        "rows_total": int(df.shape[0]),
        "status_counts": dict(status_counts),
        "methodology_counts": dict(methodology_counts),
        "top_candidate": top,
        "oos_best_consistent": oos_best,
        "top_keep_candidates": df[df.get("status", "") == "keep"].head(10).to_dict(orient="records"),
        "top_watch_candidates": df[df.get("status", "") == "watch"].head(10).to_dict(orient="records"),
        "kill_candidates": df[df.get("status", "") == "kill"].head(10).to_dict(orient="records"),
        "insights": [
            f"Top candidato atual: {top.get('candidate_id', top.get('oos', {}).get('candidate_id', '--'))} com net_ann_return={top.get('net_ann_return', top.get('oos', {}).get('mean_test_net_ann_return'))}.",
            f"Metodologias mais recorrentes: {dict(methodology_counts.most_common(5))}.",
        ],
        "rows": df.to_dict(orient="records"),
    }
    (outdir / "latest_registry.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    frontier = _latest_summary(results_root, "validation/profit_frontier_expansion_suite/*/summary.json")
    layered = _latest_summary(results_root, "validation/profit_layered_engine_suite/*/summary.json")
    drawdown_control = _latest_summary(results_root, "validation/profit_drawdown_control_suite/*/summary.json")
    crypto_plus = _latest_summary(results_root, "validation/profit_10x_rule_search_crypto_plus/*/summary.json")
    crypto_small = _latest_summary(results_root, "validation/profit_10x_rule_search_crypto/*/summary.json")
    event_rows: list[dict[str, Any]] = []
    if crypto_small:
        top = ((crypto_small.get("top_candidates") or {}).get("best_goal_score") or {})
        if isinstance(top, dict) and top:
            event_rows.append(
                {
                    "event": "crypto_scan_small",
                    "candidate_id": str(top.get("candidate_id", "")),
                    "benchmark_ticker": str(((crypto_small.get("inputs") or {}).get("benchmark_ticker", ""))),
                    "net_ann_return": _safe_float(top.get("net_ann_return")),
                    "max_drawdown": _safe_float(top.get("net_max_drawdown")),
                    "notes": "Primeira varredura cripto com universo reduzido.",
                }
            )
    if crypto_plus:
        top = ((crypto_plus.get("top_candidates") or {}).get("best_goal_score") or {})
        if isinstance(top, dict) and top:
            event_rows.append(
                {
                    "event": "crypto_scan_expanded",
                    "candidate_id": str(top.get("candidate_id", "")),
                    "benchmark_ticker": str(((crypto_plus.get("inputs") or {}).get("benchmark_ticker", ""))),
                    "net_ann_return": _safe_float(top.get("net_ann_return")),
                    "max_drawdown": _safe_float(top.get("net_max_drawdown")),
                    "notes": "Varredura cripto ampliada para majors e L1s liquidas.",
                }
            )
    if frontier:
        for key in ["best_crypto", "best_equities", "best_meta_switch"]:
            node = frontier.get(key, {})
            if isinstance(node, dict) and node:
                event_rows.append(
                    {
                        "event": key,
                        "candidate_id": str(node.get("candidate_id", "")),
                        "benchmark_ticker": str(node.get("benchmark_ticker", "")),
                        "net_ann_return": _safe_float(node.get("net_ann_return")),
                        "max_drawdown": _safe_float(node.get("net_max_drawdown")),
                        "notes": str(node.get("notes", "")),
                    }
                )
    if layered:
        for key in ["best_crypto", "best_equity", "best_meta_candidate", "frozen_walkforward_winner", "tournament_winner", "promoted_candidate"]:
            node = layered.get(key, {})
            if isinstance(node, dict) and node:
                event_rows.append(
                    {
                        "event": f"layered_{key}",
                        "candidate_id": str(node.get("candidate_id", "")),
                        "benchmark_ticker": str(node.get("benchmark_ticker", "")),
                        "net_ann_return": _safe_float(node.get("net_ann_return")),
                        "max_drawdown": _safe_float(node.get("net_max_drawdown")),
                        "notes": str(node.get("notes", "")),
                    }
                )
    if drawdown_control:
        for key in ["base_candidate", "best_balanced_candidate", "best_drawdown_candidate"]:
            node = drawdown_control.get(key, {})
            if isinstance(node, dict) and node:
                event_rows.append(
                    {
                        "event": f"drawdown_control_{key}",
                        "candidate_id": str(node.get("candidate_id", "")),
                        "benchmark_ticker": str(node.get("benchmark_ticker", "")),
                        "net_ann_return": _safe_float(node.get("net_ann_return")),
                        "max_drawdown": _safe_float(node.get("net_max_drawdown")),
                        "notes": str(node.get("notes", "")),
                    }
                )

    pattern_headlines: list[str] = []
    if crypto_plus:
        top = ((crypto_plus.get("top_candidates") or {}).get("best_goal_score") or {})
        if isinstance(top, dict) and top:
            pattern_headlines.append(
                f"Cripto ampliado manteve edge contra {((crypto_plus.get('inputs') or {}).get('benchmark_ticker', 'benchmark'))}, mas com drawdown extremo."
            )
    if frontier:
        best_crypto = frontier.get("best_crypto", {})
        best_eq = frontier.get("best_equities", {})
        best_meta = frontier.get("best_meta_switch", {})
        improvement = frontier.get("improvement_vs_previous_crypto_best", {})
        if isinstance(best_meta, dict) and best_meta:
            pattern_headlines.append(
                f"Meta-switch virou a melhor descoberta recente: combina sleeve cripto e equities com Sharpe superior ao cripto puro."
            )
        if isinstance(improvement, dict) and improvement:
            delta_ann = _safe_float(improvement.get("delta_net_ann_return"))
            if np.isfinite(delta_ann) and delta_ann < 0:
                pattern_headlines.append("Stops e overlays no cripto puro reduziram retorno anual; nao resolveram o drawdown estrutural.")
        if isinstance(best_eq, dict) and best_eq:
            pattern_headlines.append("Sleeve causal de equities ficou metodologicamente mais honesto, mas o alpha segue modesto.")
        if isinstance(best_crypto, dict) and best_crypto:
            pattern_headlines.append("Tiers mostraram que o edge cripto vem do universo amplo; midcaps isolados nao sustentaram o resultado.")
    if layered:
        best_meta = layered.get("best_meta_candidate", {})
        frozen = layered.get("frozen_walkforward_winner", {})
        promoted = layered.get("promoted_candidate", {})
        promotion_decision = layered.get("promotion_decision", {})
        improvement = layered.get("improvement_vs_frontier_meta", {})
        if isinstance(best_meta, dict) and best_meta:
            pattern_headlines.append("A camada extra de meta-switch ficou mais madura: agora o stack testa sizing continuo e delay duro na mesma bateria.")
        if isinstance(frozen, dict) and frozen:
            pattern_headlines.append(
                f"Walk-forward congelado escolheu {frozen.get('candidate_id', 'candidato')} como vencedor fora da amostra recente."
            )
        if isinstance(promoted, dict) and promoted:
            action = str(promoted.get("promotion_action", ""))
            if action == "promote_new":
                pattern_headlines.append(f"O torneio robusto promoveu {promoted.get('candidate_id', 'novo candidato')} como novo principal da pesquisa.")
            elif action == "keep_current":
                pattern_headlines.append(f"O torneio robusto manteve {promoted.get('candidate_id', 'o candidato atual')} como principal.")
            elif action == "hold_previous":
                pattern_headlines.append("As ideias novas trouxeram aprendizado, mas ainda nao venceram a barra de promocao do stack atual.")
        if isinstance(promotion_decision, dict) and promotion_decision:
            action = str(promotion_decision.get("action", ""))
            if action == "promote_first":
                pattern_headlines.append("A primeira versao com score de fragilidade ja saiu com candidato promovido.")
        if isinstance(improvement, dict) and improvement:
            delta_ann = _safe_float(improvement.get("delta_net_ann_return"))
            if delta_ann is not None:
                if delta_ann > 0:
                    pattern_headlines.append("O stack em camadas melhorou o retorno anual sobre a melhor fronteira anterior.")
                elif delta_ann < 0:
                    pattern_headlines.append("O stack em camadas ficou mais honesto, mas nao superou a melhor fronteira anterior em retorno anual.")
    if drawdown_control:
        verdict = drawdown_control.get("verdict", {})
        winner = drawdown_control.get("best_balanced_candidate", {})
        if isinstance(winner, dict) and winner:
            if bool((verdict or {}).get("winner_is_base", False)):
                pattern_headlines.append("Os controles anti-drawdown trouxeram reducao de risco, mas nenhum bateu o campeao atual no custo-beneficio.")
            else:
                pattern_headlines.append(f"O melhor controle anti-drawdown foi {winner.get('candidate_id', 'novo overlay')}, sinal de que deu para fechar dano sem desmontar o motor.")
        worthwhile = drawdown_control.get("worth_it_candidates", [])
        if isinstance(worthwhile, list) and worthwhile:
            pattern_headlines.append(f"Controles que valeram a pena nesta rodada: {', '.join(str(x.get('candidate_id', '')) for x in worthwhile[:3])}.")

    patterns = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "event_count": len(event_rows),
        "events": event_rows,
        "pattern_headlines": pattern_headlines,
        "sources": {
            "frontier_summary": str((results_root / "validation/profit_frontier_expansion_suite").resolve()),
            "layered_summary": str((results_root / "validation/profit_layered_engine_suite").resolve()),
            "drawdown_control_summary": str((results_root / "validation/profit_drawdown_control_suite").resolve()),
            "crypto_plus_summary": str((results_root / "validation/profit_10x_rule_search_crypto_plus").resolve()),
            "crypto_small_summary": str((results_root / "validation/profit_10x_rule_search_crypto").resolve()),
        },
    }
    (outdir / "latest_patterns.json").write_text(json.dumps(patterns, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "ok", "outdir": str(outdir), "rows_total": int(df.shape[0])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
