#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"


def read_json(path: Path, fallback: Any = None) -> Any:
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return fallback
    try:
        return json.loads(raw)
    except Exception:
        fixed = (
            raw.replace("NaN", "null")
            .replace("Infinity", "null")
            .replace("-null", "null")
        )
        try:
            return json.loads(fixed)
        except Exception:
            return fallback


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_num(value: Any) -> float | None:
    try:
        num = float(value)
    except Exception:
        return None
    return num if num == num and abs(num) != float("inf") else None


def deep_clean(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [deep_clean(item) for item in value]
    if isinstance(value, dict):
        return {key: deep_clean(item) for key, item in value.items()}
    return value


def latest_validation_dir(name: str) -> Path | None:
    root = RESULTS_ROOT / "validation" / name
    if not root.exists():
        return None
    candidates = sorted([p for p in root.iterdir() if p.is_dir()], reverse=True)
    return candidates[0] if candidates else None


def latest_lab_from_finance_ready() -> tuple[dict[str, Any], Path | None]:
    latest_pointer = read_json(RESULTS_ROOT / "ops" / "finance_product_ready" / "latest_finance_product_ready.json", {})
    report_path_raw = str(latest_pointer.get("finance_product_ready_json") or "").strip()
    report = read_json(Path(report_path_raw), {}) if report_path_raw else {}
    run_dir_raw = str(report.get("run_dir") or latest_pointer.get("run_dir") or "").strip()
    run_dir = Path(run_dir_raw) if run_dir_raw else None
    return report, run_dir


def latest_json_summary(validation_name: str) -> dict[str, Any]:
    latest_dir = latest_validation_dir(validation_name)
    if not latest_dir:
        return {}
    return read_json(latest_dir / "summary.json", {})


def latest_csv_rows(validation_name: str, filename: str) -> list[dict[str, str]]:
    latest_dir = latest_validation_dir(validation_name)
    if not latest_dir:
        return []
    return read_csv(latest_dir / filename)


def signal_status_from_regime(regime_asset: str) -> str:
    regime = (regime_asset or "").strip().lower()
    if regime in {"estavel", "stable"}:
        return "validated"
    if regime in {"transicao", "transition"}:
        return "watch"
    return "inconclusive"


def first_list_item(value: Any) -> Any:
    if isinstance(value, list) and value:
        return value[0]
    return {}


def select_attack_mode_from_registry(profit_registry: dict[str, Any]) -> dict[str, Any]:
    rows = profit_registry.get("rows", []) if isinstance(profit_registry, dict) else []
    if not isinstance(rows, list):
        return {}
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        methodology = str(row.get("methodology") or "").strip().lower()
        status = str(row.get("status") or "").strip().lower()
        if status not in {"keep", "watch"}:
            continue
        if not any(
            token in methodology
            for token in [
                "historical_closure_attack",
                "alpha_hardening",
                "attack_entry_ranking",
                "confidence_calibration",
                "alpha_improvement",
                "alpha_hardening_attack",
            ]
        ):
            continue
        candidates.append(row)
    if not candidates:
        return {}
    candidates.sort(
        key=lambda row: (
            1 if str(row.get("status") or "").strip().lower() == "keep" else 0,
            1
            if any(
                token in str(row.get("methodology") or "").strip().lower()
                for token in ["historical_closure_attack", "alpha_hardening"]
            )
            else 0,
            to_num(row.get("net_total_return")) if to_num(row.get("net_total_return")) is not None else -1e18,
            to_num(row.get("net_ann_return")) if to_num(row.get("net_ann_return")) is not None else -1e18,
            to_num(row.get("sharpe")) if to_num(row.get("sharpe")) is not None else -1e18,
        ),
        reverse=True,
    )
    return candidates[0]


def count_volume_support(universe_rows: list[dict[str, Any]]) -> tuple[int, int]:
    source_root = REPO_ROOT / "data" / "raw" / "finance" / "yfinance_daily"
    total = 0
    supported = 0
    for row in universe_rows:
        asset = str((row or {}).get("asset") or "").strip()
        if not asset:
            continue
        total += 1
        path = source_root / f"{asset}.csv"
        if not path.exists():
            continue
        try:
            header = path.open("r", encoding="utf-8").readline().strip().lower().split(",")
        except Exception:
            continue
        if "volume" in header:
            supported += 1
    return supported, total


def human_shadow_label(mode_key: str, candidate_id: str = "", fallback: str = "") -> str:
    mode_key = str(mode_key or "").strip().lower()
    candidate_id = str(candidate_id or "").strip().lower()
    if mode_key == "mode_attack":
        return "Ataque diário"
    if mode_key == "mode_main":
        return "Principal diário"
    if mode_key == "mode_attack_guard":
        return "Ataque com guarda"
    if mode_key == "mode_main_guard":
        return "Principal com guarda"
    if mode_key == "canonical_main":
        return "Canônico legado"
    if mode_key == "canonical_challenger":
        return "Challenger legado"
    if mode_key == "paper_live":
        return "Paper trading ao vivo"
    if mode_key == "paper_replay":
        return "Replay histórico"
    if mode_key == "research_top":
        return "Pesquisa líder"
    if mode_key == "research_oos":
        return "Pesquisa fora da amostra"
    if "alpha_attack" in candidate_id or "criticality" in candidate_id:
        return "Ataque quantitativo"
    if "conviction" in candidate_id or "guard" in candidate_id:
        return "Proteção quantitativa"
    return fallback or mode_key or "Shadow"


def human_shadow_variant(mode_key: str, candidate_id: str = "") -> str:
    mode_key = str(mode_key or "").strip().lower()
    candidate_id = str(candidate_id or "").strip().lower()
    if mode_key == "mode_attack":
        return "Varia o tamanho da aposta conforme a confiança e libera mais risco quando o cripto está saudável."
    if mode_key == "mode_main":
        return "Fica mais perto do modo principal e alterna entre risco moderado e proteção."
    if mode_key == "mode_attack_guard":
        return "Mesmo ataque, mas com trava extra quando a cauda do risco piora."
    if mode_key == "mode_main_guard":
        return "Modo principal com freio extra para meses e cenários mais duros."
    if mode_key == "canonical_main":
        return "Foto antiga do ataque original, útil para comparar a geração nova com a fase anterior do laboratório."
    if mode_key == "canonical_challenger":
        return "Versão antiga mais agressiva, guardada como referência histórica do laboratório."
    if mode_key == "paper_live":
        return "Paper trading diário com capital virtual, sem ordem real, mas acumulando como se estivesse rodando todo dia."
    if mode_key == "paper_replay":
        return "Replay histórico congelado para comparar o jeito atual de operar com um caminho longo e consistente."
    if mode_key == "research_top":
        return "Melhor pesquisa líquida publicada até agora. Serve como referência de longo prazo, não como carteira diária."
    if mode_key == "research_oos":
        return "Melhor candidata fora da amostra, para medir robustez sem depender só do trecho bonito."
    if "criticality" in candidate_id:
        return "Usa criticidade estrutural para distinguir oportunidade real de histeria geral do mercado."
    return "Modo de comparação entre cenários do laboratório."


def human_shadow_forecast(mode_key: str, recommended_live_mode: dict[str, Any], mode_confidence: dict[str, Any]) -> str:
    mode_name = str(recommended_live_mode.get("mode") or "").strip().lower()
    confidence = str(
        recommended_live_mode.get("confidence_level")
        or mode_confidence.get("confidence_level")
        or ""
    ).strip().lower()
    if mode_key in {"mode_attack", "mode_attack_guard"}:
        if "ataque" in mode_name and confidence in {"alta", "high", "média", "media", "medium"}:
            return "Se a leitura continuar limpa, este modo segue como candidato natural para atacar."
        return "Só vale acelerar aqui quando a confiança voltar a subir e a vigilância parar de pedir cautela."
    if mode_key in {"mode_main", "mode_main_guard"}:
        if "prote" in mode_name or "principal" in mode_name:
            return "Hoje este modo parece mais próximo do uso real, porque o ambiente pede mais disciplina."
        return "Este modo tende a assumir o controle quando o ataque perde clareza."
    if mode_key in {"paper_live", "paper_replay"}:
        return "Serve para acompanhar se a lógica continua coerente sem precisar colocar dinheiro real."
    return "Use este modo como comparação e contexto, não como ordem automática."


def parse_weights(value: Any) -> dict[str, float]:
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items() if to_num(v) is not None}
    raw = str(value or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(k): float(v) for k, v in parsed.items() if to_num(v) is not None}


def weights_preview(weights: dict[str, float], limit: int = 4) -> tuple[list[dict[str, Any]], str]:
    if not weights:
        return [], ""
    rows = sorted(
        [{"asset": asset, "weight": weight} for asset, weight in weights.items()],
        key=lambda row: abs(float(row["weight"])),
        reverse=True,
    )
    preview = rows[:limit]
    text = ", ".join(f"{row['asset']} {row['weight'] * 100:.0f}%" for row in preview)
    return rows, text


def describe_last_action(mode_key: str, payload: dict[str, Any]) -> str:
    weights = parse_weights(payload.get("weights"))
    latest_source = payload.get("latest_source") if isinstance(payload.get("latest_source"), dict) else {}
    executed_weights = parse_weights(latest_source.get("executed_weights_json"))
    executed_assets = str(latest_source.get("executed_assets") or latest_source.get("selected_assets") or "").strip()
    gross_exposure = to_num(payload.get("gross_exposure")) or to_num(latest_source.get("core_gross_exposure"))
    target_exposure = to_num(payload.get("latest_target_exposure")) or to_num(payload.get("target_exposure"))

    source_weights = executed_weights or weights
    _, preview = weights_preview(source_weights)
    if preview:
        prefix = "Última carteira executada" if executed_weights else "Carteira alvo"
        if gross_exposure is not None:
            return f"{prefix}: {preview}. Exposição total em torno de {gross_exposure * 100:.0f}%."
        return f"{prefix}: {preview}."
    if executed_assets:
        assets = [item.strip() for item in executed_assets.split(",") if item.strip()]
        shown = ", ".join(assets[:4])
        suffix = "..." if len(assets) > 4 else ""
        return f"Última seleção executada: {shown}{suffix}."
    if target_exposure is not None:
        return f"Último ajuste conhecido: exposição alvo perto de {target_exposure * 100:.0f}%."
    if mode_key == "paper_live":
        return "Última ação: o paper trading ajustou a exposição virtual e continuou acumulando sem ordem real."
    if mode_key == "paper_replay":
        return "Última ação: replay histórico atualizado para comparar o caminho atual com o histórico completo."
    return "Sem detalhe operacional fino publicado neste modo."


def build_shadow_modes(
    operation_agent: dict[str, Any],
    recommended_live_mode: dict[str, Any],
    mode_confidence: dict[str, Any],
    canonical_shadow: dict[str, Any],
    invest_shadow: dict[str, Any],
    profit_registry: dict[str, Any],
) -> list[dict[str, Any]]:
    shadow_modes: list[dict[str, Any]] = []

    def append_mode(
        *,
        slug: str,
        label: str,
        source: str,
        payload: dict[str, Any],
        candidate_id: str = "",
        status: str = "running",
        accumulating: bool = True,
        what_it_is: str = "",
        what_varies: str = "",
        metrics_source: dict[str, Any] | None = None,
    ) -> None:
        metrics = metrics_source or payload
        weights = parse_weights(payload.get("weights"))
        if not weights:
            latest_source = payload.get("latest_source") if isinstance(payload.get("latest_source"), dict) else {}
            weights = parse_weights(latest_source.get("executed_weights_json"))
        weights_rows, weights_text = weights_preview(weights)
        latest_date = str(
            payload.get("latest_date")
            or payload.get("latest_signal_date")
            or payload.get("effective_date")
            or payload.get("signal_date")
            or payload.get("price_date")
            or ""
        )
        shadow_modes.append(
            {
                "slug": slug,
                "label": label,
                "source": source,
                "status": status,
                "running": status == "running",
                "accumulating": accumulating,
                "candidate_id": candidate_id or str(payload.get("candidate_id") or ""),
                "what_it_is": what_it_is,
                "what_varies": what_varies,
                "latest_date": latest_date,
                "last_action": describe_last_action(slug, payload),
                "forecast": human_shadow_forecast(slug, recommended_live_mode, mode_confidence),
                "net_ann_return": to_num(metrics.get("net_ann_return")) or to_num(metrics.get("daily_ann_return")),
                "net_total_return": to_num(metrics.get("net_total_return")) or to_num(metrics.get("daily_total_return")),
                "net_sharpe": to_num(metrics.get("net_sharpe")) or to_num(metrics.get("daily_sharpe")) or to_num(metrics.get("sharpe")),
                "net_max_drawdown": to_num(metrics.get("net_max_drawdown")) or to_num(metrics.get("daily_max_drawdown")) or to_num(metrics.get("max_drawdown")),
                "gross_exposure": to_num(payload.get("gross_exposure")) or to_num(payload.get("latest_target_exposure")),
                "weights": weights_rows[:12],
                "weights_preview": weights_text,
                "notes": str(payload.get("notes") or ""),
            }
        )

    for mode_key in ["mode_attack", "mode_main", "mode_attack_guard", "mode_main_guard"]:
        payload = operation_agent.get(mode_key)
        if not isinstance(payload, dict):
            continue
        candidate_id = str(payload.get("candidate_id") or "")
        append_mode(
            slug=mode_key,
            label=human_shadow_label(mode_key, candidate_id, str(payload.get("label") or "")),
            source="daily_operation_agent",
            payload=payload,
            candidate_id=candidate_id,
            status="running",
            accumulating=True,
            what_it_is=str(payload.get("label") or human_shadow_label(mode_key, candidate_id)),
            what_varies=human_shadow_variant(mode_key, candidate_id),
        )

    candidates = canonical_shadow.get("candidates")
    if isinstance(candidates, list):
        for row in candidates[:2]:
            if not isinstance(row, dict):
                continue
            candidate_key = str(row.get("candidate_key") or "")
            append_mode(
                slug=f"canonical_{candidate_key}",
                label=human_shadow_label(f"canonical_{candidate_key}", str(row.get("profile_name") or ""), str(row.get("profile_name") or "")),
                source="legacy_shadow_canonical",
                payload={
                    "latest_date": str(((row.get("latest_signal") or {}) if isinstance(row.get("latest_signal"), dict) else {}).get("ym") or ""),
                    "latest_source": row.get("latest_signal") if isinstance(row.get("latest_signal"), dict) else {},
                    "notes": "Referência histórica do shadow antigo para comparar a geração atual.",
                },
                candidate_id=str(row.get("profile_name") or ""),
                status="historical",
                accumulating=False,
                what_it_is="Modo legado do shadow que ainda serve para comparar a geração atual com a fase anterior.",
                what_varies=human_shadow_variant(f"canonical_{candidate_key}", str(row.get("profile_name") or "")),
                metrics_source=row,
            )

    latest = invest_shadow.get("latest") if isinstance(invest_shadow.get("latest"), dict) else {}
    live = invest_shadow.get("live") if isinstance(invest_shadow.get("live"), dict) else {}
    live_portfolio = live.get("portfolio") if isinstance(live.get("portfolio"), dict) else {}
    append_mode(
        slug="paper_live",
        label=human_shadow_label("paper_live"),
        source="invest_shadow_live",
        payload={
            "latest_date": live.get("latest_signal_date") or latest.get("signal_date"),
            "latest_target_exposure": live.get("latest_target_exposure") or latest.get("target_exposure"),
            "notes": "Paper trading diário com capital virtual e comparação contra benchmark.",
        },
        status="running",
        accumulating=True,
        what_it_is="Teste diário com capital virtual, para ver se a lógica de hoje continua fazendo sentido fora do backtest.",
        what_varies=human_shadow_variant("paper_live"),
        metrics_source={
            "net_ann_return": live_portfolio.get("ann_return"),
            "net_total_return": live_portfolio.get("total_return"),
            "net_sharpe": live_portfolio.get("sharpe"),
            "net_max_drawdown": live_portfolio.get("max_drawdown"),
        },
    )

    historical = invest_shadow.get("historical_proxy_replay") if isinstance(invest_shadow.get("historical_proxy_replay"), dict) else {}
    append_mode(
        slug="paper_replay",
        label=human_shadow_label("paper_replay"),
        source="invest_shadow_replay",
        payload={
            "latest_date": historical.get("end_date"),
            "latest_target_exposure": historical.get("latest_target_exposure"),
            "notes": "Replay histórico do mesmo paper trading para ter uma linha longa de comparação.",
        },
        status="historical",
        accumulating=False,
        what_it_is="Replay histórico congelado do paper trading, para comparar o comportamento diário com um caminho longo.",
        what_varies=human_shadow_variant("paper_replay"),
        metrics_source={
            "net_ann_return": historical.get("ann_return"),
            "net_total_return": historical.get("total_return"),
            "net_sharpe": historical.get("sharpe"),
            "net_max_drawdown": historical.get("max_drawdown"),
        },
    )

    top_candidate = profit_registry.get("top_candidate") if isinstance(profit_registry.get("top_candidate"), dict) else {}
    if top_candidate:
        append_mode(
            slug="research_top",
            label=human_shadow_label("research_top", str(top_candidate.get("candidate_id") or ""), str(top_candidate.get("label") or "")),
            source="profit_research_registry",
            payload={
                "latest_date": str(top_candidate.get("generated_at_utc") or ""),
                "notes": "Melhor candidata de pesquisa líquida publicada até agora.",
            },
            candidate_id=str(top_candidate.get("candidate_id") or ""),
            status="research",
            accumulating=False,
            what_it_is="Melhor pesquisa líquida encontrada até agora no laboratório, sem obrigação de ser a carteira diária.",
            what_varies=human_shadow_variant("research_top", str(top_candidate.get("candidate_id") or "")),
            metrics_source=top_candidate,
        )

    oos_best = profit_registry.get("oos_best_consistent")
    if isinstance(oos_best, dict) and oos_best:
        append_mode(
            slug="research_oos",
            label=human_shadow_label("research_oos", str(oos_best.get("candidate_id") or "")),
            source="profit_research_registry",
            payload={
                "latest_date": str(oos_best.get("generated_at_utc") or ""),
                "notes": "Melhor candidata fora da amostra, útil para medir robustez sem depender só do trecho bonito.",
            },
            candidate_id=str(oos_best.get("candidate_id") or ""),
            status="research",
            accumulating=False,
            what_it_is="Melhor candidata fora da amostra. Serve para medir consistência histórica, não execução diária.",
            what_varies=human_shadow_variant("research_oos", str(oos_best.get("candidate_id") or "")),
            metrics_source={
                "net_ann_return": oos_best.get("mean_test_net_ann_return"),
                "net_total_return": oos_best.get("mean_test_edge_vs_spy"),
                "net_sharpe": oos_best.get("mean_test_net_sharpe"),
                "net_max_drawdown": None,
            },
        )

    return shadow_modes


def build_snapshot() -> dict[str, Any]:
    finance_ready, lab_run_dir = latest_lab_from_finance_ready()
    lab_summary = read_json((lab_run_dir / "summary.json") if lab_run_dir else Path("missing"), {})
    lab_timeseries = read_csv((lab_run_dir / "macro_timeseries_T120.csv") if lab_run_dir else Path("missing"))
    lab_sector_diag = read_csv((lab_run_dir / "sector_regime_diagnostics.csv") if lab_run_dir else Path("missing"))
    lab_asset_diag = read_csv((lab_run_dir / "asset_regime_diagnostics.csv") if lab_run_dir else Path("missing"))
    action_playbook = read_json((lab_run_dir / "action_playbook_T120.json") if lab_run_dir else Path("missing"), [])

    profit_registry = read_json(RESULTS_ROOT / "ops" / "profit_research" / "latest_registry.json", {})
    profit_patterns = read_json(RESULTS_ROOT / "ops" / "profit_research" / "latest_patterns.json", {})
    invest_shadow = read_json(RESULTS_ROOT / "ops" / "invest_shadow" / "latest_summary.json", {})
    canonical_shadow = read_json(RESULTS_ROOT / "ops" / "profit_shadow_target_800_attack" / "canonical_latest_run.json", {})

    layered_summary = latest_json_summary("profit_layered_engine_suite")
    drawdown_summary = latest_json_summary("profit_drawdown_control_suite")
    equity_summary = latest_json_summary("profit_equity_improvement_suite")
    meta_injection_summary = latest_json_summary("profit_meta_equity_injection")
    crypto_10x_summary = latest_json_summary("profit_10x_rule_search_crypto_plus")
    crypto_winner_frequency = latest_csv_rows("profit_10x_rule_search_crypto_plus", "winner_asset_frequency.csv")
    universe_expansion_verdict = read_json(
        RESULTS_ROOT / "validation" / "universe_expansion_pack_compare" / "final_structural_verdict.json",
        {},
    )
    group_method_summary = latest_json_summary("profit_group_methodology_suite")
    group_oos_summary = latest_json_summary("profit_group_oos_validation")
    operation_agent = read_json(RESULTS_ROOT / "ops" / "agents" / "daily_operation" / "latest_summary.json", {})
    vigilance_agent = read_json(RESULTS_ROOT / "ops" / "agents" / "daily_vigilance" / "latest_summary.json", {})
    operation_confidence = operation_agent.get("mode_confidence", {}) if isinstance(operation_agent, dict) else {}
    recommended_live_mode = operation_agent.get("recommended_live_mode", {}) if isinstance(operation_agent, dict) else {}
    group_top_candidates = group_method_summary.get("top_candidates", {}) if isinstance(group_method_summary, dict) else {}
    group_best_net_blended = group_top_candidates.get("best_net_blended", {}) if isinstance(group_top_candidates, dict) else {}
    group_oos_best_list = group_oos_summary.get("best_consistent_candidates", {}) if isinstance(group_oos_summary, dict) else {}
    group_oos_best = first_list_item(group_oos_best_list)
    universe_summary = universe_expansion_verdict.get("summary", {}) if isinstance(universe_expansion_verdict, dict) else {}
    universe_verdict = universe_expansion_verdict.get("verdict", {}) if isinstance(universe_expansion_verdict, dict) else {}

    latest_lab_row = lab_timeseries[-1] if lab_timeseries else {}
    latest_playbook = action_playbook[-1] if isinstance(action_playbook, list) and action_playbook else {}
    gate = lab_summary.get("deployment_gate", {}) if isinstance(lab_summary, dict) else {}

    sector_pressure = []
    for row in lab_sector_diag:
      risk_mean = to_num(row.get("risk_mean"))
      conf_mean = to_num(row.get("confidence_mean"))
      n_assets = to_num(row.get("n_assets"))
      if risk_mean is None:
          continue
      sector_pressure.append(
          {
              "sector": str(row.get("sector") or "").strip(),
              "risk_mean": risk_mean,
              "confidence_mean": conf_mean,
              "n_assets": n_assets,
              "alert": str(row.get("alerta_setor") or "").strip().lower(),
              "plan": str(row.get("plano_acao") or "").strip(),
              "impact_score": risk_mean * (0.5 + max(0.0, conf_mean or 0.0) * 0.5),
          }
      )
    sector_pressure.sort(key=lambda row: row["impact_score"], reverse=True)

    current_universe = []
    for row in lab_asset_diag:
        ticker = str(row.get("ticker") or "").strip()
        if not ticker:
            continue
        confidence = to_num(row.get("confidence_score")) or 0.0
        quality = to_num(row.get("stability_score"))
        risk_score = to_num(row.get("risk_score"))
        signal_status = signal_status_from_regime(str(row.get("regime_asset") or ""))
        current_universe.append(
            {
                "asset": ticker,
                "domain": "finance",
                "timestamp": str(finance_ready.get("data_last_date") or latest_lab_row.get("date") or ""),
                "run_id": str(lab_summary.get("run_id") or ""),
                "data_adequacy": "ok",
                "source_type": "lab_corr_asset_diag",
                "regime": str(row.get("regime_asset") or "").upper(),
                "confidence": confidence,
                "quality": quality if quality is not None else max(0.0, 1.0 - (risk_score or 0.5)),
                "instability_score": risk_score or 0.0,
                "status": signal_status,
                "signal_status": signal_status,
                "reason": str(row.get("sector") or "").strip(),
                "risk_truth_status": signal_status,
                "group": str(row.get("sector") or "").strip(),
                "risk_score": risk_score,
                "sector": str(row.get("sector") or "").strip(),
            }
        )
    current_universe.sort(
        key=lambda row: (
            {"validated": 0, "watch": 1, "inconclusive": 2}.get(str(row["signal_status"]), 3),
            -float(row["confidence"]),
            str(row["asset"]),
        )
    )

    crypto_watchlist = []
    for row in crypto_winner_frequency[:20]:
        ticker = str(row.get("ticker") or "").strip()
        rebalance_count = to_num(row.get("rebalance_count"))
        if not ticker:
            continue
        crypto_watchlist.append(
            {
                "asset": ticker,
                "domain": "crypto",
                "timestamp": str(finance_ready.get("data_last_date") or latest_lab_row.get("date") or ""),
                "run_id": str(crypto_10x_summary.get("outdir") or ""),
                "data_adequacy": "ok",
                "source_type": "crypto_research",
                "regime": "WATCH",
                "confidence": min(0.99, 0.35 + ((rebalance_count or 0.0) / 400.0)),
                "quality": 0.65,
                "instability_score": 0.45,
                "status": "watch",
                "signal_status": "watch",
                "reason": "cripto_liquido_no_vencedor_10x",
                "risk_truth_status": "watch",
                "group": "crypto",
                "risk_score": 0.45,
                "sector": "crypto",
                "rebalance_count": rebalance_count,
            }
        )

    merged_universe = {}
    for row in current_universe + crypto_watchlist:
        key = f"{row['asset']}__{row['domain']}"
        merged_universe[key] = row
    current_universe = list(merged_universe.values())
    current_universe.sort(
        key=lambda row: (
            {"validated": 0, "watch": 1, "inconclusive": 2}.get(str(row["signal_status"]), 3),
            -float(row["confidence"]),
            str(row["asset"]),
        )
    )

    top_candidate = profit_registry.get("top_candidate", {}) if isinstance(profit_registry, dict) else {}
    oos_best = profit_registry.get("oos_best_consistent", {}) if isinstance(profit_registry, dict) else {}
    attack_mode_from_registry = select_attack_mode_from_registry(profit_registry)
    shadow_latest = invest_shadow.get("latest", {}) if isinstance(invest_shadow, dict) else {}
    shadow_replay = invest_shadow.get("historical_proxy_replay", {}) if isinstance(invest_shadow, dict) else {}
    shadow_portfolio = shadow_replay.get("portfolio", {}) if isinstance(shadow_replay, dict) else {}
    shadow_live = invest_shadow.get("live", {}) if isinstance(invest_shadow, dict) else {}
    shadow_live_portfolio = shadow_live.get("portfolio", {}) if isinstance(shadow_live, dict) else {}
    shadow_modes = build_shadow_modes(
        operation_agent if isinstance(operation_agent, dict) else {},
        recommended_live_mode if isinstance(recommended_live_mode, dict) else {},
        operation_confidence if isinstance(operation_confidence, dict) else {},
        canonical_shadow if isinstance(canonical_shadow, dict) else {},
        invest_shadow if isinstance(invest_shadow, dict) else {},
        profit_registry if isinstance(profit_registry, dict) else {},
    )
    best_shadow_by_return = max(
        shadow_modes,
        key=lambda row: to_num(row.get("net_ann_return")) if to_num(row.get("net_ann_return")) is not None else -1e18,
        default={},
    )
    best_shadow_by_drawdown = max(
        shadow_modes,
        key=lambda row: to_num(row.get("net_max_drawdown")) if to_num(row.get("net_max_drawdown")) is not None else -1e18,
        default={},
    )

    volume_supported_assets, volume_total_assets = count_volume_support(current_universe)

    snapshot = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "as_of_date": str(finance_ready.get("data_last_date") or latest_lab_row.get("date") or ""),
        "sources": {
            "finance_product_ready": str(RESULTS_ROOT / "ops" / "finance_product_ready" / "latest_finance_product_ready.json"),
            "profit_registry": str(RESULTS_ROOT / "ops" / "profit_research" / "latest_registry.json"),
            "profit_patterns": str(RESULTS_ROOT / "ops" / "profit_research" / "latest_patterns.json"),
            "invest_shadow": str(RESULTS_ROOT / "ops" / "invest_shadow" / "latest_summary.json"),
            "lab_run_dir": str(lab_run_dir or ""),
        },
        "finance": {
            "overall_readiness": finance_ready.get("overall_readiness"),
            "data_last_date": finance_ready.get("data_last_date"),
            "operational_state": finance_ready.get("operational_state"),
            "risk_level_next_month": finance_ready.get("risk_level_next_month"),
            "confidence_score": finance_ready.get("confidence_score"),
            "mode_confidence": operation_confidence,
            "recommended_live_mode": recommended_live_mode,
            "lab_run_id": lab_summary.get("run_id"),
            "gate_blocked": gate.get("blocked"),
            "gate_reasons": gate.get("reasons") or [],
            "latest_state": latest_lab_row,
            "latest_playbook": latest_playbook,
        },
        "profit_research": {
            "rows_total": profit_registry.get("rows_total"),
            "status_counts": profit_registry.get("status_counts"),
            "top_candidate": top_candidate,
            "oos_best_consistent": oos_best,
            "insights": profit_registry.get("insights") or [],
            "pattern_headlines": profit_patterns.get("pattern_headlines") or [],
            "hypothesis_headlines": profit_patterns.get("hypothesis_headlines") or [],
            "hypotheses": profit_patterns.get("hypotheses") or [],
            "event_count": profit_patterns.get("event_count") or 0,
        },
        "agents": {
            "daily_operation": operation_agent,
            "daily_vigilance": vigilance_agent,
        },
        "confidence": {
            "recommended_live_mode": recommended_live_mode,
            "mode_confidence": operation_confidence,
            "vigilance_status": vigilance_agent.get("status"),
            "vigilance_alerts": vigilance_agent.get("alerts") or [],
        },
        "shadow": {
            "run_id": invest_shadow.get("run_id"),
            "latest": shadow_latest,
            "live": {
                "status": shadow_live.get("status"),
                "start_date": shadow_live.get("start_date"),
                "end_date": shadow_live.get("end_date"),
                "n_days": shadow_live.get("n_days"),
                "capital_start": shadow_live.get("capital_start"),
                "capital_end": shadow_live.get("capital_end"),
                "latest_target_exposure": shadow_live.get("latest_target_exposure"),
                "latest_executed_exposure": shadow_live.get("latest_executed_exposure"),
                "latest_regime": shadow_live.get("latest_regime"),
                "latest_signal_date": shadow_live.get("latest_signal_date"),
                "edge_vs_benchmark_total_return": shadow_live.get("edge_vs_benchmark_total_return"),
                "portfolio": {
                    "total_return": shadow_live_portfolio.get("total_return"),
                    "ann_return": shadow_live_portfolio.get("ann_return"),
                    "ann_vol": shadow_live_portfolio.get("ann_vol"),
                    "sharpe": shadow_live_portfolio.get("sharpe"),
                    "max_drawdown": shadow_live_portfolio.get("max_drawdown"),
                },
            },
            "historical_proxy_replay": {
                "status": shadow_replay.get("status"),
                "start_date": shadow_replay.get("start_date"),
                "end_date": shadow_replay.get("end_date"),
                "n_days": shadow_replay.get("n_days"),
                "capital_start": shadow_replay.get("capital_start"),
                "capital_end": shadow_replay.get("capital_end"),
                "latest_target_exposure": shadow_replay.get("latest_target_exposure"),
                "latest_executed_exposure": shadow_replay.get("latest_executed_exposure"),
                "latest_regime": shadow_replay.get("latest_regime"),
                "latest_signal_date": shadow_replay.get("latest_signal_date"),
                "ann_return": shadow_portfolio.get("ann_return"),
                "sharpe": shadow_portfolio.get("sharpe"),
                "max_drawdown": shadow_portfolio.get("max_drawdown"),
                "total_return": shadow_portfolio.get("total_return"),
                "edge_vs_benchmark_total_return": shadow_replay.get("edge_vs_benchmark_total_return"),
            },
        },
        "shadow_modes": shadow_modes,
        "shadow_mode_overview": {
            "total": len(shadow_modes),
            "running": len([row for row in shadow_modes if row.get("running") is True]),
            "accumulating": len([row for row in shadow_modes if row.get("accumulating") is True]),
            "best_by_return": {
                "label": best_shadow_by_return.get("label"),
                "net_ann_return": best_shadow_by_return.get("net_ann_return"),
            },
            "best_by_drawdown": {
                "label": best_shadow_by_drawdown.get("label"),
                "net_max_drawdown": best_shadow_by_drawdown.get("net_max_drawdown"),
            },
        },
        "layered_engine": {
            "best_meta_candidate": attack_mode_from_registry or layered_summary.get("best_meta_candidate"),
            "drawdown_best_balanced": drawdown_summary.get("best_balanced_candidate"),
            "equity_best_overall": equity_summary.get("best_overall_candidate"),
            "meta_equity_winner": meta_injection_summary.get("winner"),
            "best_crypto_rule": crypto_10x_summary.get("best_hit_rate_candidate"),
        },
        "proof": {
            "history_window": {
                "start": (
                    (universe_summary.get("target_800_cov075") or {}).get("period_start")
                    or group_best_net_blended.get("period_start")
                    or "2016-02-18"
                ),
                "end": str(finance_ready.get("data_last_date") or latest_lab_row.get("date") or ""),
            },
            "attack_mode": attack_mode_from_registry or layered_summary.get("best_meta_candidate") or {},
            "robust_mode": drawdown_summary.get("best_balanced_candidate") or {},
            "research_best": top_candidate,
            "research_oos": oos_best or group_oos_best,
            "group_suite_best": group_best_net_blended,
            "physics_math_why": [
                "A matriz de correlacao mostra quando os ativos deixam de andar sozinhos e passam a obedecer um movimento coletivo.",
                "O espectro ajuda a medir concentracao de risco. Quando um fator domina tudo, o motor reduz confianca e corta agressividade.",
                "Momentum e persistencia medem inercia do mercado. A ideia nao e prever cada candle, e sim explorar padroes que costumam durar mais de um dia.",
                "O meta-switch junta essas leituras para escolher entre ataque, robustez e caixa sem fingir certeza absoluta.",
            ],
        },
        "universe_expansion": {
            "verdict": universe_verdict,
            "baseline_528": (universe_summary.get("baseline_528_cov090") or {}),
            "target_800_cov075": (universe_summary.get("target_800_cov075") or {}),
            "target_800_cov090": (universe_summary.get("target_800_cov090") or {}),
        },
        "data_quality": {
            "price_source": "yfinance_daily_bundle",
            "price_history_start": "2016-01-05",
            "price_history_end": str(finance_ready.get("data_last_date") or latest_lab_row.get("date") or ""),
            "volume_supported_assets": volume_supported_assets,
            "volume_total_assets": volume_total_assets,
            "volume_coverage_ratio": (volume_supported_assets / volume_total_assets) if volume_total_assets else 0.0,
            "volume_note": "Os CSVs publicados hoje nao trazem volume. O site deve usar confianca, drawdown e volatilidade em vez de fingir volume.",
        },
        "charts": {
            "sector_pressure": sector_pressure[:8],
            "asset_watchlist": current_universe[:40],
            "crypto_watchlist": crypto_watchlist[:12],
            "allocation_mix": [
                {"label": "Risco alvo", "value": latest_playbook.get("exposure")},
                {
                    "label": "Caixa alvo",
                    "value": (1 - float(latest_playbook.get("exposure"))) if to_num(latest_playbook.get("exposure")) is not None else None,
                },
            ],
        },
        "current_universe": current_universe[:500],
    }
    return snapshot


def build_dashboard_overview(snapshot: dict[str, Any]) -> dict[str, Any]:
    universe = snapshot.get("current_universe", [])
    total = len(universe) or 1
    validated = len([row for row in universe if row.get("signal_status") == "validated"])
    watch = len([row for row in universe if row.get("signal_status") == "watch"])
    groups = []
    for row in snapshot.get("charts", {}).get("sector_pressure", []):
        groups.append(
            {
                "group": row.get("sector"),
                "mean_mase": row.get("risk_mean"),
                "mean_dir_acc": row.get("confidence_mean"),
            }
        )
    return {
        "status": "ok",
        "generated_at_utc": snapshot.get("generated_at_utc"),
        "summary_cards": {
            "pct_assets_mase_lt_1": validated / total,
            "pct_assets_dir_acc_gt_052": (validated + watch) / total,
        },
        "groups": groups,
        "source": "site_finance_snapshot",
    }


def bundle_price_history(snapshot: dict[str, Any]) -> dict[str, Any]:
    public_root = REPO_ROOT / "website-ui" / "public" / "data" / "raw" / "finance" / "yfinance_daily"
    public_root.mkdir(parents=True, exist_ok=True)

    required_assets = {
        "SPY",
        "QQQ",
        "IWM",
        "TLT",
        "GLD",
        "BTC-USD",
        "ETH-USD",
        "SOL-USD",
        "BNB-USD",
        "XRP-USD",
        "LINK-USD",
    }
    for row in snapshot.get("current_universe", []):
        asset = str((row or {}).get("asset") or "").strip()
        if asset:
            required_assets.add(asset)

    copied: list[str] = []
    missing: list[str] = []
    source_root = REPO_ROOT / "data" / "raw" / "finance" / "yfinance_daily"
    for asset in sorted(required_assets):
        src = source_root / f"{asset}.csv"
        dst = public_root / f"{asset}.csv"
        if src.exists():
            shutil.copyfile(src, dst)
            copied.append(asset)
        else:
            missing.append(asset)

    return {
        "copied_count": len(copied),
        "missing_count": len(missing),
        "copied_assets": copied,
        "missing_assets": missing,
    }


def main() -> None:
    snapshot = deep_clean(build_snapshot())
    dashboard_overview = deep_clean(build_dashboard_overview(snapshot))

    site_root = RESULTS_ROOT / "ops" / "site_data"
    site_root.mkdir(parents=True, exist_ok=True)
    dashboard_root = RESULTS_ROOT / "dashboard"
    dashboard_root.mkdir(parents=True, exist_ok=True)
    public_root = REPO_ROOT / "website-ui" / "public" / "data" / "site"
    public_root.mkdir(parents=True, exist_ok=True)
    bundle_manifest = bundle_price_history(snapshot)

    latest_site = site_root / "latest_site_snapshot.json"
    latest_site.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    (dashboard_root / "overview.json").write_text(
        json.dumps(dashboard_overview, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (public_root / "latest_site_snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (public_root / "price_bundle_manifest.json").write_text(
        json.dumps(bundle_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(str(latest_site))


if __name__ == "__main__":
    main()
