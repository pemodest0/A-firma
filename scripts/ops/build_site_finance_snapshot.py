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
    shadow_latest = invest_shadow.get("latest", {}) if isinstance(invest_shadow, dict) else {}
    shadow_replay = invest_shadow.get("historical_proxy_replay", {}) if isinstance(invest_shadow, dict) else {}
    shadow_portfolio = shadow_replay.get("portfolio", {}) if isinstance(shadow_replay, dict) else {}
    shadow_live = invest_shadow.get("live", {}) if isinstance(invest_shadow, dict) else {}
    shadow_live_portfolio = shadow_live.get("portfolio", {}) if isinstance(shadow_live, dict) else {}

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
            "event_count": profit_patterns.get("event_count") or 0,
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
        "layered_engine": {
            "best_meta_candidate": layered_summary.get("best_meta_candidate"),
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
            "attack_mode": layered_summary.get("best_meta_candidate") or {},
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
