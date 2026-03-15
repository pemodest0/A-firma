#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class SourceSpec:
    suite: str
    csv_relpath: str
    sort_keys: tuple[str, ...]


@dataclass(frozen=True)
class CategorySpec:
    name: str
    sources: tuple[SourceSpec, ...]


BATTERY_AXES: dict[str, list[str]] = {
    "walkforward_2021_2025": [
        "scripts/bench/validation/run_profit_one_year_walkforward_audit.py",
        "scripts/bench/validation/run_profit_alpha_walkforward_tournament.py",
        "scripts/bench/validation/run_profit_drawdown_control_suite.py",
    ],
    "cost_stress_10_30_50bps": [
        "scripts/bench/validation/run_profit_attack_validation_suite.py",
        "scripts/bench/validation/run_profit_equity_improvement_suite.py",
        "execution/cost_model.py",
    ],
    "min_holding_7_14_21": [
        "scripts/bench/validation/run_profit_attack_entry_ranking_suite.py",
        "scripts/bench/validation/run_profit_alpha_improvement_suite.py",
        "scripts/bench/validation/run_profit_champion_selection_rotation_suite.py",
    ],
    "concentration_top1_top3": [
        "scripts/bench/validation/run_profit_official_post_fiscal_validation.py",
        "scripts/bench/validation/run_profit_u800_alpha_suite.py",
        "scripts/bench/validation/run_profit_universe_resilience_suite.py",
        "scripts/bench/validation/run_profit_champion_timing_robustness_suite.py",
    ],
    "regime_stable_transition_stress_dispersion": [
        "scripts/bench/validation/run_profit_equity_improvement_suite.py",
        "scripts/bench/validation/run_profit_regime_simulation_suite.py",
        "scripts/bench/validation/run_profit_sleeve_sizing_synthetic_suite.py",
    ],
}


CATEGORIES: tuple[CategorySpec, ...] = (
    CategorySpec(
        name="official_serious",
        sources=(
            SourceSpec(
                suite="profit_official_post_fiscal_validation_global",
                csv_relpath="results/validation/profit_official_post_fiscal_validation/20260315T002633Z/candidate_compare.csv",
                sort_keys=("net_total_return", "net_ann_return", "net_sharpe"),
            ),
            SourceSpec(
                suite="profit_official_post_fiscal_validation_brazil",
                csv_relpath="results/validation/profit_official_post_fiscal_validation_brazil/20260315T024647Z/candidate_compare.csv",
                sort_keys=("net_total_return", "net_ann_return", "net_sharpe"),
            ),
            SourceSpec(
                suite="profit_champion_drawdown_suite",
                csv_relpath="results/validation/profit_champion_drawdown_suite/20260314T052137Z/candidate_compare.csv",
                sort_keys=("net_total_return", "net_ann_return", "net_sharpe"),
            ),
        ),
    ),
    CategorySpec(
        name="attack_fast",
        sources=(
            SourceSpec(
                suite="profit_one_year_payoff_63d",
                csv_relpath="results/validation/profit_one_year_payoff_audit/20260315T065136Z/candidate_compare.csv",
                sort_keys=("median_return_252d", "hit_rate_6x_252d", "p90_return_252d"),
            ),
            SourceSpec(
                suite="profit_one_year_payoff_126d",
                csv_relpath="results/validation/profit_one_year_payoff_audit/20260315T065322Z/candidate_compare.csv",
                sort_keys=("median_return_252d", "hit_rate_6x_252d", "p90_return_252d"),
            ),
            SourceSpec(
                suite="profit_one_year_payoff_252d",
                csv_relpath="results/validation/profit_one_year_payoff_audit/20260315T041459Z/candidate_compare.csv",
                sort_keys=("median_return_252d", "hit_rate_6x_252d", "p90_return_252d"),
            ),
        ),
    ),
    CategorySpec(
        name="drawdown_control",
        sources=(
            SourceSpec(
                suite="profit_drawdown_control_suite",
                csv_relpath="results/validation/profit_drawdown_control_suite/20260307T055254Z/candidate_compare.csv",
                sort_keys=("net_sharpe", "net_ann_return", "net_total_return"),
            ),
            SourceSpec(
                suite="profit_regime_simulation_suite",
                csv_relpath="results/validation/profit_regime_simulation_suite/20260308T044624Z/candidate_compare.csv",
                sort_keys=("net_sharpe", "net_ann_return", "net_total_return"),
            ),
        ),
    ),
    CategorySpec(
        name="meta_context",
        sources=(
            SourceSpec(
                suite="profit_meta_mode_selector_suite",
                csv_relpath="results/validation/profit_meta_mode_selector_suite/20260312T233431Z/candidate_compare.csv",
                sort_keys=("net_sharpe", "net_ann_return", "net_total_return"),
            ),
            SourceSpec(
                suite="profit_layered_engine_suite_meta",
                csv_relpath="results/validation/profit_layered_engine_suite/20260307T054325Z/meta_candidate_compare.csv",
                sort_keys=("net_sharpe", "net_ann_return", "net_total_return"),
            ),
            SourceSpec(
                suite="profit_regime_simulation_suite",
                csv_relpath="results/validation/profit_regime_simulation_suite/20260308T044624Z/candidate_compare.csv",
                sort_keys=("net_sharpe", "net_ann_return", "net_total_return"),
            ),
        ),
    ),
    CategorySpec(
        name="research_turbo",
        sources=(
            SourceSpec(
                suite="profit_structural_signal_suite",
                csv_relpath="results/validation/profit_structural_signal_suite/20260309T153606Z/candidate_compare.csv",
                sort_keys=("net_total_return", "net_ann_return", "net_sharpe"),
            ),
            SourceSpec(
                suite="profit_alpha_war_suite",
                csv_relpath="results/validation/profit_alpha_war_suite/20260308T092621Z/candidate_compare.csv",
                sort_keys=("net_total_return", "net_ann_return", "net_sharpe"),
            ),
            SourceSpec(
                suite="profit_attack_entry_ranking_suite",
                csv_relpath="results/validation/profit_attack_entry_ranking_suite/20260309T095758Z/candidate_compare.csv",
                sort_keys=("net_total_return", "net_ann_return", "net_sharpe"),
            ),
            SourceSpec(
                suite="profit_crypto_resolution_suite",
                csv_relpath="results/validation/profit_crypto_resolution_suite/20260309T064200Z/candidate_compare.csv",
                sort_keys=("net_total_return", "net_ann_return", "net_sharpe"),
            ),
        ),
    ),
)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _metric_sort_tuple(row: dict[str, Any], keys: tuple[str, ...]) -> tuple[float, ...]:
    values: list[float] = []
    for key in keys:
        val = _safe_float(row.get(key))
        values.append(val if val == val else float("-inf"))
    return tuple(values)


def _candidate_name(row: dict[str, Any]) -> str:
    for key in ("candidate_id", "scenario_id", "mode_id", "name", "label"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return "unknown_candidate"


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _best_rows_for_category(spec: CategorySpec, top_n: int) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for source in spec.sources:
        csv_path = ROOT / source.csv_relpath
        rows = _read_csv_rows(csv_path)
        for raw in rows:
            candidate_id = _candidate_name(raw)
            enriched = dict(raw)
            enriched["candidate_id"] = candidate_id
            enriched["source_suite"] = source.suite
            enriched["_sort_tuple"] = _metric_sort_tuple(enriched, source.sort_keys)
            previous = merged.get(candidate_id)
            if previous is None or enriched["_sort_tuple"] > previous["_sort_tuple"]:
                merged[candidate_id] = enriched
    ranked = sorted(merged.values(), key=lambda row: row["_sort_tuple"], reverse=True)
    out: list[dict[str, Any]] = []
    for row in ranked[:top_n]:
        slim = {
            "candidate_id": row["candidate_id"],
            "source_suite": row.get("source_suite"),
            "net_ann_return": _safe_float(row.get("net_ann_return")),
            "net_total_return": _safe_float(row.get("net_total_return")),
            "net_sharpe": _safe_float(row.get("net_sharpe")),
            "net_max_drawdown": _safe_float(row.get("net_max_drawdown")),
            "median_return_252d": _safe_float(row.get("median_return_252d")),
            "hit_rate_6x_252d": _safe_float(row.get("hit_rate_6x_252d")),
            "touch_loss_50_252d": _safe_float(row.get("touch_loss_50_252d")),
            "avg_turnover_daily": _safe_float(row.get("avg_turnover_daily")),
        }
        out.append(slim)
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
    ap = argparse.ArgumentParser(description="Cataloga top 5 campeoes por categoria e mapeia cobertura da bateria de validacao.")
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--outdir-root", default="results/validation/profit_champion_battery_catalog")
    args = ap.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = ROOT / args.outdir_root / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    category_rows: dict[str, list[dict[str, Any]]] = {}
    for spec in CATEGORIES:
        rows = _best_rows_for_category(spec, top_n=int(args.top_n))
        category_rows[spec.name] = rows
        _write_csv(outdir / f"{spec.name}.csv", rows)

    summary = {
        "suite": "profit_champion_battery_catalog",
        "run_id": run_id,
        "top_n": int(args.top_n),
        "categories": category_rows,
        "battery_axes": BATTERY_AXES,
        "notes": [
            "O catalogo usa artefatos ja materializados para montar os top 5 por categoria.",
            "Walk-forward, cost stress e regime ja possuem suites dedicadas no laboratorio.",
            "Min holding e concentracao ja possuem cobertura parcial, mas ainda nao estao padronizados cross-suite para todos os campeoes.",
        ],
    }
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({"status": "ok", "outdir": str(outdir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
