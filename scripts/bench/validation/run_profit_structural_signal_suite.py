#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.portfolio.exogenous_features import (  # noqa: E402
    build_exogenous_feature_panel,
    build_structural_stress_signal,
    feature_spectral_extremes,
)
from engine.structural.run_manifest import write_run_manifest  # noqa: E402
from scripts.bench.validation.run_profit_alpha_improvement_suite import _safe_float, _write_json  # noqa: E402
from scripts.bench.validation.run_profit_exogenous_feature_suite import (  # noqa: E402
    _build_base_fast_entry,
    _combo_bundle,
    _ecosystem_compare,
    _monthly_spectral_panel,
    _variant_bundle,
)
from scripts.bench.validation.run_profit_exogenous_feature_suite import _result_row  # noqa: E402
from scripts.bench.validation.run_profit_investment_yearbook import _calendar_rows  # noqa: E402
from scripts.bench.validation.run_profit_sector_pressure_suite import _research_row  # noqa: E402
from scripts.bench.validation.run_profit_alpha_hardening_suite import _build_candidates  # noqa: E402


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _year_improvement_table(baseline_rows: pd.DataFrame, best_rows: pd.DataFrame) -> pd.DataFrame:
    if baseline_rows.empty or best_rows.empty:
        return pd.DataFrame()
    base = baseline_rows.set_index("year")
    best = best_rows.set_index("year")
    idx = base.index.intersection(best.index)
    out = pd.DataFrame(index=idx)
    out["baseline_value_brl"] = pd.to_numeric(base.loc[idx, "ending_capital_brl"], errors="coerce")
    out["best_value_brl"] = pd.to_numeric(best.loc[idx, "ending_capital_brl"], errors="coerce")
    out["delta_brl"] = out["best_value_brl"] - out["baseline_value_brl"]
    denom = out["baseline_value_brl"].replace(0.0, pd.NA)
    out["delta_pct_vs_baseline"] = ((out["best_value_brl"] / denom) - 1.0) * 100.0
    out = out.reset_index()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Testa sinais de transição crítica, crowding e estresse estrutural sobre o ataque atual.")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--capital-brl", type=float, default=1000.0)
    ap.add_argument("--outdir-root", default="results/validation/profit_structural_signal_suite")
    args = ap.parse_args()

    outdir = (ROOT / args.outdir_root / _run_id()).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    prices_dir = (ROOT / args.prices_dir).resolve()

    built = _build_candidates(
        prices_dir=prices_dir,
        crypto_groups=(ROOT / args.crypto_asset_groups).resolve(),
        crypto_meta=(ROOT / args.crypto_asset_metadata).resolve(),
        equity_groups=(ROOT / args.equity_asset_groups).resolve(),
        equity_meta=(ROOT / args.equity_asset_metadata).resolve(),
        benchmark_crypto=str(args.benchmark_crypto),
        benchmark_equity=str(args.benchmark_equity),
    )
    context = dict(built["context"])
    protect_alloc = built["allocations"]["baseline_guard"]
    base = _build_base_fast_entry(context=context, protect_alloc=protect_alloc)

    exo = build_exogenous_feature_panel(
        prices_dir=prices_dir,
        crypto_returns=context["crypto_returns"],
        crypto_prices=context["crypto_prices"],
        benchmark_crypto=str(args.benchmark_crypto),
        macro_index=base["raw_score"].index,
    )
    spectral_panel = _monthly_spectral_panel(context=context, prices_dir=prices_dir)
    structural_stress = build_structural_stress_signal(spectral_panel=spectral_panel, index=base["raw_score"].index)
    panel = exo.panel.copy()
    panel["structural_stress"] = pd.to_numeric(structural_stress.reindex(panel.index), errors="coerce").fillna(0.5).clip(0.0, 1.0)

    specs = [
        ("critical_slowing_down", "critical_slowing_down", "penalty", 0.14, "single_structural"),
        ("crowding", "crowding", "penalty", 0.14, "single_structural"),
        ("structural_stress", "structural_stress", "penalty", 0.16, "single_structural"),
    ]
    combo_configs: dict[str, list[tuple[str, str, float]]] = {
        "critical_plus_crowding": [
            ("critical_slowing_down", "penalty", 0.10),
            ("crowding", "penalty", 0.08),
        ],
        "critical_plus_stress": [
            ("critical_slowing_down", "penalty", 0.08),
            ("structural_stress", "penalty", 0.10),
        ],
        "all_structural": [
            ("critical_slowing_down", "penalty", 0.08),
            ("crowding", "penalty", 0.06),
            ("structural_stress", "penalty", 0.10),
        ],
    }

    results = {"baseline_fast_entry": base["result"]}
    weights = {"baseline_fast_entry": base["weight"]}
    rows = [
        _result_row(
            base["result"],
            baseline=base["result"],
            family="baseline",
            label="Ataque atual",
        )
    ]

    for cid, col, mode, weight, family in specs:
        result, attack_weight = _variant_bundle(
            candidate_id=cid,
            feature_panel=panel,
            feature_col=col,
            feature_mode=mode,
            feature_weight=weight,
            base_score=base["raw_score"],
            attack_alloc=base["attack_alloc"],
            protect_alloc=protect_alloc,
        )
        results[cid] = result
        weights[cid] = attack_weight
        rows.append(_result_row(result, baseline=base["result"], family=family, label=cid))

    for cid, adjustments in combo_configs.items():
        result, attack_weight = _combo_bundle(
            candidate_id=cid,
            panel=panel,
            base_score=base["raw_score"],
            attack_alloc=base["attack_alloc"],
            protect_alloc=protect_alloc,
            adjustments=adjustments,
        )
        results[cid] = result
        weights[cid] = attack_weight
        rows.append(_result_row(result, baseline=base["result"], family="combo_structural", label=cid))

    compare_df = pd.DataFrame(rows).sort_values(["net_total_return", "net_sharpe"], ascending=[False, False]).reset_index(drop=True)
    compare_df.to_csv(outdir / "candidate_compare.csv", index=False)

    best_row = compare_df.iloc[0].to_dict() if not compare_df.empty else {}
    best_id = str(best_row.get("candidate_id", "baseline_fast_entry"))
    best_result = results[best_id]
    best_weight = weights[best_id]

    yearbook_rows: list[dict[str, Any]] = []
    baseline_years = pd.DataFrame(_calendar_rows(result=results["baseline_fast_entry"], capital_brl=float(args.capital_brl)))
    best_years = pd.DataFrame(_calendar_rows(result=best_result, capital_brl=float(args.capital_brl)))
    yearbook_rows.extend(baseline_years.to_dict(orient="records"))
    if best_id != "baseline_fast_entry":
        yearbook_rows.extend(best_years.to_dict(orient="records"))
    pd.DataFrame(yearbook_rows).to_csv(outdir / "yearbook_reais.csv", index=False)
    year_delta = _year_improvement_table(baseline_years, best_years)
    year_delta.to_csv(outdir / "year_improvement.csv", index=False)

    spectral_effects = feature_spectral_extremes(
        feature_panel=panel,
        spectral_panel=spectral_panel,
        feature_cols=["critical_slowing_down", "crowding", "structural_stress"],
    )
    spectral_effects.to_csv(outdir / "feature_spectral_effects.csv", index=False)
    ecosystem_df = _ecosystem_compare(
        spectral_panel=spectral_panel,
        baseline_weight=weights["baseline_fast_entry"],
        variant_weight=best_weight,
    )
    ecosystem_df.to_csv(outdir / "ecosystem_compare.csv", index=False)
    spectral_panel.to_csv(outdir / "spectral_panel.csv", index=True)
    panel.to_csv(outdir / "structural_feature_panel.csv", index=True)

    research_rows = [
        _research_row(
            base["result"],
            outdir=ROOT / "results" / "validation",
            status="keep",
            methodology="attack_fast_entry_baseline",
            label="Ataque atual",
        )
    ]
    if best_id != "baseline_fast_entry":
        research_rows.append(
            _research_row(
                best_result,
                outdir=ROOT / "results" / "validation",
                status="watch",
                methodology="structural_signal_overlay",
                label=f"Sinal estrutural: {best_id}",
            )
        )
    pd.DataFrame(research_rows).to_csv(outdir / "research_rows.csv", index=False)

    summary = {
        "suite": "profit_structural_signal_suite",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline": rows[0],
        "best_overall": best_row,
        "best_candidate_id": best_id,
        "worth_keeping": bool(best_id != "baseline_fast_entry" and _safe_float(best_result.net_total_return) > _safe_float(base["result"].net_total_return)),
        "year_improvement_mean_pct": _safe_float(year_delta["delta_pct_vs_baseline"].mean()) if not year_delta.empty else None,
        "files": {
            "candidate_compare_csv": str(outdir / "candidate_compare.csv"),
            "yearbook_reais_csv": str(outdir / "yearbook_reais.csv"),
            "year_improvement_csv": str(outdir / "year_improvement.csv"),
            "feature_spectral_effects_csv": str(outdir / "feature_spectral_effects.csv"),
            "ecosystem_compare_csv": str(outdir / "ecosystem_compare.csv"),
            "structural_feature_panel_csv": str(outdir / "structural_feature_panel.csv"),
        },
    }
    _write_json(outdir / "summary.json", summary)
    write_run_manifest(
        outdir,
        script="scripts/bench/validation/run_profit_structural_signal_suite.py",
        params={
            "crypto_asset_groups": str(args.crypto_asset_groups),
            "crypto_asset_metadata": str(args.crypto_asset_metadata),
            "equity_asset_groups": str(args.equity_asset_groups),
            "equity_asset_metadata": str(args.equity_asset_metadata),
            "prices_dir": str(args.prices_dir),
            "benchmark_crypto": str(args.benchmark_crypto),
            "benchmark_equity": str(args.benchmark_equity),
            "capital_brl": float(args.capital_brl),
        },
        paths={
            "summary_json": str(outdir / "summary.json"),
            "candidate_compare_csv": str(outdir / "candidate_compare.csv"),
            "yearbook_reais_csv": str(outdir / "yearbook_reais.csv"),
            "year_improvement_csv": str(outdir / "year_improvement.csv"),
            "feature_spectral_effects_csv": str(outdir / "feature_spectral_effects.csv"),
            "ecosystem_compare_csv": str(outdir / "ecosystem_compare.csv"),
            "structural_feature_panel_csv": str(outdir / "structural_feature_panel.csv"),
        },
        extra={
            "suite": "profit_structural_signal_suite",
            "best_candidate_id": str(best_result.candidate_id),
            "best_total_return": _safe_float(best_result.net_total_return),
            "best_ann_return": _safe_float(best_result.net_ann_return),
            "best_sharpe": _safe_float(best_result.net_sharpe),
            "best_max_drawdown": _safe_float(best_result.net_max_drawdown),
            "baseline_candidate_id": str(base["result"].candidate_id),
            "baseline_total_return": _safe_float(base["result"].net_total_return),
        },
        repo_root=ROOT,
    )


if __name__ == "__main__":
    main()
