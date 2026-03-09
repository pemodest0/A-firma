#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.structural.run_manifest import write_run_manifest  # noqa: E402
from scripts.bench.validation.run_profit_alpha_hardening_suite import _build_candidates  # noqa: E402
from scripts.bench.validation.run_profit_attack_validation_suite import _bootstrap_returns  # noqa: E402
from scripts.bench.validation.run_profit_frontier_expansion_suite import _write_json  # noqa: E402
from scripts.bench.validation.run_profit_historical_closure_suite import (  # noqa: E402
    _compound_periods,
    _confidence_rows,
    _frozen_row,
)
from scripts.bench.validation.run_profit_investment_yearbook import _calendar_rows  # noqa: E402
from scripts.bench.validation.run_profit_pbo_suite import _common_monthly_matrix, _pbo_for_metric  # noqa: E402
from scripts.bench.validation.run_profit_sector_pressure_suite import _research_row  # noqa: E402


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _human_label(candidate_id: str) -> str:
    mapping = {
        "alpha_attack_major8_equity25": "Ataque oficial com liquidação cripto",
        "alpha_attack_major8_equity25_legacy": "Ataque anterior",
        "meta_major8_eq_a2r1": "Modo principal",
        "meta_major8_eq_a2r1_mc_guard": "Modo principal com guarda",
    }
    return mapping.get(str(candidate_id), str(candidate_id))


def _bootstrap_summary(bundle, *, seed: int) -> dict[str, Any]:
    ret = pd.to_numeric(bundle.result.net_ret, errors="coerce").dropna().astype(float)
    bench = pd.to_numeric(bundle.result.benchmark_net_ret, errors="coerce").reindex(ret.index).fillna(0.0).astype(float)
    if ret.empty:
        return {"status": "empty"}
    monthly = pd.DataFrame({"ret": ret, "bench": bench}, index=pd.to_datetime(ret.index)).groupby(pd.to_datetime(ret.index).to_period("M")).agg(
        ret=("ret", lambda x: float(np.prod(1.0 + pd.Series(x).to_numpy(dtype=float)) - 1.0)),
        bench=("bench", lambda x: float(np.prod(1.0 + pd.Series(x).to_numpy(dtype=float)) - 1.0)),
    )
    daily_boot = _bootstrap_returns(
        returns=ret.to_numpy(dtype=float),
        benchmark=bench.to_numpy(dtype=float),
        periods_per_year=252.0,
        rng=np.random.default_rng(int(seed)),
        n_iter=800,
        sample_len=int(len(ret)),
        block_size=21,
    )
    monthly_boot = _bootstrap_returns(
        returns=pd.to_numeric(monthly["ret"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        benchmark=pd.to_numeric(monthly["bench"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        periods_per_year=12.0,
        rng=np.random.default_rng(int(seed) + 17),
        n_iter=800,
        sample_len=int(len(monthly)),
        block_size=3,
    )
    return {"daily": daily_boot, "monthly": monthly_boot}


def main() -> None:
    ap = argparse.ArgumentParser(description="Suite de promoção do novo ataque com overlay de liquidação cripto.")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--capital-brl", type=float, default=10000.0)
    ap.add_argument("--outdir-root", default="results/validation/profit_attack_promotion_suite")
    args = ap.parse_args()

    outdir = (ROOT / args.outdir_root / _run_id()).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    built = _build_candidates(
        prices_dir=(ROOT / args.prices_dir).resolve(),
        crypto_groups=(ROOT / args.crypto_asset_groups).resolve(),
        crypto_meta=(ROOT / args.crypto_asset_metadata).resolve(),
        equity_groups=(ROOT / args.equity_asset_groups).resolve(),
        equity_meta=(ROOT / args.equity_asset_metadata).resolve(),
        benchmark_crypto=str(args.benchmark_crypto),
        benchmark_equity=str(args.benchmark_equity),
    )

    attack = built["attack"]
    attack_legacy = built["allocations"]["attack_legacy"].bundle
    attack_legacy = attack_legacy.__class__(
        result=attack_legacy.result.__class__(**{**attack_legacy.result.__dict__, "candidate_id": "alpha_attack_major8_equity25_legacy"}),
        benchmark_gross_ret=attack_legacy.benchmark_gross_ret,
        profile=attack_legacy.profile,
        benchmark_profile=attack_legacy.benchmark_profile,
    )
    baseline = built["baseline"]
    baseline_guard = built["baseline_guard"]

    bundles = [attack, attack_legacy, baseline, baseline_guard]

    frozen_rows = [_frozen_row(bundle, capital_brl=float(args.capital_brl)) for bundle in bundles]
    for row in frozen_rows:
        row["candidate_label"] = _human_label(row["candidate_id"])
    frozen_df = pd.DataFrame(frozen_rows).sort_values(["net_total_return", "net_sharpe"], ascending=[False, False]).reset_index(drop=True)
    frozen_df.to_csv(outdir / "frozen_overview.csv", index=False)

    yearbook_rows: list[dict[str, Any]] = []
    for bundle in bundles:
        rows = _calendar_rows(result=bundle.result, capital_brl=float(args.capital_brl))
        for row in rows:
            row["candidate_label"] = _human_label(row["candidate_id"])
        yearbook_rows.extend(rows)
    yearbook_df = pd.DataFrame(yearbook_rows)
    yearbook_df.to_csv(outdir / "yearbook_reais.csv", index=False)

    month_rows: list[dict[str, Any]] = []
    quarter_rows: list[dict[str, Any]] = []
    for bundle in bundles:
        monthly = _compound_periods(bundle, freq="M")
        quarterly = _compound_periods(bundle, freq="Q")
        if not monthly.empty:
            monthly["candidate_label"] = _human_label(bundle.result.candidate_id)
            month_rows.extend(monthly.to_dict("records"))
        if not quarterly.empty:
            quarterly["candidate_label"] = _human_label(bundle.result.candidate_id)
            quarter_rows.extend(quarterly.to_dict("records"))
    month_df = pd.DataFrame(month_rows)
    quarter_df = pd.DataFrame(quarter_rows)
    month_df.to_csv(outdir / "monthly_tournament.csv", index=False)
    quarter_df.to_csv(outdir / "quarterly_tournament.csv", index=False)

    boot_rows: list[dict[str, Any]] = []
    confidence_rows: list[dict[str, Any]] = []
    for i, bundle in enumerate(bundles, start=1):
        boot = _bootstrap_summary(bundle, seed=100 + i)
        for horizon, payload in boot.items():
            if str(payload.get("status")) != "ok":
                continue
            boot_rows.append(
                {
                    "candidate_id": str(bundle.result.candidate_id),
                    "candidate_label": _human_label(bundle.result.candidate_id),
                    "horizon": str(horizon),
                    "prob_positive": payload.get("prob_positive"),
                    "prob_beat_benchmark": payload.get("prob_beat_benchmark"),
                    "ann_return_p05": payload.get("strategy_ann_return_p05"),
                    "ann_return_p50": payload.get("strategy_ann_return_p50"),
                    "ann_return_p95": payload.get("strategy_ann_return_p95"),
                    "mdd_p05": payload.get("strategy_mdd_p05"),
                    "mdd_p50": payload.get("strategy_mdd_p50"),
                    "mdd_p95": payload.get("strategy_mdd_p95"),
                }
            )
        confidence_rows.extend(_confidence_rows(str(bundle.result.candidate_id), boot))
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(outdir / "bootstrap_compare.csv", index=False)
    confidence_df = pd.DataFrame(confidence_rows)
    confidence_df.to_csv(outdir / "confidence_bands.csv", index=False)

    pbo_matrix = _common_monthly_matrix(bundles)
    pbo_rows: list[dict[str, Any]] = []
    pbo_summary: dict[str, Any] = {}
    if not pbo_matrix.empty and len(pbo_matrix.columns) >= 2:
        for metric in ["total_return", "sharpe"]:
            detail_df, metric_summary = _pbo_for_metric(pbo_matrix, metric=metric, n_slices=8)
            detail_df["metric"] = metric
            pbo_rows.extend(detail_df.to_dict("records"))
            pbo_summary[metric] = metric_summary
    pbo_df = pd.DataFrame(pbo_rows)
    pbo_df.to_csv(outdir / "pbo_compare.csv", index=False)

    best = frozen_df.iloc[0].to_dict() if not frozen_df.empty else {}
    legacy_row = frozen_df[frozen_df["candidate_id"] == "alpha_attack_major8_equity25_legacy"].head(1)
    promoted_row = frozen_df[frozen_df["candidate_id"] == "alpha_attack_major8_equity25"].head(1)
    summary = {
        "suite": "profit_attack_promotion_suite",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "best_overall": best,
        "promoted_attack": promoted_row.iloc[0].to_dict() if not promoted_row.empty else {},
        "legacy_attack": legacy_row.iloc[0].to_dict() if not legacy_row.empty else {},
        "pbo": pbo_summary,
        "worth_promoting": bool(
            not promoted_row.empty
            and not legacy_row.empty
            and float(promoted_row.iloc[0]["net_total_return"]) > float(legacy_row.iloc[0]["net_total_return"])
            and float(promoted_row.iloc[0]["net_sharpe"]) >= float(legacy_row.iloc[0]["net_sharpe"]) * 0.98
        ),
        "artifacts": {
            "frozen_overview_csv": str(outdir / "frozen_overview.csv"),
            "yearbook_reais_csv": str(outdir / "yearbook_reais.csv"),
            "monthly_tournament_csv": str(outdir / "monthly_tournament.csv"),
            "quarterly_tournament_csv": str(outdir / "quarterly_tournament.csv"),
            "bootstrap_compare_csv": str(outdir / "bootstrap_compare.csv"),
            "confidence_bands_csv": str(outdir / "confidence_bands.csv"),
            "pbo_compare_csv": str(outdir / "pbo_compare.csv"),
        },
    }
    _write_json(outdir / "summary.json", summary)

    research_rows = [
        _research_row(attack.result, outdir=ROOT / "results" / "validation", status="keep", methodology="attack_promotion_exogenous_liquidation", label="Ataque com overlay de liquidação cripto"),
        _research_row(attack_legacy.result, outdir=ROOT / "results" / "validation", status="watch", methodology="attack_promotion_legacy_reference", label="Ataque anterior"),
    ]
    pd.DataFrame(research_rows).to_csv(outdir / "research_registry_rows.csv", index=False)

    write_run_manifest(
        outdir,
        script="scripts/bench/validation/run_profit_attack_promotion_suite.py",
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
            "frozen_overview_csv": str(outdir / "frozen_overview.csv"),
            "yearbook_reais_csv": str(outdir / "yearbook_reais.csv"),
            "bootstrap_compare_csv": str(outdir / "bootstrap_compare.csv"),
            "pbo_compare_csv": str(outdir / "pbo_compare.csv"),
        },
        extra={
            "suite": "profit_attack_promotion_suite",
            "best_candidate_id": str(best.get("candidate_id") or ""),
            "worth_promoting": bool(summary["worth_promoting"]),
        },
        repo_root=ROOT,
    )
    print(outdir)


if __name__ == "__main__":
    main()
