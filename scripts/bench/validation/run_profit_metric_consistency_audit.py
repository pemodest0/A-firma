#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bench.validation.run_profit_alpha_hardening_suite import _blend_allocation_bundles, _build_candidates  # noqa: E402
from scripts.bench.validation.run_profit_alpha_improvement_suite import _safe_float, _write_json  # noqa: E402
from scripts.bench.validation.run_profit_champion_extension_suite import _build_criticality_free_energy_bundle, _profit_lock_scale  # noqa: E402
from scripts.bench.validation.run_profit_champion_timing_robustness_suite import _underperform_prob_rolling  # noqa: E402
from scripts.bench.validation.run_profit_country_compare_suite import (  # noqa: E402
    _build_official_bundle,
    _filter_brazil_equities,
    _link_prices_dir,
    _write_synthetic_benchmark,
)
from scripts.bench.validation.run_profit_investment_yearbook import _calendar_rows  # noqa: E402
from scripts.bench.validation.run_profit_marketmode_criticality_suite import _build_structure_layers  # noqa: E402
from scripts.bench.validation.run_profit_official_post_fiscal_validation import _leave_one_year_out, _topk_crypto_share  # noqa: E402


def _run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _compare_scalar(*, scope: str, metric: str, expected: Any, actual: Any, tolerance: float = 1e-10) -> dict[str, Any]:
    exp = _safe_float(expected, float("nan"))
    act = _safe_float(actual, float("nan"))
    if not math.isfinite(exp) and not math.isfinite(act):
        abs_diff = 0.0
        within = True
    else:
        abs_diff = abs(float(exp) - float(act))
        within = abs_diff <= float(tolerance)
    return {
        "scope": str(scope),
        "metric": str(metric),
        "expected": exp,
        "actual": act,
        "abs_diff": float(abs_diff),
        "tolerance": float(tolerance),
        "within_tolerance": bool(within),
    }


def _series_metrics(result: Any) -> dict[str, float]:
    ret = pd.to_numeric(result.net_ret, errors="coerce").dropna().astype(float)
    bench = pd.to_numeric(result.benchmark_net_ret, errors="coerce").reindex(ret.index).fillna(0.0).astype(float)
    turnover = pd.to_numeric(result.turnover, errors="coerce").reindex(ret.index).fillna(0.0).astype(float)
    total = float((1.0 + ret).prod() - 1.0) if not ret.empty else float("nan")
    span_years = max(1.0 / 252.0, float(len(ret)) / 252.0) if not ret.empty else float("nan")
    ann = float((1.0 + total) ** (1.0 / span_years) - 1.0) if ret.size and total > -1.0 else -1.0
    wealth = (1.0 + ret).cumprod()
    peak = wealth.cummax()
    mdd = float((wealth / peak - 1.0).min()) if not wealth.empty else float("nan")
    mu = float(ret.mean()) if not ret.empty else float("nan")
    sd = float(ret.std(ddof=1)) if len(ret) > 1 else float("nan")
    sharpe = float((mu / sd) * (252.0 ** 0.5)) if math.isfinite(sd) and sd > 0.0 else float("nan")
    bench_total = float((1.0 + bench).prod() - 1.0) if not bench.empty else float("nan")
    return {
        "net_total_return": total,
        "net_ann_return": ann,
        "net_max_drawdown": mdd,
        "net_sharpe": sharpe,
        "edge_vs_benchmark": float(total - bench_total) if math.isfinite(total) and math.isfinite(bench_total) else float("nan"),
        "avg_turnover_daily": float(turnover.mean()) if not turnover.empty else float("nan"),
        "underperform_prob_63": _underperform_prob_rolling(ret, bench, horizon=63),
    }


def _build_global_official_allocations(
    *,
    prices_dir: Path,
    crypto_groups: Path,
    crypto_meta: Path,
    equity_groups: Path,
    equity_meta: Path,
    benchmark_crypto: str,
    benchmark_equity: str,
) -> dict[str, Any]:
    built = _build_candidates(
        prices_dir=prices_dir,
        crypto_groups=crypto_groups,
        crypto_meta=crypto_meta,
        equity_groups=equity_groups,
        equity_meta=equity_meta,
        benchmark_crypto=str(benchmark_crypto),
        benchmark_equity=str(benchmark_equity),
    )
    context = dict(built["context"])
    attack_alloc = built["allocations"]["attack"]
    protect_alloc = built["allocations"]["baseline_guard"]
    structure_daily, _spectral_panel, criticality, _structural_stress = _build_structure_layers(context)
    baseline_bundle, _baseline_score, baseline_weight = _build_criticality_free_energy_bundle(
        candidate_id="criticality_free_energy_attack",
        notes="baseline causal antes da trava parcial de lucro",
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        base_score=context["attack_score_exogenous"],
        structure_daily=structure_daily,
        criticality=criticality,
    )
    profit_lock_weight = (baseline_weight * _profit_lock_scale(baseline_bundle.bundle.result.net_ret)).clip(0.0, 1.0)
    official_bundle = _blend_allocation_bundles(
        candidate_id="champion_profit_lock_partial",
        notes="trava parcial de lucro usando apenas historico realizado",
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        attack_weight=profit_lock_weight,
    )
    return {
        "context": context,
        "baseline_alloc": baseline_bundle,
        "official_alloc": official_bundle,
    }


def _published_row(df: pd.DataFrame, key: str, value: str) -> dict[str, Any]:
    sub = df.loc[df[key].astype(str) == str(value)]
    if sub.empty:
        raise KeyError(f"row not found: {key}={value}")
    return sub.iloc[0].to_dict()


def _snapshot_dir(artifact_dir: Path, label: str) -> Path | None:
    path = artifact_dir / "input_snapshot" / str(label)
    return path if path.exists() else None


def _snapshot_metadata(snapshot_dir: Path | None, fallback: Path) -> Path:
    if snapshot_dir is None:
        return fallback
    candidate = snapshot_dir / "metadata" / fallback.name
    return candidate if candidate.exists() else fallback


def _artifact_csv(artifact_dir: Path, *names: str) -> Path:
    for name in names:
        path = artifact_dir / str(name)
        if path.exists():
            return path
    raise FileNotFoundError(f"none of the candidate files exist in {artifact_dir}: {names}")


def _country_summary_rows(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload.get("rows")
    if isinstance(rows, list):
        return {str(row.get("scenario_label")): row for row in rows if isinstance(row, dict)}
    out: dict[str, dict[str, Any]] = {}
    for key in ("brazil_crypto_official", "brazil_only_best", "global_mixed_official"):
        value = payload.get(key)
        if isinstance(value, dict):
            label = str(value.get("scenario_label") or key)
            out[label] = value
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Audita consistência das melhores métricas do motor contra os artefatos publicados.")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--capital-brl", type=float, default=10000.0)
    ap.add_argument("--global-artifact", default="results/validation/profit_official_post_fiscal_validation/20260315T002633Z")
    ap.add_argument("--country-artifact", default="results/validation/profit_country_compare_suite/20260315T023950Z")
    ap.add_argument("--outdir-root", default="results/validation/profit_metric_consistency_audit")
    args = ap.parse_args()

    outdir = (ROOT / args.outdir_root / _run_id()).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    prices_dir = (ROOT / args.prices_dir).resolve()
    crypto_groups = (ROOT / args.crypto_asset_groups).resolve()
    crypto_meta = (ROOT / args.crypto_asset_metadata).resolve()
    equity_groups = (ROOT / args.equity_asset_groups).resolve()
    equity_meta = (ROOT / args.equity_asset_metadata).resolve()

    comparisons: list[dict[str, Any]] = []
    notes: list[str] = []

    global_artifact = (ROOT / args.global_artifact).resolve()
    global_summary = json.loads((global_artifact / "summary.json").read_text(encoding="utf-8"))
    global_compare = pd.read_csv(global_artifact / "candidate_compare.csv")
    global_yearbook = pd.read_csv(global_artifact / "yearbook_reais.csv")
    global_snapshot = _snapshot_dir(global_artifact, "official_global")
    global_prices_dir = (global_snapshot / "prices").resolve() if global_snapshot is not None else prices_dir
    global_crypto_groups = _snapshot_metadata(global_snapshot, crypto_groups)
    global_crypto_meta = _snapshot_metadata(global_snapshot, crypto_meta)
    global_equity_groups = _snapshot_metadata(global_snapshot, equity_groups)
    global_equity_meta = _snapshot_metadata(global_snapshot, equity_meta)

    global_allocs = _build_global_official_allocations(
        prices_dir=global_prices_dir,
        crypto_groups=global_crypto_groups,
        crypto_meta=global_crypto_meta,
        equity_groups=global_equity_groups,
        equity_meta=global_equity_meta,
        benchmark_crypto=str(args.benchmark_crypto),
        benchmark_equity=str(args.benchmark_equity),
    )
    global_official = global_allocs["official_alloc"].bundle.result
    global_baseline = global_allocs["baseline_alloc"].bundle.result
    global_official_metrics = _series_metrics(global_official)
    global_baseline_metrics = _series_metrics(global_baseline)
    global_crypto_tickers = list(global_allocs["context"]["crypto_tiers"]["crypto_all"])
    off_top1_mean, off_top1_max = _topk_crypto_share(global_allocs["official_alloc"].weights, crypto_tickers=global_crypto_tickers, k=1)
    off_top3_mean, off_top3_max = _topk_crypto_share(global_allocs["official_alloc"].weights, crypto_tickers=global_crypto_tickers, k=3)
    base_top1_mean, base_top1_max = _topk_crypto_share(global_allocs["baseline_alloc"].weights, crypto_tickers=global_crypto_tickers, k=1)
    base_top3_mean, base_top3_max = _topk_crypto_share(global_allocs["baseline_alloc"].weights, crypto_tickers=global_crypto_tickers, k=3)
    leave_one_year_df = _leave_one_year_out(global_official.net_ret)
    leave_one_year_worst = leave_one_year_df.sort_values("remaining_total_return").head(1).iloc[0].to_dict()

    for metric in ["net_ann_return", "net_total_return", "net_sharpe", "net_max_drawdown", "edge_vs_benchmark", "avg_turnover_daily", "underperform_prob_63"]:
        comparisons.append(_compare_scalar(scope="global_official.result_vs_series", metric=metric, expected=getattr(global_official, metric, global_official_metrics.get(metric)), actual=global_official_metrics.get(metric)))
        comparisons.append(_compare_scalar(scope="global_baseline.result_vs_series", metric=metric, expected=getattr(global_baseline, metric, global_baseline_metrics.get(metric)), actual=global_baseline_metrics.get(metric)))

    global_official_row = _published_row(global_compare, "candidate_id", "champion_profit_lock_partial")
    global_baseline_row = _published_row(global_compare, "candidate_id", "criticality_free_energy_attack")
    for metric in ["net_ann_return", "net_total_return", "net_sharpe", "net_max_drawdown", "edge_vs_benchmark", "avg_turnover_daily", "underperform_prob_63"]:
        comparisons.append(_compare_scalar(scope="global_official.published_compare", metric=metric, expected=global_official_metrics.get(metric), actual=global_official_row.get(metric)))
        comparisons.append(_compare_scalar(scope="global_baseline.published_compare", metric=metric, expected=global_baseline_metrics.get(metric), actual=global_baseline_row.get(metric)))

    comparisons.extend(
        [
            _compare_scalar(scope="global_official.published_compare", metric="crypto_top1_share_mean", expected=off_top1_mean, actual=global_official_row.get("crypto_top1_share_mean")),
            _compare_scalar(scope="global_official.published_compare", metric="crypto_top3_share_mean", expected=off_top3_mean, actual=global_official_row.get("crypto_top3_share_mean")),
            _compare_scalar(scope="global_baseline.published_compare", metric="crypto_top1_share_mean", expected=base_top1_mean, actual=global_baseline_row.get("crypto_top1_share_mean")),
            _compare_scalar(scope="global_baseline.published_compare", metric="crypto_top3_share_mean", expected=base_top3_mean, actual=global_baseline_row.get("crypto_top3_share_mean")),
        ]
    )

    comparisons.extend(
        [
            _compare_scalar(scope="global_summary", metric="official_net_ann_return", expected=global_official_metrics["net_ann_return"], actual=global_summary.get("official_net_ann_return")),
            _compare_scalar(scope="global_summary", metric="official_net_total_return", expected=global_official_metrics["net_total_return"], actual=global_summary.get("official_net_total_return")),
            _compare_scalar(scope="global_summary", metric="official_net_max_drawdown", expected=global_official_metrics["net_max_drawdown"], actual=global_summary.get("official_net_max_drawdown")),
            _compare_scalar(scope="global_summary", metric="official_underperform_prob_63", expected=global_official_metrics["underperform_prob_63"], actual=global_summary.get("official_underperform_prob_63")),
            _compare_scalar(scope="global_summary", metric="official_crypto_top1_share_mean", expected=off_top1_mean, actual=global_summary.get("official_crypto_top1_share_mean")),
            _compare_scalar(scope="global_summary", metric="official_crypto_top3_share_mean", expected=off_top3_mean, actual=global_summary.get("official_crypto_top3_share_mean")),
            _compare_scalar(scope="global_summary", metric="baseline_net_ann_return", expected=global_baseline_metrics["net_ann_return"], actual=global_summary.get("baseline_net_ann_return")),
            _compare_scalar(scope="global_summary", metric="baseline_net_total_return", expected=global_baseline_metrics["net_total_return"], actual=global_summary.get("baseline_net_total_return")),
            _compare_scalar(scope="global_summary", metric="baseline_net_max_drawdown", expected=global_baseline_metrics["net_max_drawdown"], actual=global_summary.get("baseline_net_max_drawdown")),
            _compare_scalar(scope="global_summary", metric="leave_one_year_out.remaining_total_return", expected=leave_one_year_worst["remaining_total_return"], actual=(global_summary.get("leave_one_year_out_worst_case") or {}).get("remaining_total_return")),
            _compare_scalar(scope="global_summary", metric="leave_one_year_out.remaining_ann_return", expected=leave_one_year_worst["remaining_ann_return"], actual=(global_summary.get("leave_one_year_out_worst_case") or {}).get("remaining_ann_return")),
        ]
    )

    global_yearbook_expected = pd.DataFrame(_calendar_rows(result=global_official, capital_brl=float(args.capital_brl)))
    global_yearbook_published = global_yearbook.loc[global_yearbook["candidate_id"].astype(str) == "champion_profit_lock_partial"].copy()
    merged_global_yearbook = global_yearbook_expected.merge(global_yearbook_published, on=["candidate_id", "year"], how="inner", suffixes=("_expected", "_published"))
    for _, row in merged_global_yearbook.iterrows():
        year = int(row["year"])
        for metric in ["year_total_return", "ending_capital_brl", "operation_days", "turnover_sum"]:
            comparisons.append(
                _compare_scalar(
                    scope=f"global_yearbook.{year}",
                    metric=metric,
                    expected=row[f"{metric}_expected"],
                    actual=row[f"{metric}_published"],
                    tolerance=1e-8 if metric != "operation_days" else 0.0,
                )
            )

    # Brazil + crypto and Brazil-only manual published rows
    manual_artifact = (ROOT / args.country_artifact).resolve()
    manual_compare = pd.read_csv(_artifact_csv(manual_artifact, "manual_candidate_compare.csv", "candidate_compare.csv"))
    manual_summary = json.loads(_artifact_csv(manual_artifact, "manual_compare_summary.json", "summary.json").read_text(encoding="utf-8"))
    brazil_snapshot = _snapshot_dir(manual_artifact, "brazil_crypto")
    if brazil_snapshot is not None:
        br_groups = brazil_snapshot / "metadata" / "equity_groups_brazil_only.csv"
        br_meta = brazil_snapshot / "metadata" / "equity_meta_brazil_only.csv"
        br_prices_dir = (brazil_snapshot / "prices").resolve()
        br_benchmark = "BR_SYNTH"
    else:
        tmp_brazil = outdir / "tmp_brazil"
        tmp_brazil.mkdir(parents=True, exist_ok=True)
        br_groups, br_meta, br_tickers = _filter_brazil_equities(equity_groups=equity_groups, equity_meta=equity_meta, outdir=tmp_brazil)
        br_prices_dir = tmp_brazil / "prices_brazil"
        _link_prices_dir(source_dir=prices_dir, target_dir=br_prices_dir)
        br_benchmark = "BR_SYNTH"
        _write_synthetic_benchmark(prices_dir=prices_dir, tickers=br_tickers, outdir=br_prices_dir, benchmark_ticker=br_benchmark)
    built_br = _build_candidates(
        prices_dir=br_prices_dir,
        crypto_groups=_snapshot_metadata(brazil_snapshot, crypto_groups),
        crypto_meta=_snapshot_metadata(brazil_snapshot, crypto_meta),
        equity_groups=br_groups,
        equity_meta=br_meta,
        benchmark_crypto=str(args.benchmark_crypto),
        benchmark_equity=br_benchmark,
        equity_profile_id="br_local_equity",
    )
    br_baseline, br_official = _build_official_bundle(built_br)
    br_base_result = built_br["context"]["equity_base"].result
    br_attack_result = built_br["context"]["equity_attack"].result
    scenario_map = {
        "brazil_crypto_official": br_official,
        "brazil_only_equity_base": br_base_result,
        "brazil_only_equity_attack": br_attack_result,
    }
    for label, result in scenario_map.items():
        metrics = _series_metrics(result)
        published = _published_row(manual_compare, "scenario_label", label)
        for metric in ["net_ann_return", "net_total_return", "net_max_drawdown", "underperform_prob_63"]:
            comparisons.append(_compare_scalar(scope=f"{label}.published_compare", metric=metric, expected=metrics.get(metric), actual=published.get(metric)))

    rows_payload = _country_summary_rows(manual_summary)
    for label, result in scenario_map.items():
        metrics = _series_metrics(result)
        published = rows_payload.get(label, {})
        for metric in ["net_ann_return", "net_total_return", "net_max_drawdown", "underperform_prob_63"]:
            if published:
                comparisons.append(_compare_scalar(scope=f"{label}.published_summary", metric=metric, expected=metrics.get(metric), actual=published.get(metric)))

    if abs(off_top1_mean) < 1e-12 and abs(off_top3_mean) < 1e-12:
        notes.append(
            "crypto_top1_share_mean e crypto_top3_share_mean do modo oficial ficam estruturalmente zerados porque os pesos publicados do bundle oficial estao no nivel de sleeve (crypto/equity/cash), nao no nivel de cripto individual."
        )

    compare_df = pd.DataFrame(comparisons)
    compare_df.to_csv(outdir / "metric_checks.csv", index=False)
    failing = compare_df.loc[~compare_df["within_tolerance"]].copy()
    failing.to_csv(outdir / "metric_failures.csv", index=False)
    summary = {
        "suite": "profit_metric_consistency_audit",
        "generated_at": datetime.now(UTC).isoformat(),
        "checks_total": int(compare_df.shape[0]),
        "checks_passed": int(compare_df["within_tolerance"].sum()),
        "checks_failed": int((~compare_df["within_tolerance"]).sum()),
        "global_official_years_checked": sorted(merged_global_yearbook["year"].astype(int).tolist()),
        "notes": notes,
        "artifacts": {
            "metric_checks_csv": str(outdir / "metric_checks.csv"),
            "metric_failures_csv": str(outdir / "metric_failures.csv"),
        },
        "input_sources": {
            "global_prices_dir": str(global_prices_dir),
            "global_snapshot": str(global_snapshot) if global_snapshot is not None else None,
            "brazil_prices_dir": str(br_prices_dir),
            "brazil_snapshot": str(brazil_snapshot) if brazil_snapshot is not None else None,
        },
    }
    _write_json(outdir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
