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
from scripts.bench.validation.run_profit_alpha_hardening_suite import (  # noqa: E402
    AllocationBundle,
    _blend_allocation_bundles,
    _build_candidates,
)
from scripts.bench.validation.run_profit_alpha_improvement_suite import _safe_float, _write_json  # noqa: E402
from scripts.bench.validation.run_profit_attack_entry_ranking_suite import _result_row  # noqa: E402
from scripts.bench.validation.run_profit_champion_extension_suite import (  # noqa: E402
    _build_criticality_free_energy_bundle,
    _fragility_bridge_weight,
    _fragility_decile_scale,
    _profit_lock_scale,
)
from scripts.bench.validation.run_profit_investment_yearbook import _calendar_rows  # noqa: E402
from scripts.bench.validation.run_profit_marketmode_criticality_suite import _build_structure_layers  # noqa: E402
from scripts.bench.validation.run_profit_pbo_suite import _pbo_for_metric, _pbo_verdict  # noqa: E402
from scripts.bench.validation.run_profit_champion_timing_robustness_suite import (  # noqa: E402
    _candidate_pbo_profile,
    _common_monthly_matrix,
    _underperform_prob_rolling,
)


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _topk_crypto_share(weights: pd.DataFrame, *, crypto_tickers: list[str], k: int) -> tuple[float, float]:
    if weights.empty or not crypto_tickers:
        return float("nan"), float("nan")
    crypto = weights.reindex(columns=list(crypto_tickers)).fillna(0.0)
    total = pd.to_numeric(crypto.sum(axis=1), errors="coerce").fillna(0.0).astype(float)
    if crypto.empty:
        return float("nan"), float("nan")
    ordered = np.sort(crypto.to_numpy(dtype=float), axis=1)
    topk = pd.Series(ordered[:, -min(int(k), ordered.shape[1]):].sum(axis=1), index=crypto.index, dtype=float)
    share = (topk / total.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return float(share.mean()), float(share.max())


def _leave_one_year_out(net_ret: pd.Series) -> pd.DataFrame:
    ret = pd.to_numeric(net_ret, errors="coerce").dropna().astype(float)
    if ret.empty:
        return pd.DataFrame(columns=["excluded_year", "remaining_total_return", "remaining_ann_return"])
    years = sorted(pd.to_datetime(ret.index).year.unique().tolist())
    rows: list[dict[str, Any]] = []
    for year in years:
        kept = ret[pd.to_datetime(ret.index).year != int(year)]
        if kept.empty:
            continue
        total = float(np.prod(1.0 + kept.to_numpy(dtype=float)) - 1.0)
        span_years = max(1.0 / 252.0, float(len(kept)) / 252.0)
        ann = float((1.0 + total) ** (1.0 / span_years) - 1.0) if total > -1.0 else -1.0
        rows.append(
            {
                "excluded_year": int(year),
                "remaining_total_return": total,
                "remaining_ann_return": ann,
            }
        )
    return pd.DataFrame(rows)


def _forward_return_distribution(net_ret: pd.Series, *, horizon: int, mask: pd.Series | None = None) -> pd.Series:
    ret = pd.to_numeric(net_ret, errors="coerce").dropna().astype(float)
    if ret.empty or len(ret) <= int(horizon):
        return pd.Series(dtype=float)
    values = ret.to_numpy(dtype=float)
    out = np.full(ret.shape[0], np.nan, dtype=float)
    for i in range(0, len(values) - int(horizon)):
        out[i] = float(np.prod(1.0 + values[i + 1 : i + 1 + int(horizon)]) - 1.0)
    series = pd.Series(out, index=ret.index, dtype=float).dropna()
    if mask is not None:
        aligned_mask = pd.Series(mask, copy=True).reindex(series.index).fillna(False).astype(bool)
        series = series[aligned_mask]
    return series


def _forecast_bands(net_ret: pd.Series, *, current_regime: str, regime_series: pd.Series) -> pd.DataFrame:
    regime = pd.Series(regime_series, copy=True)
    regime.index = pd.to_datetime(regime.index)
    ret = pd.Series(net_ret, copy=True)
    ret.index = pd.to_datetime(ret.index)
    conditional_mask = regime.reindex(ret.index).ffill().eq(str(current_regime))
    rows: list[dict[str, Any]] = []
    for horizon in (21, 63, 126, 252):
        unconditional = _forward_return_distribution(ret, horizon=horizon)
        conditional = _forward_return_distribution(ret, horizon=horizon, mask=conditional_mask)
        for label, series in (("all_history", unconditional), (f"regime_{current_regime}", conditional)):
            if series.empty:
                continue
            rows.append(
                {
                    "window": label,
                    "horizon_days": int(horizon),
                    "samples": int(series.shape[0]),
                    "p10": float(series.quantile(0.10)),
                    "p50": float(series.quantile(0.50)),
                    "p90": float(series.quantile(0.90)),
                    "win_rate": float((series > 0.0).mean()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Validação oficial pós-fiscal do modo live com métricas robustas e bandas probabilísticas.")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--capital-brl", type=float, default=10000.0)
    ap.add_argument("--outdir-root", default="results/validation/profit_official_post_fiscal_validation")
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
    context = dict(built["context"])
    attack_alloc: AllocationBundle = built["allocations"]["attack"]
    protect_alloc: AllocationBundle = built["allocations"]["baseline_guard"]

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

    liquidation = pd.to_numeric(context["exogenous_panel"].get("liquidation"), errors="coerce").reindex(baseline_weight.index).shift(1).fillna(0.0)
    fragility_weight = (baseline_weight * _fragility_decile_scale(liquidation)).clip(0.0, 1.0)
    fragility_bundle = _blend_allocation_bundles(
        candidate_id="champion_fragility_decile",
        notes="reduz tamanho quando fragilidade entra no pior decil",
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        attack_weight=fragility_weight,
    )
    bridge_bundle = _blend_allocation_bundles(
        candidate_id="champion_fragility_u800_bridge",
        notes="ponte de fragilidade como challenger",
        attack_alloc=baseline_bundle,
        protect_alloc=fragility_bundle,
        attack_weight=_fragility_bridge_weight(liquidation),
    )
    profit_lock_weight = (baseline_weight * _profit_lock_scale(baseline_bundle.bundle.result.net_ret)).clip(0.0, 1.0)
    official_bundle = _blend_allocation_bundles(
        candidate_id="champion_profit_lock_partial",
        notes="trava parcial de lucro usando apenas historico realizado",
        attack_alloc=attack_alloc,
        protect_alloc=protect_alloc,
        attack_weight=profit_lock_weight,
    )

    candidate_bundles = {
        baseline_bundle.bundle.result.candidate_id: baseline_bundle.bundle,
        fragility_bundle.bundle.result.candidate_id: fragility_bundle,
        bridge_bundle.bundle.result.candidate_id: bridge_bundle,
        official_bundle.bundle.result.candidate_id: official_bundle,
    }
    results = {cid: bundle.result for cid, bundle in candidate_bundles.items()}

    rows: list[dict[str, Any]] = []
    crypto_tickers = list(context["crypto_tiers"]["crypto_all"])
    for cid, bundle in candidate_bundles.items():
        result = bundle.result
        row = _result_row(result, baseline=official_bundle.bundle.result, family="official_post_fiscal", label=cid)
        row["underperform_prob_63"] = _underperform_prob_rolling(result.net_ret, result.benchmark_net_ret, horizon=63)
        top1_mean, top1_max = _topk_crypto_share(bundle.weights, crypto_tickers=crypto_tickers, k=1)
        top3_mean, top3_max = _topk_crypto_share(bundle.weights, crypto_tickers=crypto_tickers, k=3)
        row["crypto_top1_share_mean"] = top1_mean
        row["crypto_top1_share_max"] = top1_max
        row["crypto_top3_share_mean"] = top3_mean
        row["crypto_top3_share_max"] = top3_max
        rows.append(row)
    compare_df = pd.DataFrame(rows)

    monthly_matrix = _common_monthly_matrix(results)
    pbo_metric_summary: dict[str, Any] = {}
    pbo_profile_frames: list[pd.DataFrame] = []
    if not monthly_matrix.empty and len(monthly_matrix.columns) >= 2:
        for metric in ("total_return", "sharpe"):
            split_df, metric_summary = _pbo_for_metric(monthly_matrix, metric=metric, n_slices=8)
            metric_summary["verdict"] = _pbo_verdict(float(metric_summary.get("pbo", float("nan"))))
            pbo_metric_summary[str(metric)] = metric_summary
            if not split_df.empty:
                split_df["metric"] = str(metric)
                pbo_profile_frames.append(_candidate_pbo_profile(split_df, metric=str(metric)))
    pbo_profile_df = pd.concat(pbo_profile_frames, axis=0, ignore_index=True) if pbo_profile_frames else pd.DataFrame()
    pbo_candidate_summary = (
        pbo_profile_df.groupby("candidate_id", as_index=False).agg(
            pbo_win_splits=("pbo_win_splits", "sum"),
            pbo_below_median_rate=("pbo_below_median_rate", "mean"),
            pbo_median_oos_rank=("pbo_median_oos_rank", "mean"),
        )
        if not pbo_profile_df.empty
        else pd.DataFrame(columns=["candidate_id", "pbo_win_splits", "pbo_below_median_rate", "pbo_median_oos_rank"])
    )
    compare_df = compare_df.merge(pbo_candidate_summary, on="candidate_id", how="left")
    compare_df.to_csv(outdir / "candidate_compare.csv", index=False)

    leave_one_year_df = _leave_one_year_out(official_bundle.bundle.result.net_ret)
    leave_one_year_df.to_csv(outdir / "leave_one_year_out.csv", index=False)

    current_regime = str(structure_daily["regime"].dropna().iloc[-1]) if "regime" in structure_daily and not structure_daily["regime"].dropna().empty else "unknown"
    forecast_df = _forecast_bands(
        official_bundle.bundle.result.net_ret,
        current_regime=current_regime,
        regime_series=pd.Series(structure_daily.get("regime"), copy=True),
    )
    forecast_df.to_csv(outdir / "forecast_bands.csv", index=False)

    calendar_rows: list[dict[str, Any]] = []
    for result in results.values():
        calendar_rows.extend(_calendar_rows(result=result, capital_brl=float(args.capital_brl)))
    pd.DataFrame(calendar_rows).sort_values(["year", "candidate_id"]).to_csv(outdir / "yearbook_reais.csv", index=False)

    official_row = compare_df.loc[compare_df["candidate_id"] == "champion_profit_lock_partial"].iloc[0].to_dict()
    baseline_row = compare_df.loc[compare_df["candidate_id"] == "criticality_free_energy_attack"].iloc[0].to_dict()
    leave_one_year_worst = leave_one_year_df.sort_values("remaining_total_return").head(1).to_dict("records")
    current_regime_forecast = forecast_df[forecast_df["window"] == f"regime_{current_regime}"].copy()
    current_regime_forecast = current_regime_forecast.sort_values("horizon_days")

    summary = {
        "suite": "profit_official_post_fiscal_validation",
        "official_candidate": "champion_profit_lock_partial",
        "baseline_candidate": "criticality_free_energy_attack",
        "official_underperform_prob_63": _safe_float(official_row.get("underperform_prob_63")),
        "official_crypto_top1_share_mean": _safe_float(official_row.get("crypto_top1_share_mean")),
        "official_crypto_top3_share_mean": _safe_float(official_row.get("crypto_top3_share_mean")),
        "official_net_ann_return": _safe_float(official_row.get("net_ann_return")),
        "official_net_total_return": _safe_float(official_row.get("net_total_return")),
        "official_net_max_drawdown": _safe_float(official_row.get("net_max_drawdown")),
        "baseline_net_ann_return": _safe_float(baseline_row.get("net_ann_return")),
        "baseline_net_total_return": _safe_float(baseline_row.get("net_total_return")),
        "baseline_net_max_drawdown": _safe_float(baseline_row.get("net_max_drawdown")),
        "pbo_overall": pbo_metric_summary,
        "leave_one_year_out_worst_case": leave_one_year_worst[0] if leave_one_year_worst else {},
        "current_regime": current_regime,
        "current_regime_forecast": current_regime_forecast.to_dict("records"),
        "notes": [
            "Bandas de forecast são distribuições históricas condicionais, não promessa.",
            "Concentração cripto aqui mede share médio do top1 e top3 dentro do sleeve cripto do modo oficial.",
            "Leave-one-year-out mostra o quanto o resultado total depende de anos específicos.",
        ],
    }
    _write_json(outdir / "summary.json", summary)

    write_run_manifest(
        outdir,
        script="scripts/bench/validation/run_profit_official_post_fiscal_validation.py",
        params=vars(args),
        extra={
            "suite": "profit_official_post_fiscal_validation",
            "official_candidate": "champion_profit_lock_partial",
            "baseline_candidate": "criticality_free_energy_attack",
        },
    )
    print(str(outdir))


if __name__ == "__main__":
    main()
