#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.structural.run_manifest import write_run_manifest  # noqa: E402
from scripts.bench.validation.run_profit_alpha_hardening_suite import (  # noqa: E402
    _blend_allocation_bundles,
    _build_candidates,
)
from scripts.bench.validation.run_profit_alpha_improvement_suite import _write_json  # noqa: E402
from scripts.bench.validation.run_profit_attack_entry_ranking_suite import _result_row  # noqa: E402
from scripts.bench.validation.run_profit_champion_extension_suite import (  # noqa: E402
    _build_criticality_free_energy_bundle,
    _profit_lock_scale,
)
from scripts.bench.validation.run_profit_champion_timing_robustness_suite import _underperform_prob_rolling  # noqa: E402
from scripts.bench.validation.run_profit_investment_yearbook import _calendar_rows  # noqa: E402
from scripts.bench.validation.run_profit_marketmode_criticality_suite import _build_structure_layers  # noqa: E402


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _filter_brazil_equities(*, equity_groups: Path, equity_meta: Path, outdir: Path) -> tuple[Path, Path, list[str]]:
    groups = pd.read_csv(equity_groups)
    meta = pd.read_csv(equity_meta)
    groups_mask = groups["group"].astype(str).str.startswith("equities_br_") | groups["asset"].astype(str).str.endswith(".SA")
    br_groups = groups.loc[groups_mask].copy()
    selected_assets = set(br_groups["asset"].astype(str))
    meta_key = "ticker" if "ticker" in meta.columns else "asset_id"
    meta_mask = meta[meta_key].astype(str).isin(selected_assets) | meta["sector_internal"].astype(str).str.startswith("equities_br_")
    br_meta = meta.loc[meta_mask].copy()
    br_meta = br_meta[br_meta[meta_key].astype(str).isin(selected_assets)].copy()
    tickers = sorted(set(br_meta[meta_key].astype(str)))
    if not tickers:
        raise SystemExit("nenhum ticker Brasil encontrado no universo de equities")
    groups_path = outdir / "equity_groups_brazil_only.csv"
    meta_path = outdir / "equity_meta_brazil_only.csv"
    br_groups.to_csv(groups_path, index=False)
    br_meta.to_csv(meta_path, index=False)
    return groups_path, meta_path, tickers


def _link_prices_dir(*, source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(source_dir.glob("*.csv")):
        dst = target_dir / src.name
        if dst.exists():
            continue
        try:
            os.symlink(src.resolve(), dst)
        except OSError:
            shutil.copy2(src, dst)


def _write_synthetic_benchmark(*, prices_dir: Path, tickers: list[str], outdir: Path, benchmark_ticker: str) -> Path:
    series_map: dict[str, pd.Series] = {}
    for ticker in tickers:
        path = prices_dir / f"{ticker}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "date" not in frame.columns or "price" not in frame.columns:
            continue
        date = pd.to_datetime(frame["date"], errors="coerce")
        price = pd.to_numeric(frame["price"], errors="coerce")
        clean = pd.DataFrame({"date": date, "price": price}).dropna().sort_values("date")
        if clean.shape[0] < 252:
            continue
        clean = clean.drop_duplicates("date", keep="last").set_index("date")["price"].astype(float)
        series_map[str(ticker)] = clean
    if not series_map:
        raise SystemExit("falha ao montar benchmark sintetico Brasil: sem series suficientes")
    panel = pd.concat(series_map, axis=1, sort=True).sort_index()
    simple_ret = panel.pct_change().replace([np.inf, -np.inf], np.nan).mean(axis=1, skipna=True).fillna(0.0)
    price = (1.0 + simple_ret).cumprod() * 100.0
    log_price = np.log(price.clip(lower=1e-9))
    frame = pd.DataFrame(
        {
            "date": price.index,
            "price": price.to_numpy(dtype=float),
            "log_price": log_price.to_numpy(dtype=float),
            "r": log_price.diff().fillna(0.0).to_numpy(dtype=float),
        }
    )
    out_path = outdir / f"{benchmark_ticker}.csv"
    frame.to_csv(out_path, index=False)
    return out_path


def _build_official_bundle(built: dict[str, Any]) -> tuple[Any, Any]:
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
    return baseline_bundle.bundle.result, official_bundle.bundle.result


def _scenario_row(*, scenario: str, label: str, result: Any, baseline: Any) -> dict[str, Any]:
    row = _result_row(result, baseline=baseline, family=scenario, label=label)
    row["scenario"] = str(scenario)
    row["scenario_label"] = str(label)
    row["underperform_prob_63"] = _underperform_prob_rolling(result.net_ret, result.benchmark_net_ret, horizon=63)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Compara global misto, Brasil+cripto e Brasil puro com o motor oficial pos-fiscal.")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--capital-brl", type=float, default=10000.0)
    ap.add_argument("--outdir-root", default="results/validation/profit_country_compare_suite")
    args = ap.parse_args()

    outdir = (ROOT / args.outdir_root / _run_id()).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    prices_dir = (ROOT / args.prices_dir).resolve()

    br_groups, br_meta, br_tickers = _filter_brazil_equities(
        equity_groups=(ROOT / args.equity_asset_groups).resolve(),
        equity_meta=(ROOT / args.equity_asset_metadata).resolve(),
        outdir=outdir,
    )
    br_prices_dir = outdir / "prices_brazil"
    _link_prices_dir(source_dir=prices_dir, target_dir=br_prices_dir)
    br_benchmark_ticker = "BR_SYNTH"
    _write_synthetic_benchmark(prices_dir=prices_dir, tickers=br_tickers, outdir=br_prices_dir, benchmark_ticker=br_benchmark_ticker)

    built_global = _build_candidates(
        prices_dir=prices_dir,
        crypto_groups=(ROOT / args.crypto_asset_groups).resolve(),
        crypto_meta=(ROOT / args.crypto_asset_metadata).resolve(),
        equity_groups=(ROOT / args.equity_asset_groups).resolve(),
        equity_meta=(ROOT / args.equity_asset_metadata).resolve(),
        benchmark_crypto=str(args.benchmark_crypto),
        benchmark_equity=str(args.benchmark_equity),
    )
    global_baseline, global_official = _build_official_bundle(built_global)

    built_br = _build_candidates(
        prices_dir=br_prices_dir,
        crypto_groups=(ROOT / args.crypto_asset_groups).resolve(),
        crypto_meta=(ROOT / args.crypto_asset_metadata).resolve(),
        equity_groups=br_groups,
        equity_meta=br_meta,
        benchmark_crypto=str(args.benchmark_crypto),
        benchmark_equity=br_benchmark_ticker,
        equity_profile_id="br_local_equity",
    )
    br_crypto_baseline, br_crypto_official = _build_official_bundle(built_br)
    br_equity_base = built_br["context"]["equity_base"].result
    br_equity_attack = built_br["context"]["equity_attack"].result

    rows = [
        _scenario_row(
            scenario="global_mixed",
            label="global_mixed_official",
            result=global_official,
            baseline=global_baseline,
        ),
        _scenario_row(
            scenario="brazil_crypto",
            label="brazil_crypto_official",
            result=br_crypto_official,
            baseline=br_crypto_baseline,
        ),
        _scenario_row(
            scenario="brazil_only",
            label="brazil_only_equity_base",
            result=br_equity_base,
            baseline=br_equity_base,
        ),
        _scenario_row(
            scenario="brazil_only",
            label="brazil_only_equity_attack",
            result=br_equity_attack,
            baseline=br_equity_base,
        ),
    ]
    compare_df = pd.DataFrame(rows)
    compare_df.to_csv(outdir / "candidate_compare.csv", index=False)

    brazil_only_df = compare_df.loc[compare_df["scenario"] == "brazil_only"].copy()
    brazil_only_best = brazil_only_df.sort_values(["net_total_return", "sharpe"], ascending=[False, False]).iloc[0].to_dict()
    global_row = compare_df.loc[compare_df["scenario_label"] == "global_mixed_official"].iloc[0].to_dict()
    br_crypto_row = compare_df.loc[compare_df["scenario_label"] == "brazil_crypto_official"].iloc[0].to_dict()

    calendar_rows: list[dict[str, Any]] = []
    for scenario, label, result in [
        ("global_mixed", "global_mixed_official", global_official),
        ("brazil_crypto", "brazil_crypto_official", br_crypto_official),
        ("brazil_only", "brazil_only_equity_base", br_equity_base),
        ("brazil_only", "brazil_only_equity_attack", br_equity_attack),
    ]:
        for row in _calendar_rows(result=result, capital_brl=float(args.capital_brl)):
            row["scenario"] = scenario
            row["scenario_label"] = label
            calendar_rows.append(row)
    pd.DataFrame(calendar_rows).sort_values(["scenario", "year", "scenario_label"]).to_csv(outdir / "yearbook_reais.csv", index=False)

    summary = {
        "suite": "profit_country_compare_suite",
        "global_equity_count": int(len(built_global["context"]["equity_assets"])),
        "brazil_equity_count": int(len(br_tickers)),
        "brazil_benchmark_ticker": br_benchmark_ticker,
        "global_mixed_official": global_row,
        "brazil_crypto_official": br_crypto_row,
        "brazil_only_best": brazil_only_best,
        "recommendation": {
            "best_net_total_return_label": str(compare_df.sort_values("net_total_return", ascending=False).iloc[0]["scenario_label"]),
            "best_sharpe_label": str(compare_df.sort_values("sharpe", ascending=False).iloc[0]["scenario_label"]),
        },
    }
    _write_json(outdir / "summary.json", summary)
    write_run_manifest(
        outdir / "RUN_MANIFEST.json",
        {
            "suite": "profit_country_compare_suite",
            "equity_groups": str(args.equity_asset_groups),
            "equity_meta": str(args.equity_asset_metadata),
            "crypto_groups": str(args.crypto_asset_groups),
            "crypto_meta": str(args.crypto_asset_metadata),
            "prices_dir": str(args.prices_dir),
            "benchmark_equity_global": str(args.benchmark_equity),
            "benchmark_equity_brazil": br_benchmark_ticker,
            "capital_brl": float(args.capital_brl),
        },
    )
    print(f"[ok] country compare at {outdir}")


if __name__ == "__main__":
    main()
