#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.structural.run_manifest import write_run_manifest  # noqa: E402
from execution.net_assumptions import NetAssumptionProfile, load_net_assumption_profiles  # noqa: E402
from scripts.bench.validation.run_profit_frontier_expansion_suite import (  # noqa: E402
    EQUITY_EXCLUDED,
    StrategyResult,
    _build_equity_group_map,
    _ensure_benchmark_columns,
    _evaluate_net,
    _load_asset_table,
    _load_daily_universe,
    _select_crypto_tiers,
    _simulate_asset_rule,
    _write_json,
)
from scripts.bench.validation.run_profit_layered_engine_suite import (  # noqa: E402
    StrategyBundle,
    _load_structural_regime_series_local,
    _simulate_equity_group_sleeve_v2,
    _simulate_equity_group_sleeve_v3,
    _simulate_equity_trail_switch_bundle,
)
from scripts.bench.validation.run_profit_regime_simulation_suite import (  # noqa: E402
    _apply_mc_guard,
    _blended_profile,
    _build_hmm_meta_allocation,
    _build_meta_hrp_allocation,
    _build_meta_v1_allocation,
)


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _result_row(result: StrategyResult) -> dict[str, Any]:
    return {
        "candidate_id": result.candidate_id,
        "suite": result.suite,
        "family": result.family,
        "benchmark_ticker": result.benchmark_ticker,
        "net_ann_return": result.net_ann_return,
        "net_total_return": result.net_total_return,
        "net_sharpe": result.net_sharpe,
        "net_max_drawdown": result.net_max_drawdown,
        "edge_vs_benchmark_net_total_return": result.edge_vs_benchmark,
        "avg_turnover_daily": result.avg_turnover_daily,
        "notes": result.notes,
    }


def _calendar_rows(
    *,
    result: StrategyResult,
    capital_brl: float,
    operation_threshold: float = 1e-8,
) -> list[dict[str, Any]]:
    ret = pd.to_numeric(result.net_ret, errors="coerce").dropna().astype(float)
    bench = pd.to_numeric(result.benchmark_net_ret, errors="coerce").reindex(ret.index).fillna(0.0).astype(float)
    turnover = pd.to_numeric(result.turnover, errors="coerce").reindex(ret.index).fillna(0.0).astype(float)
    if ret.empty:
        return []
    wealth = (1.0 + ret).cumprod()
    bench_wealth = (1.0 + bench).cumprod()
    rows: list[dict[str, Any]] = []
    for year, sub in ret.groupby(ret.index.year):
        idx = sub.index
        year_turnover = turnover.loc[idx]
        year_bench = bench.loc[idx]
        year_total = float(np.prod(1.0 + sub.to_numpy(dtype=float)) - 1.0)
        bench_total = float(np.prod(1.0 + year_bench.to_numpy(dtype=float)) - 1.0)
        start_wealth = float(wealth.shift(1).reindex(idx).ffill().iloc[0]) if idx[0] != wealth.index[0] else 1.0
        end_wealth = float(wealth.loc[idx[-1]])
        start_capital = float(capital_brl * start_wealth)
        end_capital = float(capital_brl * end_wealth)
        rows.append(
            {
                "candidate_id": result.candidate_id,
                "year": int(year),
                "days": int(len(idx)),
                "year_total_return": year_total,
                "benchmark_total_return": bench_total,
                "edge_total_return": year_total - bench_total,
                "starting_capital_brl": start_capital,
                "ending_capital_brl": end_capital,
                "profit_brl": end_capital - start_capital,
                "operation_days": int((year_turnover.abs() > float(operation_threshold)).sum()),
                "turnover_sum": float(year_turnover.abs().sum()),
                "avg_turnover_daily": float(year_turnover.mean()) if not year_turnover.empty else float("nan"),
            }
        )
    return rows


def _buy_hold_result(
    *,
    candidate_id: str,
    ticker: str,
    returns: pd.Series,
    profile: NetAssumptionProfile,
) -> StrategyResult:
    gross = pd.to_numeric(returns, errors="coerce").fillna(0.0).astype(float)
    turnover = pd.Series(np.zeros(len(gross), dtype=float), index=gross.index, dtype=float)
    perf = _evaluate_net(
        gross_ret=gross,
        turnover=turnover,
        profile=profile,
        benchmark_ret=gross,
        benchmark_profile=profile,
    )
    return StrategyResult(
        suite="buy_hold",
        candidate_id=candidate_id,
        family="buy_hold",
        benchmark_ticker=ticker,
        gross_ret=gross,
        turnover=turnover,
        net_ret=perf["net_ret"],
        benchmark_net_ret=perf["benchmark_net_ret"],
        net_ann_return=float(perf["net_ann_return"]),
        net_total_return=float(perf["net_total_return"]),
        net_sharpe=float(perf["net_sharpe"]),
        net_max_drawdown=float(perf["net_max_drawdown"]),
        edge_vs_benchmark=0.0,
        avg_turnover_daily=0.0,
        hit_rate_10x_5y=float("nan"),
        years_to_10x_full=float("nan"),
        notes=f"buy and hold {ticker}",
    )


def _rename_result(result: StrategyResult, *, candidate_id: str, family: str | None = None, notes_prefix: str | None = None) -> StrategyResult:
    notes = result.notes
    if notes_prefix:
        notes = f"{notes_prefix}; {notes}" if notes else str(notes_prefix)
    return replace(
        result,
        candidate_id=str(candidate_id),
        family=str(family or result.family),
        notes=notes,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Yearbook de lucro e operacoes por ano para o stack atual e variantes de ativos.")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--net-assumptions", default="config/profit_net_assumptions.json")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--capital-brl", type=float, default=10000.0)
    ap.add_argument("--outdir-root", default="results/validation/profit_investment_yearbook")
    args = ap.parse_args()

    outdir = (ROOT / args.outdir_root / _run_id()).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    prices_dir = (ROOT / args.prices_dir).resolve()

    profiles = load_net_assumption_profiles((ROOT / args.net_assumptions).resolve())
    foreign_profile = profiles["profiles"]["foreign_financial_brazil_resident"]
    crypto_profile = profiles["profiles"]["crypto_global_brazil_resident_conservative"]
    blended_profile = _blended_profile(
        crypto_profile,
        foreign_profile,
        profile_id="yearbook_blended",
        label="Yearbook blended",
    )

    crypto_assets = _load_asset_table((ROOT / args.crypto_asset_groups).resolve(), (ROOT / args.crypto_asset_metadata).resolve())
    crypto_returns, crypto_prices, crypto_viability = _load_daily_universe(
        prices_dir=prices_dir,
        asset_table=crypto_assets,
        min_history_days=600,
        max_abs_daily_return=1.5,
    )
    crypto_returns, crypto_prices = _ensure_benchmark_columns(
        crypto_returns,
        crypto_prices,
        prices_dir,
        [str(args.benchmark_crypto), "ETH-USD"],
    )
    crypto_tiers = _select_crypto_tiers(crypto_assets, crypto_viability)

    equity_assets = _load_asset_table((ROOT / args.equity_asset_groups).resolve(), (ROOT / args.equity_asset_metadata).resolve())
    equity_assets = equity_assets[~equity_assets["asset_group"].astype(str).isin(EQUITY_EXCLUDED)].copy()
    equity_returns, equity_prices, _ = _load_daily_universe(
        prices_dir=prices_dir,
        asset_table=equity_assets,
        min_history_days=1200,
        max_abs_daily_return=0.8,
    )
    equity_returns, equity_prices = _ensure_benchmark_columns(
        equity_returns,
        equity_prices,
        prices_dir,
        [str(args.benchmark_equity)],
    )
    equity_group_map = _build_equity_group_map(equity_assets, equity_returns)

    crypto_all = _simulate_asset_rule(
        candidate_id="crypto_all__momvol21_hard15",
        family="crypto",
        allowed_tickers=crypto_tiers["crypto_all"],
        returns=crypto_returns,
        prices=crypto_prices,
        asset_table=crypto_assets,
        benchmark_ticker=str(args.benchmark_crypto),
        fallback_ticker=str(args.benchmark_crypto),
        score_mode="mom_vol_adj",
        lookback_days=21,
        rebalance_days=7,
        top_k=3,
        asset_ma_days=0,
        market_ma_days=200,
        relative_to_benchmark=False,
        skip_recent_days=0,
        trailing_stop_dd=None,
        hard_stop_loss=0.15,
        stop_to_cash=True,
        profile=crypto_profile,
        benchmark_profile=crypto_profile,
    )
    crypto_major = _simulate_asset_rule(
        candidate_id="crypto_major8__momvol21",
        family="crypto",
        allowed_tickers=crypto_tiers["crypto_major8"],
        returns=crypto_returns,
        prices=crypto_prices,
        asset_table=crypto_assets,
        benchmark_ticker=str(args.benchmark_crypto),
        fallback_ticker=str(args.benchmark_crypto),
        score_mode="mom_vol_adj",
        lookback_days=21,
        rebalance_days=7,
        top_k=3,
        asset_ma_days=0,
        market_ma_days=200,
        relative_to_benchmark=False,
        skip_recent_days=0,
        trailing_stop_dd=None,
        hard_stop_loss=None,
        stop_to_cash=True,
        profile=crypto_profile,
        benchmark_profile=crypto_profile,
    )
    if crypto_all is None or crypto_major is None:
        raise SystemExit("failed to rebuild crypto sleeves")
    crypto_all_bundle = StrategyBundle(
        result=crypto_all,
        benchmark_gross_ret=pd.to_numeric(crypto_returns[str(args.benchmark_crypto)], errors="coerce").fillna(0.0).astype(float),
        profile=crypto_profile,
        benchmark_profile=crypto_profile,
    )
    crypto_major_bundle = StrategyBundle(
        result=crypto_major,
        benchmark_gross_ret=pd.to_numeric(crypto_returns[str(args.benchmark_crypto)], errors="coerce").fillna(0.0).astype(float),
        profile=crypto_profile,
        benchmark_profile=crypto_profile,
    )

    eq_a1 = _simulate_equity_group_sleeve_v2(
        candidate_id="equities_v2__slow189__g3__a1",
        returns=equity_returns,
        prices=equity_prices,
        asset_table=equity_assets,
        equity_groups=equity_group_map,
        benchmark_ticker=str(args.benchmark_equity),
        group_lookback_fast=63,
        group_lookback_slow=189,
        group_top_k=3,
        assets_per_group=1,
        asset_lookback=126,
        asset_ma_days=200,
        market_ma_days=200,
        profile=foreign_profile,
        benchmark_profile=foreign_profile,
    )
    eq_r3 = _simulate_equity_group_sleeve_v3(
        candidate_id="equities_v3__slow252__g3__a2__br30__cap45",
        returns=equity_returns,
        prices=equity_prices,
        asset_table=equity_assets,
        equity_groups=equity_group_map,
        benchmark_ticker=str(args.benchmark_equity),
        group_lookback_fast=63,
        group_lookback_slow=252,
        group_top_k=3,
        assets_per_group=2,
        asset_lookback=126,
        asset_ma_days=200,
        market_ma_days=200,
        min_group_breadth=0.30,
        max_group_weight=0.45,
        profile=foreign_profile,
        benchmark_profile=foreign_profile,
    )
    eq_a2 = _simulate_equity_group_sleeve_v2(
        candidate_id="equities_v2__slow189__g4__a1",
        returns=equity_returns,
        prices=equity_prices,
        asset_table=equity_assets,
        equity_groups=equity_group_map,
        benchmark_ticker=str(args.benchmark_equity),
        group_lookback_fast=63,
        group_lookback_slow=189,
        group_top_k=4,
        assets_per_group=1,
        asset_lookback=126,
        asset_ma_days=200,
        market_ma_days=200,
        profile=foreign_profile,
        benchmark_profile=foreign_profile,
    )
    eq_r1 = _simulate_equity_group_sleeve_v3(
        candidate_id="equities_v3__slow189__g3__a2__br35__cap40",
        returns=equity_returns,
        prices=equity_prices,
        asset_table=equity_assets,
        equity_groups=equity_group_map,
        benchmark_ticker=str(args.benchmark_equity),
        group_lookback_fast=63,
        group_lookback_slow=189,
        group_top_k=3,
        assets_per_group=2,
        asset_lookback=126,
        asset_ma_days=200,
        market_ma_days=200,
        min_group_breadth=0.35,
        max_group_weight=0.40,
        profile=foreign_profile,
        benchmark_profile=foreign_profile,
    )
    if eq_a1 is None or eq_r3 is None or eq_a2 is None or eq_r1 is None:
        raise SystemExit("failed to rebuild equity sleeves")

    regime_series = _load_structural_regime_series_local(ROOT)
    equity_meta_a1r3 = _simulate_equity_trail_switch_bundle(
        candidate_id="equities_meta__trail_switch__a1__r3",
        aggressive_bundle=eq_a1,
        robust_bundle=eq_r3,
        regime_series=regime_series,
        spy_prices=pd.to_numeric(equity_prices[str(args.benchmark_equity)], errors="coerce"),
    )
    equity_meta_a2r1 = _simulate_equity_trail_switch_bundle(
        candidate_id="equities_meta__trail_switch__a2__r1",
        aggressive_bundle=eq_a2,
        robust_bundle=eq_r1,
        regime_series=regime_series,
        spy_prices=pd.to_numeric(equity_prices[str(args.benchmark_equity)], errors="coerce"),
    )

    btc_prices = pd.to_numeric(crypto_prices[str(args.benchmark_crypto)], errors="coerce")
    spy_prices = pd.to_numeric(equity_prices[str(args.benchmark_equity)], errors="coerce")

    candidates: list[StrategyResult] = []
    baseline_all_a1r3 = _build_meta_v1_allocation(
        crypto_bundle=crypto_all_bundle,
        equity_bundle=equity_meta_a1r3,
        btc_prices=btc_prices,
        spy_prices=spy_prices,
        profile=blended_profile,
    )
    baseline_all_a2r1 = _build_meta_v1_allocation(
        crypto_bundle=crypto_all_bundle,
        equity_bundle=equity_meta_a2r1,
        btc_prices=btc_prices,
        spy_prices=spy_prices,
        profile=blended_profile,
    )
    baseline_major_a1r3 = _build_meta_v1_allocation(
        crypto_bundle=crypto_major_bundle,
        equity_bundle=equity_meta_a1r3,
        btc_prices=btc_prices,
        spy_prices=spy_prices,
        profile=blended_profile,
    )
    baseline_major_a2r1 = _build_meta_v1_allocation(
        crypto_bundle=crypto_major_bundle,
        equity_bundle=equity_meta_a2r1,
        btc_prices=btc_prices,
        spy_prices=spy_prices,
        profile=blended_profile,
    )
    mc_guard, _ = _apply_mc_guard(
        candidate_id="meta_mc_guard__regime21",
        base=baseline_all_a1r3,
        returns=pd.concat(
            {
                "crypto": pd.to_numeric(crypto_all_bundle.result.gross_ret, errors="coerce"),
                "equity": pd.to_numeric(equity_meta_a1r3.result.gross_ret, errors="coerce"),
            },
            axis=1,
            sort=False,
        ).dropna(how="all"),
        regime=regime_series,
        profile=blended_profile,
        lookback=252,
        horizon=21,
        n_paths=400,
        step=42,
    )
    hrp_bundle = _build_meta_hrp_allocation(
        crypto_bundle=crypto_all_bundle,
        equity_bundle=equity_meta_a1r3,
        btc_prices=btc_prices,
        spy_prices=spy_prices,
        profile=blended_profile,
        lookback=63,
    )
    hmm_bundle, _ = _build_hmm_meta_allocation(
        candidate_id="meta_hmm__btc_spy_challenger",
        crypto_bundle=crypto_all_bundle,
        equity_bundle=equity_meta_a1r3,
        btc_prices=btc_prices,
        spy_prices=spy_prices,
        profile=blended_profile,
    )

    candidates.extend(
        [
            _rename_result(
                baseline_all_a1r3.bundle.result,
                candidate_id="meta_all22_eq_a1r3",
                family="meta_switch_assets",
                notes_prefix="crypto_set=all22;equity_set=a1r3",
            ),
            _rename_result(
                baseline_all_a2r1.bundle.result,
                candidate_id="meta_all22_eq_a2r1",
                family="meta_switch_assets",
                notes_prefix="crypto_set=all22;equity_set=a2r1",
            ),
            _rename_result(
                baseline_major_a1r3.bundle.result,
                candidate_id="meta_major8_eq_a1r3",
                family="meta_switch_assets",
                notes_prefix="crypto_set=major8;equity_set=a1r3",
            ),
            _rename_result(
                baseline_major_a2r1.bundle.result,
                candidate_id="meta_major8_eq_a2r1",
                family="meta_switch_assets",
                notes_prefix="crypto_set=major8;equity_set=a2r1",
            ),
            _rename_result(
                mc_guard.bundle.result,
                candidate_id="meta_all22_eq_a1r3_mc_guard",
                family="meta_switch_mc_guard",
                notes_prefix="crypto_set=all22;equity_set=a1r3",
            ),
            _rename_result(
                hrp_bundle.bundle.result,
                candidate_id="meta_all22_eq_a1r3_hrp",
                family="meta_switch_hrp",
                notes_prefix="crypto_set=all22;equity_set=a1r3",
            ),
            _rename_result(
                hmm_bundle.bundle.result,
                candidate_id="meta_all22_eq_a1r3_hmm",
                family="meta_switch_hmm",
                notes_prefix="crypto_set=all22;equity_set=a1r3",
            ),
            _buy_hold_result(
                candidate_id="buy_hold__btc",
                ticker=str(args.benchmark_crypto),
                returns=pd.to_numeric(crypto_returns[str(args.benchmark_crypto)], errors="coerce").fillna(0.0).astype(float),
                profile=crypto_profile,
            ),
            _buy_hold_result(
                candidate_id="buy_hold__spy",
                ticker=str(args.benchmark_equity),
                returns=pd.to_numeric(equity_returns[str(args.benchmark_equity)], errors="coerce").fillna(0.0).astype(float),
                profile=foreign_profile,
            ),
        ]
    )

    candidate_df = pd.DataFrame([_result_row(result) for result in candidates]).sort_values(
        ["net_ann_return", "net_total_return", "net_sharpe"], ascending=[False, False, False]
    )
    candidate_df.to_csv(outdir / "candidate_compare.csv", index=False)

    year_df = pd.DataFrame(
        [row for result in candidates for row in _calendar_rows(result=result, capital_brl=float(args.capital_brl))]
    ).sort_values(["year", "profit_brl"], ascending=[True, False])
    year_df.to_csv(outdir / "calendar_year_operations.csv", index=False)

    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "outdir": str(outdir),
        "capital_brl": float(args.capital_brl),
        "best_total_profit_candidate": candidate_df.iloc[0].to_dict() if not candidate_df.empty else {},
        "best_candidates_tested": candidate_df["candidate_id"].drop_duplicates().tolist(),
        "insights": [
            "Comparativo anual montado com lucro líquido e dias com operação por ano.",
            "operation_days conta dias com turnover > 0, não número bruto de ordens individuais.",
            "Foram comparados sets de ativos cripto amplos vs major8 e duas pernas de equities.",
        ],
        "artifacts": {
            "candidate_compare_csv": str(outdir / "candidate_compare.csv"),
            "calendar_year_operations_csv": str(outdir / "calendar_year_operations.csv"),
        },
    }
    _write_json(outdir / "summary.json", summary)
    write_run_manifest(
        outdir,
        script="scripts/bench/validation/run_profit_investment_yearbook.py",
        params={
            "capital_brl": float(args.capital_brl),
            "benchmark_equity": args.benchmark_equity,
            "benchmark_crypto": args.benchmark_crypto,
        },
        paths=summary["artifacts"],
        extra={"summary_json": str(outdir / "summary.json")},
    )


if __name__ == "__main__":
    main()
