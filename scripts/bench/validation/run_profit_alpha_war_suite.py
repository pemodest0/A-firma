#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.structural.run_manifest import write_run_manifest  # noqa: E402
from execution.net_assumptions import load_net_assumption_profiles  # noqa: E402
from scripts.bench.validation.run_profit_frontier_expansion_suite import (  # noqa: E402
    EQUITY_EXCLUDED,
    StrategyResult,
    _ensure_benchmark_columns,
    _load_asset_table,
    _load_daily_universe,
    _select_crypto_tiers,
    _simulate_asset_rule,
    _write_json,
)
from scripts.bench.validation.run_profit_layered_engine_suite import (  # noqa: E402
    StrategyBundle,
    _build_equity_group_map,
    _load_structural_regime_series_local,
    _simulate_equity_group_sleeve_v2,
    _simulate_equity_group_sleeve_v3,
    _simulate_equity_trail_switch_bundle,
)
from scripts.bench.validation.run_profit_regime_simulation_suite import (  # noqa: E402
    _blended_profile,
    _build_meta_v1_allocation,
    _evaluate_allocation_candidate,
)
from scripts.bench.validation.run_profit_sector_pressure_suite import (  # noqa: E402
    _research_row,
    _result_row,
    _simulate_equity_group_sleeve_v4_sector_pressure,
    _simulate_equity_group_sleeve_v5_hybrid_rank,
)


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _tail_return(series: pd.Series, lookback: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    min_periods = max(10, int(lookback) // 3)
    return (1.0 + values).rolling(int(lookback), min_periods=min_periods).apply(np.prod, raw=True) - 1.0


def _build_alpha_meta_allocation(
    *,
    candidate_id: str,
    crypto_bundle: StrategyBundle,
    equity_bundle: StrategyBundle,
    btc_prices: pd.Series,
    spy_prices: pd.Series,
    profile,
    entry_lookback: int,
    exit_lookback: int,
    entry_margin: float,
    exit_margin: float,
    risk_off_mode: str,
    min_crypto_hold_days: int,
) -> StrategyBundle:
    idx = (
        crypto_bundle.result.gross_ret.index.intersection(equity_bundle.result.gross_ret.index)
        .intersection(btc_prices.index)
        .intersection(spy_prices.index)
    )
    crypto_ret = pd.to_numeric(crypto_bundle.result.gross_ret.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    equity_ret = pd.to_numeric(equity_bundle.result.gross_ret.reindex(idx), errors="coerce").fillna(0.0).astype(float)
    btc = pd.to_numeric(btc_prices.reindex(idx), errors="coerce").astype(float)
    spy = pd.to_numeric(spy_prices.reindex(idx), errors="coerce").astype(float)

    btc_ok = (btc.shift(1) > btc.shift(1).rolling(200, min_periods=100).mean()).fillna(False)
    spy_ok = (spy.shift(1) > spy.shift(1).rolling(200, min_periods=100).mean()).fillna(False)
    crypto_entry = _tail_return(crypto_ret, int(entry_lookback))
    equity_entry = _tail_return(equity_ret, int(entry_lookback))
    crypto_exit = _tail_return(crypto_ret, int(exit_lookback))
    equity_exit = _tail_return(equity_ret, int(exit_lookback))

    weights = pd.DataFrame(0.0, index=idx, columns=["crypto", "equity", "cash"], dtype=float)
    source = pd.Series(index=idx, dtype=object)
    state = "cash"
    hold_days = 0

    for dt in idx:
        btc_good = bool(btc_ok.loc[dt])
        spy_good = bool(spy_ok.loc[dt])
        both_bad = not btc_good and not spy_good
        ce = float(crypto_entry.loc[dt]) if pd.notna(crypto_entry.loc[dt]) else -1.0
        ee = float(equity_entry.loc[dt]) if pd.notna(equity_entry.loc[dt]) else -1.0
        cx = float(crypto_exit.loc[dt]) if pd.notna(crypto_exit.loc[dt]) else -1.0
        ex = float(equity_exit.loc[dt]) if pd.notna(equity_exit.loc[dt]) else -1.0

        choose = state
        if both_bad:
            if risk_off_mode == "equity25":
                weights.loc[dt, "equity"] = 0.25
                weights.loc[dt, "cash"] = 0.75
                choose = "equity25"
            elif risk_off_mode == "equity50":
                weights.loc[dt, "equity"] = 0.50
                weights.loc[dt, "cash"] = 0.50
                choose = "equity50"
            else:
                weights.loc[dt, "cash"] = 1.0
                choose = "cash"
            state = choose
            hold_days = 0
            source.loc[dt] = choose
            continue

        enter_crypto = btc_good and (ce > ee + float(entry_margin))
        exit_crypto = spy_good and (ex >= cx + float(exit_margin))

        if state == "crypto":
            hold_days += 1
            if hold_days < int(min_crypto_hold_days) and btc_good:
                choose = "crypto"
            elif not btc_good and spy_good:
                choose = "equity"
                hold_days = 0
            elif exit_crypto:
                choose = "equity"
                hold_days = 0
            else:
                choose = "crypto"
        else:
            if enter_crypto:
                choose = "crypto"
                hold_days = 1
            elif spy_good:
                choose = "equity"
                hold_days = 0
            elif btc_good:
                choose = "crypto"
                hold_days = 1
            else:
                choose = "cash"
                hold_days = 0

        if choose == "crypto":
            weights.loc[dt, "crypto"] = 1.0
        elif choose == "equity":
            weights.loc[dt, "equity"] = 1.0
        elif choose == "equity25":
            weights.loc[dt, "equity"] = 0.25
            weights.loc[dt, "cash"] = 0.75
        elif choose == "equity50":
            weights.loc[dt, "equity"] = 0.50
            weights.loc[dt, "cash"] = 0.50
        else:
            weights.loc[dt, "cash"] = 1.0
        state = choose
        source.loc[dt] = choose

    btc_bench = btc.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    spy_bench = spy.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    bench = 0.5 * btc_bench + 0.5 * spy_bench
    return _evaluate_allocation_candidate(
        candidate_id=candidate_id,
        family="meta_switch_alpha_search",
        weights=weights,
        crypto_ret=crypto_ret,
        equity_ret=equity_ret,
        benchmark_ret=bench,
        profile=profile,
        benchmark_profile=profile,
        notes=(
            f"entry_lb={entry_lookback};exit_lb={exit_lookback};entry_margin={entry_margin:.2f};"
            f"exit_margin={exit_margin:.2f};risk_off={risk_off_mode};hold={min_crypto_hold_days}"
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Busca focada para maximizar lucro final no melhor meta atual.")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--outdir-root", default="results/validation/profit_alpha_war_suite")
    args = ap.parse_args()

    outdir = (ROOT / args.outdir_root / _run_id()).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    prices_dir = (ROOT / args.prices_dir).resolve()

    profiles = load_net_assumption_profiles(ROOT / "config" / "profit_net_assumptions.json")
    foreign_profile = profiles["profiles"]["foreign_financial_brazil_resident"]
    crypto_profile = profiles["profiles"]["crypto_global_brazil_resident_conservative"]
    blended_profile = _blended_profile(
        crypto_profile,
        foreign_profile,
        profile_id="alpha_war_blended",
        label="Alpha war blended",
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
    regime_series = _load_structural_regime_series_local(ROOT)

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
    if eq_a2 is None or eq_r1 is None:
        raise SystemExit("falha ao reconstruir a base de equities")

    equity_bundles: list[StrategyBundle] = [
        _simulate_equity_trail_switch_bundle(
            candidate_id="equities_meta__trail_switch__a2__r1",
            aggressive_bundle=eq_a2,
            robust_bundle=eq_r1,
            regime_series=regime_series,
            spy_prices=pd.to_numeric(equity_prices[str(args.benchmark_equity)], errors="coerce"),
        )
    ]

    eq_sp, _ = _simulate_equity_group_sleeve_v4_sector_pressure(
        candidate_id="equities_v4__sector_pressure_p25",
        returns=equity_returns,
        prices=equity_prices,
        asset_table=equity_assets,
        equity_groups=equity_group_map,
        benchmark_ticker=str(args.benchmark_equity),
        regime_series=regime_series,
        group_lookback_fast=63,
        group_lookback_slow=189,
        group_top_k=4,
        assets_per_group=1,
        asset_lookback=126,
        asset_ma_days=200,
        market_ma_days=200,
        pressure_lookback=120,
        pressure_horizon=21,
        pressure_penalty=0.25,
        profile=foreign_profile,
        benchmark_profile=foreign_profile,
    )
    if eq_sp is not None:
        equity_bundles.append(
            _simulate_equity_trail_switch_bundle(
                candidate_id="equities_meta__trail_switch__sector_p25",
                aggressive_bundle=eq_sp,
                robust_bundle=eq_r1,
                regime_series=regime_series,
                spy_prices=pd.to_numeric(equity_prices[str(args.benchmark_equity)], errors="coerce"),
            )
        )

    eq_hybrid, _ = _simulate_equity_group_sleeve_v5_hybrid_rank(
        candidate_id="equities_v5__hybrid_p15_s15",
        returns=equity_returns,
        prices=equity_prices,
        asset_table=equity_assets,
        equity_groups=equity_group_map,
        benchmark_ticker=str(args.benchmark_equity),
        regime_series=regime_series,
        group_lookback_fast=63,
        group_lookback_slow=189,
        group_top_k=4,
        assets_per_group=1,
        asset_lookback=126,
        asset_ma_days=200,
        market_ma_days=200,
        pressure_lookback=120,
        pressure_horizon=21,
        pressure_penalty=0.15,
        systemic_penalty=0.15,
        profile=foreign_profile,
        benchmark_profile=foreign_profile,
    )
    if eq_hybrid is not None:
        equity_bundles.append(
            _simulate_equity_trail_switch_bundle(
                candidate_id="equities_meta__trail_switch__hybrid_p15_s15",
                aggressive_bundle=eq_hybrid,
                robust_bundle=eq_r1,
                regime_series=regime_series,
                spy_prices=pd.to_numeric(equity_prices[str(args.benchmark_equity)], errors="coerce"),
            )
        )

    crypto_results: list[StrategyResult] = []
    for score_mode in ["mom_total", "mom_vol_adj"]:
        for lookback_days in [21, 42, 63]:
            for rebalance_days in [7, 14, 21]:
                for top_k in [1, 2, 3]:
                    candidate_id = f"crypto_major8__{score_mode}_lb{lookback_days:03d}_rb{rebalance_days:02d}_k{top_k}"
                    result = _simulate_asset_rule(
                        candidate_id=candidate_id,
                        family="crypto_major8_search",
                        allowed_tickers=crypto_tiers["crypto_major8"],
                        returns=crypto_returns,
                        prices=crypto_prices,
                        asset_table=crypto_assets,
                        benchmark_ticker=str(args.benchmark_crypto),
                        fallback_ticker=str(args.benchmark_crypto),
                        score_mode=score_mode,
                        lookback_days=lookback_days,
                        rebalance_days=rebalance_days,
                        top_k=top_k,
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
                    if result is not None:
                        crypto_results.append(result)

    if not crypto_results:
        raise SystemExit("falha ao gerar sleeves de cripto")

    crypto_df = pd.DataFrame([_result_row(result) for result in crypto_results]).sort_values(
        ["net_total_return", "net_ann_return", "net_sharpe"],
        ascending=[False, False, False],
    )
    crypto_df.to_csv(outdir / "crypto_sleeve_compare.csv", index=False)

    top_crypto_ids = crypto_df.head(2)["candidate_id"].astype(str).tolist()
    crypto_lookup = {result.candidate_id: result for result in crypto_results if result.candidate_id in set(top_crypto_ids)}

    btc_prices = pd.to_numeric(crypto_prices[str(args.benchmark_crypto)], errors="coerce")
    spy_prices = pd.to_numeric(equity_prices[str(args.benchmark_equity)], errors="coerce")

    baseline_crypto = next((result for result in crypto_results if result.candidate_id == "crypto_major8__mom_vol_adj_lb021_rb07_k3"), None)
    if baseline_crypto is None:
        baseline_crypto = crypto_results[0]
    baseline_crypto_bundle = StrategyBundle(
        result=baseline_crypto,
        benchmark_gross_ret=pd.to_numeric(crypto_returns[str(args.benchmark_crypto)], errors="coerce").reindex(baseline_crypto.gross_ret.index).fillna(0.0).astype(float),
        profile=crypto_profile,
        benchmark_profile=crypto_profile,
    )
    baseline_equity = equity_bundles[0]
    baseline = _build_meta_v1_allocation(
        crypto_bundle=baseline_crypto_bundle,
        equity_bundle=baseline_equity,
        btc_prices=btc_prices,
        spy_prices=spy_prices,
        profile=blended_profile,
    )
    baseline = replace(
        baseline,
        bundle=replace(
            baseline.bundle,
            result=replace(
                baseline.bundle.result,
                candidate_id="meta_major8_eq_a2r1",
                notes="baseline atual de lucro maximo",
            ),
        ),
    )

    candidates: list[StrategyResult] = [baseline.bundle.result]
    research_rows: list[dict[str, Any]] = [
        _research_row(
            baseline.bundle.result,
            outdir=outdir,
            status="keep",
            methodology="alpha_war_baseline",
            label="Base atual de lucro maximo",
        )
    ]

    for crypto_id in top_crypto_ids:
        crypto_result = crypto_lookup[crypto_id]
        crypto_bundle = StrategyBundle(
            result=crypto_result,
            benchmark_gross_ret=pd.to_numeric(crypto_returns[str(args.benchmark_crypto)], errors="coerce").reindex(crypto_result.gross_ret.index).fillna(0.0).astype(float),
            profile=crypto_profile,
            benchmark_profile=crypto_profile,
        )
        for equity_bundle in equity_bundles[:2]:
            for entry_lookback in [21]:
                for exit_lookback in [63]:
                    for entry_margin in [0.00, 0.05]:
                        for exit_margin in [0.05]:
                            for risk_off_mode in ["cash", "equity25"]:
                                for min_hold in [0, 10]:
                                    cid = (
                                        f"alpha__{crypto_id}__{equity_bundle.result.candidate_id}"
                                        f"__e{entry_lookback}_x{exit_lookback}"
                                        f"__m{int(round(entry_margin*100)):02d}{int(round(exit_margin*100)):02d}"
                                        f"__{risk_off_mode}__h{min_hold:02d}"
                                    )
                                    bundle = _build_alpha_meta_allocation(
                                        candidate_id=cid,
                                        crypto_bundle=crypto_bundle,
                                        equity_bundle=equity_bundle,
                                        btc_prices=btc_prices,
                                        spy_prices=spy_prices,
                                        profile=blended_profile,
                                        entry_lookback=entry_lookback,
                                        exit_lookback=exit_lookback,
                                        entry_margin=entry_margin,
                                        exit_margin=exit_margin,
                                        risk_off_mode=risk_off_mode,
                                        min_crypto_hold_days=min_hold,
                                    )
                                    candidates.append(bundle.result)

    candidate_df = pd.DataFrame([_result_row(result) for result in candidates]).sort_values(
        ["net_total_return", "net_ann_return", "net_sharpe"],
        ascending=[False, False, False],
    )
    candidate_df.to_csv(outdir / "candidate_compare.csv", index=False)

    best = candidate_df.iloc[0].to_dict() if not candidate_df.empty else {}
    baseline_row = candidate_df[candidate_df["candidate_id"] == "meta_major8_eq_a2r1"].iloc[0].to_dict()
    improved = bool(
        best
        and str(best.get("candidate_id", "")) != "meta_major8_eq_a2r1"
        and float(best.get("net_total_return", float("-inf"))) > float(baseline_row.get("net_total_return", float("-inf")))
    )

    top_rows = candidate_df.head(10).to_dict(orient="records")
    for i, row in enumerate(top_rows):
        status = "watch"
        if i == 0 and improved:
            status = "keep"
        elif row.get("candidate_id") == "meta_major8_eq_a2r1":
            status = "keep" if not improved else "watch"
        research_rows.append(
            _research_row(
                next(result for result in candidates if result.candidate_id == row["candidate_id"]),
                outdir=outdir,
                status=status,
                methodology="alpha_war_search",
                label=f"Busca focada de lucro {i + 1}",
            )
        )

    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "outdir": str(outdir),
        "baseline_candidate": baseline_row,
        "best_candidate": best,
        "improved_over_baseline": improved,
        "search_space": {
            "top_crypto_candidates_kept": top_crypto_ids,
            "equity_bundles": [bundle.result.candidate_id for bundle in equity_bundles[:2]],
            "entry_lookbacks": [21],
            "exit_lookbacks": [63],
            "entry_margins": [0.00, 0.05],
            "exit_margins": [0.05],
            "risk_off_modes": ["cash", "equity25"],
            "min_crypto_hold_days": [0, 10],
        },
        "insights": [
            "Busca focada em lucro puro no melhor stack já provado.",
            "O search variou só o que ainda parecia promissor: perna cripto, tempo em caixa e troca entre cripto e ações.",
            "O objetivo principal desta rodada foi lucro final, não suavização de risco.",
        ],
        "artifacts": {
            "candidate_compare_csv": str(outdir / "candidate_compare.csv"),
            "crypto_sleeve_compare_csv": str(outdir / "crypto_sleeve_compare.csv"),
            "profit_research_rows_json": str(outdir / "profit_research_rows.json"),
        },
    }
    _write_json(outdir / "summary.json", summary)
    (outdir / "profit_research_rows.json").write_text(json.dumps(research_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_run_manifest(
        outdir,
        script="scripts/bench/validation/run_profit_alpha_war_suite.py",
        params=summary["search_space"],
        paths=summary["artifacts"],
        extra={"summary_json": str(outdir / "summary.json")},
    )


if __name__ == "__main__":
    main()
