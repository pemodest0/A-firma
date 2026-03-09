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

from engine.portfolio import (  # noqa: E402
    estimate_regime_moments,
    estimate_transition_matrix,
    simulate_regime_conditioned_paths,
    summarize_portfolio_distribution,
)
from engine.structural.run_manifest import write_run_manifest  # noqa: E402
from scripts.bench.validation.run_profit_alpha_hardening_suite import (  # noqa: E402
    AllocationBundle,
    _build_alpha_meta_allocation_bundle,
    _build_candidates,
    _candidate_bundle,
)
from scripts.bench.validation.run_profit_frontier_expansion_suite import (  # noqa: E402
    EQUITY_EXCLUDED,
    _build_equity_group_map,
    _ensure_benchmark_columns,
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
)
from scripts.bench.validation.run_profit_sector_pressure_suite import (  # noqa: E402
    _blended_profile,
    _research_row,
    _simulate_equity_group_sleeve_v4_sector_pressure,
)
from execution.net_assumptions import load_net_assumption_profiles  # noqa: E402


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _human_label(candidate_id: str) -> str:
    mapping = {
        "meta_major8_eq_a2r1": "Modo principal de lucro",
        "alpha_attack_major8_equity25": "Modo ataque de lucro máximo",
        "meta_major8_eq_a2r1_mc_guard": "Modo principal com guarda de Monte Carlo",
        "alpha_attack_major8_equity25_mc_guard": "Modo ataque com guarda de Monte Carlo",
    }
    return mapping.get(str(candidate_id), str(candidate_id))


def _load_profiles() -> tuple[Any, Any, Any]:
    profiles = load_net_assumption_profiles(ROOT / "config" / "profit_net_assumptions.json")
    foreign_profile = profiles["profiles"]["foreign_financial_brazil_resident"]
    crypto_profile = profiles["profiles"]["crypto_global_brazil_resident_conservative"]
    blended_profile = _blended_profile(
        crypto_profile,
        foreign_profile,
        profile_id="universe_resilience_blended",
        label="Universe resilience blended",
    )
    return foreign_profile, crypto_profile, blended_profile


def _selection_frequency_for_crypto_rule(
    *,
    allowed_tickers: list[str],
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark_ticker: str,
    lookback_days: int,
    rebalance_days: int,
    top_k: int,
    score_mode: str,
    asset_ma_days: int,
    market_ma_days: int,
) -> pd.DataFrame:
    from scripts.bench.validation.run_profit_frontier_expansion_suite import _precompute_scores_skip, _top_k_indices  # noqa: E402

    all_tickers = list(returns.columns.astype(str))
    ticker_to_col = {ticker: idx for idx, ticker in enumerate(all_tickers)}
    allowed_idx = np.array([ticker_to_col[t] for t in allowed_tickers if t in ticker_to_col], dtype=int)
    if allowed_idx.size == 0:
        return pd.DataFrame(columns=["ticker", "rebalance_count"])

    score_map, asset_ma_filters, benchmark_filters = _precompute_scores_skip(
        returns,
        prices,
        lookbacks=[int(lookback_days)],
        asset_ma_days_list=[0, int(asset_ma_days), int(market_ma_days)],
        benchmark_ticker=benchmark_ticker,
        skip_recent_days=0,
    )
    score_df = score_map[(int(lookback_days), str(score_mode))]
    score_arr = score_df.reindex(index=returns.index, columns=all_tickers).to_numpy(dtype=float)
    asset_ma_arr = asset_ma_filters[int(asset_ma_days)].reindex(index=returns.index, columns=all_tickers).fillna(False).to_numpy(dtype=bool)
    benchmark_ok = benchmark_filters[int(market_ma_days)].reindex(returns.index).fillna(False).to_numpy(dtype=bool)
    ret_arr = returns.reindex(columns=all_tickers).to_numpy(dtype=float)
    warmup = max(int(lookback_days), int(asset_ma_days), int(market_ma_days)) + 2
    rebalance_positions = list(range(int(max(1, warmup)), ret_arr.shape[0], int(max(1, rebalance_days))))
    counts: dict[str, int] = {}
    for pos in rebalance_positions:
        score_row = score_arr[pos]
        valid = np.zeros(score_row.shape[0], dtype=bool)
        valid[allowed_idx] = True
        valid &= np.isfinite(score_row)
        valid &= asset_ma_arr[pos]
        valid &= score_row > 0.0
        if pos > 0:
            valid &= np.isfinite(ret_arr[pos - 1])
        if int(market_ma_days) > 0 and not bool(benchmark_ok[pos]):
            valid[:] = False
        selected_idx = _top_k_indices(score_row, valid, int(top_k))
        for idx in selected_idx:
            ticker = str(all_tickers[int(idx)])
            counts[ticker] = counts.get(ticker, 0) + 1
    return (
        pd.DataFrame([{"ticker": ticker, "rebalance_count": count} for ticker, count in counts.items()])
        .sort_values(["rebalance_count", "ticker"], ascending=[False, True])
        .reset_index(drop=True)
    )


def _random_keep(asset_table: pd.DataFrame, *, fraction: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    df = asset_table.drop_duplicates(subset=["ticker"], keep="first").copy()
    if df.empty:
        return df
    keep_n = max(1, int(round(float(fraction) * df.shape[0])))
    picks = rng.choice(df.index.to_numpy(dtype=int), size=keep_n, replace=False)
    return df.loc[sorted(picks)].reset_index(drop=True)


def _build_custom_candidates(
    *,
    prices_dir: Path,
    crypto_groups: Path,
    crypto_meta: Path,
    equity_groups: Path,
    equity_meta: Path,
    benchmark_crypto: str,
    benchmark_equity: str,
    crypto_drop_tickers: list[str] | None = None,
    equity_drop_sectors: list[str] | None = None,
    random_keep_fraction: float | None = None,
    random_seed: int = 7,
    crypto_allowed_mode: str = "major8",
) -> dict[str, Any]:
    foreign_profile, crypto_profile, blended_profile = _load_profiles()
    crypto_assets = _load_asset_table(crypto_groups, crypto_meta)
    if crypto_drop_tickers:
        drops = {str(x).strip() for x in crypto_drop_tickers if str(x).strip()}
        crypto_assets = crypto_assets[~crypto_assets["ticker"].astype(str).isin(drops)].copy()
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
        [str(benchmark_crypto), "ETH-USD"],
    )
    crypto_tiers = _select_crypto_tiers(crypto_assets, crypto_viability)
    if not crypto_tiers.get("crypto_major8"):
        raise SystemExit("crypto_major8 ficou vazio na perturbacao")

    equity_assets = _load_asset_table(equity_groups, equity_meta)
    equity_assets = equity_assets[~equity_assets["asset_group"].astype(str).isin(EQUITY_EXCLUDED)].copy()
    if equity_drop_sectors:
        drops = {str(x).strip() for x in equity_drop_sectors if str(x).strip()}
        equity_assets = equity_assets[~equity_assets["asset_group"].astype(str).isin(drops)].copy()
    if random_keep_fraction is not None:
        equity_assets = _random_keep(equity_assets, fraction=float(random_keep_fraction), seed=int(random_seed))
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
        [str(benchmark_equity)],
    )
    equity_group_map = _build_equity_group_map(equity_assets, equity_returns)
    regime_series = _load_structural_regime_series_local(ROOT)

    eq_a2 = _simulate_equity_group_sleeve_v2(
        candidate_id="equities_v2__slow189__g4__a1",
        returns=equity_returns,
        prices=equity_prices,
        asset_table=equity_assets,
        equity_groups=equity_group_map,
        benchmark_ticker=str(benchmark_equity),
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
        benchmark_ticker=str(benchmark_equity),
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
    eq_sp, _ = _simulate_equity_group_sleeve_v4_sector_pressure(
        candidate_id="equities_v4__sector_pressure_p25",
        returns=equity_returns,
        prices=equity_prices,
        asset_table=equity_assets,
        equity_groups=equity_group_map,
        benchmark_ticker=str(benchmark_equity),
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
    if eq_a2 is None or eq_r1 is None or eq_sp is None:
        raise SystemExit("falha ao reconstruir sleeves de equities na perturbacao")

    equity_base = _simulate_equity_trail_switch_bundle(
        candidate_id="equities_meta__trail_switch__a2__r1",
        aggressive_bundle=eq_a2,
        robust_bundle=eq_r1,
        regime_series=regime_series,
        spy_prices=pd.to_numeric(equity_prices[str(benchmark_equity)], errors="coerce"),
    )
    equity_attack = _simulate_equity_trail_switch_bundle(
        candidate_id="equities_meta__trail_switch__sector_p25",
        aggressive_bundle=eq_sp,
        robust_bundle=eq_r1,
        regime_series=regime_series,
        spy_prices=pd.to_numeric(equity_prices[str(benchmark_equity)], errors="coerce"),
    )

    allowed_crypto = list(crypto_tiers["crypto_all"]) if str(crypto_allowed_mode).strip().lower() == "all22" else list(crypto_tiers["crypto_major8"])
    if not allowed_crypto:
        raise SystemExit(f"universo cripto vazio para modo={crypto_allowed_mode}")

    baseline_crypto_result = _simulate_asset_rule(
        candidate_id="crypto_major8__mom_vol_adj_lb021_rb07_k3",
        family="crypto_major8_search",
        allowed_tickers=allowed_crypto,
        returns=crypto_returns,
        prices=crypto_prices,
        asset_table=crypto_assets,
        benchmark_ticker=str(benchmark_crypto),
        fallback_ticker=str(benchmark_crypto),
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
    attack_crypto_result = _simulate_asset_rule(
        candidate_id="crypto_major8__mom_total_lb021_rb07_k1",
        family="crypto_major8_search",
        allowed_tickers=allowed_crypto,
        returns=crypto_returns,
        prices=crypto_prices,
        asset_table=crypto_assets,
        benchmark_ticker=str(benchmark_crypto),
        fallback_ticker=str(benchmark_crypto),
        score_mode="mom_total",
        lookback_days=21,
        rebalance_days=7,
        top_k=1,
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
    if baseline_crypto_result is None or attack_crypto_result is None:
        raise SystemExit("falha ao reconstruir sleeves de cripto na perturbacao")

    baseline_crypto_bundle = StrategyBundle(
        result=baseline_crypto_result,
        benchmark_gross_ret=pd.to_numeric(crypto_returns[str(benchmark_crypto)], errors="coerce").reindex(baseline_crypto_result.gross_ret.index).fillna(0.0).astype(float),
        profile=crypto_profile,
        benchmark_profile=crypto_profile,
    )
    attack_crypto_bundle = StrategyBundle(
        result=attack_crypto_result,
        benchmark_gross_ret=pd.to_numeric(crypto_returns[str(benchmark_crypto)], errors="coerce").reindex(attack_crypto_result.gross_ret.index).fillna(0.0).astype(float),
        profile=crypto_profile,
        benchmark_profile=crypto_profile,
    )

    btc_prices = pd.to_numeric(crypto_prices[str(benchmark_crypto)], errors="coerce")
    spy_prices = pd.to_numeric(equity_prices[str(benchmark_equity)], errors="coerce")

    from scripts.bench.validation.run_profit_regime_simulation_suite import _build_meta_v1_allocation  # noqa: E402

    baseline = _build_meta_v1_allocation(
        crypto_bundle=baseline_crypto_bundle,
        equity_bundle=equity_base,
        btc_prices=btc_prices,
        spy_prices=spy_prices,
        profile=blended_profile,
    )
    baseline = replace(
        baseline,
        bundle=_candidate_bundle(candidate_id="meta_major8_eq_a2r1", bundle=baseline.bundle),
    )

    attack = _build_alpha_meta_allocation_bundle(
        candidate_id="alpha_attack_major8_equity25",
        crypto_bundle=attack_crypto_bundle,
        equity_bundle=equity_attack,
        btc_prices=btc_prices,
        spy_prices=spy_prices,
        profile=blended_profile,
        entry_lookback=21,
        exit_lookback=63,
        entry_margin=0.05,
        exit_margin=0.05,
        risk_off_mode="equity25",
        min_crypto_hold_days=0,
    )

    base_returns = pd.concat(
        {
            "crypto": pd.to_numeric(baseline_crypto_bundle.result.gross_ret, errors="coerce"),
            "equity": pd.to_numeric(equity_base.result.gross_ret, errors="coerce"),
            "benchmark": 0.5 * pd.to_numeric(crypto_returns[str(benchmark_crypto)], errors="coerce") + 0.5 * pd.to_numeric(equity_returns[str(benchmark_equity)], errors="coerce"),
        },
        axis=1,
        sort=False,
    ).dropna(how="all")
    attack_returns = pd.concat(
        {
            "crypto": pd.to_numeric(attack_crypto_bundle.result.gross_ret, errors="coerce"),
            "equity": pd.to_numeric(equity_attack.result.gross_ret, errors="coerce"),
            "benchmark": 0.5 * pd.to_numeric(crypto_returns[str(benchmark_crypto)], errors="coerce") + 0.5 * pd.to_numeric(equity_returns[str(benchmark_equity)], errors="coerce"),
        },
        axis=1,
        sort=False,
    ).dropna(how="all")
    baseline_guard, _ = _apply_mc_guard(
        candidate_id="meta_major8_eq_a2r1_mc_guard",
        base=baseline,
        returns=base_returns[["crypto", "equity"]],
        regime=regime_series,
        profile=blended_profile,
        lookback=252,
        horizon=21,
        n_paths=400,
        step=42,
    )
    attack_guard, _ = _apply_mc_guard(
        candidate_id="alpha_attack_major8_equity25_mc_guard",
        base=attack,
        returns=attack_returns[["crypto", "equity"]],
        regime=regime_series,
        profile=blended_profile,
        lookback=252,
        horizon=21,
        n_paths=400,
        step=42,
    )
    return {
        "baseline": baseline,
        "attack": attack,
        "baseline_guard": baseline_guard,
        "attack_guard": attack_guard,
        "returns": {
            "baseline": base_returns,
            "attack": attack_returns,
            "baseline_guard": base_returns,
            "attack_guard": attack_returns,
        },
        "crypto_major8": list(crypto_tiers["crypto_major8"]),
        "crypto_all22": list(crypto_tiers.get("crypto_all", [])),
        "equity_assets": equity_assets,
        "regime_series": regime_series,
    }


def _result_row(
    *,
    scenario: str,
    label: str,
    bundle: AllocationBundle,
    notes: str,
) -> dict[str, Any]:
    result = bundle.bundle.result
    return {
        "scenario": scenario,
        "candidate_id": str(result.candidate_id),
        "candidate_label": label,
        "net_ann_return": _safe_float(result.net_ann_return),
        "net_total_return": _safe_float(result.net_total_return),
        "net_sharpe": _safe_float(result.net_sharpe),
        "net_max_drawdown": _safe_float(result.net_max_drawdown),
        "edge_vs_benchmark": _safe_float(result.edge_vs_benchmark),
        "avg_turnover_daily": _safe_float(result.avg_turnover_daily),
        "notes": notes,
    }


def _large_mc_summary(
    *,
    returns_frame: pd.DataFrame,
    regime: pd.Series,
    weights: pd.Series,
    start_state: str,
    horizon: int,
    n_paths: int,
    random_state: int,
) -> dict[str, Any]:
    aligned = returns_frame.dropna(how="any").copy()
    if aligned.empty:
        return {}
    reg = regime.reindex(aligned.index).astype(str).str.lower().ffill().bfill()
    moments = estimate_regime_moments(aligned, reg, min_obs=20)
    states, transition = estimate_transition_matrix(reg, state_order=sorted(moments.keys()))
    if start_state not in moments:
        start_state = states[0]
    sim, state_paths = simulate_regime_conditioned_paths(
        regime_moments=moments,
        transition_matrix=transition,
        states=states,
        start_state=start_state,
        horizon=int(horizon),
        n_paths=int(n_paths),
        random_state=int(random_state),
    )
    port_stats = summarize_portfolio_distribution(sim, weights.to_numpy(dtype=float))
    benchmark_stats = summarize_portfolio_distribution(sim, np.array([0.0, 0.0, 1.0], dtype=float))
    port_daily = np.einsum("tha,a->th", sim, weights.to_numpy(dtype=float), optimize=True)
    bench_daily = sim[:, :, 2]
    port_terminal = np.prod(1.0 + port_daily, axis=1) - 1.0
    bench_terminal = np.prod(1.0 + bench_daily, axis=1) - 1.0
    diff = port_terminal - bench_terminal
    return {
        "start_state": start_state,
        "horizon_days": int(horizon),
        "n_paths": int(n_paths),
        "portfolio": port_stats,
        "benchmark": benchmark_stats,
        "underperform_prob": float(np.mean(diff < 0.0)),
        "beat_prob": float(np.mean(diff > 0.0)),
        "alpha_p05": float(np.nanquantile(diff, 0.05)),
        "alpha_p50": float(np.nanquantile(diff, 0.50)),
        "alpha_p95": float(np.nanquantile(diff, 0.95)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Perturbacao de universo, dependencia de poucos ativos e Monte Carlo amplo por regime.")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-groups-clean", default="data/asset_groups_target_800_clean.csv")
    ap.add_argument("--equity-asset-metadata-clean", default="data/asset_metadata_target_800_clean.csv")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--outdir-root", default="results/validation/profit_universe_resilience_suite")
    args = ap.parse_args()

    outdir = (ROOT / args.outdir_root / _run_id()).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    prices_dir = (ROOT / args.prices_dir).resolve()
    built = _build_custom_candidates(
        prices_dir=prices_dir,
        crypto_groups=(ROOT / args.crypto_asset_groups).resolve(),
        crypto_meta=(ROOT / args.crypto_asset_metadata).resolve(),
        equity_groups=(ROOT / args.equity_asset_groups).resolve(),
        equity_meta=(ROOT / args.equity_asset_metadata).resolve(),
        benchmark_crypto=str(args.benchmark_crypto),
        benchmark_equity=str(args.benchmark_equity),
    )

    # Identify the names that most often carry the aggressive crypto sleeve.
    crypto_assets = _load_asset_table((ROOT / args.crypto_asset_groups).resolve(), (ROOT / args.crypto_asset_metadata).resolve())
    crypto_returns, crypto_prices, _ = _load_daily_universe(
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
    top_freq = _selection_frequency_for_crypto_rule(
        allowed_tickers=built["crypto_major8"],
        returns=crypto_returns,
        prices=crypto_prices,
        benchmark_ticker=str(args.benchmark_crypto),
        lookback_days=21,
        rebalance_days=7,
        top_k=1,
        score_mode="mom_total",
        asset_ma_days=0,
        market_ma_days=200,
    )
    top_freq.to_csv(outdir / "top_crypto_frequency.csv", index=False)
    top1 = top_freq["ticker"].head(1).astype(str).tolist()
    top2 = top_freq["ticker"].head(2).astype(str).tolist()
    top3 = top_freq["ticker"].head(3).astype(str).tolist()

    scenarios: list[dict[str, Any]] = [
        {"name": "base", "notes": "configuracao atual sem perturbacao"},
        {"name": "crypto_all22", "notes": "troca major8 por universo cripto mais amplo", "crypto_mode": "all22"},
        {"name": "drop_top1_crypto", "notes": f"remove o nome mais frequente: {','.join(top1)}", "drop_crypto": top1},
        {"name": "drop_top2_crypto", "notes": f"remove os dois nomes mais frequentes: {','.join(top2)}", "drop_crypto": top2},
        {"name": "drop_top3_crypto", "notes": f"remove os tres nomes mais frequentes: {','.join(top3)}", "drop_crypto": top3},
        {"name": "equity_clean_pack", "notes": "usa target_800_clean em vez de clean_plus", "equity_pack": "clean"},
        {"name": "equity_random80", "notes": "mantem 80% aleatorio do universo de acoes", "equity_random_keep": 0.80, "seed": 17},
        {"name": "equity_random65", "notes": "mantem 65% aleatorio do universo de acoes", "equity_random_keep": 0.65, "seed": 23},
        {"name": "drop_sector_technology", "notes": "remove setor technology", "drop_sectors": ["technology"]},
        {"name": "drop_sector_materials", "notes": "remove setor materials", "drop_sectors": ["materials"]},
        {"name": "drop_sector_financials", "notes": "remove setor financials", "drop_sectors": ["financials"]},
        {"name": "drop_sector_consumer_discretionary", "notes": "remove setor consumer_discretionary", "drop_sectors": ["consumer_discretionary"]},
    ]

    perturb_rows: list[dict[str, Any]] = []
    mc_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        if str(scenario["name"]) == "base":
            perturbed = built
        else:
            equity_groups = (ROOT / args.equity_asset_groups).resolve()
            equity_meta = (ROOT / args.equity_asset_metadata).resolve()
            if str(scenario.get("equity_pack", "")) == "clean":
                equity_groups = (ROOT / args.equity_asset_groups_clean).resolve()
                equity_meta = (ROOT / args.equity_asset_metadata_clean).resolve()
            perturbed = _build_custom_candidates(
                prices_dir=prices_dir,
                crypto_groups=(ROOT / args.crypto_asset_groups).resolve(),
                crypto_meta=(ROOT / args.crypto_asset_metadata).resolve(),
                equity_groups=equity_groups,
                equity_meta=equity_meta,
                benchmark_crypto=str(args.benchmark_crypto),
                benchmark_equity=str(args.benchmark_equity),
                crypto_drop_tickers=scenario.get("drop_crypto"),
                equity_drop_sectors=scenario.get("drop_sectors"),
                random_keep_fraction=scenario.get("equity_random_keep"),
                random_seed=int(scenario.get("seed", 7)),
                crypto_allowed_mode="all22" if str(scenario.get("crypto_mode", "")) == "all22" else "major8",
            )

        for key in ["baseline", "attack"]:
            perturb_rows.append(
                _result_row(
                    scenario=str(scenario["name"]),
                    label=_human_label(perturbed[key].bundle.result.candidate_id),
                    bundle=perturbed[key],
                    notes=str(scenario["notes"]),
                )
            )
        for key in ["baseline", "attack", "baseline_guard", "attack_guard"]:
            bundle = perturbed[key]
            returns_frame = perturbed["returns"][key]
            weights = bundle.weights.reindex(returns_frame.index).ffill().fillna(0.0).iloc[-1][["crypto", "equity", "cash"]]
            sim_weights = pd.Series(
                {
                    "crypto": float(weights.get("crypto", 0.0)),
                    "equity": float(weights.get("equity", 0.0)),
                    "benchmark": 0.0,
                }
            )
            state = str(perturbed["regime_series"].reindex(returns_frame.index).ffill().bfill().iloc[-1]).lower()
            for horizon, paths in [(21, 4000), (63, 3000), (126, 2500)]:
                mc = _large_mc_summary(
                    returns_frame=returns_frame[["crypto", "equity", "benchmark"]].dropna(how="any"),
                    regime=perturbed["regime_series"],
                    weights=sim_weights,
                    start_state=state,
                    horizon=int(horizon),
                    n_paths=int(paths),
                    random_state=7 + int(horizon),
                )
                mc_rows.append(
                    {
                        "scenario": str(scenario["name"]),
                        "candidate_id": str(bundle.bundle.result.candidate_id),
                        "candidate_label": _human_label(bundle.bundle.result.candidate_id),
                        "horizon_days": int(horizon),
                        "n_paths": int(paths),
                        "start_state": str(mc.get("start_state", "")),
                        "beat_prob": _safe_float(mc.get("beat_prob")),
                        "underperform_prob": _safe_float(mc.get("underperform_prob")),
                        "alpha_p05": _safe_float(mc.get("alpha_p05")),
                        "alpha_p50": _safe_float(mc.get("alpha_p50")),
                        "alpha_p95": _safe_float(mc.get("alpha_p95")),
                        "ruin_prob_m10": _safe_float(((mc.get("portfolio") or {}) if isinstance(mc.get("portfolio"), dict) else {}).get("ruin_prob_m10")),
                        "ruin_prob_m20": _safe_float(((mc.get("portfolio") or {}) if isinstance(mc.get("portfolio"), dict) else {}).get("ruin_prob_m20")),
                        "terminal_p05": _safe_float(((mc.get("portfolio") or {}) if isinstance(mc.get("portfolio"), dict) else {}).get("terminal_p05")),
                        "terminal_p50": _safe_float(((mc.get("portfolio") or {}) if isinstance(mc.get("portfolio"), dict) else {}).get("terminal_p50")),
                        "terminal_p95": _safe_float(((mc.get("portfolio") or {}) if isinstance(mc.get("portfolio"), dict) else {}).get("terminal_p95")),
                    }
                )

    perturb_df = pd.DataFrame(perturb_rows).sort_values(
        ["candidate_id", "net_total_return", "net_ann_return"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    mc_df = pd.DataFrame(mc_rows).sort_values(
        ["candidate_id", "scenario", "horizon_days"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    perturb_df.to_csv(outdir / "perturbation_compare.csv", index=False)
    mc_df.to_csv(outdir / "monte_carlo_compare.csv", index=False)

    attack_base = perturb_df[(perturb_df["scenario"] == "base") & (perturb_df["candidate_id"] == "alpha_attack_major8_equity25")].head(1)
    attack_perturbed = perturb_df[perturb_df["candidate_id"] == "alpha_attack_major8_equity25"].copy()
    base_total = float(attack_base.iloc[0]["net_total_return"]) if not attack_base.empty else float("nan")
    base_ann = float(attack_base.iloc[0]["net_ann_return"]) if not attack_base.empty else float("nan")
    if np.isfinite(base_total):
        attack_perturbed["total_retention"] = pd.to_numeric(attack_perturbed["net_total_return"], errors="coerce") / float(base_total)
    else:
        attack_perturbed["total_retention"] = float("nan")
    if np.isfinite(base_ann):
        attack_perturbed["ann_retention"] = pd.to_numeric(attack_perturbed["net_ann_return"], errors="coerce") / float(base_ann)
    else:
        attack_perturbed["ann_retention"] = float("nan")
    attack_perturbed.to_csv(outdir / "attack_retention.csv", index=False)

    attack_mc = mc_df[(mc_df["scenario"] == "base") & (mc_df["candidate_id"] == "alpha_attack_major8_equity25")].copy()
    guard_mc = mc_df[(mc_df["scenario"] == "base") & (mc_df["candidate_id"] == "alpha_attack_major8_equity25_mc_guard")].copy()

    research_rows = []
    for candidate_id in ["meta_major8_eq_a2r1", "alpha_attack_major8_equity25", "meta_major8_eq_a2r1_mc_guard", "alpha_attack_major8_equity25_mc_guard"]:
        subset = perturb_df[(perturb_df["scenario"] == "base") & (perturb_df["candidate_id"] == candidate_id)].copy()
        if subset.empty:
            continue
        row = subset.iloc[0]
        bundle = built["baseline"] if candidate_id == "meta_major8_eq_a2r1" else built["attack"] if candidate_id == "alpha_attack_major8_equity25" else built["baseline_guard"] if candidate_id == "meta_major8_eq_a2r1_mc_guard" else built["attack_guard"]
        result = replace(
            bundle.bundle.result,
            net_ann_return=float(row["net_ann_return"]),
            net_total_return=float(row["net_total_return"]),
            net_sharpe=float(row["net_sharpe"]),
            net_max_drawdown=float(row["net_max_drawdown"]),
            edge_vs_benchmark=float(row["edge_vs_benchmark"]),
        )
        status = "keep" if candidate_id in {"meta_major8_eq_a2r1", "alpha_attack_major8_equity25"} else "watch"
        research_rows.append(
            _research_row(
                result,
                outdir=outdir,
                status=status,
                methodology="universe_resilience_and_regime_mc",
                label=f"{_human_label(candidate_id)} com teste de perturbacao",
            )
        )
    (outdir / "profit_research_rows.json").write_text(json.dumps(research_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "top_crypto_frequency": top_freq.head(10).to_dict(orient="records"),
        "base_attack": attack_base.iloc[0].to_dict() if not attack_base.empty else {},
        "attack_retention_best_nonbase": attack_perturbed[attack_perturbed["scenario"] != "base"].sort_values(["net_total_return", "net_ann_return"], ascending=[False, False]).head(1).to_dict(orient="records"),
        "attack_retention_worst_nonbase": attack_perturbed[attack_perturbed["scenario"] != "base"].sort_values(["net_total_return", "net_ann_return"], ascending=[True, True]).head(1).to_dict(orient="records"),
        "attack_mc_base": attack_mc.to_dict(orient="records"),
        "attack_guard_mc_base": guard_mc.to_dict(orient="records"),
        "insights": [
            "A perturbacao de universo mede se o lucro sobrevive quando o conjunto de ativos e setores muda.",
            "A remocao dos nomes mais recorrentes testa se o alpha depende de poucos foguetes ou se o motor continua de pe sem eles.",
            "O Monte Carlo por regime amplia a leitura de confianca com faixa provavel de retorno, risco de ruina e chance de ficar abaixo do benchmark.",
        ],
        "artifacts": {
            "perturbation_compare_csv": str(outdir / "perturbation_compare.csv"),
            "attack_retention_csv": str(outdir / "attack_retention.csv"),
            "top_crypto_frequency_csv": str(outdir / "top_crypto_frequency.csv"),
            "monte_carlo_compare_csv": str(outdir / "monte_carlo_compare.csv"),
            "profit_research_rows_json": str(outdir / "profit_research_rows.json"),
        },
    }
    _write_json(outdir / "summary.json", summary)
    write_run_manifest(
        outdir,
        script="scripts/bench/validation/run_profit_universe_resilience_suite.py",
        params={
            "benchmark_crypto": args.benchmark_crypto,
            "benchmark_equity": args.benchmark_equity,
            "n_perturb_scenarios": len(scenarios),
        },
        paths=summary["artifacts"],
        extra={"summary_json": str(outdir / "summary.json")},
    )


if __name__ == "__main__":
    main()
