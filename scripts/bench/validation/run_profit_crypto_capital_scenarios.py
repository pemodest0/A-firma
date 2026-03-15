#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from execution.cost_model import summarize_return_series  # noqa: E402
from execution.net_assumptions import apply_net_assumptions, load_net_assumption_profiles  # noqa: E402
from scripts.bench.validation.run_profit_alpha_improvement_suite import _write_json  # noqa: E402
from scripts.bench.validation.run_profit_one_year_payoff_audit import _forward_path_frame  # noqa: E402
from scripts.bench.validation.run_profit_10x_rule_search import (  # noqa: E402
    RuleConfig,
    _ensure_benchmark_columns,
    _load_asset_table,
    _load_daily_universe,
    _precompute_scores,
    _safe_float,
    _top_k_indices,
)


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    family: str
    assets: tuple[str, ...]
    weights: tuple[float, ...] = ()
    rebalance_days: int = 0
    dynamic_candidate_id: str = ""


DEFAULT_SCENARIOS = [
    Scenario("btc_hold", "single_asset", ("BTC-USD",), (1.0,), 0, ""),
    Scenario("eth_hold", "single_asset", ("ETH-USD",), (1.0,), 0, ""),
    Scenario("sol_hold", "single_asset", ("SOL-USD",), (1.0,), 0, ""),
    Scenario("xrp_hold", "single_asset", ("XRP-USD",), (1.0,), 0, ""),
    Scenario("btc_eth_70_30_hold", "fixed_basket", ("BTC-USD", "ETH-USD"), (0.7, 0.3), 0, ""),
    Scenario("major3_eq_hold", "fixed_basket", ("BTC-USD", "ETH-USD", "SOL-USD"), (1 / 3, 1 / 3, 1 / 3), 0, ""),
    Scenario("major5_eq_monthly", "fixed_basket", ("BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "BNB-USD"), (0.2, 0.2, 0.2, 0.2, 0.2), 21, ""),
    Scenario("dyn_k2_fast", "dynamic_motor", (), (), 0, "all_assets__lb021__rb07__k2__mom_total__ama000__mma000__riskon"),
    Scenario("dyn_k3_rel", "dynamic_motor", (), (), 0, "all_assets__lb021__rb07__k3__mom_vol_adj__ama000__mma000__relshy"),
    Scenario("dyn_k3_total", "dynamic_motor", (), (), 0, "all_assets__lb021__rb07__k3__mom_total__ama000__mma000__riskon"),
    Scenario("dyn_k2_slow", "dynamic_motor", (), (), 0, "all_assets__lb126__rb07__k2__mom_total__ama200__mma200__riskon"),
]


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _weights_dict(assets: tuple[str, ...], weights: tuple[float, ...]) -> dict[str, float]:
    total = float(sum(weights))
    if total <= 0.0:
        raise ValueError("weights must sum positive")
    return {str(asset): float(weight) / total for asset, weight in zip(assets, weights)}


def _simulate_fixed_basket(
    *,
    scenario: Scenario,
    returns: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    idx = returns.index
    weights = _weights_dict(scenario.assets, scenario.weights)
    target = pd.Series(weights, dtype=float)
    target = target[target.index.isin(returns.columns)]
    if target.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    current = pd.Series(0.0, index=target.index, dtype=float)
    gross = pd.Series(0.0, index=idx, dtype=float)
    turnover = pd.Series(0.0, index=idx, dtype=float)

    first = True
    rebalance_days = int(max(0, scenario.rebalance_days))
    last_rebalance_pos = 0
    for pos, dt in enumerate(idx):
        if first:
            current = target.copy()
            turnover.loc[dt] = 1.0
            first = False
            last_rebalance_pos = pos
        elif rebalance_days > 0 and (pos - last_rebalance_pos) >= rebalance_days:
            turn = 0.5 * float((current.reindex(target.index).fillna(0.0) - target).abs().sum())
            turnover.loc[dt] = turnover.loc[dt] + turn
            current = target.copy()
            last_rebalance_pos = pos

        day_ret = pd.to_numeric(returns.loc[dt, current.index], errors="coerce").fillna(0.0).astype(float)
        gross.loc[dt] = float((current * day_ret).sum())
        grown = current * (1.0 + day_ret)
        total = float(grown.sum())
        if total > 0.0:
            current = (grown / total).astype(float)

    # liquidation cost at the campaign end
    turnover.iloc[-1] = turnover.iloc[-1] + 1.0
    return gross.astype(float), turnover.astype(float)


def _parse_rule_candidate_id(candidate_id: str, *, groups: tuple[str, ...]) -> RuleConfig:
    parts = str(candidate_id).split("__")
    if len(parts) != 8:
        raise ValueError(f"invalid candidate id: {candidate_id}")
    return RuleConfig(
        family_id=str(parts[0]),
        groups=tuple(groups),
        lookback_days=int(parts[1].replace("lb", "")),
        rebalance_days=int(parts[2].replace("rb", "")),
        top_k=int(parts[3].replace("k", "")),
        score_mode=str(parts[4]),
        asset_ma_days=int(parts[5].replace("ama", "")),
        market_ma_days=int(parts[6].replace("mma", "")),
        relative_to_shy=(parts[7] == "relshy"),
    )


def _simulate_dynamic_crypto(
    *,
    scenario: Scenario,
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    asset_table: pd.DataFrame,
    score_map: dict[tuple[int, str], pd.DataFrame],
    asset_ma_filters: dict[int, pd.DataFrame],
    benchmark_filters: dict[int, pd.Series],
    benchmark_ticker: str,
    fallback_ticker: str,
) -> tuple[pd.Series, pd.Series]:
    cfg = _parse_rule_candidate_id(scenario.dynamic_candidate_id, groups=tuple(sorted(asset_table["asset_group"].astype(str).unique().tolist())))
    all_tickers = list(returns.columns.astype(str))
    ticker_to_col = {ticker: idx for idx, ticker in enumerate(all_tickers)}
    allowed_assets = asset_table[asset_table["asset_group"].astype(str).isin(list(cfg.groups))]["ticker"].astype(str).tolist()
    allowed_idx = np.array([ticker_to_col[ticker] for ticker in allowed_assets if ticker in ticker_to_col], dtype=int)
    if allowed_idx.size == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    score_df = score_map[(int(cfg.lookback_days), str(cfg.score_mode))]
    score_arr = score_df.reindex(index=returns.index, columns=all_tickers).to_numpy(dtype=float)
    asset_ma_arr = asset_ma_filters[int(cfg.asset_ma_days)].reindex(index=returns.index, columns=all_tickers).fillna(False).to_numpy(dtype=bool)
    benchmark_ok = benchmark_filters[int(cfg.market_ma_days)].reindex(returns.index).fillna(False).to_numpy(dtype=bool)
    ret_arr = returns.reindex(columns=all_tickers).to_numpy(dtype=float)
    shy_score = None
    if bool(cfg.relative_to_shy) and fallback_ticker in score_df.columns:
        shy_score = pd.to_numeric(score_df[fallback_ticker], errors="coerce").to_numpy(dtype=float)

    warmup = max(int(cfg.lookback_days), int(cfg.asset_ma_days), int(cfg.market_ma_days)) + 2
    rebalance_positions = list(range(int(max(1, warmup)), ret_arr.shape[0], int(max(1, cfg.rebalance_days))))
    if not rebalance_positions:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    gross = pd.Series(0.0, index=returns.index, dtype=float)
    turnover = pd.Series(0.0, index=returns.index, dtype=float)
    prev_weights: dict[str, float] = {"CASH": 1.0}

    for pos_idx, pos in enumerate(rebalance_positions):
        next_pos = rebalance_positions[pos_idx + 1] if pos_idx + 1 < len(rebalance_positions) else ret_arr.shape[0]
        score_row = score_arr[pos]
        valid = np.zeros(score_row.shape[0], dtype=bool)
        valid[allowed_idx] = True
        valid &= np.isfinite(score_row)
        if pos > 0:
            valid &= np.isfinite(ret_arr[pos - 1])
        valid &= asset_ma_arr[pos]
        valid &= score_row > 0.0
        if shy_score is not None and pos < shy_score.shape[0] and np.isfinite(shy_score[pos]):
            valid &= score_row > float(shy_score[pos])
        if int(cfg.market_ma_days) > 0 and not bool(benchmark_ok[pos]):
            valid[:] = False

        selected_idx = _top_k_indices(score_row, valid, int(cfg.top_k))
        if not selected_idx and fallback_ticker in ticker_to_col:
            selected_idx = [ticker_to_col[fallback_ticker]]

        if not selected_idx:
            new_weights = {"CASH": 1.0}
            turnover.iloc[pos] = 0.5 * float(sum(abs(float(prev_weights.get(k, 0.0)) - float(new_weights.get(k, 0.0))) for k in sorted(set(prev_weights) | set(new_weights))))
            prev_weights = new_weights
            continue

        w = 1.0 / float(len(selected_idx))
        new_weights = {all_tickers[i]: w for i in selected_idx}
        turnover.iloc[pos] = 0.5 * float(sum(abs(float(prev_weights.get(k, 0.0)) - float(new_weights.get(k, 0.0))) for k in sorted(set(prev_weights) | set(new_weights))))
        prev_weights = dict(new_weights)
        block = np.nan_to_num(ret_arr[pos:next_pos, selected_idx], nan=0.0)
        gross.iloc[pos:next_pos] = block.mean(axis=1).astype(float)

    turnover.iloc[-1] = turnover.iloc[-1] + 1.0
    return gross.astype(float), turnover.astype(float)


def _approx_nav_before(net_ret: pd.Series, capital_brl: float) -> pd.Series:
    equity = (1.0 + pd.to_numeric(net_ret, errors="coerce").fillna(0.0).astype(float)).cumprod()
    return (equity.shift(1).fillna(1.0) * float(capital_brl)).astype(float)


def _months_over_exemption(turnover: pd.Series, net_ret: pd.Series, *, capital_brl: float, sell_fraction_proxy: float, monthly_limit_brl: float) -> tuple[int, float]:
    nav_prev = _approx_nav_before(net_ret, capital_brl)
    sales_brl = (nav_prev * pd.to_numeric(turnover, errors="coerce").reindex(nav_prev.index).fillna(0.0).clip(lower=0.0) * float(sell_fraction_proxy)).astype(float)
    ym = pd.to_datetime(sales_brl.index).strftime("%Y-%m")
    by_month = sales_brl.groupby(ym).sum()
    over = by_month[by_month > float(monthly_limit_brl)]
    return int(over.shape[0]), float(by_month.max()) if not by_month.empty else 0.0


def _scenario_row(
    *,
    scenario: Scenario,
    capital_brl: float,
    gross: pd.Series,
    turnover: pd.Series,
    benchmark_ret: pd.Series,
    profile: Any,
) -> dict[str, Any]:
    net_frame = apply_net_assumptions(
        gross,
        turnover,
        profile=profile,
        periods_index=gross.index,
        initial_capital_brl=float(capital_brl),
    )
    benchmark_frame = apply_net_assumptions(
        benchmark_ret,
        pd.Series(np.zeros(len(benchmark_ret), dtype=float), index=benchmark_ret.index, dtype=float),
        profile=profile,
        periods_index=benchmark_ret.index,
        initial_capital_brl=float(capital_brl),
    )
    net_ret = pd.to_numeric(net_frame["net_ret"], errors="coerce").fillna(0.0).astype(float)
    summary = summarize_return_series(net_ret, periods_per_year=252)
    bench_summary = summarize_return_series(pd.to_numeric(benchmark_frame["net_ret"], errors="coerce").fillna(0.0).astype(float), periods_per_year=252)
    nav_prev = _approx_nav_before(net_ret, capital_brl)
    transaction_cost_brl = float((pd.to_numeric(net_frame["transaction_cost_ret"], errors="coerce").fillna(0.0) * nav_prev).sum())
    tax_brl = float((pd.to_numeric(net_frame["tax_ret"], errors="coerce").fillna(0.0) * nav_prev).sum())
    withholding_brl = float((pd.to_numeric(net_frame["withholding_ret"], errors="coerce").fillna(0.0) * nav_prev).sum())
    months_over_exemption, max_monthly_sales_brl = _months_over_exemption(
        turnover,
        net_ret,
        capital_brl=float(capital_brl),
        sell_fraction_proxy=float(profile.sell_turnover_fraction_proxy),
        monthly_limit_brl=float(profile.monthly_sales_exemption_brl),
    )
    payoff = _forward_path_frame(net_ret, horizon_days=252, monthly_start=False)
    terminal_multiple = pd.to_numeric(payoff["terminal_multiple"], errors="coerce").astype(float) if not payoff.empty else pd.Series(dtype=float)
    max_multiple = pd.to_numeric(payoff["max_multiple"], errors="coerce").astype(float) if not payoff.empty else pd.Series(dtype=float)
    min_multiple = pd.to_numeric(payoff["min_multiple"], errors="coerce").astype(float) if not payoff.empty else pd.Series(dtype=float)

    row = {
        "scenario_id": str(scenario.scenario_id),
        "family": str(scenario.family),
        "capital_brl": float(capital_brl),
        "assets": ",".join(scenario.assets) if scenario.assets else "",
        "weights": ",".join(f"{w:.4f}" for w in scenario.weights) if scenario.weights else "",
        "rebalance_days": int(scenario.rebalance_days),
        "dynamic_candidate_id": str(scenario.dynamic_candidate_id),
        "net_ann_return": _safe_float(summary.get("annualized_return")),
        "net_total_return": _safe_float(summary.get("total_return")),
        "net_sharpe": _safe_float(summary.get("sharpe")),
        "net_max_drawdown": _safe_float(summary.get("max_drawdown")),
        "edge_vs_benchmark": _safe_float(summary.get("total_return")) - _safe_float(bench_summary.get("total_return")),
        "avg_turnover_daily": float(pd.to_numeric(turnover, errors="coerce").fillna(0.0).mean()),
        "final_value_brl": float(capital_brl * (1.0 + _safe_float(summary.get("total_return"), -1.0))),
        "transaction_cost_brl": transaction_cost_brl,
        "tax_brl": tax_brl,
        "withholding_brl": withholding_brl,
        "months_over_35k_sales": int(months_over_exemption),
        "max_monthly_sales_brl": max_monthly_sales_brl,
        "median_return_252d": float((terminal_multiple - 1.0).quantile(0.50)) if not terminal_multiple.empty else float("nan"),
        "p10_return_252d": float((terminal_multiple - 1.0).quantile(0.10)) if not terminal_multiple.empty else float("nan"),
        "p90_return_252d": float((terminal_multiple - 1.0).quantile(0.90)) if not terminal_multiple.empty else float("nan"),
        "hit_rate_2x_252d": float((max_multiple >= 2.0).mean()) if not max_multiple.empty else float("nan"),
        "hit_rate_3x_252d": float((max_multiple >= 3.0).mean()) if not max_multiple.empty else float("nan"),
        "touch_loss_50_252d": float((min_multiple <= 0.5).mean()) if not min_multiple.empty else float("nan"),
        "median_end_value_252d_brl": float(capital_brl * terminal_multiple.quantile(0.50)) if not terminal_multiple.empty else float("nan"),
        "p10_end_value_252d_brl": float(capital_brl * terminal_multiple.quantile(0.10)) if not terminal_multiple.empty else float("nan"),
        "p90_end_value_252d_brl": float(capital_brl * terminal_multiple.quantile(0.90)) if not terminal_multiple.empty else float("nan"),
    }
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Cenarios cripto-only por ativo, cesta fixa e sleeve dinamico, com friccao e imposto proxy.")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--net-assumptions-config", default="config/profit_net_assumptions.json")
    ap.add_argument("--benchmark-ticker", default="BTC-USD")
    ap.add_argument("--fallback-ticker", default="BTC-USD")
    ap.add_argument("--capitals", default="100,200,400,1000,40000")
    ap.add_argument("--outdir-root", default="results/validation/profit_crypto_capital_scenarios")
    args = ap.parse_args()

    outdir = (ROOT / args.outdir_root / _run_id()).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    prices_dir = (ROOT / args.prices_dir).resolve()
    asset_groups = (ROOT / args.asset_groups).resolve()
    asset_metadata = (ROOT / args.asset_metadata).resolve()
    capitals = [float(token.strip()) for token in str(args.capitals).split(",") if token.strip()]

    asset_table = _load_asset_table(asset_groups, asset_metadata)
    returns, prices, viability = _load_daily_universe(
        prices_dir=prices_dir,
        asset_table=asset_table,
        min_history_days=252,
        max_abs_daily_return=2.0,
    )
    returns, prices = _ensure_benchmark_columns(returns, prices, prices_dir, [str(args.benchmark_ticker)])
    returns = returns[returns.index >= pd.Timestamp("2016-02-18")].copy()
    prices = prices.reindex(returns.index).copy()
    if returns.empty:
        raise SystemExit("no crypto returns after date filter")

    lookbacks = sorted(
        {
            int(s.dynamic_candidate_id.split("__")[1].replace("lb", ""))
            for s in DEFAULT_SCENARIOS
            if s.dynamic_candidate_id
        }
    )
    ma_days = sorted(
        {
            int(s.dynamic_candidate_id.split("__")[5].replace("ama", ""))
            for s in DEFAULT_SCENARIOS
            if s.dynamic_candidate_id
        }
        | {
            int(s.dynamic_candidate_id.split("__")[6].replace("mma", ""))
            for s in DEFAULT_SCENARIOS
            if s.dynamic_candidate_id
        }
        | {0}
    )
    score_map, asset_ma_filters, benchmark_filters = _precompute_scores(
        returns,
        prices,
        lookbacks=lookbacks,
        asset_ma_days_list=ma_days,
        benchmark_ticker=str(args.benchmark_ticker),
    )
    profiles = load_net_assumption_profiles((ROOT / args.net_assumptions_config).resolve())
    profile = profiles["profiles"]["crypto_global_brazil_resident_conservative"]
    benchmark_ret = pd.to_numeric(returns[str(args.benchmark_ticker)], errors="coerce").reindex(returns.index).fillna(0.0).astype(float)

    base_series: dict[str, tuple[Scenario, pd.Series, pd.Series]] = {}
    for scenario in DEFAULT_SCENARIOS:
        if scenario.family == "dynamic_motor":
            gross, turnover = _simulate_dynamic_crypto(
                scenario=scenario,
                returns=returns,
                prices=prices,
                asset_table=asset_table,
                score_map=score_map,
                asset_ma_filters=asset_ma_filters,
                benchmark_filters=benchmark_filters,
                benchmark_ticker=str(args.benchmark_ticker),
                fallback_ticker=str(args.fallback_ticker),
            )
        else:
            gross, turnover = _simulate_fixed_basket(
                scenario=scenario,
                returns=returns,
            )
        if gross.empty:
            continue
        base_series[scenario.scenario_id] = (scenario, gross, turnover)

    rows: list[dict[str, Any]] = []
    for capital in capitals:
        for scenario_id, (scenario, gross, turnover) in base_series.items():
            rows.append(
                _scenario_row(
                    scenario=scenario,
                    capital_brl=float(capital),
                    gross=gross,
                    turnover=turnover,
                    benchmark_ret=benchmark_ret,
                    profile=profile,
                )
            )

    compare_df = pd.DataFrame(rows).sort_values(
        ["capital_brl", "median_return_252d", "net_total_return"],
        ascending=[True, False, False],
        na_position="last",
    ).reset_index(drop=True)
    compare_df.to_csv(outdir / "scenario_compare.csv", index=False)

    summary_rows = []
    for capital, sub in compare_df.groupby("capital_brl", sort=True):
        best_median = sub.sort_values(["median_return_252d", "net_total_return"], ascending=[False, False]).iloc[0].to_dict()
        safest = sub.sort_values(["touch_loss_50_252d", "median_return_252d"], ascending=[True, False]).iloc[0].to_dict()
        cheapest = sub.sort_values(["transaction_cost_brl", "median_return_252d"], ascending=[True, False]).iloc[0].to_dict()
        summary_rows.append(
            {
                "capital_brl": float(capital),
                "best_median_scenario_id": best_median["scenario_id"],
                "best_median_return_252d": _safe_float(best_median.get("median_return_252d")),
                "safest_scenario_id": safest["scenario_id"],
                "safest_touch_loss_50_252d": _safe_float(safest.get("touch_loss_50_252d")),
                "cheapest_scenario_id": cheapest["scenario_id"],
                "cheapest_transaction_cost_brl": _safe_float(cheapest.get("transaction_cost_brl")),
            }
        )
    summary = {
        "suite": "profit_crypto_capital_scenarios",
        "capitals": capitals,
        "profile_id": "crypto_global_brazil_resident_conservative",
        "crypto_asset_count": int(viability.shape[0]),
        "notes": [
            "Para capitais pequenos, a isencao mensal de R$ 35 mil tende a zerar o imposto proxy em boa parte dos cenarios; a friccao de giro vira o drag principal.",
            "Setorialmente, o universo atual e um unico setor logico: crypto. A comparacao util aqui e por ativo, cesta fixa e sleeve dinamico.",
            "As metricas de 252 dias usam daily-start e resultado liquido com custo/imposto proxy do motor.",
        ],
        "by_capital": summary_rows,
    }
    _write_json(outdir / "summary.json", summary)

    write_run_manifest(
        outdir,
        script="scripts/bench/validation/run_profit_crypto_capital_scenarios.py",
        params={
            "prices_dir": str(prices_dir),
            "asset_groups": str(asset_groups),
            "asset_metadata": str(asset_metadata),
            "benchmark_ticker": str(args.benchmark_ticker),
            "fallback_ticker": str(args.fallback_ticker),
            "capitals": capitals,
        },
        extra={"suite": "profit_crypto_capital_scenarios"},
    )
    print(str(outdir))


if __name__ == "__main__":
    main()
