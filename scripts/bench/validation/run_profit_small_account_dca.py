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
from execution.net_assumptions import NetAssumptionProfile, load_net_assumption_profiles  # noqa: E402
from scripts.bench.validation.run_profit_alpha_improvement_suite import _write_json  # noqa: E402
from scripts.bench.validation.run_profit_crypto_capital_scenarios import (  # noqa: E402
    DEFAULT_SCENARIOS,
    Scenario,
    _ensure_benchmark_columns,
    _load_asset_table,
    _load_daily_universe,
    _precompute_scores,
    _simulate_dynamic_crypto,
    _simulate_fixed_basket,
)


DEFAULT_SCENARIO_IDS = (
    "btc_hold",
    "btc_eth_70_30_hold",
    "major5_eq_monthly",
    "dyn_k2_slow",
    "dyn_k2_fast",
    "dyn_k3_rel",
)


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _scenario_map() -> dict[str, Scenario]:
    return {str(s.scenario_id): s for s in DEFAULT_SCENARIOS}


def _parse_csv_numbers(text: str, *, cast=float) -> list[Any]:
    return [cast(token.strip()) for token in str(text).split(",") if token.strip()]


def _month_bounds(index: pd.DatetimeIndex) -> list[tuple[pd.Period, int, int]]:
    periods = index.to_period("M")
    out: list[tuple[pd.Period, int, int]] = []
    for period in pd.Index(periods.unique()).sort_values():
        positions = np.flatnonzero(periods == period)
        if positions.size <= 0:
            continue
        out.append((period, int(positions[0]), int(positions[-1])))
    return out


def _window_slices(index: pd.DatetimeIndex, *, horizon_months: int) -> list[dict[str, Any]]:
    bounds = _month_bounds(index)
    if horizon_months <= 0 or len(bounds) < horizon_months:
        return []
    windows: list[dict[str, Any]] = []
    for start_idx in range(0, len(bounds) - horizon_months + 1):
        start_period, start_pos, _ = bounds[start_idx]
        end_period, _, end_pos = bounds[start_idx + horizon_months - 1]
        contribution_positions = [bounds[i][1] for i in range(start_idx + 1, start_idx + horizon_months)]
        windows.append(
            {
                "start_period": str(start_period),
                "end_period": str(end_period),
                "start_pos": int(start_pos),
                "end_pos": int(end_pos),
                "contribution_positions": contribution_positions,
            }
        )
    return windows


def _resolve_progressive_rate(gain_brl: float, profile: NetAssumptionProfile) -> float:
    gain = float(max(0.0, gain_brl))
    if not profile.capital_gains_brackets:
        return float(max(0.0, profile.capital_gains_tax_rate))
    for up_to_brl, rate in profile.capital_gains_brackets:
        if gain <= float(up_to_brl):
            return float(max(0.0, rate))
    return float(max(0.0, profile.capital_gains_brackets[-1][1]))


@dataclass(frozen=True)
class DcaPathResult:
    final_value_brl: float
    total_contributed_brl: float
    profit_brl: float
    transaction_cost_brl: float
    tax_brl: float
    withholding_brl: float
    max_drawdown: float
    months_over_exemption: int
    max_monthly_sales_brl: float
    contribution_count: int


def _simulate_dca_path(
    *,
    gross_ret: pd.Series,
    turnover: pd.Series,
    profile: NetAssumptionProfile,
    initial_capital_brl: float,
    monthly_contribution_brl: float,
    contribution_positions: list[int],
) -> DcaPathResult:
    idx = pd.DatetimeIndex(gross_ret.index)
    gross = pd.to_numeric(gross_ret, errors="coerce").reindex(idx).fillna(0.0).astype(float).to_numpy(dtype=float)
    turn = (
        pd.to_numeric(turnover, errors="coerce")
        .reindex(idx)
        .fillna(0.0)
        .clip(lower=0.0)
        .astype(float)
        .to_numpy(dtype=float)
    )
    month_labels = idx.strftime("%Y-%m").to_numpy(dtype=object)
    cost_rate = float(max(0.0, profile.total_cost_bps_assumed)) / 10000.0
    withholding_rate = float(max(0.0, profile.withholding_bps_on_sales)) / 10000.0
    sell_fraction = float(np.clip(float(profile.sell_turnover_fraction_proxy), 0.0, 1.0))

    market_value_brl = float(max(0.0, initial_capital_brl))
    book_basis_brl = float(max(0.0, initial_capital_brl))
    total_contributed_brl = float(max(0.0, initial_capital_brl))
    contribution_count = 1 if initial_capital_brl > 0.0 else 0

    transaction_cost_brl = 0.0
    tax_brl = 0.0
    withholding_brl = 0.0
    carry_loss_brl = 0.0
    month_sales_brl = 0.0
    month_realized_gain_brl = 0.0
    month_withholding_brl = 0.0
    month_last_stamp: pd.Timestamp | None = None
    current_label: str | None = None
    months_over_exemption = 0
    max_monthly_sales_brl = 0.0

    equity_path: list[float] = []
    contribution_set = set(int(x) for x in contribution_positions)
    monthly_limit = float(max(0.0, profile.monthly_sales_exemption_brl))

    def finalize_month() -> None:
        nonlocal carry_loss_brl, month_sales_brl, month_realized_gain_brl, month_withholding_brl
        nonlocal tax_brl, months_over_exemption, max_monthly_sales_brl, market_value_brl
        max_monthly_sales_brl = max(max_monthly_sales_brl, float(month_sales_brl))
        if monthly_limit > 0.0 and month_sales_brl > monthly_limit:
            months_over_exemption += 1

        taxable_brl = 0.0
        if not (monthly_limit > 0.0 and month_sales_brl <= monthly_limit):
            if profile.loss_compensation_enabled:
                effective_brl = month_realized_gain_brl + carry_loss_brl
                taxable_brl = float(max(0.0, effective_brl))
                carry_loss_brl = float(min(0.0, effective_brl))
            else:
                taxable_brl = float(max(0.0, month_realized_gain_brl))
        elif profile.loss_compensation_enabled and month_realized_gain_brl < 0.0:
            carry_loss_brl += month_realized_gain_brl

        if taxable_brl > 0.0:
            rate = _resolve_progressive_rate(taxable_brl, profile)
            month_tax_brl = taxable_brl * rate
            if profile.withholding_compensates_tax:
                month_tax_brl = max(0.0, month_tax_brl - month_withholding_brl)
            tax_brl += month_tax_brl
            market_value_brl = float(max(0.0, market_value_brl - month_tax_brl))

        month_sales_brl = 0.0
        month_realized_gain_brl = 0.0
        month_withholding_brl = 0.0

    for pos, stamp in enumerate(idx):
        label = str(month_labels[pos])
        if current_label is None:
            current_label = label
        elif label != current_label and month_last_stamp is not None:
            finalize_month()
            current_label = label

        if pos in contribution_set and monthly_contribution_brl > 0.0:
            contribution_cost_brl = float(monthly_contribution_brl) * cost_rate
            market_value_brl += float(monthly_contribution_brl) - contribution_cost_brl
            book_basis_brl += float(monthly_contribution_brl)
            total_contributed_brl += float(monthly_contribution_brl)
            contribution_count += 1
            transaction_cost_brl += contribution_cost_brl

        nav_before = float(max(0.0, market_value_brl))
        daily_turnover = float(turn[pos])
        txn_cost_today = nav_before * daily_turnover * cost_rate
        withholding_proxy_today = nav_before * daily_turnover * sell_fraction * withholding_rate
        period_ret = float(gross[pos]) - (daily_turnover * cost_rate) - (daily_turnover * sell_fraction * withholding_rate)
        market_value_brl = float(max(0.0, nav_before * (1.0 + period_ret)))
        transaction_cost_brl += txn_cost_today
        withholding_brl += withholding_proxy_today

        sale_notional_brl = float(
            min(
                market_value_brl,
                max(0.0, market_value_brl * daily_turnover * sell_fraction),
            )
        )
        month_sales_brl += sale_notional_brl
        month_withholding_brl += float(max(0.0, sale_notional_brl * withholding_rate))

        if sale_notional_brl > 0.0 and market_value_brl > 0.0 and book_basis_brl > 0.0:
            sale_fraction_of_book = float(np.clip(sale_notional_brl / market_value_brl, 0.0, 1.0))
            cost_basis_sold_brl = float(book_basis_brl * sale_fraction_of_book)
            realized_gain_brl = float(sale_notional_brl - cost_basis_sold_brl)
            month_realized_gain_brl += realized_gain_brl
            remaining_book_brl = float(max(0.0, book_basis_brl - cost_basis_sold_brl))
            buy_notional_brl = sale_notional_brl
            book_basis_brl = remaining_book_brl + buy_notional_brl

        month_last_stamp = stamp
        equity_path.append(market_value_brl)

    if month_last_stamp is not None:
        finalize_month()

    equity = pd.Series(equity_path, index=idx, dtype=float)
    running_peak = equity.cummax().replace(0.0, np.nan)
    drawdown = (equity / running_peak) - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
    return DcaPathResult(
        final_value_brl=float(market_value_brl),
        total_contributed_brl=float(total_contributed_brl),
        profit_brl=float(market_value_brl - total_contributed_brl),
        transaction_cost_brl=float(transaction_cost_brl),
        tax_brl=float(tax_brl),
        withholding_brl=float(withholding_brl),
        max_drawdown=float(max_drawdown),
        months_over_exemption=int(months_over_exemption),
        max_monthly_sales_brl=float(max_monthly_sales_brl),
        contribution_count=int(contribution_count),
    )


def _aggregate_paths(
    *,
    scenario: Scenario,
    initial_capital_brl: float,
    monthly_contribution_brl: float,
    horizon_months: int,
    path_results: list[DcaPathResult],
) -> dict[str, Any]:
    frame = pd.DataFrame([result.__dict__ for result in path_results])
    if frame.empty:
        return {}
    final_value = pd.to_numeric(frame["final_value_brl"], errors="coerce").astype(float)
    profit = pd.to_numeric(frame["profit_brl"], errors="coerce").astype(float)
    total_contributed = pd.to_numeric(frame["total_contributed_brl"], errors="coerce").astype(float)
    multiple = final_value / total_contributed.replace(0.0, np.nan)
    return {
        "scenario_id": str(scenario.scenario_id),
        "family": str(scenario.family),
        "assets": ",".join(scenario.assets) if scenario.assets else "",
        "weights": ",".join(f"{w:.4f}" for w in scenario.weights) if scenario.weights else "",
        "dynamic_candidate_id": str(scenario.dynamic_candidate_id),
        "initial_capital_brl": float(initial_capital_brl),
        "monthly_contribution_brl": float(monthly_contribution_brl),
        "horizon_months": int(horizon_months),
        "window_count": int(frame.shape[0]),
        "median_final_value_brl": float(final_value.quantile(0.50)),
        "p10_final_value_brl": float(final_value.quantile(0.10)),
        "p90_final_value_brl": float(final_value.quantile(0.90)),
        "median_profit_brl": float(profit.quantile(0.50)),
        "p10_profit_brl": float(profit.quantile(0.10)),
        "p90_profit_brl": float(profit.quantile(0.90)),
        "win_rate_profit": float((profit > 0.0).mean()),
        "loss_rate_10pct_contributed": float((profit <= (-0.10 * total_contributed)).mean()),
        "median_multiple_on_contributed": float(multiple.quantile(0.50)),
        "median_transaction_cost_brl": float(pd.to_numeric(frame["transaction_cost_brl"], errors="coerce").quantile(0.50)),
        "median_tax_brl": float(pd.to_numeric(frame["tax_brl"], errors="coerce").quantile(0.50)),
        "median_withholding_brl": float(pd.to_numeric(frame["withholding_brl"], errors="coerce").quantile(0.50)),
        "median_max_drawdown": float(pd.to_numeric(frame["max_drawdown"], errors="coerce").quantile(0.50)),
        "worst_final_value_brl": float(final_value.min()),
        "best_final_value_brl": float(final_value.max()),
        "median_total_contributed_brl": float(total_contributed.quantile(0.50)),
        "median_contribution_count": float(pd.to_numeric(frame["contribution_count"], errors="coerce").quantile(0.50)),
        "months_over_35k_any_rate": float((pd.to_numeric(frame["months_over_exemption"], errors="coerce") > 0).mean()),
        "max_monthly_sales_brl_p90": float(pd.to_numeric(frame["max_monthly_sales_brl"], errors="coerce").quantile(0.90)),
    }


def _build_base_series(
    *,
    scenarios: list[Scenario],
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    asset_table: pd.DataFrame,
    score_map: dict[tuple[int, str], pd.DataFrame],
    asset_ma_filters: dict[int, pd.DataFrame],
    benchmark_filters: dict[int, pd.Series],
    benchmark_ticker: str,
    fallback_ticker: str,
) -> dict[str, tuple[Scenario, pd.Series, pd.Series]]:
    out: dict[str, tuple[Scenario, pd.Series, pd.Series]] = {}
    for scenario in scenarios:
        if scenario.family == "dynamic_motor":
            gross, turnover = _simulate_dynamic_crypto(
                scenario=scenario,
                returns=returns,
                prices=prices,
                asset_table=asset_table,
                score_map=score_map,
                asset_ma_filters=asset_ma_filters,
                benchmark_filters=benchmark_filters,
                benchmark_ticker=str(benchmark_ticker),
                fallback_ticker=str(fallback_ticker),
            )
        else:
            gross, turnover = _simulate_fixed_basket(scenario=scenario, returns=returns)
        if gross.empty:
            continue
        out[str(scenario.scenario_id)] = (scenario, gross.astype(float), turnover.astype(float))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Simulacao de conta pequena com aporte mensal e custo/imposto proxy.")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--asset-groups", default="data/asset_groups_crypto_top_liquid_plus.csv")
    ap.add_argument("--asset-metadata", default="data/asset_metadata_crypto_top_liquid_plus.csv")
    ap.add_argument("--net-assumptions-config", default="config/profit_net_assumptions.json")
    ap.add_argument("--benchmark-ticker", default="BTC-USD")
    ap.add_argument("--fallback-ticker", default="BTC-USD")
    ap.add_argument("--scenario-ids", default=",".join(DEFAULT_SCENARIO_IDS))
    ap.add_argument("--initial-capitals", default="400")
    ap.add_argument("--monthly-contributions", default="50,100")
    ap.add_argument("--horizon-months", default="12,24,36,48")
    ap.add_argument("--outdir-root", default="results/validation/profit_small_account_dca")
    args = ap.parse_args()

    outdir = (ROOT / args.outdir_root / _run_id()).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    scenario_ids = tuple(str(token).strip() for token in str(args.scenario_ids).split(",") if str(token).strip())
    scenario_map = _scenario_map()
    scenarios = [scenario_map[token] for token in scenario_ids if token in scenario_map]
    if not scenarios:
        raise SystemExit("no valid scenarios selected")

    prices_dir = (ROOT / args.prices_dir).resolve()
    asset_groups = (ROOT / args.asset_groups).resolve()
    asset_metadata = (ROOT / args.asset_metadata).resolve()
    initial_capitals = _parse_csv_numbers(str(args.initial_capitals), cast=float)
    monthly_contributions = _parse_csv_numbers(str(args.monthly_contributions), cast=float)
    horizon_months = _parse_csv_numbers(str(args.horizon_months), cast=int)

    asset_table = _load_asset_table(asset_groups, asset_metadata)
    dynamic_selected = any(bool(s.dynamic_candidate_id) for s in scenarios)
    if not dynamic_selected:
        required_tickers = {str(args.benchmark_ticker), str(args.fallback_ticker)}
        for scenario in scenarios:
            required_tickers.update(str(asset) for asset in scenario.assets)
        asset_table = asset_table[asset_table["ticker"].astype(str).isin(sorted(required_tickers))].copy()
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
            for s in scenarios
            if s.dynamic_candidate_id
        }
    )
    ma_days = sorted(
        {
            int(s.dynamic_candidate_id.split("__")[5].replace("ama", ""))
            for s in scenarios
            if s.dynamic_candidate_id
        }
        | {
            int(s.dynamic_candidate_id.split("__")[6].replace("mma", ""))
            for s in scenarios
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

    base_series = _build_base_series(
        scenarios=scenarios,
        returns=returns,
        prices=prices,
        asset_table=asset_table,
        score_map=score_map,
        asset_ma_filters=asset_ma_filters,
        benchmark_filters=benchmark_filters,
        benchmark_ticker=str(args.benchmark_ticker),
        fallback_ticker=str(args.fallback_ticker),
    )

    rows: list[dict[str, Any]] = []
    for initial_capital in initial_capitals:
        for monthly_contribution in monthly_contributions:
            for months in horizon_months:
                for scenario_id, (scenario, gross, turnover) in base_series.items():
                    windows = _window_slices(pd.DatetimeIndex(gross.index), horizon_months=int(months))
                    path_results: list[DcaPathResult] = []
                    for window in windows:
                        path_results.append(
                            _simulate_dca_path(
                                gross_ret=gross.iloc[window["start_pos"] : window["end_pos"] + 1],
                                turnover=turnover.iloc[window["start_pos"] : window["end_pos"] + 1],
                                profile=profile,
                                initial_capital_brl=float(initial_capital),
                                monthly_contribution_brl=float(monthly_contribution),
                                contribution_positions=[
                                    int(pos - window["start_pos"]) for pos in window["contribution_positions"]
                                ],
                            )
                        )
                    row = _aggregate_paths(
                        scenario=scenario,
                        initial_capital_brl=float(initial_capital),
                        monthly_contribution_brl=float(monthly_contribution),
                        horizon_months=int(months),
                        path_results=path_results,
                    )
                    if row:
                        rows.append(row)

    compare_df = pd.DataFrame(rows).sort_values(
        ["initial_capital_brl", "monthly_contribution_brl", "horizon_months", "median_profit_brl"],
        ascending=[True, True, True, False],
        na_position="last",
    ).reset_index(drop=True)
    compare_df.to_csv(outdir / "scenario_compare.csv", index=False)

    by_combo: list[dict[str, Any]] = []
    for (initial_capital, monthly_contribution, months), sub in compare_df.groupby(
        ["initial_capital_brl", "monthly_contribution_brl", "horizon_months"], sort=True
    ):
        best_profit = sub.sort_values(["median_profit_brl", "median_multiple_on_contributed"], ascending=[False, False]).iloc[0].to_dict()
        safest = sub.sort_values(["median_max_drawdown", "median_profit_brl"], ascending=[False, False]).iloc[0].to_dict()
        cleanest = sub.sort_values(["median_transaction_cost_brl", "median_tax_brl"], ascending=[True, True]).iloc[0].to_dict()
        by_combo.append(
            {
                "initial_capital_brl": float(initial_capital),
                "monthly_contribution_brl": float(monthly_contribution),
                "horizon_months": int(months),
                "best_profit_scenario_id": str(best_profit["scenario_id"]),
                "best_profit_median_profit_brl": _safe_float(best_profit.get("median_profit_brl")),
                "best_profit_median_final_value_brl": _safe_float(best_profit.get("median_final_value_brl")),
                "safest_scenario_id": str(safest["scenario_id"]),
                "safest_median_max_drawdown": _safe_float(safest.get("median_max_drawdown")),
                "cleanest_scenario_id": str(cleanest["scenario_id"]),
                "cleanest_median_transaction_cost_brl": _safe_float(cleanest.get("median_transaction_cost_brl")),
                "cleanest_median_tax_brl": _safe_float(cleanest.get("median_tax_brl")),
            }
        )

    summary = {
        "suite": "profit_small_account_dca",
        "profile_id": "crypto_global_brazil_resident_conservative",
        "initial_capitals": initial_capitals,
        "monthly_contributions": monthly_contributions,
        "horizon_months": horizon_months,
        "scenario_ids": list(scenario_ids),
        "crypto_asset_count": int(viability.shape[0]),
        "notes": [
            "As janelas comecam no primeiro dia util de cada mes e terminam no ultimo dia util do mes final do horizonte.",
            "Cada aporte mensal entra no primeiro dia util do novo mes e paga friccao de compra em uma mao.",
            "A camada fiscal usa o mesmo proxy de inventario mensal da modelagem oficial de cripto, agora aplicada no nivel do patrimonio com aportes.",
        ],
        "by_combo": by_combo,
    }
    _write_json(outdir / "summary.json", summary)

    write_run_manifest(
        outdir,
        script="scripts/bench/validation/run_profit_small_account_dca.py",
        params={
            "scenario_ids": list(scenario_ids),
            "initial_capitals": initial_capitals,
            "monthly_contributions": monthly_contributions,
            "horizon_months": horizon_months,
            "prices_dir": str(prices_dir),
            "asset_groups": str(asset_groups),
            "asset_metadata": str(asset_metadata),
        },
        extra={"suite": "profit_small_account_dca"},
    )
    print(str(outdir))


if __name__ == "__main__":
    main()
