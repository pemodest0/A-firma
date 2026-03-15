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
from execution.cost_model import summarize_return_series  # noqa: E402
from execution.net_assumptions import apply_net_assumptions, blend_profiles, load_net_assumption_profiles  # noqa: E402
from scripts.bench.validation.run_profit_alpha_improvement_suite import _write_json  # noqa: E402
from scripts.bench.validation.run_profit_alpha_hardening_suite import _build_candidates  # noqa: E402
from scripts.bench.validation.run_profit_country_compare_suite import (  # noqa: E402
    _build_official_bundle,
    _filter_brazil_equities,
    _link_prices_dir,
    _write_synthetic_benchmark,
)
from scripts.bench.validation.run_profit_10x_rule_search import (  # noqa: E402
    RuleConfig,
    _ensure_benchmark_columns,
    _load_asset_table,
    _load_daily_universe,
    _precompute_scores,
    _safe_float,
    _top_k_indices,
)


DEFAULT_CRYPTO_CANDIDATES = [
    "all_assets__lb021__rb07__k2__mom_total__ama000__mma000__riskon",
    "all_assets__lb021__rb07__k3__mom_vol_adj__ama000__mma000__relshy",
    "all_assets__lb021__rb07__k3__mom_total__ama000__mma000__riskon",
    "all_assets__lb126__rb07__k2__mom_total__ama200__mma200__riskon",
]


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_rule_candidate_id(candidate_id: str, *, groups: tuple[str, ...]) -> RuleConfig:
    parts = str(candidate_id).split("__")
    if len(parts) != 8:
        raise ValueError(f"invalid candidate id: {candidate_id}")
    family_id = parts[0]
    return RuleConfig(
        family_id=str(family_id),
        groups=tuple(groups),
        lookback_days=int(parts[1].replace("lb", "")),
        rebalance_days=int(parts[2].replace("rb", "")),
        top_k=int(parts[3].replace("k", "")),
        score_mode=str(parts[4]),
        asset_ma_days=int(parts[5].replace("ama", "")),
        market_ma_days=int(parts[6].replace("mma", "")),
        relative_to_shy=(parts[7] == "relshy"),
    )


def _start_positions(index: pd.DatetimeIndex, *, monthly_start: bool) -> list[int]:
    if len(index) == 0:
        return []
    if not monthly_start:
        return list(range(len(index)))
    start_labels = pd.Series(index=index, data=np.arange(len(index), dtype=int)).groupby(index.to_period("M")).head(1)
    return [int(v) for v in start_labels.to_list()]


def _forward_path_frame(net_returns: pd.Series, *, horizon_days: int, monthly_start: bool = True) -> pd.DataFrame:
    ret = pd.to_numeric(net_returns, errors="coerce").fillna(0.0).astype(float)
    if ret.empty or ret.shape[0] <= int(horizon_days):
        return pd.DataFrame(columns=["start_date", "terminal_multiple", "max_multiple", "min_multiple"])
    horizon = int(horizon_days)
    wealth = (1.0 + ret).cumprod().astype(float)
    future = wealth.shift(-1)
    terminal_multiple = (wealth.shift(-horizon) / wealth).astype(float)
    rolling_max = future.iloc[::-1].rolling(horizon, min_periods=horizon).max().iloc[::-1]
    rolling_min = future.iloc[::-1].rolling(horizon, min_periods=horizon).min().iloc[::-1]
    max_multiple = (rolling_max / wealth).astype(float)
    min_multiple = (rolling_min / wealth).astype(float)
    starts = _start_positions(pd.DatetimeIndex(ret.index), monthly_start=monthly_start)
    positions = [int(pos) for pos in starts if int(pos) + horizon < len(ret)]
    if not positions:
        return pd.DataFrame(columns=["start_date", "terminal_multiple", "max_multiple", "min_multiple"])
    frame = pd.DataFrame(
        {
            "start_date": [str(pd.Timestamp(ret.index[pos]).date()) for pos in positions],
            "terminal_multiple": terminal_multiple.iloc[positions].to_numpy(dtype=float),
            "max_multiple": max_multiple.iloc[positions].to_numpy(dtype=float),
            "min_multiple": min_multiple.iloc[positions].to_numpy(dtype=float),
        }
    )
    return frame.dropna(subset=["terminal_multiple", "max_multiple", "min_multiple"]).reset_index(drop=True)


def _payoff_row_from_frame(
    *,
    scenario: str,
    candidate_id: str,
    frame: pd.DataFrame,
    base_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if frame.empty:
        return {
            "scenario": str(scenario),
            "candidate_id": str(candidate_id),
            "starts_considered": 0,
            "hit_rate_5x_252d": float("nan"),
            "hit_rate_6x_252d": float("nan"),
            "touch_loss_50_252d": float("nan"),
            "touch_loss_90_252d": float("nan"),
            "end_below_50_252d": float("nan"),
            "end_below_90_252d": float("nan"),
            "median_return_252d": float("nan"),
            "p10_return_252d": float("nan"),
            "p90_return_252d": float("nan"),
            "best_return_252d": float("nan"),
            "worst_return_252d": float("nan"),
            "median_end_value_100_brl": float("nan"),
            "median_end_value_200_brl": float("nan"),
            **(base_metrics or {}),
        }

    terminal = pd.to_numeric(frame["terminal_multiple"], errors="coerce").astype(float)
    max_path = pd.to_numeric(frame["max_multiple"], errors="coerce").astype(float)
    min_path = pd.to_numeric(frame["min_multiple"], errors="coerce").astype(float)
    terminal_ret = terminal - 1.0
    row = {
        "scenario": str(scenario),
        "candidate_id": str(candidate_id),
        "starts_considered": int(frame.shape[0]),
        "hit_rate_5x_252d": float((max_path >= 5.0).mean()),
        "hit_rate_6x_252d": float((max_path >= 6.0).mean()),
        "touch_loss_50_252d": float((min_path <= 0.5).mean()),
        "touch_loss_90_252d": float((min_path <= 0.1).mean()),
        "end_below_50_252d": float((terminal <= 0.5).mean()),
        "end_below_90_252d": float((terminal <= 0.1).mean()),
        "median_return_252d": float(terminal_ret.quantile(0.50)),
        "p10_return_252d": float(terminal_ret.quantile(0.10)),
        "p90_return_252d": float(terminal_ret.quantile(0.90)),
        "best_return_252d": float(terminal_ret.max()),
        "worst_return_252d": float(terminal_ret.min()),
        "median_end_value_100_brl": float(100.0 * terminal.quantile(0.50)),
        "median_end_value_200_brl": float(200.0 * terminal.quantile(0.50)),
    }
    if base_metrics:
        row.update(base_metrics)
    return row


def _one_year_payoff_row(
    *,
    scenario: str,
    candidate_id: str,
    net_returns: pd.Series,
    horizon_days: int,
    monthly_start: bool,
    base_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    frame = _forward_path_frame(net_returns, horizon_days=horizon_days, monthly_start=monthly_start)
    return _payoff_row_from_frame(
        scenario=scenario,
        candidate_id=candidate_id,
        frame=frame,
        base_metrics=base_metrics,
    )


def _simulate_crypto_candidate_series(
    *,
    cfg: RuleConfig,
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    asset_table: pd.DataFrame,
    score_map: dict[tuple[int, str], pd.DataFrame],
    asset_ma_filters: dict[int, pd.DataFrame],
    benchmark_filters: dict[int, pd.Series],
    benchmark_ticker: str,
    fallback_ticker: str,
    net_profiles: dict[str, Any],
) -> tuple[dict[str, Any], pd.Series]:
    all_tickers = list(returns.columns.astype(str))
    ticker_to_col = {ticker: idx for idx, ticker in enumerate(all_tickers)}
    allowed_assets = asset_table[asset_table["asset_group"].astype(str).isin(list(cfg.groups))]["ticker"].astype(str).tolist()
    allowed_idx = np.array([ticker_to_col[ticker] for ticker in allowed_assets if ticker in ticker_to_col], dtype=int)
    if allowed_idx.size == 0:
        return {}, pd.Series(dtype=float)

    score_df = score_map[(int(cfg.lookback_days), str(cfg.score_mode))]
    score_arr = score_df.reindex(index=returns.index, columns=all_tickers).to_numpy(dtype=float)
    asset_ma_arr = asset_ma_filters[int(cfg.asset_ma_days)].reindex(index=returns.index, columns=all_tickers).fillna(False).to_numpy(dtype=bool)
    benchmark_ok = benchmark_filters[int(cfg.market_ma_days)].reindex(returns.index).fillna(False).to_numpy(dtype=bool)
    ret_arr = returns.reindex(columns=all_tickers).to_numpy(dtype=float)

    shy_score = None
    if bool(cfg.relative_to_shy) and fallback_ticker in score_df.columns:
        shy_score = pd.to_numeric(score_df[fallback_ticker], errors="coerce").to_numpy(dtype=float)

    asset_meta = asset_table.drop_duplicates(subset=["ticker"], keep="first").set_index("ticker")
    foreign_flag = np.array(
        [1.0 if str(asset_meta.get("jurisdiction", pd.Series(dtype=object)).get(ticker, "foreign")) == "foreign" else 0.0 for ticker in all_tickers],
        dtype=float,
    )
    fallback_idx = int(ticker_to_col[fallback_ticker]) if fallback_ticker in ticker_to_col else -1

    warmup = max(int(cfg.lookback_days), int(cfg.asset_ma_days), int(cfg.market_ma_days)) + 2
    rebalance_positions = list(range(int(max(1, warmup)), ret_arr.shape[0], int(max(1, cfg.rebalance_days))))
    if not rebalance_positions:
        return {}, pd.Series(dtype=float)

    daily_ret = np.zeros(ret_arr.shape[0], dtype=float)
    daily_turnover = np.zeros(ret_arr.shape[0], dtype=float)
    daily_foreign_share = np.zeros(ret_arr.shape[0], dtype=float)
    prev_weights: dict[str, float] = {"CASH": 1.0}

    for pos_idx, pos in enumerate(rebalance_positions):
        next_pos = rebalance_positions[pos_idx + 1] if pos_idx + 1 < len(rebalance_positions) else ret_arr.shape[0]
        score_row = score_arr[pos]
        valid = np.zeros(score_row.shape[0], dtype=bool)
        valid[allowed_idx] = True
        valid &= np.isfinite(score_row)
        valid &= np.isfinite(ret_arr[pos - 1])
        valid &= asset_ma_arr[pos]
        valid &= score_row > 0.0
        if shy_score is not None and pos < shy_score.shape[0] and np.isfinite(shy_score[pos]):
            valid &= score_row > float(shy_score[pos])
        if int(cfg.market_ma_days) > 0 and not bool(benchmark_ok[pos]):
            valid[:] = False

        selected_idx = _top_k_indices(score_row, valid, int(cfg.top_k))
        if not selected_idx and fallback_idx >= 0:
            selected_idx = [fallback_idx]

        if not selected_idx:
            weights: dict[str, float] = {"CASH": 1.0}
            period_ret = np.zeros(next_pos - pos, dtype=float)
            foreign_share = 0.0
        else:
            weight = 1.0 / float(len(selected_idx))
            weights = {all_tickers[idx]: weight for idx in selected_idx}
            period_block = np.nan_to_num(ret_arr[pos:next_pos, selected_idx], nan=0.0)
            period_ret = period_block.mean(axis=1).astype(float)
            foreign_share = float(np.mean(foreign_flag[selected_idx]))

        daily_ret[pos:next_pos] = period_ret
        daily_foreign_share[pos:next_pos] = foreign_share
        daily_turnover[pos] = 0.5 * float(
            sum(abs(float(prev_weights.get(key, 0.0)) - float(weights.get(key, 0.0))) for key in sorted(set(prev_weights) | set(weights)))
        )
        prev_weights = dict(weights)

    daily_ret_s = pd.Series(daily_ret, index=returns.index, dtype=float)
    turnover_s = pd.Series(daily_turnover, index=returns.index, dtype=float)
    foreign_share_s = pd.Series(daily_foreign_share, index=returns.index, dtype=float)
    avg_foreign_share = float(foreign_share_s.mean()) if not foreign_share_s.empty else 1.0
    foreign_profile = net_profiles["profiles"]["foreign_financial_brazil_resident"]
    br_profile = net_profiles["profiles"]["br_local_equity"]
    blended_profile = blend_profiles(avg_foreign_share, foreign_profile=foreign_profile, br_profile=br_profile)
    net_frame = apply_net_assumptions(daily_ret_s, turnover_s, profile=blended_profile, periods_index=returns.index)
    net_ret = pd.to_numeric(net_frame["net_ret"], errors="coerce").fillna(0.0).astype(float)
    net_summary = summarize_return_series(net_ret, periods_per_year=252)

    return (
        {
            "candidate_id": cfg.candidate_id,
            "net_ann_return": _safe_float(net_summary.get("annualized_return")),
            "net_total_return": _safe_float(net_summary.get("total_return")),
            "net_max_drawdown": _safe_float(net_summary.get("max_drawdown")),
            "avg_turnover_daily": float(turnover_s.mean()) if not turnover_s.empty else float("nan"),
        },
        net_ret,
    )


def _official_series(
    *,
    prices_dir: Path,
    crypto_groups: Path,
    crypto_meta: Path,
    equity_groups: Path,
    equity_meta: Path,
    benchmark_crypto: str,
    benchmark_equity: str,
    equity_profile_id: str = "foreign_financial_brazil_resident",
) -> tuple[Any, Any]:
    built = _build_candidates(
        prices_dir=prices_dir,
        crypto_groups=crypto_groups,
        crypto_meta=crypto_meta,
        equity_groups=equity_groups,
        equity_meta=equity_meta,
        benchmark_crypto=str(benchmark_crypto),
        benchmark_equity=str(benchmark_equity),
        equity_profile_id=str(equity_profile_id),
    )
    return _build_official_bundle(built)


def _crypto_candidate_rows(
    *,
    prices_dir: Path,
    asset_groups: Path,
    asset_metadata: Path,
    net_assumptions_config: Path,
    candidate_ids: list[str],
    horizon_days: int,
    monthly_start: bool,
) -> list[dict[str, Any]]:
    asset_table = _load_asset_table(asset_groups, asset_metadata)
    returns, prices, _viability = _load_daily_universe(
        prices_dir=prices_dir,
        asset_table=asset_table,
        min_history_days=252,
        max_abs_daily_return=2.0,
    )
    returns, prices = _ensure_benchmark_columns(returns, prices, prices_dir, ["BTC-USD"])
    returns = returns[returns.index >= pd.Timestamp("2016-02-18")].copy()
    prices = prices.reindex(returns.index).copy()
    lookbacks = sorted(
        {
            int(part.split("__")[1].replace("lb", ""))
            for part in candidate_ids
        }
    )
    ma_days = sorted(
        {
            int(token.replace("ama", ""))
            for cid in candidate_ids
            for token in [cid.split("__")[5]]
        }
        | {
            int(token.replace("mma", ""))
            for cid in candidate_ids
            for token in [cid.split("__")[6]]
        }
        | {0}
    )
    score_map, asset_ma_filters, benchmark_filters = _precompute_scores(
        returns,
        prices,
        lookbacks=lookbacks,
        asset_ma_days_list=ma_days,
        benchmark_ticker="BTC-USD",
    )
    net_profiles = load_net_assumption_profiles(net_assumptions_config)
    all_groups = tuple(sorted(asset_table["asset_group"].astype(str).unique().tolist()))
    rows: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        cfg = _parse_rule_candidate_id(candidate_id, groups=all_groups)
        perf_row, net_ret = _simulate_crypto_candidate_series(
            cfg=cfg,
            returns=returns,
            prices=prices,
            asset_table=asset_table,
            score_map=score_map,
            asset_ma_filters=asset_ma_filters,
            benchmark_filters=benchmark_filters,
            benchmark_ticker="BTC-USD",
            fallback_ticker="BTC-USD",
            net_profiles=net_profiles,
        )
        if net_ret.empty:
            continue
        base_metrics = {
            "full_net_ann_return": _safe_float(perf_row.get("net_ann_return")),
            "full_net_total_return": _safe_float(perf_row.get("net_total_return")),
            "full_net_max_drawdown": _safe_float(perf_row.get("net_max_drawdown")),
            "avg_turnover_daily": _safe_float(perf_row.get("avg_turnover_daily")),
        }
        rows.append(
            _one_year_payoff_row(
                scenario="crypto_aggressive",
                candidate_id=str(candidate_id),
                net_returns=net_ret,
                horizon_days=horizon_days,
                monthly_start=monthly_start,
                base_metrics=base_metrics,
            )
        )
    return rows


def _official_rows(
    *,
    prices_dir: Path,
    crypto_groups: Path,
    crypto_meta: Path,
    equity_groups: Path,
    equity_meta: Path,
    benchmark_crypto: str,
    benchmark_equity: str,
    horizon_days: int,
    monthly_start: bool,
    outdir: Path,
) -> list[dict[str, Any]]:
    global_baseline, global_official = _official_series(
        prices_dir=prices_dir,
        crypto_groups=crypto_groups,
        crypto_meta=crypto_meta,
        equity_groups=equity_groups,
        equity_meta=equity_meta,
        benchmark_crypto=benchmark_crypto,
        benchmark_equity=benchmark_equity,
    )

    br_groups, br_meta, br_tickers = _filter_brazil_equities(
        equity_groups=equity_groups,
        equity_meta=equity_meta,
        outdir=outdir,
    )
    br_prices_dir = outdir / "prices_brazil"
    _link_prices_dir(source_dir=prices_dir, target_dir=br_prices_dir)
    br_benchmark_ticker = "BR_SYNTH"
    _write_synthetic_benchmark(prices_dir=prices_dir, tickers=br_tickers, outdir=br_prices_dir, benchmark_ticker=br_benchmark_ticker)

    br_baseline, br_official = _official_series(
        prices_dir=br_prices_dir,
        crypto_groups=crypto_groups,
        crypto_meta=crypto_meta,
        equity_groups=br_groups,
        equity_meta=br_meta,
        benchmark_crypto=benchmark_crypto,
        benchmark_equity=br_benchmark_ticker,
        equity_profile_id="br_local_equity",
    )

    built_br = _build_candidates(
        prices_dir=br_prices_dir,
        crypto_groups=crypto_groups,
        crypto_meta=crypto_meta,
        equity_groups=br_groups,
        equity_meta=br_meta,
        benchmark_crypto=str(benchmark_crypto),
        benchmark_equity=br_benchmark_ticker,
        equity_profile_id="br_local_equity",
    )
    br_equity_base = built_br["context"]["equity_base"].result

    rows = []
    for scenario, result in [
        ("global_mixed_official", global_official),
        ("brazil_crypto_official", br_official),
        ("brazil_only_equity_base", br_equity_base),
    ]:
        rows.append(
            _one_year_payoff_row(
                scenario=scenario,
                candidate_id=str(result.candidate_id),
                net_returns=result.net_ret,
                horizon_days=horizon_days,
                monthly_start=monthly_start,
                base_metrics={
                    "full_net_ann_return": _safe_float(result.net_ann_return),
                    "full_net_total_return": _safe_float(result.net_total_return),
                    "full_net_max_drawdown": _safe_float(result.net_max_drawdown),
                    "avg_turnover_daily": _safe_float(result.avg_turnover_daily),
                },
            )
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Audita payoff de 252 dias para cenarios agressivos e oficiais do motor.")
    ap.add_argument("--prices-dir", default="data/raw/finance/yfinance_daily")
    ap.add_argument("--crypto-asset-groups", default="data/asset_groups_crypto_top_liquid.csv")
    ap.add_argument("--crypto-asset-metadata", default="data/asset_metadata_crypto_top_liquid.csv")
    ap.add_argument("--equity-asset-groups", default="data/asset_groups_target_800_clean_plus.csv")
    ap.add_argument("--equity-asset-metadata", default="data/asset_metadata_target_800_clean_plus.csv")
    ap.add_argument("--net-assumptions-config", default="config/profit_net_assumptions.json")
    ap.add_argument("--benchmark-crypto", default="BTC-USD")
    ap.add_argument("--benchmark-equity", default="SPY")
    ap.add_argument("--horizon-days", type=int, default=252)
    ap.add_argument("--monthly-start", action="store_true")
    ap.add_argument("--skip-crypto", action="store_true")
    ap.add_argument("--skip-officials", action="store_true")
    ap.add_argument("--candidate-ids", default=",".join(DEFAULT_CRYPTO_CANDIDATES))
    ap.add_argument("--outdir-root", default="results/validation/profit_one_year_payoff_audit")
    args = ap.parse_args()

    outdir = (ROOT / args.outdir_root / _run_id()).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    candidate_ids = [token.strip() for token in str(args.candidate_ids).split(",") if token.strip()]
    prices_dir = (ROOT / args.prices_dir).resolve()
    crypto_groups = (ROOT / args.crypto_asset_groups).resolve()
    crypto_meta = (ROOT / args.crypto_asset_metadata).resolve()
    equity_groups = (ROOT / args.equity_asset_groups).resolve()
    equity_meta = (ROOT / args.equity_asset_metadata).resolve()
    net_assumptions_config = (ROOT / args.net_assumptions_config).resolve()

    rows = []
    if not bool(args.skip_crypto):
        rows.extend(
            _crypto_candidate_rows(
                prices_dir=prices_dir,
                asset_groups=crypto_groups,
                asset_metadata=crypto_meta,
                net_assumptions_config=net_assumptions_config,
                candidate_ids=candidate_ids,
                horizon_days=int(args.horizon_days),
                monthly_start=bool(args.monthly_start),
            )
        )
    if not bool(args.skip_officials):
        rows.extend(
            _official_rows(
                prices_dir=prices_dir,
                crypto_groups=crypto_groups,
                crypto_meta=crypto_meta,
                equity_groups=equity_groups,
                equity_meta=equity_meta,
                benchmark_crypto=str(args.benchmark_crypto),
                benchmark_equity=str(args.benchmark_equity),
                horizon_days=int(args.horizon_days),
                monthly_start=bool(args.monthly_start),
                outdir=outdir,
            )
        )
    if not rows:
        raise SystemExit("no scenarios selected")

    compare_df = pd.DataFrame(rows).sort_values(
        ["hit_rate_6x_252d", "median_return_252d", "end_below_50_252d"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    compare_df.to_csv(outdir / "candidate_compare.csv", index=False)

    best_6x = compare_df.sort_values(["hit_rate_6x_252d", "median_return_252d"], ascending=[False, False]).iloc[0].to_dict()
    best_median = compare_df.sort_values(["median_return_252d", "hit_rate_6x_252d"], ascending=[False, False]).iloc[0].to_dict()
    safest = compare_df.sort_values(["end_below_50_252d", "touch_loss_50_252d", "median_return_252d"], ascending=[True, True, False]).iloc[0].to_dict()

    summary = {
        "suite": "profit_one_year_payoff_audit",
        "horizon_days": int(args.horizon_days),
        "monthly_start": bool(args.monthly_start),
        "target_multiple_notes": {
            "five_x": "5x final wealth = +400%",
            "six_x": "6x final wealth = +500%",
        },
        "best_hit_rate_6x_252d": best_6x,
        "best_median_return_252d": best_median,
        "lowest_end_below_50_252d": safest,
        "notes": [
            "As probabilidades usam janelas forward de 252 dias uteis.",
            "Perder tudo foi aproximado como terminar ou tocar abaixo de 10% do capital (-90%).",
            "Se monthly_start=true, cada janela comeca no primeiro pregao util de cada mes; se false, usa todos os dias.",
        ],
    }
    _write_json(outdir / "summary.json", summary)

    write_run_manifest(
        outdir,
        script="scripts/bench/validation/run_profit_one_year_payoff_audit.py",
        params={
            "prices_dir": str(prices_dir),
            "crypto_asset_groups": str(crypto_groups),
            "crypto_asset_metadata": str(crypto_meta),
            "equity_asset_groups": str(equity_groups),
            "equity_asset_metadata": str(equity_meta),
            "net_assumptions_config": str(net_assumptions_config),
            "benchmark_crypto": str(args.benchmark_crypto),
            "benchmark_equity": str(args.benchmark_equity),
            "horizon_days": int(args.horizon_days),
            "monthly_start": bool(args.monthly_start),
            "candidate_ids": candidate_ids,
        },
        extra={"suite": "profit_one_year_payoff_audit"},
    )
    print(str(outdir))


if __name__ == "__main__":
    main()
